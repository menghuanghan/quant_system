"""
EventSignalProcessor - 事件信号宽表处理器（纯cuDF GPU版本）

生成 dwd_event_signal 宽表，捕捉公司行为产生的瞬时Alpha或风险预警

数据源：
    - repurchase: 回购（公告日 ann_date）
    - share_float: 解禁（解禁日 float_date）
    - pledge: 质押（end_date 作为统计日期）
    - dividend: 分红（公告日期 ann_date）

处理逻辑：
    1. 事件映射：将稀疏的事件记录映射到 trade_date
    2. 信号生成：
       - 0/1 信号：事件发生当日标记为 1
       - 持续性窗口：例如"分红预案公告后30天内"
       - 预警特征：如解禁前N天生成 days_to_unlock
"""

import logging
from typing import Optional, Set

import cudf
import cupy as cp
import pandas as pd

from .base import BaseProcessor
from .config import (
    DATA_SOURCE_PATHS,
    DWD_OUTPUT_CONFIG,
    PROCESSING_CONFIG,
)

logger = logging.getLogger(__name__)


def _build_date_mapping(trade_dates: Set[str]) -> dict:
    """
    构建非交易日到最近下一个交易日的映射表
    
    这是解决事件日期对齐问题的核心方法：
    - 事件（如分红公告）可能发生在周末/节假日
    - 需要映射到最近的下一个交易日才能与骨架表 join
    
    Args:
        trade_dates: 交易日集合 (YYYY-MM-DD格式)
    
    Returns:
        dict: {日期字符串: 最近下一个交易日}
    """
    sorted_dates = sorted(trade_dates)
    date_mapping = {}
    
    # 为交易日建立自映射
    for d in sorted_dates:
        date_mapping[d] = d
    
    # 填充非交易日映射
    if sorted_dates:
        min_date = sorted_dates[0]
        max_date = sorted_dates[-1]
        
        # 从最大日期往前遍历，填充空档
        from datetime import datetime, timedelta
        
        current = datetime.strptime(max_date, '%Y-%m-%d')
        end = datetime.strptime(min_date, '%Y-%m-%d')
        
        next_trade_date = max_date
        while current >= end:
            date_str = current.strftime('%Y-%m-%d')
            if date_str in trade_dates:
                next_trade_date = date_str
            date_mapping[date_str] = next_trade_date
            current -= timedelta(days=1)
    
    return date_mapping


class EventSignalProcessor(BaseProcessor):
    """
    事件信号宽表处理器 - 纯GPU版本
    
    核心改进（v2）：
    1. 非交易日事件映射到最近下一个交易日
    2. 完善的缺失值填充策略
    3. 状态型数据（质押率）前向填充
    
    输出字段：
        - trade_date, ts_code: 主键
        # 回购事件
        - is_repurchase_ann: 是否回购公告日
        - repurchase_amount: 回购金额（如有）
        - in_repurchase_window: 回购公告后30天内
        # 解禁事件
        - is_unlock_day: 是否解禁日
        - unlock_share: 解禁股数
        - unlock_ratio: 解禁比例
        - days_to_unlock: 距离下次解禁天数（预警）
        - in_unlock_window: 解禁前后窗口期
        # 质押风险
        - pledge_ratio: 质押比例
        - pledge_ratio_high: 高质押风险标记（>50%）
        # 分红事件
        - is_dividend_ann: 是否分红预案公告日
        - cash_div: 每股现金分红
        - stk_div: 每股送转
        - ex_date: 除权除息日
        - in_dividend_window: 分红公告后窗口期
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        super().__init__(use_gpu=use_gpu, start_date=start_date, end_date=end_date)
        self.output_path = DWD_OUTPUT_CONFIG.output_dir / DWD_OUTPUT_CONFIG.event_signal
        self._date_mapping: Optional[dict] = None
        self._trade_dates_set: Optional[Set[str]] = None
    
    def _get_date_mapping(self) -> dict:
        """获取日期映射表（懒加载）"""
        if self._date_mapping is None:
            trade_dates = self.get_trade_dates()
            self._trade_dates_set = set(trade_dates)
            self._date_mapping = _build_date_mapping(self._trade_dates_set)
            logger.info(f"构建日期映射表完成，覆盖 {len(self._date_mapping)} 个日期")
        return self._date_mapping
    
    def _map_date_to_trade_date(self, df: cudf.DataFrame, date_col: str) -> cudf.DataFrame:
        """
        将事件日期映射到最近的交易日
        
        Args:
            df: 包含日期列的DataFrame
            date_col: 日期列名
        
        Returns:
            添加了 mapped_trade_date 列的DataFrame（保持 object/str 类型以兼容骨架表）
        """
        date_mapping = self._get_date_mapping()
        
        # 转换为Python list进行映射
        dates = df[date_col].to_arrow().to_pylist()
        mapped_dates = [date_mapping.get(d, d) for d in dates]
        
        # 保持 object/str 类型（与骨架表一致）
        df['mapped_trade_date'] = cudf.Series(mapped_dates)
        
        # 统计映射情况
        original_set = set(dates)
        mapped_set = set(mapped_dates)
        non_trade_days = original_set - self._trade_dates_set
        if non_trade_days:
            logger.info(f"  {date_col}: {len(non_trade_days)} 个非交易日事件已映射")
        
        return df
    
    def _load_repurchase(self) -> cudf.DataFrame:
        """加载回购数据"""
        logger.info("加载回购数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.repurchase)
        
        if len(df) == 0:
            logger.warning("无法加载回购数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'ann_date')
        df = df[df['ann_date'] <= self.end_date]
        df = df[df['ann_date'] >= self.start_date]
        
        logger.info(f"加载回购数据完成，共 {len(df)} 行")
        return df
    
    def _load_share_float(self) -> cudf.DataFrame:
        """加载解禁数据"""
        logger.info("加载解禁数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.share_float)
        
        if len(df) == 0:
            logger.warning("无法加载解禁数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'float_date')
        df = self.normalize_date_column(df, 'ann_date')
        
        # 解禁日在范围内
        df = df[df['float_date'] <= self.end_date]
        
        logger.info(f"加载解禁数据完成，共 {len(df)} 行")
        return df
    
    def _load_pledge(self) -> cudf.DataFrame:
        """加载质押数据"""
        logger.info("加载质押数据...")
        
        df = self.read_parquet_dir(DATA_SOURCE_PATHS.pledge_dir)
        
        if len(df) == 0:
            logger.warning("无法加载质押数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'end_date')
        df = df[df['end_date'] <= self.end_date]
        
        logger.info(f"加载质押数据完成，共 {len(df)} 行")
        return df
    
    def _load_dividend(self) -> cudf.DataFrame:
        """加载分红数据"""
        logger.info("加载分红数据...")
        
        df = self.read_parquet_dir(DATA_SOURCE_PATHS.dividend_dir)
        
        if len(df) == 0:
            logger.warning("无法加载分红数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'ann_date')
        df = self.normalize_date_column(df, 'ex_date')
        df = self.normalize_date_column(df, 'record_date')
        
        # 过滤范围
        df = df[df['ann_date'] <= self.end_date]
        
        logger.info(f"加载分红数据完成，共 {len(df)} 行")
        return df
    
    def _process_repurchase(self, df: cudf.DataFrame, skeleton: cudf.DataFrame) -> cudf.DataFrame:
        """处理回购事件"""
        if len(df) == 0:
            return skeleton.assign(
                is_repurchase_ann=0,
                repurchase_amount=0.0,
                in_repurchase_window=0
            )
        
        logger.info("处理回购事件...")
        
        # 关键修复：映射到交易日
        df = self._map_date_to_trade_date(df, 'ann_date')
        
        # 按股票-交易日汇总（使用映射后的日期）
        repurchase_agg = df.groupby(['ts_code', 'mapped_trade_date']).agg({
            'amount': 'sum'
        }).reset_index()
        repurchase_agg = repurchase_agg.rename(columns={
            'mapped_trade_date': 'trade_date',
            'amount': 'repurchase_amount'
        })
        repurchase_agg['is_repurchase_ann'] = 1
        
        result = skeleton.merge(repurchase_agg, on=['ts_code', 'trade_date'], how='left')
        result['is_repurchase_ann'] = result['is_repurchase_ann'].fillna(0).astype('int32')
        result['repurchase_amount'] = result['repurchase_amount'].fillna(0.0)
        
        # 计算回购窗口（公告后30天内）
        result = result.sort_values(['ts_code', 'trade_date'])
        
        # 使用cumsum计算累计回购公告数
        result['cum_repurchase'] = result.groupby('ts_code')['is_repurchase_ann'].cumsum()
        # 30天前的累计数
        result['cum_repurchase_30d_ago'] = result.groupby('ts_code')['cum_repurchase'].shift(30).fillna(0)
        # 过去30天内的回购公告数
        result['repurchase_30d_count'] = result['cum_repurchase'] - result['cum_repurchase_30d_ago']
        result['in_repurchase_window'] = (result['repurchase_30d_count'] > 0).astype('int32')
        
        # 清理临时列
        result = result.drop(columns=['cum_repurchase', 'cum_repurchase_30d_ago', 'repurchase_30d_count'])
        
        matched = (result['is_repurchase_ann'] == 1).sum()
        logger.info(f"回购事件处理完成，捕获 {matched} 条记录")
        return result
    
    def _process_share_float(self, df: cudf.DataFrame, skeleton: cudf.DataFrame) -> cudf.DataFrame:
        """处理解禁事件
        
        PIT修复（v2）：days_to_unlock 只考虑已公告（ann_date <= trade_date）的解禁事件
        
        逻辑说明：
        1. is_unlock_day: 标记解禁实际发生日期（float_date），用于事后分析
        2. days_to_unlock: 只有在公告日（ann_date）之后才能"看到"该解禁事件
           - 在 ann_date 之前: days_to_unlock = -1（不知道有这个解禁）
           - 在 ann_date 之后: days_to_unlock = 距离 float_date 的交易日数
        """
        MIN_SHARE_FLOAT_ROWS = 500  # 低于此阈值认为数据不充分
        
        if len(df) == 0:
            logger.warning(
                "解禁数据 (share_float) 为空，days_to_unlock 等字段将全为默认值。"
                "请检查数据采集任务 'share_float' 是否正常执行。"
            )
            return skeleton.assign(
                is_unlock_day=0,
                unlock_share=0.0,
                unlock_ratio=0.0,
                days_to_unlock=-1,
                in_unlock_window=0
            )
        
        if len(df) < MIN_SHARE_FLOAT_ROWS:
            logger.warning(
                f"解禁数据 (share_float) 仅 {len(df)} 行，远低于预期 "
                f"（阈值 {MIN_SHARE_FLOAT_ROWS}），days_to_unlock 等字段可能不准确。"
                f"建议重新采集: python scripts/run_full_collection.py "
                f"--domains fundamental --tasks share_float"
            )
        
        logger.info("处理解禁事件...")
        
        # =============================================
        # 1. 处理解禁日标记（is_unlock_day）- 使用 float_date
        # =============================================
        df = self._map_date_to_trade_date(df, 'float_date')
        # 保存 float_date 映射结果，避免后续被覆盖
        df['float_trade_date'] = df['mapped_trade_date'].copy()
        
        # 按股票-交易日汇总
        float_agg = df.groupby(['ts_code', 'float_trade_date']).agg({
            'float_share': 'sum',
            'float_ratio': 'sum'
        }).reset_index()
        float_agg = float_agg.rename(columns={
            'float_trade_date': 'trade_date',
            'float_share': 'unlock_share',
            'float_ratio': 'unlock_ratio'
        })
        float_agg['is_unlock_day'] = 1
        
        result = skeleton.merge(float_agg, on=['ts_code', 'trade_date'], how='left')
        result['is_unlock_day'] = result['is_unlock_day'].fillna(0).astype('int32')
        result['unlock_share'] = result['unlock_share'].fillna(0.0)
        result['unlock_ratio'] = result['unlock_ratio'].fillna(0.0)
        
        # =============================================
        # 2. 计算 days_to_unlock（PIT 合规版本）
        #    只有 ann_date <= trade_date 的解禁才"可见"
        # =============================================
        logger.info("计算 days_to_unlock（PIT 合规）...")
        
        # 将 ann_date 也映射到交易日（取下一个交易日）
        df = self._map_date_to_trade_date(df, 'ann_date')
        df['ann_trade_date'] = df['mapped_trade_date'].copy()
        
        # 构建解禁事件表：每个事件有 (ts_code, ann_trade_date, float_trade_date)
        unlock_events = df[['ts_code', 'ann_trade_date', 'float_trade_date']].drop_duplicates()
        
        # 构建交易日序号映射
        trade_dates = sorted(skeleton['trade_date'].unique().to_arrow().to_pylist())
        date_to_pos = {d: i for i, d in enumerate(trade_dates)}
        min_date = trade_dates[0]  # 骨架表最早日期
        
        # 为 result 添加序号
        result = result.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        result['_trade_pos'] = result['trade_date'].to_pandas().map(date_to_pos).values
        
        # 为解禁事件添加序号
        # 关键修复：对于 ann_date 在骨架表之前的事件，将 _ann_pos 设为 -1（始终可见）
        unlock_events_pd = unlock_events.to_pandas()
        unlock_events_pd['_ann_pos'] = unlock_events_pd['ann_trade_date'].map(date_to_pos)
        unlock_events_pd['_float_pos'] = unlock_events_pd['float_trade_date'].map(date_to_pos)
        
        # 如果 ann_trade_date < min_date 或不在字典中，设为 -1（始终可见）
        is_early_announce = (unlock_events_pd['_ann_pos'].isna()) | (unlock_events_pd['ann_trade_date'] < min_date)
        unlock_events_pd.loc[is_early_announce, '_ann_pos'] = -1
        
        # float_date 不在骨架表范围内的事件直接跳过（无效事件）
        unlock_events_pd = unlock_events_pd.dropna(subset=['_float_pos'])
        unlock_events_pd['_ann_pos'] = unlock_events_pd['_ann_pos'].astype(int)
        unlock_events_pd['_float_pos'] = unlock_events_pd['_float_pos'].astype(int)
        
        # 统计历史公告
        hist_announce = (unlock_events_pd['_ann_pos'] == -1).sum()
        if hist_announce > 0:
            logger.info(f"  历史公告（ann_date 在骨架表之前）: {hist_announce} 条，始终可见")
        
        # 初始化 days_to_unlock 为 -1（无解禁预警）
        result['days_to_unlock'] = -1
        
        # =============================================
        # 向量化计算 days_to_unlock（性能优化版本）
        # 使用 numba JIT + numpy 广播机制
        # =============================================
        import numpy as np
        from numba import njit, prange
        
        @njit(parallel=True, cache=True)
        def _compute_days_to_unlock_numba(trade_positions, ann_positions, float_positions):
            """
            Numba JIT 加速版：计算每个交易日到最近可见解禁的天数
            
            trade_positions: shape (N,) - 交易日序号
            ann_positions: shape (M,) - 解禁公告日序号 (-1 表示始终可见)
            float_positions: shape (M,) - 解禁日序号
            
            返回: shape (N,) - 每个交易日的 days_to_unlock (-1 表示无可见解禁)
            """
            N = len(trade_positions)
            M = len(float_positions)
            
            result = np.full(N, -1, dtype=np.int32)
            
            if M == 0:
                return result
            
            # 并行处理每个交易日
            for i in prange(N):
                trade_pos = trade_positions[i]
                min_days = np.int32(2147483647)  # INT_MAX
                
                # 遍历所有解禁事件，找最近的可见解禁
                for j in range(M):
                    ann_pos = ann_positions[j]
                    float_pos = float_positions[j]
                    
                    # 可见性条件：ann_pos <= trade_pos（已公告）且 float_pos >= trade_pos（未来解禁）
                    if ann_pos <= trade_pos and float_pos >= trade_pos:
                        days = float_pos - trade_pos
                        if days < min_days:
                            min_days = days
                
                if min_days != 2147483647:
                    result[i] = min_days
            
            return result
        
        # 按股票分组处理（避免全量笛卡尔积内存爆炸）
        stocks = result['ts_code'].unique().to_arrow().to_pylist()
        result_pd = result[['ts_code', '_trade_pos']].to_pandas()
        
        # 预分配结果数组
        days_to_unlock_all = np.full(len(result_pd), -1, dtype=np.int32)
        
        # 为加速查找，提前建立索引
        result_pd_indexed = result_pd.reset_index()
        stock_groups = result_pd_indexed.groupby('ts_code')
        event_groups = unlock_events_pd.groupby('ts_code')
        
        logger.info(f"  Numba JIT 并行处理 {len(stocks)} 只股票的 days_to_unlock...")
        
        # 预热 numba JIT（第一次调用会编译）
        _warmup = _compute_days_to_unlock_numba(
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32)
        )
        
        for ts_code in stocks:
            # 获取该股票在结果表中的索引和交易日序号
            stock_rows = stock_groups.get_group(ts_code)
            indices = stock_rows['index'].values
            trade_positions = stock_rows['_trade_pos'].values.astype(np.int32)
            
            # 获取该股票的解禁事件
            if ts_code not in event_groups.groups:
                continue  # 无解禁事件，保持 -1
            
            stock_events = event_groups.get_group(ts_code)
            ann_positions = stock_events['_ann_pos'].values.astype(np.int32)
            float_positions = stock_events['_float_pos'].values.astype(np.int32)
            
            # 向量化计算（无内层循环）
            dtu = _compute_days_to_unlock_numba(trade_positions, ann_positions, float_positions)
            
            # 写入结果
            days_to_unlock_all[indices] = dtu
        
        result['days_to_unlock'] = cudf.Series(days_to_unlock_all).astype('int32')
        
        # 清理临时列
        result = result.drop(columns=['_trade_pos'], errors='ignore')
        
        # =============================================
        # 3. 解禁窗口（解禁前后各15天）- 也使用 PIT 逻辑
        # =============================================
        result = result.sort_values(['ts_code', 'trade_date'])
        
        # in_unlock_window: 距离解禁 <= 15 天（包括解禁前后）
        # 解禁前：days_to_unlock 在 [0, 15] 区间
        # 解禁后：需要另外处理
        result['pre_unlock_window'] = ((result['days_to_unlock'] >= 0) & 
                                        (result['days_to_unlock'] <= 15)).astype('int32')
        
        # 解禁后窗口：使用 cumsum + shift（解禁发生后15天内）
        result['cum_unlock'] = result.groupby('ts_code')['is_unlock_day'].cumsum()
        result['cum_unlock_15d_ago'] = result.groupby('ts_code')['cum_unlock'].shift(15).fillna(0)
        result['post_unlock_count'] = result['cum_unlock'] - result['cum_unlock_15d_ago']
        result['post_unlock_window'] = (result['post_unlock_count'] > 0).astype('int32')
        
        result['in_unlock_window'] = ((result['pre_unlock_window'] == 1) | 
                                       (result['post_unlock_window'] == 1)).astype('int32')
        result = result.drop(columns=['pre_unlock_window', 'cum_unlock', 'cum_unlock_15d_ago', 
                                       'post_unlock_count', 'post_unlock_window'])
        
        matched = (result['is_unlock_day'] == 1).sum()
        pit_visible = (result['days_to_unlock'] >= 0).sum()
        logger.info(f"解禁事件处理完成: 实际解禁 {matched} 天, PIT可见 {pit_visible} 条记录")
        return result
    
    def _process_pledge(self, df: cudf.DataFrame, skeleton: cudf.DataFrame) -> cudf.DataFrame:
        """处理质押数据"""
        if len(df) == 0:
            return skeleton.assign(
                pledge_ratio=0.0,
                pledge_ratio_high=0
            )
        
        logger.info("处理质押数据...")
        
        # 关键修复：质押数据的 end_date 是统计截止日，中登公司实际公告日期通常滞后 3-5 天
        # 为避免 Look-Ahead Bias，增加 5 天公告滞后偏移
        import pandas as pd
        PLEDGE_ANNOUNCEMENT_LAG_DAYS = 5
        
        # 转换日期类型为 datetime64 以便做日期加法
        if df['end_date'].dtype == 'object' or str(df['end_date'].dtype) == 'str':
            df['end_date'] = cudf.to_datetime(df['end_date'], format='%Y-%m-%d')
        
        df['end_date'] = df['end_date'] + pd.Timedelta(days=PLEDGE_ANNOUNCEMENT_LAG_DAYS)
        
        # 转回字符串格式以便映射（与 _map_date_to_trade_date 兼容）
        df['end_date'] = df['end_date'].dt.strftime('%Y-%m-%d')
        logger.info(f"质押数据增加 {PLEDGE_ANNOUNCEMENT_LAG_DAYS} 天公告滞后偏移")
        
        # 映射到交易日（返回 object/str 类型）
        df = self._map_date_to_trade_date(df, 'end_date')
        
        # 质押是累积数据，使用PIT方式处理
        pledge_data = df[['ts_code', 'mapped_trade_date', 'pledge_ratio']].copy()
        pledge_data = pledge_data.rename(columns={'mapped_trade_date': 'trade_date'})
        
        # 去重（取每个股票每天最新数据）
        pledge_data = pledge_data.sort_values(['ts_code', 'trade_date'])
        pledge_data = pledge_data.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
        
        # 合并到骨架表（两者都是 object 类型）
        result = skeleton.merge(pledge_data, on=['ts_code', 'trade_date'], how='left')
        
        # 关键修复：前向填充质押比例（状态型数据）
        result = result.sort_values(['ts_code', 'trade_date'])
        result['pledge_ratio'] = result.groupby('ts_code')['pledge_ratio'].ffill()
        
        # 兜底填充（股票上市初期可能无质押数据）
        result['pledge_ratio'] = result['pledge_ratio'].fillna(0.0)
        
        # 高质押风险标记（>50%）
        result['pledge_ratio_high'] = (result['pledge_ratio'] > 50).astype('int32')
        
        filled_count = (result['pledge_ratio'] > 0).sum()
        logger.info(f"质押数据处理完成，{filled_count} 条记录有质押信息")
        return result
    
    def _process_dividend(self, df: cudf.DataFrame, skeleton: cudf.DataFrame) -> cudf.DataFrame:
        """处理分红事件"""
        if len(df) == 0:
            return skeleton.assign(
                is_dividend_ann=0,
                cash_div=0.0,
                stk_div=0.0,
                in_dividend_window=0
            )
        
        logger.info("处理分红事件...")
        
        # 关键修复：映射到交易日
        df = self._map_date_to_trade_date(df, 'ann_date')
        
        # 分红预案公告
        div_data = df[['ts_code', 'mapped_trade_date', 'cash_div', 'stk_div']].copy()
        div_data = div_data.rename(columns={'mapped_trade_date': 'trade_date'})
        
        # 填充空值（原始数据中 cash_div/stk_div 可能为空）
        div_data['cash_div'] = div_data['cash_div'].fillna(0.0)
        div_data['stk_div'] = div_data['stk_div'].fillna(0.0)
        
        # 汇总（一天可能有多次公告）
        div_agg = div_data.groupby(['ts_code', 'trade_date']).agg({
            'cash_div': 'sum',
            'stk_div': 'sum'
        }).reset_index()
        div_agg['is_dividend_ann'] = 1
        
        result = skeleton.merge(div_agg, on=['ts_code', 'trade_date'], how='left')
        result['is_dividend_ann'] = result['is_dividend_ann'].fillna(0).astype('int32')
        result['cash_div'] = result['cash_div'].fillna(0.0)
        result['stk_div'] = result['stk_div'].fillna(0.0)
        
        # 分红窗口（公告后30天）
        result = result.sort_values(['ts_code', 'trade_date'])
        
        result['cum_dividend'] = result.groupby('ts_code')['is_dividend_ann'].cumsum()
        result['cum_dividend_30d_ago'] = result.groupby('ts_code')['cum_dividend'].shift(30).fillna(0)
        result['dividend_30d_count'] = result['cum_dividend'] - result['cum_dividend_30d_ago']
        result['in_dividend_window'] = (result['dividend_30d_count'] > 0).astype('int32')
        
        result = result.drop(columns=['cum_dividend', 'cum_dividend_30d_ago', 'dividend_30d_count'])
        
        matched = (result['is_dividend_ann'] == 1).sum()
        logger.info(f"分红事件处理完成，捕获 {matched} 条记录")
        return result
    
    def process(self) -> cudf.DataFrame:
        """处理并生成事件信号宽表"""
        logger.info("开始处理事件信号宽表...")
        
        # 初始化日期映射（预先构建以便复用）
        _ = self._get_date_mapping()
        
        # 1. 构建骨架表
        skeleton = self.build_skeleton_table()
        
        # 2. 加载各事件数据
        repurchase = self._load_repurchase()
        share_float = self._load_share_float()
        pledge = self._load_pledge()
        dividend = self._load_dividend()
        
        # 3. 分别处理各类事件
        result = skeleton.copy()
        
        # 回购
        repurchase_result = self._process_repurchase(repurchase, skeleton)
        repurchase_cols = ['ts_code', 'trade_date', 'is_repurchase_ann', 'repurchase_amount', 'in_repurchase_window']
        repurchase_result = repurchase_result[[c for c in repurchase_cols if c in repurchase_result.columns]]
        result = result.merge(repurchase_result, on=['ts_code', 'trade_date'], how='left')
        
        # 解禁
        float_result = self._process_share_float(share_float, skeleton)
        float_cols = ['ts_code', 'trade_date', 'is_unlock_day', 'unlock_share', 'unlock_ratio', 'days_to_unlock', 'in_unlock_window']
        float_result = float_result[[c for c in float_cols if c in float_result.columns]]
        result = result.merge(float_result, on=['ts_code', 'trade_date'], how='left')
        
        # 质押
        pledge_result = self._process_pledge(pledge, skeleton)
        pledge_cols = ['ts_code', 'trade_date', 'pledge_ratio', 'pledge_ratio_high']
        pledge_result = pledge_result[[c for c in pledge_cols if c in pledge_result.columns]]
        result = result.merge(pledge_result, on=['ts_code', 'trade_date'], how='left')
        
        # 分红
        dividend_result = self._process_dividend(dividend, skeleton)
        div_cols = ['ts_code', 'trade_date', 'is_dividend_ann', 'cash_div', 'stk_div', 'in_dividend_window']
        dividend_result = dividend_result[[c for c in div_cols if c in dividend_result.columns]]
        result = result.merge(dividend_result, on=['ts_code', 'trade_date'], how='left')
        
        # 4. 生成综合事件信号
        logger.info("生成综合事件信号...")
        
        # 任一事件发生
        event_cols = ['is_repurchase_ann', 'is_unlock_day', 'is_dividend_ann']
        event_cols_present = [c for c in event_cols if c in result.columns]
        if event_cols_present:
            result['has_event'] = 0
            for col in event_cols_present:
                result['has_event'] = result['has_event'] + result[col].fillna(0)
            result['has_event'] = (result['has_event'] > 0).astype('int32')
        
        # 风险事件（解禁+高质押）
        result['has_risk_event'] = 0
        if 'is_unlock_day' in result.columns:
            result['has_risk_event'] = result['has_risk_event'] + result['is_unlock_day'].fillna(0)
        if 'pledge_ratio_high' in result.columns:
            result['has_risk_event'] = result['has_risk_event'] + result['pledge_ratio_high'].fillna(0)
        result['has_risk_event'] = (result['has_risk_event'] > 0).astype('int32')
        
        # 5. 兜底填充所有数值列（确保无 NaN）
        logger.info("执行兜底填充，确保无NaN...")
        
        # 整型字段填0
        int_cols = ['is_repurchase_ann', 'in_repurchase_window', 'is_unlock_day', 'in_unlock_window',
                    'pledge_ratio_high', 'is_dividend_ann', 'in_dividend_window', 'has_event', 'has_risk_event']
        for col in int_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0).astype('int32')
        
        # 浮点型字段填0
        float_cols = ['repurchase_amount', 'unlock_share', 'unlock_ratio', 'pledge_ratio', 'cash_div', 'stk_div']
        for col in float_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0.0)
        
        # days_to_unlock 填 -1（表示无解禁预警）
        if 'days_to_unlock' in result.columns:
            result['days_to_unlock'] = result['days_to_unlock'].fillna(-1).astype('int32')
        
        # 6. 排序
        result = result.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        
        # 7. float64 → float32（节省内存）
        result = self.convert_float64_to_float32(result)
        
        # 验证无NaN
        nan_counts = result.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            logger.warning(f"仍有NaN列: {dict(nan_cols.to_pandas())}")
        else:
            logger.info("✓ 所有字段已清理，无NaN")
        
        logger.info(f"事件信号宽表处理完成，共 {len(result)} 行")
        return result
    
    def save(self, df: cudf.DataFrame):
        """保存处理结果"""
        self.save_parquet(df, self.output_path)
        logger.info(f"事件信号宽表已保存到 {self.output_path}")
