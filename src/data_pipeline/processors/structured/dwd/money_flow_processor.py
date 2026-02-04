"""
MoneyFlowProcessor - 资金博弈宽表处理器（纯cuDF GPU版本）

生成 dwd_money_flow 宽表，捕捉机构动向和市场情绪

数据源：
    - money_flow: 个股资金流向（日频）
    - margin_detail: 融资融券明细（日频）
    - top_list: 龙虎榜（日频，稀疏）
    - top_inst: 龙虎榜机构明细（日频，稀疏）
    - hsgt_flow: 沪深港通资金流向（日频，市场级）

处理逻辑：
    1. 日频数据直接与骨架表 Join
    2. 缺失值填充为 0（表示当日无相关交易）
    3. 计算净流入比例等衍生指标
    4. 沪深港通数据作为市场级特征广播到每个股票
"""

import logging
from typing import Optional

import cudf
import cupy as cp

from .base import BaseProcessor
from .config import (
    DATA_SOURCE_PATHS,
    DWD_OUTPUT_CONFIG,
    PROCESSING_CONFIG,
)

logger = logging.getLogger(__name__)


class MoneyFlowProcessor(BaseProcessor):
    """
    资金博弈宽表处理器 - 纯GPU版本
    
    输出字段：
        - trade_date, ts_code: 主键
        # 资金流向
        - buy_sm_amount, sell_sm_amount: 小单买卖金额
        - buy_md_amount, sell_md_amount: 中单买卖金额
        - buy_lg_amount, sell_lg_amount: 大单买卖金额
        - buy_elg_amount, sell_elg_amount: 超大单买卖金额
        - net_mf_amount: 主力净流入金额
        - net_mf_amount_pct: 主力净流入占比
        # 两融
        - rzye: 融资余额
        - rqye: 融券余额
        - rzmre: 融资买入额
        - rzche: 融资偿还额
        - rzrqye: 融资融券余额
        # 龙虎榜
        - is_top_list: 是否上龙虎榜
        - top_net_amount: 龙虎榜净买入
        - top_inst_net_buy: 机构净买入
        # 沪深港通
        - hsgt_north: 北向资金净流入
        - hsgt_north_ma5, hsgt_north_ma20: 北向资金5/20日移动平均
        # 数据质量标志
        - is_bj_stock: 是否北交所股票（北交所无分单明细数据，模型需感知）
    
    缺失值处理：
        - 所有数值字段：NaN → 0（确保无泄漏到模型层）
        - 龙虎榜字段：未上榜股票填0
        - 两融余额字段：前向填充后再填0（从未纳入两融的股票）
        - 北交所股票：明细数据不可靠，通过is_bj_stock标识
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        super().__init__(use_gpu=use_gpu, start_date=start_date, end_date=end_date)
        self.output_path = DWD_OUTPUT_CONFIG.output_dir / DWD_OUTPUT_CONFIG.money_flow
    
    def _load_money_flow(self) -> cudf.DataFrame:
        """加载个股资金流向数据"""
        logger.info("加载个股资金流向数据...")
        
        df = self.read_parquet_dir(DATA_SOURCE_PATHS.money_flow_dir)
        
        if len(df) == 0:
            logger.warning("无法加载资金流向数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'trade_date')
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        # 选择核心字段
        cols = [
            'ts_code', 'trade_date',
            'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount',
            'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
            'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
            'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
            'net_mf_vol', 'net_mf_amount'
        ]
        df = df[[c for c in cols if c in df.columns]]
        
        logger.info(f"加载资金流向数据完成，共 {len(df)} 行")
        return df
    
    def _load_margin_detail(self) -> cudf.DataFrame:
        """加载融资融券明细数据"""
        logger.info("加载融资融券明细数据...")
        
        df = self.read_parquet_dir(DATA_SOURCE_PATHS.margin_detail_dir)
        
        if len(df) == 0:
            logger.warning("无法加载融资融券数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'trade_date')
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        # 选择核心字段
        cols = [
            'ts_code', 'trade_date',
            'rzye', 'rqye', 'rzmre', 'rqyl', 'rzche', 'rqchl', 'rqmcl', 'rzrqye'
        ]
        df = df[[c for c in cols if c in df.columns]]
        
        logger.info(f"加载融资融券数据完成，共 {len(df)} 行")
        return df
    
    def _load_top_list(self) -> cudf.DataFrame:
        """加载龙虎榜数据"""
        logger.info("加载龙虎榜数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.top_list)
        
        if len(df) == 0:
            logger.warning("无法加载龙虎榜数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'trade_date')
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        # 按股票-日期汇总（一只股票一天可能多次上榜）
        agg_df = df.groupby(['ts_code', 'trade_date']).agg({
            'amount': 'sum',           # 总成交额
            'l_buy': 'sum',            # 买入总额
            'l_sell': 'sum',           # 卖出总额
            'net_amount': 'sum',       # 净买入
        }).reset_index()
        
        agg_df = agg_df.rename(columns={
            'amount': 'top_amount',
            'l_buy': 'top_l_buy',
            'l_sell': 'top_l_sell',
            'net_amount': 'top_net_amount'
        })
        agg_df['is_top_list'] = 1
        
        logger.info(f"加载龙虎榜数据完成，共 {len(agg_df)} 行（汇总后）")
        return agg_df
    
    def _load_top_inst(self) -> cudf.DataFrame:
        """加载龙虎榜机构明细数据"""
        logger.info("加载龙虎榜机构明细数据...")
        
        df = self.read_parquet_dir(DATA_SOURCE_PATHS.top_inst_dir)
        
        if len(df) == 0:
            logger.warning("无法加载龙虎榜机构明细数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'trade_date')
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        # 按股票-日期汇总机构净买入
        agg_df = df.groupby(['ts_code', 'trade_date']).agg({
            'buy': 'sum',
            'sell': 'sum',
            'net_buy': 'sum',
        }).reset_index()
        
        agg_df = agg_df.rename(columns={
            'buy': 'top_inst_buy',
            'sell': 'top_inst_sell',
            'net_buy': 'top_inst_net_buy'
        })
        
        logger.info(f"加载龙虎榜机构明细完成，共 {len(agg_df)} 行（汇总后）")
        return agg_df
    
    def _load_hsgt_flow(self) -> cudf.DataFrame:
        """
        加载沪深港通资金流向数据（市场级数据）
        
        字段说明：
            - hgt: 沪股通净流入（亿元）
            - sgt: 深股通净流入（亿元）
            - north_money: 北向资金总净流入（亿元）
            - ggt_ss: 港股通(沪)净流入
            - ggt_sz: 港股通(深)净流入
            - south_money: 南向资金总净流入
        """
        logger.info("加载沪深港通资金流向数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.hsgt_flow)
        
        if len(df) == 0:
            logger.warning("无法加载沪深港通资金流向数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'trade_date')
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        # 选择并重命名字段
        rename_map = {
            'hgt': 'hsgt_hgt',           # 沪股通
            'sgt': 'hsgt_sgt',           # 深股通
            'north_money': 'hsgt_north', # 北向资金
            'ggt_ss': 'hsgt_ggt_ss',     # 港股通(沪)
            'ggt_sz': 'hsgt_ggt_sz',     # 港股通(深)
            'south_money': 'hsgt_south', # 南向资金
        }
        
        for old_name, new_name in rename_map.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        cols = ['trade_date'] + [v for v in rename_map.values() if v in df.columns]
        df = df[cols]
        
        logger.info(f"加载沪深港通资金流向完成，共 {len(df)} 行")
        return df
    
    def _calculate_derived_features(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """计算衍生特征"""
        logger.info("计算资金流衍生特征...")
        
        # 主力资金（大单+超大单）
        if 'buy_lg_amount' in df.columns and 'buy_elg_amount' in df.columns:
            df['buy_main_amount'] = df['buy_lg_amount'].fillna(0) + df['buy_elg_amount'].fillna(0)
            df['sell_main_amount'] = df['sell_lg_amount'].fillna(0) + df['sell_elg_amount'].fillna(0)
            df['net_main_amount'] = df['buy_main_amount'] - df['sell_main_amount']
        
        # 散户资金（小单+中单）
        if 'buy_sm_amount' in df.columns and 'buy_md_amount' in df.columns:
            df['buy_retail_amount'] = df['buy_sm_amount'].fillna(0) + df['buy_md_amount'].fillna(0)
            df['sell_retail_amount'] = df['sell_sm_amount'].fillna(0) + df['sell_md_amount'].fillna(0)
            df['net_retail_amount'] = df['buy_retail_amount'] - df['sell_retail_amount']
        
        # 计算主力净流入占比（需要成交额数据，这里用买卖总额近似）
        if 'buy_main_amount' in df.columns and 'sell_main_amount' in df.columns:
            total_amount = (df['buy_main_amount'] + df['sell_main_amount'] + 
                          df['buy_retail_amount'] + df['sell_retail_amount'])
            # 避免除零
            total_amount = total_amount.where(total_amount > 0, cp.nan)
            df['net_main_amount_pct'] = df['net_main_amount'] / total_amount
        
        # 融资余额变化
        if 'rzye' in df.columns:
            df = df.sort_values(['ts_code', 'trade_date'])
            df['rzye_chg'] = df.groupby('ts_code')['rzye'].diff()
        
        # 北向资金动量特征（市场整体流入的滞后值作为动量指标）
        # 计算北向资金流入5日/20日移动平均
        if 'hsgt_north' in df.columns:
            # 先对每日北向资金计算（市场级，不需要groupby）
            unique_dates = df[['trade_date', 'hsgt_north']].drop_duplicates().sort_values('trade_date')
            
            # 计算5日和20日累计（用cumsum-shift模式近似rolling）
            unique_dates['hsgt_north_cumsum'] = unique_dates['hsgt_north'].fillna(0).cumsum()
            unique_dates['hsgt_north_ma5'] = (
                unique_dates['hsgt_north_cumsum'] - 
                unique_dates['hsgt_north_cumsum'].shift(5).fillna(0)
            ) / 5
            unique_dates['hsgt_north_ma20'] = (
                unique_dates['hsgt_north_cumsum'] - 
                unique_dates['hsgt_north_cumsum'].shift(20).fillna(0)
            ) / 20
            
            # 合并回主表
            unique_dates = unique_dates[['trade_date', 'hsgt_north_ma5', 'hsgt_north_ma20']]
            df = df.merge(unique_dates, on='trade_date', how='left')
        
        logger.info("衍生特征计算完成")
        return df
    
    def process(self) -> cudf.DataFrame:
        """处理并生成资金博弈宽表"""
        logger.info("开始处理资金博弈宽表...")
        
        # 1. 构建骨架表
        skeleton = self.build_skeleton_table()
        
        # 2. 加载各数据源
        money_flow = self._load_money_flow()
        margin_detail = self._load_margin_detail()
        top_list = self._load_top_list()
        top_inst = self._load_top_inst()
        hsgt_flow = self._load_hsgt_flow()
        
        # 3. 合并数据（左连接骨架表）
        logger.info("合并数据到骨架表...")
        
        result = skeleton
        
        if len(money_flow) > 0:
            result = result.merge(money_flow, on=['ts_code', 'trade_date'], how='left')
        
        if len(margin_detail) > 0:
            result = result.merge(margin_detail, on=['ts_code', 'trade_date'], how='left')
        
        if len(top_list) > 0:
            result = result.merge(top_list, on=['ts_code', 'trade_date'], how='left')
        
        if len(top_inst) > 0:
            result = result.merge(top_inst, on=['ts_code', 'trade_date'], how='left')
        
        # 沪深港通是市场级数据，只按trade_date合并（广播到所有股票）
        if len(hsgt_flow) > 0:
            result = result.merge(hsgt_flow, on='trade_date', how='left')
        
        # 4. 填充缺失值（确保无 NaN 泄漏到模型层）
        logger.info("填充缺失值...")
        
        # 4.1 资金流数据缺失意味着当日无交易或无数据，填充为0
        amount_cols = [c for c in result.columns if 'amount' in c.lower() or 'vol' in c.lower()]
        for col in amount_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)
        
        # 4.2 龙虎榜字段填充为0（未上榜=无资金流入流出）
        top_list_cols = [
            'is_top_list', 'top_amount', 'top_l_buy', 'top_l_sell', 'top_net_amount',
            'top_inst_buy', 'top_inst_sell', 'top_inst_net_buy'
        ]
        for col in top_list_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)
        if 'is_top_list' in result.columns:
            result['is_top_list'] = result['is_top_list'].astype('int32')
        
        # 4.3 融资融券处理
        # 余额字段：先前向填充（持仓概念），再填0（从未被纳入两融的股票）
        margin_balance_cols = ['rzye', 'rqye', 'rzrqye']
        for col in margin_balance_cols:
            if col in result.columns:
                result = result.sort_values(['ts_code', 'trade_date'])
                result[col] = result.groupby('ts_code')[col].ffill()
                result[col] = result[col].fillna(0)  # 从未纳入两融的股票填0
        
        # 流量字段：直接填0（当日无融资融券操作）
        margin_flow_cols = ['rzmre', 'rqyl', 'rzche', 'rqchl', 'rqmcl', 'rzye_chg']
        for col in margin_flow_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)
        
        # 4.4 沪深港通字段填充为0（无数据日期填0）
        hsgt_cols = [c for c in result.columns if c.startswith('hsgt_')]
        for col in hsgt_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)
        
        # 4.5 添加北交所标志位（北交所无分单明细数据，模型需感知）
        result['is_bj_stock'] = result['ts_code'].str.endswith('.BJ').astype('int32')
        
        # 5. 计算衍生特征
        result = self._calculate_derived_features(result)
        
        # 6. 兜底：确保所有数值字段无NaN（衍生特征如 net_main_amount_pct 可能产生NaN）
        logger.info("执行兜底NaN填充...")
        numeric_cols = result.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
        for col in numeric_cols:
            if col not in ['trade_date']:  # 排除日期字段
                nan_count = result[col].isna().sum()
                if nan_count > 0:
                    result[col] = result[col].fillna(0)
                    logger.debug(f"  {col}: 填充 {nan_count} 个NaN为0")
        
        # 7. 排序
        result = result.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        
        logger.info(f"资金博弈宽表处理完成，共 {len(result)} 行")
        return result
    
    def save(self, df: cudf.DataFrame):
        """保存处理结果"""
        self.save_parquet(df, self.output_path)
        logger.info(f"资金博弈宽表已保存到 {self.output_path}")
