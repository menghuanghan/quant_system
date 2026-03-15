"""
ChipStructureProcessor - 筹码结构宽表处理器（纯cuDF GPU版本）

生成 dwd_chip_structure 宽表，捕捉股东结构和筹码分布

数据源：
    - top10_holders: 十大股东（季度，使用ann_date进行PIT）
    - share_structure: 股本结构/股东户数（季度，使用ann_date进行PIT）

处理逻辑：
    1. PIT 对齐：使用 ann_date（公告日）作为生效日期，严禁使用 end_date
    2. 公告可用性修正：ann_date 后移一个交易日作为可交易生效日
    3. 稀疏转稠密：季度数据通过 ffill 前向填充到每个交易日
    4. 计算筹码集中度等衍生指标
"""

import logging
from typing import Optional

import cudf
import cupy as cp

from .base import BaseProcessor, ffill_by_group_gpu
from .config import (
    DATA_SOURCE_PATHS,
    DWD_OUTPUT_CONFIG,
    PROCESSING_CONFIG,
)

logger = logging.getLogger(__name__)


class ChipStructureProcessor(BaseProcessor):
    """
    筹码结构宽表处理器 - 纯GPU版本
    
    输出字段：
        - trade_date, ts_code: 主键
        # 十大股东
        - top10_hold_ratio: 十大股东持股比例合计
        - top10_hold_amount: 十大股东持股数量合计
        - top1_hold_ratio: 第一大股东持股比例
        - top10_inst_ratio: 十大股东中机构持股比例
        # 股东户数
        - holder_num: 股东户数
        - holder_num_chg: 股东户数变化
        - holder_num_chg_pct: 股东户数变化率
        # PIT元数据
        - chip_report_date: 财报期
        - chip_ann_date: 公告日期
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        super().__init__(use_gpu=use_gpu, start_date=start_date, end_date=end_date)
        self.output_path = DWD_OUTPUT_CONFIG.output_dir / DWD_OUTPUT_CONFIG.chip_structure
    
    def _load_top10_holders(self) -> cudf.DataFrame:
        """加载十大股东数据"""
        logger.info("加载十大股东数据...")
        
        df = self.read_parquet_dir(DATA_SOURCE_PATHS.top10_holders_dir)
        
        if len(df) == 0:
            logger.warning("无法加载十大股东数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'ann_date')
        df = self.normalize_date_column(df, 'end_date')
        
        # 过滤日期范围（使用ann_date进行PIT过滤）
        df = df[df['ann_date'] <= self.end_date]
        
        # ====== PIT修复：过滤end_date > ann_date的脏数据 ======
        # 财报期（end_date）必须早于或等于公告日（ann_date），否则是数据错误
        n_before = len(df)
        df = df[df['end_date'] <= df['ann_date']]
        n_after = len(df)
        if n_before > n_after:
            logger.warning(
                f"PIT过滤: 移除了 {n_before - n_after} 行 end_date > ann_date 的脏数据"
            )
        
        logger.info(f"加载十大股东数据完成，共 {len(df)} 行")
        return df
    
    def _load_share_structure(self) -> cudf.DataFrame:
        """加载股本结构/股东户数数据"""
        logger.info("加载股本结构数据...")
        
        df = self.read_parquet_dir(DATA_SOURCE_PATHS.share_structure_dir)
        
        if len(df) == 0:
            logger.warning("无法加载股本结构数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'ann_date')
        df = self.normalize_date_column(df, 'end_date')
        
        # 过滤日期范围
        df = df[df['ann_date'] <= self.end_date]
        
        # ====== PIT修复：过滤end_date > ann_date的脏数据 ======
        n_before = len(df)
        df = df[df['end_date'] <= df['ann_date']]
        n_after = len(df)
        if n_before > n_after:
            logger.warning(
                f"PIT过滤: 移除了 {n_before - n_after} 行 end_date > ann_date 的脏数据"
            )
        
        logger.info(f"加载股本结构数据完成，共 {len(df)} 行")
        return df
    
    def _aggregate_top10_holders(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """汇总十大股东数据
        
        注意：Tushare top10_holders 接口同时返回"前十大股东"和"前十大流通股东"两组数据，
        每期通常有 20 行记录。必须先过滤只保留"前十大股东"，否则 hold_ratio sum 会翻倍。
        """
        if len(df) == 0:
            return cudf.DataFrame()
        
        logger.info("汇总十大股东数据...")
        
        # ── 关键修复：只保留"前十大股东"，排除"前十大流通股东" ──
        if 'holder_type' in df.columns:
            df['holder_type'] = df['holder_type'].fillna('')
            
            # 检查是否存在 前十大股东/前十大流通股东 分类
            has_classification = df['holder_type'].str.contains('前十大', regex=False).any()
            
            if has_classification:
                n_before = len(df)
                df_total = df[df['holder_type'].str.contains('前十大股东', regex=False) & 
                             ~df['holder_type'].str.contains('流通', regex=False)]
                n_after = len(df_total)
                logger.info(
                    f"过滤十大股东类型: {n_before} → {n_after} 行 "
                    f"(排除了 {n_before - n_after} 行'前十大流通股东'记录)"
                )
                df = df_total
            else:
                # 旧格式数据无 holder_type 分类 → 按股票/报告期只取前10名
                logger.info("holder_type 无'前十大'分类标签，按持股比例取前10名")
                df = df.sort_values(['ts_code', 'ann_date', 'end_date', 'hold_ratio'], 
                                   ascending=[True, True, True, False])
                df['_rank'] = df.groupby(['ts_code', 'ann_date', 'end_date']).cumcount()
                df = df[df['_rank'] < 10].drop(columns=['_rank'])
        
        # 按股票和报告期汇总
        agg_df = df.groupby(['ts_code', 'ann_date', 'end_date']).agg({
            'hold_ratio': 'sum',           # 十大股东持股比例合计
            'hold_amount': 'sum',          # 十大股东持股数量合计
        }).reset_index()
        
        agg_df = agg_df.rename(columns={
            'hold_ratio': 'top10_hold_ratio',
            'hold_amount': 'top10_hold_amount',
        })
        
        # ── 安全钳：top10_hold_ratio 理论上不应超过 100% ──
        overflow = (agg_df['top10_hold_ratio'] > 100).sum()
        if overflow > 0:
            logger.warning(
                f"仍有 {overflow} 条记录 top10_hold_ratio > 100%，执行 clip(0, 100)"
            )
            agg_df['top10_hold_ratio'] = agg_df['top10_hold_ratio'].clip(upper=100.0)
        
        # 计算第一大股东持股比例
        # 按持股比例排序，取最大值
        top1_df = df.sort_values(['ts_code', 'ann_date', 'hold_ratio'], ascending=[True, True, False])
        top1_df = top1_df.drop_duplicates(subset=['ts_code', 'ann_date'], keep='first')
        top1_df = top1_df[['ts_code', 'ann_date', 'hold_ratio']].rename(
            columns={'hold_ratio': 'top1_hold_ratio'}
        )
        
        agg_df = agg_df.merge(top1_df, on=['ts_code', 'ann_date'], how='left')
        
        # 计算机构持股比例
        if 'holder_type' in df.columns:
            holder_type_lower = df['holder_type'].str.lower()
            
            # 使用多个条件判断机构类型
            is_inst = (
                holder_type_lower.str.contains('机构', regex=False) |
                holder_type_lower.str.contains('基金', regex=False) |
                holder_type_lower.str.contains('社保', regex=False) |
                holder_type_lower.str.contains('qfii', regex=False) |
                holder_type_lower.str.contains('保险', regex=False) |
                holder_type_lower.str.contains('券商', regex=False) |
                holder_type_lower.str.contains('信托', regex=False)
            )
            
            inst_df = df[is_inst]
            if len(inst_df) > 0:
                inst_agg = inst_df.groupby(['ts_code', 'ann_date']).agg({
                    'hold_ratio': 'sum'
                }).reset_index()
                inst_agg = inst_agg.rename(columns={'hold_ratio': 'top10_inst_ratio'})
                agg_df = agg_df.merge(inst_agg, on=['ts_code', 'ann_date'], how='left')
            
            if 'top10_inst_ratio' not in agg_df.columns:
                agg_df['top10_inst_ratio'] = 0.0
            agg_df['top10_inst_ratio'] = agg_df['top10_inst_ratio'].fillna(0)
        
        agg_df = agg_df.rename(columns={
            'ann_date': 'chip_ann_date',
            'end_date': 'chip_report_date',
        })
        
        logger.info(f"十大股东数据汇总完成，共 {len(agg_df)} 行")
        return agg_df
    
    def _process_share_structure(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """处理股本结构数据"""
        if len(df) == 0:
            return cudf.DataFrame()
        
        logger.info("处理股本结构数据...")
        
        # 选择需要的字段
        cols = ['ts_code', 'ann_date', 'end_date', 'holder_num']
        df = df[[c for c in cols if c in df.columns]]
        
        if 'holder_num' not in df.columns:
            logger.warning("股本结构数据中没有 holder_num 字段")
            return cudf.DataFrame()
        
        # 去重（保留每个股票每个公告日最新数据）
        df = df.sort_values(['ts_code', 'ann_date', 'end_date'], ascending=[True, True, False])
        df = df.drop_duplicates(subset=['ts_code', 'ann_date'], keep='first')
        
        # 计算股东户数变化
        df = df.sort_values(['ts_code', 'ann_date'])
        df['holder_num_prev'] = df.groupby('ts_code')['holder_num'].shift(1)
        df['holder_num_chg'] = df['holder_num'] - df['holder_num_prev']
        df['holder_num_chg_pct'] = df['holder_num_chg'] / df['holder_num_prev'].where(
            df['holder_num_prev'] > 0, cp.nan
        )
        df = df.drop(columns=['holder_num_prev'])
        
        df = df.rename(columns={
            'ann_date': 'holder_ann_date',
            'end_date': 'holder_report_date',
        })
        
        logger.info(f"股本结构数据处理完成，共 {len(df)} 行")
        return df
    
    def _pit_align_to_daily(
        self,
        df: cudf.DataFrame,
        ann_date_col: str,
        value_cols: list,
        report_date_col: str = None,  # 新增：报告期字段，用于去重时保留最新报告期
    ) -> cudf.DataFrame:
        """
        将低频PIT数据对齐到日频
        
        逻辑：
        1. 将 ann_date 映射到“下一交易日”（防止盘后发布造成当日不可见）
        2. 对于每个stock，在映射后的可用日及之后使用该值
        3. 直到下一个可用日出现新值
        
        Args:
            df: 低频数据
            ann_date_col: 公告日字段名（如 chip_ann_date）
            value_cols: 需要填充的值字段列表
            report_date_col: 报告期字段名（如 chip_report_date），用于同一公告日多条记录时保留最新报告期
        """
        if len(df) == 0:
            return cudf.DataFrame()
        
        logger.info(f"PIT对齐到日频，公告日字段: {ann_date_col}...")
        
        # 获取交易日列表
        trade_dates = self.get_trade_dates()
        
        # 构建所有股票的日期序列
        stocks = df['ts_code'].unique().to_arrow().to_pylist()
        
        # 创建骨架
        dates_df = cudf.DataFrame({'trade_date': trade_dates, '_key': 1})
        stocks_df = cudf.DataFrame({'ts_code': stocks, '_key': 1})
        skeleton = dates_df.merge(stocks_df, on='_key').drop(columns=['_key'])

        # 公告日后移一个交易日，作为可交易生效日
        df = self._map_ann_date_to_next_trade_date(df, ann_date_col, trade_dates)
        if len(df) == 0:
            return cudf.DataFrame()
        
        # **关键修复**：后移映射后，同一股票同一可用日可能有多条记录（如周末公告映射到同一交易日）
        # 需要去重保留最新记录，避免合并时一对多膨胀
        if report_date_col and report_date_col in df.columns:
            logger.info(f"去重：按 (ts_code, trade_date) 保留 {report_date_col} 最新的记录")
            df = df.sort_values(
                ['ts_code', 'trade_date', ann_date_col, report_date_col],
                ascending=[True, True, False, False]
            )
            df = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
        else:
            # 没有报告期字段时，按公告日较新的记录保留
            df = df.sort_values(['ts_code', 'trade_date', ann_date_col], ascending=[True, True, False])
            df = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
        
        # 合并低频数据（按可用日）
        result = skeleton.merge(df, on=['ts_code', 'trade_date'], how='left')
        
        # 前向填充
        result = result.sort_values(['ts_code', 'trade_date'])
        for col in value_cols:
            if col in result.columns:
                result[col] = result.groupby('ts_code')[col].ffill()
        
        return result

    def _map_ann_date_to_next_trade_date(
        self,
        df: cudf.DataFrame,
        ann_date_col: str,
        trade_dates: list,
    ) -> cudf.DataFrame:
        """将公告日映射到下一交易日（严格后移一个交易日）。"""
        if len(df) == 0 or ann_date_col not in df.columns:
            return df

        import bisect

        ann_dates = sorted(df[ann_date_col].dropna().unique().to_arrow().to_pylist())
        mapped_ann_dates = []
        mapped_trade_dates = []

        for ann_date in ann_dates:
            idx = bisect.bisect_right(trade_dates, ann_date)
            if idx < len(trade_dates):
                mapped_ann_dates.append(ann_date)
                mapped_trade_dates.append(trade_dates[idx])

        if not mapped_ann_dates:
            logger.warning(f"{ann_date_col} 后移交易日映射后无可用数据")
            return df.head(0)

        mapping_df = cudf.DataFrame({
            ann_date_col: mapped_ann_dates,
            'trade_date': mapped_trade_dates,
        })

        before = len(df)
        df = df.merge(mapping_df, on=ann_date_col, how='left')

        dropped = int(df['trade_date'].isna().sum())
        if dropped > 0:
            logger.warning(
                f"{ann_date_col} 后移交易日映射: 有 {dropped} 行超出交易日历范围，已丢弃"
            )
            df = df.dropna(subset=['trade_date'])

        logger.info(f"{ann_date_col} 后移交易日映射完成: {before} -> {len(df)} 行")
        return df
    
    def process(self) -> cudf.DataFrame:
        """处理并生成筹码结构宽表"""
        logger.info("开始处理筹码结构宽表...")
        
        # 1. 构建骨架表
        skeleton = self.build_skeleton_table()
        
        # 2. 加载和处理数据源
        top10_holders = self._load_top10_holders()
        share_structure = self._load_share_structure()
        
        # 3. 汇总十大股东数据
        top10_agg = self._aggregate_top10_holders(top10_holders)
        
        # 4. 处理股本结构数据
        share_struct_processed = self._process_share_structure(share_structure)
        
        # 5. PIT对齐到日频
        result = skeleton.copy()
        
        if len(top10_agg) > 0:
            top10_value_cols = ['top10_hold_ratio', 'top10_hold_amount', 'top1_hold_ratio', 
                               'top10_inst_ratio', 'chip_report_date']
            # 修复：传入 report_date_col 以在同一公告日多条记录时保留最新报告期
            top10_daily = self._pit_align_to_daily(
                top10_agg, 'chip_ann_date', top10_value_cols, 
                report_date_col='chip_report_date'
            )
            if len(top10_daily) > 0:
                # 移除重复的骨架列
                top10_daily = top10_daily.drop(columns=['chip_ann_date'], errors='ignore')
                result = result.merge(
                    top10_daily, 
                    on=['ts_code', 'trade_date'], 
                    how='left'
                )
        
        if len(share_struct_processed) > 0:
            holder_value_cols = ['holder_num', 'holder_num_chg', 'holder_num_chg_pct', 'holder_report_date']
            # 修复：传入 report_date_col 以在同一公告日多条记录时保留最新报告期
            holder_daily = self._pit_align_to_daily(
                share_struct_processed, 'holder_ann_date', holder_value_cols,
                report_date_col='holder_report_date'
            )
            if len(holder_daily) > 0:
                holder_daily = holder_daily.drop(columns=['holder_ann_date'], errors='ignore')
                result = result.merge(
                    holder_daily[['ts_code', 'trade_date'] + [c for c in holder_value_cols if c in holder_daily.columns]],
                    on=['ts_code', 'trade_date'],
                    how='left'
                )
        
        # 6. 计算衍生特征
        logger.info("计算筹码集中度特征...")
        
        # 筹码集中度（十大股东持股比例越高，筹码越集中）
        if 'top10_hold_ratio' in result.columns:
            result['chip_concentration'] = result['top10_hold_ratio'] / 100.0  # 归一化
        
        # 股东户数减少表示筹码集中
        if 'holder_num_chg_pct' in result.columns:
            result['holder_decrease'] = (result['holder_num_chg_pct'] < 0).astype('int32')
        
        # 7. 排序
        result = result.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        
        # 8. float64 → float32（节省内存）
        result = self.convert_float64_to_float32(result)
        
        logger.info(f"筹码结构宽表处理完成，共 {len(result)} 行")
        return result
    
    def save(self, df: cudf.DataFrame):
        """保存处理结果"""
        self.save_parquet(df, self.output_path)
        logger.info(f"筹码结构宽表已保存到 {self.output_path}")
