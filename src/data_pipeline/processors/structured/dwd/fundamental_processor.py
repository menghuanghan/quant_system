"""
FundamentalProcessor - PIT基本面宽表处理器（纯cuDF GPU版本）

生成 dwd_stock_fundamental 宽表，严格遵循Point-in-Time原则

核心处理逻辑：
1. 财报数据以 ann_date（公告日）为准，而不是 end_date（报告期）
2. 利润表和现金流量表是YTD累计值，需要先拆分为单季度数据
3. 单季度数据滚动4季求和得到TTM
4. 日频数据通过前向填充（ffill）对齐到每个交易日
"""

import logging
from typing import Optional, List

import cudf
import cupy as cp

from .base import BaseProcessor
from .config import (
    DATA_SOURCE_PATHS,
    DWD_OUTPUT_CONFIG,
    FUNDAMENTAL_CONFIG,
    PROCESSING_CONFIG,
)

logger = logging.getLogger(__name__)


class FundamentalProcessor(BaseProcessor):
    """
    PIT基本面宽表处理器 - 纯GPU版本
    
    输出字段：
        - trade_date, ts_code: 主键
        - total_mv, circ_mv: 市值
        - pe_ttm, pb, ps_ttm: 估值指标
        - revenue_ttm, n_income_attr_p_ttm: TTM指标
        - revenue_sq, n_income_attr_p_sq: 单季度数据
        - roe, roa, gross_margin: 财务指标
        - revenue_yoy, net_profit_yoy: 增长率
        - report_date, ann_date: PIT元数据
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        super().__init__(use_gpu=use_gpu, start_date=start_date, end_date=end_date)
        self.output_path = DWD_OUTPUT_CONFIG.output_dir / DWD_OUTPUT_CONFIG.stock_fundamental
    
    def _load_income_statement(self) -> cudf.DataFrame:
        """加载利润表数据"""
        logger.info("加载利润表数据...")
        
        df = self.read_parquet_dir(DATA_SOURCE_PATHS.income_statement_dir)
        
        if len(df) == 0:
            raise ValueError("无法加载利润表数据")
        
        # 标准化日期
        df = self.normalize_date_column(df, 'ann_date')
        df = self.normalize_date_column(df, 'end_date')
        
        # 保留end_date为YYYYMMDD格式用于季度判断
        df['end_date_str'] = df['end_date'].astype(str).str.replace('-', '')
        
        # 过滤report_type=1（合并报表）
        df = df[df['report_type'] == '1']
        
        logger.info(f"加载利润表数据完成，共 {len(df)} 行")
        return df
    
    def _load_balance_sheet(self) -> cudf.DataFrame:
        """加载资产负债表数据"""
        logger.info("加载资产负债表数据...")
        
        df = self.read_parquet_dir(DATA_SOURCE_PATHS.balance_sheet_dir)
        
        if len(df) == 0:
            raise ValueError("无法加载资产负债表数据")
        
        df = self.normalize_date_column(df, 'ann_date')
        df = self.normalize_date_column(df, 'end_date')
        df['end_date_str'] = df['end_date'].astype(str).str.replace('-', '')
        
        df = df[df['report_type'] == '1']
        
        logger.info(f"加载资产负债表数据完成，共 {len(df)} 行")
        return df
    
    def _load_cash_flow(self) -> cudf.DataFrame:
        """加载现金流量表数据"""
        logger.info("加载现金流量表数据...")
        
        df = self.read_parquet_dir(DATA_SOURCE_PATHS.cash_flow_dir)
        
        if len(df) == 0:
            logger.warning("无法加载现金流量表数据")
            return cudf.DataFrame() if self.use_gpu else None
        
        df = self.normalize_date_column(df, 'ann_date')
        df = self.normalize_date_column(df, 'end_date')
        df['end_date_str'] = df['end_date'].astype(str).str.replace('-', '')
        
        df = df[df['report_type'] == '1']
        
        logger.info(f"加载现金流量表数据完成，共 {len(df)} 行")
        return df
    
    def _load_financial_indicator(self) -> cudf.DataFrame:
        """加载财务指标数据"""
        logger.info("加载财务指标数据...")
        
        df = self.read_parquet_dir(DATA_SOURCE_PATHS.financial_indicator_dir)
        
        if len(df) == 0:
            logger.warning("无法加载财务指标数据")
            return cudf.DataFrame() if self.use_gpu else None
        
        df = self.normalize_date_column(df, 'ann_date')
        df = self.normalize_date_column(df, 'end_date')
        df['end_date_str'] = df['end_date'].astype(str).str.replace('-', '')
        
        logger.info(f"加载财务指标数据完成，共 {len(df)} 行")
        return df
    
    def _load_daily_basic(self) -> cudf.DataFrame:
        """加载每日基本指标（市值、估值等）"""
        logger.info("加载每日基本指标数据...")
        
        df = self.read_parquet_dir(
            DATA_SOURCE_PATHS.daily_basic_dir,
            columns=['ts_code', 'trade_date', 'total_mv', 'circ_mv', 'pe_ttm', 'pb', 'ps_ttm']
        )
        
        if len(df) == 0:
            raise ValueError("无法加载每日基本指标数据")
        
        df = self.normalize_date_column(df, 'trade_date')
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载每日基本指标数据完成，共 {len(df)} 行")
        return df
    
    def _get_quarter(self, end_date_str: cudf.Series) -> cudf.Series:
        """从end_date提取季度（1-4）"""
        month = end_date_str.str.slice(4, 6).astype('int32')
        quarter = cudf.Series(cp.zeros(len(month), dtype='int32'))
        quarter = quarter.where(month != 3, 1)
        quarter = quarter.where(month != 6, 2)
        quarter = quarter.where(month != 9, 3)
        quarter = quarter.where(month != 12, 4)
        return quarter
    
    def _calculate_single_quarter(
        self,
        df: cudf.DataFrame,
        cumulative_fields: List[str],
    ) -> cudf.DataFrame:
        """
        计算单季度数据（纯GPU操作）
        
        单季度 = 当季累计 - 上季累计
        Q1单季 = Q1累计（无需减）
        """
        logger.info("计算单季度数据...")
        
        # 添加季度和年份列
        df['quarter'] = self._get_quarter(df['end_date_str'])
        df['year'] = df['end_date_str'].str.slice(0, 4).astype('int32')
        
        # 过滤掉非季末日期
        df = df[df['quarter'] > 0]
        
        # 按股票和年份排序
        df = df.sort_values(['ts_code', 'year', 'quarter'])
        
        # 为每个累计字段计算单季度值
        for field in cumulative_fields:
            if field not in df.columns:
                continue
            
            sq_field = f"{field}_sq"
            
            # 上一季累计值（同一年内）
            df['prev_cum'] = df.groupby(['ts_code', 'year'])[field].shift(1)
            
            # Q1的前一季度累计值应该是0
            df['prev_cum'] = df['prev_cum'].where(df['quarter'] != 1, 0)
            df['prev_cum'] = df['prev_cum'].fillna(0)
            
            # 单季度 = 当季累计 - 上季累计
            df[sq_field] = df[field] - df['prev_cum']
        
        df = df.drop(columns=['prev_cum'])
        
        logger.info(f"单季度计算完成，处理了 {len(cumulative_fields)} 个字段")
        return df
    
    def _calculate_ttm(
        self,
        df: cudf.DataFrame,
        sq_fields: List[str],
    ) -> cudf.DataFrame:
        """
        计算TTM（滚动12个月/最近4季度）指标
        
        TTM = 最近4个季度的单季度值之和
        
        由于cuDF的groupby rolling与numba有兼容性问题，
        这里使用shift的方式实现滚动求和
        """
        logger.info("计算TTM指标...")
        
        df = df.sort_values(['ts_code', 'year', 'quarter'])
        
        for field in sq_fields:
            sq_field = f"{field}_sq"
            ttm_field = f"{field}_ttm"
            
            if sq_field not in df.columns:
                continue
            
            # 使用shift实现滚动4季求和
            # TTM = 当季 + 前1季 + 前2季 + 前3季
            sq = df[sq_field]
            sq_lag1 = df.groupby('ts_code')[sq_field].shift(1)
            sq_lag2 = df.groupby('ts_code')[sq_field].shift(2)
            sq_lag3 = df.groupby('ts_code')[sq_field].shift(3)
            
            df[ttm_field] = sq + sq_lag1 + sq_lag2 + sq_lag3
        
        logger.info(f"TTM计算完成，处理了 {len(sq_fields)} 个字段")
        return df
    
    def _prepare_pit_fundamental(
        self,
        income_df: cudf.DataFrame,
        balance_df: cudf.DataFrame,
        indicator_df: cudf.DataFrame,
    ) -> cudf.DataFrame:
        """准备PIT基本面数据"""
        logger.info("准备PIT基本面数据...")
        
        # 1. 处理利润表：单季拆分 + TTM
        income_fields = [f for f in FUNDAMENTAL_CONFIG.INCOME_CUMULATIVE_FIELDS 
                        if f in income_df.columns]
        
        if income_fields:
            income_sq = self._calculate_single_quarter(income_df, income_fields)
            income_ttm = self._calculate_ttm(income_sq, income_fields)
        else:
            income_ttm = income_df
        
        # 选择利润表输出字段
        income_output_cols = ['ts_code', 'ann_date', 'end_date', 'quarter', 'year']
        income_output_cols += [f"{f}_sq" for f in ['revenue', 'n_income_attr_p'] 
                               if f"{f}_sq" in income_ttm.columns]
        income_output_cols += [f"{f}_ttm" for f in ['revenue', 'operate_profit', 
                               'total_profit', 'n_income_attr_p'] 
                               if f"{f}_ttm" in income_ttm.columns]
        
        income_output = income_ttm[[c for c in income_output_cols if c in income_ttm.columns]]
        
        # 2. 处理资产负债表（时点值，无需拆分）
        balance_output_cols = ['ts_code', 'ann_date', 'end_date', 'total_assets', 
                               'total_liab', 'total_hldr_eqy_exc_min_int']
        balance_output = balance_df[[c for c in balance_output_cols if c in balance_df.columns]]
        
        if 'total_hldr_eqy_exc_min_int' in balance_output.columns:
            balance_output = balance_output.rename(columns={'total_hldr_eqy_exc_min_int': 'total_equity'})
        
        # 3. 处理财务指标
        if len(indicator_df) > 0:
            indicator_cols = ['ts_code', 'ann_date', 'end_date', 'roe', 'roa', 
                             'gross_margin', 'netprofit_margin', 'debt_to_assets',
                             'netprofit_yoy', 'or_yoy']
            indicator_output = indicator_df[[c for c in indicator_cols if c in indicator_df.columns]]
            
            # 重命名
            rename_map = {'or_yoy': 'revenue_yoy', 'netprofit_yoy': 'net_profit_yoy'}
            for old, new in rename_map.items():
                if old in indicator_output.columns:
                    indicator_output = indicator_output.rename(columns={old: new})
        else:
            indicator_output = cudf.DataFrame()
        
        # 4. 合并所有财务数据
        fundamental = income_output.copy()
        
        if len(balance_output) > 0:
            balance_merge = balance_output.drop_duplicates(
                subset=['ts_code', 'ann_date', 'end_date'], keep='last'
            )
            fundamental = fundamental.merge(
                balance_merge,
                on=['ts_code', 'ann_date', 'end_date'],
                how='outer'
            )
        
        if len(indicator_output) > 0:
            indicator_merge = indicator_output.drop_duplicates(
                subset=['ts_code', 'ann_date', 'end_date'], keep='last'
            )
            fundamental = fundamental.merge(
                indicator_merge,
                on=['ts_code', 'ann_date', 'end_date'],
                how='outer'
            )
        
        # 重命名end_date为report_date
        fundamental = fundamental.rename(columns={'end_date': 'report_date'})
        
        logger.info(f"PIT基本面数据准备完成，共 {len(fundamental)} 行")
        return fundamental
    
    def _get_full_trade_dates(self) -> list:
        """获取完整历史交易日（不限制 start_date，用于 ffill 需要历史数据的场景）"""
        cal_df = cudf.read_parquet(str(DATA_SOURCE_PATHS.trade_calendar))
        cal_df = cal_df[cal_df['is_open'] == 1]
        cal_df = self.normalize_date_column(cal_df, 'cal_date')
        # 只过滤 end_date，不过滤 start_date
        cal_df = cal_df[cal_df['cal_date'] <= self.end_date]
        trade_dates = sorted(cal_df['cal_date'].unique().to_arrow().to_pylist())
        return trade_dates
    
    def _resample_to_daily(
        self,
        fundamental: cudf.DataFrame,
        daily_basic: cudf.DataFrame,
    ) -> cudf.DataFrame:
        """
        将季频财务数据重采样到日频（纯GPU操作）
        
        使用PIT原则：数据只能在公告日之后使用
        
        注意：使用完整历史交易日构建骨架表，确保ffill能正确填充历史数据，
        输出日期范围的过滤在 process() 方法中进行。
        """
        logger.info("将财务数据重采样到日频...")
        
        # 获取所有股票和完整历史交易日（不限制start_date）
        trade_dates = self._get_full_trade_dates()
        stock_codes = fundamental['ts_code'].unique().to_arrow().to_pylist()
        
        logger.info(f"重采样: {len(stock_codes)} 只股票, {len(trade_dates)} 个交易日")
        
        # 构建骨架表
        skeleton = cudf.DataFrame({
            'trade_date': trade_dates * len(stock_codes),
            'ts_code': [code for code in stock_codes for _ in trade_dates]
        })
        
        # 将财务数据按公告日对齐
        fundamental['trade_date'] = fundamental['ann_date']
        
        # **关键修复**：同一股票同一公告日可能有多条记录（不同报告期），需要去重保留最新报告期
        # 先按报告期降序排序，然后去重保留第一条（即最新报告期）
        if 'report_date' in fundamental.columns:
            logger.info("去重：按 (ts_code, trade_date) 保留 report_date 最新的记录")
            fundamental = fundamental.sort_values(
                ['ts_code', 'trade_date', 'report_date'], 
                ascending=[True, True, False]
            )
            dup_before = len(fundamental)
            fundamental = fundamental.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
            if len(fundamental) < dup_before:
                logger.info(f"去重: {dup_before} → {len(fundamental)} 行")
        
        # 选择需要填充的列
        ffill_cols = [col for col in fundamental.columns 
                     if col not in ['ts_code', 'trade_date', 'ann_date']]
        
        # 合并到骨架表
        result = skeleton.merge(
            fundamental.drop(columns=['ann_date']),
            on=['ts_code', 'trade_date'],
            how='left'
        )
        
        # 按股票分组前向填充
        result = result.sort_values(['ts_code', 'trade_date'])
        
        for col in ffill_cols:
            if col in result.columns:
                result[col] = result.groupby('ts_code')[col].ffill()
        
        # 合并每日市值估值数据
        logger.info("合并每日市值估值数据...")
        result = result.merge(
            daily_basic,
            on=['ts_code', 'trade_date'],
            how='left'
        )
        
        logger.info(f"日频重采样完成，共 {len(result)} 行")
        return result
    
    def process(self) -> cudf.DataFrame:
        """执行数据处理 - 纯GPU操作"""
        # 1. 加载财务报表数据
        income_df = self._load_income_statement()
        balance_df = self._load_balance_sheet()
        indicator_df = self._load_financial_indicator()
        daily_basic = self._load_daily_basic()
        
        # 2. 准备PIT基本面数据
        fundamental = self._prepare_pit_fundamental(
            income_df, balance_df, indicator_df
        )
        
        del income_df, balance_df, indicator_df
        
        # 3. 重采样到日频
        df = self._resample_to_daily(fundamental, daily_basic)
        
        del fundamental, daily_basic
        
        # 4. 选择输出列
        output_columns = [
            'trade_date', 'ts_code',
            'total_mv', 'circ_mv', 'pe_ttm', 'pb', 'ps_ttm',
            'revenue_ttm', 'operate_profit_ttm', 'total_profit_ttm', 'n_income_attr_p_ttm',
            'revenue_sq', 'n_income_attr_p_sq',
            'roe', 'roa', 'gross_margin', 'netprofit_margin', 'debt_to_assets',
            'revenue_yoy', 'net_profit_yoy',
            'total_assets', 'total_liab', 'total_equity',
            'report_date',
        ]
        
        existing_columns = [col for col in output_columns if col in df.columns]
        df = df[existing_columns]
        df = df.sort_values(['trade_date', 'ts_code'])
        
        # 5. 按 start_date 过滤输出（骨架表包含完整历史用于ffill，输出只保留目标日期范围）
        logger.info(f"过滤输出日期范围: {self.start_date} ~ {self.end_date}")
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        logger.info(f"过滤后数据行数: {len(df)}")
        
        return df
    
    def save(self, df: cudf.DataFrame):
        """保存处理结果"""
        self.save_parquet(df, self.output_path)


def main():
    """独立运行测试"""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    processor = FundamentalProcessor(use_gpu=True)
    df = processor.run()
    
    print(f"\n输出数据样例：")
    print(df.head(10).to_arrow().to_pandas())
    
    return df


if __name__ == "__main__":
    main()
