"""
MacroEnvProcessor - 宏观环境宽表处理器（纯cuDF GPU版本）

生成 dwd_macro_env 宽表，为模型增加"环境感知"能力（Market Regime）

数据源：
    基础宏观：
    - cn_gdp: GDP（季度）
    - cn_cpi: CPI（月度）
    - cn_ppi: PPI工业生产者出厂价格（月度）
    - cn_pmi: PMI（月度）
    - cn_m2: M2货币供应量（月度）
    - lpr: LPR利率（不定期）
    - shibor: SHIBOR利率（日度）
    - market_congestion: 市场拥挤度（日度）
    - stock_bond_spread: 股债利差（日度）
    
    深度风险因子：
    - a_pe_pb_ew_median: A股估值中位数（判断市场低估）
    - buffett_indicator: 巴菲特指标（总市值/GDP，长周期择时）
    - break_net_stock: 破净股占比（极度悲观信号）
    
    指数与基准：
    - index_daily: 核心指数（沪深300、中证500、创业板指等）
    
    衍生品数据：
    - repo_daily: 回购利率（GC001、R-001）
    - fut_daily: 股指期货（IF、IC、IH、IM）
"""

import logging
from typing import Optional, Dict, List
from pathlib import Path
import os

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


class MacroEnvProcessor(BaseProcessor):
    """
    宏观环境宽表处理器 - 纯GPU版本
    
    输出字段（主键：trade_date）：
        # GDP相关（季度）
        - gdp_yoy: GDP同比增速
        
        # CPI相关（月度）
        - cpi_yoy: CPI同比
        - cpi_mom: CPI环比
        
        # PPI相关（月度）
        - ppi_yoy: PPI工业生产者出厂价格同比
        
        # PMI相关（月度）
        - pmi: 制造业PMI
        - pmi_prod: PMI生产指数
        - pmi_new_order: PMI新订单指数
        
        # 货币供应（月度）
        - m2: M2货币供应量
        - m2_yoy: M2同比增速
        
        # 利率（日度/不定期）
        - lpr_1y: 1年期LPR
        - lpr_5y: 5年期LPR
        - shibor_on: SHIBOR隔夜
        - shibor_1w/1m/3m/6m/1y: SHIBOR各期限
        
        # 市场风险指标（日度）
        - market_congestion: 市场拥挤度
        - stock_bond_spread: 股债利差
        
        # 深度风险因子（日度）
        - pb_median: A股PB中位数
        - pb_ew: A股PB等权均值
        - pb_quantile_10y: PB近10年分位数
        - buffett_indicator: 巴菲特指标（总市值/GDP）
        - buffett_quantile_10y: 巴菲特指标近10年分位数
        - break_net_ratio: 破净股占比
        
        # 指数基准（日度）
        - sh300_pct_chg: 沪深300涨跌幅
        - sh300_amplitude: 沪深300振幅
        - sh300_turnover: 沪深300换手率代理
        - zz500_pct_chg: 中证500涨跌幅
        - cyb_pct_chg: 创业板指涨跌幅
        - kc50_pct_chg: 科创50涨跌幅
        - zz1000_pct_chg: 中证1000涨跌幅
        
        # 流动性（日度）
        - liquidity_gc001_close: GC001回购利率收盘价
        - liquidity_gc001_weight: GC001加权利率
        - liquidity_r001_close: R-001回购利率收盘价
        
        # 期货基差（日度）
        - if_basis_rate: IF基差率（贴水为负）
        - ic_basis_rate: IC基差率
        - ih_basis_rate: IH基差率
        - im_basis_rate: IM基差率
        - if_total_oi: IF总持仓量
        - ic_total_oi: IC总持仓量
        
        # 衍生指标
        - pmi_regime: PMI状态（0收缩/1扩张/2强扩张）
        - macro_regime: 综合宏观状态
    """
    
    # 核心指数配置
    CORE_INDICES = {
        '000300_SH': ('sh300', '沪深300'),
        '000905_SH': ('zz500', '中证500'),
        '399006_SZ': ('cyb', '创业板指'),
        '000016_SH': ('sz50', '上证50'),
        '000852_SH': ('zz1000', '中证1000'),
        '000688_SH': ('kc50', '科创50'),
    }
    
    # 期货品种配置
    FUTURES_CONFIG = {
        'IF': {'spot_index': '000300_SH', 'name': '沪深300'},
        'IC': {'spot_index': '000905_SH', 'name': '中证500'},
        'IH': {'spot_index': '000016_SH', 'name': '上证50'},
        'IM': {'spot_index': '000852_SH', 'name': '中证1000'},
    }
    
    # ========================================================================
    # 产品上市日期配置 (防止Look-Ahead Bias)
    # ========================================================================
    PRODUCT_LAUNCH_DATES = {
        # 股指期货上市日
        'IF': '2010-04-16',  # 沪深300期货
        'IC': '2015-04-16',  # 中证500期货
        'IH': '2015-04-16',  # 上证50期货
        'IM': '2022-07-22',  # 中证1000期货
        # 指数基日/上市日
        'kc50': '2019-12-31',   # 科创50基日
        'zz1000': '2014-10-17', # 中证1000发布日
        # 利率改革日
        'lpr': '2019-08-20',    # LPR改革日（新报价机制）
    }
    
    def __init__(
        self,
        use_gpu: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        super().__init__(use_gpu=use_gpu, start_date=start_date, end_date=end_date)
        self.output_path = DWD_OUTPUT_CONFIG.output_dir / DWD_OUTPUT_CONFIG.macro_env
    
    def _build_date_skeleton(self) -> cudf.DataFrame:
        """构建交易日骨架（不含股票维度）"""
        trade_dates = self.get_trade_dates()
        return cudf.DataFrame({'trade_date': trade_dates})
    
    def _load_gdp(self) -> cudf.DataFrame:
        """加载GDP数据"""
        logger.info("加载GDP数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.cn_gdp)
        
        if len(df) == 0:
            logger.warning("无法加载GDP数据")
            return cudf.DataFrame()
        
        # quarter格式: 2024Q1, 2024Q2等
        # 需要转换为trade_date（取季末日期作为数据发布的大致日期）
        df['quarter_str'] = df['quarter'].astype(str)
        df['year'] = df['quarter_str'].str.slice(0, 4).astype('int32')
        df['q'] = df['quarter_str'].str.slice(5, 6).astype('int32')
        
        # 季末日期映射（实际发布日期通常滞后1-2个月，这里使用发布月末作为近似）
        # Q1 -> 4月末, Q2 -> 7月末, Q3 -> 10月末, Q4 -> 次年1月末
        def quarter_to_date(row):
            year = int(row['year'])
            q = int(row['q'])
            if q == 1:
                return f"{year}-04-30"
            elif q == 2:
                return f"{year}-07-31"
            elif q == 3:
                return f"{year}-10-31"
            else:  # Q4
                return f"{year+1}-01-31"
        
        # cuDF不直接支持apply，使用条件赋值
        df['trade_date'] = ''
        df.loc[df['q'] == 1, 'trade_date'] = df.loc[df['q'] == 1, 'year'].astype(str) + '-04-30'
        df.loc[df['q'] == 2, 'trade_date'] = df.loc[df['q'] == 2, 'year'].astype(str) + '-07-31'
        df.loc[df['q'] == 3, 'trade_date'] = df.loc[df['q'] == 3, 'year'].astype(str) + '-10-31'
        # Q4特殊处理
        mask_q4 = df['q'] == 4
        if mask_q4.any():
            df.loc[mask_q4, 'trade_date'] = (df.loc[mask_q4, 'year'] + 1).astype(str) + '-01-31'
        
        # 选择需要的字段（只保留同比增速，不保留绝对值避免YTD累计值锯齿陷阱）
        cols = ['trade_date', 'gdp_yoy']
        df = df[[c for c in cols if c in df.columns]]
        
        logger.info(f"加载GDP数据完成，共 {len(df)} 行")
        return df
    
    def _load_cpi(self) -> cudf.DataFrame:
        """加载CPI数据"""
        logger.info("加载CPI数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.cn_cpi)
        
        if len(df) == 0:
            logger.warning("无法加载CPI数据")
            return cudf.DataFrame()
        
        # month格式: 202401等
        df['month_str'] = df['month'].astype(str)
        
        # 转换为月末日期（实际发布日期通常在次月9日左右，这里用月末作为近似）
        df['year'] = df['month_str'].str.slice(0, 4)
        df['mon'] = df['month_str'].str.slice(4, 6)
        
        # PIT修复：CPI实际在次月9-12日公布，保守使用次月15日作为可用日期
        # 例如：202401的CPI在2024-02-09公布，应映射到2024-02-15
        df['year_int'] = df['year'].astype('int32')
        df['mon_int'] = df['mon'].astype('int32')
        
        # 计算次月：月份+1，超过12则年份+1
        df['pub_year'] = df['year_int']
        df['pub_mon'] = df['mon_int'] + 1
        
        # 处理跨年情况
        mask_overflow = df['pub_mon'] > 12
        df.loc[mask_overflow, 'pub_year'] = df.loc[mask_overflow, 'year_int'] + 1
        df.loc[mask_overflow, 'pub_mon'] = 1
        
        # 构建公布日期（次月15日，PIT合规）
        df['trade_date'] = (
            df['pub_year'].astype(str) + '-' + 
            df['pub_mon'].astype(str).str.zfill(2) + '-15'
        )
        
        # 清理临时列
        df = df.drop(columns=['year_int', 'mon_int', 'pub_year', 'pub_mon'], errors='ignore')
        
        # 选择需要的字段（使用全国同比）
        df = df.rename(columns={'nt_yoy': 'cpi_yoy', 'nt_mom': 'cpi_mom'})
        cols = ['trade_date', 'cpi_yoy', 'cpi_mom']
        df = df[[c for c in cols if c in df.columns]]
        
        logger.info(f"加载CPI数据完成，共 {len(df)} 行")
        return df
    
    def _load_pmi(self) -> cudf.DataFrame:
        """加载PMI数据"""
        logger.info("加载PMI数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.cn_pmi)
        
        if len(df) == 0:
            logger.warning("无法加载PMI数据")
            return cudf.DataFrame()
        
        # month格式: 202401等
        df['month_str'] = df['month'].astype(str)
        df['year'] = df['month_str'].str.slice(0, 4)
        df['mon'] = df['month_str'].str.slice(4, 6)
        
        # PIT修复：PMI实际在次月1日公布，使用次月3日作为保守可用日期
        df['year_int'] = df['year'].astype('int32')
        df['mon_int'] = df['mon'].astype('int32')
        df['pub_year'] = df['year_int']
        df['pub_mon'] = df['mon_int'] + 1
        mask_overflow = df['pub_mon'] > 12
        df.loc[mask_overflow, 'pub_year'] = df.loc[mask_overflow, 'year_int'] + 1
        df.loc[mask_overflow, 'pub_mon'] = 1
        df['trade_date'] = (
            df['pub_year'].astype(str) + '-' + 
            df['pub_mon'].astype(str).str.zfill(2) + '-03'
        )
        df = df.drop(columns=['year_int', 'mon_int', 'pub_year', 'pub_mon'], errors='ignore')
        
        # 选择需要的字段
        cols = ['trade_date', 'pmi', 'pmi_prod', 'pmi_new_order']
        df = df[[c for c in cols if c in df.columns]]
        
        logger.info(f"加载PMI数据完成，共 {len(df)} 行")
        return df
    
    def _load_ppi(self) -> cudf.DataFrame:
        """加载PPI数据"""
        logger.info("加载PPI数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.cn_ppi)
        
        if len(df) == 0:
            logger.warning("无法加载PPI数据")
            return cudf.DataFrame()
        
        # month格式: 202401等
        df['month_str'] = df['month'].astype(str)
        df['year'] = df['month_str'].str.slice(0, 4)
        df['mon'] = df['month_str'].str.slice(4, 6)
        
        # PIT修复：PPI实际在次月9-12日公布，使用次月15日作为保守可用日期
        df['year_int'] = df['year'].astype('int32')
        df['mon_int'] = df['mon'].astype('int32')
        df['pub_year'] = df['year_int']
        df['pub_mon'] = df['mon_int'] + 1
        mask_overflow = df['pub_mon'] > 12
        df.loc[mask_overflow, 'pub_year'] = df.loc[mask_overflow, 'year_int'] + 1
        df.loc[mask_overflow, 'pub_mon'] = 1
        df['trade_date'] = (
            df['pub_year'].astype(str) + '-' + 
            df['pub_mon'].astype(str).str.zfill(2) + '-15'
        )
        df = df.drop(columns=['year_int', 'mon_int', 'pub_year', 'pub_mon', 'month_str', 'year', 'mon'], errors='ignore')
        
        # 选择需要的字段（工业生产者出厂价格同比）
        cols = ['trade_date', 'ppi_yoy']
        df = df[[c for c in cols if c in df.columns]]
        
        logger.info(f"加载PPI数据完成，共 {len(df)} 行")
        return df
    
    def _load_m2(self) -> cudf.DataFrame:
        """加载M2数据"""
        logger.info("加载M2数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.cn_m2)
        
        if len(df) == 0:
            logger.warning("无法加载M2数据")
            return cudf.DataFrame()
        
        # month格式: 202401等
        df['month_str'] = df['month'].astype(str)
        df['year'] = df['month_str'].str.slice(0, 4)
        df['mon'] = df['month_str'].str.slice(4, 6)
        
        # PIT修复：M2实际在次月10-15日公布，使用次月15日作为保守可用日期
        df['year_int'] = df['year'].astype('int32')
        df['mon_int'] = df['mon'].astype('int32')
        df['pub_year'] = df['year_int']
        df['pub_mon'] = df['mon_int'] + 1
        mask_overflow = df['pub_mon'] > 12
        df.loc[mask_overflow, 'pub_year'] = df.loc[mask_overflow, 'year_int'] + 1
        df.loc[mask_overflow, 'pub_mon'] = 1
        df['trade_date'] = (
            df['pub_year'].astype(str) + '-' + 
            df['pub_mon'].astype(str).str.zfill(2) + '-15'
        )
        df = df.drop(columns=['year_int', 'mon_int', 'pub_year', 'pub_mon'], errors='ignore')
        
        # 选择需要的字段
        cols = ['trade_date', 'm2', 'm2_yoy']
        df = df[[c for c in cols if c in df.columns]]
        
        logger.info(f"加载M2数据完成，共 {len(df)} 行")
        return df
    
    def _load_lpr(self) -> cudf.DataFrame:
        """加载LPR数据"""
        logger.info("加载LPR数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.lpr)
        
        if len(df) == 0:
            logger.warning("无法加载LPR数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'date')
        df = df.rename(columns={'date': 'trade_date'})
        
        # 选择需要的字段
        cols = ['trade_date', 'lpr_1y', 'lpr_5y']
        df = df[[c for c in cols if c in df.columns]]
        
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载LPR数据完成，共 {len(df)} 行")
        return df
    
    def _load_shibor(self) -> cudf.DataFrame:
        """加载SHIBOR数据"""
        logger.info("加载SHIBOR数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.shibor)
        
        if len(df) == 0:
            logger.warning("无法加载SHIBOR数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'date')
        df = df.rename(columns={'date': 'trade_date'})
        
        # 重命名字段
        rename_map = {
            'on': 'shibor_on',
            '1w': 'shibor_1w',
            '1m': 'shibor_1m',
            '3m': 'shibor_3m',
            '6m': 'shibor_6m',
            '1y': 'shibor_1y',
        }
        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        # 选择需要的字段
        cols = ['trade_date', 'shibor_on', 'shibor_1w', 'shibor_1m', 'shibor_3m', 'shibor_6m', 'shibor_1y']
        df = df[[c for c in cols if c in df.columns]]
        
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载SHIBOR数据完成，共 {len(df)} 行")
        return df
    
    def _load_market_congestion(self) -> cudf.DataFrame:
        """加载市场拥挤度数据"""
        logger.info("加载市场拥挤度数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.market_congestion)
        
        if len(df) == 0:
            logger.warning("无法加载市场拥挤度数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'date')
        df = df.rename(columns={'date': 'trade_date', 'congestion': 'market_congestion'})
        
        cols = ['trade_date', 'market_congestion']
        df = df[[c for c in cols if c in df.columns]]
        
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载市场拥挤度数据完成，共 {len(df)} 行")
        return df
    
    def _load_stock_bond_spread(self) -> cudf.DataFrame:
        """加载股债利差数据"""
        logger.info("加载股债利差数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.stock_bond_spread)
        
        if len(df) == 0:
            logger.warning("无法加载股债利差数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'date')
        df = df.rename(columns={'date': 'trade_date', 'spread': 'stock_bond_spread'})
        
        cols = ['trade_date', 'stock_bond_spread']
        df = df[[c for c in cols if c in df.columns]]
        
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载股债利差数据完成，共 {len(df)} 行")
        return df
    
    def _load_margin_summary(self) -> cudf.DataFrame:
        """
        加载全市场两融余额数据
        
        处理逻辑：
            - 按 trade_date 分组，将沪深两市（SSE/SZSE）的融资融券余额求和
            - 生成 market_total_rzye（全市场融资余额）和 market_total_rqye（全市场融券余额）
            - 注：北交所(BSE)两融规模较小，也纳入合计
        """
        logger.info("加载全市场两融余额数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.margin_summary)
        
        if len(df) == 0:
            logger.warning("无法加载全市场两融余额数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'trade_date')
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        # 按 trade_date 分组，求和沪深两市（及北交所）的融资融券余额
        agg_df = df.groupby('trade_date').agg({
            'rzye': 'sum',   # 融资余额
            'rqye': 'sum',   # 融券余额
            'rzrqye': 'sum', # 融资融券余额
        }).reset_index()
        
        # 重命名为全市场字段
        agg_df = agg_df.rename(columns={
            'rzye': 'market_total_rzye',
            'rqye': 'market_total_rqye',
            'rzrqye': 'market_total_rzrqye',
        })
        
        logger.info(f"加载全市场两融余额数据完成，共 {len(agg_df)} 行")
        return agg_df
    
    # ========================================================================
    # 深度风险因子加载方法
    # ========================================================================
    
    def _load_pe_pb_valuation(self) -> cudf.DataFrame:
        """加载A股估值中位数数据（判断市场是否低估）"""
        logger.info("加载A股估值数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.a_pe_pb_ew_median)
        
        if len(df) == 0:
            logger.warning("无法加载A股估值数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'date')
        df = df.rename(columns={'date': 'trade_date'})
        
        # 原始数据已包含 pb_median, pb_ew 列，只需重命名分位数列
        rename_map = {
            'quantileInRecent10YearsMiddlePB': 'pb_quantile_10y',     # 近10年分位数
            'quantileInAllHistoryMiddlePB': 'pb_quantile_all',        # 全历史分位数
        }
        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        cols = ['trade_date', 'pb_median', 'pb_ew', 'pb_quantile_10y', 'pb_quantile_all']
        df = df[[c for c in cols if c in df.columns]]
        
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载A股估值数据完成，共 {len(df)} 行")
        return df
    
    def _load_buffett_indicator(self) -> cudf.DataFrame:
        """加载巴菲特指标（总市值/GDP，长周期择时指标）"""
        logger.info("加载巴菲特指标...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.buffett_indicator)
        
        if len(df) == 0:
            logger.warning("无法加载巴菲特指标")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'date')
        df = df.rename(columns={'date': 'trade_date'})
        
        # 计算巴菲特指标 = 总市值 / GDP
        if 'total_market_cap' in df.columns and 'gdp' in df.columns:
            df['buffett_indicator'] = df['total_market_cap'] / df['gdp']
        
        # 选择关键字段
        rename_map = {
            'quantile_10y': 'buffett_quantile_10y',
            'quantile_all': 'buffett_quantile_all',
        }
        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        cols = ['trade_date', 'buffett_indicator', 'buffett_quantile_10y', 'buffett_quantile_all']
        df = df[[c for c in cols if c in df.columns]]
        
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载巴菲特指标完成，共 {len(df)} 行")
        return df
    
    def _load_break_net_stock(self) -> cudf.DataFrame:
        """加载破净股占比（情绪极度悲观信号）"""
        logger.info("加载破净股占比...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.break_net_stock)
        
        if len(df) == 0:
            logger.warning("无法加载破净股占比")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'date')
        df = df.rename(columns={'date': 'trade_date'})
        
        # 选择关键字段
        cols = ['trade_date', 'break_net_ratio']
        df = df[[c for c in cols if c in df.columns]]
        
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载破净股占比完成，共 {len(df)} 行")
        return df
    
    # ========================================================================
    # 指数与基准加载方法
    # ========================================================================
    
    def _load_index_daily(self) -> cudf.DataFrame:
        """加载核心指数日线数据"""
        logger.info("加载核心指数数据...")
        
        index_dir = DATA_SOURCE_PATHS.index_daily_dir
        all_dfs = []
        
        for fname, (prefix, name) in self.CORE_INDICES.items():
            path = index_dir / f"{fname}.parquet"
            if not path.exists():
                logger.warning(f"指数文件不存在: {path}")
                continue
            
            df = self.read_parquet(path)
            if len(df) == 0:
                continue
            
            df = self.normalize_date_column(df, 'trade_date')
            
            # ====== 金额单位转换（千元 → 元）======
            # Tushare index_daily 的 amount 单位是千元
            df = self.convert_qian_yuan_to_yuan(df, ['amount'])
            
            # 计算振幅
            if 'high' in df.columns and 'low' in df.columns and 'pre_close' in df.columns:
                df['amplitude'] = (df['high'] - df['low']) / df['pre_close'] * 100
            
            # 计算成交额占比代理（使用 amount / 均值作为换手率代理）
            if 'amount' in df.columns:
                df['turnover_proxy'] = df['amount'] / df['amount'].mean()
            
            # 透视为宽表列
            rename_map = {
                'pct_chg': f'{prefix}_pct_chg',
                'amplitude': f'{prefix}_amplitude',
                'turnover_proxy': f'{prefix}_turnover',
                'close': f'{prefix}_close',
                'vol': f'{prefix}_vol',
                'amount': f'{prefix}_amount',
            }
            
            cols_to_keep = ['trade_date']
            for old, new in rename_map.items():
                if old in df.columns:
                    df = df.rename(columns={old: new})
                    cols_to_keep.append(new)
            
            df = df[cols_to_keep]
            all_dfs.append(df)
            logger.debug(f"加载 {name} ({fname}) 完成，{len(df)} 行")
        
        if not all_dfs:
            logger.warning("无法加载任何指数数据")
            return cudf.DataFrame()
        
        # 合并所有指数
        result = all_dfs[0]
        for df in all_dfs[1:]:
            result = result.merge(df, on='trade_date', how='outer')
        
        result = result[(result['trade_date'] >= self.start_date) & (result['trade_date'] <= self.end_date)]
        
        logger.info(f"加载核心指数数据完成，共 {len(result)} 行，{len(result.columns)} 列")
        return result
    
    # ========================================================================
    # 衍生品数据加载方法
    # ========================================================================
    
    def _load_repo_daily(self) -> cudf.DataFrame:
        """加载回购利率数据（GC001、R-001）"""
        logger.info("加载回购利率数据...")
        
        repo_dir = DATA_SOURCE_PATHS.repo_daily_dir
        repo_codes = {
            '204001_SH': 'gc001',  # 上交所 GC001
            '131810_SZ': 'r001',   # 深交所 R-001
        }
        
        all_dfs = []
        for fname, prefix in repo_codes.items():
            path = repo_dir / f"{fname}.parquet"
            if not path.exists():
                logger.warning(f"回购利率文件不存在: {path}")
                continue
            
            df = self.read_parquet(path)
            if len(df) == 0:
                continue
            
            df = self.normalize_date_column(df, 'trade_date')
            
            # ====== 金额单位转换（万元 → 元）======
            # Tushare repo_daily 的 amount 单位是万元
            df = self.convert_wan_yuan_to_yuan(df, ['amount'])
            
            # 透视为宽表列
            rename_map = {
                'close': f'liquidity_{prefix}_close',
                'weight': f'liquidity_{prefix}_weight',
                'high': f'liquidity_{prefix}_high',
                'low': f'liquidity_{prefix}_low',
                'amount': f'liquidity_{prefix}_amount',
            }
            
            cols_to_keep = ['trade_date']
            for old, new in rename_map.items():
                if old in df.columns:
                    df = df.rename(columns={old: new})
                    cols_to_keep.append(new)
            
            df = df[cols_to_keep]
            all_dfs.append(df)
        
        if not all_dfs:
            logger.warning("无法加载回购利率数据")
            return cudf.DataFrame()
        
        # 合并
        result = all_dfs[0]
        for df in all_dfs[1:]:
            result = result.merge(df, on='trade_date', how='outer')
        
        result = result[(result['trade_date'] >= self.start_date) & (result['trade_date'] <= self.end_date)]
        
        logger.info(f"加载回购利率数据完成，共 {len(result)} 行")
        return result
    
    def _load_futures_data(self, index_data: cudf.DataFrame) -> cudf.DataFrame:
        """加载股指期货数据并计算基差
        
        Args:
            index_data: 指数数据（用于计算基差）
        """
        logger.info("加载股指期货数据...")
        
        fut_dir = DATA_SOURCE_PATHS.fut_daily_dir
        
        results = []
        
        for fut_code, config in self.FUTURES_CONFIG.items():
            spot_index = config['spot_index']
            spot_col = f"{self.CORE_INDICES.get(spot_index, ('idx', ''))[0]}_close"
            
            # 加载该品种的所有合约
            if not fut_dir.exists():
                logger.warning(f"期货目录不存在: {fut_dir}")
                continue
            
            # 找到所有该品种的合约文件
            contract_files = [
                f for f in os.listdir(fut_dir) 
                if f.startswith(f'{fut_code}') and f.endswith('.parquet') and '_CFX' in f
            ]
            
            if not contract_files:
                continue
            
            # 加载所有合约并聚合
            daily_data = {}  # {trade_date: {'total_oi': 0, 'main_close': 0, 'main_oi': 0}}
            
            for cf in contract_files:
                path = fut_dir / cf
                try:
                    df = self.read_parquet(path)
                    if len(df) == 0:
                        continue
                    
                    df = self.normalize_date_column(df, 'trade_date')
                    
                    # cuDF -> pandas 进行聚合（小数据量）
                    df_pd = df.to_pandas()
                    
                    for _, row in df_pd.iterrows():
                        td = row['trade_date']
                        if td not in daily_data:
                            daily_data[td] = {'total_oi': 0, 'main_close': 0, 'main_oi': 0}
                        
                        # 累加持仓量
                        if 'oi' in row and not pd.isna(row['oi']):
                            daily_data[td]['total_oi'] += row['oi']
                        
                        # 找主力合约（持仓量最大）
                        if 'oi' in row and not pd.isna(row['oi']):
                            if row['oi'] > daily_data[td]['main_oi']:
                                daily_data[td]['main_oi'] = row['oi']
                                daily_data[td]['main_close'] = row.get('close', 0)
                                
                except Exception as e:
                    logger.debug(f"加载期货文件失败 {cf}: {e}")
                    continue
            
            if not daily_data:
                continue
            
            # 转换为 DataFrame
            fut_df = pd.DataFrame([
                {'trade_date': td, **data} for td, data in daily_data.items()
            ])
            
            if len(fut_df) == 0:
                continue
            
            fut_df = cudf.from_pandas(fut_df)
            fut_df = self.normalize_date_column(fut_df, 'trade_date')
            
            # 重命名
            prefix = fut_code.lower()
            fut_df = fut_df.rename(columns={
                'total_oi': f'{prefix}_total_oi',
                'main_close': f'{prefix}_close',
            })
            
            # 保留需要的列
            fut_df = fut_df[['trade_date', f'{prefix}_total_oi', f'{prefix}_close']]
            
            # 计算基差率（需要合并指数数据）
            if len(index_data) > 0 and spot_col in index_data.columns:
                fut_df = fut_df.merge(
                    index_data[['trade_date', spot_col]], 
                    on='trade_date', 
                    how='left'
                )
                
                # 基差率 = (期货 - 现货) / 现货
                fut_df[f'{prefix}_basis_rate'] = (
                    (fut_df[f'{prefix}_close'] - fut_df[spot_col]) / fut_df[spot_col]
                )
                
                # 删除临时列
                fut_df = fut_df.drop(columns=[spot_col])
            
            results.append(fut_df)
            logger.debug(f"加载 {fut_code} 期货完成，{len(fut_df)} 行")
        
        if not results:
            logger.warning("无法加载期货数据")
            return cudf.DataFrame()
        
        # 合并所有品种
        result = results[0]
        for df in results[1:]:
            result = result.merge(df, on='trade_date', how='outer')
        
        result = result[(result['trade_date'] >= self.start_date) & (result['trade_date'] <= self.end_date)]
        
        logger.info(f"加载股指期货数据完成，共 {len(result)} 行")
        return result
    
    def _clean_lookahead_bias(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        清理"未来函数"数据 - 防止Look-Ahead Bias
        
        问题：bfill/ffill 会把上市后的数据回填到上市前，导致训练时出现"幽灵数据"
        解决：在产品上市日之前，强制将相关字段置为 0
        """
        logger.info("清理未来函数数据（Look-Ahead Bias）...")
        
        cleaned_count = 0
        
        # 1. 中证1000期货 (IM) - 上市日: 2022-07-22
        im_launch = self.PRODUCT_LAUNCH_DATES.get('IM', '2022-07-22')
        im_cols = ['im_total_oi', 'im_close', 'im_basis_rate']
        im_cols_present = [c for c in im_cols if c in df.columns]
        if im_cols_present:
            mask = df['trade_date'] < im_launch
            for col in im_cols_present:
                df.loc[mask, col] = 0
            cleaned_count += int(mask.sum())
            logger.info(f"  IM期货: 清理了 {int(mask.sum())} 行上市前数据 (< {im_launch})")
        
        # 2. 中证500期货 (IC) - 上市日: 2015-04-16
        ic_launch = self.PRODUCT_LAUNCH_DATES.get('IC', '2015-04-16')
        ic_cols = ['ic_total_oi', 'ic_close', 'ic_basis_rate']
        ic_cols_present = [c for c in ic_cols if c in df.columns]
        if ic_cols_present:
            mask = df['trade_date'] < ic_launch
            for col in ic_cols_present:
                df.loc[mask, col] = 0
            logger.info(f"  IC期货: 清理了 {int(mask.sum())} 行上市前数据 (< {ic_launch})")
        
        # 3. 上证50期货 (IH) - 上市日: 2015-04-16
        ih_launch = self.PRODUCT_LAUNCH_DATES.get('IH', '2015-04-16')
        ih_cols = ['ih_total_oi', 'ih_close', 'ih_basis_rate']
        ih_cols_present = [c for c in ih_cols if c in df.columns]
        if ih_cols_present:
            mask = df['trade_date'] < ih_launch
            for col in ih_cols_present:
                df.loc[mask, col] = 0
            logger.info(f"  IH期货: 清理了 {int(mask.sum())} 行上市前数据 (< {ih_launch})")
        
        # 4. 沪深300期货 (IF) - 上市日: 2010-04-16
        if_launch = self.PRODUCT_LAUNCH_DATES.get('IF', '2010-04-16')
        if_cols = ['if_total_oi', 'if_close', 'if_basis_rate']
        if_cols_present = [c for c in if_cols if c in df.columns]
        if if_cols_present:
            mask = df['trade_date'] < if_launch
            for col in if_cols_present:
                df.loc[mask, col] = 0
            logger.info(f"  IF期货: 清理了 {int(mask.sum())} 行上市前数据 (< {if_launch})")
        
        # 5. 科创50 (KC50) - 基日: 2019-12-31
        kc50_launch = self.PRODUCT_LAUNCH_DATES.get('kc50', '2019-12-31')
        kc50_cols = [c for c in df.columns if c.startswith('kc50_')]
        if kc50_cols:
            mask = df['trade_date'] < kc50_launch
            for col in kc50_cols:
                df.loc[mask, col] = 0
            logger.info(f"  科创50: 清理了 {int(mask.sum())} 行基日前数据 (< {kc50_launch})")
        
        # 6. 中证1000指数 (ZZ1000) - 发布日: 2014-10-17
        zz1000_launch = self.PRODUCT_LAUNCH_DATES.get('zz1000', '2014-10-17')
        zz1000_cols = [c for c in df.columns if c.startswith('zz1000_')]
        if zz1000_cols:
            mask = df['trade_date'] < zz1000_launch
            for col in zz1000_cols:
                df.loc[mask, col] = 0
            logger.info(f"  中证1000: 清理了 {int(mask.sum())} 行发布前数据 (< {zz1000_launch})")
        
        # 7. LPR利率 - 改革日: 2019-08-20 (此前使用贷款基准利率，机制不同)
        lpr_launch = self.PRODUCT_LAUNCH_DATES.get('lpr', '2019-08-20')
        lpr_cols = ['lpr_1y', 'lpr_5y', 'lpr_trend']
        lpr_cols_present = [c for c in lpr_cols if c in df.columns]
        if lpr_cols_present:
            mask = df['trade_date'] < lpr_launch
            for col in lpr_cols_present:
                df.loc[mask, col] = 0
            logger.info(f"  LPR利率: 清理了 {int(mask.sum())} 行改革前数据 (< {lpr_launch})")
        
        logger.info("未来函数数据清理完成")
        return df
    
    def _calculate_derived_features(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """计算衍生特征"""
        logger.info("计算宏观环境衍生特征...")
        
        df = df.sort_values('trade_date')
        
        # 1. 宏观环境状态（基于PMI荣枯线）
        if 'pmi' in df.columns:
            # PMI > 50 扩张，< 50 收缩
            df['pmi_regime'] = 0  # 收缩
            df.loc[df['pmi'] > 50, 'pmi_regime'] = 1  # 扩张
            df.loc[df['pmi'] > 52, 'pmi_regime'] = 2  # 强扩张
        
        # 2. 利率趋势（基于LPR变化）
        if 'lpr_1y' in df.columns:
            df['lpr_1y_prev'] = df['lpr_1y'].shift(1)
            df['lpr_trend'] = 0  # 平稳
            df.loc[df['lpr_1y'] > df['lpr_1y_prev'], 'lpr_trend'] = 1   # 上升
            df.loc[df['lpr_1y'] < df['lpr_1y_prev'], 'lpr_trend'] = -1  # 下降
            df = df.drop(columns=['lpr_1y_prev'])
        
        # 3. 货币环境（基于M2增速）
        if 'm2_yoy' in df.columns:
            # M2同比增速 > 10% 宽松，< 8% 紧缩
            df['money_regime'] = 0  # 中性
            df.loc[df['m2_yoy'] > 10, 'money_regime'] = 1   # 宽松
            df.loc[df['m2_yoy'] < 8, 'money_regime'] = -1   # 紧缩
        
        # 4. 市场风险偏好（基于股债利差）
        if 'stock_bond_spread' in df.columns:
            # 计算历史分位数（简化：与60日均值比较）
            df['spread_ma60'] = df['stock_bond_spread'].rolling(window=60, min_periods=1).mean()
            df['risk_appetite'] = 0  # 中性
            df.loc[df['stock_bond_spread'] > df['spread_ma60'], 'risk_appetite'] = 1   # 风险偏好上升
            df.loc[df['stock_bond_spread'] < df['spread_ma60'], 'risk_appetite'] = -1  # 风险偏好下降
            df = df.drop(columns=['spread_ma60'])
        
        # 5. 综合市场状态（regime）
        regime_cols = ['pmi_regime', 'money_regime', 'risk_appetite']
        regime_cols_present = [c for c in regime_cols if c in df.columns]
        if regime_cols_present:
            df['macro_score'] = 0
            for col in regime_cols_present:
                df['macro_score'] = df['macro_score'] + df[col].fillna(0)
            # 综合状态
            df['macro_regime'] = 0  # 中性
            df.loc[df['macro_score'] >= 2, 'macro_regime'] = 1   # 偏多
            df.loc[df['macro_score'] <= -2, 'macro_regime'] = -1  # 偏空
        
        logger.info("宏观环境衍生特征计算完成")
        return df
    
    def process(self) -> cudf.DataFrame:
        """处理并生成宏观环境宽表"""
        logger.info("开始处理宏观环境宽表...")
        
        # 1. 构建交易日骨架
        skeleton = self._build_date_skeleton()
        
        # ================================================================
        # 2. 加载基础宏观数据
        # ================================================================
        gdp = self._load_gdp()
        cpi = self._load_cpi()
        ppi = self._load_ppi()
        pmi = self._load_pmi()
        m2 = self._load_m2()
        lpr = self._load_lpr()
        shibor = self._load_shibor()
        congestion = self._load_market_congestion()
        spread = self._load_stock_bond_spread()
        margin_summary = self._load_margin_summary()  # 全市场两融余额
        
        # ================================================================
        # 3. 加载深度风险因子
        # ================================================================
        valuation = self._load_pe_pb_valuation()
        buffett = self._load_buffett_indicator()
        break_net = self._load_break_net_stock()
        
        # ================================================================
        # 4. 加载指数与基准数据
        # ================================================================
        index_data = self._load_index_daily()
        
        # ================================================================
        # 5. 加载衍生品数据
        # ================================================================
        repo = self._load_repo_daily()
        futures = self._load_futures_data(index_data)
        
        # ================================================================
        # 6. 合并所有数据到骨架表
        # ================================================================
        logger.info("合并所有数据到交易日骨架...")
        
        result = skeleton.copy()
        
        # 低频数据合并后需要ffill
        low_freq_data = [
            (gdp, ['gdp_yoy']),
            (cpi, ['cpi_yoy', 'cpi_mom']),
            (ppi, ['ppi_yoy']),
            (pmi, ['pmi', 'pmi_prod', 'pmi_new_order']),
            (m2, ['m2', 'm2_yoy']),
            (lpr, ['lpr_1y', 'lpr_5y']),
            # 深度风险因子也需要ffill（月频/周频数据）
            (break_net, ['break_net_ratio']),
        ]
        
        for df, cols in low_freq_data:
            if len(df) > 0:
                df_cols = ['trade_date'] + [c for c in cols if c in df.columns]
                result = result.merge(df[df_cols], on='trade_date', how='left')
        
        # 高频数据直接合并
        high_freq_data = [
            (shibor, ['shibor_on', 'shibor_1w', 'shibor_1m', 'shibor_3m', 'shibor_6m', 'shibor_1y']),
            (congestion, ['market_congestion']),
            (spread, ['stock_bond_spread']),
            # 全市场两融余额（日频）
            (margin_summary, ['market_total_rzye', 'market_total_rqye', 'market_total_rzrqye']),
            # 深度风险因子（日频）
            (valuation, ['pb_median', 'pb_ew', 'pb_quantile_10y', 'pb_quantile_all']),
            (buffett, ['buffett_indicator', 'buffett_quantile_10y', 'buffett_quantile_all']),
        ]
        
        for df, cols in high_freq_data:
            if len(df) > 0:
                df_cols = ['trade_date'] + [c for c in cols if c in df.columns]
                result = result.merge(df[df_cols], on='trade_date', how='left')
        
        # 合并指数数据（日频，所有列）
        if len(index_data) > 0:
            result = result.merge(index_data, on='trade_date', how='left')
        
        # 合并回购利率数据（日频）
        if len(repo) > 0:
            result = result.merge(repo, on='trade_date', how='left')
        
        # 合并期货数据（日频）
        if len(futures) > 0:
            result = result.merge(futures, on='trade_date', how='left')
        
        # ================================================================
        # 7. 前向/后向填充缺失值
        # ================================================================
        logger.info("填充缺失值...")
        
        result = result.sort_values('trade_date')
        
        # 低频宏观数据使用 ffill（禁止 bfill，避免未来数据泄露）
        # 例如：GDP Q2 于 7月31日 发布，若使用 bfill 会将其回填到 4-6月，导致 Look-Ahead Bias
        low_freq_cols = [
            'gdp_yoy',
            'cpi_yoy', 'cpi_mom',
            'ppi_yoy',  # PPI工业生产者出厂价格同比
            'pmi', 'pmi_prod', 'pmi_new_order',
            'm2', 'm2_yoy',
            'lpr_1y', 'lpr_5y',
            'break_net_ratio',
        ]
        
        for col in low_freq_cols:
            if col in result.columns:
                result[col] = result[col].ffill()
        
        # SHIBOR 使用 ffill（禁止 bfill，避免未来数据泄露）
        shibor_cols = ['shibor_on', 'shibor_1w', 'shibor_1m', 'shibor_3m', 'shibor_6m', 'shibor_1y']
        for col in shibor_cols:
            if col in result.columns:
                result[col] = result[col].ffill()
        
        # 市场风险指标使用 ffill（禁止 bfill，避免未来数据泄露）
        risk_cols = ['market_congestion', 'stock_bond_spread']
        for col in risk_cols:
            if col in result.columns:
                result[col] = result[col].ffill()
        
        # 深度风险因子使用 ffill（禁止 bfill，避免未来数据泄露）
        deep_risk_cols = [
            'pb_median', 'pb_ew', 'pb_quantile_10y', 'pb_quantile_all',
            'buffett_indicator', 'buffett_quantile_10y', 'buffett_quantile_all',
        ]
        for col in deep_risk_cols:
            if col in result.columns:
                result[col] = result[col].ffill()
        
        # 指数数据使用 ffill（禁止 bfill，避免未来数据泄露）
        for prefix in ['sh300', 'zz500', 'cyb', 'sz50', 'zz1000', 'kc50']:
            for suffix in ['_pct_chg', '_amplitude', '_turnover', '_close', '_vol', '_amount']:
                col = f'{prefix}{suffix}'
                if col in result.columns:
                    result[col] = result[col].ffill()
        
        # 回购利率使用 ffill（禁止 bfill，避免未来数据泄露）
        for prefix in ['gc001', 'r001']:
            for suffix in ['_close', '_weight', '_high', '_low', '_amount']:
                col = f'liquidity_{prefix}{suffix}'
                if col in result.columns:
                    result[col] = result[col].ffill()
        
        # 期货数据使用 ffill（禁止 bfill，避免未来数据泄露）
        for prefix in ['if', 'ic', 'ih', 'im']:
            for suffix in ['_total_oi', '_close', '_basis_rate']:
                col = f'{prefix}{suffix}'
                if col in result.columns:
                    result[col] = result[col].ffill()
        
        # ================================================================
        # 8. 计算衍生特征
        # ================================================================
        result = self._calculate_derived_features(result)
        
        # ================================================================
        # 8.5 清理未来函数数据（Look-Ahead Bias）
        # ================================================================
        result = self._clean_lookahead_bias(result)
        
        # ================================================================
        # 9. 最终兜底：将剩余 NaN 填0
        # ================================================================
        for col in result.columns:
            if col == 'trade_date':
                continue
            if result[col].isna().any():
                result[col] = result[col].fillna(0)
        
        # 10. 排序
        result = result.sort_values('trade_date').reset_index(drop=True)
        
        # ================================================================
        # 10.5 主键去重（修复重复日期问题）
        # ================================================================
        # 某些数据源可能在同一 trade_date 有多条记录（如多个交易所汇总）
        # 使用 keep='last' 保留最后一条（假设后加载的数据更新更及时）
        n_before = len(result)
        result = result.drop_duplicates(subset=['trade_date'], keep='last')
        n_after = len(result)
        if n_before > n_after:
            logger.warning(
                f"主键去重: 移除了 {n_before - n_after} 行重复的 trade_date 记录"
            )
        
        # 11. float64 → float32（节省内存）
        result = self.convert_float64_to_float32(result)
        
        logger.info(f"宏观环境宽表处理完成，共 {len(result)} 行，{len(result.columns)} 列")
        return result
    
    def save(self, df: cudf.DataFrame):
        """保存处理结果"""
        self.save_parquet(df, self.output_path)
        logger.info(f"宏观环境宽表已保存到 {self.output_path}")
