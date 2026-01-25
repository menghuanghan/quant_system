"""
资产质量异常（Asset Quality Anomaly）采集模块

数据类型包括：
- 个股商誉明细
- 商誉减值预期明细
- 破净股统计
"""

import logging
from typing import Optional
from datetime import datetime

import pandas as pd

from ..base import (
    BaseCollector,
    DataSource,
    DataSourceManager,
    retry_on_failure,
    StandardFields,
    CollectorRegistry
)

logger = logging.getLogger(__name__)


@CollectorRegistry.register("stock_goodwill")
class StockGoodwillCollector(BaseCollector):
    """
    个股商誉明细采集器
    
    获取上市公司商誉明细数据
    主数据源：AkShare (stock_sy_profile_em)
    备用数据源：Tushare (fina_indicator中的goodwill字段)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 股票代码
        'stock_name',           # 股票名称
        'report_date',          # 报告期
        'goodwill',             # 商誉(亿元)
        'net_assets',           # 净资产(亿元)
        'goodwill_ratio',       # 商誉占净资产比例(%)
        'total_assets',         # 总资产(亿元)
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集个股商誉明细数据
        
        Args:
            ts_code: 股票代码 (可选，不填则获取全市场)
            report_date: 报告期 (YYYYMMDD格式，可选)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 商誉明细数据
        """
        # 优先使用AkShare
        try:
            df = self._collect_from_akshare(ts_code, report_date, **kwargs)
            if not df.empty:
                logger.info(f"从 AkShare 成功采集商誉明细数据: {len(df)} 条")
                return df
        except Exception as e:
            logger.warning(f"从 AkShare 采集商誉明细失败: {e}")
        
        # 降级到Tushare
        try:
            df = self._collect_from_tushare(ts_code, report_date, **kwargs)
            if not df.empty:
                logger.info(f"从 Tushare 成功采集商誉明细数据: {len(df)} 条")
                return df
        except Exception as e:
            logger.error(f"从 Tushare 采集商誉明细失败: {e}")
        
        logger.error("所有数据源均采集失败")
        
        # 确保返回符合OUTPUT_FIELDS格式的空DataFrame
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare interface: stock_sy_em (Goodwill)
        # It takes 'date' in YYYYMMDD, return data for specified report date
        # If not provided, we might get empty results or current.
        if report_date:
            date_param = report_date
        else:
            # Try to get latest
            date_param = datetime.now().strftime('%Y%m%d')
            
        try:
            df = ak.stock_sy_em(date=date_param)
        except Exception as e:
            logger.warning(f"AkShare stock_sy_em failed: {e}")
            return pd.DataFrame()
            
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Standardize columns
        column_mapping = {
            '股票代码': 'ts_code',
            '股票简称': 'stock_name',
            '公告日期': 'report_date',
            '商誉': 'goodwill',
            '商誉占净资产比例': 'goodwill_ratio',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # Format ts_code (AkShare is often just digits)
        if 'ts_code' in df.columns:
            def format_code(c):
                c = str(c).zfill(6)
                if c.startswith('6'): return c + '.SH'
                if c.startswith(('0', '3')): return c + '.SZ'
                if c.startswith(('4', '8')): return c + '.BJ'
                return c
            df['ts_code'] = df['ts_code'].apply(format_code)

        if ts_code:
            df = df[df['ts_code'] == ts_code]
            
        return df
    
    @retry_on_failure(max_retries=3)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从Tushare采集数据"""
        import tushare as ts
        
        pro = ts.pro_api()
        
        # Tushare需要通过fina_indicator获取商誉数据
        # 需要2000+积分
        if ts_code:
            df = pro.fina_indicator(
                ts_code=ts_code,
                period=report_date,
                fields='ts_code,end_date,goodwill,tangible_asset,total_assets'
            )
        else:
            # 批量获取需要按报告期查询
            if not report_date:
                # 默认最近一个季度
                report_date = datetime.now().strftime('%Y%m%d')
            
            df = pro.fina_indicator(
                period=report_date,
                fields='ts_code,end_date,goodwill,tangible_asset,total_assets'
            )
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名
        column_mapping = {
            'end_date': 'report_date',
            'tangible_asset': 'net_assets',  # 使用有形资产作为净资产的近似
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 计算商誉占比
        df['goodwill_ratio'] = (df['goodwill'] / df['net_assets'] * 100).round(2)
        
        # 缺失字段
        df['stock_name'] = None
        
        return df


@CollectorRegistry.register("goodwill_impairment")
class GoodwillImpairmentCollector(BaseCollector):
    """
    商誉减值预期明细采集器
    
    获取商誉减值预期数据
    主数据源：AkShare (stock_sy_jz_em)
    备用数据源：无
    """
    
    OUTPUT_FIELDS = [
        'ts_code',                  # 股票代码
        'stock_name',               # 股票名称
        'report_date',              # 报告期
        'goodwill',                 # 商誉(亿元)
        'expected_impairment',      # 预期减值(亿元)
        'impairment_ratio',         # 减值占商誉比例(%)
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集商誉减值预期明细数据
        
        Args:
            ts_code: 股票代码 (可选)
            report_date: 报告期 (YYYYMMDD格式，可选)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 商誉减值预期数据
        """
        manager = DataSourceManager()
        df = pd.DataFrame()
        
        try:
            df = self._collect_from_akshare(ts_code, report_date, **kwargs)
            
            if not df.empty:
                logger.info(f"成功从 AkShare 采集商誉减值数据: {len(df)} 条")
        except Exception as e:
            logger.error(f"从 AkShare 采集商誉减值数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            logger.warning("未能采集到商誉减值数据")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有必需字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare interface: stock_sy_jz_em (Impairment Expectation)
        try:
            df = ak.stock_sy_jz_em()
        except Exception as e:
            logger.warning(f"AkShare stock_sy_jz_em failed: {e}")
            return pd.DataFrame()
        
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        
        # Standardize columns
        column_mapping = {
            '股票代码': 'ts_code',
            '股票简称': 'stock_name',
            '报告期': 'report_date',
            '商誉': 'goodwill',
            '预期减值': 'expected_impairment',
            '减值占比': 'impairment_ratio',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # Format ts_code
        if 'ts_code' in df.columns:
            def format_code(c):
                c = str(c).zfill(6)
                if c.startswith('6'): return c + '.SH'
                if c.startswith(('0', '3')): return c + '.SZ'
                if c.startswith(('4', '8')): return c + '.BJ'
                return c
            df['ts_code'] = df['ts_code'].apply(format_code)

        if ts_code:
            df = df[df['ts_code'] == ts_code]
        
        # Date filtering and formatting
        if report_date:
            df = df[df['report_date'].astype(str).str.contains(report_date)]
            
        if 'report_date' in df.columns:
             df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce').dt.strftime('%Y%m%d')
        
        return df


@CollectorRegistry.register("break_net_stock")
class BreakNetStockCollector(BaseCollector):
    """
    破净股统计采集器
    
    获取破净股统计数据（市净率<1的股票）
    主数据源：AkShare (stock_a_below_net_asset_statistics)
    备用数据源：Tushare (daily_basic筛选pb<1)
    """
    
    OUTPUT_FIELDS = [
        'date',                 # 统计日期
        'total_count',          # 破净股总数
        'break_net_count_sh',   # 上海破净股数量
        'break_net_count_sz',   # 深圳破净股数量
        'break_net_ratio',      # 破净股占比(%)
    ]
    
    def collect(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集破净股统计数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 破净股统计数据
        """
        # Determine if we need history
        need_history = False
        if start_date and end_date and start_date != end_date:
            need_history = True
        elif start_date and start_date != datetime.now().strftime('%Y%m%d'):
            need_history = True
            
        # Strategy: Use Tushare for history, AkShare for current snapshot
        if need_history:
             try:
                df = self._collect_from_tushare(start_date, end_date, **kwargs)
                if not df.empty:
                    logger.info(f"从 Tushare 成功采集破净股统计数据: {len(df)} 条")
                    return df
             except Exception as e:
                logger.error(f"从 Tushare 采集破净股统计失败: {e}")
                
             # Fallback to AkShare (will likely return empty or only current)
             try:
                df = self._collect_from_akshare(start_date, end_date, **kwargs)
                if not df.empty:
                     return df
             except Exception as e:
                logger.warning(f"从 AkShare 采集破净股统计失败: {e}")
        else:
            # Current snapshot - AkShare first
            try:
                df = self._collect_from_akshare(start_date, end_date, **kwargs)
                if not df.empty:
                    logger.info(f"从 AkShare 成功采集破净股统计数据: {len(df)} 条")
                    return df
            except Exception as e:
                logger.warning(f"从 AkShare 采集破净股统计失败: {e}")
            
            # Fallback to Tushare
            try:
                df = self._collect_from_tushare(start_date, end_date, **kwargs)
                if not df.empty:
                    logger.info(f"从 Tushare 成功采集破净股统计数据: {len(df)} 条")
                    return df
            except Exception as e:
                logger.error(f"从 Tushare 采集破净股统计失败: {e}")

        logger.error("所有数据源均采集失败")
        
        # 确保返回符合OUTPUT_FIELDS格式的空DataFrame
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare没有直接的破净股统计接口，我们使用市场全景图数据筛选PB\u003c1的股票
        # Use stock_zh_a_spot_em for current snapshot as a workaround for broken historical LG interfaces
        try:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return pd.DataFrame()
                
            pb_col = '市净率' if '市净率' in df.columns else 'pb'
            if pb_col in df.columns:
                df[pb_col] = pd.to_numeric(df[pb_col], errors='coerce')
                break_net_df = df[df[pb_col] < 1].copy()
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"AkShare获取破净股快照失败: {e}")
            return pd.DataFrame()
        
        if break_net_df.empty and df.empty:
            return pd.DataFrame()
        
        # Snapshot date
        current_date = datetime.now().strftime('%Y%m%d')
        
        summary_df = pd.DataFrame([{
            'date': current_date,
            'total_count': len(break_net_df),
            'break_net_count_sh': len(break_net_df[break_net_df['代码'].str.startswith('6')]) if '代码' in break_net_df.columns else 0,
            'break_net_count_sz': len(break_net_df[break_net_df['代码'].str.startswith(('0', '3'))]) if '代码' in break_net_df.columns else 0,
            'break_net_ratio': round(len(break_net_df) / len(df) * 100, 2) if not df.empty else 0,
        }])
        
        return summary_df
    
    @retry_on_failure(max_retries=3)
    def _collect_from_tushare(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从Tushare采集数据（执行历史统计）"""
        import tushare as ts
        import time
        
        pro = self.tushare_api
        
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            # 如果没指定开始日期，默认只采集结束日期那一天
            start_date = end_date
        
        logger.info(f"正在从 Tushare 统计 {start_date} 到 {end_date} 的破净股数据...")
        
        # 获取交易日历，只统计交易日
        try:
            df_cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open=1)
            trade_days = df_cal['cal_date'].tolist()
        except Exception as e:
            logger.warning(f"获取交易日历失败，将尝试直接统计全量日期: {e}")
            trade_days = pd.date_range(start=start_date, end=end_date).strftime('%Y%m%d').tolist()

        result_list = []
        # 为了不消耗过多积分和触发限流，如果日期范围过大，建议只采样
        if len(trade_days) > 31:
             logger.info(f"日期范围较大 ({len(trade_days)}天)，将按月频率进行采样统计以避免限流")
             # 每月取最后一个交易日
             df_cal['month'] = df_cal['cal_date'].str[:6]
             trade_days = df_cal.groupby('month')['cal_date'].max().tolist()

        for trade_date in trade_days:
            try:
                # 获取当日所有股票的市净率
                df_daily = pro.daily_basic(trade_date=trade_date, fields='ts_code,trade_date,pb')
                
                if df_daily.empty:
                    continue
                
                # 筛选破净股（pb<1）
                break_net = df_daily[df_daily['pb'] < 1]
                
                # 统计
                total = len(break_net)
                sh_count = len(break_net[break_net['ts_code'].str.startswith('6')])
                sz_count = len(break_net[break_net['ts_code'].str.startswith(('0', '3'))])
                ratio = (total / len(df_daily) * 100) if len(df_daily) > 0 else 0
                
                result_list.append({
                    'date': trade_date,
                    'total_count': total,
                    'break_net_count_sh': sh_count,
                    'break_net_count_sz': sz_count,
                    'break_net_ratio': round(ratio, 2)
                })
                
                # 稍微延迟，避免过快
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"统计 {trade_date} 破净股失败: {e}")
        
        if not result_list:
            return pd.DataFrame()
        
        return pd.DataFrame(result_list)


# 便捷函数
def get_stock_goodwill(
    ts_code: Optional[str] = None,
    report_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取个股商誉明细数据
    
    Args:
        ts_code: 股票代码 (可选)
        report_date: 报告期 (YYYYMMDD格式，可选)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 商誉明细数据
    """
    collector = StockGoodwillCollector()
    return collector.collect(
        ts_code=ts_code,
        report_date=report_date,
        **kwargs
    )


def get_goodwill_impairment(
    ts_code: Optional[str] = None,
    report_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取商誉减值预期明细数据
    
    Args:
        ts_code: 股票代码 (可选)
        report_date: 报告期 (YYYYMMDD格式，可选)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 商誉减值预期数据
    """
    collector = GoodwillImpairmentCollector()
    return collector.collect(
        ts_code=ts_code,
        report_date=report_date,
        **kwargs
    )


def get_break_net_stock(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取破净股统计数据
    
    Args:
        start_date: 开始日期 (YYYYMMDD格式)
        end_date: 结束日期 (YYYYMMDD格式)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 破净股统计数据
    """
    collector = BreakNetStockCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
