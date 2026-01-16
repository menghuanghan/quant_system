"""
估值扩散与拥挤度（Valuation Dispersion & Crowding）采集模块

数据类型包括：
- A股等权重与中位数市盈率/市净率
- 大盘拥挤度
- 股债利差
- 巴菲特指标
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


@CollectorRegistry.register("a_pe_pb_ew_median")
class APEPBEWMedianCollector(BaseCollector):
    """
    A股等权重与中位数市盈率/市净率采集器
    
    获取A股市场的等权重和中位数市盈率、市净率指标
    主数据源：AkShare (stock_a_pe_pb_ew_median)
    备用数据源：Tushare (index_dailybasic)
    """
    
    OUTPUT_FIELDS = [
        'date',             # 日期
        'pe_ew',            # 等权重市盈率
        'pe_median',        # 中位数市盈率
        'pb_ew',            # 等权重市净率
        'pb_median',        # 中位数市净率
    ]
    
    def collect(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集A股等权重与中位数市盈率/市净率数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 市盈率/市净率数据
        """
        # 优先使用AkShare
        try:
            df = self._collect_from_akshare(start_date, end_date, **kwargs)
            if not df.empty:
                logger.info(f"从 AkShare 成功采集 A股市盈率/市净率数据: {len(df)} 条")
                return df
        except Exception as e:
            logger.warning(f"从 AkShare 采集 A股市盈率/市净率失败: {e}")
        
        # 降级到Tushare
        try:
            df = self._collect_from_tushare(start_date, end_date, **kwargs)
            if not df.empty:
                logger.info(f"从 Tushare 成功采集 A股市盈率/市净率数据: {len(df)} 条")
                return df
        except Exception as e:
            logger.error(f"从 Tushare 采集 A股市盈率/市净率失败: {e}")
        
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
        
        # AkShare接口：stock_a_all_pb（获取全市场市净率数据）
        df = ak.stock_a_all_pb()
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名
        column_mapping = {
            '日期': 'date',
            '等权重市盈率': 'pe_ew',
            '中位数市盈率': 'pe_median',
            '等权重市净率': 'pb_ew',
            '中位数市净率': 'pb_median',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 日期过滤
        if start_date or end_date:
            df['date'] = pd.to_datetime(df['date'])
            if start_date:
                start = pd.to_datetime(start_date)
                df = df[df['date'] >= start]
            if end_date:
                end = pd.to_datetime(end_date)
                df = df[df['date'] <= end]
            df['date'] = df['date'].dt.strftime('%Y%m%d')
        
        return df
    
    @retry_on_failure(max_retries=3)
    def _collect_from_tushare(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从Tushare采集数据（通过指数daily_basic计算）"""
        import tushare as ts
        
        pro = ts.pro_api()
        
        # 获取沪深300等主要指数的估值数据作为参考
        df = pro.index_dailybasic(
            ts_code='000300.SH',
            start_date=start_date,
            end_date=end_date,
            fields='trade_date,pe,pb'
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名
        column_mapping = {
            'trade_date': 'date',
            'pe': 'pe_median',  # 使用指数PE作为中位数的近似
            'pb': 'pb_median',  # 使用指数PB作为中位数的近似
        }
        df = self._standardize_columns(df, column_mapping)
        
        # Tushare没有直接的等权重数据，设为None
        df['pe_ew'] = None
        df['pb_ew'] = None
        
        return df


@CollectorRegistry.register("market_congestion")
class MarketCongestionCollector(BaseCollector):
    """
    大盘拥挤度采集器
    
    获取A股市场拥挤度指标（衡量市场整体估值是否过热）
    主数据源：AkShare (stock_a_congestion_lg)
    备用数据源：无（Tushare无此数据）
    注意：AkShare仅返回 date, close, congestion 三个字段
    """
    
    OUTPUT_FIELDS = [
        'date',                 # 日期
        'close',                # 收盘价
        'congestion',           # 拥挤度
    ]
    
    def collect(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集大盘拥挤度数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 拥挤度数据
        """
        manager = DataSourceManager()
        df = pd.DataFrame()
        
        # 只有AkShare有此数据
        try:
            df = self._collect_from_akshare(start_date, end_date, **kwargs)
            
            if not df.empty:
                logger.info(f"成功从 AkShare 采集大盘拥挤度数据: {len(df)} 条")
        except Exception as e:
            logger.error(f"从 AkShare 采集大盘拥挤度数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            logger.warning("未能采集到大盘拥挤度数据")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有必需字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare接口：stock_a_congestion_lg
        df = ak.stock_a_congestion_lg()
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名 - AkShare实际返回: ['date', 'close', 'congestion']
        column_mapping = {
            '日期': 'date',
            '收盘价': 'close',
            '大盘拥挤度': 'congestion',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 日期过滤
        if start_date or end_date:
            df['date'] = pd.to_datetime(df['date'])
            if start_date:
                start = pd.to_datetime(start_date)
                df = df[df['date'] >= start]
            if end_date:
                end = pd.to_datetime(end_date)
                df = df[df['date'] <= end]
            df['date'] = df['date'].dt.strftime('%Y%m%d')
        
        return df


@CollectorRegistry.register("stock_bond_spread")
class StockBondSpreadCollector(BaseCollector):
    """
    股债利差采集器
    
    获取股债利差数据（股票收益率与债券收益率之差，衡量股票相对债券的吸引力）
    主数据源：AkShare (stock_ebs_lg)
    备用数据源：无
    """
    
    OUTPUT_FIELDS = [
        'date',                 # 日期
        'index_close',          # 沪深300指数收盘价
        'spread',               # 股债利差
        'spread_ma',            # 股债利差均线
    ]
    
    def collect(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股债利差数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 股债利差数据
        """
        manager = DataSourceManager()
        df = pd.DataFrame()
        
        try:
            df = self._collect_from_akshare(start_date, end_date, **kwargs)
            
            if not df.empty:
                logger.info(f"成功从 AkShare 采集股债利差数据: {len(df)} 条")
        except Exception as e:
            logger.error(f"从 AkShare 采集股债利差数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            logger.warning("未能采集到股债利差数据")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有必需字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare接口：stock_ebs_lg (股债利差-理杏仁)
        df = ak.stock_ebs_lg()
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名 - AkShare实际返回: ['日期', '沪深300指数', '股债利差', '股债利差均线']
        column_mapping = {
            '日期': 'date',
            '沪深300指数': 'index_close',
            '股债利差': 'spread',
            '股债利差均线': 'spread_ma',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 日期过滤
        if start_date or end_date:
            df['date'] = pd.to_datetime(df['date'])
            if start_date:
                start = pd.to_datetime(start_date)
                df = df[df['date'] >= start]
            if end_date:
                end = pd.to_datetime(end_date)
                df = df[df['date'] <= end]
            df['date'] = df['date'].dt.strftime('%Y%m%d')
        
        return df


@CollectorRegistry.register("buffett_indicator")
class BuffettIndicatorCollector(BaseCollector):
    """
    巴菲特指标采集器
    
    获取巴菲特指标数据（总市值/GDP，衡量股市整体估值水平）
    主数据源：AkShare (stock_buffett_index_lg)
    备用数据源：无
    """
    
    OUTPUT_FIELDS = [
        'date',                 # 日期
        'close',                # 收盘价
        'total_market_cap',     # 总市值（亿元）
        'gdp',                  # GDP（亿元）
        'quantile_10y',         # 近十年分位数
        'quantile_all',         # 总历史分位数
    ]
    
    def collect(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集巴菲特指标数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 巴菲特指标数据
        """
        manager = DataSourceManager()
        df = pd.DataFrame()
        
        try:
            df = self._collect_from_akshare(start_date, end_date, **kwargs)
            
            if not df.empty:
                logger.info(f"成功从 AkShare 采集巴菲特指标数据: {len(df)} 条")
        except Exception as e:
            logger.error(f"从 AkShare 采集巴菲特指标数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            logger.warning("未能采集到巴菲特指标数据")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有必需字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare接口：stock_buffett_index_lg (巴菲特指标-理杏仁)
        df = ak.stock_buffett_index_lg()
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名 - AkShare实际返回: ['日期', '收盘价', '总市值', 'GDP', '近十年分位数', '总历史分位数']
        column_mapping = {
            '日期': 'date',
            '收盘价': 'close',
            '总市值': 'total_market_cap',
            'GDP': 'gdp',
            '近十年分位数': 'quantile_10y',
            '总历史分位数': 'quantile_all',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 日期过滤
        if start_date or end_date:
            df['date'] = pd.to_datetime(df['date'])
            if start_date:
                start = pd.to_datetime(start_date)
                df = df[df['date'] >= start]
            if end_date:
                end = pd.to_datetime(end_date)
                df = df[df['date'] <= end]
            df['date'] = df['date'].dt.strftime('%Y%m%d')
        
        return df


# 便捷函数
def get_a_pe_pb_ew_median(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取A股等权重与中位数市盈率/市净率数据
    
    Args:
        start_date: 开始日期 (YYYYMMDD格式)
        end_date: 结束日期 (YYYYMMDD格式)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 市盈率/市净率数据
    """
    collector = APEPBEWMedianCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )


def get_market_congestion(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取大盘拥挤度数据
    
    Args:
        start_date: 开始日期 (YYYYMMDD格式)
        end_date: 结束日期 (YYYYMMDD格式)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 拥挤度数据
    """
    collector = MarketCongestionCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )


def get_stock_bond_spread(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取股债利差数据
    
    Args:
        start_date: 开始日期 (YYYYMMDD格式)
        end_date: 结束日期 (YYYYMMDD格式)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 股债利差数据
    """
    collector = StockBondSpreadCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )


def get_buffett_indicator(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取巴菲特指标数据
    
    Args:
        start_date: 开始日期 (YYYYMMDD格式)
        end_date: 结束日期 (YYYYMMDD格式)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 巴菲特指标数据
    """
    collector = BuffettIndicatorCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
