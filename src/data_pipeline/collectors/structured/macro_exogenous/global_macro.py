"""
国际宏观数据（Global Macro Data）采集模块

数据类型包括：
- 美国国债收益率
- 全球经济日历
- 大宗商品价格
"""

import logging
from typing import Optional, List
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


@CollectorRegistry.register("us_treasury")
class USTreasuryCollector(BaseCollector):
    """
    美国国债收益率采集器
    
    采集美国国债收益率曲线
    主数据源：Tushare (us_tycr) - 需2000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'date',                 # 日期
        'm1',                   # 1月期收益率（%）
        'm2',                   # 2月期
        'm3',                   # 3月期
        'm6',                   # 6月期
        'y1',                   # 1年期
        'y2',                   # 2年期
        'y3',                   # 3年期
        'y5',                   # 5年期
        'y7',                   # 7年期
        'y10',                  # 10年期
        'y20',                  # 20年期
        'y30',                  # 30年期
    ]
    
    def collect(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集美国国债收益率数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的美国国债收益率数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条美国国债收益率数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取美国国债收益率失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条美国国债收益率数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取美国国债收益率失败: {e}")
        
        logger.error("所有数据源均无法获取美国国债收益率数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取美国国债收益率"""
        pro = self.tushare_api
        
        params = {}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = pro.us_tycr(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取美国国债收益率"""
        import akshare as ak
        
        try:
            df = ak.bond_investing_global(country='美国', index_name='美国10年期国债')
        except Exception as e:
            logger.warning(f"AkShare获取美国国债收益率失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '日期': 'date',
            '收盘': 'y10',
        }
        df = self._standardize_columns(df, column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("eco_calendar")
class EcoCalendarCollector(BaseCollector):
    """
    全球经济日历采集器
    
    采集全球主要经济事件/数据发布日历
    主数据源：Tushare (eco_cal) - 需5000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'date',                 # 日期
        'time',                 # 时间
        'country',              # 国家/地区
        'event',                # 事件名称
        # 'importance',         # 重要性（数据源不稳定）
        # 'actual',             # 公布值（数据源不稳定）
        # 'consensus',          # 预测值（数据源不稳定）
        # 'previous',           # 前值（数据源不稳定）
        # 'unit',               # 单位（数据源不提供）
    ]
    
    def collect(
        self,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        country: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集经济日历数据
        
        Args:
            date: 日期
            start_date: 开始日期
            end_date: 结束日期
            country: 国家/地区
        
        Returns:
            DataFrame: 标准化的经济日历数据
        """
        # 优先使用Tushare（需5000积分，可能不够）
        try:
            df = self._collect_from_tushare(date, start_date, end_date, country)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条经济日历数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取经济日历失败（可能积分不足）: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条经济日历数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取经济日历失败: {e}")
        
        logger.error("所有数据源均无法获取经济日历数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        country: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取经济日历"""
        pro = self.tushare_api
        
        params = {}
        if date:
            params['date'] = date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if country:
            params['country'] = country
        
        df = pro.eco_cal(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取经济日历"""
        import akshare as ak
        
        try:
            df = ak.news_economic_baidu(symbol='全球')
        except Exception as e:
            logger.warning(f"AkShare获取经济日历失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '日期': 'date',
            '时间': 'time',
            '国家': 'country',
            '事件': 'event',
            '前值': 'previous',
            '预期': 'consensus',
            '公布值': 'actual',
            '重要性': 'importance',
        }
        df = self._standardize_columns(df, column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_us_treasury(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取美国国债收益率
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 美国国债收益率数据
    
    Example:
        >>> df = get_us_treasury(start_date='20250101', end_date='20251231')
    """
    collector = USTreasuryCollector()
    return collector.collect(start_date=start_date, end_date=end_date)


def get_eco_calendar(
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    country: Optional[str] = None
) -> pd.DataFrame:
    """
    获取全球经济日历
    
    Args:
        date: 日期
        start_date: 开始日期
        end_date: 结束日期
        country: 国家/地区
    
    Returns:
        DataFrame: 经济日历数据
    
    Example:
        >>> df = get_eco_calendar(date='20250115')
        >>> df = get_eco_calendar(country='中国')
    """
    collector = EcoCalendarCollector()
    return collector.collect(date=date, start_date=start_date, end_date=end_date, country=country)
