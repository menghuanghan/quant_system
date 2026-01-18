"""
统一研报采集接口

聚合多个数据源，提供标准化的研报采集API
"""

import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd

from ..base import UnstructuredCollector
from .eastmoney_report_collector import EastMoneyReportCollector, ReportRating, RatingChange
from .analyst_collector import AnalystCollector

logger = logging.getLogger(__name__)


class ReportCollector:
    """
    统一研报采集接口
    
    数据源：
    1. 东方财富研报 - 评级、目标价、PDF链接
    2. 分析师数据 - 排名、业绩
    
    使用示例:
        >>> collector = ReportCollector()
        >>> df = collector.collect_reports(
        ...     start_date='2025-01-01',
        ...     end_date='2025-01-31',
        ...     stock_codes=['000001', '600000']
        ... )
    """
    
    REPORT_FIELDS = [
        'report_id',
        'title',
        'stock_code',
        'stock_name',
        'broker',
        'analyst',
        'rating',
        'rating_change',
        'target_price',
        'target_price_low',
        'target_price_high',
        'pub_date',
        'pdf_url',
        'source',
    ]
    
    def __init__(self):
        self._report_collector = None
        self._analyst_collector = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def report_collector(self):
        if self._report_collector is None:
            self._report_collector = EastMoneyReportCollector()
        return self._report_collector
    
    @property
    def analyst_collector(self):
        if self._analyst_collector is None:
            self._analyst_collector = AnalystCollector()
        return self._analyst_collector
    
    def collect_reports(
        self,
        start_date: str,
        end_date: str,
        stock_codes: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集研报数据（主接口）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表
        
        Returns:
            研报DataFrame
        """
        return self.report_collector.collect(
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes,
            **kwargs
        )
    
    def collect_market_reports(
        self,
        start_date: str,
        end_date: str,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        采集市场热门研报
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            top_n: 热门股票数量
        
        Returns:
            研报DataFrame
        """
        return self.report_collector.collect_market_reports(
            start_date=start_date,
            end_date=end_date,
            top_n=top_n
        )
    
    def collect_analyst_rank(self) -> pd.DataFrame:
        """
        采集分析师排名
        
        Returns:
            分析师排名DataFrame
        """
        return self.analyst_collector.collect_analyst_rank()
    
    def collect_incremental(
        self,
        days: int = 7,
        stock_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        增量采集（调度器接口）
        
        Args:
            days: 采集最近N天
            stock_codes: 股票代码列表
        
        Returns:
            研报DataFrame
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        return self.collect_reports(
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes
        )
    
    def get_rating_summary(
        self,
        stock_code: str,
        months: int = 6
    ) -> Dict:
        """
        获取个股评级汇总
        
        Args:
            stock_code: 股票代码
            months: 统计最近N个月
        
        Returns:
            评级汇总字典
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=months * 30)).strftime('%Y-%m-%d')
        
        df = self.collect_reports(
            start_date=start_date,
            end_date=end_date,
            stock_codes=[stock_code]
        )
        
        if df.empty:
            return {
                'stock_code': stock_code,
                'report_count': 0,
                'buy_count': 0,
                'hold_count': 0,
                'sell_count': 0,
                'avg_target_price': None,
            }
        
        return {
            'stock_code': stock_code,
            'report_count': len(df),
            'buy_count': len(df[df['rating'] == ReportRating.BUY]),
            'hold_count': len(df[df['rating'].isin([ReportRating.NEUTRAL, ReportRating.HOLD])]),
            'sell_count': len(df[df['rating'].isin([ReportRating.SELL, ReportRating.UNDERWEIGHT])]),
            'avg_target_price': df['target_price'].dropna().mean() if df['target_price'].notna().any() else None,
        }


# ============= 便捷函数接口 =============

def get_reports(
    stock_codes: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    获取研报数据
    
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        研报DataFrame
    """
    collector = ReportCollector()
    return collector.collect_reports(
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes
    )


def get_reports_incremental(
    days: int = 7,
    stock_codes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    增量获取研报（调度器接口）
    
    Args:
        days: 天数
        stock_codes: 股票代码列表
    
    Returns:
        研报DataFrame
    """
    collector = ReportCollector()
    return collector.collect_incremental(
        days=days,
        stock_codes=stock_codes
    )


def get_analyst_rank() -> pd.DataFrame:
    """
    获取分析师排名
    
    Returns:
        分析师排名DataFrame
    """
    collector = ReportCollector()
    return collector.collect_analyst_rank()
