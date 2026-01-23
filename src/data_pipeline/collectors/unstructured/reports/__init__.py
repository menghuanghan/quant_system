"""
研报与分析师数据采集子模块

包含：
- EastMoneyReportCollector: 东方财富研报采集器
- AnalystCollector: 分析师数据采集器
- ReportCollector: 统一研报采集接口
- ThsRatingCollector: 同花顺投资评级采集器（补充字段）
- RatingChangeTracker: 评级变化追踪器
"""

from .eastmoney_report_collector import (
    EastMoneyReportCollector,
    ReportRating,
    RatingChange,
    get_stock_reports,
    get_market_reports,
    get_eps_forecast,
)

from .report_collector import (
    ReportCollector,
    get_reports,
    get_reports_incremental,
)


__all__ = [
    # Constants
    'ReportRating',
    'RatingChange',
    'RATING_HIERARCHY',
    
    # Collectors
    'EastMoneyReportCollector',
    'ReportCollector',
    
    # Main functions
    'get_reports',
    'get_reports_incremental',
    'get_eps_forecast',
    
    # Source-specific functions
    'get_stock_reports',
    'get_market_reports',
]

