"""
研报与分析师数据采集子模块

包含：
- EastMoneyReportCollector: 东方财富研报采集器
- AnalystCollector: 分析师数据采集器
- ReportCollector: 统一研报采集接口
"""

from .eastmoney_report_collector import (
    EastMoneyReportCollector,
    ReportRating,
    RatingChange,
    get_stock_reports,
    get_market_reports,
    get_eps_forecast,
)

from .analyst_collector import (
    AnalystCollector,
    get_analyst_rank,
    get_analyst_detail,
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
    
    # Collectors
    'EastMoneyReportCollector',
    'AnalystCollector',
    'ReportCollector',
    
    # Main functions
    'get_reports',
    'get_reports_incremental',
    'get_analyst_rank',
    'get_eps_forecast',
    
    # Source-specific functions
    'get_stock_reports',
    'get_market_reports',
    'get_analyst_detail',
]

