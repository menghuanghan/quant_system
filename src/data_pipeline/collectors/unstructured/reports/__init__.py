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

from .ths_rating_collector import (
    ThsRatingCollector,
    get_stock_ths_ratings,
    enrich_reports_with_ths_data,
)

from .rating_change_tracker import (
    RatingChangeTracker,
    detect_rating_changes,
    create_rating_tracker,
    RATING_HIERARCHY,
)


__all__ = [
    # Constants
    'ReportRating',
    'RatingChange',
    'RATING_HIERARCHY',
    
    # Collectors
    'EastMoneyReportCollector',
    'AnalystCollector',
    'ReportCollector',
    'ThsRatingCollector',
    'RatingChangeTracker',
    
    # Main functions
    'get_reports',
    'get_reports_incremental',
    'get_analyst_rank',
    'get_eps_forecast',
    'detect_rating_changes',
    'create_rating_tracker',
    
    # Source-specific functions
    'get_stock_reports',
    'get_market_reports',
    'get_stock_ths_ratings',
    'enrich_reports_with_ths_data',
    'get_analyst_detail',
]

