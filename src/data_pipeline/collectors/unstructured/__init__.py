"""
非结构化数据采集模块

本模块用于采集非结构化数据，当前支持：
- announcements: 上市公司公告采集
- news: 财经新闻采集

基础设施：
- base: 基类定义、元数据结构、工具函数
- rate_limiter: 请求速率限制
- proxy_pool: 代理池管理
- request_utils: 请求伪装与工具
"""

# 基础设施
from .base import (
    UnstructuredCollector,
    AnnouncementMetadata,
    AnnouncementCategory,
    DataSourceType,
    CollectionProgress,
    DateRangeIterator,
    generate_task_id,
    parse_date_range,
)

from .rate_limiter import (
    RateLimiter,
    AdaptiveRateLimiter,
    RateLimitConfig,
    TokenBucket,
    get_rate_limiter,
    set_rate_limiter,
)

from .proxy_pool import (
    ProxyPool,
    ProxyPoolConfig,
    ProxyInfo,
    RotationStrategy,
    get_proxy_pool,
    set_proxy_pool,
)

from .request_utils import (
    RequestDisguiser,
    RequestSession,
    create_session,
    safe_request,
    safe_download_file,
)

# 公告采集
from .announcements import (
    AnnouncementCollector,
    TushareAnnouncementCollector,
    AKShareAnnouncementCollector,
    CninfoAnnouncementCrawler,
    get_announcements,
    get_announcement_by_date,
    get_announcements_incremental,
    get_correction_announcements,
    get_full_market_history,
    get_tushare_announcements,
    get_akshare_announcements,
    get_cninfo_announcements,
)

# 新闻采集
from .news import (
    NewsCategory,
    NewsCollector,
    CCTVNewsCollector,
    EastMoneyNewsCollector,
    SinaFinanceCrawler,
    STCNCrawler,
    ExchangeNewsCrawler,
    get_news,
    get_news_by_date,
    get_news_incremental,
    get_stock_related_news,
    get_cctv_news,
    get_cctv_news_recent,
    get_eastmoney_news,
    get_stock_news,
    get_sina_news,
    get_stcn_news,
    get_exchange_news,
)

# 研报采集
from .reports import (
    ReportRating,
    RatingChange,
    ReportCollector,
    EastMoneyReportCollector,
    AnalystCollector,
    get_reports,
    get_reports_incremental,
    get_analyst_rank,
    get_stock_reports,
    get_market_reports,
    get_analyst_detail,
    get_eps_forecast,
)


__all__ = [
    # 基类和元数据
    'UnstructuredCollector',
    'AnnouncementMetadata',
    'AnnouncementCategory',
    'DataSourceType',
    'CollectionProgress',
    'DateRangeIterator',
    'generate_task_id',
    'parse_date_range',
    
    # 速率限制
    'RateLimiter',
    'AdaptiveRateLimiter',
    'RateLimitConfig',
    'TokenBucket',
    'get_rate_limiter',
    'set_rate_limiter',
    
    # 代理池
    'ProxyPool',
    'ProxyPoolConfig',
    'ProxyInfo',
    'RotationStrategy',
    'get_proxy_pool',
    'set_proxy_pool',
    
    # 请求工具
    'RequestDisguiser',
    'RequestSession',
    'create_session',
    'safe_request',
    'safe_download_file',
    
    # 公告采集器
    'AnnouncementCollector',
    'TushareAnnouncementCollector',
    'AKShareAnnouncementCollector',
    'CninfoAnnouncementCrawler',
    'get_announcements',
    'get_announcement_by_date',
    'get_announcements_incremental',
    'get_correction_announcements',
    'get_full_market_history',
    'get_tushare_announcements',
    'get_akshare_announcements',
    'get_cninfo_announcements',
    
    # 新闻采集器
    'NewsCategory',
    'NewsCollector',
    'CCTVNewsCollector',
    'EastMoneyNewsCollector',
    'SinaFinanceCrawler',
    'STCNCrawler',
    'ExchangeNewsCrawler',
    'get_news',
    'get_news_by_date',
    'get_news_incremental',
    'get_stock_related_news',
    'get_cctv_news',
    'get_cctv_news_recent',
    'get_eastmoney_news',
    'get_stock_news',
    'get_sina_news',
    'get_stcn_news',
    'get_exchange_news',
    
    # 研报采集器
    'ReportRating',
    'RatingChange',
    'ReportCollector',
    'EastMoneyReportCollector',
    'AnalystCollector',
    'get_reports',
    'get_reports_incremental',
    'get_analyst_rank',
    'get_stock_reports',
    'get_market_reports',
    'get_analyst_detail',
    'get_eps_forecast',
]


__version__ = '1.3.0'
__author__ = 'Quant Team'


