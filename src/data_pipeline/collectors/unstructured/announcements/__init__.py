"""
公告采集子模块

包含：
- TushareAnnouncementCollector: Tushare公告采集器
- AKShareAnnouncementCollector: AKShare公告采集器
- CninfoAnnouncementCrawler: 巨潮资讯爬虫
- AnnouncementCollector: 统一采集接口
- StreamingAnnouncementCollector: 流式公告采集器（推荐）
"""

from .tushare_collector import (
    TushareAnnouncementCollector,
    get_tushare_announcements,
)

from .akshare_collector import (
    AKShareAnnouncementCollector,
    get_akshare_announcements,
)

from .cninfo_crawler import (
    CninfoAnnouncementCrawler,
    get_cninfo_announcements,
)

from .announcement_collector import (
    AnnouncementCollector,
    get_announcements,
    get_announcement_by_date,
    get_announcements_incremental,
    get_correction_announcements,
    get_full_market_history,
)

# 流式采集器（推荐，支持即时清洗与防泄露）
from .streaming_announcement_collector import (
    StreamingAnnouncementCollector,
    collect_announcements_streaming,
    verify_time_cleaning,
)


__all__ = [
    # Collectors
    'TushareAnnouncementCollector',
    'AKShareAnnouncementCollector',
    'CninfoAnnouncementCrawler',
    'AnnouncementCollector',
    'StreamingAnnouncementCollector',  # 流式采集器
    
    # Main functions
    'get_announcements',
    'get_announcement_by_date',
    'get_announcements_incremental',
    'get_correction_announcements',
    'get_full_market_history',
    'collect_announcements_streaming',  # 流式采集接口
    'verify_time_cleaning',  # 防泄露验证
    
    # Source-specific functions
    'get_tushare_announcements',
    'get_akshare_announcements',
    'get_cninfo_announcements',
]
