"""
公告采集子模块

包含：
- TushareAnnouncementCollector: Tushare公告采集器
- AKShareAnnouncementCollector: AKShare公告采集器
- CninfoAnnouncementCrawler: 巨潮资讯爬虫
- AnnouncementCollector: 统一采集接口
- get_cninfo_announcements_with_text: 带文本提取的公告采集（推荐）
"""

from .akshare_collector import (
    AKShareAnnouncementCollector,
    get_akshare_announcements,
)

from .cninfo_crawler import (
    CninfoAnnouncementCrawler,
    get_cninfo_announcements,
    get_cninfo_announcements_with_text,  # 带文本提取
)

from .announcement_collector import (
    AnnouncementCollector,
    get_announcements,
    get_announcement_by_date,
    get_announcements_incremental,
    get_correction_announcements,
    get_full_market_history,
)


__all__ = [
    # Collectors
    'AKShareAnnouncementCollector',
    'CninfoAnnouncementCrawler',
    'AnnouncementCollector',
    
    # Main functions
    'get_announcements',
    'get_announcement_by_date',
    'get_announcements_incremental',
    'get_correction_announcements',
    'get_full_market_history',
    
    # Source-specific functions
    'get_akshare_announcements',
    'get_cninfo_announcements',
    'get_cninfo_announcements_with_text',  # 带文本提取
]
