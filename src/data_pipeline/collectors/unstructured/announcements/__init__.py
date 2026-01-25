"""
公告采集子模块

包含：
- CninfoAnnouncementCrawler: 巨潮资讯爬虫
- AnnouncementCollector: 统一采集接口
"""

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


__all__ = [
    # Collectors
    'CninfoAnnouncementCrawler',
    'AnnouncementCollector',
    
    # Main functions
    'get_announcements',
    'get_announcement_by_date',
    'get_announcements_incremental',
    'get_correction_announcements',
    'get_full_market_history',
    
    # Source-specific functions
    'get_cninfo_announcements',
]
