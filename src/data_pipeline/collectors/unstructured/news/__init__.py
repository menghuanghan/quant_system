"""
新闻采集子模块

包含：
- CCTVNewsCollector: 央视新闻联播采集器
- OfficialExchangeNewsCrawler: 交易所官方公告采集器
- NewsCollector: 统一采集接口
"""

from .cctv_collector import (
    CCTVNewsCollector,
    NewsCategory,
    get_cctv_news,
    get_cctv_news_recent,
)



from .official_exchange_news_crawler import (
    OfficialExchangeNewsCrawler,
    get_official_exchange_news,
)

from .news_collector import (
    NewsCollector,
    get_news,
    get_news_by_date,
    get_news_incremental,
    get_stock_related_news,
)


__all__ = [
    # Category enum
    'NewsCategory',
    
    # Collectors
    'CCTVNewsCollector',

    'OfficialExchangeNewsCrawler',
    'NewsCollector',
    
    # Main functions
    'get_news',
    'get_news_by_date',
    'get_news_incremental',
    'get_stock_related_news',
    
    # Source-specific functions
    'get_cctv_news',
    'get_cctv_news_recent',

    'get_official_exchange_news',
]
