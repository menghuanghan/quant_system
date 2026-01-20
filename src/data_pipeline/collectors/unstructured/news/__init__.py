"""
新闻采集子模块

包含：
- CCTVNewsCollector: 央视新闻联播采集器
- EastMoneyNewsCollector: 东方财富新闻采集器
- SinaFinanceCrawler: 新浪财经爬虫
- STCNCrawler: 证券时报爬虫
- ExchangeNewsCrawler: 交易所公告解读
- NewsCollector: 统一采集接口
- StreamingNewsCollector: 流式新闻采集器（推荐）
"""

from .cctv_collector import (
    CCTVNewsCollector,
    NewsCategory,
    get_cctv_news,
    get_cctv_news_recent,
)

from .eastmoney_collector import (
    EastMoneyNewsCollector,
    get_eastmoney_news,
    get_stock_news,
)

from .sina_crawler import (
    SinaFinanceCrawler,
    get_sina_news,
)

from .stcn_crawler import (
    STCNCrawler,
    get_stcn_news,
)

from .exchange_news_crawler import (
    ExchangeNewsCrawler,
    get_exchange_news,
)

from .news_collector import (
    NewsCollector,
    get_news,
    get_news_by_date,
    get_news_incremental,
    get_stock_related_news,
)

# 流式采集器（推荐，支持即时清洗与防泄露）
from .streaming_news_collector import (
    StreamingNewsCollector,
    collect_news_streaming,
    collect_stock_news_streaming,
)


__all__ = [
    # Category enum
    'NewsCategory',
    
    # Collectors
    'CCTVNewsCollector',
    'EastMoneyNewsCollector',
    'SinaFinanceCrawler',
    'STCNCrawler',
    'ExchangeNewsCrawler',
    'NewsCollector',
    'StreamingNewsCollector',  # 流式采集器
    
    # Main functions
    'get_news',
    'get_news_by_date',
    'get_news_incremental',
    'get_stock_related_news',
    'collect_news_streaming',  # 流式采集接口
    'collect_stock_news_streaming',  # 个股新闻流式采集
    
    # Source-specific functions
    'get_cctv_news',
    'get_cctv_news_recent',
    'get_eastmoney_news',
    'get_stock_news',
    'get_sina_news',
    'get_stcn_news',
    'get_exchange_news',
]
