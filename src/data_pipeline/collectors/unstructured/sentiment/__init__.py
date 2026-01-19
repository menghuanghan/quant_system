"""
舆情与市场情绪采集模块

本模块包含两个核心子模块：

1. 市场热度数据 (Structured)
   - 实时热榜采集 (AkShare)
   - 历史热度代理指标 (换手率 + 新闻条数)
   
2. 投资者舆情文本 (Unstructured)
   - 互动易问答 (Tushare cninfo_interaction)
   - 股吧/雪球评论爬虫
   - 事件驱动回溯采集

使用示例:
    ```python
    from src.data_pipeline.collectors.unstructured.sentiment import (
        MarketHeatCollector,
        InvestorSentimentCollector,
        get_market_heat,
        get_investor_sentiment,
    )
    
    # 采集市场热度
    heat_df = get_market_heat(trade_date='2025-01-15')
    
    # 采集投资者舆情（事件驱动）
    sentiment_df = get_investor_sentiment(
        ts_codes=['000001.SZ'],
        start_date='2024-01-01',
        end_date='2025-01-15',
        event_driven=True  # 只采集波动日
    )
    ```
"""

from .market_heat import (
    MarketHeatCollector,
    HotListSource,
    HeatConfig,
    get_market_heat,
    get_realtime_hotlist,
    get_historical_heat_proxy,
)

from .investor_sentiment import (
    InvestorSentimentCollector,
    SentimentSource,
    SentimentConfig,
    EventFilter,
    get_investor_sentiment,
    get_cninfo_interaction,
    get_guba_comments,
    get_xueqiu_comments,
    get_event_driven_sentiment,
)


__all__ = [
    # 市场热度
    'MarketHeatCollector',
    'HotListSource',
    'HeatConfig',
    'get_market_heat',
    'get_realtime_hotlist',
    'get_historical_heat_proxy',
    
    # 投资者舆情
    'InvestorSentimentCollector',
    'SentimentSource',
    'SentimentConfig',
    'EventFilter',
    'get_investor_sentiment',
    'get_cninfo_interaction',
    'get_guba_comments',
    'get_xueqiu_comments',
    'get_event_driven_sentiment',
]
