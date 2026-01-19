"""
事件驱动型数据采集模块

专门服务于事件驱动策略 (Event-Driven Strategy)

事件类型：
- 并购重组 (merger)
- 违规处罚 (penalty)
- 实控人变更 (control_change)
- 重大合同 (contract)
- 权益变动 (equity_change)
- 重大诉讼 (litigation)

数据源组合：
1. 主力军 - 巨潮资讯 (Cninfo)
   - PDF原文下载
   - 精准分类ID回溯
   - 适合历史数据回溯

2. 辅助军 - 东方财富数据中心
   - 结构化标签（金额、原因等）
   - 适合作为模型训练Label
   - 与巨潮PDF通过 股票代码+日期 对齐

使用示例：
    >>> from src.data_pipeline.collectors.unstructured.events import (
    ...     get_cninfo_events,
    ...     get_eastmoney_events,
    ...     align_events_with_pdf
    ... )
    
    >>> # 采集并购重组事件（含PDF）
    >>> cninfo_df = get_cninfo_events('20240101', '20241231', event_types=['merger'])
    
    >>> # 获取结构化标签
    >>> eastmoney_df = get_eastmoney_events('20240101', '20241231', event_types=['merger'])
    
    >>> # 对齐数据
    >>> aligned_df = align_events_with_pdf(eastmoney_df, cninfo_df)
"""

from .base_event import (
    BaseEventCollector,
    EventDocument,
    EventType,
    EventSource,
    get_event_collector
)

from .cninfo_event import (
    CninfoEventCollector,
    get_cninfo_events,
    EVENT_CATEGORIES
)

from .eastmoney_event import (
    EastMoneyEventCollector,
    get_eastmoney_events,
    align_events_with_pdf
)

__all__ = [
    # 基类
    'BaseEventCollector',
    'EventDocument',
    'EventType',
    'EventSource',
    'get_event_collector',
    
    # 巨潮采集器
    'CninfoEventCollector',
    'get_cninfo_events',
    'EVENT_CATEGORIES',
    
    # 东财采集器
    'EastMoneyEventCollector',
    'get_eastmoney_events',
    'align_events_with_pdf',
]
