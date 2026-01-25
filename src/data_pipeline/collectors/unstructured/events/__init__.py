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

数据源：
- 巨潮资讯 (Cninfo)
  - 精准分类ID定向采集
  - 历史数据回溯能力强
  - 包含PDF URL链接

使用示例：
    >>> from src.data_pipeline.collectors.unstructured.events import get_cninfo_events
    
    >>> # 采集并购重组事件
    >>> cninfo_df = get_cninfo_events('20240101', '20241231', event_types=['merger'])
    
    >>> # 采集特定股票的所有事件
    >>> df = get_cninfo_events('20240101', '20241231', stock_codes=['000001.SZ'])
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
]
