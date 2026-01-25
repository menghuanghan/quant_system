"""
事件驱动数据采集基类（简化版）

职责：只负责采集事件数据并返回DataFrame
"""

import re
import hashlib
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

import pandas as pd

from ..base import UnstructuredCollector

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型枚举"""
    MERGER = 'merger'
    PENALTY = 'penalty'
    CONTROL_CHANGE = 'control_change'
    MAJOR_CONTRACT = 'contract'
    EQUITY_CHANGE = 'equity_change'
    LITIGATION = 'litigation'
    BANKRUPTCY = 'bankruptcy'
    SUSPENSION = 'suspension'
    OTHER = 'other'


class EventSource(Enum):
    """数据源枚举"""
    CNINFO = 'cninfo'
    EASTMONEY = 'eastmoney'
    SSE = 'sse'
    SZSE = 'szse'


class EventDocument:
    """事件文档数据结构（简化版）- 不含content"""
    
    def __init__(
        self,
        id: str,
        ts_code: str,
        stock_name: str,
        event_type: str,
        event_subtype: str,
        title: str,
        date: str,
        source: str,
        url: str = "",
        content: str = "",  # 保留参数但不存储
        effective_date: str = "",
        labels: Dict[str, Any] = None,
        **kwargs
    ):
        self.id = id
        self.ts_code = ts_code
        self.stock_name = stock_name
        self.event_type = event_type
        self.event_subtype = event_subtype
        self.title = title
        self.date = date
        self.source = source
        self.url = url
        # content字段不再存储
        self.effective_date = effective_date
        self.labels = labels or {}
        self.extra = kwargs
    
    def to_dict(self) -> dict:
        """转换为字典 - 不含content"""
        return {
            'id': self.id,
            'ts_code': self.ts_code,
            'stock_name': self.stock_name,
            'event_type': self.event_type,
            'event_subtype': self.event_subtype,
            'title': self.title,
            'date': self.date,
            'source': self.source,
            'url': self.url,
            'effective_date': self.effective_date,
            'labels': str(self.labels),
            **self.extra
        }


class BaseEventCollector(UnstructuredCollector):
    """
    事件采集器基类
    
    职责：
    - 采集事件列表和详情
    - 返回包含事件数据的DataFrame
    - 不负责PDF下载和存储
    """
    
    SOURCE = EventSource.CNINFO
    
    # 事件关键词映射（用于分类）
    EVENT_KEYWORDS = {
        EventType.MERGER.value: ['并购', '重组', '收购', '资产出售', '资产转让', '股权转让'],
        EventType.PENALTY.value: ['处罚', '立案', '调查', '违规', '警告', '罚款'],
        EventType.CONTROL_CHANGE.value: ['实控人', '控制权', '第一大股东'],
        EventType.MAJOR_CONTRACT.value: ['重大合同', '合同', '中标'],
        EventType.EQUITY_CHANGE.value: ['股权', '权益变动', '增持', '减持', '股份变动'],
        EventType.LITIGATION.value: ['诉讼', '仲裁', '起诉'],
        EventType.BANKRUPTCY.value: ['破产', '重整', '清算'],
        EventType.SUSPENSION.value: ['停牌', '复牌', '暂停'],
    }
    
    def _generate_id(self, ts_code: str, title: str, date: str) -> str:
        """生成唯一ID"""
        key = f"{ts_code}_{title}_{date}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _classify_event(self, title: str, content: str = "") -> str:
        """事件分类"""
        text = title + content
        
        for event_type, keywords in self.EVENT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return event_type
        
        return EventType.OTHER.value
    
    def _extract_labels(self, title: str, content: str = "") -> Dict[str, Any]:
        """提取结构化标签"""
        labels = {}
        text = title + content
        
        # 提取金额
        amount_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:万元|亿元|元)', text)
        if amount_match:
            amount_str = amount_match.group(1)
            unit = amount_match.group(0)
            
            amount = float(amount_str)
            if '亿元' in unit:
                amount *= 100000000
            elif '万元' in unit:
                amount *= 10000
            
            labels['amount'] = amount
        
        # 提取比例
        ratio_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if ratio_match:
            labels['ratio'] = float(ratio_match.group(1))
        
        return labels
    
    def _extract_event_subtype(self, title: str, event_type: str) -> str:
        """提取事件子类型"""
        subtype_keywords = {
            EventType.MERGER.value: {
                '预案': '重组预案',
                '草案': '重组草案',
                '报告书': '重组报告书',
                '审核': '审核进展',
                '批复': '监管批复',
            },
            EventType.PENALTY.value: {
                '警示函': '警示函',
                '处罚': '行政处罚',
                '立案': '立案调查',
            },
            EventType.CONTROL_CHANGE.value: {
                '实际控制人': '实控人变更',
                '控股股东': '控股股东变更',
            },
        }
        
        event_subtypes = subtype_keywords.get(event_type, {})
        for keyword, subtype in event_subtypes.items():
            if keyword in title:
                return subtype
        return ''
    
    def _detect_correction(self, title: str) -> bool:
        """检测是否为更正公告"""
        correction_keywords = ['更正', '修订', '修正', '补充', '补正']
        return any(kw in title for kw in correction_keywords)
    
    def to_dataframe(self, events: List) -> pd.DataFrame:
        """
        将事件列表转换为DataFrame
        
        Args:
            events: EventDocument对象列表或字典列表
            
        Returns:
            DataFrame
        """
        if not events:
            return pd.DataFrame()
        
        records = []
        for event in events:
            if hasattr(event, 'to_dict'):
                records.append(event.to_dict())
            elif isinstance(event, dict):
                records.append(event)
        
        return pd.DataFrame(records)


def get_event_collector(source: str):
    """获取事件采集器实例"""
    from .cninfo_event import CninfoEventCollector
    
    collectors = {
        'cninfo': CninfoEventCollector,
    }
    
    collector_cls = collectors.get(source.lower())
    if not collector_cls:
        raise ValueError(f"未知的事件数据源: {source}")
    
    return collector_cls()
