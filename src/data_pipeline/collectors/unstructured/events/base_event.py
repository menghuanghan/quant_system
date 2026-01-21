"""
事件驱动数据采集基类

定义事件采集器的通用接口、数据结构和工具方法
"""

import re
import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型枚举"""
    MERGER = 'merger'                    # 并购重组
    PENALTY = 'penalty'                  # 违规处罚
    CONTROL_CHANGE = 'control_change'    # 实控人变更
    MAJOR_CONTRACT = 'contract'          # 重大合同
    EQUITY_CHANGE = 'equity_change'      # 权益变动
    LITIGATION = 'litigation'            # 重大诉讼
    BANKRUPTCY = 'bankruptcy'            # 破产重整
    SUSPENSION = 'suspension'            # 停复牌
    OTHER = 'other'


class EventSource(Enum):
    """数据源枚举"""
    CNINFO = 'cninfo'           # 巨潮资讯（主力军）
    EASTMONEY = 'eastmoney'     # 东方财富（辅助/标签）
    SSE = 'sse'                 # 上交所
    SZSE = 'szse'               # 深交所


@dataclass
class EventDocument:
    """
    事件文档数据结构
    
    包含事件元数据和PDF路径信息
    """
    id: str                             # 唯一ID (md5)
    ts_code: str                        # 股票代码 (000001.SZ)
    stock_name: str                     # 股票名称
    event_type: str                     # 事件类型 (EventType.value)
    event_subtype: str                  # 事件子类型
    
    title: str                          # 标题
    summary: str                        # 摘要/关键信息
    
    ann_date: str                       # 公告日期 (YYYY-MM-DD)
    effective_date: str                 # 生效日期（如有）
    
    source: str                         # 数据来源 (EventSource.value)
    url: str                            # 原始链接
    pdf_url: str                        # PDF下载链接
    local_path: str                     # 本地PDF路径
    
    # 结构化标签（从东财补全）
    labels: Dict[str, Any] = field(default_factory=dict)
    # 如：{'amount': 10000000, 'penalty_reason': '信披违规', 'acquirer': 'XXX公司'}
    
    # 元数据
    is_correction: bool = False         # 是否为更正公告
    original_id: str = ''               # 原始公告ID（如为更正）
    create_time: str = ''               # 采集时间


class BaseEventCollector:
    """
    事件采集器基类
    
    提供通用功能：
    1. PDF下载与路径管理
    2. 事件分类与标签
    3. 元数据存储
    4. 数据去重
    """
    
    SOURCE = EventSource.CNINFO
    
    # 数据存储路径
    DATA_DIR = Path("data/raw/unstructured/events")
    
    # 事件类型对应的存储目录
    EVENT_DIRS = {
        EventType.MERGER.value: 'merger_acquisition',
        EventType.PENALTY.value: 'penalty',
        EventType.CONTROL_CHANGE.value: 'control_change',
        EventType.MAJOR_CONTRACT.value: 'contract',
        EventType.EQUITY_CHANGE.value: 'equity_change',
        EventType.LITIGATION.value: 'litigation',
        EventType.BANKRUPTCY.value: 'bankruptcy',
        EventType.SUSPENSION.value: 'suspension',
        EventType.OTHER.value: 'other',
    }
    
    def __init__(self):
        self._ensure_dirs()
        self._existing_ids = set()
    
    def _ensure_dirs(self):
        """确保存储目录存在"""
        for event_dir in self.EVENT_DIRS.values():
            (self.DATA_DIR / event_dir).mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / 'meta').mkdir(parents=True, exist_ok=True)
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        event_types: Optional[List[str]] = None,
        stock_codes: Optional[List[str]] = None,
        download_pdf: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集事件数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD 或 YYYY-MM-DD)
            end_date: 结束日期
            event_types: 事件类型列表 (EventType.value)
            stock_codes: 股票代码列表（可选）
            download_pdf: 是否下载PDF
            
        Returns:
            事件数据DataFrame
        """
        raise NotImplementedError("子类需要实现collect方法")
    
    def _generate_id(self, ts_code: str, title: str, ann_date: str) -> str:
        """生成唯一ID"""
        key = f"{ts_code}_{title}_{ann_date}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_pdf_path(
        self,
        event_type: str,
        ts_code: str,
        title: str,
        ann_date: str
    ) -> Path:
        """
        生成PDF存储路径
        
        路径格式: data/raw/events/{event_type}/{year}/{ts_code}_{safe_title}.pdf
        """
        # 获取事件目录
        event_dir = self.EVENT_DIRS.get(event_type, 'other')
        
        # 提取年份
        year = ann_date[:4] if ann_date else datetime.now().strftime('%Y')
        
        # 安全文件名
        safe_title = re.sub(r'[\\/:*?"<>|]', '', title)[:50]
        filename = f"{ts_code.replace('.', '_')}_{safe_title}.pdf"
        
        return self.DATA_DIR / event_dir / year / filename
    
    def _download_pdf(self, url: str, save_path: Path, timeout: int = 60) -> bool:
        """下载PDF文件"""
        if not url:
            return False
        
        try:
            import requests
            from ..request_utils import RequestDisguiser
            
            disguiser = RequestDisguiser()
            headers = disguiser.get_headers()
            headers['Referer'] = 'http://www.cninfo.com.cn/'
            
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            
            if response.status_code == 200:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.debug(f"下载成功: {save_path}")
                return True
            else:
                logger.warning(f"下载失败: {url}, 状态码: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"下载异常: {url}, 错误: {e}")
        
        return False
    
    def _load_existing_ids(self, event_type: Optional[str] = None) -> set:
        """加载已有事件ID用于去重"""
        existing_ids = set()
        
        # 从元数据文件加载
        meta_dir = self.DATA_DIR / 'meta'
        if meta_dir.exists():
            for meta_file in meta_dir.glob('*.jsonl'):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            import json
                            record = json.loads(line.strip())
                            if event_type and record.get('event_type') != event_type:
                                continue
                            existing_ids.add(record.get('id', ''))
                except Exception as e:
                    logger.debug(f"读取元数据失败: {meta_file}, {e}")
        
        return existing_ids
    
    def _save_metadata(self, events: List[EventDocument], filename: Optional[str] = None):
        """保存事件元数据"""
        if not events:
            return
        
        if not filename:
            filename = f"{self.SOURCE.value}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        meta_path = self.DATA_DIR / 'meta' / filename
        
        import json
        with open(meta_path, 'a', encoding='utf-8') as f:
            for event in events:
                record = {
                    'id': event.id,
                    'ts_code': event.ts_code,
                    'stock_name': event.stock_name,
                    'event_type': event.event_type,
                    'event_subtype': event.event_subtype,
                    'title': event.title,
                    'summary': event.summary,
                    'ann_date': event.ann_date,
                    'effective_date': event.effective_date,
                    'source': event.source,
                    'url': event.url,
                    'pdf_url': event.pdf_url,
                    'local_path': event.local_path,
                    'labels': event.labels,
                    'is_correction': event.is_correction,
                    'original_id': event.original_id,
                    'create_time': event.create_time or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"保存元数据: {len(events)} 条 -> {meta_path}")
    
    def to_dataframe(self, events: List[EventDocument]) -> pd.DataFrame:
        """转换为DataFrame"""
        if not events:
            return pd.DataFrame()
        
        records = []
        for e in events:
            records.append({
                'id': e.id,
                'ts_code': e.ts_code,
                'stock_name': e.stock_name,
                'event_type': e.event_type,
                'event_subtype': e.event_subtype,
                'title': e.title,
                'summary': e.summary,
                'ann_date': e.ann_date,
                'effective_date': e.effective_date,
                'source': e.source,
                'url': e.url,
                'pdf_url': e.pdf_url,
                'local_path': e.local_path,
                'labels': str(e.labels) if e.labels else '',
                'is_correction': e.is_correction,
                'create_time': e.create_time,
            })
        
        return pd.DataFrame(records)
    
    def _classify_event(self, title: str, content: str = "") -> str:
        """根据标题分类事件类型"""
        text = f"{title} {content}".lower()
        
        # 并购重组
        if any(kw in text for kw in ['并购', '重组', '收购', '资产重组', '合并', '分立', '要约收购']):
            return EventType.MERGER.value
        
        # 违规处罚
        if any(kw in text for kw in ['处罚', '警示', '监管函', '谴责', '罚款', '立案调查', '违规']):
            return EventType.PENALTY.value
        
        # 实控人变更
        if any(kw in text for kw in ['实际控制人', '控股股东', '控制权', '易主']):
            return EventType.CONTROL_CHANGE.value
        
        # 重大合同
        if any(kw in text for kw in ['重大合同', '中标', '战略合作', '订单', '框架协议']):
            return EventType.MAJOR_CONTRACT.value
        
        # 权益变动
        if any(kw in text for kw in ['权益变动', '举牌', '增持', '减持', '要约']):
            return EventType.EQUITY_CHANGE.value
        
        # 重大诉讼
        if any(kw in text for kw in ['诉讼', '仲裁', '起诉', '索赔']):
            return EventType.LITIGATION.value
        
        # 破产重整
        if any(kw in text for kw in ['破产', '重整', '清算']):
            return EventType.BANKRUPTCY.value
        
        return EventType.OTHER.value
    
    def _extract_event_subtype(self, title: str, event_type: str) -> str:
        """提取事件子类型"""
        if event_type == EventType.MERGER.value:
            if '预案' in title:
                return 'draft'
            elif '草案' in title:
                return 'proposal'
            elif '报告书' in title:
                return 'report'
            elif '审核' in title or '批复' in title:
                return 'approval'
            elif '回复' in title:
                return 'response'
            else:
                return 'announcement'
        
        elif event_type == EventType.PENALTY.value:
            if '处罚决定' in title:
                return 'decision'
            elif '监管函' in title:
                return 'regulatory_letter'
            elif '警示函' in title:
                return 'warning_letter'
            elif '问询函' in title:
                return 'inquiry_letter'
            else:
                return 'notice'
        
        elif event_type == EventType.CONTROL_CHANGE.value:
            if '简式' in title:
                return 'brief_report'
            elif '详式' in title:
                return 'detailed_report'
            else:
                return 'report'
        
        return 'general'
    
    def _detect_correction(self, title: str) -> bool:
        """检测是否为更正公告"""
        correction_keywords = ['更正', '补充', '修订', '修正', '补充说明', '更正说明']
        return any(kw in title for kw in correction_keywords)


def get_event_collector(source: str = 'cninfo') -> BaseEventCollector:
    """
    获取事件采集器实例
    
    Args:
        source: 数据源 ('cninfo', 'eastmoney')
        
    Returns:
        事件采集器实例
    """
    from .cninfo_event import CninfoEventCollector
    from .eastmoney_event import EastMoneyEventCollector
    
    collectors = {
        'cninfo': CninfoEventCollector,
        'eastmoney': EastMoneyEventCollector,
    }
    
    collector_class = collectors.get(source.lower())
    if not collector_class:
        raise ValueError(f"未知数据源: {source}, 可选: {list(collectors.keys())}")
    
    return collector_class()
