"""
政策采集基类（简化版）

职责：只负责采集政策数据并返回DataFrame
"""

import re
import hashlib
import logging
from typing import Optional, List
from datetime import datetime
from enum import Enum

import pandas as pd

from ..base import UnstructuredCollector
from ..request_utils import safe_request

logger = logging.getLogger(__name__)


class PolicySource(Enum):
    """政策来源"""
    CSRC = "csrc"      # 证监会
    GOV = "gov"        # 国务院
    NDRC = "ndrc"      # 发改委
    OTHER = "other"


class PolicyCategory(Enum):
    """政策类别"""
    MACRO = "macro"            # 宏观政策
    STOCK = "stock"            # 股市政策
    BOND = "bond"              # 债市政策
    FUND = "fund"              # 基金政策
    FUTURES = "futures"        # 期货政策
    IPO = "ipo"                # IPO相关
    SUPERVISION = "supervision"  # 监管规则
    INDUSTRY = "industry"      # 行业政策
    OTHER = "other"


class BasePolicyCollector(UnstructuredCollector):
    """
    政策采集基类
    
    职责：
    - 采集政策列表和详情
    - 返回包含政策数据的DataFrame
    - 不负责文件下载和存储
    """
    
    SOURCE = PolicySource.OTHER
    BASE_URL = ""
    
    # 发文字号正则模式
    DOC_NO_PATTERNS = [
        # 括号内格式: (发改环资〔2025〕1751号)
        r'\(([^\(\)]+〔\d{4}〕\d+号)\)',
        r'\(([^\(\)]+\[\d{4}\]\d+号)\)',
        # 证监会: 证监发〔2024〕1号
        r'(证监[发办函]\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        # 国务院: 国发〔2024〕1号
        r'(国[发办函]\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        # 发改委: 发改环资〔2025〕1751号
        r'(发改\w{1,4}\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        # 通用格式
        r'(\w{2,6}[发办函]\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        # 公告格式: 2024年第1号公告
        r'(\d{4}\s*年\s*第?\s*\d+\s*号\s*公告)',
        # 令格式
        r'([\u4e00-\u9fa5]+令\s*第?\s*\d+\s*号)',
    ]
    
    def _generate_id(self, doc_no: str, title: str, source: str) -> str:
        """生成唯一ID"""
        key = doc_no if doc_no else f"{title}_{source}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _extract_doc_no(self, text: str) -> str:
        """从文本中提取发文字号"""
        if not text:
            return ""
        
        for pattern in self.DOC_NO_PATTERNS:
            match = re.search(pattern, text)
            if match:
                doc_no = match.group(1)
                doc_no = re.sub(r'\s+', '', doc_no)
                return doc_no
        
        return ""
    
    def _extract_publish_date(self, text: str) -> str:
        """从文本中提取发布日期"""
        if not text:
            return ""
        
        patterns = [
            r'(\d{4})\s*[-年./]\s*(\d{1,2})\s*[-月./]\s*(\d{1,2})\s*日?',
            r'(\d{4})(\d{2})(\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                year, month, day = match.groups()
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        return ""
    
    def _classify_policy(self, title: str, content: str) -> str:
        """政策分类"""
        text = title + content
        
        # 关键词映射
        keywords = {
            PolicyCategory.IPO.value: ['IPO', 'ipo', '首发', '首次公开发行', '上市', '发行股票'],
            PolicyCategory.STOCK.value: ['股票', '证券', 'A股', '二级市场', '股市'],
            PolicyCategory.FUTURES.value: ['期货', '衍生品', '期权'],
            PolicyCategory.FUND.value: ['基金', '资管', '公募', '私募'],
            PolicyCategory.SUPERVISION.value: ['监管', '处罚', '违规', '稽查'],
            PolicyCategory.MACRO.value: ['宏观', '经济', '财政', '货币政策'],
        }
        
        for category, kws in keywords.items():
            if any(kw in text for kw in kws):
                return category
        
        return PolicyCategory.OTHER.value
    
    def _extract_tags(self, title: str, content: str) -> List[str]:
        """提取标签"""
        tags = []
        text = title + content
        
        tag_keywords = {
            '再融资': ['再融资', '配股', '增发', '可转债'],
            '退市': ['退市', '终止上市'],
            '并购重组': ['并购', '重组', '收购', '资产重组'],
            '信息披露': ['信息披露', '公告', '报告'],
            '投资者保护': ['投资者保护', '维权'],
        }
        
        for tag, keywords in tag_keywords.items():
            if any(kw in text for kw in keywords):
                tags.append(tag)
        
        return tags
    
    def to_dataframe(self, documents: List) -> pd.DataFrame:
        """
        将政策文档列表转换为DataFrame
        
        Args:
            documents: PolicyDocument对象列表或字典列表
            
        Returns:
            DataFrame
        """
        if not documents:
            return pd.DataFrame()
        
        # 转换为字典列表
        records = []
        for doc in documents:
            if hasattr(doc, 'to_dict'):
                records.append(doc.to_dict())
            elif isinstance(doc, dict):
                records.append(doc)
        
        return pd.DataFrame(records)
    
    def _get_playwright_browser(self):
        """获取Playwright浏览器实例（如果需要）"""
        if not hasattr(self, '_scraper') or self._scraper is None:
            try:
                from ..scraper_base import get_scraper
                self._scraper = get_scraper()
                self._scraper.init_browser('playwright', headless=True)
            except Exception as e:
                logger.warning(f"Playwright初始化失败: {e}")
                self._scraper = None
        return self._scraper
    
    def _close_browser(self):
        """关闭浏览器"""
        if hasattr(self, '_scraper') and self._scraper:
            try:
                self._scraper.close()
            except:
                pass
            self._scraper = None


# 简化的数据类
class PolicyDocument:
    """政策文档数据结构（简化版）- 不含content"""
    
    def __init__(
        self,
        id: str,
        source_dept: str,
        doc_no: str,
        title: str,
        date: str,
        source: str,
        category: str = "",
        tags: List[str] = None,
        url: str = "",
        content: str = "",  # 保留参数但不存储
        **kwargs
    ):
        self.id = id
        self.source_dept = source_dept
        self.doc_no = doc_no
        self.title = title
        self.date = date
        self.source = source
        self.category = category
        self.tags = tags or []
        self.url = url
        # content字段不再存储
        self.extra = kwargs
    
    def to_dict(self) -> dict:
        """转换为字典 - 不含content"""
        return {
            'id': self.id,
            'source_dept': self.source_dept,
            'doc_no': self.doc_no,
            'title': self.title,
            'date': self.date,
            'source': self.source,
            'category': self.category,
            'tags': ','.join(self.tags) if isinstance(self.tags, list) else self.tags,
            'url': self.url,
            **self.extra
        }


def get_policy_collector(source: str):
    """获取政策采集器实例"""
    from .gov_council import GovCouncilCollector
    from .ndrc import NDRCCollector
    
    collectors = {
        'gov': GovCouncilCollector,
        'ndrc': NDRCCollector,
    }
    
    collector_cls = collectors.get(source.lower())
    if not collector_cls:
        raise ValueError(f"未知的政策来源: {source}，支持的来源: {list(collectors.keys())}")
    
    return collector_cls()
