"""
政策采集基类

处理附件下载、发文字号提取、元数据管理等通用功能

数据存储规范：
- 元数据索引：JSONL格式存储政策元信息
- 原始文件：保留PDF/Word原文件用于后续解析
"""

import os
import re
import json
import hashlib
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import time

import pandas as pd
import requests

from ..base import UnstructuredCollector
from ..request_utils import safe_request, RequestDisguiser

logger = logging.getLogger(__name__)


class PolicySource(Enum):
    """政策来源"""
    CSRC = "csrc"              # 证监会
    GOV = "gov"                # 国务院
    NDRC = "ndrc"              # 发改委
    OTHER = "other"            # 其他


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
    OTHER = "other"            # 其他


@dataclass
class PolicyDocument:
    """政策文档数据结构"""
    id: str                              # 唯一ID (MD5 hash)
    source_dept: str                     # 发文部门
    doc_no: str                          # 发文字号（核心去重键）
    title: str                           # 标题
    publish_date: str                    # 发布日期
    source: str                          # 数据来源
    category: str = ""                   # 政策类别
    tags: List[str] = field(default_factory=list)  # 标签
    file_type: str = ""                  # 文件类型 (html/pdf/doc)
    url: str = ""                        # 原始URL
    local_path: str = ""                 # 本地存储路径
    content_text: str = ""               # 文本内容（如果是网页或已解析）
    summary: str = ""                    # 摘要
    effective_date: str = ""             # 生效日期
    status: str = "active"               # 状态 (active/abolished)
    create_time: str = ""                # 采集时间
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PolicyDocument':
        """从字典创建"""
        return cls(**data)


class BasePolicyCollector(UnstructuredCollector):
    """
    政策采集基类
    
    功能：
    1. 政策列表采集
    2. 详情页采集
    3. 附件下载（PDF/Word）
    4. 发文字号提取
    5. 元数据管理
    """
    
    # 子类需要覆盖
    SOURCE = PolicySource.OTHER
    BASE_URL = ""
    
    # 文件存储路径
    DATA_DIR = Path("data/raw/unstructured/policy")
    META_DIR = DATA_DIR / "meta"
    FILE_DIR = DATA_DIR / "files"
    
    # 发文字号正则模式
    DOC_NO_PATTERNS = [
        # 通用格式（括号内）: (发改环资〔2025〕1751号)
        r'\(([^\(\)]+〔\d{4}〕\d+号)\)',
        r'\(([^\(\)]+\[\d{4}\]\d+号)\)',
        # 证监会: 证监发〔2024〕1号
        r'(证监[发办函]\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        # 国务院: 国发〔2024〕1号
        r'(国[发办函]\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        # 财政部: 财金〔2025〕1号, 财预〔2025〕1号
        r'(财\w{1,2}\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        # 发改委: 发改环资〔2025〕1751号, 发改办投资〔2025〕991号
        r'(发改\w{1,4}\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        # 工信部: 工信部联〔2025〕1号
        r'(工信\w{1,4}\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        # 通用格式: XX发〔2024〕1号（兜底）
        r'(\w{2,6}[发办函]\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        # 公告格式: 2024年第1号公告
        r'(\d{4}\s*年\s*第?\s*\d+\s*号\s*公告)',
        # 令格式: 中国证监会令第XX号
        r'([\u4e00-\u9fa5]+令\s*第?\s*\d+\s*号)',
    ]
    
    def __init__(self):
        super().__init__()
        self._disguiser = RequestDisguiser()
        self._ensure_dirs()
        self._browser = None
    
    def _ensure_dirs(self):
        """确保目录存在"""
        self.META_DIR.mkdir(parents=True, exist_ok=True)
        self.FILE_DIR.mkdir(parents=True, exist_ok=True)
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        categories: Optional[List[str]] = None,
        max_pages: int = 10,
        download_files: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集政策数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            categories: 政策类别过滤
            max_pages: 最大采集页数
            download_files: 是否下载附件
            
        Returns:
            政策元数据DataFrame
        """
        raise NotImplementedError("子类需要实现collect方法")
    
    def _generate_id(self, doc_no: str, title: str, source: str) -> str:
        """生成唯一ID"""
        # 优先使用发文字号，其次使用标题+来源
        key = doc_no if doc_no else f"{title}_{source}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _extract_doc_no(self, text: str) -> str:
        """
        从文本中提取发文字号
        
        发文字号是政策的唯一标识，用于去重
        """
        if not text:
            return ""
        
        for pattern in self.DOC_NO_PATTERNS:
            match = re.search(pattern, text)
            if match:
                doc_no = match.group(1)
                # 标准化格式
                doc_no = re.sub(r'\s+', '', doc_no)
                return doc_no
        
        return ""
    
    def _extract_publish_date(self, text: str) -> str:
        """从文本中提取发布日期"""
        if not text:
            return ""
        
        # 常见日期格式
        patterns = [
            r'(\d{4})\s*[-年./]\s*(\d{1,2})\s*[-月./]\s*(\d{1,2})\s*日?',
            r'(\d{4})(\d{2})(\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                year, month, day = match.groups()
                try:
                    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
                except:
                    continue
        
        return ""
    
    def _classify_policy(self, title: str, content: str = "") -> str:
        """根据标题和内容分类政策"""
        text = f"{title} {content}".lower()
        
        # 关键词分类
        if any(kw in text for kw in ['ipo', '首发', '上市', '发行']):
            return PolicyCategory.IPO.value
        elif any(kw in text for kw in ['股票', '股市', 'a股', '证券']):
            return PolicyCategory.STOCK.value
        elif any(kw in text for kw in ['债券', '债市', '信用债']):
            return PolicyCategory.BOND.value
        elif any(kw in text for kw in ['基金', '私募', '公募']):
            return PolicyCategory.FUND.value
        elif any(kw in text for kw in ['期货', '期权', '衍生品']):
            return PolicyCategory.FUTURES.value
        elif any(kw in text for kw in ['监管', '处罚', '违规', '稽查']):
            return PolicyCategory.SUPERVISION.value
        elif any(kw in text for kw in ['宏观', '货币', '财政', '利率']):
            return PolicyCategory.MACRO.value
        elif any(kw in text for kw in ['行业', '产业', '制造']):
            return PolicyCategory.INDUSTRY.value
        
        return PolicyCategory.OTHER.value
    
    def _extract_tags(self, title: str, content: str = "") -> List[str]:
        """提取政策标签"""
        tags = []
        text = f"{title} {content}"
        
        # 关键词标签
        tag_keywords = {
            'IPO': ['ipo', '首发', '上市发行'],
            '退市': ['退市', '摘牌'],
            '并购': ['并购', '重组', '收购'],
            '减持': ['减持', '增持'],
            '分红': ['分红', '派息', '股息'],
            '注册制': ['注册制'],
            '科创板': ['科创板'],
            '创业板': ['创业板'],
            '北交所': ['北交所', '北京证券交易所'],
            '新三板': ['新三板'],
            '融资融券': ['融资融券', '两融'],
            '股票回购': ['回购'],
            '信息披露': ['信息披露', '披露'],
            '投资者保护': ['投资者保护'],
        }
        
        for tag, keywords in tag_keywords.items():
            if any(kw.lower() in text.lower() for kw in keywords):
                tags.append(tag)
        
        return tags
    
    def _download_file(
        self,
        url: str,
        save_path: Union[str, Path],
        timeout: int = 30
    ) -> bool:
        """
        下载文件（PDF/Word等）
        
        Args:
            url: 文件URL
            save_path: 保存路径
            timeout: 超时时间
            
        Returns:
            是否下载成功
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果文件已存在，跳过
            if save_path.exists():
                logger.debug(f"文件已存在，跳过: {save_path}")
                return True
            
            response = requests.get(
                url,
                headers=self._disguiser.get_headers(),
                timeout=timeout,
                stream=True
            )
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"文件下载成功: {save_path}")
                return True
            else:
                logger.warning(f"文件下载失败: {url}, 状态码: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"文件下载异常: {url}, 错误: {e}")
            return False
    
    def _get_file_path(
        self,
        doc_no: str,
        title: str,
        publish_date: str,
        file_type: str
    ) -> Path:
        """生成文件存储路径"""
        # 按年份和来源组织
        year = publish_date[:4] if publish_date else datetime.now().strftime('%Y')
        source_name = self.SOURCE.value
        
        # 文件名：发文字号或标题
        if doc_no:
            filename = re.sub(r'[〔〕\[\]（）\s]', '_', doc_no)
        else:
            # 清理标题作为文件名
            filename = re.sub(r'[\\/:*?"<>|]', '_', title)[:50]
        
        filename = f"{filename}.{file_type}"
        
        return self.FILE_DIR / year / source_name / filename
    
    def _save_metadata(
        self,
        documents: List[PolicyDocument],
        filename: str = None
    ):
        """保存元数据到JSONL文件"""
        if not documents:
            return
        
        if filename is None:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"{self.SOURCE.value}_{date_str}.jsonl"
        
        filepath = self.META_DIR / filename
        
        with open(filepath, 'a', encoding='utf-8') as f:
            for doc in documents:
                f.write(doc.to_json() + '\n')
        
        logger.info(f"保存元数据: {len(documents)} 条 -> {filepath}")
    
    def _load_existing_ids(self) -> set:
        """加载已采集的政策ID，用于去重"""
        existing_ids = set()
        
        for jsonl_file in self.META_DIR.glob(f"{self.SOURCE.value}_*.jsonl"):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            existing_ids.add(data.get('id', ''))
                            # 也用发文字号去重
                            if data.get('doc_no'):
                                existing_ids.add(data['doc_no'])
            except Exception as e:
                logger.warning(f"加载元数据失败: {jsonl_file}, 错误: {e}")
        
        return existing_ids
    
    def _get_playwright_browser(self):
        """获取Playwright浏览器实例"""
        if self._browser is None:
            try:
                from ..scraper_base import PlaywrightDriver
                self._browser = PlaywrightDriver(headless=True)
            except ImportError:
                logger.warning("Playwright未安装，无法使用浏览器采集")
                return None
        return self._browser
    
    def _close_browser(self):
        """关闭浏览器"""
        if self._browser:
            self._browser.close()
            self._browser = None
    
    def to_dataframe(self, documents: List[PolicyDocument]) -> pd.DataFrame:
        """将文档列表转换为DataFrame"""
        if not documents:
            return pd.DataFrame()
        
        records = [doc.to_dict() for doc in documents]
        return pd.DataFrame(records)


def get_policy_collector(source: Union[str, PolicySource]) -> BasePolicyCollector:
    """
    获取指定来源的政策采集器
    
    Args:
        source: 政策来源
        
    Returns:
        采集器实例
    """
    if isinstance(source, str):
        source = PolicySource(source.lower())
    
    if source == PolicySource.CSRC:
        from .csrc import CSRCCollector
        return CSRCCollector()
    elif source == PolicySource.GOV:
        from .gov_council import GovCouncilCollector
        return GovCouncilCollector()
    elif source == PolicySource.NDRC:
        from .gov_council import NDRCCollector
        return NDRCCollector()
    else:
        raise ValueError(f"不支持的政策来源: {source}")
