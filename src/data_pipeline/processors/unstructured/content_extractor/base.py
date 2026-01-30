"""
内容提取器基类模块

定义内容提取的统一接口和数据结构
"""

import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """数据源类型枚举"""
    # 公告来源
    CNINFO_ANNOUNCEMENT = "cninfo_announcement"     # 巨潮资讯公告PDF
    CNINFO_EVENT = "cninfo_event"                   # 巨潮资讯事件详情页->PDF
    
    # 研报来源
    EASTMONEY_REPORT = "eastmoney_report"           # 东方财富研报PDF
    
    # 政策来源
    GOV_POLICY = "gov_policy"                       # 国务院政策HTML
    NDRC_POLICY = "ndrc_policy"                     # 发改委政策HTML
    
    # 新闻来源
    CCTV_NEWS = "cctv_news"                         # CCTV新闻（已有content）
    EXCHANGE_NEWS = "exchange_news"                 # 交易所公告（从title提取）
    
    # 通用
    GENERIC_PDF = "generic_pdf"                     # 通用PDF
    GENERIC_HTML = "generic_html"                   # 通用HTML
    UNKNOWN = "unknown"


class ContentType(Enum):
    """内容类型枚举"""
    PDF = "pdf"
    HTML = "html"
    HTML_WITH_PDF = "html_with_pdf"     # HTML页面包含PDF链接(如巨潮事件详情)
    TEXT = "text"                        # 纯文本
    TITLE_ONLY = "title_only"           # 仅标题（如交易所公告）


@dataclass
class ExtractorResult:
    """提取结果数据结构"""
    # 成功标志
    success: bool
    
    # 提取的文本内容
    content_text: Optional[str] = None
    
    # 源信息
    source_url: str = ""
    source_type: DataSourceType = DataSourceType.UNKNOWN
    content_type: ContentType = ContentType.TEXT
    
    # 元数据
    title: Optional[str] = None
    page_count: Optional[int] = None          # PDF页数
    char_count: Optional[int] = None          # 字符数
    word_count: Optional[int] = None          # 词数（中文按字计）
    
    # 处理信息
    process_time_ms: float = 0.0              # 处理耗时（毫秒）
    extractor_version: str = "1.0.0"
    
    # 错误信息
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # 实际PDF链接（用于巨潮详情页解析）
    actual_pdf_url: Optional[str] = None
    
    # 额外元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """计算字符数和词数"""
        if self.content_text and self.char_count is None:
            self.char_count = len(self.content_text)
        if self.content_text and self.word_count is None:
            # 中文按字计，英文按空格分词
            self.word_count = len(self.content_text.replace(' ', ''))
    
    @staticmethod
    def failure(
        source_url: str,
        error_message: str,
        error_code: str = "EXTRACT_ERROR",
        source_type: DataSourceType = DataSourceType.UNKNOWN
    ) -> 'ExtractorResult':
        """创建失败结果的便捷方法"""
        return ExtractorResult(
            success=False,
            source_url=source_url,
            source_type=source_type,
            error_message=error_message,
            error_code=error_code,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'content_text': self.content_text,
            'source_url': self.source_url,
            'source_type': self.source_type.value,
            'content_type': self.content_type.value,
            'title': self.title,
            'page_count': self.page_count,
            'char_count': self.char_count,
            'word_count': self.word_count,
            'process_time_ms': self.process_time_ms,
            'extractor_version': self.extractor_version,
            'error_message': self.error_message,
            'error_code': self.error_code,
            'actual_pdf_url': self.actual_pdf_url,
            'metadata': self.metadata,
        }


class BaseExtractor(ABC):
    """
    内容提取器基类
    
    所有具体提取器（PDF、HTML等）都应继承此类
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None
    ):
        """
        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            user_agent: 自定义User-Agent
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent or self._default_user_agent()
        self._session = None
    
    @staticmethod
    def _default_user_agent() -> str:
        """默认User-Agent"""
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    
    @abstractmethod
    def extract(self, url: str, **kwargs) -> ExtractorResult:
        """
        从URL提取文本内容
        
        Args:
            url: 源URL
            **kwargs: 额外参数
        
        Returns:
            ExtractorResult: 提取结果
        """
        pass
    
    @abstractmethod
    def extract_from_bytes(
        self,
        content: bytes,
        source_url: str = "",
        **kwargs
    ) -> ExtractorResult:
        """
        从字节流提取文本内容
        
        Args:
            content: 字节内容
            source_url: 源URL（用于记录）
            **kwargs: 额外参数
        
        Returns:
            ExtractorResult: 提取结果
        """
        pass
    
    def _generate_content_id(self, url: str, content: bytes) -> str:
        """生成内容ID（基于URL和内容的哈希）"""
        hash_input = f"{url}:{len(content)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _clean_text(self, text: str) -> str:
        """
        清洗文本，去除冗余信息
        
        Args:
            text: 原始文本
        
        Returns:
            清洗后的文本
        """
        import re
        
        if not text:
            return ""
        
        # 1. 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 2. 移除控制字符（保留换行和制表符）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 3. 移除连续空格（保留单个空格）
        text = re.sub(r' +', ' ', text)
        
        # 4. 移除连续换行（保留最多两个）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 5. 移除每行首尾空格
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # 6. 移除常见的页眉页脚模式
        # 移除页码
        text = re.sub(r'\n\s*[-—]\s*\d+\s*[-—]\s*\n', '\n', text)
        text = re.sub(r'\n\s*第\s*\d+\s*页\s*(共\s*\d+\s*页)?\s*\n', '\n', text)
        
        # 7. 移除常见PDF格式冗余
        # 巨潮资讯页脚
        text = re.sub(r'本公司.*?信息披露.*?指定.*?媒体.*?\n?', '', text)
        text = re.sub(r'公告编号[：:]\s*\d+[-/]\d+\s*', '', text)
        
        # 8. 合并被分割的段落（连续的短行可能是同一段）
        # 这个处理需要根据实际情况调整
        
        # 9. 最终清理
        text = text.strip()
        
        return text


class TextCleaningMixin:
    """文本清洗混入类，提供GPU加速的文本清洗功能"""
    
    @staticmethod
    def clean_text_batch_gpu(texts: List[str]) -> List[str]:
        """
        使用cuDF进行批量文本清洗（GPU加速）
        
        注意：仅在处理大批量数据时使用此方法才有性能优势
        
        Args:
            texts: 文本列表
        
        Returns:
            清洗后的文本列表
        """
        try:
            import cudf
            
            # 创建cuDF Series
            s = cudf.Series(texts)
            
            # GPU加速的字符串操作
            # 1. 统一换行符
            s = s.str.replace('\r\n', '\n', regex=False)
            s = s.str.replace('\r', '\n', regex=False)
            
            # 2. 移除连续空格
            s = s.str.replace(r' +', ' ', regex=True)
            
            # 3. 移除连续换行（保留最多两个）
            s = s.str.replace(r'\n{3,}', '\n\n', regex=True)
            
            # 4. 去除首尾空白
            s = s.str.strip()
            
            return s.to_arrow().to_pylist()
            
        except ImportError:
            logger.warning("cuDF not available, falling back to CPU processing")
            return [TextCleaningMixin._clean_text_cpu(t) for t in texts]
        except Exception as e:
            logger.warning(f"GPU text cleaning failed: {e}, falling back to CPU")
            return [TextCleaningMixin._clean_text_cpu(t) for t in texts]
    
    @staticmethod
    def _clean_text_cpu(text: str) -> str:
        """CPU版本的文本清洗"""
        import re
        
        if not text:
            return ""
        
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
