"""
内容提取器工厂

根据数据源类型或URL特征自动选择合适的提取器

支持的数据源：
- 公告(CNINFO_ANNOUNCEMENT): 巨潮资讯PDF -> PDFExtractor
- 事件(CNINFO_EVENT): 巨潮资讯详情页 -> CninfoDetailParser -> PDFExtractor
- 研报(EASTMONEY_REPORT): 东方财富PDF -> PDFExtractor
- 国务院政策(GOV_POLICY): 政府网站HTML -> HTMLExtractor
- 发改委政策(NDRC_POLICY): 政府网站HTML -> HTMLExtractor
- CCTV新闻(CCTV_NEWS): 已有content -> 直接返回
- 交易所公告(EXCHANGE_NEWS): 从title提取 -> 直接返回title
"""

import re
import logging
from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse

from .base import (
    BaseExtractor,
    ExtractorResult,
    DataSourceType,
    ContentType,
)
from .pdf_extractor import PDFExtractor
from .html_extractor import HTMLExtractor
from .cninfo_detail_parser import CninfoDetailParser

logger = logging.getLogger(__name__)


class ContentExtractorFactory:
    """
    内容提取器工厂
    
    职责：
    1. 根据数据源类型创建对应的提取器
    2. 根据URL特征自动识别数据源类型
    3. 缓存提取器实例以提高性能
    """
    
    # URL模式 -> 数据源类型映射
    URL_PATTERNS = {
        # PDF URL模式
        r'static\.cninfo\.com\.cn.*\.pdf': DataSourceType.CNINFO_ANNOUNCEMENT,
        r'www\.cninfo\.com\.cn/new/disclosure/detail': DataSourceType.CNINFO_EVENT,
        r'pdf\.dfcfw\.com.*\.pdf': DataSourceType.EASTMONEY_REPORT,
        r'data\.eastmoney\.com.*\.pdf': DataSourceType.EASTMONEY_REPORT,
        
        # HTML URL模式
        r'www\.gov\.cn/zhengce': DataSourceType.GOV_POLICY,
        r'www\.gov\.cn/.*content': DataSourceType.GOV_POLICY,
        r'www\.ndrc\.gov\.cn': DataSourceType.NDRC_POLICY,
        r'ndrc\.gov\.cn': DataSourceType.NDRC_POLICY,
        
        # 交易所公告
        r'www\.sse\.com\.cn': DataSourceType.EXCHANGE_NEWS,
        r'www\.szse\.cn': DataSourceType.EXCHANGE_NEWS,
    }
    
    # 数据源类型 -> 提取器类映射
    EXTRACTOR_MAPPING = {
        DataSourceType.CNINFO_ANNOUNCEMENT: PDFExtractor,
        DataSourceType.CNINFO_EVENT: CninfoDetailParser,
        DataSourceType.EASTMONEY_REPORT: PDFExtractor,
        DataSourceType.GOV_POLICY: HTMLExtractor,
        DataSourceType.NDRC_POLICY: HTMLExtractor,
        DataSourceType.GENERIC_PDF: PDFExtractor,
        DataSourceType.GENERIC_HTML: HTMLExtractor,
    }
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
        cache_extractors: bool = True
    ):
        """
        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            user_agent: 自定义User-Agent
            cache_extractors: 是否缓存提取器实例
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.cache_extractors = cache_extractors
        
        # 提取器缓存
        self._extractor_cache: Dict[DataSourceType, BaseExtractor] = {}
        
        # 共享的PDF提取器（用于CninfoDetailParser等）
        self._shared_pdf_extractor: Optional[PDFExtractor] = None
    
    @property
    def shared_pdf_extractor(self) -> PDFExtractor:
        """获取共享的PDF提取器"""
        if self._shared_pdf_extractor is None:
            self._shared_pdf_extractor = PDFExtractor(
                timeout=self.timeout,
                max_retries=self.max_retries,
                user_agent=self.user_agent
            )
        return self._shared_pdf_extractor
    
    def detect_source_type(self, url: str) -> DataSourceType:
        """
        根据URL检测数据源类型
        
        Args:
            url: 源URL
        
        Returns:
            DataSourceType: 检测到的数据源类型
        """
        if not url:
            return DataSourceType.UNKNOWN
        
        url_lower = url.lower()
        
        # 按模式匹配
        for pattern, source_type in self.URL_PATTERNS.items():
            if re.search(pattern, url_lower):
                return source_type
        
        # 根据URL后缀判断
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        
        if path_lower.endswith('.pdf'):
            return DataSourceType.GENERIC_PDF
        elif path_lower.endswith(('.html', '.htm', '.shtml')):
            return DataSourceType.GENERIC_HTML
        
        # 如果无法判断，默认为HTML
        return DataSourceType.GENERIC_HTML
    
    def get_extractor(
        self,
        source_type: DataSourceType
    ) -> Optional[BaseExtractor]:
        """
        获取指定数据源类型的提取器
        
        Args:
            source_type: 数据源类型
        
        Returns:
            对应的提取器实例，或None（对于不需要提取的类型）
        """
        # 不需要提取器的类型
        if source_type in (DataSourceType.CCTV_NEWS, DataSourceType.EXCHANGE_NEWS):
            return None
        
        # 检查缓存
        if self.cache_extractors and source_type in self._extractor_cache:
            return self._extractor_cache[source_type]
        
        # 获取提取器类
        extractor_cls = self.EXTRACTOR_MAPPING.get(source_type)
        if extractor_cls is None:
            logger.warning(f"No extractor for source type: {source_type}")
            return None
        
        # 创建提取器实例
        if extractor_cls == CninfoDetailParser:
            # CninfoDetailParser需要共享PDF提取器
            extractor = CninfoDetailParser(
                timeout=self.timeout,
                max_retries=self.max_retries,
                user_agent=self.user_agent,
                pdf_extractor=self.shared_pdf_extractor
            )
        else:
            extractor = extractor_cls(
                timeout=self.timeout,
                max_retries=self.max_retries,
                user_agent=self.user_agent
            )
        
        # 缓存
        if self.cache_extractors:
            self._extractor_cache[source_type] = extractor
        
        return extractor
    
    def create_extractor_for_url(
        self,
        url: str
    ) -> Optional[BaseExtractor]:
        """
        根据URL创建合适的提取器
        
        Args:
            url: 源URL
        
        Returns:
            对应的提取器实例
        """
        source_type = self.detect_source_type(url)
        return self.get_extractor(source_type)
    
    def extract(
        self,
        url: str,
        source_type: Optional[DataSourceType] = None,
        title: Optional[str] = None,
        existing_content: Optional[str] = None,
        **kwargs
    ) -> ExtractorResult:
        """
        提取URL内容
        
        Args:
            url: 源URL
            source_type: 数据源类型（可选，自动检测）
            title: 标题（用于EXCHANGE_NEWS等类型）
            existing_content: 已有内容（用于CCTV_NEWS等类型）
            **kwargs: 传递给提取器的额外参数
        
        Returns:
            ExtractorResult: 提取结果
        """
        # 确定数据源类型
        if source_type is None:
            source_type = self.detect_source_type(url)
        
        # 处理不需要提取的类型
        if source_type == DataSourceType.CCTV_NEWS:
            # CCTV新闻已有content
            if existing_content:
                return ExtractorResult(
                    success=True,
                    content_text=existing_content,
                    source_url=url,
                    source_type=source_type,
                    content_type=ContentType.TEXT,
                    title=title,
                )
            return ExtractorResult.failure(
                source_url=url,
                error_message="CCTV news requires existing_content parameter",
                error_code="MISSING_CONTENT",
                source_type=source_type
            )
        
        if source_type == DataSourceType.EXCHANGE_NEWS:
            # 交易所公告从title提取
            if title:
                return ExtractorResult(
                    success=True,
                    content_text=title,
                    source_url=url,
                    source_type=source_type,
                    content_type=ContentType.TITLE_ONLY,
                    title=title,
                )
            return ExtractorResult.failure(
                source_url=url,
                error_message="Exchange news requires title parameter",
                error_code="MISSING_TITLE",
                source_type=source_type
            )
        
        # 获取提取器
        extractor = self.get_extractor(source_type)
        if extractor is None:
            return ExtractorResult.failure(
                source_url=url,
                error_message=f"No extractor available for source type: {source_type}",
                error_code="NO_EXTRACTOR",
                source_type=source_type
            )
        
        # 执行提取
        return extractor.extract(url, source_type=source_type, **kwargs)
    
    def clear_cache(self):
        """清除提取器缓存"""
        self._extractor_cache.clear()
        self._shared_pdf_extractor = None
