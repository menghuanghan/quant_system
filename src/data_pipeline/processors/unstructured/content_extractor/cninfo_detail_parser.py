"""
巨潮资讯详情页解析器

专门用于处理巨潮资讯事件详情页：
- 解析 http://www.cninfo.com.cn/new/disclosure/detail?announcementId=xxx 这类URL
- 提取真实的PDF下载链接
- 下载并提取PDF内容

工作流程：
1. 如果提供了直接的PDF URL，直接使用
2. 否则尝试从详情页HTML解析获取PDF链接
3. 或者尝试根据announcementId构造PDF链接
4. 调用PDFExtractor提取文本

注意：对于事件数据，建议使用数据中的pdf_url字段（如果存在）
"""

import re
import json
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import (
    BaseExtractor,
    ExtractorResult,
    DataSourceType,
    ContentType,
)
from .pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)


class CninfoDetailParser(BaseExtractor):
    """
    巨潮资讯详情页解析器
    
    处理两类URL：
    1. 详情页URL: http://www.cninfo.com.cn/new/disclosure/detail?announcementId=xxx
    2. 直接PDF URL: http://static.cninfo.com.cn/finalpage/xxx/xxx.PDF
    
    对于详情页URL，会先解析获取真实PDF链接，然后提取内容
    """
    
    VERSION = "1.0.0"
    
    # 巨潮资讯配置
    BASE_URL = "http://www.cninfo.com.cn"
    STATIC_URL = "http://static.cninfo.com.cn"
    DETAIL_API = "http://www.cninfo.com.cn/new/disclosure/detail"
    
    # 获取公告详情的API（用于获取adjunctUrl）
    ANNOUNCEMENT_API = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    
    # 请求头
    DEFAULT_HEADERS = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'keep-alive',
        'Referer': 'http://www.cninfo.com.cn/',
    }
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
        pdf_extractor: Optional[PDFExtractor] = None
    ):
        """
        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            user_agent: 自定义User-Agent
            pdf_extractor: 可选的PDF提取器实例（共享时提高性能）
        """
        super().__init__(timeout, max_retries, user_agent)
        self._pdf_extractor = pdf_extractor
    
    @property
    def pdf_extractor(self) -> PDFExtractor:
        """获取或创建PDF提取器"""
        if self._pdf_extractor is None:
            self._pdf_extractor = PDFExtractor(
                timeout=self.timeout,
                max_retries=self.max_retries,
                user_agent=self.user_agent
            )
        return self._pdf_extractor
    
    def _create_session(self) -> requests.Session:
        """创建带重试机制的HTTP会话"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_headers(self) -> Dict[str, str]:
        """生成请求头"""
        headers = dict(self.DEFAULT_HEADERS)
        headers['User-Agent'] = self.user_agent
        return headers
    
    def _is_detail_page_url(self, url: str) -> bool:
        """判断是否为详情页URL"""
        return 'disclosure/detail' in url.lower() and 'announcementId' in url
    
    def _is_direct_pdf_url(self, url: str) -> bool:
        """判断是否为直接PDF URL"""
        return url.lower().endswith('.pdf') or 'finalpage' in url.lower()
    
    def _extract_announcement_id(self, url: str) -> Optional[str]:
        """从URL中提取公告ID"""
        # 尝试从URL参数中提取
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        
        if 'announcementId' in params:
            return params['announcementId'][0]
        
        # 尝试从URL路径中提取
        match = re.search(r'announcementId[=/](\d+)', url)
        if match:
            return match.group(1)
        
        return None
    
    def _fetch_pdf_url_from_detail(
        self,
        url: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        从详情页获取PDF下载链接
        
        Args:
            url: 详情页URL
        
        Returns:
            (PDF URL, 错误信息)
        """
        session = self._create_session()
        headers = self._get_headers()
        
        try:
            # 请求详情页
            response = session.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            response.encoding = 'utf-8'
            html_content = response.text
            
            # 方法1: 从页面JavaScript中提取adjunctUrl
            pdf_url = self._extract_adjunct_url_from_html(html_content)
            if pdf_url:
                return self._build_full_pdf_url(pdf_url), None
            
            # 方法2: 从页面中查找PDF链接
            pdf_url = self._extract_pdf_link_from_html(html_content)
            if pdf_url:
                return self._build_full_pdf_url(pdf_url), None
            
            # 方法3: 使用API获取
            announcement_id = self._extract_announcement_id(url)
            if announcement_id:
                pdf_url = self._fetch_pdf_url_from_api(announcement_id, session)
                if pdf_url:
                    return pdf_url, None
            
            return None, "Could not find PDF URL in detail page"
            
        except requests.exceptions.RequestException as e:
            return None, f"Failed to fetch detail page: {str(e)}"
        finally:
            session.close()
    
    def _extract_adjunct_url_from_html(self, html_content: str) -> Optional[str]:
        """从HTML中提取adjunctUrl"""
        # 方法1: 匹配JavaScript变量
        patterns = [
            r'adjunctUrl["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            r'"adjunctUrl"\s*:\s*"([^"]+)"',
            r"'adjunctUrl'\s*:\s*'([^']+)'",
            r'pdfUrl["\']?\s*[:=]\s*["\']([^"\']+\.pdf)["\']',
            r'downloadUrl["\']?\s*[:=]\s*["\']([^"\']+\.pdf)["\']',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                url = match.group(1)
                if url and ('.pdf' in url.lower() or 'finalpage' in url.lower()):
                    return url
        
        return None
    
    def _extract_pdf_link_from_html(self, html_content: str) -> Optional[str]:
        """从HTML中提取PDF链接"""
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 查找PDF链接
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if '.pdf' in href.lower() or 'finalpage' in href.lower():
                    return href
            
            # 查找iframe中的PDF
            for iframe in soup.find_all('iframe', src=True):
                src = iframe['src']
                if '.pdf' in src.lower() or 'finalpage' in src.lower():
                    return src
            
            # 查找embed中的PDF
            for embed in soup.find_all('embed', src=True):
                src = embed['src']
                if '.pdf' in src.lower() or 'finalpage' in src.lower():
                    return src
                    
        except Exception as e:
            logger.debug(f"Failed to parse HTML for PDF link: {e}")
        
        return None
    
    def _fetch_pdf_url_from_api(
        self,
        announcement_id: str,
        session: requests.Session
    ) -> Optional[str]:
        """通过API获取PDF URL"""
        # 巨潮资讯的公告详情API
        api_url = f"{self.BASE_URL}/new/announcement/query"
        
        headers = self._get_headers()
        headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=UTF-8'
        headers['Accept'] = 'application/json'
        
        data = {
            'announcementId': announcement_id,
        }
        
        try:
            response = session.post(
                api_url,
                headers=headers,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                # 解析返回的JSON
                if isinstance(result, dict):
                    adjunct_url = result.get('adjunctUrl') or result.get('pdfUrl')
                    if adjunct_url:
                        return self._build_full_pdf_url(adjunct_url)
                        
        except Exception as e:
            logger.debug(f"API call failed: {e}")
        
        return None
    
    def _guess_pdf_url_from_id(
        self,
        announcement_id: str,
        date_hint: Optional[str] = None
    ) -> Optional[str]:
        """
        根据announcementId猜测PDF URL
        
        巨潮资讯的PDF URL格式通常为:
        http://static.cninfo.com.cn/finalpage/YYYY-MM-DD/{announcementId}.PDF
        
        Args:
            announcement_id: 公告ID
            date_hint: 日期提示（格式：YYYY-MM-DD 或 YYYYMMDD）
        
        Returns:
            猜测的PDF URL，如果无法猜测返回None
        """
        if not announcement_id:
            return None
        
        # 如果提供了日期提示
        if date_hint:
            # 统一日期格式为 YYYY-MM-DD
            if len(date_hint) == 8 and date_hint.isdigit():
                date_str = f"{date_hint[:4]}-{date_hint[4:6]}-{date_hint[6:8]}"
            else:
                date_str = date_hint
            
            pdf_url = f"{self.STATIC_URL}/finalpage/{date_str}/{announcement_id}.PDF"
            return pdf_url
        
        # 没有日期提示，尝试最近几天
        # 这是一个备用方案，可能不太准确
        from datetime import datetime, timedelta
        
        today = datetime.now()
        for days_back in range(0, 30):  # 尝试过去30天
            date = today - timedelta(days=days_back)
            date_str = date.strftime('%Y-%m-%d')
            pdf_url = f"{self.STATIC_URL}/finalpage/{date_str}/{announcement_id}.PDF"
            
            # 发送HEAD请求检查URL是否有效
            try:
                response = requests.head(pdf_url, timeout=5)
                if response.status_code == 200:
                    return pdf_url
            except:
                continue
        
        return None
    
    def _build_full_pdf_url(self, relative_url: str) -> str:
        """构建完整的PDF URL"""
        if relative_url.startswith('http'):
            return relative_url
        
        # 巨潮资讯的PDF通常托管在static.cninfo.com.cn
        if relative_url.startswith('/'):
            return f"{self.STATIC_URL}{relative_url}"
        else:
            return f"{self.STATIC_URL}/{relative_url}"
    
    def extract(
        self,
        url: str,
        source_type: DataSourceType = DataSourceType.CNINFO_EVENT,
        pdf_url: Optional[str] = None,
        date: Optional[str] = None,
        **kwargs
    ) -> ExtractorResult:
        """
        从巨潮资讯URL提取文本内容
        
        支持多种方式：
        1. 如果提供了pdf_url参数，直接使用它
        2. 如果url是直接的PDF URL，直接使用
        3. 如果url是详情页URL，尝试解析获取PDF链接
        4. 如果提供了date，可以根据announcementId构造PDF URL
        
        Args:
            url: 巨潮资讯URL（详情页URL或PDF URL）
            source_type: 数据源类型
            pdf_url: 直接提供的PDF URL（优先使用）
            date: 公告日期（用于构造PDF URL，格式：YYYY-MM-DD 或 YYYYMMDD）
            **kwargs: 额外参数
        
        Returns:
            ExtractorResult: 提取结果
        """
        start_time = time.time()
        
        # 验证URL
        if not url or not isinstance(url, str):
            return ExtractorResult.failure(
                source_url=str(url),
                error_message="Invalid URL",
                error_code="INVALID_URL",
                source_type=source_type
            )
        
        actual_pdf_url = None
        
        # 方法1: 使用直接提供的pdf_url参数
        if pdf_url and isinstance(pdf_url, str) and pdf_url.strip():
            actual_pdf_url = pdf_url
            logger.debug(f"Using provided pdf_url: {pdf_url}")
        
        # 方法2: 判断URL是否直接是PDF URL
        elif self._is_direct_pdf_url(url):
            # 直接是PDF URL
            actual_pdf_url = url
            logger.debug(f"Direct PDF URL: {url}")
        
        # 方法3: 详情页URL，尝试多种方法获取PDF链接
        elif self._is_detail_page_url(url):
            logger.debug(f"Detail page URL, extracting PDF link: {url}")
            
            # 先尝试从详情页解析
            actual_pdf_url, error = self._fetch_pdf_url_from_detail(url)
            
            # 如果失败，且提供了日期，尝试构造URL
            if not actual_pdf_url and date:
                announcement_id = self._extract_announcement_id(url)
                if announcement_id:
                    actual_pdf_url = self._guess_pdf_url_from_id(announcement_id, date)
                    if actual_pdf_url:
                        logger.debug(f"Constructed PDF URL from date: {actual_pdf_url}")
            
            if not actual_pdf_url:
                return ExtractorResult.failure(
                    source_url=url,
                    error_message="Could not find PDF URL. Try providing pdf_url or date parameter.",
                    error_code="PDF_URL_NOT_FOUND",
                    source_type=source_type
                )
        else:
            # 尝试作为普通URL处理
            actual_pdf_url, error = self._fetch_pdf_url_from_detail(url)
            if not actual_pdf_url:
                # 尝试直接作为PDF URL
                actual_pdf_url = url
        
        if not actual_pdf_url:
            return ExtractorResult.failure(
                source_url=url,
                error_message="Could not determine PDF URL",
                error_code="PDF_URL_NOT_FOUND",
                source_type=source_type
            )
        
        logger.info(f"Extracting PDF from: {actual_pdf_url}")
        
        # 使用PDF提取器提取内容
        result = self.pdf_extractor.extract(
            actual_pdf_url,
            source_type=source_type,
            **kwargs
        )
        
        # 更新结果信息
        result.source_url = url  # 保持原始URL
        result.actual_pdf_url = actual_pdf_url  # 记录实际PDF URL
        result.content_type = ContentType.HTML_WITH_PDF
        result.process_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def extract_from_bytes(
        self,
        content: bytes,
        source_url: str = "",
        source_type: DataSourceType = DataSourceType.CNINFO_EVENT,
        **kwargs
    ) -> ExtractorResult:
        """
        从字节流提取内容
        
        对于详情页解析器，字节流可能是：
        1. HTML详情页内容
        2. PDF内容
        
        Args:
            content: 字节内容
            source_url: 源URL
            source_type: 数据源类型
            **kwargs: 额外参数
        
        Returns:
            ExtractorResult: 提取结果
        """
        # 检查是否为PDF
        if content.startswith(b'%PDF'):
            return self.pdf_extractor.extract_from_bytes(
                content,
                source_url=source_url,
                source_type=source_type,
                **kwargs
            )
        
        # 尝试作为HTML处理
        try:
            html_content = content.decode('utf-8')
        except UnicodeDecodeError:
            html_content = content.decode('gbk', errors='ignore')
        
        # 从HTML中提取PDF URL
        pdf_url = self._extract_adjunct_url_from_html(html_content)
        if not pdf_url:
            pdf_url = self._extract_pdf_link_from_html(html_content)
        
        if pdf_url:
            full_pdf_url = self._build_full_pdf_url(pdf_url)
            result = self.pdf_extractor.extract(
                full_pdf_url,
                source_type=source_type,
                **kwargs
            )
            result.source_url = source_url
            result.actual_pdf_url = full_pdf_url
            result.content_type = ContentType.HTML_WITH_PDF
            return result
        
        return ExtractorResult.failure(
            source_url=source_url,
            error_message="Could not find PDF URL in HTML content",
            error_code="PDF_URL_NOT_FOUND",
            source_type=source_type
        )
    
    def extract_batch(
        self,
        urls: List[str],
        source_type: DataSourceType = DataSourceType.CNINFO_EVENT,
        max_workers: int = 4,
        **kwargs
    ) -> List[ExtractorResult]:
        """
        批量提取巨潮资讯内容
        
        Args:
            urls: URL列表
            source_type: 数据源类型
            max_workers: 最大并发数
            **kwargs: 额外参数
        
        Returns:
            ExtractorResult列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(
                    self.extract, url, source_type, **kwargs
                ): url
                for url in urls
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch extraction failed for {url}: {e}")
                    results.append(ExtractorResult.failure(
                        source_url=url,
                        error_message=str(e),
                        error_code="BATCH_ERROR",
                        source_type=source_type
                    ))
        
        return results
