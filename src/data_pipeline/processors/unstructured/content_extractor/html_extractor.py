"""
HTML文本提取器

使用trafilatura和readability-lxml从HTML页面中提取正文内容
支持：
- 国务院政策页面
- 发改委政策页面
- 其他政府网站

特性：
- 自动去除导航、广告、侧边栏
- 保留正文结构
- 支持多种编码
"""

import re
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import (
    BaseExtractor,
    ExtractorResult,
    DataSourceType,
    ContentType,
    TextCleaningMixin,
)

logger = logging.getLogger(__name__)


class HTMLExtractor(BaseExtractor, TextCleaningMixin):
    """
    HTML正文提取器
    
    使用trafilatura和readability-lxml提取网页正文
    
    特点：
    1. 智能正文识别：自动识别并提取文章主体内容
    2. 噪音过滤：自动去除广告、导航、侧边栏等
    3. 多引擎备份：trafilatura为主，readability为备
    4. 编码自适应：自动检测并处理各种编码
    """
    
    VERSION = "1.0.0"
    
    # 默认请求头
    DEFAULT_HEADERS = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
    }
    
    # 政策网站特定配置
    SITE_CONFIGS = {
        'www.gov.cn': {
            'encoding': 'utf-8',
            'referer': 'https://www.gov.cn/',
            'source_type': DataSourceType.GOV_POLICY,
        },
        'ndrc.gov.cn': {
            'encoding': 'utf-8',
            'referer': 'https://www.ndrc.gov.cn/',
            'source_type': DataSourceType.NDRC_POLICY,
        },
    }
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
        prefer_trafilatura: bool = True,
        include_tables: bool = True,
        include_links: bool = False
    ):
        """
        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            user_agent: 自定义User-Agent
            prefer_trafilatura: 优先使用trafilatura（否则使用readability）
            include_tables: 是否包含表格内容
            include_links: 是否包含链接文本
        """
        super().__init__(timeout, max_retries, user_agent)
        self.prefer_trafilatura = prefer_trafilatura
        self.include_tables = include_tables
        self.include_links = include_links
        self._trafilatura = None
        self._readability = None
    
    @property
    def trafilatura(self):
        """懒加载trafilatura"""
        if self._trafilatura is None:
            try:
                import trafilatura
                self._trafilatura = trafilatura
                logger.debug("trafilatura initialized successfully")
            except ImportError:
                raise ImportError(
                    "trafilatura not installed. "
                    "Install it with: pip install trafilatura"
                )
        return self._trafilatura
    
    @property
    def readability(self):
        """懒加载readability-lxml"""
        if self._readability is None:
            try:
                from readability import Document
                self._readability = Document
                logger.debug("readability-lxml initialized successfully")
            except ImportError:
                logger.warning(
                    "readability-lxml not installed. "
                    "Install it with: pip install readability-lxml"
                )
                self._readability = False  # 标记为不可用
        return self._readability
    
    def _create_session(self) -> requests.Session:
        """创建带重试机制的HTTP会话"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_site_config(self, url: str) -> Dict[str, Any]:
        """获取网站特定配置"""
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # 查找匹配的配置
        for site_domain, config in self.SITE_CONFIGS.items():
            if site_domain in domain:
                return config
        
        # 默认配置
        return {
            'encoding': None,  # 自动检测
            'referer': None,
            'source_type': DataSourceType.GENERIC_HTML,
        }
    
    def _get_headers(self, url: str) -> Dict[str, str]:
        """根据URL生成请求头"""
        headers = dict(self.DEFAULT_HEADERS)
        headers['User-Agent'] = self.user_agent
        
        config = self._get_site_config(url)
        if config.get('referer'):
            headers['Referer'] = config['referer']
        
        return headers
    
    def _download_html(
        self,
        url: str
    ) -> Tuple[Optional[str], Optional[str], DataSourceType]:
        """
        下载HTML页面
        
        Args:
            url: 页面URL
        
        Returns:
            (HTML内容, 错误信息, 数据源类型)
        """
        session = self._create_session()
        headers = self._get_headers(url)
        config = self._get_site_config(url)
        
        try:
            response = session.get(
                url,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # 设置编码 - 优先从配置获取，否则从meta标签检测
            if config.get('encoding'):
                response.encoding = config['encoding']
            else:
                # 尝试从HTML中的meta标签检测编码
                detected_encoding = self._detect_encoding_from_html(response.content)
                if detected_encoding:
                    response.encoding = detected_encoding
                elif response.encoding is None or response.encoding == 'ISO-8859-1':
                    response.encoding = response.apparent_encoding or 'utf-8'
            
            return response.text, None, config.get('source_type', DataSourceType.GENERIC_HTML)
            
        except requests.exceptions.Timeout:
            return None, f"Request timeout after {self.timeout}s", DataSourceType.GENERIC_HTML
        except requests.exceptions.HTTPError as e:
            return None, f"HTTP error: {e.response.status_code}", DataSourceType.GENERIC_HTML
        except requests.exceptions.RequestException as e:
            return None, f"Request failed: {str(e)}", DataSourceType.GENERIC_HTML
        finally:
            session.close()
    
    def _detect_encoding_from_html(self, content: bytes) -> Optional[str]:
        """从HTML meta标签检测编码"""
        import re
        
        # 先用ASCII解码前1000字节来查找meta charset
        try:
            head = content[:1000].decode('ascii', errors='ignore')
            
            # 匹配 <meta charset="utf-8">
            match = re.search(r'<meta[^>]+charset=["\']?([^"\'>\s]+)', head, re.I)
            if match:
                return match.group(1).strip()
            
            # 匹配 <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
            match = re.search(r'content=["\'][^"\']*charset=([^"\';\s]+)', head, re.I)
            if match:
                return match.group(1).strip()
                
        except Exception:
            pass
        
        return None
    
    def extract(
        self,
        url: str,
        source_type: Optional[DataSourceType] = None,
        **kwargs
    ) -> ExtractorResult:
        """
        从URL下载并提取HTML正文内容
        
        Args:
            url: 页面URL
            source_type: 数据源类型（可选，自动检测）
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
                source_type=source_type or DataSourceType.GENERIC_HTML
            )
        
        # 下载HTML
        html_content, error, detected_type = self._download_html(url)
        if error:
            return ExtractorResult.failure(
                source_url=url,
                error_message=error,
                error_code="DOWNLOAD_ERROR",
                source_type=source_type or detected_type
            )
        
        # 使用指定的或检测到的数据源类型
        final_source_type = source_type or detected_type
        
        # 提取文本
        result = self.extract_from_bytes(
            html_content.encode('utf-8'),
            source_url=url,
            source_type=final_source_type,
            **kwargs
        )
        
        # 更新处理时间
        result.process_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def extract_from_bytes(
        self,
        content: bytes,
        source_url: str = "",
        source_type: DataSourceType = DataSourceType.GENERIC_HTML,
        **kwargs
    ) -> ExtractorResult:
        """
        从字节流提取HTML正文内容
        
        Args:
            content: HTML字节内容
            source_url: 源URL
            source_type: 数据源类型
            **kwargs: 额外参数
        
        Returns:
            ExtractorResult: 提取结果
        """
        start_time = time.time()
        
        try:
            # 解码HTML
            try:
                html_text = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    html_text = content.decode('gbk')
                except UnicodeDecodeError:
                    html_text = content.decode('utf-8', errors='ignore')
            
            # 尝试提取正文
            extracted_text = None
            extraction_method = None
            
            # 优先使用trafilatura
            if self.prefer_trafilatura:
                extracted_text = self._extract_with_trafilatura(html_text, source_url)
                if extracted_text:
                    extraction_method = 'trafilatura'
            
            # 如果trafilatura失败，尝试readability
            if not extracted_text and self.readability and self.readability is not False:
                extracted_text = self._extract_with_readability(html_text)
                if extracted_text:
                    extraction_method = 'readability'
            
            # 如果两者都失败，尝试基础提取
            if not extracted_text:
                extracted_text = self._extract_basic(html_text)
                if extracted_text:
                    extraction_method = 'basic'
            
            if not extracted_text or not extracted_text.strip():
                return ExtractorResult.failure(
                    source_url=source_url,
                    error_message="No text content extracted from HTML",
                    error_code="EMPTY_CONTENT",
                    source_type=source_type
                )
            
            # 清洗文本
            clean_text = self._clean_html_text(extracted_text, source_type)
            
            return ExtractorResult(
                success=True,
                content_text=clean_text,
                source_url=source_url,
                source_type=source_type,
                content_type=ContentType.HTML,
                process_time_ms=(time.time() - start_time) * 1000,
                extractor_version=self.VERSION,
                metadata={'extraction_method': extraction_method}
            )
            
        except Exception as e:
            logger.error(f"HTML extraction failed for {source_url}: {e}")
            return ExtractorResult.failure(
                source_url=source_url,
                error_message=str(e),
                error_code="EXTRACT_ERROR",
                source_type=source_type
            )
    
    def _extract_with_trafilatura(
        self,
        html_text: str,
        url: str = ""
    ) -> Optional[str]:
        """使用trafilatura提取正文"""
        try:
            text = self.trafilatura.extract(
                html_text,
                url=url,
                include_tables=self.include_tables,
                include_links=self.include_links,
                include_comments=False,
                no_fallback=False,
                favor_precision=True,
            )
            # 验证提取结果是否有效
            if text and len(text.strip()) > 50:
                return text
            return None
        except Exception as e:
            logger.debug(f"trafilatura extraction failed: {e}")
            return None
    
    def _extract_with_readability(self, html_text: str) -> Optional[str]:
        """使用readability-lxml提取正文"""
        try:
            doc = self.readability(html_text)
            # 获取内容
            content_html = doc.summary()
            # 将HTML转为纯文本
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content_html, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            # 验证提取结果是否有效
            if text and len(text.strip()) > 50:
                return text
            return None
        except Exception as e:
            logger.debug(f"readability extraction failed: {e}")
            return None
    
    def _extract_basic(self, html_text: str) -> Optional[str]:
        """基础HTML文本提取（备用方案）"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_text, 'html.parser')
            
            # 移除脚本和样式
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 
                            'aside', 'iframe', 'noscript']):
                tag.decompose()
            
            # 尝试找到正文容器（增强政府网站支持）
            content_selectors = [
                # 通用选择器
                'article',
                '.article-content',
                '.content',
                '.main-content',
                '#content',
                '#article',
                '.news-content',
                '.policy-content',
                '.text-content',
                'main',
                # 政府网站特定选择器
                '.pages_content',          # 国务院网站
                '#UCAP-CONTENT',           # 国务院网站
                '.article',
                '.article-con',
                '#zoom',                   # 部分政府网站
                '.TRS_Editor',             # TRS系统
                '.TRS_PreAppend',
                '#fontzoom',
                '.bt_content',
                '.news_txt',
                '.zwgk_content',           # 政务公开
                '.con_con',
                '.detail_content',
                '.word_content',
                '#myContent',
                '.conment',                # 有时是typo
                '.news-detail',
            ]
            
            content = None
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content and len(content.get_text(strip=True)) > 100:  # 确保内容有实质
                    break
                content = None
            
            if content:
                text = content.get_text(separator='\n', strip=True)
            else:
                # 退而求其次，获取body内容
                body = soup.find('body')
                text = body.get_text(separator='\n', strip=True) if body else soup.get_text(separator='\n', strip=True)
            
            return text
            
        except Exception as e:
            logger.debug(f"basic extraction failed: {e}")
            return None
    
    def _clean_html_text(
        self,
        text: str,
        source_type: DataSourceType = DataSourceType.GENERIC_HTML
    ) -> str:
        """
        清洗HTML提取的文本
        
        Args:
            text: 原始文本
            source_type: 数据源类型
        
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 基础清洗
        text = self._clean_text(text)
        
        # 根据数据源类型进行特定清洗
        if source_type == DataSourceType.GOV_POLICY:
            text = self._clean_gov_policy(text)
        elif source_type == DataSourceType.NDRC_POLICY:
            text = self._clean_ndrc_policy(text)
        
        return text
    
    def _clean_gov_policy(self, text: str) -> str:
        """清洗国务院政策文本"""
        # 移除常见导航文字
        patterns = [
            r'首页.*?当前位置.*?\n',
            r'来源[：:]\s*.*?\n',
            r'发布时间[：:]\s*\d{4}[-/]\d{2}[-/]\d{2}.*?\n',
            r'字号[：:]\s*\[.*?\]\s*\n',
            r'打印\s*本页\s*',
            r'关闭窗口\s*',
            r'分享到[：:]\s*',
            r'微信\s*微博\s*',
            r'责任编辑[：:]\s*\S+\s*',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 移除空白段落
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _clean_ndrc_policy(self, text: str) -> str:
        """清洗发改委政策文本"""
        # 移除常见导航文字
        patterns = [
            r'首页.*?当前位置.*?\n',
            r'来源[：:]\s*.*?\n',
            r'发布时间[：:]\s*\d{4}[-/]\d{2}[-/]\d{2}.*?\n',
            r'字号[：:]\s*\[.*?\]\s*\n',
            r'打印\s*',
            r'关闭\s*',
            r'分享\s*',
            r'国家发展和改革委员会.*?\n',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 移除空白段落
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def extract_batch(
        self,
        urls: List[str],
        source_type: Optional[DataSourceType] = None,
        max_workers: int = 4,
        **kwargs
    ) -> List[ExtractorResult]:
        """
        批量提取HTML正文
        
        Args:
            urls: URL列表
            source_type: 数据源类型
            max_workers: 最大并发数
            **kwargs: 额外参数
        
        Returns:
            ExtractorResult列表
        """
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
                        source_type=source_type or DataSourceType.GENERIC_HTML
                    ))
        
        return results
