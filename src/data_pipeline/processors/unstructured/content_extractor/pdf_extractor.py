"""
PDF文本提取器

使用PyMuPDF (fitz)从PDF文件中提取文本内容
支持：
- 巨潮资讯公告PDF
- 东方财富研报PDF
- 事件相关PDF

特性：
- 内存流处理（不写入磁盘）
- 多线程并行提取（大批量时）
- 文本清洗与格式化
"""

import io
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


class PDFExtractor(BaseExtractor, TextCleaningMixin):
    """
    PDF文本提取器
    
    使用PyMuPDF(fitz)提取PDF文本内容
    
    特点：
    1. 速度快：PyMuPDF是最快的Python PDF库
    2. 内存处理：不需要将PDF保存到磁盘
    3. 表格支持：可以提取简单表格内容
    4. 文本清洗：自动清理冗余格式
    """
    
    VERSION = "1.0.0"
    
    # 默认请求头
    DEFAULT_HEADERS = {
        'Accept': 'application/pdf,*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'keep-alive',
    }
    
    # PDF来源特定的Referer
    REFERERS = {
        DataSourceType.CNINFO_ANNOUNCEMENT: 'http://www.cninfo.com.cn/',
        DataSourceType.CNINFO_EVENT: 'http://www.cninfo.com.cn/',
        DataSourceType.EASTMONEY_REPORT: 'https://www.eastmoney.com/',
    }
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
        extract_images: bool = False,
        extract_tables: bool = True,
        max_pages: Optional[int] = None,
        extraction_timeout: int = 60,
        timeout_fallback_pages: int = 20
    ):
        """
        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            user_agent: 自定义User-Agent
            extract_images: 是否提取图片中的文字(OCR)，暂不支持
            extract_tables: 是否尝试提取表格
            max_pages: 最大处理页数，None表示全部
            extraction_timeout: 文本提取超时时间（秒），默认60秒
            timeout_fallback_pages: 超时后回退处理的页数，默认20页
        """
        super().__init__(timeout, max_retries, user_agent)
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.max_pages = max_pages
        self.extraction_timeout = extraction_timeout
        self.timeout_fallback_pages = timeout_fallback_pages
        self._fitz = None
    
    @property
    def fitz(self):
        """懒加载PyMuPDF"""
        if self._fitz is None:
            try:
                import fitz
                self._fitz = fitz
                logger.debug("PyMuPDF initialized successfully")
            except ImportError:
                raise ImportError(
                    "PyMuPDF not installed. "
                    "Install it with: pip install PyMuPDF"
                )
        return self._fitz
    
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
    
    def _get_headers(
        self,
        url: str,
        source_type: DataSourceType = DataSourceType.GENERIC_PDF
    ) -> Dict[str, str]:
        """根据URL和数据源类型生成请求头"""
        headers = dict(self.DEFAULT_HEADERS)
        headers['User-Agent'] = self.user_agent
        
        # 设置Referer
        if source_type in self.REFERERS:
            headers['Referer'] = self.REFERERS[source_type]
        elif 'cninfo' in url.lower():
            headers['Referer'] = 'http://www.cninfo.com.cn/'
        elif 'dfcfw' in url.lower() or 'eastmoney' in url.lower():
            headers['Referer'] = 'https://www.eastmoney.com/'
        
        return headers
    
    def _download_pdf(
        self,
        url: str,
        source_type: DataSourceType = DataSourceType.GENERIC_PDF
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """
        下载PDF文件到内存
        
        Args:
            url: PDF的URL
            source_type: 数据源类型
        
        Returns:
            (PDF字节内容, 错误信息) - 成功时错误信息为None
        """
        session = self._create_session()
        headers = self._get_headers(url, source_type)
        
        try:
            response = session.get(
                url,
                headers=headers,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            # 检查Content-Type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower() and 'octet-stream' not in content_type.lower():
                # 有些服务器不返回正确的Content-Type，检查内容
                content = response.content
                if not content.startswith(b'%PDF'):
                    return None, f"Response is not a PDF: {content_type}"
                return content, None
            
            return response.content, None
            
        except requests.exceptions.Timeout:
            return None, f"Request timeout after {self.timeout}s"
        except requests.exceptions.HTTPError as e:
            return None, f"HTTP error: {e.response.status_code}"
        except requests.exceptions.RequestException as e:
            return None, f"Request failed: {str(e)}"
        finally:
            session.close()
    
    def extract(
        self,
        url: str,
        source_type: DataSourceType = DataSourceType.GENERIC_PDF,
        **kwargs
    ) -> ExtractorResult:
        """
        从URL下载并提取PDF文本内容
        
        Args:
            url: PDF的URL
            source_type: 数据源类型
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
        
        # 下载PDF
        pdf_bytes, error = self._download_pdf(url, source_type)
        if error:
            return ExtractorResult.failure(
                source_url=url,
                error_message=error,
                error_code="DOWNLOAD_ERROR",
                source_type=source_type
            )
        
        # 提取文本
        result = self.extract_from_bytes(
            pdf_bytes,
            source_url=url,
            source_type=source_type,
            **kwargs
        )
        
        # 更新处理时间
        result.process_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def extract_from_bytes(
        self,
        content: bytes,
        source_url: str = "",
        source_type: DataSourceType = DataSourceType.GENERIC_PDF,
        **kwargs
    ) -> ExtractorResult:
        """
        从字节流提取PDF文本内容
        
        Args:
            content: PDF字节内容
            source_url: 源URL（用于记录）
            source_type: 数据源类型
            **kwargs: 额外参数
        
        Returns:
            ExtractorResult: 提取结果
        """
        start_time = time.time()
        is_timeout_truncated = False
        actual_pages_processed = 0
        
        try:
            # 使用内存流打开PDF
            pdf_stream = io.BytesIO(content)
            doc = self.fitz.open(stream=pdf_stream, filetype="pdf")
            
            # 检查页数
            page_count = len(doc)
            max_pages = self.max_pages or page_count
            pages_to_process = min(page_count, max_pages)
            
            # 对于超大PDF（>100页），记录警告
            if page_count > 100:
                logger.warning(
                    f"Large PDF detected: {page_count} pages, "
                    f"processing up to {pages_to_process} pages with {self.extraction_timeout}s timeout"
                )
            
            # 提取文本
            text_parts = []
            for page_num in range(pages_to_process):
                # 检查是否超时
                elapsed = time.time() - start_time
                if elapsed > self.extraction_timeout:
                    # 超时：如果已处理的页数少于fallback页数，继续处理到fallback页数
                    if page_num < self.timeout_fallback_pages:
                        # 继续处理，直到达到fallback页数
                        pass
                    else:
                        # 超时且已超过fallback页数，停止处理
                        logger.warning(
                            f"PDF extraction timeout after {elapsed:.1f}s at page {page_num}/{pages_to_process}. "
                            f"Truncating to {page_num} pages."
                        )
                        is_timeout_truncated = True
                        break
                
                page = doc[page_num]
                
                # 提取页面文本
                page_text = page.get_text("text")
                
                # 如果启用表格提取，尝试更好地保留表格格式
                if self.extract_tables:
                    # 尝试使用blocks模式获取更好的布局
                    blocks = page.get_text("blocks")
                    if blocks:
                        # blocks按位置排序
                        blocks.sort(key=lambda b: (b[1], b[0]))  # 按y,x排序
                        block_texts = [b[4] for b in blocks if b[6] == 0]  # 只取文本块
                        if block_texts:
                            page_text = '\n'.join(block_texts)
                
                if page_text.strip():
                    text_parts.append(page_text)
                
                actual_pages_processed = page_num + 1
            
            doc.close()
            
            # 合并文本
            raw_text = '\n\n'.join(text_parts)
            
            # 如果超时截断，在文本开头添加标记
            if is_timeout_truncated:
                raw_text = f"[注：由于文档过大（{page_count}页），仅提取前{actual_pages_processed}页内容]\n\n" + raw_text
            
            # 清洗文本
            clean_text = self._clean_pdf_text(raw_text, source_type)
            
            if not clean_text.strip():
                return ExtractorResult.failure(
                    source_url=source_url,
                    error_message="No text content extracted from PDF",
                    error_code="EMPTY_CONTENT",
                    source_type=source_type
                )
            
            return ExtractorResult(
                success=True,
                content_text=clean_text,
                source_url=source_url,
                source_type=source_type,
                content_type=ContentType.PDF,
                page_count=page_count,
                process_time_ms=(time.time() - start_time) * 1000,
                extractor_version=self.VERSION,
            )
            
        except Exception as e:
            logger.error(f"PDF extraction failed for {source_url}: {e}")
            return ExtractorResult.failure(
                source_url=source_url,
                error_message=str(e),
                error_code="EXTRACT_ERROR",
                source_type=source_type
            )
    
    def _clean_pdf_text(
        self,
        text: str,
        source_type: DataSourceType = DataSourceType.GENERIC_PDF
    ) -> str:
        """
        清洗PDF提取的文本
        
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
        if source_type in (
            DataSourceType.CNINFO_ANNOUNCEMENT,
            DataSourceType.CNINFO_EVENT
        ):
            text = self._clean_cninfo_pdf(text)
        elif source_type == DataSourceType.EASTMONEY_REPORT:
            text = self._clean_eastmoney_pdf(text)
        
        return text
    
    def _clean_cninfo_pdf(self, text: str) -> str:
        """清洗巨潮资讯PDF"""
        # 移除证券代码和简称重复行
        text = re.sub(r'证券代码[：:]\s*\d+\s*证券简称[：:]\s*\S+\s*', '', text)
        
        # 移除公告编号
        text = re.sub(r'公告编号[：:]\s*\d+[-/]?\d*\s*', '', text)
        
        # 移除常见页眉
        text = re.sub(r'深圳证券交易所|上海证券交易所|北京证券交易所', '', text)
        
        # 移除常见页脚
        text = re.sub(r'本公司.*?信息披露.*?指定.*?媒体.*?\n?', '', text)
        text = re.sub(r'以上内容.*?投资者.*?参考.*?\n?', '', text)
        
        # 移除空白段落
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _clean_eastmoney_pdf(self, text: str) -> str:
        """清洗东方财富研报PDF"""
        # 移除研报页眉（券商名称等）
        text = re.sub(r'^.*?证券.*?研究.*?报告.*?\n', '', text, flags=re.MULTILINE)
        
        # 移除免责声明
        disclaimer_patterns = [
            r'免责声明[：:].*?(?=\n\n|\Z)',
            r'分析师声明[：:].*?(?=\n\n|\Z)',
            r'重要提示[：:].*?(?=\n\n|\Z)',
            r'投资评级说明[：:].*?(?=\n\n|\Z)',
        ]
        for pattern in disclaimer_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # 移除评级定义说明
        text = re.sub(r'买入.*?增持.*?中性.*?减持.*?\n', '', text)
        
        # 移除空白段落
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def extract_batch(
        self,
        urls: List[str],
        source_type: DataSourceType = DataSourceType.GENERIC_PDF,
        max_workers: int = 4,
        **kwargs
    ) -> List[ExtractorResult]:
        """
        批量提取PDF文本
        
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
                        source_type=source_type
                    ))
        
        return results
    
    def extract_batch_gpu(
        self,
        urls: List[str],
        source_type: DataSourceType = DataSourceType.GENERIC_PDF,
        max_workers: int = 4,
        **kwargs
    ) -> List[ExtractorResult]:
        """
        批量提取PDF文本（GPU加速文本清洗）
        
        先并行下载和提取文本，然后使用GPU批量清洗文本
        
        Args:
            urls: URL列表
            source_type: 数据源类型
            max_workers: 最大并发数
            **kwargs: 额外参数
        
        Returns:
            ExtractorResult列表
        """
        # 先进行普通批量提取（不做最终清洗）
        raw_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(
                    self._extract_raw, url, source_type, **kwargs
                ): url
                for url in urls
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    raw_results.append(result)
                except Exception as e:
                    logger.error(f"Batch extraction failed for {url}: {e}")
                    raw_results.append(ExtractorResult.failure(
                        source_url=url,
                        error_message=str(e),
                        error_code="BATCH_ERROR",
                        source_type=source_type
                    ))
        
        # GPU批量清洗文本
        texts_to_clean = []
        indices_to_clean = []
        for i, result in enumerate(raw_results):
            if result.success and result.content_text:
                texts_to_clean.append(result.content_text)
                indices_to_clean.append(i)
        
        if texts_to_clean:
            cleaned_texts = self.clean_text_batch_gpu(texts_to_clean)
            for idx, cleaned in zip(indices_to_clean, cleaned_texts):
                raw_results[idx].content_text = cleaned
        
        return raw_results
    
    def _extract_raw(
        self,
        url: str,
        source_type: DataSourceType = DataSourceType.GENERIC_PDF,
        **kwargs
    ) -> ExtractorResult:
        """提取PDF原始文本（不做最终清洗，用于批量GPU处理）"""
        start_time = time.time()
        
        # 下载PDF
        pdf_bytes, error = self._download_pdf(url, source_type)
        if error:
            return ExtractorResult.failure(
                source_url=url,
                error_message=error,
                error_code="DOWNLOAD_ERROR",
                source_type=source_type
            )
        
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            doc = self.fitz.open(stream=pdf_stream, filetype="pdf")
            
            page_count = len(doc)
            max_pages = self.max_pages or page_count
            pages_to_process = min(page_count, max_pages)
            
            text_parts = []
            for page_num in range(pages_to_process):
                page = doc[page_num]
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(page_text)
            
            doc.close()
            raw_text = '\n\n'.join(text_parts)
            
            if not raw_text.strip():
                return ExtractorResult.failure(
                    source_url=url,
                    error_message="No text content",
                    error_code="EMPTY_CONTENT",
                    source_type=source_type
                )
            
            return ExtractorResult(
                success=True,
                content_text=raw_text,
                source_url=url,
                source_type=source_type,
                content_type=ContentType.PDF,
                page_count=page_count,
                process_time_ms=(time.time() - start_time) * 1000,
                extractor_version=self.VERSION,
            )
            
        except Exception as e:
            return ExtractorResult.failure(
                source_url=url,
                error_message=str(e),
                error_code="EXTRACT_ERROR",
                source_type=source_type
            )
