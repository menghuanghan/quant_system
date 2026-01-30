"""
统一内容提取器接口

提供面向用户的高级API，支持：
- 单个URL提取
- 批量URL提取
- DataFrame行级提取
- GPU加速批量处理

这是内容提取器模块的主入口
"""

import logging
import time
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .base import (
    ExtractorResult,
    DataSourceType,
    ContentType,
    TextCleaningMixin,
)
from .factory import ContentExtractorFactory

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """批量提取结果"""
    total: int
    success_count: int
    failed_count: int
    results: List[ExtractorResult]
    elapsed_time_seconds: float
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success_count / self.total if self.total > 0 else 0.0
    
    def get_failed_urls(self) -> List[str]:
        """获取失败的URL列表"""
        return [r.source_url for r in self.results if not r.success]
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        records = [r.to_dict() for r in self.results]
        return pd.DataFrame(records)


class ContentExtractor(TextCleaningMixin):
    """
    统一内容提取器
    
    这是内容提取器模块的主接口，提供：
    1. 简单的单URL提取
    2. 批量URL提取（支持并发和GPU加速）
    3. DataFrame行级提取
    4. 自动数据源识别
    
    使用示例：
    ```python
    from src.data_pipeline.processors.unstructured.content_extractor import ContentExtractor
    
    extractor = ContentExtractor()
    
    # 单个URL提取
    result = extractor.extract("http://static.cninfo.com.cn/xxx.PDF")
    if result.success:
        print(result.content_text)
    
    # 批量提取
    urls = ["url1", "url2", "url3"]
    batch_result = extractor.extract_batch(urls)
    print(f"成功率: {batch_result.success_rate:.2%}")
    
    # DataFrame提取
    df = pd.read_parquet("announcements.parquet")
    df_with_content = extractor.extract_from_dataframe(
        df, 
        url_column='url',
        output_column='content_text'
    )
    ```
    """
    
    VERSION = "1.0.0"
    
    # 数据域 -> 数据源类型映射
    DOMAIN_MAPPING = {
        'announcements': DataSourceType.CNINFO_ANNOUNCEMENT,
        'events': DataSourceType.CNINFO_EVENT,
        'reports': DataSourceType.EASTMONEY_REPORT,
        'policy/gov': DataSourceType.GOV_POLICY,
        'policy/ndrc': DataSourceType.NDRC_POLICY,
        'news/cctv': DataSourceType.CCTV_NEWS,
        'news/exchange': DataSourceType.EXCHANGE_NEWS,
    }
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
        max_workers: int = 4,
        use_gpu: bool = True
    ):
        """
        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            user_agent: 自定义User-Agent
            max_workers: 批量处理时的最大并发数
            use_gpu: 是否使用GPU加速（用于文本清洗）
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.max_workers = max_workers
        self.use_gpu = use_gpu
        
        # 创建工厂
        self._factory = ContentExtractorFactory(
            timeout=timeout,
            max_retries=max_retries,
            user_agent=user_agent,
            cache_extractors=True
        )
        
        # 检查GPU可用性
        self._gpu_available = self._check_gpu_availability() if use_gpu else False
    
    def _check_gpu_availability(self) -> bool:
        """检查GPU是否可用"""
        try:
            import cudf
            # 尝试创建一个小的cuDF Series
            s = cudf.Series(["test"])
            s.str.lower()
            return True
        except Exception as e:
            logger.warning(f"GPU (cuDF) not available: {e}")
            return False
    
    def extract(
        self,
        url: str,
        source_type: Optional[DataSourceType] = None,
        domain: Optional[str] = None,
        title: Optional[str] = None,
        existing_content: Optional[str] = None,
        **kwargs
    ) -> ExtractorResult:
        """
        从URL提取文本内容
        
        Args:
            url: 源URL
            source_type: 数据源类型（可选，自动检测）
            domain: 数据域名称（如'announcements', 'reports'等，用于类型推断）
            title: 标题（用于EXCHANGE_NEWS等类型）
            existing_content: 已有内容（用于CCTV_NEWS等类型）
            **kwargs: 传递给具体提取器的参数
        
        Returns:
            ExtractorResult: 提取结果
        """
        # 如果提供了domain，尝试推断source_type
        if source_type is None and domain:
            source_type = self.DOMAIN_MAPPING.get(domain)
        
        return self._factory.extract(
            url=url,
            source_type=source_type,
            title=title,
            existing_content=existing_content,
            **kwargs
        )
    
    def extract_batch(
        self,
        urls: List[str],
        source_type: Optional[DataSourceType] = None,
        domain: Optional[str] = None,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> BatchResult:
        """
        批量提取URL内容
        
        Args:
            urls: URL列表
            source_type: 数据源类型（可选，自动检测）
            domain: 数据域名称
            max_workers: 最大并发数（None则使用默认值）
            progress_callback: 进度回调函数 (completed, total)
            **kwargs: 传递给具体提取器的参数
        
        Returns:
            BatchResult: 批量提取结果
        """
        start_time = time.time()
        workers = max_workers or self.max_workers
        total = len(urls)
        
        # 推断source_type
        if source_type is None and domain:
            source_type = self.DOMAIN_MAPPING.get(domain)
        
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_url = {
                executor.submit(
                    self.extract, url, source_type, domain, **kwargs
                ): url
                for url in urls
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Extraction failed for {url}: {e}")
                    results.append(ExtractorResult.failure(
                        source_url=url,
                        error_message=str(e),
                        error_code="BATCH_ERROR",
                        source_type=source_type or DataSourceType.UNKNOWN
                    ))
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        
        # 如果启用GPU且可用，进行批量文本清洗
        if self.use_gpu and self._gpu_available:
            results = self._gpu_post_process(results)
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        
        return BatchResult(
            total=total,
            success_count=success_count,
            failed_count=total - success_count,
            results=results,
            elapsed_time_seconds=elapsed
        )
    
    def _gpu_post_process(
        self,
        results: List[ExtractorResult]
    ) -> List[ExtractorResult]:
        """使用GPU进行批量后处理"""
        # 收集需要处理的文本
        texts_to_clean = []
        indices = []
        
        for i, result in enumerate(results):
            if result.success and result.content_text:
                texts_to_clean.append(result.content_text)
                indices.append(i)
        
        if not texts_to_clean:
            return results
        
        # GPU批量清洗
        try:
            cleaned_texts = self.clean_text_batch_gpu(texts_to_clean)
            for idx, cleaned in zip(indices, cleaned_texts):
                results[idx].content_text = cleaned
        except Exception as e:
            logger.warning(f"GPU post-processing failed: {e}")
        
        return results
    
    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        url_column: str = 'url',
        output_column: str = 'content_text',
        source_type_column: Optional[str] = None,
        title_column: Optional[str] = None,
        content_column: Optional[str] = None,
        domain: Optional[str] = None,
        max_workers: Optional[int] = None,
        inplace: bool = False,
        add_metadata: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        从DataFrame的URL列提取文本内容
        
        Args:
            df: 输入DataFrame
            url_column: URL列名
            output_column: 输出内容列名
            source_type_column: 数据源类型列名（可选）
            title_column: 标题列名（用于EXCHANGE_NEWS）
            content_column: 已有内容列名（用于CCTV_NEWS）
            domain: 数据域名称
            max_workers: 最大并发数
            inplace: 是否原地修改
            add_metadata: 是否添加元数据列（成功状态、错误信息等）
            progress_callback: 进度回调函数
            **kwargs: 传递给提取器的参数
        
        Returns:
            包含提取内容的DataFrame
        """
        if not inplace:
            df = df.copy()
        
        # 检查必需列
        if url_column not in df.columns:
            raise ValueError(f"URL column '{url_column}' not found in DataFrame")
        
        # 准备提取
        urls = df[url_column].tolist()
        titles = df[title_column].tolist() if title_column and title_column in df.columns else [None] * len(urls)
        contents = df[content_column].tolist() if content_column and content_column in df.columns else [None] * len(urls)
        
        # 确定source_type
        source_type = None
        if domain:
            source_type = self.DOMAIN_MAPPING.get(domain)
        
        # 批量提取
        results = []
        workers = max_workers or self.max_workers
        total = len(urls)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {}
            for i, (url, title, content) in enumerate(zip(urls, titles, contents)):
                # 跳过空URL
                if pd.isna(url) or not url:
                    results.append((i, ExtractorResult.failure(
                        source_url="",
                        error_message="Empty URL",
                        error_code="EMPTY_URL",
                        source_type=source_type or DataSourceType.UNKNOWN
                    )))
                    continue
                
                future = executor.submit(
                    self.extract,
                    url,
                    source_type=source_type,
                    domain=domain,
                    title=title,
                    existing_content=content,
                    **kwargs
                )
                future_to_idx[future] = i
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    results.append((idx, ExtractorResult.failure(
                        source_url=urls[idx],
                        error_message=str(e),
                        error_code="EXTRACT_ERROR",
                        source_type=source_type or DataSourceType.UNKNOWN
                    )))
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        
        # 按索引排序
        results.sort(key=lambda x: x[0])
        
        # 填充结果
        df[output_column] = [r.content_text for _, r in results]
        
        if add_metadata:
            df[f'{output_column}_success'] = [r.success for _, r in results]
            df[f'{output_column}_error'] = [r.error_message for _, r in results]
            df[f'{output_column}_char_count'] = [r.char_count for _, r in results]
            df[f'{output_column}_process_time_ms'] = [r.process_time_ms for _, r in results]
        
        return df
    
    def detect_source_type(self, url: str) -> DataSourceType:
        """
        检测URL对应的数据源类型
        
        Args:
            url: 源URL
        
        Returns:
            DataSourceType: 数据源类型
        """
        return self._factory.detect_source_type(url)
    
    def get_supported_domains(self) -> List[str]:
        """获取支持的数据域列表"""
        return list(self.DOMAIN_MAPPING.keys())
    
    def get_statistics(self, batch_result: BatchResult) -> Dict[str, Any]:
        """
        获取批量提取的统计信息
        
        Args:
            batch_result: 批量提取结果
        
        Returns:
            统计信息字典
        """
        success_results = [r for r in batch_result.results if r.success]
        
        total_chars = sum(r.char_count or 0 for r in success_results)
        total_time = sum(r.process_time_ms or 0 for r in success_results)
        
        # 按错误类型分组
        error_counts = {}
        for r in batch_result.results:
            if not r.success:
                error_code = r.error_code or "UNKNOWN"
                error_counts[error_code] = error_counts.get(error_code, 0) + 1
        
        return {
            'total': batch_result.total,
            'success_count': batch_result.success_count,
            'failed_count': batch_result.failed_count,
            'success_rate': batch_result.success_rate,
            'elapsed_time_seconds': batch_result.elapsed_time_seconds,
            'total_characters': total_chars,
            'avg_characters': total_chars / len(success_results) if success_results else 0,
            'total_process_time_ms': total_time,
            'avg_process_time_ms': total_time / len(success_results) if success_results else 0,
            'error_counts': error_counts,
            'throughput_per_second': batch_result.total / batch_result.elapsed_time_seconds if batch_result.elapsed_time_seconds > 0 else 0,
        }


# 便捷函数
def extract_content(
    url: str,
    source_type: Optional[DataSourceType] = None,
    **kwargs
) -> ExtractorResult:
    """
    便捷函数：从URL提取内容
    
    Args:
        url: 源URL
        source_type: 数据源类型
        **kwargs: 其他参数
    
    Returns:
        ExtractorResult: 提取结果
    """
    extractor = ContentExtractor()
    return extractor.extract(url, source_type=source_type, **kwargs)


def extract_content_batch(
    urls: List[str],
    source_type: Optional[DataSourceType] = None,
    max_workers: int = 4,
    **kwargs
) -> BatchResult:
    """
    便捷函数：批量提取内容
    
    Args:
        urls: URL列表
        source_type: 数据源类型
        max_workers: 最大并发数
        **kwargs: 其他参数
    
    Returns:
        BatchResult: 批量结果
    """
    extractor = ContentExtractor(max_workers=max_workers)
    return extractor.extract_batch(urls, source_type=source_type, **kwargs)
