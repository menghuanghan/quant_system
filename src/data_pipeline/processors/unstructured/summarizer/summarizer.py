"""
摘要生成器主模块

提供统一的API接口，支持：
- 单条文本摘要生成
- 批量摘要生成（并发/串行）
- DataFrame行级摘要
- 自动数据类型识别

这是summarizer模块的主入口
"""

import logging
import time
import asyncio
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .base import (
    SummaryResult,
    DataType,
    SummarizerConfig,
    BaseSummarizer,
    TextCleaningMixin,
)
from .llm_client import LLMClient, AsyncLLMClient, LLMResponse
from .prompts import PromptTemplates
from .text_preprocessor import TextPreprocessor, FinancialTextExtractor

logger = logging.getLogger(__name__)


@dataclass
class BatchSummaryResult:
    """批量摘要结果"""
    total: int
    success_count: int
    failed_count: int
    results: List[SummaryResult]
    elapsed_time_seconds: float
    
    # 统计信息
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_compression_ratio: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success_count / self.total if self.total > 0 else 0.0
    
    @property
    def throughput(self) -> float:
        """吞吐量（每秒处理数）"""
        return self.total / self.elapsed_time_seconds if self.elapsed_time_seconds > 0 else 0.0
    
    def get_failed_indices(self) -> List[int]:
        """获取失败的索引列表"""
        return [i for i, r in enumerate(self.results) if not r.success]
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        records = [r.to_dict() for r in self.results]
        return pd.DataFrame(records)


class Summarizer(BaseSummarizer, TextCleaningMixin):
    """
    摘要生成器
    
    基于LLM的生成式摘要工具，将content_text转换为50-100字的content摘要。
    
    特点：
    1. 针对金融文本优化的Prompt
    2. 自动文本预处理（清洗、分块）
    3. 支持批量处理和DataFrame操作
    4. GPU加速（Ollama自动管理）
    
    使用示例：
    ```python
    from src.data_pipeline.processors.unstructured.summarizer import Summarizer, DataType
    
    summarizer = Summarizer()
    
    # 单条摘要
    result = summarizer.summarize(
        content_text="这里是公告原文...",
        data_type=DataType.ANNOUNCEMENT
    )
    if result.success:
        print(result.content)  # 50-100字摘要
    
    # 批量摘要
    batch_result = summarizer.summarize_batch(
        texts=["文本1", "文本2"],
        data_types=[DataType.ANNOUNCEMENT, DataType.REPORT]
    )
    print(f"成功率: {batch_result.success_rate:.2%}")
    
    # DataFrame摘要
    df = pd.read_parquet("data.parquet")
    df_with_summary = summarizer.summarize_dataframe(
        df,
        text_column='content_text',
        output_column='content',
        type_column='data_type'
    )
    ```
    """
    
    VERSION = "1.0.0"
    
    # 数据域到DataType的映射
    DOMAIN_MAPPING = {
        'announcements': DataType.ANNOUNCEMENT,
        'reports': DataType.REPORT,
        'policy': DataType.POLICY,
        'policy/gov': DataType.POLICY,
        'policy/ndrc': DataType.POLICY_INDUSTRY,
        'news': DataType.NEWS,
        'news/cctv': DataType.NEWS_MARKET,
        'news/exchange': DataType.NEWS_MARKET,
        'events': DataType.EVENT,
    }
    
    def __init__(
        self,
        config: Optional[SummarizerConfig] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """
        初始化摘要生成器
        
        Args:
            config: 配置对象
            llm_client: LLM客户端（可选，用于注入自定义客户端）
        """
        super().__init__(config)
        
        self.llm_client = llm_client or LLMClient(self.config)
        self.preprocessor = TextPreprocessor(self.config)
        self.financial_extractor = FinancialTextExtractor()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"摘要生成器初始化完成 - 模型: {self.config.model_name}"
        )
    
    def summarize(
        self,
        content_text: str,
        data_type: Union[DataType, str] = DataType.GENERIC,
        title: Optional[str] = None,
        preprocess: bool = True,
        **kwargs
    ) -> SummaryResult:
        """
        生成摘要
        
        Args:
            content_text: 输入的原始文本（来自content_extractor）
            data_type: 数据类型，用于选择Prompt
            title: 标题（可用于辅助理解内容）
            preprocess: 是否进行预处理
            **kwargs: 额外参数
            
        Returns:
            SummaryResult: 摘要结果
        """
        start_time = time.time()
        
        # 转换数据类型
        if isinstance(data_type, str):
            data_type = self._parse_data_type(data_type)
        
        # 输入验证
        if not content_text or not content_text.strip():
            return SummaryResult.failure(
                original_text="",
                error_message="输入文本为空",
                error_code="EMPTY_INPUT",
                data_type=data_type
            )
        
        try:
            # 预处理
            if preprocess:
                processed_text = self._preprocess_text(content_text, data_type)
            else:
                processed_text = content_text
            
            if not processed_text:
                return SummaryResult.failure(
                    original_text=content_text,
                    error_message="预处理后文本为空",
                    error_code="EMPTY_AFTER_PREPROCESS",
                    data_type=data_type
                )
            
            # 构建Prompt
            messages = PromptTemplates.build_messages(
                content=processed_text,
                data_type=data_type,
                title=title
            )
            
            # 调用LLM
            response = self.llm_client.chat(messages)
            
            # 后处理摘要
            summary = self._postprocess_summary(response.content)
            
            # 构建结果
            elapsed_ms = (time.time() - start_time) * 1000
            
            return SummaryResult(
                success=True,
                content=summary,
                original_text=content_text,
                data_type=data_type,
                prompt_used=messages[1]['content'][:200] + "...",
                model_name=response.model,
                process_time_ms=elapsed_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            
        except Exception as e:
            self.logger.error(f"摘要生成失败: {e}")
            return SummaryResult.failure(
                original_text=content_text,
                error_message=str(e),
                error_code="SUMMARIZE_ERROR",
                data_type=data_type
            )
    
    def summarize_batch(
        self,
        texts: List[str],
        data_types: Optional[List[Union[DataType, str]]] = None,
        titles: Optional[List[str]] = None,
        max_workers: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> BatchSummaryResult:
        """
        批量生成摘要
        
        Args:
            texts: 文本列表
            data_types: 数据类型列表，长度需与texts一致
            titles: 标题列表（可选）
            max_workers: 最大并发数（建议1，因为Ollama串行更高效）
            progress_callback: 进度回调函数 (completed, total)
            **kwargs: 额外参数
            
        Returns:
            BatchSummaryResult: 批量结果
        """
        start_time = time.time()
        total = len(texts)
        
        if total == 0:
            return BatchSummaryResult(
                total=0,
                success_count=0,
                failed_count=0,
                results=[],
                elapsed_time_seconds=0.0
            )
        
        # 准备数据类型列表
        if data_types is None:
            data_types = [DataType.GENERIC] * total
        elif len(data_types) != total:
            data_types = data_types + [DataType.GENERIC] * (total - len(data_types))
        
        # 准备标题列表
        if titles is None:
            titles = [None] * total
        elif len(titles) != total:
            titles = titles + [None] * (total - len(titles))
        
        results = []
        success_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        # 串行处理（Ollama对串行处理更友好）
        if max_workers <= 1:
            for i, (text, dtype, title) in enumerate(zip(texts, data_types, titles)):
                result = self.summarize(text, dtype, title, **kwargs)
                results.append(result)
                
                if result.success:
                    success_count += 1
                    total_input_tokens += result.input_tokens or 0
                    total_output_tokens += result.output_tokens or 0
                
                if progress_callback:
                    progress_callback(i + 1, total)
        else:
            # 并发处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.summarize, text, dtype, title, **kwargs
                    ): i
                    for i, (text, dtype, title) in enumerate(zip(texts, data_types, titles))
                }
                
                # 预分配结果列表
                results = [None] * total
                completed = 0
                
                for future in as_completed(futures):
                    idx = futures[future]
                    result = future.result()
                    results[idx] = result
                    
                    if result.success:
                        success_count += 1
                        total_input_tokens += result.input_tokens or 0
                        total_output_tokens += result.output_tokens or 0
                    
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
        
        elapsed = time.time() - start_time
        
        # 计算平均压缩比
        compression_ratios = [
            r.compression_ratio for r in results
            if r.success and r.compression_ratio
        ]
        avg_ratio = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0.0
        
        return BatchSummaryResult(
            total=total,
            success_count=success_count,
            failed_count=total - success_count,
            results=results,
            elapsed_time_seconds=elapsed,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            avg_compression_ratio=avg_ratio,
        )
    
    def summarize_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'content_text',
        output_column: str = 'content',
        type_column: Optional[str] = None,
        title_column: Optional[str] = None,
        domain_column: Optional[str] = None,
        default_type: DataType = DataType.GENERIC,
        max_workers: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        对DataFrame中的文本列生成摘要
        
        Args:
            df: 输入DataFrame
            text_column: 文本列名
            output_column: 输出列名
            type_column: 数据类型列名（可选）
            title_column: 标题列名（可选）
            domain_column: 数据域列名（用于推断数据类型）
            default_type: 默认数据类型
            max_workers: 并发数
            progress_callback: 进度回调
            **kwargs: 额外参数
            
        Returns:
            pd.DataFrame: 添加了摘要列的DataFrame
        """
        df = df.copy()
        
        # 提取文本
        texts = df[text_column].fillna('').tolist()
        
        # 确定数据类型
        if type_column and type_column in df.columns:
            data_types = [
                self._parse_data_type(t) if pd.notna(t) else default_type
                for t in df[type_column]
            ]
        elif domain_column and domain_column in df.columns:
            data_types = [
                self.DOMAIN_MAPPING.get(d, default_type) if pd.notna(d) else default_type
                for d in df[domain_column]
            ]
        else:
            data_types = [default_type] * len(texts)
        
        # 提取标题
        titles = None
        if title_column and title_column in df.columns:
            titles = df[title_column].tolist()
        
        # 批量处理
        batch_result = self.summarize_batch(
            texts=texts,
            data_types=data_types,
            titles=titles,
            max_workers=max_workers,
            progress_callback=progress_callback,
            **kwargs
        )
        
        # 添加结果列
        df[output_column] = [r.content if r.success else None for r in batch_result.results]
        df[f'{output_column}_success'] = [r.success for r in batch_result.results]
        df[f'{output_column}_error'] = [r.error_message for r in batch_result.results]
        
        self.logger.info(
            f"DataFrame摘要完成 - 总数: {batch_result.total}, "
            f"成功: {batch_result.success_count}, "
            f"成功率: {batch_result.success_rate:.2%}"
        )
        
        return df
    
    def _preprocess_text(self, text: str, data_type: DataType) -> str:
        """预处理文本"""
        # 清洗
        cleaned = self.preprocessor.clean(text)
        
        if not cleaned:
            return ""
        
        # 如果太长，尝试提取关键部分
        if len(cleaned) > self.config.max_input_chars:
            # 对于金融文本，尝试提取关键章节
            if data_type in [DataType.ANNOUNCEMENT, DataType.REPORT]:
                key_sections = self.financial_extractor.extract_key_sections(cleaned)
                if len(key_sections) <= self.config.max_input_chars:
                    return key_sections
            
            # 否则使用智能截断
            return self.preprocessor.truncate_smart(
                cleaned,
                self.config.max_input_chars
            )
        
        return cleaned
    
    def _postprocess_summary(self, summary: str) -> str:
        """后处理摘要"""
        if not summary:
            return ""
        
        # 移除可能的前缀
        prefixes_to_remove = [
            "摘要：", "摘要:", "总结：", "总结:", 
            "核心内容：", "核心内容:"
        ]
        for prefix in prefixes_to_remove:
            if summary.startswith(prefix):
                summary = summary[len(prefix):]
        
        # 移除首尾空白
        summary = summary.strip()
        
        # 如果超过最大长度，截断
        if len(summary) > self.config.max_summary_chars:
            # 尝试在句子边界截断
            import re
            sentences = re.split(r'[。！？]', summary)
            truncated = ""
            for sent in sentences:
                if len(truncated) + len(sent) + 1 <= self.config.max_summary_chars:
                    truncated += sent + "。"
                else:
                    break
            summary = truncated.strip() if truncated else summary[:self.config.max_summary_chars]
        
        return summary
    
    def _parse_data_type(self, type_str: str) -> DataType:
        """解析数据类型字符串"""
        if isinstance(type_str, DataType):
            return type_str
        
        type_str = type_str.lower().strip()
        
        # 尝试直接匹配
        try:
            return DataType(type_str)
        except ValueError:
            pass
        
        # 尝试从域名映射
        if type_str in self.DOMAIN_MAPPING:
            return self.DOMAIN_MAPPING[type_str]
        
        # 关键词匹配
        if '公告' in type_str or 'announcement' in type_str:
            return DataType.ANNOUNCEMENT
        if '研报' in type_str or 'report' in type_str:
            return DataType.REPORT
        if '政策' in type_str or 'policy' in type_str:
            return DataType.POLICY
        if '新闻' in type_str or 'news' in type_str:
            return DataType.NEWS
        if '事件' in type_str or 'event' in type_str:
            return DataType.EVENT
        
        return DataType.GENERIC
    
    def warm_up(self) -> float:
        """
        预热模型
        
        Returns:
            float: 预热耗时（毫秒）
        """
        return self.llm_client.warm_up()
    
    def is_available(self) -> bool:
        """检查服务是否可用"""
        return self.llm_client.is_available()
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        return self.llm_client.get_model_info()


# ==================== 便捷函数 ====================


def summarize(
    content_text: str,
    data_type: Union[DataType, str] = DataType.GENERIC,
    **kwargs
) -> SummaryResult:
    """
    便捷函数：生成摘要
    
    使用默认配置创建Summarizer并生成摘要。
    
    Args:
        content_text: 原始文本
        data_type: 数据类型
        **kwargs: 额外参数
        
    Returns:
        SummaryResult: 摘要结果
    """
    summarizer = Summarizer()
    return summarizer.summarize(content_text, data_type, **kwargs)


def summarize_batch(
    texts: List[str],
    data_types: Optional[List[Union[DataType, str]]] = None,
    **kwargs
) -> BatchSummaryResult:
    """
    便捷函数：批量生成摘要
    """
    summarizer = Summarizer()
    return summarizer.summarize_batch(texts, data_types, **kwargs)


# 异步版本
class AsyncSummarizer:
    """
    异步摘要生成器
    
    用于需要高并发处理的场景。
    
    使用示例：
    ```python
    import asyncio
    
    async def main():
        summarizer = AsyncSummarizer()
        
        # 并发生成多个摘要
        tasks = [
            summarizer.summarize("文本1", DataType.ANNOUNCEMENT),
            summarizer.summarize("文本2", DataType.REPORT),
        ]
        results = await asyncio.gather(*tasks)
    
    asyncio.run(main())
    ```
    """
    
    def __init__(self, config: Optional[SummarizerConfig] = None):
        self.config = config or SummarizerConfig()
        self.llm_client = AsyncLLMClient(self.config)
        self.preprocessor = TextPreprocessor(self.config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def summarize(
        self,
        content_text: str,
        data_type: DataType = DataType.GENERIC,
        **kwargs
    ) -> SummaryResult:
        """异步生成摘要"""
        start_time = time.time()
        
        if not content_text:
            return SummaryResult.failure(
                original_text="",
                error_message="输入文本为空",
                error_code="EMPTY_INPUT",
                data_type=data_type
            )
        
        try:
            # 预处理
            cleaned = self.preprocessor.clean(content_text)
            if len(cleaned) > self.config.max_input_chars:
                cleaned = self.preprocessor.truncate_smart(
                    cleaned, self.config.max_input_chars
                )
            
            # 构建Prompt
            messages = PromptTemplates.build_messages(
                content=cleaned,
                data_type=data_type
            )
            
            # 异步调用LLM
            response = await self.llm_client.generate(
                messages[1]['content'],
                system=messages[0]['content']
            )
            
            # 后处理
            summary = response.content.strip()
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return SummaryResult(
                success=True,
                content=summary,
                original_text=content_text,
                data_type=data_type,
                model_name=response.model,
                process_time_ms=elapsed_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            
        except Exception as e:
            self.logger.error(f"异步摘要生成失败: {e}")
            return SummaryResult.failure(
                original_text=content_text,
                error_message=str(e),
                error_code="ASYNC_SUMMARIZE_ERROR",
                data_type=data_type
            )
    
    async def summarize_batch(
        self,
        texts: List[str],
        data_types: Optional[List[DataType]] = None,
        max_concurrent: int = 4,
        **kwargs
    ) -> BatchSummaryResult:
        """异步批量生成摘要"""
        import asyncio
        from asyncio import Semaphore
        
        start_time = time.time()
        total = len(texts)
        
        if total == 0:
            return BatchSummaryResult(
                total=0, success_count=0, failed_count=0,
                results=[], elapsed_time_seconds=0.0
            )
        
        if data_types is None:
            data_types = [DataType.GENERIC] * total
        
        # 使用信号量限制并发
        semaphore = Semaphore(max_concurrent)
        
        async def bounded_summarize(text, dtype):
            async with semaphore:
                return await self.summarize(text, dtype, **kwargs)
        
        # 并发执行
        tasks = [
            bounded_summarize(text, dtype)
            for text, dtype in zip(texts, data_types)
        ]
        results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        
        return BatchSummaryResult(
            total=total,
            success_count=success_count,
            failed_count=total - success_count,
            results=results,
            elapsed_time_seconds=elapsed,
        )
