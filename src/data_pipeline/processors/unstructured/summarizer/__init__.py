"""
摘要生成器模块 (Summarizer)

基于LLM的生成式摘要工具，将content_text转换为简洁的content摘要。

主要组件：
- Summarizer: 主接口类，提供单条/批量摘要生成
- LLMClient: Ollama客户端封装
- PromptTemplates: 针对不同数据类型的Prompt模板
- TextPreprocessor: 文本预处理器

使用示例：
```python
from src.data_pipeline.processors.unstructured.summarizer import Summarizer

summarizer = Summarizer()

# 单条摘要
result = summarizer.summarize(
    content_text="这里是公告原文...",
    data_type="announcement"
)
print(result.content)  # 50-100字摘要

# 批量摘要
results = summarizer.summarize_batch(
    texts=["文本1", "文本2"],
    data_types=["announcement", "report"]
)
```
"""

from .base import (
    SummaryResult,
    DataType,
    SummarizerConfig,
)
from .llm_client import LLMClient
from .prompts import PromptTemplates
from .text_preprocessor import TextPreprocessor
from .summarizer import Summarizer, BatchSummaryResult, summarize, summarize_batch

__all__ = [
    # 主接口
    'Summarizer',
    'BatchSummaryResult',
    
    # 数据结构
    'SummaryResult',
    'DataType',
    'SummarizerConfig',
    
    # 组件
    'LLMClient',
    'PromptTemplates',
    'TextPreprocessor',
    
    # 便捷函数
    'summarize',
    'summarize_batch',
]

__version__ = "1.0.0"
