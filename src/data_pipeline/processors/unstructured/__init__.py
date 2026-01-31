"""
非结构化数据处理模块

提供非结构化数据的处理工具：
- content_extractor: 源文本提取器（URL -> content_text）
- summarizer: 摘要生成器（content_text -> content）
- scorer: 量化打分器（content -> score）
- scheduler: 调度器（协调各工具完成端到端处理）
"""

from .content_extractor import (
    ContentExtractor,
    ContentExtractorFactory,
    PDFExtractor,
    HTMLExtractor,
    CninfoDetailParser,
    ExtractorResult,
    BatchResult,
    DataSourceType,
    ContentType,
    extract_content,
    extract_content_batch,
)

from .summarizer import (
    Summarizer,
    BatchSummaryResult,
    SummaryResult,
    DataType,
    SummarizerConfig,
    LLMClient,
    PromptTemplates,
    TextPreprocessor,
    summarize,
    summarize_batch,
)

from .scorer import (
    Scorer,
    BatchScoreResult,
    ScoreResult,
    ScorerConfig,
    ScoreLevel,
    ScoringMethod,
    RuleScorer,
    LLMScorer,
    score,
    score_batch,
)

from .scheduler import (
    UnstructuredScheduler,
    ProcessingConfig,
    ProcessingResult,
    BatchProcessingResult,
    DataCategory,
    ProcessingStatus,
    PDFPipeline,
    ExchangePipeline,
    process_month,
    process_year,
    process_all,
)

from .filter import (
    AnnouncementFilter,
    FilterConfig,
    FilterResult,
    TITLE_BLACKLIST_KEYWORDS,
    filter_month as filter_announcements_month,
    filter_year as filter_announcements_year,
    filter_all as filter_announcements_all,
)

__all__ = [
    # ===== Content Extractor =====
    # 主接口
    'ContentExtractor',
    'ContentExtractorFactory',
    # 具体提取器
    'PDFExtractor',
    'HTMLExtractor', 
    'CninfoDetailParser',
    # 数据结构
    'ExtractorResult',
    'BatchResult',
    # 枚举类型
    'DataSourceType',
    'ContentType',
    # 便捷函数
    'extract_content',
    'extract_content_batch',
    
    # ===== Summarizer =====
    # 主接口
    'Summarizer',
    # 数据结构
    'BatchSummaryResult',
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
    
    # ===== Scorer =====
    # 主接口
    'Scorer',
    # 具体打分器
    'RuleScorer',
    'LLMScorer',
    # 数据结构
    'BatchScoreResult',
    'ScoreResult',
    'ScorerConfig',
    'ScoreLevel',
    'ScoringMethod',
    # 便捷函数
    'score',
    'score_batch',
    
    # ===== Scheduler =====
    # 主接口
    'UnstructuredScheduler',
    # 流水线
    'PDFPipeline',
    'ExchangePipeline',
    # 数据结构
    'ProcessingConfig',
    'ProcessingResult',
    'BatchProcessingResult',
    'DataCategory',
    'ProcessingStatus',
    # 便捷函数
    'process_month',
    'process_year',
    'process_all',
    
    # ===== Filter =====
    # 主接口
    'AnnouncementFilter',
    # 数据结构
    'FilterConfig',
    'FilterResult',
    'TITLE_BLACKLIST_KEYWORDS',
    # 便捷函数
    'filter_announcements_month',
    'filter_announcements_year',
    'filter_announcements_all',
]
