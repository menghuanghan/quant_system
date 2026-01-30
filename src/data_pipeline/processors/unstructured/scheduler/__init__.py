"""
非结构化数据处理调度器

负责调度 content_extractor、summarizer、scorer 三个工具，
完成对 data/raw/unstructured 下原始数据的处理。

支持的数据类型：
- announcements: 公告 (PDF -> content -> summary -> score)
- reports: 研报 (PDF -> content -> summary -> score)
- events: 事件 (PDF -> content -> summary -> score)
- exchange: 交易所公告 (title -> regex -> score)
- news/cctv: CCTV新闻 (content -> keywords -> sentiment -> beta_signal)
- policy/gov: 国务院政策 (HTML -> sector_mapping -> industry_scores)
- policy/ndrc: 发改委政策 (HTML -> sector_mapping -> industry_scores)

输出：
- data/processed/unstructured/{category}/{year}/{month}.parquet
- 公告/研报/事件字段：id, ts_code, date, score, reason
- 交易所公告字段：id, ts_code, date, score
- CCTV新闻字段：date, id, market_sentiment, beta_signal, keywords, tone_analysis
- 政策数据字段：date, id, summary, benefited_industries, harmed_industries, industry_scores
"""

from .base import (
    ProcessingConfig,
    ProcessingResult,
    BatchProcessingResult,
    DataCategory,
    ProcessingStatus,
    CCTV_OUTPUT_COLUMNS,
    POLICY_OUTPUT_COLUMNS,
)
from .pipeline import (
    PDFPipeline,
    ExchangePipeline,
    CCTVPipeline,
    PolicyPipeline,
    CCTVProcessingResult,
    PolicyProcessingResult,
    BasePipeline,
)
from .scheduler import (
    UnstructuredScheduler,
    process_month,
    process_year,
    process_all,
)

__all__ = [
    # 配置和数据结构
    'ProcessingConfig',
    'ProcessingResult',
    'BatchProcessingResult',
    'DataCategory',
    'ProcessingStatus',
    'CCTV_OUTPUT_COLUMNS',
    'POLICY_OUTPUT_COLUMNS',
    # 流水线
    'PDFPipeline',
    'ExchangePipeline',
    'CCTVPipeline',
    'PolicyPipeline',
    'CCTVProcessingResult',
    'PolicyProcessingResult',
    'BasePipeline',
    # 调度器
    'UnstructuredScheduler',
    # 便捷函数
    'process_month',
    'process_year',
    'process_all',
]
