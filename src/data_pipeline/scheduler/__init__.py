"""
调度器模块

包含结构化和非结构化数据的调度能力：
- structured/: 结构化数据调度（K线、财报等）
- unstructured/: 非结构化数据调度（新闻、公告、研报等）

非结构化数据调度器快速使用：
    >>> from src.data_pipeline.scheduler.unstructured import (
    ...     UnstructuredHistoryScheduler,
    ...     run_backfill
    ... )
    >>> run_backfill(2021, 2025)
"""

# 导出非结构化调度器的主要接口
from .unstructured import (
    UnstructuredHistoryScheduler,
    run_backfill,
    get_backfill_progress,
    reset_failed_tasks,
    CheckpointManager,
    get_checkpoint_manager
)