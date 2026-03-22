"""
非结构化数据增量采集调度模块

负责按交易日（或自然日）采集公告、事件、新闻、政策、研报等非结构化增量数据，
并将结果落盘到 data/raw/unstructured_increment/date=YYYY-MM-DD/。
"""

from .config import (
    UnstructuredIncrementTask,
    UNSTRUCTURED_INCREMENT_TASKS,
    TASK_NAME_MAP,
    get_increment_tasks,
)
from .scheduler import (
    IncrementTaskResult,
    IncrementCollectionReport,
    UnstructuredIncrementScheduler,
)


__all__ = [
    "UnstructuredIncrementTask",
    "UNSTRUCTURED_INCREMENT_TASKS",
    "TASK_NAME_MAP",
    "get_increment_tasks",
    "IncrementTaskResult",
    "IncrementCollectionReport",
    "UnstructuredIncrementScheduler",
]
