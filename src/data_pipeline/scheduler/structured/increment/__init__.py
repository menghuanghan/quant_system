"""结构化增量采集调度器模块"""

from .config import (
    CORE_INDEX_CODES,
    DOMAIN_NAMES,
    ETF_OPTIONAL_TASK,
    OPTIONAL_STATIC_TASKS,
    TIME_DEPENDENT_TASKS,
    IncrementCollectionTask,
    get_increment_tasks,
)
from .scheduler import (
    IncrementCollectionReport,
    IncrementTaskResult,
    StaticRefreshItem,
    StaticRefreshReport,
    StructuredIncrementScheduler,
)

__all__ = [
    "CORE_INDEX_CODES",
    "DOMAIN_NAMES",
    "ETF_OPTIONAL_TASK",
    "OPTIONAL_STATIC_TASKS",
    "TIME_DEPENDENT_TASKS",
    "IncrementCollectionTask",
    "get_increment_tasks",
    "IncrementCollectionReport",
    "IncrementTaskResult",
    "StaticRefreshItem",
    "StaticRefreshReport",
    "StructuredIncrementScheduler",
]
