"""
非结构化数据调度器模块

提供非结构化数据的采集调度：
- full: 全量采集调度器
- increment: 增量采集调度器
"""

from .full import (
    UnstructuredFullCollectionScheduler,
    TaskStatus,
    TaskResult,
    CollectionProgress,
    CheckpointData,
    CollectionTask,
    DataType,
    StockScope,
    StoragePattern,
    ALL_TASKS,
    TASKS_BY_TYPE,
    TYPE_NAMES,
    get_enabled_tasks,
    get_tasks_by_type,
    get_tasks_sorted_by_priority,
    list_all_tasks,
    get_task_count,
)
from .increment import (
    UnstructuredIncrementTask,
    UNSTRUCTURED_INCREMENT_TASKS,
    TASK_NAME_MAP,
    get_increment_tasks,
    IncrementTaskResult,
    IncrementCollectionReport,
    UnstructuredIncrementScheduler,
)


__all__ = [
    # 调度器
    "UnstructuredFullCollectionScheduler",
    "TaskStatus",
    "TaskResult",
    "CollectionProgress",
    "CheckpointData",
    # 配置
    "CollectionTask",
    "DataType",
    "StockScope",
    "StoragePattern",
    "ALL_TASKS",
    "TASKS_BY_TYPE",
    "TYPE_NAMES",
    # 函数
    "get_enabled_tasks",
    "get_tasks_by_type",
    "get_tasks_sorted_by_priority",
    "list_all_tasks",
    "get_task_count",

    # 增量调度器
    "UnstructuredIncrementTask",
    "UNSTRUCTURED_INCREMENT_TASKS",
    "TASK_NAME_MAP",
    "get_increment_tasks",
    "IncrementTaskResult",
    "IncrementCollectionReport",
    "UnstructuredIncrementScheduler",
]
