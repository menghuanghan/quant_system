"""
非结构化数据全量调度器模块

提供原始数据采集调度器
"""

from .collection import (
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
]
