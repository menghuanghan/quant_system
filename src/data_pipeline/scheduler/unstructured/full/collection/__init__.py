"""
非结构化数据全量采集模块

提供非结构化原始数据采集调度器，统一调度：
- announcements: 上市公司公告
- events: 事件驱动型数据
- news: 新闻
- policy: 政策
- reports: 研报
"""

from .scheduler import (
    UnstructuredFullCollectionScheduler,
    TaskStatus,
    TaskResult,
    CollectionProgress,
    CheckpointData,
)

from .config import (
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
)


def list_all_tasks():
    """列出所有采集任务"""
    result = {}
    for data_type, tasks in TASKS_BY_TYPE.items():
        result[data_type.value] = {
            "type_name": TYPE_NAMES.get(data_type, data_type.value),
            "tasks": [
                {
                    "name": t.name,
                    "description": t.description,
                    "storage_pattern": t.storage_pattern.value,
                    "stock_scope": t.stock_scope.value,
                    "enabled": t.enabled,
                }
                for t in tasks
            ]
        }
    return result


def get_task_count():
    """获取任务统计"""
    enabled = get_enabled_tasks()
    return {
        "total_tasks": len(ALL_TASKS),
        "enabled_tasks": len(enabled),
        "stock_related_tasks": len([t for t in enabled if t.stock_scope == StockScope.ALL_A]),
        "by_type": {
            dtype.value: len([t for t in tasks if t.enabled])
            for dtype, tasks in TASKS_BY_TYPE.items()
        }
    }


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
