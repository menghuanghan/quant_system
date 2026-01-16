"""
全量调度器模块

提供原始数据采集调度器和数据清洗调度器
"""

# 导出采集调度器
from .collection.scheduler import (
    FullCollectionScheduler,
    TaskStatus,
    TaskResult,
    CollectionProgress,
)

# 导出配置常量
from .collection.config import (
    CollectionTask,
    DataCategory,
    StockScope,
    CollectionFrequency,
    ALL_TASKS,
    TASKS_BY_DOMAIN,
    DOMAIN_NAMES,
    get_enabled_tasks,
    get_tasks_sorted_by_priority,
    get_tasks_by_domain,
)


def list_all_tasks():
    """列出所有采集任务"""
    result = {}
    for domain, tasks in TASKS_BY_DOMAIN.items():
        result[domain] = {
            "domain_name": DOMAIN_NAMES.get(domain, domain),
            "tasks": [
                {
                    "name": t.name,
                    "description": t.description,
                    "category": t.category.value,
                    "stock_scope": t.stock_scope.value,
                    "enabled": t.enabled,
                    "realtime": t.realtime,
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
        "time_dependent_tasks": len([t for t in enabled if t.category == DataCategory.TIME_DEPENDENT]),
        "time_independent_tasks": len([t for t in enabled if t.category == DataCategory.TIME_INDEPENDENT]),
        "stock_related_tasks": len([t for t in enabled if t.stock_scope == StockScope.ALL_A]),
    }


__all__ = [
    # 调度器
    "FullCollectionScheduler",
    "TaskStatus",
    "TaskResult",
    "CollectionProgress",
    # 配置
    "CollectionTask",
    "DataCategory",
    "StockScope",
    "CollectionFrequency",
    "ALL_TASKS",
    "TASKS_BY_DOMAIN",
    "DOMAIN_NAMES",
    # 函数
    "get_enabled_tasks",
    "get_tasks_sorted_by_priority",
    "get_tasks_by_domain",
    "list_all_tasks",
    "get_task_count",
]