"""
增量调度器模块

提供增量数据采集调度器，支持：
- 采集所有结构化非实时数据
- 只采集时间相关非实时数据
- 只采集时间无关非实时数据
"""

# 导出采集调度器
from .scheduler import (
    IncrementCollectionScheduler,
    TaskStatus,
    TaskResult,
    CollectionProgress,
)

# 导出配置常量和函数
from .config import (
    CollectionTask,
    DataCategory,
    StockScope,
    CollectionFrequency,
    CollectionMode,
    ALL_TASKS,
    TASKS_BY_DOMAIN,
    DOMAIN_NAMES,
    DEFAULT_INCREMENT_OUTPUT_DIR,
    get_enabled_tasks,
    get_tasks_sorted_by_priority,
    get_tasks_by_domain,
    get_tasks_by_mode,
    get_tasks_by_mode_and_domain,
    get_increment_task_stats,
)


def list_all_tasks(mode: CollectionMode = CollectionMode.ALL):
    """
    列出所有采集任务
    
    Args:
        mode: 采集模式，用于筛选任务
    """
    result = {}
    for domain, tasks in TASKS_BY_DOMAIN.items():
        filtered_tasks = []
        for t in tasks:
            if not t.enabled or t.realtime:
                continue
            if mode == CollectionMode.TIME_DEPENDENT and t.category != DataCategory.TIME_DEPENDENT:
                continue
            if mode == CollectionMode.TIME_INDEPENDENT and t.category != DataCategory.TIME_INDEPENDENT:
                continue
            filtered_tasks.append({
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
                "stock_scope": t.stock_scope.value,
                "enabled": t.enabled,
                "realtime": t.realtime,
            })
        
        if filtered_tasks:
            result[domain] = {
                "domain_name": DOMAIN_NAMES.get(domain, domain),
                "tasks": filtered_tasks
            }
    return result


def get_task_count(mode: CollectionMode = CollectionMode.ALL):
    """
    获取任务统计
    
    Args:
        mode: 采集模式，用于统计
    """
    return get_increment_task_stats(mode)


__all__ = [
    # 调度器
    "IncrementCollectionScheduler",
    "TaskStatus",
    "TaskResult",
    "CollectionProgress",
    # 配置
    "CollectionTask",
    "DataCategory",
    "StockScope",
    "CollectionFrequency",
    "CollectionMode",
    "ALL_TASKS",
    "TASKS_BY_DOMAIN",
    "DOMAIN_NAMES",
    "DEFAULT_INCREMENT_OUTPUT_DIR",
    # 函数
    "get_enabled_tasks",
    "get_tasks_sorted_by_priority",
    "get_tasks_by_domain",
    "get_tasks_by_mode",
    "get_tasks_by_mode_and_domain",
    "get_increment_task_stats",
    "list_all_tasks",
    "get_task_count",
]