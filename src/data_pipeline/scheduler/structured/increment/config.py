"""
增量数据采集调度器配置模块

复用全量调度器的配置，提供增量采集特定的功能：
- 支持采集所有结构化非实时数据
- 支持只采集时间相关非实时数据
- 支持只采集时间无关非实时数据
"""

from enum import Enum
from typing import List

# 从全量调度器导入配置
from src.data_pipeline.scheduler.structured.full.collection.config import (
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
    get_tasks_by_category,
)


class CollectionMode(Enum):
    """采集模式"""
    ALL = "all"                            # 采集所有非实时数据
    TIME_DEPENDENT = "time_dependent"      # 只采集时间相关非实时数据
    TIME_INDEPENDENT = "time_independent"  # 只采集时间无关非实时数据


# 增量采集默认输出目录
DEFAULT_INCREMENT_OUTPUT_DIR = "data/raw/inc_structured"


def get_tasks_by_mode(mode: CollectionMode) -> List[CollectionTask]:
    """
    根据采集模式获取任务列表
    
    Args:
        mode: 采集模式
            - ALL: 采集所有非实时数据
            - TIME_DEPENDENT: 只采集时间相关非实时数据
            - TIME_INDEPENDENT: 只采集时间无关非实时数据
    
    Returns:
        符合条件的任务列表
    """
    enabled_tasks = get_enabled_tasks()
    
    if mode == CollectionMode.ALL:
        return enabled_tasks
    elif mode == CollectionMode.TIME_DEPENDENT:
        return [t for t in enabled_tasks if t.category == DataCategory.TIME_DEPENDENT]
    elif mode == CollectionMode.TIME_INDEPENDENT:
        return [t for t in enabled_tasks if t.category == DataCategory.TIME_INDEPENDENT]
    else:
        return enabled_tasks


def get_tasks_by_mode_and_domain(mode: CollectionMode, domain: str) -> List[CollectionTask]:
    """
    根据采集模式和数据域获取任务列表
    
    Args:
        mode: 采集模式
        domain: 数据域名称
    
    Returns:
        符合条件的任务列表
    """
    domain_tasks = get_tasks_by_domain(domain)
    enabled_tasks = [t for t in domain_tasks if t.enabled and not t.realtime]
    
    if mode == CollectionMode.ALL:
        return enabled_tasks
    elif mode == CollectionMode.TIME_DEPENDENT:
        return [t for t in enabled_tasks if t.category == DataCategory.TIME_DEPENDENT]
    elif mode == CollectionMode.TIME_INDEPENDENT:
        return [t for t in enabled_tasks if t.category == DataCategory.TIME_INDEPENDENT]
    else:
        return enabled_tasks


def get_increment_task_stats(mode: CollectionMode = CollectionMode.ALL):
    """
    获取增量采集任务统计
    
    Args:
        mode: 采集模式
    
    Returns:
        统计信息字典
    """
    tasks = get_tasks_by_mode(mode)
    
    return {
        "mode": mode.value,
        "total_tasks": len(tasks),
        "time_dependent_tasks": len([t for t in tasks if t.category == DataCategory.TIME_DEPENDENT]),
        "time_independent_tasks": len([t for t in tasks if t.category == DataCategory.TIME_INDEPENDENT]),
        "stock_related_tasks": len([t for t in tasks if t.stock_scope == StockScope.ALL_A]),
        "domains": {
            domain: len(get_tasks_by_mode_and_domain(mode, domain))
            for domain in TASKS_BY_DOMAIN.keys()
        }
    }


__all__ = [
    # 从全量调度器导入
    "CollectionTask",
    "DataCategory",
    "StockScope",
    "CollectionFrequency",
    "ALL_TASKS",
    "TASKS_BY_DOMAIN",
    "DOMAIN_NAMES",
    "get_enabled_tasks",
    "get_tasks_sorted_by_priority",
    "get_tasks_by_domain",
    "get_tasks_by_category",
    # 增量调度器新增
    "CollectionMode",
    "DEFAULT_INCREMENT_OUTPUT_DIR",
    "get_tasks_by_mode",
    "get_tasks_by_mode_and_domain",
    "get_increment_task_stats",
]
