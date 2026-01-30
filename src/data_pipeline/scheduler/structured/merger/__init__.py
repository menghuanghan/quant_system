"""
结构化数据合并器模块

提供增量数据与全量数据的合并功能，支持GPU加速
"""

from .config import (
    MergeConfig,
    MergeMode,
    DateSortOrder,
    MergeTask,
    MERGE_TASKS_BY_DOMAIN,
    get_merge_tasks_by_domain,
    get_all_merge_tasks,
)

from .merger import (
    GPUDataMerger,
    DataMerger,
    MergeResult,
    MergeProgress,
    MergeStatus,
)

__all__ = [
    # 配置
    "MergeConfig",
    "MergeMode",
    "DateSortOrder",
    "MergeTask",
    "MERGE_TASKS_BY_DOMAIN",
    "get_merge_tasks_by_domain",
    "get_all_merge_tasks",
    # 合并器
    "GPUDataMerger",
    "DataMerger",
    "MergeResult",
    "MergeProgress",
    "MergeStatus",
]
