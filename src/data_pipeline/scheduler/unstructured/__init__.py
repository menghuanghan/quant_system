"""
非结构化数据调度器模块

提供全量历史数据回填能力，支持：
- 断点续传：中断后自动恢复
- 资源编排：内存监控、并发控制、存储限制
- 策略路由：热/冷数据差异化处理
- 异常熔断：403/封禁时自动暂停

核心组件：
- CheckpointManager: 状态管理与断点续传
- UnstructuredHistoryScheduler: 主调度器
- CollectorRegistry: 采集器注册表

快速使用：
    >>> from src.data_pipeline.scheduler.unstructured import (
    ...     UnstructuredHistoryScheduler,
    ...     run_backfill,
    ...     get_backfill_progress
    ... )
    >>> 
    >>> # 方式1: 使用调度器对象
    >>> scheduler = UnstructuredHistoryScheduler()
    >>> scheduler.register_collector('news', NewsCollector)
    >>> scheduler.run_backfill(2021, 2025)
    >>> 
    >>> # 方式2: 快速函数
    >>> run_backfill(2024, 2025, sources=['news_sina'])
    >>> 
    >>> # 查看进度
    >>> progress = get_backfill_progress()
"""

# 状态管理
from .checkpoint import (
    TaskState,
    TaskRecord,
    BackfillStatus,
    CheckpointManager,
    get_checkpoint_manager,
    set_checkpoint_manager
)

# 配置模块
from .config import (
    # 数据温度
    DataTemperature,
    TemperaturePolicy,
    TEMPERATURE_POLICIES,
    get_temperature_for_date,
    get_policy_for_month,
    
    # 时间槽
    TimeSlot,
    generate_time_slots,
    
    # 采集器配置
    CollectorConfig,
    CollectorRegistry,
    get_default_collector_configs,
    register_default_collectors,
    
    # 调度器配置
    SchedulerConfig,
    
    # 工具函数
    get_output_path,
    validate_parquet_file,
    estimate_storage_usage
)

# 调度器
from .history_scheduler import (
    # 异常
    SchedulerError,
    CircuitBreakerError,
    RateLimitError,
    StorageLimitError,
    DependencyError,
    
    # 熔断器
    CircuitBreakerState,
    CircuitBreaker,
    
    # 资源监控
    ResourceMonitor,
    
    # 任务执行
    TaskResult,
    TaskExecutor,
    
    # 主调度器
    UnstructuredHistoryScheduler,
    
    # 便捷函数
    run_backfill,
    get_backfill_progress,
    reset_failed_tasks
)

__all__ = [
    # 状态管理
    'TaskState',
    'TaskRecord',
    'BackfillStatus',
    'CheckpointManager',
    'get_checkpoint_manager',
    'set_checkpoint_manager',
    
    # 数据温度
    'DataTemperature',
    'TemperaturePolicy',
    'TEMPERATURE_POLICIES',
    'get_temperature_for_date',
    'get_policy_for_month',
    
    # 时间槽
    'TimeSlot',
    'generate_time_slots',
    
    # 采集器配置
    'CollectorConfig',
    'CollectorRegistry',
    'get_default_collector_configs',
    'register_default_collectors',
    
    # 调度器配置
    'SchedulerConfig',
    
    # 工具函数
    'get_output_path',
    'validate_parquet_file',
    'estimate_storage_usage',
    
    # 异常
    'SchedulerError',
    'CircuitBreakerError',
    'RateLimitError',
    'StorageLimitError',
    'DependencyError',
    
    # 熔断器
    'CircuitBreakerState',
    'CircuitBreaker',
    
    # 资源监控
    'ResourceMonitor',
    
    # 任务执行
    'TaskResult',
    'TaskExecutor',
    
    # 主调度器
    'UnstructuredHistoryScheduler',
    
    # 便捷函数
    'run_backfill',
    'get_backfill_progress',
    'reset_failed_tasks'
]