"""
非结构化数据调度器模块

提供全量历史数据回填能力，支持：
- 断点续传：中断后自动恢复
- 资源编排：内存监控、并发控制、存储限制
- 策略路由：热/冷数据差异化处理
- 异常熔断：403/封禁时自动暂停
- 数据类型选择：支持6种非结构化数据（公告/新闻/研报/舆情/政策/事件）
- 上下文依赖注入：舆情需要异动表，研报需要股票池

核心组件：
- CheckpointManager: 状态管理与断点续传
- EnhancedHistoryScheduler: 增强版主调度器
- DataType: 数据类型枚举
- CollectorDefinition: 采集器定义

快速使用：
    >>> from src.data_pipeline.scheduler.unstructured import (
    ...     EnhancedHistoryScheduler,
    ...     run_full_history,
    ...     DataType
    ... )
    >>> 
    >>> # 方式1: 采集全部数据
    >>> scheduler = EnhancedHistoryScheduler()
    >>> scheduler.run('20240101', '20240331')
    >>> 
    >>> # 方式2: 选择性采集
    >>> scheduler.run('20240101', '20240331', data_types=['news', 'announcements'])
    >>> 
    >>> # 方式3: 快速函数
    >>> run_full_history('20240101', '20240331', data_types=['news'])
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

# 增强版调度器
from .full_history_scheduler import (
    EnhancedHistoryScheduler,
    run_full_history,
    run_q1_2024,
    run_news_only,
    run_announcements_only,
    ContextManager,
)

# 数据类型定义
from .data_types import (
    DataType,
    CollectorDefinition,
    ContextRequirement,
    get_all_collector_definitions,
    get_collectors_by_type,
    get_enabled_collectors,
    get_collectors_by_names,
    get_default_context,
    DataTypeGroup,
    print_collector_summary,
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
    'reset_failed_tasks',
    
    # 增强版调度器
    'EnhancedHistoryScheduler',
    'run_full_history',
    'run_q1_2024',
    'run_news_only',
    'run_announcements_only',
    'ContextManager',
    
    # 数据类型
    'DataType',
    'CollectorDefinition',
    'ContextRequirement',
    'get_all_collector_definitions',
    'get_collectors_by_type',
    'get_enabled_collectors',
    'get_collectors_by_names',
    'get_default_context',
    'DataTypeGroup',
    'print_collector_summary',
]