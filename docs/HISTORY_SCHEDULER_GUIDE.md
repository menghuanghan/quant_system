# 全量历史调度器使用指南

## 概述

`UnstructuredHistoryScheduler` 是企业级全量数据回填调度器，专为无人值守运行设计。

### 核心能力

| 能力 | 描述 |
|------|------|
| **断点续传** | 基于 JSON 文件的状态持久化，中断后自动恢复 |
| **资源编排** | 内存监控、并发控制、存储限制检测 |
| **策略路由** | 热/温/冷数据差异化处理（并发、文件保存策略） |
| **异常熔断** | 403/封禁时自动暂停，防止被永久封禁 |
| **输出验证** | Parquet 文件完整性检查 |

## 快速开始

### 1. 基础使用

```python
from src.data_pipeline.scheduler.unstructured import (
    UnstructuredHistoryScheduler,
    run_backfill
)

# 方式1: 快速函数
run_backfill(2021, 2025)

# 方式2: 使用调度器对象（推荐）
scheduler = UnstructuredHistoryScheduler()
scheduler.register_collector('news_sina', NewsCollector)
scheduler.run_backfill(start_year=2021, end_year=2025)
```

### 2. 自定义配置

```python
from src.data_pipeline.scheduler.unstructured import (
    UnstructuredHistoryScheduler,
    SchedulerConfig
)
from pathlib import Path

config = SchedulerConfig(
    start_year=2023,
    end_year=2025,
    global_max_workers=4,         # 全局最大并发
    max_memory_gb=8.0,            # 内存限制
    max_storage_gb=200.0,         # 存储限制
    output_base_dir=Path("data/raw/unstructured"),
    state_dir=Path("data/state"),
    circuit_break_threshold=5,    # 连续失败N次触发熔断
    circuit_break_duration=300    # 熔断持续时间（秒）
)

scheduler = UnstructuredHistoryScheduler(config=config)
```

### 3. 注册采集器

```python
from src.data_pipeline.collectors.unstructured import (
    StreamingNewsCollector,
    StreamingAnnouncementCollector
)

# 新闻采集器（普通优先级）
scheduler.register_collector(
    name='news_sina',
    collector_class=StreamingNewsCollector,
    priority=10,
    memory_intensive=False
)

# 公告采集器（内存密集型，降低并发）
scheduler.register_collector(
    name='announcement_cninfo',
    collector_class=StreamingAnnouncementCollector,
    priority=20,
    memory_intensive=True  # 自动降低并发到2
)
```

### 4. 运行回填

```python
# 全量回填
scheduler.run_backfill()

# 仅最近2年
scheduler.run_backfill(start_year=2024, end_year=2025)

# 仅特定数据源
scheduler.run_backfill(sources=['news_sina'])

# Dry Run（仅打印计划，不执行）
scheduler.run_backfill(dry_run=True)
```

## 数据温度策略

系统根据数据年份自动应用差异化策略：

| 温度 | 年份范围 | save_pdf | text_only | max_workers |
|------|----------|----------|-----------|-------------|
| **HOT** | 最近2年 | ✅ | ❌ | 6 |
| **WARM** | 3-4年前 | ❌ | ❌ | 4 |
| **COLD** | 5年以上 | ❌ | ✅ | 2 |

### 自定义策略覆盖

```python
from src.data_pipeline.scheduler.unstructured import CollectorConfig

config = CollectorConfig(
    name='heavy_collector',
    collector_class=HeavyCollector,
    override_policy={
        'max_workers': 1,      # 覆盖默认并发
        'batch_size': 20,      # 覆盖默认批量
        'request_delay': 3.0   # 覆盖默认请求间隔
    }
)
```

## 断点续传

### 状态文件位置

```
data/state/backfill_status.json
```

### 状态文件结构

```json
{
  "version": "1.0",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-15T10:30:00",
  "tasks": {
    "news_sina": {
      "2025-12": {"state": "COMPLETED", "record_count": 1000, ...},
      "2025-11": {"state": "FAILED", "error_message": "timeout", ...}
    }
  },
  "global_stats": {
    "total_tasks": 360,
    "completed": 120,
    "failed": 5,
    "pending": 235
  }
}
```

### 手动操作状态

```python
from src.data_pipeline.scheduler.unstructured import (
    get_checkpoint_manager,
    get_backfill_progress,
    reset_failed_tasks
)

# 查看进度
progress = get_backfill_progress()
print(f"完成: {progress['completed']}/{progress['total_tasks']}")

# 重置失败任务
reset_failed_tasks()  # 全部重置
reset_failed_tasks('news_sina')  # 仅重置特定数据源

# 直接操作 CheckpointManager
manager = get_checkpoint_manager()
manager.print_summary()
```

## 熔断器机制

当连续失败达到阈值时，调度器自动熔断：

```
CLOSED (正常) → 连续失败5次 → OPEN (熔断)
                              ↓
                         等待300秒
                              ↓
                         HALF_OPEN (试探)
                              ↓
                     成功 → CLOSED
                     失败 → OPEN
```

### 手动控制熔断器

```python
# 查看状态
state = scheduler.circuit_breaker.state
print(f"熔断器状态: {state.value}")

# 手动重置
scheduler.reset_circuit_breaker()
```

## 资源监控

```python
# 获取实时状态
progress = scheduler.get_progress()
print(f"内存使用: {progress['memory_usage_gb']:.2f} GB")
print(f"存储使用: {progress['storage_usage_gb']:.2f} GB")
print(f"熔断器: {progress['circuit_breaker_state']}")
```

## 目录结构

```
data/
├── raw/
│   └── unstructured/
│       ├── news_sina/
│       │   ├── 2025/
│       │   │   ├── news_sina_2025-12.parquet
│       │   │   └── news_sina_2025-11.parquet
│       │   └── 2024/
│       └── announcement_cninfo/
│           └── ...
├── state/
│   ├── backfill_status.json      # 主状态文件
│   └── backups/                  # 自动备份
│       ├── backfill_status_20240115_103000.json
│       └── ...
└── temp/                         # 临时文件
```

## 最佳实践

### 1. 首次运行前测试

```python
# 先 dry run 确认计划
scheduler.run_backfill(dry_run=True)

# 小范围测试
scheduler.run_backfill(start_year=2025, end_year=2025, sources=['news_sina'])
```

### 2. 长时间运行建议

```python
# 使用 nohup 或 screen
# nohup python run_backfill.py > backfill.log 2>&1 &

# 定期检查进度
from src.data_pipeline.scheduler.unstructured import get_backfill_progress
progress = get_backfill_progress()
```

### 3. 处理中断

中断后直接重新运行即可，调度器会：
1. 加载状态文件
2. 跳过 COMPLETED 任务
3. 优先重试 FAILED 任务（最多3次）
4. 继续处理 PENDING 任务

### 4. 存储空间不足

```python
# 检查使用情况
from src.data_pipeline.scheduler.unstructured import estimate_storage_usage
stats = estimate_storage_usage(Path("data/raw/unstructured"))
print(f"总使用: {stats['total_size_gb']:.2f} GB")

# 如果接近限制，考虑：
# 1. 对旧数据启用 text_only_mode
# 2. 清理不需要的原始文件
# 3. 增加存储限制
```

## 故障排除

### 1. 熔断频繁触发

可能原因：
- IP 被临时封禁
- 请求过于频繁

解决方案：
```python
# 增加请求间隔
config = SchedulerConfig(
    circuit_break_duration=600  # 增加恢复等待时间
)

# 或覆盖采集器策略
collector_config.override_policy = {
    'request_delay': 5.0  # 增加请求间隔
}
```

### 2. 内存溢出

```python
# 对内存密集型采集器特殊处理
scheduler.register_collector(
    name='heavy_collector',
    collector_class=HeavyCollector,
    memory_intensive=True  # 自动降低并发，任务间 GC
)
```

### 3. 状态文件损坏

```bash
# 从备份恢复
cp data/state/backups/backfill_status_YYYYMMDD_HHMMSS.json \
   data/state/backfill_status.json
```

## API 参考

### UnstructuredHistoryScheduler

| 方法 | 描述 |
|------|------|
| `register_collector(name, cls, **kwargs)` | 注册采集器 |
| `run_backfill(start_year, end_year, sources, dry_run)` | 运行回填 |
| `stop()` | 停止调度器 |
| `reset_circuit_breaker()` | 重置熔断器 |
| `get_progress()` | 获取进度 |

### CheckpointManager

| 方法 | 描述 |
|------|------|
| `get_task(source, month)` | 获取任务记录 |
| `update_task(source, month, state, **kwargs)` | 更新任务状态 |
| `get_pending_tasks(source, include_failed)` | 获取待处理任务 |
| `reset_failed_tasks(source)` | 重置失败任务 |
| `print_summary()` | 打印进度摘要 |

### 便捷函数

```python
from src.data_pipeline.scheduler.unstructured import (
    run_backfill,           # 快速回填
    get_backfill_progress,  # 获取进度
    reset_failed_tasks,     # 重置失败
    get_checkpoint_manager  # 获取状态管理器
)
```
