# Scheduler - 非结构化数据处理调度器

调度 `content_extractor`、`summarizer`、`scorer` 三个工具，完成对 `data/raw/unstructured` 下原始数据的端到端处理。

## 功能特点

- **多种数据类别**：支持 announcements、reports、events、exchange
- **双流水线**：PDF流水线（LLM处理）+ Exchange流水线（规则打分）
- **GPU加速**：Exchange批量处理使用cuDF GPU加速
- **断点续传**：支持检查点恢复
- **进度跟踪**：详细的日志和统计

## 架构

```
scheduler/
├── __init__.py       # 模块导出
├── base.py           # 配置、数据结构、枚举
├── pipeline.py       # PDF流水线、Exchange流水线
├── scheduler.py      # 主调度器
└── README.md         # 本文档
```

## 处理流程

### PDF类数据（announcements/reports/events）

```
原始数据 (URL) 
    ↓
ContentExtractor (提取PDF内容)
    ↓
Summarizer (生成摘要，需要LLM)
    ↓
Scorer (打分，需要LLM)
    ↓
输出: id, ts_code, date, score, reason
```

### Exchange数据（news/exchange）

```
原始数据 (title)
    ↓
RuleScorer (规则打分，GPU加速)
    ↓
输出: id, ts_code, date, score
```

## 快速开始

### 命令行使用

```bash
# 发现可用数据
python scripts/run_unstructured_processing.py --discover

# 处理单个月份的Exchange数据（快速）
python scripts/run_unstructured_processing.py --category news/exchange --year 2021 --month 1

# 处理整年的Exchange数据
python scripts/run_unstructured_processing.py --category news/exchange --year 2021

# 处理公告数据（需要LLM，较慢）
python scripts/run_unstructured_processing.py --category announcements --year 2021 --month 1

# 处理所有数据
python scripts/run_unstructured_processing.py --all

# 强制重新处理
python scripts/run_unstructured_processing.py --category news/exchange --year 2021 --force
```

### Python API

```python
from src.data_pipeline.processors.unstructured import (
    UnstructuredScheduler,
    ProcessingConfig,
    DataCategory,
)

# 创建配置
config = ProcessingConfig(
    use_gpu=True,
    skip_existing=True,
)

# 创建调度器
scheduler = UnstructuredScheduler(config)

# 发现数据
discovery = scheduler.discover_data()
print(discovery)

# 处理单个月份
result = scheduler.process_month(DataCategory.EXCHANGE, 2021, 1)
print(f"成功: {result.success_count}, 失败: {result.failed_count}")

# 处理整年
results = scheduler.process_year(DataCategory.EXCHANGE, 2021)

# 处理所有数据
all_results = scheduler.process_all()
```

## 配置

```python
@dataclass
class ProcessingConfig:
    # 路径配置
    raw_data_dir: str = "data/raw/unstructured"
    processed_data_dir: str = "data/processed/unstructured"
    checkpoint_dir: str = "data/checkpoints/unstructured_scheduler"
    
    # LLM配置
    model_name: str = "qwen2.5:7b-instruct"
    ollama_host: str = "http://localhost:11434"
    llm_timeout: float = 60.0
    llm_max_retries: int = 3
    
    # 处理配置
    batch_size: int = 10
    max_workers: int = 4
    use_gpu: bool = True
    skip_existing: bool = True
```

## 数据类别

| 类别 | 路径 | 流水线 | 输出字段 |
|-----|------|-------|---------|
| announcements | announcements/ | PDF | id, ts_code, date, score, reason |
| reports | reports/ | PDF | id, ts_code, date, score, reason |
| events | events/ | PDF | id, ts_code, date, score, reason |
| exchange | news/exchange/ | Exchange | id, ts_code, date, score |

## 数据映射

### 字段映射

| 类别 | ID字段 | 代码字段 | 日期字段 | URL字段 |
|-----|--------|---------|---------|---------|
| announcements | original_id | ts_code | date | url |
| reports | report_id | stock_code | date | pdf_url |
| events | id | ts_code | date | pdf_url |
| exchange | news_id | stock_code | date | - |

## 性能

| 数据类别 | 处理方式 | 速度 | GPU加速 |
|---------|---------|------|---------|
| exchange | 规则打分 | 300-500条/秒 | ✓ |
| announcements | LLM | 1-3条/分钟 | ✗ |
| reports | LLM | 1-3条/分钟 | ✗ |
| events | LLM | 1-3条/分钟 | ✗ |

## 输出示例

### Exchange输出

```
data/processed/unstructured/news/exchange/2021/01.parquet

                 id ts_code        date  score
0  16e163e50f45d0e1  600322  2021-01-30      0
1  ed0e02a89998d9ff  600221  2021-01-29      0
2  47cfaf33e8d1d3e1  600555  2021-01-29    -50
...
```

### PDF类输出

```
data/processed/unstructured/announcements/2021/01.parquet

           id    ts_code        date  score                reason
0  1209035187  300065.SZ  2021-01-01      5  独立董事声明，无重大风险
1  1209035185  300065.SZ  2021-01-01      5  独立董事候选人声明，中性
...
```

## 检查点

检查点保存在 `data/checkpoints/unstructured_scheduler/{category}/{year}/{month}.json`：

```json
{
    "category": "news/exchange",
    "year": 2021,
    "month": 1,
    "processed_ids": ["id1", "id2", ...],
    "total_records": 61,
    "success_count": 61,
    "failed_count": 0,
    "status": "completed",
    "last_update_time": "2026-01-29T14:16:59"
}
```

## 依赖

- Python 3.10+
- pandas >= 1.5.0
- cudf-cu12 (GPU加速)
- Ollama + qwen2.5:7b-instruct (LLM)

## 测试

```bash
# 运行调度器测试
python tests/test_scheduler.py
```
