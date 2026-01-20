# 非结构化数据模块文档索引

## 📚 文档导航

### 核心文档

| 文档名称 | 描述 | 适用场景 | 路径 |
|---------|------|---------|------|
| **目录结构说明** | 详细介绍新目录设计、各模块存储策略、字段示例 | 了解目录组织原则 | [data/raw/unstructured/README.md](../data/raw/unstructured/README.md) |
| **重构总结** | 完成工作概览、性能提升、设计亮点、后续计划 | 快速了解改动全貌 | [docs/unstructured_refactoring_summary.md](./unstructured_refactoring_summary.md) |
| **迁移指南** | 代码迁移步骤、新旧代码对比、测试验证 | 迁移现有采集器 | [docs/unstructured_migration_guide.md](./unstructured_migration_guide.md) |
| **基础设施改进** | DataSink、PDF校验、HealthCheck、文本匹配等 | 了解新增工具 | [docs/unstructured_improvements.md](./unstructured_improvements.md) |

### 代码文件

| 文件名称 | 类型 | 描述 | 路径 |
|---------|------|------|------|
| **paths.py** | 核心工具 | 统一路径管理器（16个路径获取函数） | [src/utils/paths.py](../src/utils/paths.py) |
| **storage.py** | 核心工具 | DataSink存储管理（Parquet/JSONL/CSV） | [src/data_pipeline/collectors/unstructured/storage.py](../src/data_pipeline/collectors/unstructured/storage.py) |
| **text_matcher.py** | 核心工具 | 文本相似度匹配（4种算法） | [src/data_pipeline/collectors/unstructured/text_matcher.py](../src/data_pipeline/collectors/unstructured/text_matcher.py) |
| **pdf_utils.py** | 核心工具 | PDF校验与元数据提取 | [src/data_pipeline/collectors/unstructured/pdf_utils.py](../src/data_pipeline/collectors/unstructured/pdf_utils.py) |
| **scraper_base.py** | 核心工具 | 基础爬虫类（含HealthCheck、Mobile UA） | [src/data_pipeline/collectors/unstructured/scraper_base.py](../src/data_pipeline/collectors/unstructured/scraper_base.py) |
| **base_event.py** | 采集器 | 事件采集器基类（已迁移到新路径） | [src/data_pipeline/collectors/unstructured/events/base_event.py](../src/data_pipeline/collectors/unstructured/events/base_event.py) |

### 示例脚本

| 脚本名称 | 描述 | 运行命令 |
|---------|------|---------|
| **demo_path_management.py** | 路径管理器集成示例（6大模块） | `PYTHONPATH=. python scripts/demo_path_management.py` |
| **demo_improvements.py** | 基础设施改进演示（DataSink、HealthCheck等） | `.venv\Scripts\python.exe scripts\demo_improvements.py` |

---

## 🗂️ 按使用场景索引

### 场景1: 我要了解新目录结构

**推荐阅读顺序**:
1. [重构总结](./unstructured_refactoring_summary.md) - 快速了解改动
2. [目录结构说明](../data/raw/unstructured/README.md) - 详细设计文档
3. 运行示例：`python scripts/demo_path_management.py`

**关键信息**:
- 6大模块：announcements、news、reports、sentiment、policy、events
- 设计原则：存算分离、高效检索、分源管理
- 性能提升：Parquet压缩率62%，读取速度提升6.5x

---

### 场景2: 我要迁移现有采集器

**推荐阅读顺序**:
1. [迁移指南](./unstructured_migration_guide.md) - 完整迁移步骤
2. [paths.py 源码](../src/utils/paths.py) - API参考
3. [base_event.py](../src/data_pipeline/collectors/unstructured/events/base_event.py) - 已迁移示例

**迁移检查清单**:
- [ ] 备份现有数据
- [ ] 导入路径管理工具
- [ ] 替换路径生成逻辑
- [ ] 优化数据格式（CSV → Parquet）
- [ ] 运行测试验证

**代码模板**:
```python
from src.utils.paths import get_announcement_metadata_path, get_announcement_file_path

# 旧代码
meta_path = Path(f"data/raw/announcements/{year}/{month}/metadata.csv")

# 新代码
meta_path = get_announcement_metadata_path('cninfo', year, month, format='parquet')
```

---

### 场景3: 我要开发新的采集器

**推荐阅读顺序**:
1. [目录结构说明](../data/raw/unstructured/README.md) - 了解存储规范
2. [paths.py 源码](../src/utils/paths.py) - 路径API
3. [demo_path_management.py](../scripts/demo_path_management.py) - 集成示例
4. [基础设施改进](./unstructured_improvements.md) - 可用工具

**开发模板**:
```python
from src.utils.paths import get_news_path
from src.data_pipeline.collectors.unstructured.storage import get_data_sink
from src.data_pipeline.collectors.unstructured.scraper_base import ScraperBase

class MyNewsCollector:
    def __init__(self):
        self.scraper = ScraperBase(enable_health_check=True)
        self.sink = get_data_sink()
    
    def collect(self, date):
        # 1. 采集数据
        news_df = self._fetch_news(date)
        
        # 2. 使用DataSink保存（自动分区）
        self.sink.save(
            data=news_df,
            domain="news",
            sub_domain="my_source",
            partition_by="publish_date"
        )
```

---

### 场景4: 我要查看数据存储情况

**工具与方法**:

#### 方法1: 使用Python API
```python
from src.utils.paths import UnstructuredDataPaths
import json

summary = UnstructuredDataPaths.get_storage_summary()
print(json.dumps(summary, indent=2, ensure_ascii=False))
```

#### 方法2: 运行示例脚本
```bash
PYTHONPATH=. python scripts/demo_path_management.py
# 查看最后的"存储统计"部分
```

#### 方法3: 使用命令行
```powershell
# Windows PowerShell
tree /F data\raw\unstructured

# 统计文件数量
(Get-ChildItem -Recurse data\raw\unstructured -File).Count

# 统计总大小
(Get-ChildItem -Recurse data\raw\unstructured -File | Measure-Object -Property Length -Sum).Sum / 1MB
```

---

### 场景5: 我要优化存储性能

**推荐阅读**:
1. [目录结构说明 - 性能对比](../data/raw/unstructured/README.md#性能对比)
2. [重构总结 - 性能提升](./unstructured_refactoring_summary.md#-性能提升)

**关键优化点**:

| 优化项 | 方法 | 提升效果 |
|-------|------|---------|
| **存储格式** | CSV → Parquet (Snappy) | 压缩率 62% |
| **查询速度** | 列式存储 + 时间分区 | 读取快 6.5x |
| **文件管理** | 按year/month分区 | 避免单目录文件过多 |
| **元数据分离** | metadata/ 与 files/ 分离 | 查询快 10-100x |

**Parquet压缩对比**:
```python
# Snappy（默认，推荐）: 压缩快，解压快
df.to_parquet(path, compression='snappy')

# Gzip: 压缩率高，速度慢
df.to_parquet(path, compression='gzip')

# Brotli: 压缩率最高，速度最慢
df.to_parquet(path, compression='brotli')
```

---

### 场景6: 我要处理历史数据

**推荐阅读**:
1. [迁移指南 - Q1: 如何处理历史数据](./unstructured_migration_guide.md#q1-如何处理历史数据)

**方案选择**:

#### 方案A: 保持旧路径（推荐，风险低）
```python
# 兼容新旧路径
try:
    path = get_news_path('sina', date)
except:
    path = Path(f"data/raw/news/sina/{date}.json")
```

#### 方案B: 数据迁移（待实现脚本）
```bash
# 未来计划
python scripts/migrate_unstructured_data.py --dry-run
python scripts/migrate_unstructured_data.py --execute
```

---

## 🛠️ 快速参考

### 路径管理API速查

```python
from src.utils.paths import *

# 公告
get_announcement_metadata_path(source, year, month, format='parquet')
get_announcement_file_path(ts_code, ann_date, filename, use_stock_partition=False)

# 新闻
get_news_path(source, date, format='jsonl')

# 研报
get_report_metadata_path(year, month, format='parquet')
get_report_pdf_path(ts_code, report_date, org_name, rating='')

# 舆情
get_sentiment_path(source, date, format='parquet')

# 政策
get_policy_rules_path(agency, year, format='jsonl')
get_policy_file_path(agency, policy_id, filename)

# 事件
get_event_meta_path(event_type, year, month, format='parquet')
get_event_pdf_path(event_type, ts_code, ann_date, filename)
```

### DataSink API速查

```python
from src.data_pipeline.collectors.unstructured.storage import get_data_sink, StorageFormat

sink = get_data_sink()

# 保存
sink.save(
    data=df,                      # DataFrame或字典列表
    domain="news",                # 模块名称
    sub_domain="sina",            # 子域（数据源）
    partition_by="publish_date",  # 分区字段（可选）
    format=StorageFormat.PARQUET  # 格式
)

# 读取
df = sink.load(
    domain="news",
    filters={'year': 2025, 'month': 1}  # 分区过滤（可选）
)

# 统计
stats = sink.get_storage_stats(domain="news")
```

---

## 🔧 故障排查

### 问题1: ModuleNotFoundError: No module named 'src'

**原因**: Python找不到项目根目录

**解决**:
```bash
# 方式1: 设置PYTHONPATH（推荐）
export PYTHONPATH=/path/to/quant_system  # Linux/Mac
$env:PYTHONPATH = "C:\path\to\quant_system"  # Windows

# 方式2: 在脚本开头添加
import sys
sys.path.insert(0, '/path/to/quant_system')
```

### 问题2: Parquet文件无法打开

**工具**:
```bash
# 安装工具
pip install pandas pyarrow parquet-tools

# Python读取
import pandas as pd
df = pd.read_parquet('data.parquet')

# 命令行查看
parquet-tools show data.parquet
```

### 问题3: 目录权限错误

**检查**:
```python
from pathlib import Path
path = Path("data/raw/unstructured/news")
print(f"存在: {path.exists()}")
print(f"可写: {path.stat().st_mode}")

# 修复
path.mkdir(parents=True, exist_ok=True)
```

---

## 📧 联系方式

- **问题反馈**: 提交Issue到项目仓库
- **功能建议**: 创建Feature Request
- **紧急问题**: 联系项目维护者

---

**文档版本**: v1.0  
**更新日期**: 2026-01-19  
**维护者**: GitHub Copilot
