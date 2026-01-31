# 公告数据过滤器模块

## 概述

公告数据过滤器用于过滤掉A股市场中大量无Alpha价值的垃圾公告（约80%），保留有价值的公告信息。

## 背景

- A股市场每月全市场有上万条公告
- 5年累计约270万条公告
- 90%的公告是垃圾信息（董事会通知、监事辞职、独立董事提名等）
- 如果不过滤，按1秒/条处理需要20多天
- 过滤后只需处理约20%的有价值公告

## 过滤逻辑

### 第一层：事件ID过滤

根据 `events` 数据的 `original_id` 过滤掉已在事件中的公告。

原理：事件数据已经从公告中提取了关键事件，无需重复处理。

### 第二层：标题关键词过滤

根据标题中的黑名单关键词过滤垃圾公告。

黑名单关键词分类：

| 分类 | 关键词 |
|------|--------|
| 会议相关 | 通知、会议、决议、议案、审议 |
| 人事相关 | 辞职、选举、提名、候选、任命、聘任、换届、改选、补选、监事、独立董事 |
| 意见/声明类 | 意见、声明、认可、确认、认定、核查、回复、复函、说明、解释 |
| 法律/审计类 | 律师、法律、验资、审计、鉴证、评估报告、备查、备案、函件 |
| 规章制度类 | 章程、规则、制度、细则、办法 |
| 日常管理类 | 授权、委托、登记、注册、日常经营、日常关联、内控 |
| 股权相关 | 股份变动、增持、减持、质押、冻结、解除、回购、股权激励、限制性股票、期权、行权 |
| 券商/保荐相关 | 保荐、督导、现场检查、培训情况 |
| 资金管理类 | 担保、贷款、授信、借款、融资租赁、现金管理、闲置募集资金、自有资金、理财产品 |
| 形式性文件 | 权益变动报告、提示性公告、进展公告、政府补助、风险提示、投资者关系、更正、更新、修订、补充、摘要、问询函、关注函、询证函、预告、名单、名册、挂牌、承诺、延期 |
| 董事会/股东大会 | 董事会、股东大会、审核、转让 |

### 保留的有价值公告

- 业绩相关：年报、季报、业绩快报
- 重大事项：收购、重组、合并、分立
- 业务相关：合同、订单、中标
- 投资相关：对外投资、设立子公司
- 其他：重大诉讼、重大仲裁

## GPU加速

使用 RAPIDS cuDF 进行 GPU 加速：

- cuDF 的 `str.contains()` 对标题进行批量正则匹配
- cuDF 的 `isin()` 对 original_id 进行高效集合匹配
- 自动回退到 pandas（当 GPU 不可用时）

性能对比（2021年1月数据，36,404条）：
- CPU (pandas): ~2-3秒
- GPU (cuDF): ~0.3秒

## 使用方法

### Python API

```python
from src.data_pipeline.processors.unstructured.filter import (
    AnnouncementFilter,
    FilterConfig,
    filter_month,
    filter_year,
    filter_all,
    get_filter_statistics,
    print_filter_statistics,
)

# 方式1：使用便捷函数
result = filter_month(2021, 1, use_gpu=True)
print(result.summary())

# 方式2：使用过滤器类
filter_instance = AnnouncementFilter(use_gpu=True)
result = filter_instance.filter_month(2021, 1)

# 过滤整年
results = filter_year(2021, use_gpu=True)

# 过滤所有数据
all_results = filter_all(use_gpu=True)

# 获取统计信息（不修改数据）
stats = get_filter_statistics()
print_filter_statistics(stats)

# 只统计不写入（dry run）
result = filter_month(2021, 1, dry_run=True)
```

### 命令行

```bash
# 使用启动器脚本
python scripts/run_unstructured_processing.py --filter --year 2021
python scripts/run_unstructured_processing.py --filter --year 2021 --month 1
python scripts/run_unstructured_processing.py --filter --all-years

# 统计模式（不修改数据）
python scripts/run_unstructured_processing.py --filter-stats
python scripts/run_unstructured_processing.py --filter-stats --year 2021
```

## 输出

过滤后的数据会覆盖原始文件：
- `data/raw/unstructured/announcements/{year}/{month}.parquet`

如果需要备份原始文件，可以配置：
```python
config = FilterConfig(
    backup_original=True,
    backup_dir="data/raw/unstructured/announcements_backup"
)
```

## 数据结构

### FilterResult

```python
@dataclass
class FilterResult:
    year: int                     # 年份
    month: int                    # 月份
    original_count: int           # 原始记录数
    after_event_filter: int       # 事件过滤后记录数
    after_title_filter: int       # 标题过滤后记录数
    final_count: int              # 最终记录数
    event_filtered_count: int     # 被事件过滤的数量
    title_filtered_count: int     # 被标题过滤的数量
    total_filtered_count: int     # 总过滤数量
    filter_rate: float            # 过滤率
    elapsed_time: float           # 耗时（秒）
    output_path: Optional[str]    # 输出文件路径
    error_message: Optional[str]  # 错误信息
```

## 注意事项

1. **不可逆操作**：过滤会覆盖原始文件，建议先使用 `dry_run=True` 预览效果
2. **备份建议**：首次运行前建议备份原始数据
3. **GPU内存**：处理大文件时注意GPU内存使用
4. **过滤率调整**：可以通过修改 `TITLE_BLACKLIST_KEYWORDS` 调整过滤率
