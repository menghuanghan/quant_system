# 内容提取器模块 (Content Extractor)

## 概述

内容提取器是非结构化数据处理流水线的第一个工具，负责从各种数据源URL中提取原始文本内容（`content_text`）。

## 功能特性

- **PDF文本提取**: 使用PyMuPDF(fitz)从PDF文件中提取文本，支持公告、研报、事件等PDF
- **HTML正文提取**: 使用trafilatura和readability-lxml从网页中提取正文，自动去除广告和导航
- **巨潮资讯详情页解析**: 自动解析详情页获取真实PDF链接
- **工厂模式**: 根据URL自动识别数据源类型并选择合适的提取器
- **批量处理**: 支持并发批量提取，提高处理效率
- **GPU加速**: 使用cuDF进行批量文本清洗（需要RAPIDS环境）

## 支持的数据源

| 数据源 | 类型 | URL示例 | 提取器 |
|--------|------|---------|--------|
| 公告 (announcements) | PDF | `http://static.cninfo.com.cn/finalpage/xxx.PDF` | PDFExtractor |
| 研报 (reports) | PDF | `https://pdf.dfcfw.com/pdf/xxx.pdf` | PDFExtractor |
| 事件 (events) | PDF | 详情页或直接PDF URL | CninfoDetailParser + PDFExtractor |
| 国务院政策 (policy/gov) | HTML | `https://www.gov.cn/zhengce/xxx.htm` | HTMLExtractor |
| 发改委政策 (policy/ndrc) | HTML | `https://www.ndrc.gov.cn/xxx.html` | HTMLExtractor |
| CCTV新闻 (news/cctv) | 已有content | - | 直接返回existing_content |
| 交易所公告 (news/exchange) | 仅title | - | 直接返回title |

## 快速开始

### 基本使用

```python
from src.data_pipeline.processors.unstructured import ContentExtractor, DataSourceType

# 创建提取器
extractor = ContentExtractor(
    timeout=30,       # 请求超时时间
    max_workers=4,    # 批量处理并发数
    use_gpu=True      # 是否使用GPU加速
)

# 单个URL提取
result = extractor.extract(
    "http://static.cninfo.com.cn/finalpage/2021-01-01/1209035187.PDF",
    domain='announcements'
)

if result.success:
    print(f"提取成功: {result.char_count} 字符")
    print(result.content_text)
else:
    print(f"提取失败: {result.error_message}")
```

### 批量提取

```python
# 批量提取
urls = [
    "http://static.cninfo.com.cn/xxx1.PDF",
    "http://static.cninfo.com.cn/xxx2.PDF",
    "https://pdf.dfcfw.com/pdf/xxx.pdf",
]

batch_result = extractor.extract_batch(urls, domain='announcements', max_workers=4)

print(f"成功率: {batch_result.success_rate:.1%}")
print(f"总耗时: {batch_result.elapsed_time_seconds:.2f}秒")

# 获取统计信息
stats = extractor.get_statistics(batch_result)
print(f"平均字符数: {stats['avg_characters']:.0f}")
```

### 从DataFrame提取

```python
import pandas as pd

# 读取数据
df = pd.read_parquet("data/raw/unstructured/announcements/2021/01.parquet")

# 批量提取并添加到DataFrame
df = extractor.extract_from_dataframe(
    df,
    url_column='url',           # URL列名
    output_column='content_text', # 输出列名
    domain='announcements',
    add_metadata=True           # 添加元数据列
)

# 现在df包含了content_text列
print(df[['title', 'content_text']].head())
```

### 处理事件数据（带pdf_url字段）

对于事件数据，推荐直接使用数据中的`pdf_url`字段：

```python
df = pd.read_parquet("data/raw/unstructured/events/2021/01.parquet")

# 方式1: 使用pdf_url字段
result = extractor.extract(df.iloc[0]['pdf_url'], domain='events')

# 方式2: 在DataFrame中使用pdf_url列
df = extractor.extract_from_dataframe(
    df,
    url_column='pdf_url',  # 使用pdf_url列而非url列
    output_column='content_text',
    domain='events'
)
```

## 模块结构

```
content_extractor/
├── __init__.py           # 模块入口，导出公开接口
├── base.py               # 基类定义和数据结构
├── pdf_extractor.py      # PDF文本提取器
├── html_extractor.py     # HTML正文提取器
├── cninfo_detail_parser.py  # 巨潮资讯详情页解析器
├── factory.py            # 提取器工厂
├── content_extractor.py  # 统一接口和批量处理
└── README.md             # 本文档
```

## 返回结果

`ExtractorResult` 数据结构：

| 字段 | 类型 | 说明 |
|------|------|------|
| success | bool | 是否成功 |
| content_text | str | 提取的文本内容 |
| source_url | str | 源URL |
| source_type | DataSourceType | 数据源类型 |
| content_type | ContentType | 内容类型(PDF/HTML等) |
| page_count | int | PDF页数 |
| char_count | int | 字符数 |
| process_time_ms | float | 处理耗时(毫秒) |
| error_message | str | 错误信息 |
| error_code | str | 错误代码 |
| actual_pdf_url | str | 实际PDF URL(用于详情页解析) |

## 依赖

- **PyMuPDF (fitz)**: PDF文本提取
- **trafilatura**: HTML正文提取(主要)
- **readability-lxml**: HTML正文提取(备用)
- **beautifulsoup4**: HTML解析
- **requests**: HTTP请求
- **cudf** (可选): GPU加速文本清洗

## 注意事项

1. **编码处理**: 自动检测和处理UTF-8、GBK等编码
2. **反爬机制**: 内置User-Agent轮换和请求头伪装
3. **超时重试**: 支持配置超时和重试次数
4. **内存处理**: PDF在内存中处理，不写入磁盘
5. **政府网站**: 部分政府网站使用JavaScript渲染，使用基础提取器处理

## 后续工具

内容提取器是三个工具中的第一个：

1. **内容提取器** (本模块): URL → content_text
2. **内容摘要器** (待实现): content_text → content (摘要)
3. **内容评分器** (待实现): content → score (量化评分)
