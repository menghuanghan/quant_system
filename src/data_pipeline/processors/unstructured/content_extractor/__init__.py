"""
内容提取器模块 (Content Extractor)

提供从各种非结构化数据源提取文本内容的工具：
- PDF文本提取（公告、研报、事件）
- HTML正文提取（政策页面）
- 巨潮资讯详情页解析

支持的数据源：
- 公告(announcements): 巨潮资讯PDF
- 研报(reports): 东方财富PDF  
- 事件(events): 巨潮资讯详情页->PDF
- 政策(policy): 国务院/发改委HTML页面
- 新闻(news): 
    - CCTV新闻: 已有content
    - 交易所公告: 从title提取

使用示例:
```python
from src.data_pipeline.processors.unstructured import ContentExtractor, DataSourceType

# 创建提取器
extractor = ContentExtractor()

# 提取公告PDF内容
result = extractor.extract(
    "http://static.cninfo.com.cn/finalpage/2021-01-01/1209035187.PDF",
    domain='announcements'
)
if result.success:
    print(result.content_text)

# 提取研报PDF内容  
result = extractor.extract(
    "https://pdf.dfcfw.com/pdf/H3_AP202101051447329936_1.pdf",
    domain='reports'
)

# 提取政策HTML内容
result = extractor.extract(
    "https://www.gov.cn/zhengce/2021-01/31/content_5583936.htm",
    domain='policy/gov'
)

# 批量提取
urls = ["url1", "url2", "url3"]
batch_result = extractor.extract_batch(urls, domain='announcements')
print(f"成功率: {batch_result.success_rate:.2%}")

# 从DataFrame提取
import pandas as pd
df = pd.read_parquet("data.parquet")
df = extractor.extract_from_dataframe(
    df,
    url_column='url',
    output_column='content_text',
    domain='announcements'
)
```
"""

from .base import (
    BaseExtractor,
    ExtractorResult,
    DataSourceType,
    ContentType,
    TextCleaningMixin,
)
from .pdf_extractor import PDFExtractor
from .html_extractor import HTMLExtractor
from .cninfo_detail_parser import CninfoDetailParser
from .factory import ContentExtractorFactory
from .content_extractor import (
    ContentExtractor,
    BatchResult,
    extract_content,
    extract_content_batch,
)

__all__ = [
    # 基类和类型
    'BaseExtractor',
    'ExtractorResult',
    'DataSourceType', 
    'ContentType',
    'TextCleaningMixin',
    # 具体提取器
    'PDFExtractor',
    'HTMLExtractor',
    'CninfoDetailParser',
    # 工厂和主接口
    'ContentExtractorFactory',
    'ContentExtractor',
    # 批量结果
    'BatchResult',
    # 便捷函数
    'extract_content',
    'extract_content_batch',
]
