"""
公告数据过滤器模块

对公告数据进行两层过滤：
1. 第一层：根据events的original_id过滤掉已在事件中的公告
2. 第二层：根据title黑名单关键词过滤垃圾公告

支持GPU加速（使用cuDF）

使用示例：
```python
from src.data_pipeline.processors.unstructured.filter import (
    AnnouncementFilter,
    FilterConfig,
    FilterResult,
    filter_month,
    filter_year,
    filter_all,
    get_filter_statistics,
    print_filter_statistics,
)

# 使用过滤器
filter = AnnouncementFilter(use_gpu=True)
result = filter.filter_month(year=2021, month=1)
print(result)

# 便捷函数
result = filter_month(2021, 1)

# 获取统计信息
stats = get_filter_statistics()
print_filter_statistics(stats)
```
"""

from .announcement_filter import (
    AnnouncementFilter,
    FilterConfig,
    FilterResult,
    TITLE_BLACKLIST_KEYWORDS,
)
from .filter_scheduler import (
    filter_month,
    filter_year,
    filter_all,
    get_filter_statistics,
    print_filter_statistics,
)

__all__ = [
    # 过滤器
    'AnnouncementFilter',
    'FilterConfig',
    'FilterResult',
    'TITLE_BLACKLIST_KEYWORDS',
    # 便捷函数
    'filter_month',
    'filter_year',
    'filter_all',
    'get_filter_statistics',
    'print_filter_statistics',
]
