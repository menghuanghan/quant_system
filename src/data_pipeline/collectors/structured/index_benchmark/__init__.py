"""
指数与基准数据域（Index & Benchmark）采集模块

数据类型包括：
1. 股票指数：
   - 指数基本信息
   - 指数日线行情
   - 指数分钟行情（预留）
   
2. 指数成分与权重：
   - 指数成分股
   - 指数成分权重
   
3. 行业指数：
   - 申万行业指数
   - 中信行业指数
   
4. 波动率指数：
   - 中国波指(iVIX)
   
5. 全球主要指数：
   - 全球股票指数行情

使用方法:
    from src.data_pipeline.collectors.structured.index_benchmark import (
        # 股票指数
        get_index_basic,
        get_index_daily,
        get_index_weight,
        
        # 全球指数
        get_index_global,
    )
"""

# 指数数据采集器
from .stock_index import (
    IndexBasicCollector,
    IndexDailyCollector,
    IndexWeightCollector,
    IndexMemberCollector,
    get_index_basic,
    get_index_daily,
    get_index_weight,
    get_index_member,
)

# 全球指数采集器
from .global_index import (
    GlobalIndexCollector,
    get_index_global,
)

__all__ = [
    # 指数数据采集器类
    'IndexBasicCollector',
    'IndexDailyCollector',
    'IndexWeightCollector',
    'IndexMemberCollector',
    # 指数数据便捷函数
    'get_index_basic',
    'get_index_daily',
    'get_index_weight',
    'get_index_member',
    
    # 全球指数采集器类
    'GlobalIndexCollector',
    # 全球指数便捷函数
    'get_index_global',
]
