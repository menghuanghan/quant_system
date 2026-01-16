"""
宏观与外生变量数据域（Macro & Exogenous）采集模块

数据类型包括：
1. 国内宏观：
   - GDP/CPI/PPI
   - PMI
   - 社融/货币供应
   - 利率体系（Shibor/LPR）

2. 国际宏观：
   - 美/欧/日宏观指标
   - 国债收益率
   - 大宗商品宏观

3. 行业与现实经济映射：
   - 能源/生猪/物流
   - 电影票房
   - 汽车销量

使用方法:
    from src.data_pipeline.collectors.structured.macro_exogenous import (
        # 国内宏观
        get_cn_gdp,
        get_cn_cpi,
        get_cn_ppi,
        get_cn_pmi,
        get_cn_m2,
        get_shibor,
        get_lpr,
        get_sf,
        
        # 国际宏观
        get_us_treasury,
        get_eco_calendar,
        
        # 行业经济
        get_box_office,
        get_car_sales,
    )
"""

# 国内宏观数据采集器
from .china_macro import (
    ChinaGDPCollector,
    ChinaCPICollector,
    ChinaPPICollector,
    ChinaPMICollector,
    ChinaMoneySupplyCollector,
    ShiborCollector,
    LPRCollector,
    SocialFinanceCollector,
    get_cn_gdp,
    get_cn_cpi,
    get_cn_ppi,
    get_cn_pmi,
    get_cn_m2,
    get_shibor,
    get_lpr,
    get_sf,
)

# 国际宏观数据采集器
from .global_macro import (
    USTreasuryCollector,
    EcoCalendarCollector,
    get_us_treasury,
    get_eco_calendar,
)

# 行业经济数据采集器
from .industry_economy import (
    BoxOfficeCollector,
    CarSalesCollector,
    get_box_office,
    get_car_sales,
)

__all__ = [
    # 国内宏观采集器类
    'ChinaGDPCollector',
    'ChinaCPICollector',
    'ChinaPPICollector',
    'ChinaPMICollector',
    'ChinaMoneySupplyCollector',
    'ShiborCollector',
    'LPRCollector',
    'SocialFinanceCollector',
    # 国内宏观便捷函数
    'get_cn_gdp',
    'get_cn_cpi',
    'get_cn_ppi',
    'get_cn_pmi',
    'get_cn_m2',
    'get_shibor',
    'get_lpr',
    'get_sf',
    
    # 国际宏观采集器类
    'USTreasuryCollector',
    'EcoCalendarCollector',
    # 国际宏观便捷函数
    'get_us_treasury',
    'get_eco_calendar',
    
    # 行业经济采集器类
    'BoxOfficeCollector',
    'CarSalesCollector',
    # 行业经济便捷函数
    'get_box_office',
    'get_car_sales',
]
