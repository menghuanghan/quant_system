"""
深度风险与质量因子域（Deep Risk & Quality Factors）采集模块

本模块提供深度风险与质量因子相关数据的采集功能，包括：
- 估值扩散与拥挤度
- 资产质量异常
- ESG评价

所有采集器均遵循统一的接口规范，支持多数据源自动切换（Tushare → AkShare → BaoStock）
"""

# 估值扩散与拥挤度模块
from .valuation_dispersion import (
    # 采集器类
    APEPBEWMedianCollector,
    MarketCongestionCollector,
    StockBondSpreadCollector,
    BuffettIndicatorCollector,
    
    # 便捷函数
    get_a_pe_pb_ew_median,
    get_market_congestion,
    get_stock_bond_spread,
    get_buffett_indicator,
)

# 资产质量异常模块
from .asset_quality import (
    # 采集器类
    StockGoodwillCollector,
    GoodwillImpairmentCollector,
    BreakNetStockCollector,
    
    # 便捷函数
    get_stock_goodwill,
    get_goodwill_impairment,
    get_break_net_stock,
)

# ESG评级模块
from .esg_ratings import (
    # 采集器类
    MSCIESGCollector,
    HZESGCollector,
    RefinitivESGCollector,
    ZhidingESGCollector,
    
    # 便捷函数
    get_esg_msci,
    get_esg_hz,
    get_esg_refinitiv,
    get_esg_zhiding,
)

__all__ = [
    # 估值扩散与拥挤度
    'APEPBEWMedianCollector',
    'MarketCongestionCollector',
    'StockBondSpreadCollector',
    'BuffettIndicatorCollector',
    'get_a_pe_pb_ew_median',
    'get_market_congestion',
    'get_stock_bond_spread',
    'get_buffett_indicator',
    
    # 资产质量异常
    'StockGoodwillCollector',
    'GoodwillImpairmentCollector',
    'BreakNetStockCollector',
    'get_stock_goodwill',
    'get_goodwill_impairment',
    'get_break_net_stock',
    
    # ESG评级
    'MSCIESGCollector',
    'HZESGCollector',
    'RefinitivESGCollector',
    'ZhidingESGCollector',
    'get_esg_msci',
    'get_esg_hz',
    'get_esg_refinitiv',
    'get_esg_zhiding',
]
