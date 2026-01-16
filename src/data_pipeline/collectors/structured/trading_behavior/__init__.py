"""
资金与交易行为数据域（Trading Behavior）采集模块

本模块提供资金与交易行为数据的采集功能，涵盖以下数据类型：

1. 资金流向（Capital Flow）：
   - 个股资金流向
   - 行业/板块资金流向
   - 大盘资金流向
   - 沪深港通资金

2. 融资融券与杠杆行为：
   - 融资融券汇总/明细
   - 两融标的
   - 转融通

3. 特殊交易行为：
   - 龙虎榜
   - 大宗交易
   - 营业部行为

数据源优先级：Tushare > AkShare > BaoStock
"""

# 基类和工具（从父目录导入）
from ..base import (
    BaseCollector,
    DataSource,
    DataSourceManager,
    DataSourcePriority,
    CollectorRegistry,
    StandardFields,
    retry_on_failure,
    fallback_on_error,
)

# 资金流向采集器
from .capital_flow import (
    # 采集器类
    MoneyFlowCollector,
    MoneyFlowIndustryCollector,
    MoneyFlowMarketCollector,
    HSGTFlowCollector,
    # 便捷函数
    get_money_flow,
    get_money_flow_industry,
    get_money_flow_market,
    get_hsgt_flow,
)

# 融资融券采集器
from .margin_trading import (
    # 采集器类
    MarginSummaryCollector,
    MarginDetailCollector,
    MarginTargetCollector,
    SLBCollector,
    # 便捷函数
    get_margin_summary,
    get_margin_detail,
    get_margin_target,
    get_slb,
)

# 特殊交易行为采集器
from .special_trading import (
    # 采集器类
    TopListCollector,
    TopInstCollector,
    BlockTradeCollector,
    # 便捷函数
    get_top_list,
    get_top_inst,
    get_block_trade,
)


__all__ = [
    # 基类和工具
    'BaseCollector',
    'DataSource',
    'DataSourceManager',
    'DataSourcePriority',
    'CollectorRegistry',
    'StandardFields',
    'retry_on_failure',
    'fallback_on_error',
    
    # 资金流向
    'MoneyFlowCollector',
    'MoneyFlowIndustryCollector',
    'MoneyFlowMarketCollector',
    'HSGTFlowCollector',
    'get_money_flow',
    'get_money_flow_industry',
    'get_money_flow_market',
    'get_hsgt_flow',
    
    # 融资融券
    'MarginSummaryCollector',
    'MarginDetailCollector',
    'MarginTargetCollector',
    'SLBCollector',
    'get_margin_summary',
    'get_margin_detail',
    'get_margin_target',
    'get_slb',
    
    # 特殊交易行为
    'TopListCollector',
    'TopInstCollector',
    'BlockTradeCollector',
    'get_top_list',
    'get_top_inst',
    'get_block_trade',
]


# 版本信息
__version__ = '1.0.0'
__author__ = 'Quant Team'
