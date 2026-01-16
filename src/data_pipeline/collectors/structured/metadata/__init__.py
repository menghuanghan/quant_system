"""
结构化数据采集模块 - 基础元数据域（Metadata Domain）

本模块提供原始数据采集功能，涵盖以下数据类型：

1. 证券与标的基础信息（Security Master）：
   - 股票列表（A股/港股/美股）
   - 股票曾用名、代码变更
   - ST/*ST/风险警示标识
   - A+H股票

2. 交易日历与制度信息（Trading Calendar）：
   - 交易日历（股票/期货/港股/美股）
   - 停复牌信息
   - 涨跌停规则
   - 集合竞价时间

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

# 证券基础信息采集器
from .security_master import (
    # 采集器类
    StockListACollector,
    StockListHKCollector,
    StockListUSCollector,
    NameChangeCollector,
    STStatusCollector,
    AHStockCollector,
    # 便捷函数
    get_stock_list_a,
    get_stock_list_hk,
    get_stock_list_us,
    get_name_change,
    get_st_status,
    get_ah_stock,
)

# 交易日历采集器
from .trading_calendar import (
    # 采集器类
    TradeCalendarCollector,
    SuspendInfoCollector,
    PriceLimitRuleCollector,
    AuctionTimeCollector,
    HKTradeCalendarCollector,
    USTradeCalendarCollector,
    FuturesTradeCalendarCollector,
    # 便捷函数
    get_trade_calendar,
    get_suspend_info,
    get_price_limit_rule,
    get_auction_time,
    get_trade_dates,
    is_trade_date,
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
    
    # 证券基础信息采集器类
    'StockListACollector',
    'StockListHKCollector',
    'StockListUSCollector',
    'NameChangeCollector',
    'STStatusCollector',
    'AHStockCollector',
    
    # 证券基础信息便捷函数
    'get_stock_list_a',
    'get_stock_list_hk',
    'get_stock_list_us',
    'get_name_change',
    'get_st_status',
    'get_ah_stock',
    
    # 交易日历采集器类
    'TradeCalendarCollector',
    'SuspendInfoCollector',
    'PriceLimitRuleCollector',
    'AuctionTimeCollector',
    'HKTradeCalendarCollector',
    'USTradeCalendarCollector',
    'FuturesTradeCalendarCollector',
    
    # 交易日历便捷函数
    'get_trade_calendar',
    'get_suspend_info',
    'get_price_limit_rule',
    'get_auction_time',
    'get_trade_dates',
    'is_trade_date',
]


# 版本信息
__version__ = '1.0.0'
__author__ = 'Quant Team'
