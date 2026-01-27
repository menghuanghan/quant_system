"""
市场行情数据域（Market Data）采集模块

本模块提供市场行情数据的采集功能，涵盖以下数据类型：

1. K线与价格序列（Price & OHLCV）：
   - 股票日/周/月K线（前复权/后复权/不复权）
   - 指数日/周/月K线
   - ETF日/周/月K线

2. 实时与准实时行情（Realtime Market）：
   - 实时日线行情
   - 实时分钟行情
   - 实时涨跌幅/排名

3. 技术指标与衍生行情特征：
   - 常规技术指标（MA/RSI/MACD）
   - 每日基本指标（市盈率/市净率/换手率等）
   - 创新高/连续涨跌/放量缩量

4. 复权因子（Adjustment Factor）：
   - 股票复权因子（用于计算前复权/后复权价格）

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

# K线与价格序列采集器
from .price_kline import (
    # 采集器类
    StockDailyCollector,
    StockWeeklyCollector,
    StockMonthlyCollector,
    IndexDailyCollector,
    ETFDailyCollector,
    # 便捷函数
    get_stock_daily,
    get_stock_weekly,
    get_stock_monthly,
    get_index_daily,
    get_etf_daily,
)

# 实时行情采集器
from .realtime import (
    # 采集器类
    RealtimeQuoteCollector,
    TopListCollector,
    # 便捷函数
    get_realtime_quote,
    get_top_list,
)

# 技术指标采集器
from .technical import (
    # 采集器类
    DailyBasicCollector,
    TechnicalIndicatorCollector,
    StkFactorCollector,
    # 便捷函数
    get_daily_basic,
    get_technical_indicator,
    get_stk_factor,
)

# 复权因子采集器
from .adj_factor import (
    # 采集器类
    AdjFactorCollector,
    # 便捷函数
    get_adj_factor,
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
    
    # K线采集器类
    'StockDailyCollector',
    'StockWeeklyCollector',
    'StockMonthlyCollector',
    'IndexDailyCollector',
    'ETFDailyCollector',
    
    # K线便捷函数
    'get_stock_daily',
    'get_stock_weekly',
    'get_stock_monthly',
    'get_index_daily',
    'get_etf_daily',
    
    # 实时行情采集器类
    'RealtimeQuoteCollector',
    'TopListCollector',
    
    # 实时行情便捷函数
    'get_realtime_quote',
    'get_top_list',
    
    # 技术指标采集器类
    'DailyBasicCollector',
    'TechnicalIndicatorCollector',
    'StkFactorCollector',
    
    # 技术指标便捷函数
    'get_daily_basic',
    'get_technical_indicator',
    'get_stk_factor',
    
    # 复权因子采集器类
    'AdjFactorCollector',
    
    # 复权因子便捷函数
    'get_adj_factor',
]


# 版本信息
__version__ = '1.0.0'
__author__ = 'Quant Team'
