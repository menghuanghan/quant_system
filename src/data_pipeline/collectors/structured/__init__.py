"""
结构化数据采集模块

本模块按照数据域划分为不同的子模块：
- metadata: 基础元数据域（Metadata Domain）
- market_data: 市场行情数据域（Market Data Domain）
- fundamental: 公司基本面数据域（Fundamental Data Domain）
- trading_behavior: 资金与交易行为数据域（Trading Behavior Domain）
- cross_sectional: 板块/行业/主题数据域（Cross-sectional Structure Domain）
- derivatives: 衍生品与多资产数据域（Derivatives & Multi-Asset Domain）
- index_benchmark: 指数与基准数据域（Index & Benchmark Domain）
- macro_exogenous: 宏观与高频外生变量域（Macro & Exogenous Domain）
- expectations: 预期与预测分析域（Expectations & Forecasts Domain）
- deep_risk_quality: 深度风险与质量因子域（Deep Risk & Quality Factors Domain）
"""

# 导入基类和工具
from .base import (
    BaseCollector,
    DataSource,
    DataSourceManager,
    DataSourcePriority,
    CollectorRegistry,
    StandardFields,
    retry_on_failure,
    fallback_on_error,
)

# 导入基础元数据域模块
from . import metadata

# 导入市场行情数据域模块
from . import market_data

# 导入公司基本面数据域模块
from . import fundamental

# 导入资金与交易行为数据域模块
from . import trading_behavior

# 导入板块/行业/主题数据域模块
from . import cross_sectional

# 导入衍生品与多资产数据域模块
from . import derivatives

# 导入指数与基准数据域模块
from . import index_benchmark

# 导入宏观与外生变量数据域模块
from . import macro_exogenous

# 导入预期与预测分析域模块
from . import expectations

# 导入深度风险与质量因子域模块
from . import deep_risk_quality

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
    
    # 子模块
    'metadata',
    'market_data',
    'fundamental',
    'trading_behavior',
    'cross_sectional',
    'derivatives',
    'index_benchmark',
    'macro_exogenous',
    'expectations',
    'deep_risk_quality',
]


__version__ = '1.0.0'
__author__ = 'Quant Team'
