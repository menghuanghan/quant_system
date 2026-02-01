"""
结构化数据处理模块 - DWD层（明细数据层）- 纯cuDF GPU加速版本

本模块负责将原始数据进行清洗、对齐、合成，生成三张核心宽表：

1. dwd_stock_price - 基础量价宽表
2. dwd_stock_fundamental - PIT基本面宽表
3. dwd_stock_status - 状态与风险掩码表

核心原则：
- 主键: trade_date + ts_code
- PIT (Point-in-Time): 避免未来函数
- 全程cuDF GPU加速
"""

from .config import (
    DATA_SOURCE_PATHS,
    DWD_OUTPUT_CONFIG,
    MARKET_CONFIG,
    FUNDAMENTAL_CONFIG,
    DWD_FIELDS_CONFIG,
    PROCESSING_CONFIG,
)

from .base import (
    BaseProcessor,
    calculate_vwap_gpu,
    ffill_by_group_gpu,
)

from .market_data_processor import MarketDataProcessor
from .fundamental_processor import FundamentalProcessor
from .status_processor import StatusProcessor

__all__ = [
    # 配置
    'DATA_SOURCE_PATHS',
    'DWD_OUTPUT_CONFIG',
    'MARKET_CONFIG',
    'FUNDAMENTAL_CONFIG',
    'DWD_FIELDS_CONFIG',
    'PROCESSING_CONFIG',
    # 基类和工具
    'BaseProcessor',
    'calculate_vwap_gpu',
    'ffill_by_group_gpu',
    # 处理器
    'MarketDataProcessor',
    'FundamentalProcessor',
    'StatusProcessor',
]
