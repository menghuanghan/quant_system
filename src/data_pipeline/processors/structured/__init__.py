"""
结构化数据处理模块

本模块包含 DWD（明细数据层）处理器，负责将原始数据清洗、对齐、合成为三张核心宽表。

向后兼容：从 dwd 子模块重新导出核心类
"""

from .dwd import (
    # 配置
    DATA_SOURCE_PATHS,
    DWD_OUTPUT_CONFIG,
    MARKET_CONFIG,
    FUNDAMENTAL_CONFIG,
    DWD_FIELDS_CONFIG,
    PROCESSING_CONFIG,
    # 基类和工具
    BaseProcessor,
    calculate_vwap_gpu,
    ffill_by_group_gpu,
    # 处理器
    MarketDataProcessor,
    FundamentalProcessor,
    StatusProcessor,
)

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
