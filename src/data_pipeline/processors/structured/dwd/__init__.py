"""
结构化数据处理模块 - DWD层（明细数据层）- 纯cuDF GPU加速版本

本模块负责将原始数据进行清洗、对齐、合成，生成核心宽表：

核心宽表（已实现）：
1. dwd_stock_price - 基础量价宽表
2. dwd_stock_fundamental - PIT基本面宽表
3. dwd_stock_status - 状态与风险掩码表

扩展宽表（新增）：
4. dwd_money_flow - 资金博弈宽表（主力资金与筹码）
5. dwd_chip_structure - 筹码结构宽表（十大股东、股东户数）
6. dwd_stock_industry - 行业分类宽表（申万行业分类）
7. dwd_event_signal - 事件信号宽表（回购、解禁、质押、分红）
8. dwd_macro_env - 宏观环境宽表（GDP、CPI、PMI、利率等）

核心原则：
- 主键: trade_date + ts_code（宏观表除外，仅 trade_date）
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

# 核心宽表处理器
from .market_data_processor import MarketDataProcessor
from .fundamental_processor import FundamentalProcessor
from .status_processor import StatusProcessor

# 扩展宽表处理器
from .money_flow_processor import MoneyFlowProcessor
from .chip_structure_processor import ChipStructureProcessor
from .industry_processor import IndustryProcessor
from .event_signal_processor import EventSignalProcessor
from .macro_env_processor import MacroEnvProcessor

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
    # 核心处理器
    'MarketDataProcessor',
    'FundamentalProcessor',
    'StatusProcessor',
    # 扩展处理器
    'MoneyFlowProcessor',
    'ChipStructureProcessor',
    'IndustryProcessor',
    'EventSignalProcessor',
    'MacroEnvProcessor',
]
