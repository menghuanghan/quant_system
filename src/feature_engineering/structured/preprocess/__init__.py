"""
特征工程预处理模块

对 DWD 宽表数据进行清洗和预处理，为下游特征工程层提供高质量输入。

主要处理内容：
1. 极端值处理（Winsorization/Clipping）- 防止极端值压扁正常数据分布
2. 缺失值处理（倒数法）- PE→EP, PB→BP, PS→SP
3. 交易状态修正 - is_trading_final = (vol > 0) | (is_trading == 1)
4. 单位统一 - 所有金额字段统一为元
5. 数据时滞过滤 - lag_days > 180 时标记为 NaN

注意：
- Preprocess 阶段不做 Log 变换（保持原始物理意义）
- Log 变换应在 Feature Transformation 阶段进行

使用 RAPIDS cuDF 进行 GPU 加速处理。
"""

from .config import PreprocessConfig
from .base import BasePreprocessor

# 核心表预处理器
from .price_preprocessor import PricePreprocessor
from .fundamental_preprocessor import FundamentalPreprocessor
from .status_preprocessor import StatusPreprocessor

# 扩展表预处理器
from .money_flow_preprocessor import MoneyFlowPreprocessor
from .chip_preprocessor import ChipPreprocessor
from .industry_preprocessor import IndustryPreprocessor
from .macro_preprocessor import MacroPreprocessor
from .event_preprocessor import EventPreprocessor

__all__ = [
    # 配置
    "PreprocessConfig",
    "BasePreprocessor",
    # 核心表
    "PricePreprocessor",
    "FundamentalPreprocessor",
    "StatusPreprocessor",
    # 扩展表
    "MoneyFlowPreprocessor",
    "ChipPreprocessor",
    "IndustryPreprocessor",
    "MacroPreprocessor",
    "EventPreprocessor",
]
