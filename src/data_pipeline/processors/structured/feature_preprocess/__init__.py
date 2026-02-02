"""
特征工程预处理模块

对 DWD 宽表数据进行清洗和预处理，为下游特征工程层提供高质量输入。

主要处理内容：
1. 极端值处理（Winsorization/Clipping）- 防止极端值压扁正常数据分布
2. 缺失值处理（倒数法）- PE→EP, PB→BP, PS→SP
3. 交易状态修正 - is_trading_final = (vol > 0) | (is_trading == 1)
4. 对数变换 - 对 total_mv, revenue_ttm, vol 等大数级字段取对数
5. 数据时滞过滤 - lag_days > 180 时标记为 NaN

使用 RAPIDS cuDF 进行 GPU 加速处理。
"""

from .config import PreprocessConfig
from .base import BasePreprocessor
from .price_preprocessor import PricePreprocessor
from .fundamental_preprocessor import FundamentalPreprocessor
from .status_preprocessor import StatusPreprocessor

__all__ = [
    "PreprocessConfig",
    "BasePreprocessor",
    "PricePreprocessor",
    "FundamentalPreprocessor",
    "StatusPreprocessor",
]
