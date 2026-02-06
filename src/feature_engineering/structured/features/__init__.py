"""
特征生成模块

提供多个特征生成器：
- FeatureGenerator: 核心技术指标和基本面衍生特征
- MoneyFlowFeatureGenerator: 资金流特征
- ChipFeatureGenerator: 筹码结构特征
- RelativeStrengthGenerator: 相对强弱特征（需参考数据）
- IndexMemberGenerator: 指数成分特征（需参考数据）
- MacroInteractionGenerator: 宏观交互特征
- ReferenceDataLoader: 参考数据加载器
"""

from .feature_generator import FeatureGenerator
from .reference_data_loader import ReferenceDataLoader
from .money_flow_features import MoneyFlowFeatureGenerator
from .chip_features import ChipFeatureGenerator
from .relative_strength_features import RelativeStrengthGenerator, IndexMemberGenerator
from .macro_interaction_features import MacroInteractionGenerator

__all__ = [
    # 核心生成器
    "FeatureGenerator",
    # 参考数据
    "ReferenceDataLoader",
    # 扩展生成器
    "MoneyFlowFeatureGenerator",
    "ChipFeatureGenerator",
    "RelativeStrengthGenerator",
    "IndexMemberGenerator",
    "MacroInteractionGenerator",
]
