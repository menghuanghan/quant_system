"""
模型层

包含 LightGBM 和 GRU 双轨模型的训练、推断与评估组件
"""

from .base_model import BaseModel, ModelEnsemble
from .config import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_LABEL_CONFIG,
    DEFAULT_LGB_CONFIG,
    DEFAULT_SPLIT_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    FeatureConfig,
    InferenceConfig,
    LabelConfig,
    LGBConfig,
    SplitConfig,
    SplitMode,
    TargetType,
    TrainConfig,
)

__all__ = [
    # 基类
    "BaseModel",
    "ModelEnsemble",
    # 配置
    "LGBConfig",
    "SplitConfig",
    "FeatureConfig",
    "LabelConfig",
    "TrainConfig",
    "InferenceConfig",
    "SplitMode",
    "TargetType",
    # 默认配置
    "DEFAULT_LGB_CONFIG",
    "DEFAULT_SPLIT_CONFIG",
    "DEFAULT_FEATURE_CONFIG",
    "DEFAULT_LABEL_CONFIG",
    "DEFAULT_TRAIN_CONFIG",
]