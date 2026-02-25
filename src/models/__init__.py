"""
模型层

包含 LightGBM 和 GRU 双轨模型的训练、推断与评估组件
"""

from .base_model import BaseModel, ModelEnsemble
from .config import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_GRU_CONFIG,
    DEFAULT_LABEL_CONFIG,
    DEFAULT_LGB_CONFIG,
    DEFAULT_SPLIT_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    AUX_COLS,
    CATEGORICAL_FEATURES,
    FeatureConfig,
    GRUConfig,
    GRUDataConfig,
    GRUInferenceConfig,
    GRUNetworkConfig,
    GRUSplitConfig,
    GRUTrainConfig,
    ID_COLS,
    InferenceConfig,
    LABEL_COLS,
    LabelConfig,
    LGBConfig,
    SplitConfig,
    SplitMode,
    TargetType,
    TrainConfig,
    get_gru_feature_columns,
    get_gru_selected_features,
)

__all__ = [
    # 基类
    "BaseModel",
    "ModelEnsemble",
    # LGB 配置
    "LGBConfig",
    "SplitConfig",
    "FeatureConfig",
    "LabelConfig",
    "TrainConfig",
    "InferenceConfig",
    "SplitMode",
    "TargetType",
    # GRU 配置
    "GRUConfig",
    "GRUDataConfig",
    "GRUSplitConfig",
    "GRUNetworkConfig",
    "GRUTrainConfig",
    "GRUInferenceConfig",
    "get_gru_feature_columns",
    "get_gru_selected_features",
    "ID_COLS",
    "LABEL_COLS",
    "AUX_COLS",
    "CATEGORICAL_FEATURES",
    # 默认配置
    "DEFAULT_LGB_CONFIG",
    "DEFAULT_SPLIT_CONFIG",
    "DEFAULT_FEATURE_CONFIG",
    "DEFAULT_LABEL_CONFIG",
    "DEFAULT_TRAIN_CONFIG",
    "DEFAULT_GRU_CONFIG",
]