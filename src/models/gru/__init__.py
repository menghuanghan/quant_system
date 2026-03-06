"""
GRU 多任务深度学习模型模块

核心组件:
- 配置统一由 src/models/config.py 管理 (GRUConfig 等)
- dataset.py: 时序切分器 (GRUTimeSeriesSplitter) + 3D 张量构造器 (GRUTensorDataset)
- network.py: 多任务 GRU 网络 (MultiTaskGRUNetwork)
- gru_model.py: 训练模型 (GRUModel, 继承 BaseModel) + 多任务损失函数
- trainer.py: 训练调度器 (GRUTrainer) + 推断引擎 (GRUInferenceEngine)
- 模型评估复用 src/models/metrics/evaluator.py (QuantEvaluator)
"""

from ..config import (
    GRUConfig,
    GRUDataConfig,
    GRUNetworkConfig,
    GRUSplitConfig,
    GRUTrainConfig,
    GRUInferenceConfig,
    get_gru_feature_columns,
    get_gru_selected_features,
    ID_COLS,
    LABEL_COLS,
    AUX_COLS,
    CATEGORICAL_FEATURES,
)
from .dataset import (
    GRUFoldInfo,
    GRUTimeSeriesSplitter,
    GRUTensorDataset,
    create_dataloader,
)
from .network import MultiTaskGRUNetwork, TemporalAttention
from .gru_model import (
    GRUModel,
    MultiTaskLoss,
    UncertaintyWeightedLoss,
    set_seed,
)
from .trainer import GRUTrainer, GRUInferenceEngine
from .report_generator import GRUReportGenerator

__all__ = [
    # Config (from parent config.py)
    "GRUConfig",
    "GRUDataConfig",
    "GRUNetworkConfig",
    "GRUSplitConfig",
    "GRUTrainConfig",
    "GRUInferenceConfig",
    "get_gru_feature_columns",
    "get_gru_selected_features",
    "ID_COLS",
    "LABEL_COLS",
    "AUX_COLS",
    "CATEGORICAL_FEATURES",
    # Dataset
    "GRUFoldInfo",
    "GRUTimeSeriesSplitter",
    "GRUTensorDataset",
    "create_dataloader",
    # Network
    "MultiTaskGRUNetwork",
    "TemporalAttention",
    # Model + Loss
    "GRUModel",
    "MultiTaskLoss",
    "UncertaintyWeightedLoss",
    "set_seed",
    # Trainer
    "GRUTrainer",
    "GRUInferenceEngine",
    # Report
    "GRUReportGenerator",
]
