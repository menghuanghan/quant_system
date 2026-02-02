"""
LightGBM 量化因子模型模块

包含以下核心组件：
- DataLoader: 数据加载与时间序列切分
- Metrics: 自定义评价函数 (IC, RankIC)
- Config: 模型配置与超参数
- Trainer: 训练循环与 Early Stopping
- Analysis: 特征重要性与 SHAP 分析
"""

from .config import LGBMConfig
from .data_loader import DataLoader
from .metrics import ic_eval, rank_ic
from .trainer import LGBMTrainer
from .analysis import ModelAnalyzer

__all__ = [
    "LGBMConfig",
    "DataLoader",
    "ic_eval",
    "rank_ic",
    "LGBMTrainer",
    "ModelAnalyzer",
]
