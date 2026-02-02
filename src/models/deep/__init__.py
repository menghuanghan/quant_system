"""
深度学习模型模块

核心组件：
- dataset.py: StockDataset 数据集类 (3D 张量 + 滑动窗口)
- model.py: GRUModel 模型架构
- loss.py: IC Loss 损失函数
- train.py: 训练循环 (混合精度 + 早停)
"""

from .dataset import StockDataset, prepare_data
from .model import GRUModel
from .loss import ICLoss, CombinedLoss
from .train import GRUTrainer

__all__ = [
    "StockDataset",
    "prepare_data",
    "GRUModel",
    "ICLoss",
    "CombinedLoss",
    "GRUTrainer",
]
