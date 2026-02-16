"""
深度学习模型模块

核心组件：
- config.py: GRUConfig 模型配置
- dataset.py: StockDataset 数据集类 (3D 张量 + 滑动窗口)
- model.py: GRUModel 模型架构
- loss.py: IC Loss 损失函数
- train.py: 训练循环 (混合精度 + 早停)
"""

# 延迟导入以避免不必要的依赖加载
def __getattr__(name):
    if name == "GRUConfig":
        from .config import GRUConfig
        return GRUConfig
    elif name == "StockDataset":
        from .dataset import StockDataset
        return StockDataset
    elif name == "prepare_data":
        from .dataset import prepare_data
        return prepare_data
    elif name == "GRUModel":
        from .model import GRUModel
        return GRUModel
    elif name == "ICLoss":
        from .loss import ICLoss
        return ICLoss
    elif name == "CombinedLoss":
        from .loss import CombinedLoss
        return CombinedLoss
    elif name == "GRUTrainer":
        from .train import GRUTrainer
        return GRUTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GRUConfig",
    "StockDataset",
    "prepare_data",
    "GRUModel",
    "ICLoss",
    "CombinedLoss",
    "GRUTrainer",
]
