"""
LightGBM 量化因子模型模块

包含以下核心组件：
- DataLoader: 数据加载与时间序列切分
- Metrics: 自定义评价函数 (IC, RankIC)
- Config: 模型配置与超参数
- Trainer: 训练循环与 Early Stopping
- Analysis: 特征重要性与 SHAP 分析
"""

# 延迟导入以避免不必要的依赖加载
def __getattr__(name):
    if name == "LGBMConfig":
        from .config import LGBMConfig
        return LGBMConfig
    elif name == "DataLoader":
        from .data_loader import DataLoader
        return DataLoader
    elif name == "ic_eval":
        from .metrics import ic_eval
        return ic_eval
    elif name == "rank_ic":
        from .metrics import rank_ic
        return rank_ic
    elif name == "LGBMTrainer":
        from .trainer import LGBMTrainer
        return LGBMTrainer
    elif name == "ModelAnalyzer":
        from .analysis import ModelAnalyzer
        return ModelAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LGBMConfig",
    "DataLoader",
    "ic_eval",
    "rank_ic",
    "LGBMTrainer",
    "ModelAnalyzer",
]
