"""
LightGBM 模型配置

定义模型超参数、GPU 加速设置、防过拟合参数等。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[3]  # src/models/LBGM -> quant_system


@dataclass
class DataConfig:
    """数据配置"""
    
    # 数据路径
    data_path: Path = BASE_DIR / "data" / "features" / "structured" / "train.parquet"
    
    # 模型输出路径
    output_dir: Path = BASE_DIR / "models" / "lgbm"
    
    # 目标列（回归使用 ret_5d，保留涨跌幅度信息利于排序）
    target_col: str = "ret_5d"
    
    # 排除列（不作为特征）
    exclude_cols: List[str] = field(default_factory=lambda: [
        "ts_code", "trade_date",
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "label_1d", "label_5d", "label_10d", "label_20d",
        "lag_days",  # 财报时滞特征，实验性排除
    ])
    
    # 类别特征（LightGBM 原生支持，无需 One-Hot）
    categorical_features: List[str] = field(default_factory=lambda: [
        "market",  # 主板/创业板/科创板/北交所
    ])
    
    # 时间切分配置 (Time-Series Split)
    train_start: str = "2021-01-01"
    train_end: str = "2023-12-31"
    valid_start: str = "2024-01-01"
    valid_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"
    
    # Purging: 验证集/测试集与前一个集合之间的间隔天数
    # 防止预测窗口（如 5 天）的标签重叠泄露
    purge_days: int = 5


@dataclass
class LGBMParams:
    """LightGBM 超参数配置"""
    
    # 核心参数
    task: str = "train"
    boosting_type: str = "gbdt"
    objective: str = "regression"
    metric: str = "None"  # 使用自定义 IC 评价函数
    
    # GPU 加速 (关键) - 使用 CUDA 后端
    device: str = "cuda"
    gpu_platform_id: int = 0
    gpu_device_id: int = 0
    
    # 树结构参数 (防过拟合)
    num_leaves: int = 63  # 叶子数，太大会过拟合噪音
    max_depth: int = 6    # 树深度
    min_data_in_leaf: int = 100  # 叶子最少样本数
    
    # 学习参数
    learning_rate: float = 0.03  # 学习率，小一点，多跑几轮
    n_estimators: int = 2000     # 最大迭代轮数
    
    # 正则化 (防过拟合)
    lambda_l1: float = 0.1      # L1 正则
    lambda_l2: float = 0.5      # L2 正则
    feature_fraction: float = 0.8  # 每棵树随机选 80% 特征
    bagging_fraction: float = 0.8  # 每棵树随机选 80% 样本
    bagging_freq: int = 1
    
    # 其他
    verbose: int = -1  # 静默模式
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task": self.task,
            "boosting_type": self.boosting_type,
            "objective": self.objective,
            "metric": self.metric,
            "device": self.device,
            "gpu_platform_id": self.gpu_platform_id,
            "gpu_device_id": self.gpu_device_id,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": self.learning_rate,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "verbose": self.verbose,
            "seed": self.seed,
        }


@dataclass
class TrainConfig:
    """训练配置"""
    
    # 迭代控制
    num_boost_round: int = 2000
    early_stopping_rounds: int = 50  # 如果 IC 在 50 轮内没提升就停止
    
    # 日志
    verbose_eval: int = 100  # 每 100 轮打印一次
    
    # 模型保存
    save_model: bool = True
    model_name: str = "lgbm_ret5d"


@dataclass
class LGBMConfig:
    """LightGBM 总配置"""
    
    data: DataConfig = field(default_factory=DataConfig)
    params: LGBMParams = field(default_factory=LGBMParams)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    def __post_init__(self):
        """初始化后创建输出目录"""
        self.data.output_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def default(cls) -> "LGBMConfig":
        """创建默认配置"""
        return cls()
    
    @classmethod
    def cpu_mode(cls) -> "LGBMConfig":
        """创建 CPU 模式配置"""
        config = cls()
        config.params.device = "cpu"
        return config
