"""
模型超参数和全局配置
包含 LightGBM 和 GRU 模型的默认参数配置
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import os


# ====================== 路径配置 ======================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features" / "structured"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs" / "models"

# 确保目录存在
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ====================== 枚举定义 ======================

class SplitMode(Enum):
    """时序切分模式"""
    ROLLING = "rolling"           # 滚动窗口
    EXPANDING = "expanding"       # 扩展窗口
    SINGLE_FULL = "single_full"   # 单次划分（全量重训）


class TargetType(Enum):
    """标签类型"""
    REGRESSION = "regression"     # 回归（绝对/超额收益）
    RANK = "rank"                 # 排序（秩标签）
    CLASSIFICATION = "classification"  # 分类（二元标签）


# ====================== 特征配置 ======================

@dataclass
class FeatureConfig:
    """特征相关配置"""
    
    # 主键列（不参与训练）
    key_columns: List[str] = field(default_factory=lambda: [
        "trade_date", 
        "ts_code"
    ])
    
    # 类别特征列（直接传给 LightGBM 做类别分裂）
    category_columns: List[str] = field(default_factory=lambda: [
        "industry_idx",
        "sw_l1_idx", 
        "sw_l2_idx",
        "market",
        "pmi_regime",
        "lpr_trend",
        "money_regime",
        "risk_appetite",
        "macro_regime",
    ])
    
    # 标签列前缀（训练时需要排除的列）
    label_prefixes: List[str] = field(default_factory=lambda: [
        "label_",
        "ret_",
        "excess_ret_", 
        "rank_ret_",
        "sharpe_",
    ])
    
    # 默认训练标签列表
    default_target_cols: List[str] = field(default_factory=lambda: [
        "rank_ret_5d",      # 5日收益排序
        "excess_ret_10d",   # 10日超额收益
        "sharpe_20d",       # 20日夏普
    ])


# ====================== LightGBM 配置 ======================

@dataclass
class LGBConfig:
    """LightGBM 模型配置"""
    
    # 基础参数
    seed: int = 42
    num_boost_round: int = 2000
    early_stopping_rounds: int = 50
    verbose_eval: int = 100
    
    # 核心防过拟合参数
    max_depth: int = 6                    # 树深度 4~7
    num_leaves: int = 31                  # 叶子数 15~63
    min_data_in_leaf: int = 300           # 叶子最小样本数 200~500
    feature_fraction: float = 0.7         # 列抽样 0.6~0.8
    bagging_fraction: float = 0.8         # 行抽样 0.7~0.9
    bagging_freq: int = 1                 # bagging 频率
    
    # 学习率
    learning_rate: float = 0.05
    
    # 正则化
    lambda_l1: float = 0.1
    lambda_l2: float = 1.0
    
    # 目标函数（动态根据标签类型选择）
    objective_regression: str = "huber"   # 连续标签用 huber
    objective_rank: str = "regression"    # 秩标签用 mse
    huber_delta: float = 1.0              # huber 损失参数
    
    # GPU 加速（使用 CUDA）
    device: str = "cuda"
    gpu_platform_id: int = 0
    gpu_device_id: int = 0
    
    # 其他
    n_jobs: int = -1
    importance_type: str = "gain"         # 特征重要性类型
    
    def to_lgb_params(self, target_type: TargetType = TargetType.REGRESSION) -> Dict[str, Any]:
        """
        转换为 LightGBM 原生参数字典
        
        Args:
            target_type: 标签类型，决定目标函数
            
        Returns:
            params: LightGBM 参数字典
        """
        params = {
            "boosting_type": "gbdt",
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "learning_rate": self.learning_rate,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "seed": self.seed,
            "verbose": -1,
            "n_jobs": self.n_jobs,
        }
        
        # 根据标签类型选择目标函数
        if target_type == TargetType.RANK:
            params["objective"] = self.objective_rank
            params["metric"] = "mse"
        elif target_type == TargetType.CLASSIFICATION:
            params["objective"] = "binary"
            params["metric"] = "auc"
        else:
            # 回归（绝对/超额收益）
            params["objective"] = self.objective_regression
            if self.objective_regression == "huber":
                params["huber_delta"] = self.huber_delta
            params["metric"] = "huber" if self.objective_regression == "huber" else "mse"
        
        # GPU 配置（支持 cuda 和 gpu 两种模式）
        if self.device in ("cuda", "gpu"):
            params["device"] = self.device
            params["gpu_platform_id"] = self.gpu_platform_id
            params["gpu_device_id"] = self.gpu_device_id
        
        return params


# ====================== 时序切分配置 ======================

@dataclass
class SplitConfig:
    """时序切分配置"""
    
    # 窗口参数（单位：月）
    train_window_months: int = 24         # 训练窗口长度
    valid_window_months: int = 3          # 验证窗口长度
    step_months: int = 3                  # 滑动步长
    
    # 切分模式
    mode: SplitMode = SplitMode.ROLLING
    
    # 数据时间范围
    data_start_date: str = "2021-01-01"
    data_end_date: str = "2025-12-31"
    
    # 标签泄露防护：gap_days 将从 target_col 自动解析


# ====================== 标签处理配置 ======================

@dataclass  
class LabelConfig:
    """标签处理配置"""
    
    # 需要做截面 Z-Score 的标签前缀
    zscore_prefixes: List[str] = field(default_factory=lambda: [
        "ret_",
        "excess_ret_",
        "sharpe_",
    ])
    
    # 不做标准化的标签前缀（秩/分类）
    skip_normalize_prefixes: List[str] = field(default_factory=lambda: [
        "rank_",
        "label_bin_",
    ])
    
    # 去极值 clip 范围
    clip_min: float = -3.0
    clip_max: float = 3.0


# ====================== 训练配置 ======================

@dataclass
class TrainConfig:
    """训练总体配置"""
    
    # 组件配置
    lgb_config: LGBConfig = field(default_factory=LGBConfig)
    split_config: SplitConfig = field(default_factory=SplitConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    label_config: LabelConfig = field(default_factory=LabelConfig)
    
    # 输入路径
    train_data_path: Path = FEATURES_DIR / "train_lgb.parquet"
    
    # 输出路径
    model_save_dir: Path = MODELS_DIR / "lgb"
    oof_save_path: Path = MODELS_DIR / "lgb" / "oof_predictions.parquet"
    
    # 训练参数
    use_gpu_dataframe: bool = True  # 是否使用 cuDF 加速数据处理


# ====================== 实盘推断配置 ======================

@dataclass
class InferenceConfig:
    """实盘推断配置"""
    
    # 模型权重
    rolling_weight: float = 0.4      # Rolling 模型群权重
    full_weight: float = 0.6         # Single Full 模型权重
    
    # 多目标融合权重（可选：等权 or 协方差加权）
    target_weights: Optional[Dict[str, float]] = None
    
    # 模型路径
    rolling_models_dir: Path = MODELS_DIR / "lgb" / "rolling"
    full_model_path: Path = MODELS_DIR / "lgb" / "full" / "lgb_full.pkl"


# ====================== 默认实例 ======================

DEFAULT_LGB_CONFIG = LGBConfig()
DEFAULT_SPLIT_CONFIG = SplitConfig()
DEFAULT_FEATURE_CONFIG = FeatureConfig()
DEFAULT_LABEL_CONFIG = LabelConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()