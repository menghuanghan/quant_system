"""
LightGBM 模型配置

定义模型超参数、GPU 加速设置、防过拟合参数等。

改造说明（2026.02）:
- 适配全域数据 train_lgb.parquet (310 列特征)
- 动态特征识别（排除法自动发现特征）
- 完善类别特征处理
- 支持多种标签目标
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import logging

logger = logging.getLogger(__name__)

# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[3]  # src/models/LBGM -> quant_system


# ============================================================================
# 列分类定义（核心配置）
# ============================================================================

# ID 列：用于定位样本，不作为特征
ID_COLS = ["ts_code", "trade_date"]

# 标签列：预测目标及相关衍生列，不作为特征
LABEL_COLS = [
    # 原始收益率
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    # 分类标签
    "label_1d", "label_5d", "label_10d", "label_20d",
    # 超额收益
    "excess_ret_5d", "excess_ret_10d",
    # 排名收益
    "rank_ret_5d", "rank_ret_10d",
    # 风险调整收益
    "sharpe_5d", "sharpe_10d", "sharpe_20d",
    # 二分类标签
    "label_bin_5d",
]

# 辅助/掩码列：用于过滤样本，不作为特征输入
AUX_COLS = [
    "is_tradable",  # 是否可交易
    "is_risky",     # 是否风险股票
    "is_limit",     # 是否涨跌停
    "lag_days",     # 财报时滞（实验性排除）
]

# 类别特征：LightGBM 原生支持，无需 One-Hot
CATEGORICAL_FEATURES = [
    "market",       # 主板/创业板/科创板/北交所 (int8)
    "industry_idx", # 行业索引 (int32)
    "sw_l1_idx",    # 申万一级行业 (int32)
    "sw_l2_idx",    # 申万二级行业 (int32)
]


def get_feature_columns(
    all_columns: List[str],
    exclude_cols: Optional[List[str]] = None,
    include_categorical: bool = True,
) -> List[str]:
    """
    动态识别特征列（排除法）
    
    策略：从所有列中排除 ID、标签、辅助列，剩余即为特征
    
    Args:
        all_columns: 数据集的所有列名
        exclude_cols: 额外需要排除的列
        include_categorical: 是否包含类别特征
        
    Returns:
        特征列名列表
    """
    # 构建排除集合
    exclude_set: Set[str] = set(ID_COLS + LABEL_COLS + AUX_COLS)
    
    if exclude_cols:
        exclude_set.update(exclude_cols)
    
    if not include_categorical:
        exclude_set.update(CATEGORICAL_FEATURES)
    
    # 排除法获取特征列
    feature_cols = [col for col in all_columns if col not in exclude_set]
    
    logger.info(f"动态特征识别: 总列数={len(all_columns)}, 排除={len(exclude_set)}, 特征={len(feature_cols)}")
    
    return feature_cols


def get_categorical_indices(feature_cols: List[str]) -> List[int]:
    """
    获取类别特征在特征列表中的索引
    
    Args:
        feature_cols: 特征列名列表
        
    Returns:
        类别特征的索引列表
    """
    indices = []
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in feature_cols:
            indices.append(feature_cols.index(cat_col))
    return indices


@dataclass
class DataConfig:
    """数据配置"""
    
    # 数据路径（使用新的全域数据）
    data_path: Path = BASE_DIR / "data" / "features" / "structured" / "train_lgb.parquet"
    
    # 模型输出路径
    output_dir: Path = BASE_DIR / "models" / "lgbm"
    
    # 目标列（推荐使用超额收益，消除市场影响）
    target_col: str = "excess_ret_5d"
    
    # 额外排除列（在默认排除基础上）
    extra_exclude_cols: List[str] = field(default_factory=list)
    
    # 类别特征（LightGBM 原生支持，无需 One-Hot）
    categorical_features: List[str] = field(default_factory=lambda: CATEGORICAL_FEATURES.copy())
    
    # 时间切分配置 (Time-Series Split)
    # 训练: 2021-2023 (3年), 验证: 2024 (1年), 测试: 2025 (1年)
    train_start: str = "2021-01-01"
    train_end: str = "2023-12-31"
    valid_start: str = "2024-01-01"
    valid_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"
    
    # Purging: 验证集/测试集与前一个集合之间的间隔天数
    # 防止预测窗口（如 5 天）的标签泄露
    purge_days: int = 5
    
    def get_exclude_cols(self) -> List[str]:
        """获取完整的排除列列表"""
        return ID_COLS + LABEL_COLS + AUX_COLS + self.extra_exclude_cols


@dataclass
class LGBMParams:
    """LightGBM 超参数配置"""
    
    # 核心参数
    task: str = "train"
    boosting_type: str = "gbdt"
    objective: str = "regression"  # 优化回归损失
    metric: str = "mae"  # 使用 MAE 作为辅助指标（比 MSE 更稳健）
    # 注意: 实际评价使用自定义 IC 函数 (trainer.py feval)
    
    # GPU 加速 (关键) - 使用 CUDA 后端
    device: str = "cuda"
    gpu_platform_id: int = 0
    gpu_device_id: int = 0
    
    # 树结构参数 (针对大特征集调优)
    num_leaves: int = 127      # 增加叶子数以适应更多特征
    max_depth: int = 8         # 适当增加深度
    min_data_in_leaf: int = 200  # 增加最小样本数防止过拟合
    
    # 学习参数
    learning_rate: float = 0.02  # 稍微降低学习率
    n_estimators: int = 3000     # 增加最大迭代轮数
    
    # 正则化 (针对高维特征增强正则)
    lambda_l1: float = 0.2      # L1 正则
    lambda_l2: float = 1.0      # L2 正则
    feature_fraction: float = 0.7  # 每棵树随机选 70% 特征（特征多时降低）
    bagging_fraction: float = 0.8  # 每棵树随机选 80% 样本
    bagging_freq: int = 1
    
    # 特征采样（colsample_bytree 的别名）
    colsample_bytree: float = 0.7
    
    # 其他
    verbose: int = -1  # 静默模式
    seed: int = 42
    force_row_wise: bool = True  # GPU 模式下使用行方向
    
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
            "force_row_wise": self.force_row_wise,
        }


@dataclass
class TrainConfig:
    """训练配置"""
    
    # 迭代控制
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100  # 增加耐心，因为特征多收敛慢
    
    # 日志
    verbose_eval: int = 100  # 每 100 轮打印一次
    
    # 模型保存
    save_model: bool = True
    model_name: str = "lgbm_excess_ret_5d"


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
    
    @classmethod
    def for_target(cls, target_col: str) -> "LGBMConfig":
        """
        为指定目标列创建配置
        
        Args:
            target_col: 目标列名 (如 'excess_ret_5d', 'label_bin_5d')
        """
        config = cls()
        config.data.target_col = target_col
        config.train.model_name = f"lgbm_{target_col}"
        
        # 如果是分类目标，调整参数
        if target_col.startswith("label_bin"):
            config.params.objective = "binary"
            config.params.metric = "auc"
        
        return config
