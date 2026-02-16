"""
深度学习 (GRU) 模型配置

改造说明（2026.02）:
- 适配全域数据 train_gru.parquet (281 列特征)
- 动态特征识别（排除法）
- 支持多种标签目标
- GPU 加速优化
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set
import logging

logger = logging.getLogger(__name__)

# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[3]  # src/models/deep -> quant_system


# ============================================================================
# 列分类定义（与 LightGBM 共享）
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
    "is_tradable",
    "is_risky",
    "is_limit",
    "return_1d",  # GRU 数据中使用 return_1d 而非 ret_1d
]

# 类别特征：GRU 不直接使用，需要转换或排除
CATEGORICAL_FEATURES = [
    "market",
    "industry_idx",
    "sw_l1_idx",
    "sw_l2_idx",
]


def get_gru_feature_columns(
    all_columns: List[str],
    exclude_cols: Optional[List[str]] = None,
    exclude_categorical: bool = True,
) -> List[str]:
    """
    动态识别 GRU 特征列（排除法）
    
    与 LightGBM 不同，GRU 默认排除类别特征（可选嵌入层处理）
    
    Args:
        all_columns: 数据集的所有列名
        exclude_cols: 额外需要排除的列
        exclude_categorical: 是否排除类别特征（默认 True）
        
    Returns:
        特征列名列表
    """
    # 构建排除集合
    exclude_set: Set[str] = set(ID_COLS + LABEL_COLS + AUX_COLS)
    
    if exclude_cols:
        exclude_set.update(exclude_cols)
    
    if exclude_categorical:
        exclude_set.update(CATEGORICAL_FEATURES)
    
    # 排除法获取特征列
    feature_cols = [col for col in all_columns if col not in exclude_set]
    
    logger.info(f"GRU 动态特征识别: 总列数={len(all_columns)}, 排除={len(exclude_set)}, 特征={len(feature_cols)}")
    
    return feature_cols


@dataclass
class GRUDataConfig:
    """GRU 数据配置"""
    
    # 数据路径（使用 GRU 专用数据）
    data_path: Path = BASE_DIR / "data" / "features" / "structured" / "train_gru.parquet"
    
    # 窗口长度（对应一个月交易日）
    window_size: int = 20
    
    # 目标列（推荐使用超额收益，与 LightGBM 保持一致）
    target_col: str = "excess_ret_5d"
    
    # === 特征筛选配置 ===
    # 是否启用特征筛选（使用 LightGBM Top N 特征）
    use_feature_selection: bool = False
    
    # 特征列表 JSON 文件路径
    feature_selection_json: Path = BASE_DIR / "models" / "lgbm" / "top50_features.json"
    
    # 直接指定特征列表（优先级高于 JSON 文件）
    selected_features: List[str] = field(default_factory=list)
    
    # === Embedding 配置 ===
    # 是否使用 Embedding层处理类别特征
    use_embedding: bool = False
    
    # Embedding 配置（基于数据分析）
    # market: 4 个值 (0-3)
    # industry_idx: 110 个值 (0-109)
    # sw_l1_idx: 32 个值 (0-31)
    # sw_l2_idx: 129 个值 (-1 到 128)
    embedding_config: dict = field(default_factory=lambda: {
        'market': {'num_embeddings': 5, 'embed_dim': 4},      # 0-3 + padding
        'industry_idx': {'num_embeddings': 112, 'embed_dim': 16},  # 0-109 + padding + unknown
        'sw_l1_idx': {'num_embeddings': 34, 'embed_dim': 8},   # 0-31 + padding + unknown
        'sw_l2_idx': {'num_embeddings': 132, 'embed_dim': 12},  # -1→0, 0-128→1-129 + padding
    })
    
    # 使用哪些类别特征（空=使用全部）
    embedding_features: List[str] = field(default_factory=lambda: ['sw_l1_idx'])
    
    # 兼容旧配置
    use_industry_embedding: bool = False  # deprecated, use use_embedding
    num_industries: int = 32  # deprecated
    industry_embed_dim: int = 8  # deprecated
    
    # 额外排除列
    extra_exclude_cols: List[str] = field(default_factory=list)
    
    # 是否排除类别特征
    exclude_categorical: bool = True
    
    # Purging（隔离带）: 防止标签泄露
    purge_days: int = 5
    
    # 时间切分（GRU 数据从 2020.07 开始，需要前置数据构建窗口）
    # 训练: 2021-2023 (窗口从 2020.07 开始构建)
    # 验证: 2024
    # 测试: 2025
    train_start: str = "2021-01-01"
    train_end: str = "2023-12-25"  # 提前 5 天，留出隔离带
    valid_start: str = "2024-01-01"
    valid_end: str = "2024-12-25"  # 提前 5 天
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"
    
    # GPU 加速
    use_gpu: bool = True
    
    def get_exclude_cols(self) -> List[str]:
        """获取完整的排除列列表"""
        exclude = ID_COLS + LABEL_COLS + AUX_COLS + self.extra_exclude_cols
        if self.exclude_categorical:
            exclude += CATEGORICAL_FEATURES
        return exclude


@dataclass
class GRUModelConfig:
    """GRU 模型配置"""
    
    # 输入维度（运行时从数据确定）
    input_dim: int = 50  # 默认使用 Top 50 特征
    
    # GRU 配置（防过拟合优化版）
    hidden_dim: int = 32   # 降低隐层大小，减少参数
    num_layers: int = 1    # 单层 GRU，简化模型
    dropout: float = 0.5   # 增大 dropout 防止过拟合
    
    # MLP 头配置
    mlp_hidden: int = 32   # 降低 MLP 隐层
    mlp_dropout: float = 0.5
    
    # 是否双向
    bidirectional: bool = False
    
    # === Embedding 配置 ===
    use_embedding: bool = False
    embedding_config: dict = field(default_factory=lambda: {
        'market': {'num_embeddings': 5, 'embed_dim': 4},
        'industry_idx': {'num_embeddings': 112, 'embed_dim': 16},
        'sw_l1_idx': {'num_embeddings': 34, 'embed_dim': 8},
        'sw_l2_idx': {'num_embeddings': 132, 'embed_dim': 12},
    })
    embedding_features: List[str] = field(default_factory=lambda: ['sw_l1_idx'])
    
    # 兼容旧配置 (deprecated)
    use_industry_embedding: bool = False
    num_industries: int = 32
    industry_embed_dim: int = 8


@dataclass
class GRUTrainConfig:
    """GRU 训练配置"""
    
    # 基础配置
    epochs: int = 100
    batch_size: int = 2048  # RTX 5070 12GB
    num_workers: int = 4
    
    # 优化器（增强正则化）
    learning_rate: float = 5e-4   # 降低学习率
    weight_decay: float = 1e-3    # 增大权重衰减
    
    # 学习率调度
    lr_scheduler: str = "cosine"
    lr_min: float = 1e-5
    
    # 损失函数
    loss_type: str = "combined"  # combined / ic_only / mse
    mse_weight: float = 0.5
    ic_weight: float = 0.5
    
    # 早停
    patience: int = 15  # 增加耐心
    min_epochs: int = 5
    
    # 混合精度
    use_amp: bool = True
    
    # 梯度裁剪
    grad_clip: float = 1.0
    
    # 保存
    save_dir: Path = BASE_DIR / "models" / "gru"
    save_best: bool = True
    model_name: str = "gru_excess_ret_5d"
    
    # 日志
    log_interval: int = 100


@dataclass
class GRUConfig:
    """GRU 总配置"""
    
    data: GRUDataConfig = field(default_factory=GRUDataConfig)
    model: GRUModelConfig = field(default_factory=GRUModelConfig)
    train: GRUTrainConfig = field(default_factory=GRUTrainConfig)
    
    def __post_init__(self):
        """初始化后创建输出目录"""
        self.train.save_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def default(cls) -> "GRUConfig":
        """创建默认配置"""
        return cls()
    
    @classmethod
    def for_target(cls, target_col: str) -> "GRUConfig":
        """为指定目标列创建配置"""
        config = cls()
        config.data.target_col = target_col
        config.train.model_name = f"gru_{target_col}"
        return config
    
    @classmethod
    def large_batch(cls) -> "GRUConfig":
        """大 Batch Size 配置（适合高显存 GPU）"""
        config = cls()
        config.train.batch_size = 4096
        config.train.learning_rate = 2e-3  # 大 batch 可适当增加学习率
        return config
