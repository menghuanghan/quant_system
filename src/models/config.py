"""
模型超参数和全局配置
包含 LightGBM 和 GRU 模型的默认参数配置
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging
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
    ])
    
    # 标签列前缀（训练时需要排除的列）
    label_prefixes: List[str] = field(default_factory=lambda: [
        "label_",
        "ret_",
        "excess_ret_", 
        "rank_ret_",
        "sharpe_",
    ])
    
    # ==================== 宏观/无截面方差特征（需剔除）====================
    # 这些特征在同一天内所有股票数值相同，对横截面模型无区分度
    # 会导致模型被这些无意义特征主导，产生"降维打击"效应
    drop_macro_prefixes: List[str] = field(default_factory=lambda: [
        # 宏观经济指标
        "gdp_",
        "cpi_",
        "ppi_",
        "pmi",           # pmi, pmi_prod, pmi_new_order, pmi_regime
        "m2",            # m2, m2_yoy
        "lpr_",          # lpr_1y, lpr_5y, lpr_trend
        "macro_",        # 【新增】macro_amount_shibor, macro_vol_m2 等宏观聚合特征
        # 利率/货币市场
        "shibor_",       # shibor_on, shibor_1w, shibor_1m, shibor_3m, shibor_6m, shibor_1y
        # 市场总体指标（无截面方差）
        "market_total_", # market_total_rzye, market_total_rqye, market_total_rzrqye
        "market_congestion",
        "stock_bond_spread",
        "break_net_ratio",
        "buffett_",      # buffett_indicator, buffett_quantile_*
        "pb_median",
        "pb_ew",
        "pb_quantile_",
        # 陆股通总体流向（无截面方差）
        "hsgt_north",    # hsgt_north, hsgt_north_ma5, hsgt_north_ma20
        "hsgt_south",
        "hsgt_hgt",
        "hsgt_sgt",
        "hsgt_ggt_",
        "mf_north_",     # 【新增】mf_north_net 等北向资金总体流向
        # 指数行情（无截面方差）
        "sh300_",        # sh300_pct_chg, sh300_amplitude, sh300_turnover, sh300_close, sh300_vol, sh300_amount
        "zz500_",
        "zz1000_",
        "cyb_",
        "sz50_",
        "kc50_",
        "rs_",           # 【新增】rs_csi500, rs_hs300 相对指数强度（无截面方差）
        # 股指期货（无截面方差）
        "if_",           # if_total_oi, if_close, if_basis_rate
        "ic_",
        "ih_",
        "im_",
        # 货币市场流动性（无截面方差）
        "liquidity_gc001_",
        "liquidity_r001_",
        # 纯时序特征（无截面方差）
        "lag_days",      # 【新增】距上一交易日天数
        # 宏观衍生状态（无截面方差）
        "money_regime",
        "risk_appetite",
        "macro_score",
        "macro_regime",
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
    num_boost_round: int = 3000           # 增加最大迭代轮数
    early_stopping_rounds: int = 100      # 增加早停轮数，让模型学得更久
    verbose_eval: int = 100
    
    # 核心防过拟合参数 - 强制模型慢慢学，多看个股特征
    max_depth: int = 6                    # 树深度 4~7
    num_leaves: int = 31                  # 叶子数 15~63
    min_data_in_leaf: int = 300           # 叶子最小样本数，防止对单只股票过拟合
    feature_fraction: float = 0.5         # 列抽样【关键】降低到0.5，强制挖掘深层Alpha
    bagging_fraction: float = 0.8         # 行抽样 0.7~0.9
    bagging_freq: int = 1                 # bagging 频率
    
    # 学习率【关键】降低到0.01，让模型慢慢学
    learning_rate: float = 0.01
    
    # 正则化（适当降低，让模型有能力学习）
    lambda_l1: float = 0.05
    lambda_l2: float = 0.5
    
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

_logger = logging.getLogger(__name__)


# ====================== GRU 列分类定义 ======================

ID_COLS = ["ts_code", "trade_date"]

LABEL_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "label_1d", "label_5d", "label_10d", "label_20d",
    "excess_ret_5d", "excess_ret_10d",
    "rank_ret_5d", "rank_ret_10d",
    "sharpe_5d", "sharpe_10d", "sharpe_20d",
    "label_bin_5d",
]

AUX_COLS = [
    "is_tradable",
    "is_risky",
    "is_limit",
    "return_1d",
]

CATEGORICAL_FEATURES = [
    "market",
    "industry_idx",
    "sw_l1_idx",
    "sw_l2_idx",
]


def get_gru_feature_columns(
    all_columns: List[str],
    target_cols: List[str],
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    动态识别 GRU 特征列（排除法）

    排除: ID列 + 全部标签列 + 辅助列 + 类别特征 + 自定义排除列
    """
    exclude_set: Set[str] = set(ID_COLS + LABEL_COLS + AUX_COLS + CATEGORICAL_FEATURES)
    if exclude_cols:
        exclude_set.update(exclude_cols)
    feature_cols = [col for col in all_columns if col not in exclude_set]
    _logger.info(
        f"GRU 动态特征识别: 总列数={len(all_columns)}, "
        f"排除={len(exclude_set)}, 特征={len(feature_cols)}"
    )
    return feature_cols


def get_gru_selected_features(
    all_columns: List[str],
    mode: str = "rolling",
    top_n: int = 50,
    feature_importance_dir: Optional[Path] = None,
) -> List[str]:
    """
    GRU 特征选择（LGB Top N + 宏观特征）

    策略:
    1. 读取 LightGBM feature_importance.parquet → 按 importance 降序取 Top N
    2. 从 Top N 中排除类别特征和辅助/掩码列（CATEGORICAL_FEATURES + AUX_COLS）
    3. 追加所有宏观特征（FeatureConfig.drop_macro_prefixes 匹配的列），
       因为宏观特征被 LGB 剔除，但对 GRU 时序建模有价值
    4. 取交集：仅保留 all_columns 中实际存在的列
    5. 去重并保持顺序

    Args:
        all_columns: 数据文件中的全部列名
        mode: 训练模式 ("rolling" / "expanding" / "single_full")
        top_n: 取 LGB 重要性排名前 N 的特征
        feature_importance_dir: 特征重要性文件所在目录（默认 models/lgb/{mode}/）

    Returns:
        selected: 筛选后的特征列名列表
    """
    import pandas as pd

    # ---- 1. 确定 feature_importance 路径 ----
    if feature_importance_dir is None:
        # rolling / expanding 共用 rolling 路径
        lgb_mode = "single_full" if mode == "single_full" else "rolling"
        fi_path = MODELS_DIR / "lgb" / lgb_mode / "feature_importance.parquet"
    else:
        fi_path = Path(feature_importance_dir) / "feature_importance.parquet"

    # 不可用类别
    exclude_set: Set[str] = set(
        ID_COLS + LABEL_COLS + AUX_COLS + CATEGORICAL_FEATURES
    )

    # ---- 2. 读取 LGB Top N ----
    lgb_top_features: List[str] = []
    if fi_path.exists():
        fi_df = pd.read_parquet(fi_path)
        # 取跨 target 的平均重要性去重排序
        avg_imp = (
            fi_df.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
        )
        # 排除类别 / 辅助列
        for feat in avg_imp.index:
            if feat not in exclude_set:
                lgb_top_features.append(feat)
            if len(lgb_top_features) >= top_n:
                break
        _logger.info(
            f"LGB Top {top_n} 特征已加载 (from {fi_path}): "
            f"实际获得 {len(lgb_top_features)} 个"
        )
    else:
        _logger.warning(
            f"feature_importance 文件未找到: {fi_path}，"
            f"将回退到全量排除法"
        )
        return get_gru_feature_columns(all_columns, [], None)

    # ---- 3. 收集宏观特征 ----
    macro_prefixes = FeatureConfig().drop_macro_prefixes
    macro_features = []
    for col in all_columns:
        if col in exclude_set:
            continue
        for prefix in macro_prefixes:
            if col.startswith(prefix):
                macro_features.append(col)
                break

    _logger.info(f"宏观特征: {len(macro_features)} 个")

    # ---- 4. 合并去重，取 all_columns 交集 ----
    all_columns_set = set(all_columns)
    seen: Set[str] = set()
    selected: List[str] = []

    # LGB Top N 优先
    for f in lgb_top_features:
        if f in all_columns_set and f not in seen:
            selected.append(f)
            seen.add(f)

    # 宏观特征追加
    for f in macro_features:
        if f in all_columns_set and f not in seen:
            selected.append(f)
            seen.add(f)

    _logger.info(
        f"GRU 特征选择完成: LGB_Top{top_n}={len(lgb_top_features)}, "
        f"宏观={len(macro_features)}, 合计={len(selected)}"
    )
    return selected


# ====================== GRU 数据配置 ======================

@dataclass
class GRUDataConfig:
    """GRU 数据配置"""
    data_path: Path = FEATURES_DIR / "train_gru.parquet"
    seq_len: int = 20
    target_cols: List[str] = field(default_factory=lambda: [
        "rank_ret_5d",
        "excess_ret_5d",
        "sharpe_5d",
    ])
    data_start_date: str = "2021-01-01"
    data_end_date: str = "2025-12-31"
    use_gpu: bool = True


# ====================== GRU 时序切分配置 ======================

@dataclass
class GRUSplitConfig:
    """GRU 时序切分配置"""
    train_window_months: int = 24
    valid_window_months: int = 3
    step_months: int = 3
    mode: str = "rolling"  # rolling / expanding / single_full
    # 逻辑日期边界（与 GRUDataConfig 同步，由 Trainer 设置）
    data_start_date: str = "2021-01-01"
    data_end_date: str = "2025-12-31"


# ====================== GRU 网络结构配置 ======================

@dataclass
class GRUNetworkConfig:
    """GRU 神经网络配置"""
    num_features: int = 250       # 运行时从数据决定
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    use_attention: bool = True
    num_targets: int = 3          # 运行时从 target_cols 决定


# ====================== GRU 训练配置 ======================

@dataclass
class GRUTrainConfig:
    """GRU 训练配置"""
    epochs: int = 100
    batch_size: int = 2048
    num_workers: int = 0

    # AdamW
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4

    # OneCycleLR
    max_lr: float = 1e-3
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4

    # 多任务损失权重
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "rank_ret_5d": 0.6,
        "excess_ret_5d": 0.2,
        "sharpe_5d": 0.2,
    })
    # 每个任务损失类型: mse / huber
    loss_types: Dict[str, str] = field(default_factory=lambda: {
        "rank_ret_5d": "mse",
        "excess_ret_5d": "huber",
        "sharpe_5d": "huber",
    })
    use_uncertainty_weighting: bool = False

    # 早停（以 Rank IC 为准）
    patience: int = 15
    min_epochs: int = 10

    # AMP
    use_amp: bool = True

    # 梯度裁剪
    grad_clip: float = 1.0

    # 保存目录
    save_dir: Path = MODELS_DIR / "gru"

    # 随机种子
    seed: int = 42
    # single_full 模式多种子列表
    multi_seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])

    log_interval: int = 50


# ====================== GRU 推断配置 ======================

@dataclass
class GRUInferenceConfig:
    """GRU 实盘推断配置"""
    rolling_weight: float = 0.4
    full_weight: float = 0.6
    rolling_models_dir: Path = MODELS_DIR / "gru" / "rolling"
    full_models_dir: Path = MODELS_DIR / "gru" / "single_full"


# ====================== GRU 总配置 ======================

@dataclass
class GRUConfig:
    """GRU 总配置"""
    data: GRUDataConfig = field(default_factory=GRUDataConfig)
    split: GRUSplitConfig = field(default_factory=GRUSplitConfig)
    network: GRUNetworkConfig = field(default_factory=GRUNetworkConfig)
    train: GRUTrainConfig = field(default_factory=GRUTrainConfig)
    inference: GRUInferenceConfig = field(default_factory=GRUInferenceConfig)

    def __post_init__(self):
        self.train.save_dir.mkdir(parents=True, exist_ok=True)
        self.network.num_targets = len(self.data.target_cols)

    @classmethod
    def default(cls) -> "GRUConfig":
        return cls()


DEFAULT_GRU_CONFIG = GRUConfig()