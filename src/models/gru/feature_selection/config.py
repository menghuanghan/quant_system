"""
GRU 专属特征筛选配置。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GRUFeatureSelectionConfig:
    """GRU 时序特征筛选配置"""

    # 筛选模式：默认仅做平稳性硬筛选（全量平稳特征入模）
    stationary_only: bool = True

    # 基础字段
    target_col: str = "rank_ret_5d"
    trade_date_col: str = "trade_date"
    ts_code_col: str = "ts_code"

    # 数据质量门槛
    min_non_null_ratio: float = 0.60
    min_cross_section_samples: int = 80
    min_months: int = 12

    # 平稳性检验（ADF）
    adf_pvalue_threshold: float = 0.05
    adf_max_points: int = 120_000

    # 有效性门槛（RankIC/ICIR/稳定性）
    min_abs_rank_ic: float = 0.02
    min_icir: float = 0.30
    min_positive_ratio: float = 0.55

    # 输出特征上限
    top_k: int = 80

    # 是否在 stationary-only 模式下仍计算预测指标（仅诊断，不参与 hard filter）
    compute_predictive_diagnostics: bool = False

    # 宏观特征动态相关过滤（仅对白名单外宏观特征生效）
    macro_dynamic_window: int = 60
    macro_min_abs_corr: float = 0.20
    macro_min_direction_ratio: float = 0.60

    # 共线性控制（VIF）
    enable_vif: bool = True
    vif_threshold: float = 10.0
    vif_min_samples: int = 180

    # 宏观核心白名单（用户确认口径）
    core_macro_whitelist: List[str] = field(default_factory=lambda: [
        "gdp_yoy",
        "cpi_yoy",
        "ppi_yoy",
        "m2_yoy",
        "pmi",
        "pmi_prod",
        "pmi_new_order",
        "lpr_1y",
        "shibor_1m",
        "stock_bond_spread",
        "macro_score",
        "macro_regime",
        "market_sentiment",
        "beta_signal",
    ])

    # 宏观前缀（用于识别“宏观类”列）
    macro_prefixes: List[str] = field(default_factory=lambda: [
        "gdp_",
        "cpi_",
        "ppi_",
        "pmi",
        "m2",
        "lpr_",
        "macro_",
        "shibor_",
        "market_total_",
        "market_sentiment",
        "beta_signal",
        "stock_bond_spread",
        "buffett_",
        "pb_",
        "hsgt_",
        "mf_north_",
        "sh300_",
        "zz500_",
        "zz1000_",
        "cyb_",
        "sz50_",
        "kc50_",
        "rs_",
        "if_",
        "ic_",
        "ih_",
        "im_",
        "liquidity_gc001_",
        "liquidity_r001_",
        "money_regime",
        "risk_appetite",
        "macro_regime",
        "macro_score",
    ])
