"""
VIF 共线性过滤。
"""

from typing import Dict, List, Tuple
import importlib

import numpy as np
import pandas as pd


def apply_vif_filter(
    feature_vectors: Dict[str, pd.Series],
    threshold: float,
    min_samples: int,
) -> Tuple[List[str], Dict[str, object]]:
    """
    对输入特征向量执行 VIF 迭代剔除。

    Args:
        feature_vectors: {feature: pd.Series(index=trade_date, value=daily_feature_mean)}
        threshold: VIF 阈值
        min_samples: 最少样本行数

    Returns:
        (remaining_features, metadata)
    """
    if not feature_vectors:
        return [], {"vif_scores": {}, "dropped_by_vif": [], "sample_rows": 0}

    try:
        _vif_module = importlib.import_module("statsmodels.stats.outliers_influence")
        variance_inflation_factor = getattr(_vif_module, "variance_inflation_factor")
    except Exception:
        # statsmodels 不可用时，跳过 VIF（其余流程继续）
        remaining = list(feature_vectors.keys())
        return remaining, {
            "vif_scores": {},
            "dropped_by_vif": [],
            "sample_rows": 0,
            "skipped": "statsmodels_unavailable",
        }

    matrix_df = pd.DataFrame(feature_vectors)
    matrix_df = matrix_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    if matrix_df.shape[0] < min_samples or matrix_df.shape[1] <= 1:
        remaining = list(matrix_df.columns) if matrix_df.shape[1] > 0 else list(feature_vectors.keys())
        return remaining, {
            "vif_scores": {},
            "dropped_by_vif": [],
            "sample_rows": int(matrix_df.shape[0]),
            "skipped": "insufficient_samples_or_features",
        }

    remaining = list(matrix_df.columns)
    dropped: List[str] = []
    vif_scores: Dict[str, float] = {}

    while len(remaining) > 1:
        x = matrix_df[remaining]
        current_vifs = []

        for i, col in enumerate(remaining):
            try:
                vif = float(variance_inflation_factor(x.values, i))
            except Exception:
                vif = float("inf")
            current_vifs.append(vif)
            vif_scores[col] = vif

        max_vif = max(current_vifs)
        if np.isfinite(max_vif) and max_vif <= threshold:
            break

        drop_idx = int(np.argmax(current_vifs))
        drop_feature = remaining.pop(drop_idx)
        dropped.append(drop_feature)

    return remaining, {
        "vif_scores": vif_scores,
        "dropped_by_vif": dropped,
        "sample_rows": int(matrix_df.shape[0]),
    }
