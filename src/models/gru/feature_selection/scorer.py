"""
单特征时序有效性评分：RankIC / ICIR / 稳定性。
"""

from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureScore:
    feature: str
    passed: bool
    reason: str
    non_null_ratio: float
    valid_days: int
    valid_months: int
    ic_mean: float
    rank_ic_mean: float
    rank_ic_std: float
    icir: float
    positive_ratio: float
    corr_abs_mean: float
    corr_direction_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def score_feature(
    feature_name: str,
    trade_dates: pd.Series,
    target_values: np.ndarray,
    feature_values: np.ndarray,
    min_cross_section_samples: int,
    min_months: int,
    min_abs_rank_ic: float,
    min_icir: float,
    min_positive_ratio: float,
    macro_dynamic_window: int,
) -> Tuple[FeatureScore, pd.Series]:
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(trade_dates),
        "target": pd.to_numeric(target_values, errors="coerce"),
        "feature": pd.to_numeric(feature_values, errors="coerce"),
    })
    n_total = len(df)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["target", "feature"])
    non_null_ratio = len(df) / max(n_total, 1)

    if df.empty:
        empty_series = pd.Series(dtype=np.float64)
        score = FeatureScore(
            feature=feature_name,
            passed=False,
            reason="all_nan",
            non_null_ratio=non_null_ratio,
            valid_days=0,
            valid_months=0,
            ic_mean=np.nan,
            rank_ic_mean=np.nan,
            rank_ic_std=np.nan,
            icir=np.nan,
            positive_ratio=np.nan,
            corr_abs_mean=np.nan,
            corr_direction_ratio=np.nan,
        )
        return score, empty_series

    records = []
    for dt, group in df.groupby("trade_date", sort=True):
        if len(group) < min_cross_section_samples:
            continue
        if group["feature"].nunique(dropna=True) < 2:
            continue
        if group["target"].nunique(dropna=True) < 2:
            continue

        ic = group["feature"].corr(group["target"], method="pearson")
        rank_ic = group["feature"].corr(group["target"], method="spearman")

        if not np.isfinite(rank_ic):
            continue

        records.append({
            "trade_date": dt,
            "ic": float(ic) if np.isfinite(ic) else np.nan,
            "rank_ic": float(rank_ic),
            "feature_mean": float(group["feature"].mean()),
            "target_mean": float(group["target"].mean()),
        })

    if not records:
        empty_series = pd.Series(dtype=np.float64)
        score = FeatureScore(
            feature=feature_name,
            passed=False,
            reason="insufficient_cross_section",
            non_null_ratio=non_null_ratio,
            valid_days=0,
            valid_months=0,
            ic_mean=np.nan,
            rank_ic_mean=np.nan,
            rank_ic_std=np.nan,
            icir=np.nan,
            positive_ratio=np.nan,
            corr_abs_mean=np.nan,
            corr_direction_ratio=np.nan,
        )
        return score, empty_series

    daily_df = pd.DataFrame(records)
    daily_df["month"] = daily_df["trade_date"].dt.to_period("M")
    monthly_rank_ic = daily_df.groupby("month")["rank_ic"].mean()

    valid_months = int(monthly_rank_ic.shape[0])
    rank_ic_mean = float(monthly_rank_ic.mean()) if valid_months else np.nan
    rank_ic_std = float(monthly_rank_ic.std(ddof=0)) if valid_months else np.nan
    icir = float(rank_ic_mean / (rank_ic_std + 1e-12)) if np.isfinite(rank_ic_mean) else np.nan
    positive_ratio = float((monthly_rank_ic > 0).mean()) if valid_months else np.nan
    ic_mean = float(daily_df["ic"].mean())

    min_corr_periods = max(10, macro_dynamic_window // 3)
    rolling_corr = daily_df["feature_mean"].rolling(
        window=macro_dynamic_window,
        min_periods=min_corr_periods,
    ).corr(daily_df["target_mean"])
    rolling_corr = rolling_corr.dropna()

    if rolling_corr.empty:
        corr_abs_mean = np.nan
        corr_direction_ratio = np.nan
    else:
        corr_abs_mean = float(rolling_corr.abs().mean())
        pos_ratio = float((rolling_corr > 0).mean())
        neg_ratio = float((rolling_corr < 0).mean())
        corr_direction_ratio = max(pos_ratio, neg_ratio)

    passed = (
        valid_months >= min_months
        and np.isfinite(rank_ic_mean)
        and np.isfinite(icir)
        and np.isfinite(positive_ratio)
        and abs(rank_ic_mean) >= min_abs_rank_ic
        and icir >= min_icir
        and positive_ratio >= min_positive_ratio
    )

    reason = "pass" if passed else "quality_threshold"

    score = FeatureScore(
        feature=feature_name,
        passed=passed,
        reason=reason,
        non_null_ratio=non_null_ratio,
        valid_days=int(daily_df.shape[0]),
        valid_months=valid_months,
        ic_mean=ic_mean,
        rank_ic_mean=rank_ic_mean,
        rank_ic_std=rank_ic_std,
        icir=icir,
        positive_ratio=positive_ratio,
        corr_abs_mean=corr_abs_mean,
        corr_direction_ratio=corr_direction_ratio,
    )

    daily_feature_mean = daily_df.set_index("trade_date")["feature_mean"]
    return score, daily_feature_mean
