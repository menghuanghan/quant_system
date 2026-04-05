"""
平稳化变换：ADF 检验 + 差分/变化率备选。
"""

from dataclasses import dataclass
import importlib
import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    _stattools = importlib.import_module("statsmodels.tsa.stattools")
    adfuller = getattr(_stattools, "adfuller")
    _HAS_ADF = True
except Exception:
    adfuller = None
    _HAS_ADF = False
    logger.warning("statsmodels 不可用，ADF 检验将跳过（默认按通过处理）")


@dataclass
class StationaryTransformResult:
    method: str
    transformed: np.ndarray
    raw_adf_pvalue: float
    transformed_adf_pvalue: float
    passed: bool


def _sample_series(values: np.ndarray, max_points: int) -> np.ndarray:
    if len(values) <= max_points:
        return values
    idx = np.linspace(0, len(values) - 1, max_points, dtype=np.int64)
    return values[idx]


def compute_adf_pvalue(values: np.ndarray, max_points: int = 120_000) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size < 32:
        return np.nan
    if np.nanstd(arr) < 1e-12:
        return np.nan

    if not _HAS_ADF:
        return np.nan

    arr = _sample_series(arr, max_points=max_points)

    try:
        return float(adfuller(arr, regression="c", autolag="AIC")[1])
    except Exception:
        return np.nan


def _diff(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[:] = np.nan
    out[1:] = np.diff(values)
    return out


def _pct_change(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[:] = np.nan
    denom = np.abs(values[:-1]) + 1e-6
    out[1:] = np.diff(values) / denom
    return out


def to_stationary(
    values: np.ndarray,
    adf_pvalue_threshold: float = 0.05,
    adf_max_points: int = 120_000,
) -> StationaryTransformResult:
    arr = np.asarray(values, dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan

    raw_p = compute_adf_pvalue(arr, max_points=adf_max_points)
    if np.isnan(raw_p):
        # ADF 不可用或样本不足时，先按 identity 放行，后续交由 IC 质量门槛过滤
        return StationaryTransformResult(
            method="identity",
            transformed=arr,
            raw_adf_pvalue=np.nan,
            transformed_adf_pvalue=np.nan,
            passed=True,
        )

    if raw_p < adf_pvalue_threshold:
        return StationaryTransformResult(
            method="identity",
            transformed=arr,
            raw_adf_pvalue=raw_p,
            transformed_adf_pvalue=raw_p,
            passed=True,
        )

    candidates: list[tuple[str, np.ndarray, float]] = []

    diff_arr = _diff(arr)
    diff_p = compute_adf_pvalue(diff_arr, max_points=adf_max_points)
    candidates.append(("diff", diff_arr, diff_p))

    pct_arr = _pct_change(arr)
    pct_p = compute_adf_pvalue(pct_arr, max_points=adf_max_points)
    candidates.append(("pct_change", pct_arr, pct_p))

    passing = [(m, a, p) for (m, a, p) in candidates if np.isfinite(p) and p < adf_pvalue_threshold]
    if passing:
        method, transformed, pval = min(passing, key=lambda x: x[2])
        return StationaryTransformResult(
            method=method,
            transformed=transformed,
            raw_adf_pvalue=raw_p,
            transformed_adf_pvalue=pval,
            passed=True,
        )

    # 未通过平稳性门槛
    best_method = "drop"
    best_p = np.nan
    finite_candidates = [(m, p) for m, _, p in candidates if np.isfinite(p)]
    if finite_candidates:
        best_method, best_p = min(finite_candidates, key=lambda x: x[1])

    return StationaryTransformResult(
        method=best_method,
        transformed=arr,
        raw_adf_pvalue=raw_p,
        transformed_adf_pvalue=best_p,
        passed=False,
    )
