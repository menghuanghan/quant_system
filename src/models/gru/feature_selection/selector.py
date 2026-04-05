"""
GRU 专属时序特征筛选主编排。
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .config import GRUFeatureSelectionConfig
from .scorer import score_feature
from .stream_reader import ParquetColumnStreamReader
from .transformer import to_stationary
from .vif import apply_vif_filter

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    selected_features: List[str]
    transform_specs: Dict[str, str]
    metrics_df: pd.DataFrame
    artifact_dir: Path
    selector_mode: str
    candidate_count: int
    adf_pass_count: int


class GRUFeatureSelector:
    """全样本一次拟合 + 逐列流式筛选。"""

    def __init__(self, config: Optional[GRUFeatureSelectionConfig] = None):
        self.config = config or GRUFeatureSelectionConfig()

    def fit_or_load_artifacts(
        self,
        data_path: Path,
        all_columns: List[str],
        mode: str,
        output_dir: Path,
        exclude_cols: Optional[List[str]] = None,
        force_refit: bool = False,
    ) -> FeatureSelectionResult:
        output_dir = Path(output_dir)
        selected_path = output_dir / "selected_features.json"
        transform_path = output_dir / "transform_specs.json"
        metrics_path = output_dir / "feature_metrics.parquet"

        if (
            not force_refit
            and selected_path.exists()
            and transform_path.exists()
            and metrics_path.exists()
        ):
            payload = json.loads(selected_path.read_text(encoding="utf-8"))
            compatible, reason = self._is_cache_compatible(payload=payload, mode=mode)
            if compatible:
                logger.info(f"加载已有 GRU 特征筛选工件: {output_dir}")
                transform_specs = json.loads(transform_path.read_text(encoding="utf-8"))
                metrics_df = pd.read_parquet(metrics_path)
                return FeatureSelectionResult(
                    selected_features=list(payload.get("selected_features", [])),
                    transform_specs={str(k): str(v) for k, v in transform_specs.items()},
                    metrics_df=metrics_df,
                    artifact_dir=output_dir,
                    selector_mode=str(payload.get("selector_mode", self._selector_mode_name())),
                    candidate_count=int(payload.get("candidate_count", 0) or 0),
                    adf_pass_count=int(payload.get("adf_pass_count", 0) or 0),
                )

            logger.info(
                f"已有特征筛选工件与当前配置不兼容，触发重算: {reason}"
            )

        return self.fit(
            data_path=data_path,
            all_columns=all_columns,
            mode=mode,
            output_dir=output_dir,
            exclude_cols=exclude_cols,
        )

    def fit(
        self,
        data_path: Path,
        all_columns: List[str],
        mode: str,
        output_dir: Path,
        exclude_cols: Optional[List[str]] = None,
    ) -> FeatureSelectionResult:
        cfg = self.config
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        selector_mode = self._selector_mode_name()

        target_col = cfg.target_col
        trade_date_col = cfg.trade_date_col

        if target_col not in all_columns:
            raise ValueError(
                f"特征筛选目标列不存在: {target_col}，请确认训练数据包含该列"
            )

        reader = ParquetColumnStreamReader(data_path)

        meta_df = reader.read_columns_as_pandas([trade_date_col, target_col])
        meta_df[trade_date_col] = pd.to_datetime(meta_df[trade_date_col])
        target_values = pd.to_numeric(meta_df[target_col], errors="coerce").to_numpy(dtype=np.float32)
        trade_dates = meta_df[trade_date_col]

        exclude_set: Set[str] = set(exclude_cols or [])
        exclude_set.update({cfg.trade_date_col, cfg.ts_code_col, cfg.target_col})

        candidate_features = [c for c in all_columns if c not in exclude_set]
        logger.info(
            f"GRU 特征筛选开始: mode={mode}, candidates={len(candidate_features)}, "
            f"target={target_col}"
        )

        metrics_rows: List[dict] = []
        preselected_features: List[str] = []
        preselected_vectors: Dict[str, pd.Series] = {}
        transform_specs_all: Dict[str, str] = {}

        for i, feat in enumerate(candidate_features, start=1):
            raw_values = reader.read_column_as_numpy(feat, dtype=np.float64)

            if raw_values.shape[0] != target_values.shape[0]:
                metrics_rows.append({
                    "mode": mode,
                    "selector_mode": selector_mode,
                    "feature": feat,
                    "selected": False,
                    "selected_pre_vif": False,
                    "drop_reason": "length_mismatch",
                    "adf_passed": False,
                })
                continue

            raw_non_null_ratio = float(np.isfinite(raw_values).mean())
            if raw_non_null_ratio < cfg.min_non_null_ratio:
                metrics_rows.append({
                    "mode": mode,
                    "selector_mode": selector_mode,
                    "feature": feat,
                    "selected": False,
                    "selected_pre_vif": False,
                    "drop_reason": "low_non_null_ratio",
                    "non_null_ratio": raw_non_null_ratio,
                    "adf_passed": False,
                })
                continue

            transform_res = to_stationary(
                raw_values,
                adf_pvalue_threshold=cfg.adf_pvalue_threshold,
                adf_max_points=cfg.adf_max_points,
            )

            if not transform_res.passed:
                metrics_rows.append({
                    "mode": mode,
                    "selector_mode": selector_mode,
                    "feature": feat,
                    "selected": False,
                    "selected_pre_vif": False,
                    "drop_reason": "adf_not_passed",
                    "non_null_ratio": raw_non_null_ratio,
                    "transform": transform_res.method,
                    "adf_raw_pvalue": transform_res.raw_adf_pvalue,
                    "adf_transformed_pvalue": transform_res.transformed_adf_pvalue,
                    "adf_passed": False,
                })
                continue

            is_macro = self._is_macro_feature(feat)
            is_core_macro = self._is_core_macro_feature(feat)

            score = None
            daily_feature_vector = pd.Series(dtype=np.float64)
            macro_pass = True
            if not cfg.stationary_only or cfg.compute_predictive_diagnostics:
                score, daily_feature_vector = score_feature(
                    feature_name=feat,
                    trade_dates=trade_dates,
                    target_values=target_values,
                    feature_values=transform_res.transformed,
                    min_cross_section_samples=cfg.min_cross_section_samples,
                    min_months=cfg.min_months,
                    min_abs_rank_ic=cfg.min_abs_rank_ic,
                    min_icir=cfg.min_icir,
                    min_positive_ratio=cfg.min_positive_ratio,
                    macro_dynamic_window=cfg.macro_dynamic_window,
                )

                if (not cfg.stationary_only) and is_macro and not is_core_macro:
                    macro_pass = (
                        np.isfinite(score.corr_abs_mean)
                        and np.isfinite(score.corr_direction_ratio)
                        and score.corr_abs_mean >= cfg.macro_min_abs_corr
                        and score.corr_direction_ratio >= cfg.macro_min_direction_ratio
                    )

            if cfg.stationary_only:
                selected_pre_vif = True
                drop_reason = ""
            else:
                selected_pre_vif = bool(score is not None and score.passed and macro_pass)
                if score is None or not score.passed:
                    drop_reason = "quality_threshold"
                elif not macro_pass:
                    drop_reason = "macro_dynamic_filter"
                else:
                    drop_reason = ""

            row = {
                "mode": mode,
                "selector_mode": selector_mode,
                "feature": feat,
                "selected": selected_pre_vif,
                "selected_pre_vif": selected_pre_vif,
                "drop_reason": drop_reason,
                "transform": transform_res.method,
                "adf_raw_pvalue": transform_res.raw_adf_pvalue,
                "adf_transformed_pvalue": transform_res.transformed_adf_pvalue,
                "non_null_ratio": raw_non_null_ratio if score is None else score.non_null_ratio,
                "valid_days": np.nan if score is None else score.valid_days,
                "valid_months": np.nan if score is None else score.valid_months,
                "ic_mean": np.nan if score is None else score.ic_mean,
                "rank_ic_mean": np.nan if score is None else score.rank_ic_mean,
                "rank_ic_std": np.nan if score is None else score.rank_ic_std,
                "icir": np.nan if score is None else score.icir,
                "positive_ratio": np.nan if score is None else score.positive_ratio,
                "corr_abs_mean": np.nan if score is None else score.corr_abs_mean,
                "corr_direction_ratio": np.nan if score is None else score.corr_direction_ratio,
                "is_macro": is_macro,
                "is_core_macro": is_core_macro,
                "adf_passed": True,
            }
            row["ranking_score"] = (
                np.nan
                if cfg.stationary_only
                else self._ranking_score(
                    rank_ic_mean=row.get("rank_ic_mean", np.nan),
                    icir=row.get("icir", np.nan),
                    positive_ratio=row.get("positive_ratio", np.nan),
                )
            )
            metrics_rows.append(row)

            if selected_pre_vif:
                preselected_features.append(feat)
                if not cfg.stationary_only and not daily_feature_vector.empty:
                    preselected_vectors[feat] = daily_feature_vector
                transform_specs_all[feat] = transform_res.method

            if i % 50 == 0 or i == len(candidate_features):
                logger.info(
                    f"GRU 特征筛选进度: {i}/{len(candidate_features)}，"
                    f"preselected={len(preselected_features)}"
                )

        metrics_df = pd.DataFrame(metrics_rows)

        if metrics_df.empty:
            raise RuntimeError("GRU 特征筛选失败：无可用候选结果")

        if cfg.stationary_only:
            selected_features = list(preselected_features)
            vif_scores = {}
            dropped_by_vif = []
        else:
            # 先按质量排序并截断 top_k
            ranked_preselected = metrics_df[metrics_df["selected_pre_vif"] == True].copy()  # noqa: E712
            ranked_preselected = ranked_preselected.sort_values(
                by=["ranking_score", "rank_ic_mean"],
                ascending=[False, False],
            )

            selected_features = ranked_preselected["feature"].tolist()
            if cfg.top_k > 0:
                selected_features = selected_features[: cfg.top_k]

            # VIF 过滤（仅对 top_k 运行）
            vif_scores = {}
            dropped_by_vif: List[str] = []
            if cfg.enable_vif and len(selected_features) > 1:
                selected_vectors = {
                    f: preselected_vectors[f]
                    for f in selected_features
                    if f in preselected_vectors
                }
                after_vif, vif_meta = apply_vif_filter(
                    selected_vectors,
                    threshold=cfg.vif_threshold,
                    min_samples=cfg.vif_min_samples,
                )
                selected_features = [f for f in selected_features if f in set(after_vif)]
                vif_scores = dict(vif_meta.get("vif_scores", {}))
                dropped_by_vif = list(vif_meta.get("dropped_by_vif", []))

        selected_set = set(selected_features)
        dropped_vif_set = set(dropped_by_vif)
        adf_pass_count = int(metrics_df["adf_passed"].fillna(False).astype(bool).sum())

        metrics_df["vif_score"] = metrics_df["feature"].map(vif_scores)
        metrics_df["selected"] = metrics_df["feature"].isin(selected_set)

        def _final_reason(row: pd.Series) -> str:
            feat = row.get("feature")
            if feat in dropped_vif_set:
                return "vif"
            if bool(row.get("selected", False)):
                return ""
            reason = row.get("drop_reason", "")
            return reason if isinstance(reason, str) and reason else "filtered_out"

        metrics_df["drop_reason"] = metrics_df.apply(_final_reason, axis=1)
        metrics_df["selected_rank"] = np.nan
        for rank, feat in enumerate(selected_features, start=1):
            metrics_df.loc[metrics_df["feature"] == feat, "selected_rank"] = rank

        transform_specs = {f: transform_specs_all[f] for f in selected_features if f in transform_specs_all}

        # 落盘工件
        selected_payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "selector_mode": selector_mode,
            "target_col": cfg.target_col,
            "selected_features": selected_features,
            "candidate_count": int(len(candidate_features)),
            "adf_pass_count": adf_pass_count,
            "config": asdict(cfg),
        }

        (output_dir / "selected_features.json").write_text(
            json.dumps(selected_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        (output_dir / "transform_specs.json").write_text(
            json.dumps(transform_specs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if cfg.stationary_only:
            metrics_df.sort_values(
                by=["selected", "feature"],
                ascending=[False, True],
                inplace=True,
            )
        else:
            metrics_df.sort_values(
                by=["selected", "ranking_score"],
                ascending=[False, False],
                inplace=True,
            )
        metrics_df.to_parquet(output_dir / "feature_metrics.parquet", index=False)

        logger.info(
            f"GRU 特征筛选完成: selected={len(selected_features)}, artifact_dir={output_dir}"
        )

        return FeatureSelectionResult(
            selected_features=selected_features,
            transform_specs=transform_specs,
            metrics_df=metrics_df,
            artifact_dir=output_dir,
            selector_mode=selector_mode,
            candidate_count=int(len(candidate_features)),
            adf_pass_count=adf_pass_count,
        )

    def _is_macro_feature(self, feature: str) -> bool:
        return any(feature.startswith(prefix) for prefix in self.config.macro_prefixes)

    def _is_core_macro_feature(self, feature: str) -> bool:
        return feature in set(self.config.core_macro_whitelist)

    def _selector_mode_name(self) -> str:
        return "stationary_only" if self.config.stationary_only else "predictive_filter"

    @staticmethod
    def _extract_payload_selector_mode(payload: dict) -> Optional[str]:
        selector_mode = payload.get("selector_mode")
        if isinstance(selector_mode, str) and selector_mode:
            return selector_mode

        cfg = payload.get("config")
        if isinstance(cfg, dict) and "stationary_only" in cfg:
            return "stationary_only" if bool(cfg.get("stationary_only")) else "predictive_filter"

        return None

    def _is_cache_compatible(self, payload: dict, mode: str) -> tuple[bool, str]:
        payload_mode = payload.get("mode")
        if payload_mode != mode:
            return False, f"mode mismatch (cached={payload_mode}, current={mode})"

        expected_selector_mode = self._selector_mode_name()
        cached_selector_mode = self._extract_payload_selector_mode(payload)
        if cached_selector_mode != expected_selector_mode:
            return (
                False,
                f"selector_mode mismatch (cached={cached_selector_mode}, current={expected_selector_mode})",
            )

        return True, "ok"

    @staticmethod
    def _ranking_score(rank_ic_mean: float, icir: float, positive_ratio: float) -> float:
        if not np.isfinite(rank_ic_mean) or not np.isfinite(icir):
            return -np.inf
        pos = positive_ratio if np.isfinite(positive_ratio) else 0.0
        return float(abs(rank_ic_mean) * max(icir, 0.0) * max(pos, 0.0))
