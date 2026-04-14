#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日度实盘推断脚本（LGB + GRU 双层融合）

输入：
- data/features/increment/date=YYYY-MM-DD/predict_lgb.parquet
- data/features/increment/date=YYYY-MM-DD/predict_gru.parquet

输出：
- reports/predict/live_predictions_YYYYMMDD.parquet
- reports/predict/live_predictions_YYYYMMDD.csv
- reports/predict/live_inference_report_YYYYMMDD.json
- reports/predict/live_inference_report_YYYYMMDD.md

融合逻辑：
1) LGB 内部：rolling + single_full（按配置权重）
2) GRU 内部：rolling + single_full（按配置权重）
3) 家族级：LGB + GRU（脚本参数权重）
4) 排名：三目标截面百分位统一为“越小越好”后加权
   - rank_ret_5d: ascending=True
   - excess_ret_5d/sharpe_5d: ascending=False
   - 默认权重: 0.6 / 0.2 / 0.2
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import GRUInferenceConfig, InferenceConfig
from src.models.gru import GRUInferenceEngine
from src.models.lgb import InferenceEngine as LGBInferenceEngine


def setup_logging() -> logging.Logger:
    """配置日志。"""
    log_dir = PROJECT_ROOT / "logs" / "models"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "live_inference.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="日度实盘推断（LGB + GRU）")

    parser.add_argument("--date", type=str, required=True, help="推断日期，支持 YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument(
        "--increment-root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "features" / "increment"),
        help="增量特征根目录（默认 data/features/increment）",
    )
    parser.add_argument("--lgb-path", type=str, default=None, help="可选：直接指定 predict_lgb.parquet 路径")
    parser.add_argument("--gru-path", type=str, default=None, help="可选：直接指定 predict_gru.parquet 路径")

    parser.add_argument(
        "--targets",
        nargs="+",
        default=["rank_ret_5d", "excess_ret_5d", "sharpe_5d"],
        help="推断目标列表",
    )
    parser.add_argument(
        "--score-weights",
        nargs=3,
        type=float,
        default=[0.6, 0.2, 0.2],
        metavar=("RANK_W", "EXCESS_W", "SHARPE_W"),
        help="最终排名三目标权重（默认 0.6 0.2 0.2）",
    )

    parser.add_argument("--lgb-weight", type=float, default=0.5, help="家族融合：LGB 权重（默认 0.5）")
    parser.add_argument("--gru-weight", type=float, default=0.5, help="家族融合：GRU 权重（默认 0.5）")

    parser.add_argument(
        "--rolling-weight-strategy",
        type=str,
        default="linear_recency",
        choices=["uniform", "linear_recency"],
        help="rolling 层内加权策略（默认 linear_recency）",
    )
    parser.add_argument("--seq-len", type=int, default=20, help="GRU 序列长度（默认 20）")
    parser.add_argument("--batch-size", type=int, default=2048, help="GRU 推断批大小（默认 2048）")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="GRU 推断设备")

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "predict"),
        help="输出目录（默认 reports/predict）",
    )
    parser.add_argument("--top-k", type=int, default=20, help="报告展示 TopK（默认 20）")

    return parser.parse_args()


def normalize_date(date_str: str) -> str:
    """标准化日期到 YYYY-MM-DD。"""
    raw = str(date_str).strip()
    if len(raw) == 8 and raw.isdigit():
        return datetime.strptime(raw, "%Y%m%d").strftime("%Y-%m-%d")
    return pd.to_datetime(raw).strftime("%Y-%m-%d")


def ensure_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """确保并规范 key 列。"""
    if "trade_date" not in df.columns or "ts_code" not in df.columns:
        raise ValueError("输入文件缺少必需主键列: trade_date / ts_code")

    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["ts_code"] = out["ts_code"].astype(str)
    out = out.dropna(subset=["trade_date", "ts_code"]).reset_index(drop=True)
    return out


def resolve_input_paths(args: argparse.Namespace, pred_date: str) -> Tuple[Path, Path]:
    increment_root = Path(args.increment_root)
    default_dir = increment_root / f"date={pred_date}"

    lgb_path = Path(args.lgb_path) if args.lgb_path else default_dir / "predict_lgb.parquet"
    gru_path = Path(args.gru_path) if args.gru_path else default_dir / "predict_gru.parquet"

    if not lgb_path.exists():
        raise FileNotFoundError(f"LGB 输入文件不存在: {lgb_path}")
    if not gru_path.exists():
        raise FileNotFoundError(f"GRU 输入文件不存在: {gru_path}")

    return lgb_path, gru_path


def run_lgb_inference(
    features_df: pd.DataFrame,
    targets: List[str],
    rolling_weight_strategy: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = InferenceConfig(
        target_cols=targets,
        rolling_weight_strategy=rolling_weight_strategy,
    )
    engine = LGBInferenceEngine(config=cfg)

    pred_df, breakdown = engine.predict_multi(
        X=features_df,
        target_cols=targets,
        with_breakdown=True,
    )

    out = features_df[["trade_date", "ts_code"]].copy()
    for target in targets:
        out[f"lgb_pred_{target}"] = pred_df[f"y_pred_{target}"].to_numpy(dtype=np.float64)

    return out, breakdown


def run_gru_inference(
    features_df: pd.DataFrame,
    targets: List[str],
    rolling_weight_strategy: str,
    seq_len: int,
    batch_size: int,
    device: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = GRUInferenceConfig(
        target_cols=targets,
        rolling_weight_strategy=rolling_weight_strategy,
    )
    engine = GRUInferenceEngine(config=cfg)
    engine.load_models(device=device)

    pred_df, breakdown = engine.predict_from_dataframe(
        df=features_df,
        target_cols=targets,
        seq_len=seq_len,
        batch_size=batch_size,
        with_breakdown=True,
    )

    out = pred_df[["trade_date", "ts_code"]].copy()
    out["trade_date"] = out["trade_date"].astype(str).str[:10]
    out["ts_code"] = out["ts_code"].astype(str)
    for target in targets:
        out[f"gru_pred_{target}"] = pred_df[f"y_pred_{target}"].to_numpy(dtype=np.float64)

    return out, breakdown


def fuse_model_families(
    merged_df: pd.DataFrame,
    targets: List[str],
    lgb_weight: float,
    gru_weight: float,
) -> pd.DataFrame:
    """融合 LGB/GRU 家族输出。缺失时保留可用家族分数。"""
    if lgb_weight < 0 or gru_weight < 0:
        raise ValueError("家族融合权重必须非负")
    if lgb_weight == 0 and gru_weight == 0:
        raise ValueError("家族融合权重不能同时为 0")

    df = merged_df.copy()

    for target in targets:
        l_col = f"lgb_pred_{target}"
        g_col = f"gru_pred_{target}"
        f_col = f"fused_pred_{target}"

        l_series = df[l_col] if l_col in df.columns else pd.Series(np.nan, index=df.index)
        g_series = df[g_col] if g_col in df.columns else pd.Series(np.nan, index=df.index)

        fused = pd.Series(np.nan, index=df.index, dtype=np.float64)

        both_mask = l_series.notna() & g_series.notna()
        if both_mask.any():
            denom = lgb_weight + gru_weight
            fused.loc[both_mask] = (
                l_series.loc[both_mask] * lgb_weight + g_series.loc[both_mask] * gru_weight
            ) / denom

        l_only = l_series.notna() & g_series.isna()
        g_only = l_series.isna() & g_series.notna()
        fused.loc[l_only] = l_series.loc[l_only]
        fused.loc[g_only] = g_series.loc[g_only]

        df[f_col] = fused

    return df


def apply_percentile_ranking(
    df: pd.DataFrame,
    targets: List[str],
    score_weights: List[float],
) -> pd.DataFrame:
    """
    百分位评分规则：
    - rank_ret_5d: ascending=True（值越小越优）
    - excess_ret_5d / sharpe_5d: ascending=False（值越大越优 -> 百分位后小分优）
    """
    if len(targets) != 3:
        raise ValueError("当前排名逻辑要求恰好 3 个目标（rank/excess/sharpe）")
    if len(score_weights) != 3:
        raise ValueError("score_weights 长度必须为 3")

    out = df.copy()

    # 方向约定
    direction_map = {
        "rank_ret_5d": True,
        "excess_ret_5d": False,
        "sharpe_5d": False,
    }

    score_cols: List[str] = []
    for target in targets:
        pred_col = f"fused_pred_{target}"
        if pred_col not in out.columns:
            raise ValueError(f"缺少融合预测列: {pred_col}")

        score_col = f"score_{target}"
        score_cols.append(score_col)

        ascending = direction_map.get(target)
        if ascending is None:
            if target.startswith("rank"):
                ascending = True
            elif target.startswith("excess") or target.startswith("sharpe"):
                ascending = False
            else:
                ascending = False

        out[score_col] = np.nan
        valid_mask = out[pred_col].notna()
        if valid_mask.any():
            out.loc[valid_mask, score_col] = out.loc[valid_mask, pred_col].rank(
                method="average",
                pct=True,
                ascending=ascending,
            )

    # 任一目标缺失 -> 不参与最终排名
    valid_rank_mask = np.logical_and.reduce([out[c].notna().values for c in score_cols])
    out["is_rank_valid"] = valid_rank_mask

    out["final_score"] = np.nan
    if valid_rank_mask.any():
        weighted_score = np.zeros(valid_rank_mask.sum(), dtype=np.float64)
        for w, c in zip(score_weights, score_cols):
            weighted_score += float(w) * out.loc[valid_rank_mask, c].to_numpy(dtype=np.float64)

        out.loc[valid_rank_mask, "final_score"] = weighted_score

        # 分数越小越优
        out.loc[valid_rank_mask, "final_rank"] = out.loc[valid_rank_mask, "final_score"].rank(
            method="first",
            ascending=True,
        )
        out.loc[valid_rank_mask, "final_rank_pct"] = out.loc[valid_rank_mask, "final_score"].rank(
            method="average",
            pct=True,
            ascending=True,
        )
    else:
        out["final_rank"] = np.nan
        out["final_rank_pct"] = np.nan

    return out


def _to_python_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_types(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_python_types(v) for v in obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_markdown_report(
    pred_date: str,
    result_df: pd.DataFrame,
    targets: List[str],
    score_weights: List[float],
    lgb_weight: float,
    gru_weight: float,
    lgb_breakdown: Dict[str, Any],
    gru_breakdown: Dict[str, Any],
    top_k: int,
) -> str:
    valid_df = result_df[result_df["is_rank_valid"]].copy()
    valid_count = len(valid_df)
    total_count = len(result_df)

    top_df = valid_df.nsmallest(top_k, "final_rank") if valid_count > 0 else valid_df
    bottom_df = valid_df.nlargest(top_k, "final_rank") if valid_count > 0 else valid_df

    top_text = top_df[["ts_code", "final_score", "final_rank"]].to_string(index=False) if not top_df.empty else "(empty)"
    bottom_text = bottom_df[["ts_code", "final_score", "final_rank"]].to_string(index=False) if not bottom_df.empty else "(empty)"

    lines = [
        f"# Live Inference Report - {pred_date}",
        "",
        "## 运行摘要",
        f"- 总样本数: {total_count}",
        f"- 可参与排名样本数: {valid_count}",
        f"- 排名有效率: {valid_count / total_count:.2%}" if total_count > 0 else "- 排名有效率: N/A",
        "",
        "## 融合参数",
        f"- 目标列表: {targets}",
        f"- 家族融合权重: LGB={lgb_weight}, GRU={gru_weight}",
        f"- 百分位评分权重: {dict(zip(targets, score_weights))}",
        "- 方向约定: rank_ret_5d(ascending), excess_ret_5d(descending), sharpe_5d(descending)",
        "",
        f"## Top {top_k}（final_score 越小越优）",
        "```text",
        top_text,
        "```",
        "",
        f"## Bottom {top_k}（final_score 越大越劣）",
        "```text",
        bottom_text,
        "```",
        "",
        "## 可解释拆解（摘要）",
        "- LGB: rolling + single_full 两层融合",
        "- GRU: rolling + single_full 两层融合（含 feature_selection 工件约束）",
        "",
        "### LGB Breakdown (JSON)",
        "```json",
        json.dumps(_to_python_types(lgb_breakdown), ensure_ascii=False, indent=2),
        "```",
        "",
        "### GRU Breakdown (JSON)",
        "```json",
        json.dumps(_to_python_types(gru_breakdown), ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    logger = setup_logging()

    pred_date = normalize_date(args.date)
    targets = list(args.targets)
    score_weights = [float(x) for x in args.score_weights]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("🚀 Live Inference 启动")
    logger.info("推断日期: %s", pred_date)
    logger.info("目标: %s", targets)
    logger.info("=" * 70)

    try:
        lgb_path, gru_path = resolve_input_paths(args, pred_date)
        logger.info("LGB 特征输入: %s", lgb_path)
        logger.info("GRU 特征输入: %s", gru_path)

        lgb_features = ensure_key_columns(pd.read_parquet(lgb_path))
        gru_features = ensure_key_columns(pd.read_parquet(gru_path))

        # LGB 仅使用当日截面
        lgb_daily = lgb_features[lgb_features["trade_date"] == pred_date].copy()
        if lgb_daily.empty:
            raise ValueError(f"predict_lgb 中找不到目标日期 {pred_date} 的样本")

        # GRU 从最近窗口推断后再截到当日
        lgb_pred_df, lgb_breakdown = run_lgb_inference(
            features_df=lgb_daily,
            targets=targets,
            rolling_weight_strategy=args.rolling_weight_strategy,
        )

        gru_pred_all_df, gru_breakdown = run_gru_inference(
            features_df=gru_features,
            targets=targets,
            rolling_weight_strategy=args.rolling_weight_strategy,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=args.device,
        )
        gru_pred_df = gru_pred_all_df[gru_pred_all_df["trade_date"] == pred_date].copy()
        if gru_pred_df.empty:
            raise ValueError(f"GRU 推断结果中找不到目标日期 {pred_date}（请检查 seq_len 与历史窗口）")

        merged = pd.merge(
            lgb_pred_df,
            gru_pred_df,
            on=["trade_date", "ts_code"],
            how="outer",
        )

        fused = fuse_model_families(
            merged_df=merged,
            targets=targets,
            lgb_weight=float(args.lgb_weight),
            gru_weight=float(args.gru_weight),
        )

        ranked = apply_percentile_ranking(
            df=fused,
            targets=targets,
            score_weights=score_weights,
        )

        ranked = ranked.sort_values(["is_rank_valid", "final_rank", "ts_code"], ascending=[False, True, True])

        date_tag = pred_date.replace("-", "")
        parquet_path = output_dir / f"live_predictions_{date_tag}.parquet"
        csv_path = output_dir / f"live_predictions_{date_tag}.csv"
        json_path = output_dir / f"live_inference_report_{date_tag}.json"
        md_path = output_dir / f"live_inference_report_{date_tag}.md"

        ranked.to_parquet(parquet_path, index=False)
        ranked.to_csv(csv_path, index=False, encoding="utf-8")

        summary = {
            "pred_date": pred_date,
            "targets": targets,
            "score_weights": dict(zip(targets, score_weights)),
            "family_weights": {
                "lgb": float(args.lgb_weight),
                "gru": float(args.gru_weight),
            },
            "input_paths": {
                "predict_lgb": str(lgb_path),
                "predict_gru": str(gru_path),
            },
            "output_paths": {
                "predictions_parquet": str(parquet_path),
                "predictions_csv": str(csv_path),
                "report_json": str(json_path),
                "report_md": str(md_path),
            },
            "sample_stats": {
                "lgb_samples": int(len(lgb_pred_df)),
                "gru_samples": int(len(gru_pred_df)),
                "merged_samples": int(len(merged)),
                "rank_valid_samples": int(ranked["is_rank_valid"].sum()),
                "rank_valid_ratio": float(ranked["is_rank_valid"].mean()) if len(ranked) > 0 else 0.0,
            },
            "lgb_breakdown": _to_python_types(lgb_breakdown),
            "gru_breakdown": _to_python_types(gru_breakdown),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        report_md = generate_markdown_report(
            pred_date=pred_date,
            result_df=ranked,
            targets=targets,
            score_weights=score_weights,
            lgb_weight=float(args.lgb_weight),
            gru_weight=float(args.gru_weight),
            lgb_breakdown=lgb_breakdown,
            gru_breakdown=gru_breakdown,
            top_k=int(args.top_k),
        )
        md_path.write_text(report_md, encoding="utf-8")

        logger.info("✅ 推断完成")
        logger.info("输出 parquet: %s", parquet_path)
        logger.info("输出 csv: %s", csv_path)
        logger.info("输出 json 报告: %s", json_path)
        logger.info("输出 md 报告: %s", md_path)
        return 0

    except Exception as e:
        logger.error("❌ Live Inference 失败: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
