"""
训练调度主脑（trainer.py）

核心职责:
- 加载数据（内存优化：pyarrow schema → 按需读列 → 一次性张量化）
- 动态识别特征列（排除法）
- 编排 GRUTimeSeriesSplitter + GRUTensorDataset + GRUModel
- 滚窗迭代 + OOF 拼装 + 模型持久化
- Rolling 模式: 单种子，N 折 OOF
- Single_Full 模式: 多种子融合（5种子 × 1折）
- 实盘推断接口

内存优化说明（2026.02）:
- 不使用 cuDF→pandas 高峰内存路径
- pyarrow.read_schema 先获取列名，仅读取必要列
- 特征/标签一次性转为 torch.Tensor，立即释放 DataFrame
- 所有 GRUTensorDataset 共享同一组 Tensor（零拷贝），多 Fold 间不重复分配
"""

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from ..config import (
    GRUConfig,
    GRUDataConfig,
    GRUNetworkConfig,
    GRUSplitConfig,
    GRUTrainConfig,
    GRUInferenceConfig,
    ID_COLS,
    LABEL_COLS,
    AUX_COLS,
    CATEGORICAL_FEATURES,
)
from ..metrics import infer_pnl_return_col
from .feature_selection import GRUFeatureSelectionConfig, GRUFeatureSelector
from .dataset import (
    GRUFoldInfo,
    GRUTimeSeriesSplitter,
    GRUTensorDataset,
    create_dataloader,
)
from .gru_model import GRUModel, set_seed
from .report_generator import GRUReportGenerator

logger = logging.getLogger(__name__)


class _PermutedFeatureDataset(Dataset):
    """在样本级按预设映射替换单一连续特征序列，用于置换重要性评估。"""

    def __init__(self, base_dataset: GRUTensorDataset, permutation_indices: np.ndarray, feature_idx: int):
        self.base_dataset = base_dataset
        self.permutation_indices = permutation_indices
        self.feature_idx = feature_idx
        self.device = getattr(base_dataset, "device", "cpu")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x_cont, x_cat, y = self.base_dataset[idx]
        src_idx = int(self.permutation_indices[idx])
        src_x_cont, _, _ = self.base_dataset[src_idx]

        x_cont_perm = x_cont.clone()
        x_cont_perm[:, self.feature_idx] = src_x_cont[:, self.feature_idx]
        return x_cont_perm, x_cat, y


class GRUTrainer:
    """
    GRU 训练调度器

    负责:
    1. 加载数据 + 动态特征识别
    2. 多目标 GRU 并行训练（一次 fit 同时拟合所有目标）
    3. 三模式滚窗迭代 + OOF 拼装
    4. 模型持久化落盘
    5. 生成 oof_predictions.parquet

    内存管理:
    - feature_tensor / label_tensor 在 load_data() 中一次性预计算并放到 GPU
    - 所有 Fold 的 GRUTensorDataset 共享同一组 Tensor（零拷贝引用）
    - 轻量 self.df 仅保留 ts_code + trade_date（供 Splitter 使用）

    Example:
        >>> trainer = GRUTrainer(config)
        >>> trainer.load_data()
        >>> oof_df = trainer.train(mode="rolling")
    """

    def __init__(self, config: Optional[GRUConfig] = None):
        self.config = config or GRUConfig.default()
        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: Optional[List[str]] = None
        self.cat_feature_cols: List[str] = []
        self.cat_cardinalities: List[int] = []
        self.cat_embedding_dims: List[int] = []
        self.feature_selection_result = None

        # 预计算的共享张量（load_data 中创建）
        self.feature_tensor: Optional[torch.Tensor] = None
        self.cat_feature_tensor: Optional[torch.Tensor] = None
        self.label_tensor: Optional[torch.Tensor] = None
        self.dates_arr: Optional[np.ndarray] = None
        self.codes_arr: Optional[np.ndarray] = None

        # 供外部读取的元信息
        self.loaded_columns: List[str] = []
        self.target_return_col_map: Dict[str, str] = {}
        self.pnl_return_arrays: Dict[str, np.ndarray] = {}

        logger.info("GRUTrainer 初始化完成")

    def load_data(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        加载训练数据（内存优化版本 v2 — pyarrow-first 管线）

        峰值内存控制:
        - 在 pyarrow 列式存储层面逐列 float64 → float32，
          避免 pandas float64 DataFrame 的 sort/copy 造成 2× 内存翻倍
        - 逐列提取到预分配 numpy 数组，避免 pandas 块整合 (block consolidation) 的额外拷贝

        流程:
        1. pyarrow.read_schema 读取列名（不读数据）
        2. get_gru_selected_features 确定特征列
        3. pyarrow.read_table 读取 → 立即在 Arrow 层降精度 float64→float32
        4. Arrow → pandas（已是 float32，~3.9 GB 而非 ~7.3 GB）
        5. sort + 日期过滤（inplace，峰值 ~7.8 GB 而非 14.6 GB）
        6. 逐列填充预分配 numpy 数组 → GPU Tensor
        7. 释放 DataFrame，仅保留轻量 df 供 Splitter 用

        Returns:
            df: 轻量 DataFrame（仅含 ts_code + trade_date）
        """
        data_path = path or self.config.data.data_path
        target_cols = self.config.data.target_cols
        device = "cuda" if self.config.data.use_gpu and torch.cuda.is_available() else "cpu"

        logger.info(f"加载 GRU 训练数据: {data_path}")

        # ---- Step 1: 读取 parquet schema 获取列名（不读数据） ----
        schema = pq.read_schema(str(data_path))
        all_columns = schema.names
        self.loaded_columns = all_columns
        logger.info(f"Parquet schema: {len(all_columns)} 列")

        # ---- Step 2: GRU 专属流式特征筛选（全样本一次拟合） ----
        mode = self.config.split.mode
        if "rank_ret_5d" not in all_columns:
            raise ValueError(
                "当前 GRU 特征筛选固定使用 rank_ret_5d，"
                "但输入数据中未找到该列"
            )

        selector = GRUFeatureSelector(
            config=GRUFeatureSelectionConfig(
                target_col="rank_ret_5d",
                stationary_only=bool(getattr(self.config.train, "stationary_only", True)),
            )
        )
        selection_dir = self.config.train.save_dir / mode / "feature_selection"
        exclude_cols = list(set(ID_COLS + LABEL_COLS + AUX_COLS + CATEGORICAL_FEATURES))

        self.feature_selection_result = selector.fit_or_load_artifacts(
            data_path=data_path,
            all_columns=all_columns,
            mode=mode,
            output_dir=selection_dir,
            exclude_cols=exclude_cols,
        )
        self.feature_cols = list(self.feature_selection_result.selected_features)

        if not self.feature_cols:
            raise ValueError("GRU 特征筛选结果为空，请检查筛选阈值或输入数据质量")

        logger.info(
            f"GRU 特征筛选结果: {len(self.feature_cols)} 列, artifact={selection_dir}"
        )

        fs_mode = getattr(self.feature_selection_result, "selector_mode", "unknown")
        fs_candidates = int(getattr(self.feature_selection_result, "candidate_count", 0) or 0)
        fs_adf_pass = int(getattr(self.feature_selection_result, "adf_pass_count", 0) or 0)
        logger.info(
            "GRU 筛选模式审计: selector_mode=%s, predictive_filter=%s, stationary_filter=ON, "
            "candidate_count=%d, adf_pass_count=%d, selected_count=%d",
            fs_mode,
            "OFF" if fs_mode == "stationary_only" else "ON",
            fs_candidates,
            fs_adf_pass,
            len(self.feature_cols),
        )

        # ---- Step 3: 确定读取列 ----
        self.cat_feature_cols = [c for c in CATEGORICAL_FEATURES if c in all_columns]

        valid_target_cols = [c for c in target_cols if c in all_columns]
        if not valid_target_cols:
            raise ValueError(f"目标列在数据中均不存在: {target_cols}")
        if len(valid_target_cols) < len(target_cols):
            missing = set(target_cols) - set(valid_target_cols)
            logger.warning(f"目标列未找到: {missing}")
        self.config.data.target_cols = valid_target_cols
        target_cols = valid_target_cols

        self.target_return_col_map = {}
        return_cols: List[str] = []
        for target_col in target_cols:
            inferred_return_col = infer_pnl_return_col(target_col, available_cols=all_columns)
            if inferred_return_col is None:
                inferred_return_col = target_col
                logger.warning(
                    "目标 %s 未找到专用收益列，回退到目标列本身作为 pnl_return",
                    target_col,
                )
            self.target_return_col_map[target_col] = inferred_return_col
            return_cols.append(inferred_return_col)

        logger.info(f"GRU 目标收益映射: {self.target_return_col_map}")

        read_cols = list(dict.fromkeys(
            ['ts_code', 'trade_date'] + self.feature_cols + self.cat_feature_cols + target_cols + return_cols
        ))
        read_cols = [c for c in read_cols if c in all_columns]

        logger.info(f"读取列数: {len(read_cols)} / {len(all_columns)}")

        # ---- Step 4: pyarrow 读取 + 列级降精度 ----
        # 在 Arrow 列式存储层面逐列 float64→float32，
        # 峰值仅多 1 列内存（~23 MB），而非整个 DataFrame 翻倍
        t0 = time.time()
        table = pq.read_table(str(data_path), columns=read_cols)
        logger.info(
            f"PyArrow 读取完成: {table.num_rows:,} 行, "
            f"耗时={time.time() - t0:.1f}s"
        )

        t1 = time.time()
        for i in range(len(table.schema)):
            field = table.schema.field(i)
            if field.type == pa.float64():
                table = table.set_column(
                    i, field.name,
                    table.column(i).cast(pa.float32()),
                )
        gc.collect()
        logger.info(f"Arrow float64→float32 完成, 耗时={time.time() - t1:.1f}s")

        # Arrow → pandas（此时数值列已是 float32，约 3.9 GB）
        t1 = time.time()
        df = table.to_pandas()
        del table
        gc.collect()
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(
            f"Pandas 转换完成: shape={df.shape}, 内存={mem_mb:.0f}MB, "
            f"耗时={time.time() - t1:.1f}s"
        )

        # ---- Step 5: 基本预处理 ----
        if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
            df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 排序（inplace 减少一次完整拷贝）
        df.sort_values(['ts_code', 'trade_date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 日期过滤（仅截断 end_date，保留预热窗口）
        data_end = pd.Timestamp(self.config.data.data_end_date)
        mask = df['trade_date'] <= data_end
        if not mask.all():
            original_len = len(df)
            df = df[mask].reset_index(drop=True)
            logger.info(f"日期过滤: {original_len:,} -> {len(df):,}")
            gc.collect()
        else:
            logger.info(f"日期过滤: 无需截断, 保留全部 {len(df):,} 行")

        # ---- Step 5.5: 应用筛选阶段确定的时序变换 ----
        transform_specs = {}
        if self.feature_selection_result is not None:
            transform_specs = getattr(self.feature_selection_result, "transform_specs", {}) or {}

        if transform_specs:
            diff_cols = [c for c, m in transform_specs.items() if m == "diff" and c in df.columns]
            pct_cols = [c for c, m in transform_specs.items() if m == "pct_change" and c in df.columns]

            if diff_cols:
                df[diff_cols] = df.groupby('ts_code', sort=False)[diff_cols].diff()
            if pct_cols:
                df[pct_cols] = df.groupby('ts_code', sort=False)[pct_cols].pct_change()

            transformed_cols = diff_cols + pct_cols
            if transformed_cols:
                df[transformed_cols] = (
                    df[transformed_cols]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
            logger.info(
                f"应用特征变换: diff={len(diff_cols)}, pct_change={len(pct_cols)}"
            )

        # 过滤非数值特征列
        numeric_cols = df.select_dtypes(
            include=['float32', 'float64', 'Float32', 'Float64',
                     'int8', 'int16', 'int32', 'int64', 'Int32', 'Int64']
        ).columns.tolist()
        self.feature_cols = [c for c in self.feature_cols if c in numeric_cols]

        self.cat_feature_cols = [c for c in self.cat_feature_cols if c in df.columns]

        logger.info(
            f"连续特征列数: {len(self.feature_cols)}, 类别特征列数: {len(self.cat_feature_cols)}, 目标列: {target_cols}"
        )

        # ---- Step 6: 提取元数据（轻量，CPU）----
        self.dates_arr = df['trade_date'].values
        self.codes_arr = df['ts_code'].astype(str).values

        # 轻量 df 供 Splitter 使用（仅保留时间与代码列）
        self.df = df[['ts_code', 'trade_date']].copy()

        # ---- Step 7: 逐列填充预分配 numpy 数组 ----
        # 避免 df[cols].values 触发 pandas 块整合（block consolidation），
        # 块整合会分配一个等大的连续内存块，导致峰值 = df + numpy 双份
        logger.info("逐列提取特征到 numpy float32...")
        t1 = time.time()
        n_rows = len(df)

        features_np = np.empty((n_rows, len(self.feature_cols)), dtype=np.float32)
        for i, c in enumerate(self.feature_cols):
            col_data = df[c].values
            if col_data.dtype != np.float32:
                col_data = col_data.astype(np.float32)
            features_np[:, i] = col_data
        np.nan_to_num(features_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        labels_np = np.empty((n_rows, len(target_cols)), dtype=np.float32)
        for i, c in enumerate(target_cols):
            col_data = df[c].values
            if col_data.dtype != np.float32:
                col_data = col_data.astype(np.float32)
            labels_np[:, i] = col_data

        # 评估使用的真实收益列（按目标分别缓存，避免后续被 DataFrame block 引用牵连）
        self.pnl_return_arrays = {}
        for target_col in target_cols:
            ret_col = self.target_return_col_map.get(target_col, target_col)
            if ret_col not in df.columns:
                ret_col = target_col
            self.pnl_return_arrays[target_col] = np.asarray(df[ret_col].values, dtype=np.float32).copy()

        # 重要：标签 NaN（如 T+1 不可交易掩码）必须保留，
        # 后续在 Loss 中通过 masked loss 忽略无效样本。

        cat_np = None
        self.cat_cardinalities = []
        self.cat_embedding_dims = []

        if self.cat_feature_cols:
            cat_np = np.empty((n_rows, len(self.cat_feature_cols)), dtype=np.int64)
            for i, c in enumerate(self.cat_feature_cols):
                cat_series = pd.to_numeric(df[c], errors='coerce').fillna(-1).astype(np.int64)
                cat_values = cat_series.values + 1  # unknown(-1) -> 0
                cat_values = np.maximum(cat_values, 0)

                max_val = int(cat_values.max()) if len(cat_values) else 0
                cardinality = max(2, max_val + 1)
                cat_values = np.clip(cat_values, 0, cardinality - 1)

                cat_np[:, i] = cat_values
                self.cat_cardinalities.append(cardinality)
                self.cat_embedding_dims.append(max(4, min(32, int((cardinality + 1) // 2))))

        # 释放 DataFrame（最大内存节省点）
        del df
        gc.collect()
        logger.info(
            f"numpy 提取完成: features={features_np.shape}, "
            f"labels={labels_np.shape}, 耗时={time.time() - t1:.1f}s"
        )

        # ---- Step 8: 移入设备（GPU/CPU） ----
        logger.info(f"加载张量到 {device}...")
        t1 = time.time()

        self.feature_tensor = torch.from_numpy(features_np).to(device)
        del features_np
        gc.collect()

        self.label_tensor = torch.from_numpy(labels_np).to(device)
        del labels_np
        gc.collect()

        if cat_np is not None:
            self.cat_feature_tensor = torch.from_numpy(cat_np).to(device=device, dtype=torch.long)
            del cat_np
            gc.collect()
        else:
            self.cat_feature_tensor = None

        # ---- Step 9: 更新配置 ----
        self.config.network.num_features = len(self.feature_cols) + int(sum(self.cat_embedding_dims))
        self.config.network.num_targets = len(target_cols)

        feat_mb = self.feature_tensor.numel() * 4 / 1024 / 1024
        cat_mb = 0.0
        if self.cat_feature_tensor is not None:
            cat_mb = self.cat_feature_tensor.numel() * 8 / 1024 / 1024  # int64
        label_mb = self.label_tensor.numel() * 4 / 1024 / 1024
        logger.info(
            f"数据加载完成: rows={len(self.dates_arr):,}, "
            f"cont_features={len(self.feature_cols)}, cat_features={len(self.cat_feature_cols)}, targets={len(target_cols)}, "
            f"tensor_mem={feat_mb + cat_mb + label_mb:.0f}MB "
            f"(cont={feat_mb:.0f}MB, cat={cat_mb:.0f}MB, labels={label_mb:.0f}MB), "
            f"device={device}, 总耗时={time.time() - t0:.1f}s"
        )

        if self.cat_feature_cols:
            logger.info(
                f"类别特征基数: {dict(zip(self.cat_feature_cols, self.cat_cardinalities))}"
            )

        return self.df

    @staticmethod
    def _calc_mean_rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
        df = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(dates),
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )

        daily_rank_ic = []
        for _, g in df.groupby("trade_date", sort=True):
            if len(g) < 3:
                continue
            if g["y_true"].nunique(dropna=True) < 2 or g["y_pred"].nunique(dropna=True) < 2:
                continue
            ric = g["y_true"].corr(g["y_pred"], method="spearman")
            if np.isfinite(ric):
                daily_rank_ic.append(float(ric))

        if not daily_rank_ic:
            return float("nan")
        return float(np.mean(daily_rank_ic))

    @staticmethod
    def _build_cross_section_permutation_indices(dates: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(dates), dtype=np.int64)

        df = pd.DataFrame({"idx": indices, "date": dates})
        for _, g in df.groupby("date", sort=False):
            idx = g["idx"].to_numpy(dtype=np.int64)
            if len(idx) <= 1:
                continue
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            indices[idx] = shuffled

        return indices

    def _compute_permutation_importance(
        self,
        model: GRUModel,
        valid_ds: GRUTensorDataset,
        baseline_preds: np.ndarray,
        rank_channel_idx: int,
        max_features: int = 30,
        seed: int = 42,
    ) -> pd.DataFrame:
        if baseline_preds.size == 0 or not self.feature_cols:
            return pd.DataFrame(columns=["feature", "baseline_rank_ic", "permuted_rank_ic", "rank_ic_drop"])

        dates = valid_ds.get_all_dates()
        y_true_rank = valid_ds.labels[valid_ds.valid_indices, rank_channel_idx].detach().cpu().numpy()
        y_pred_rank = baseline_preds[:, rank_channel_idx]

        baseline_rank_ic = self._calc_mean_rank_ic(y_true_rank, y_pred_rank, dates)
        perm_indices = self._build_cross_section_permutation_indices(dates=dates, seed=seed)

        feature_candidates = self.feature_cols[: max_features if max_features > 0 else len(self.feature_cols)]
        rows = []

        for feat_idx, feat in enumerate(feature_candidates):
            perm_ds = _PermutedFeatureDataset(
                base_dataset=valid_ds,
                permutation_indices=perm_indices,
                feature_idx=feat_idx,
            )
            perm_loader = create_dataloader(
                perm_ds,
                batch_size=self.config.train.batch_size,
                shuffle=False,
            )

            perm_preds = model.predict(perm_loader)
            perm_rank_ic = self._calc_mean_rank_ic(
                y_true=y_true_rank,
                y_pred=perm_preds[:, rank_channel_idx],
                dates=dates,
            )

            rank_ic_drop = baseline_rank_ic - perm_rank_ic if np.isfinite(perm_rank_ic) else np.nan
            rows.append(
                {
                    "feature": feat,
                    "baseline_rank_ic": baseline_rank_ic,
                    "permuted_rank_ic": perm_rank_ic,
                    "rank_ic_drop": rank_ic_drop,
                }
            )

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values("rank_ic_drop", ascending=False).reset_index(drop=True)
            out["importance_rank"] = np.arange(1, len(out) + 1)
        return out

    def _build_feature_selection_summary(self) -> Dict[str, Any]:
        """构建特征筛选摘要（用于训练报告）。"""
        result = self.feature_selection_result
        if result is None:
            return {}

        selected_features = list(getattr(result, "selected_features", []) or [])
        transform_specs = dict(getattr(result, "transform_specs", {}) or {})
        metrics_df = getattr(result, "metrics_df", None)

        artifact_dir = Path(getattr(result, "artifact_dir", "")) if getattr(result, "artifact_dir", None) else None
        artifact_files = {}
        if artifact_dir is not None:
            artifact_files = {
                "selected_features": str(artifact_dir / "selected_features.json"),
                "transform_specs": str(artifact_dir / "transform_specs.json"),
                "feature_metrics": str(artifact_dir / "feature_metrics.parquet"),
            }

        summary: Dict[str, Any] = {
            "artifact_dir": str(artifact_dir) if artifact_dir is not None else "",
            "artifact_files": artifact_files,
            "selector_mode": str(getattr(result, "selector_mode", "")),
            "selected_count": int(len(selected_features)),
            "selected_preview": selected_features[:10],
            "selected_transform_distribution": {},
            "drop_reason_distribution": {},
            "adf_pass_count": int(getattr(result, "adf_pass_count", 0) or 0),
        }

        if not isinstance(metrics_df, pd.DataFrame) or metrics_df.empty:
            if transform_specs:
                transform_series = pd.Series(list(transform_specs.values())).fillna("identity").replace("", "identity")
                summary["selected_transform_distribution"] = {
                    str(k): int(v) for k, v in transform_series.value_counts().to_dict().items()
                }
            summary["candidate_count"] = int(getattr(result, "candidate_count", len(selected_features)) or len(selected_features))
            summary["selected_ratio"] = 1.0 if selected_features else 0.0
            return summary

        candidate_count = int(metrics_df["feature"].nunique()) if "feature" in metrics_df.columns else int(len(metrics_df))
        if "selected" in metrics_df.columns:
            selected_mask = metrics_df["selected"].fillna(False).astype(bool)
            selected_count = int(selected_mask.sum())
        else:
            selected_mask = pd.Series(False, index=metrics_df.index)
            selected_count = int(len(selected_features))

        selected_ratio = float(selected_count / candidate_count) if candidate_count > 0 else 0.0
        summary["candidate_count"] = candidate_count
        summary["selected_count"] = selected_count
        summary["selected_ratio"] = selected_ratio

        if "adf_passed" in metrics_df.columns:
            summary["adf_pass_count"] = int(metrics_df["adf_passed"].fillna(False).astype(bool).sum())

        if "drop_reason" in metrics_df.columns:
            reason_series = metrics_df.loc[~selected_mask, "drop_reason"].fillna("").replace("", "filtered_out")
            summary["drop_reason_distribution"] = {
                str(k): int(v) for k, v in reason_series.value_counts().head(10).to_dict().items()
            }

        if "transform" in metrics_df.columns:
            transform_series = metrics_df.loc[selected_mask, "transform"].fillna("identity").replace("", "identity")
        else:
            transform_series = pd.Series([transform_specs.get(f, "identity") for f in selected_features])

        if not transform_series.empty:
            summary["selected_transform_distribution"] = {
                str(k): int(v) for k, v in transform_series.value_counts().to_dict().items()
            }

        return summary

    def train(
        self,
        mode: Optional[str] = None,
        save_models: bool = True,
        save_oof: bool = True,
        generate_report: bool = True,
    ) -> pd.DataFrame:
        """
        训练入口

        GRU 是多任务并行 → 不需要 for target in target_cols 循环,
        直接开始滚窗迭代与 OOF 拼装.

        Args:
            mode: 训练模式 (rolling / expanding / single_full)
            save_models: 是否持久化模型
            save_oof: 是否保存 OOF
            generate_report: 是否生成训练报告

        Returns:
            oof_df: 全局 OOF 预测 DataFrame
        """
        if self.df is None:
            self.load_data()

        mode = mode or self.config.split.mode
        target_cols = self.config.data.target_cols

        logger.info("=" * 60)
        logger.info(f"GRU 训练启动: mode={mode}, targets={target_cols}")
        logger.info("=" * 60)

        if mode == "single_full":
            oof_df, fold_train_info = self._train_single_full(save_models, save_oof)
        else:
            oof_df, fold_train_info = self._train_fold_mode(mode, save_models, save_oof)

        # 生成训练报告
        if generate_report and not oof_df.empty:
            try:
                report_gen = GRUReportGenerator(
                    config=self.config,
                    oof_df=oof_df,
                    fold_train_info=fold_train_info,
                )
                report_path = report_gen.generate_report()
                logger.info(f"训练报告已生成: {report_path}")
            except Exception as e:
                logger.warning(f"训练报告生成失败: {e}", exc_info=True)

        return oof_df

    def _train_fold_mode(
        self,
        mode: str,
        save_models: bool,
        save_oof: bool,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Rolling / Expanding 模式训练

        Returns:
            (oof_df, fold_train_info)
        """
        config = self.config
        target_cols = config.data.target_cols
        seed = config.train.seed
        device = str(self.feature_tensor.device)
        set_seed(seed)

        feature_selection_summary = self._build_feature_selection_summary()

        fold_train_info: List[Dict[str, Any]] = []

        # 1) 初始化切分器（传入逻辑日期边界）
        splitter = GRUTimeSeriesSplitter(
            df=self.df,
            target_cols=target_cols,
            config=config.split,
            seq_len=config.data.seq_len,
            data_start_date=config.data.data_start_date,
            data_end_date=config.data.data_end_date,
        )

        oof_list = []

        # 2) 遍历 Fold
        for fold_info in splitter.split(mode=mode):
            fold_idx = fold_info.fold_idx
            logger.info(f"--- Fold {fold_idx} ---")

            # 构建训练/验证 Dataset（共享张量，零拷贝）
            train_ds = GRUTensorDataset(
                cont_features=self.feature_tensor,
                cat_features=self.cat_feature_tensor,
                labels=self.label_tensor,
                dates=self.dates_arr,
                codes=self.codes_arr,
                indices=fold_info.train_indices,
                seq_len=config.data.seq_len,
                target_cols=target_cols,
                date_range=(fold_info.train_start, fold_info.train_end),
            )
            valid_ds = GRUTensorDataset(
                cont_features=self.feature_tensor,
                cat_features=self.cat_feature_tensor,
                labels=self.label_tensor,
                dates=self.dates_arr,
                codes=self.codes_arr,
                indices=fold_info.valid_indices,
                seq_len=config.data.seq_len,
                target_cols=target_cols,
                date_range=(fold_info.valid_start, fold_info.valid_end),
            )

            train_loader = create_dataloader(
                train_ds, batch_size=config.train.batch_size, shuffle=True,
            )
            valid_loader = create_dataloader(
                valid_ds, batch_size=config.train.batch_size, shuffle=False,
            )

            # 构建模型
            model = GRUModel(
                target_cols=target_cols,
                num_features=len(self.feature_cols) + int(sum(self.cat_embedding_dims)),
                num_cont_features=len(self.feature_cols),
                num_cat_features=len(self.cat_feature_cols),
                cat_cardinalities=self.cat_cardinalities,
                cat_embedding_dims=self.cat_embedding_dims,
                config=config.train,
                device=device,
                seed=seed,
                hidden_size=config.network.hidden_size,
                num_layers=config.network.num_layers,
                dropout=config.network.dropout,
                use_attention=config.network.use_attention,
            )

            # 训练（计时）
            fold_t0 = time.time()
            model.fit(train_loader, X_valid=valid_loader)
            fold_time = time.time() - fold_t0

            # 收集 fold 训练元信息
            epochs_trained = model.train_info.get("epochs_trained", 0)
            best_epoch = epochs_trained - model.patience_counter
            fold_record = {
                "fold_idx": fold_idx,
                "train_start": fold_info.train_start.strftime("%Y-%m-%d"),
                "train_end": fold_info.train_end.strftime("%Y-%m-%d"),
                "valid_start": fold_info.valid_start.strftime("%Y-%m-%d"),
                "valid_end": fold_info.valid_end.strftime("%Y-%m-%d"),
                "train_samples": len(fold_info.train_indices),
                "valid_samples": len(fold_info.valid_indices),
                "epochs_trained": epochs_trained,
                "best_epoch": best_epoch,
                "best_rank_ic": model.best_rank_ic,
                "train_time_s": fold_time,
                "history": {
                    "train_loss": list(model.history["train_loss"]),
                    "valid_loss": list(model.history["valid_loss"]),
                    "valid_rank_ic": list(model.history["valid_rank_ic"]),
                },
                "feature_selection_summary": feature_selection_summary,
            }

            # 验证集预测
            preds = model.predict(valid_loader)  # (N_valid, num_targets)

            # 组装 OOF DataFrame
            dates = valid_ds.get_all_dates()
            codes = valid_ds.get_all_codes()

            oof_df = pd.DataFrame({
                "trade_date": dates,
                "ts_code": codes,
                "fold": fold_idx,
            })

            # 多目标：真实值 + 预测值
            for i, col in enumerate(target_cols):
                true_vals = valid_ds.labels[valid_ds.valid_indices, i].cpu().numpy()
                oof_df[f"y_true_{col}"] = true_vals
                oof_df[f"y_pred_{col}"] = preds[:, i]

                pnl_source = self.pnl_return_arrays.get(col)
                if pnl_source is not None:
                    oof_df[f"pnl_return_{col}"] = pnl_source[valid_ds.valid_indices]
                else:
                    oof_df[f"pnl_return_{col}"] = true_vals

            # rank 通道作为主信号
            rank_col = target_cols[model.rank_channel_idx]
            oof_df["y_pred"] = oof_df[f"y_pred_{rank_col}"]
            oof_df["y_true"] = oof_df[f"y_true_{rank_col}"]
            if f"pnl_return_{rank_col}" in oof_df.columns:
                oof_df["pnl_return"] = oof_df[f"pnl_return_{rank_col}"]
            else:
                oof_df["pnl_return"] = oof_df["y_true"]

            # 置换特征重要性（按 trade_date 截面 shuffle）
            try:
                perm_df = self._compute_permutation_importance(
                    model=model,
                    valid_ds=valid_ds,
                    baseline_preds=preds,
                    rank_channel_idx=model.rank_channel_idx,
                    max_features=min(30, len(self.feature_cols)),
                    seed=seed + fold_idx,
                )
            except Exception as e:
                logger.warning(f"Fold {fold_idx} 置换重要性计算失败: {e}", exc_info=True)
                perm_df = pd.DataFrame()

            if not perm_df.empty:
                perm_dir = config.train.save_dir / mode / "permutation_importance"
                perm_dir.mkdir(parents=True, exist_ok=True)
                perm_path = perm_dir / f"permutation_importance_fold{fold_idx}.parquet"
                perm_df.to_parquet(perm_path, index=False)

                fold_record["permutation_importance_path"] = str(perm_path)
                fold_record["permutation_importance_top"] = (
                    perm_df.head(10)[["feature", "rank_ic_drop"]].to_dict("records")
                )

            fold_train_info.append(fold_record)

            oof_list.append(oof_df)

            # 持久化模型
            if save_models:
                model_dir = config.train.save_dir / mode
                model_dir.mkdir(parents=True, exist_ok=True)
                model_name = (
                    f"{mode}_fold{fold_idx}_"
                    f"{fold_info.train_start.strftime('%Y%m%d')}_"
                    f"{fold_info.train_end.strftime('%Y%m%d')}_best_model.pth"
                )
                model.save(model_dir / model_name)

            # 清理显存（Dataset 内部无独立张量，仅清理模型权重和 DataLoader）
            del model, train_ds, valid_ds, train_loader, valid_loader
            gc.collect()
            torch.cuda.empty_cache()

        # 3) 合并 OOF
        if oof_list:
            all_oof = pd.concat(oof_list, ignore_index=True)
            all_oof = all_oof.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        else:
            all_oof = pd.DataFrame()

        # 4) 保存 OOF
        if save_oof and not all_oof.empty:
            oof_dir = config.train.save_dir / mode
            oof_dir.mkdir(parents=True, exist_ok=True)
            oof_path = oof_dir / "oof_predictions.parquet"
            all_oof.to_parquet(oof_path, index=False)
            logger.info(f"OOF 已保存: {oof_path}")

        return all_oof, fold_train_info

    def _train_single_full(
        self,
        save_models: bool,
        save_oof: bool,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Single_Full 模式训练

        多种子融合: 用 multi_seeds 中的每个种子分别训练，
        实盘预测时取所有模型的算术平均。
        OOF 中的预测也取多种子平均。

        Returns:
            (oof_df, fold_train_info)
        """
        config = self.config
        target_cols = config.data.target_cols
        seeds = config.train.multi_seeds
        device = str(self.feature_tensor.device)
        fold_train_info: List[Dict[str, Any]] = []
        feature_selection_summary = self._build_feature_selection_summary()

        # 1) 获取唯一 Fold（single_full 只产出1个，使用逻辑日期边界）
        splitter = GRUTimeSeriesSplitter(
            df=self.df,
            target_cols=target_cols,
            config=config.split,
            seq_len=config.data.seq_len,
            data_start_date=config.data.data_start_date,
            data_end_date=config.data.data_end_date,
        )
        fold_info = next(splitter.split(mode="single_full"))

        logger.info(f"Single_Full: 使用 {len(seeds)} 个种子 {seeds}")

        seed_oof_preds = []  # list of (N_valid, num_targets) arrays

        for seed_idx, seed in enumerate(seeds):
            logger.info(f"=== 种子 {seed_idx + 1}/{len(seeds)}: seed={seed} ===")
            set_seed(seed)

            # 构建 Dataset（共享张量，零拷贝）
            train_ds = GRUTensorDataset(
                cont_features=self.feature_tensor,
                cat_features=self.cat_feature_tensor,
                labels=self.label_tensor,
                dates=self.dates_arr,
                codes=self.codes_arr,
                indices=fold_info.train_indices,
                seq_len=config.data.seq_len,
                target_cols=target_cols,
                date_range=(fold_info.train_start, fold_info.train_end),
            )
            valid_ds = GRUTensorDataset(
                cont_features=self.feature_tensor,
                cat_features=self.cat_feature_tensor,
                labels=self.label_tensor,
                dates=self.dates_arr,
                codes=self.codes_arr,
                indices=fold_info.valid_indices,
                seq_len=config.data.seq_len,
                target_cols=target_cols,
                date_range=(fold_info.valid_start, fold_info.valid_end),
            )

            train_loader = create_dataloader(
                train_ds, batch_size=config.train.batch_size, shuffle=True,
            )
            valid_loader = create_dataloader(
                valid_ds, batch_size=config.train.batch_size, shuffle=False,
            )

            # 构建模型
            model = GRUModel(
                target_cols=target_cols,
                num_features=len(self.feature_cols) + int(sum(self.cat_embedding_dims)),
                num_cont_features=len(self.feature_cols),
                num_cat_features=len(self.cat_feature_cols),
                cat_cardinalities=self.cat_cardinalities,
                cat_embedding_dims=self.cat_embedding_dims,
                config=config.train,
                device=device,
                seed=seed,
                hidden_size=config.network.hidden_size,
                num_layers=config.network.num_layers,
                dropout=config.network.dropout,
                use_attention=config.network.use_attention,
            )

            # 训练（计时）
            seed_t0 = time.time()
            model.fit(train_loader, X_valid=valid_loader)
            seed_time = time.time() - seed_t0

            # 收集种子训练元信息
            epochs_trained = model.train_info.get("epochs_trained", 0)
            best_epoch = epochs_trained - model.patience_counter
            seed_record = {
                "seed": seed,
                "train_start": fold_info.train_start.strftime("%Y-%m-%d"),
                "train_end": fold_info.train_end.strftime("%Y-%m-%d"),
                "valid_start": fold_info.valid_start.strftime("%Y-%m-%d"),
                "valid_end": fold_info.valid_end.strftime("%Y-%m-%d"),
                "train_samples": len(fold_info.train_indices),
                "valid_samples": len(fold_info.valid_indices),
                "epochs_trained": epochs_trained,
                "best_epoch": best_epoch,
                "best_rank_ic": model.best_rank_ic,
                "train_time_s": seed_time,
                "history": {
                    "train_loss": list(model.history["train_loss"]),
                    "valid_loss": list(model.history["valid_loss"]),
                    "valid_rank_ic": list(model.history["valid_rank_ic"]),
                },
                "feature_selection_summary": feature_selection_summary,
            }

            # 验证集预测
            preds = model.predict(valid_loader)
            seed_oof_preds.append(preds)

            try:
                perm_df = self._compute_permutation_importance(
                    model=model,
                    valid_ds=valid_ds,
                    baseline_preds=preds,
                    rank_channel_idx=model.rank_channel_idx,
                    max_features=min(30, len(self.feature_cols)),
                    seed=seed,
                )
            except Exception as e:
                logger.warning(f"Single_Full seed={seed} 置换重要性计算失败: {e}", exc_info=True)
                perm_df = pd.DataFrame()

            if not perm_df.empty:
                perm_dir = config.train.save_dir / "single_full" / "permutation_importance"
                perm_dir.mkdir(parents=True, exist_ok=True)
                perm_path = perm_dir / f"permutation_importance_seed_{seed}.parquet"
                perm_df.to_parquet(perm_path, index=False)
                seed_record["permutation_importance_path"] = str(perm_path)
                seed_record["permutation_importance_top"] = (
                    perm_df.head(10)[["feature", "rank_ic_drop"]].to_dict("records")
                )

            fold_train_info.append(seed_record)

            # 持久化
            if save_models:
                model_dir = config.train.save_dir / "single_full"
                model_dir.mkdir(parents=True, exist_ok=True)
                model_name = f"single_full_best_model_seed_{seed}.pth"
                model.save(model_dir / model_name)

            del model, train_ds, valid_ds, train_loader, valid_loader
            gc.collect()
            torch.cuda.empty_cache()

        # 多种子平均
        avg_preds = np.mean(seed_oof_preds, axis=0)  # (N_valid, num_targets)

        # 组装 OOF（共享张量方式获取元数据，不额外分配内存）
        valid_ds_meta = GRUTensorDataset(
            cont_features=self.feature_tensor,
            cat_features=self.cat_feature_tensor,
            labels=self.label_tensor,
            dates=self.dates_arr,
            codes=self.codes_arr,
            indices=fold_info.valid_indices,
            seq_len=config.data.seq_len,
            target_cols=target_cols,
            date_range=(fold_info.valid_start, fold_info.valid_end),
        )

        dates = valid_ds_meta.get_all_dates()
        codes = valid_ds_meta.get_all_codes()

        oof_df = pd.DataFrame({
            "trade_date": dates,
            "ts_code": codes,
            "fold": 0,
        })

        for i, col in enumerate(target_cols):
            true_vals = valid_ds_meta.labels[valid_ds_meta.valid_indices, i].cpu().numpy()
            oof_df[f"y_true_{col}"] = true_vals
            oof_df[f"y_pred_{col}"] = avg_preds[:, i]

            pnl_source = self.pnl_return_arrays.get(col)
            if pnl_source is not None:
                oof_df[f"pnl_return_{col}"] = pnl_source[valid_ds_meta.valid_indices]
            else:
                oof_df[f"pnl_return_{col}"] = true_vals

        # 主信号
        rank_idx = 0
        for i, col in enumerate(target_cols):
            if col.startswith("rank"):
                rank_idx = i
                break
        rank_col = target_cols[rank_idx]
        oof_df["y_pred"] = oof_df[f"y_pred_{rank_col}"]
        oof_df["y_true"] = oof_df[f"y_true_{rank_col}"]
        if f"pnl_return_{rank_col}" in oof_df.columns:
            oof_df["pnl_return"] = oof_df[f"pnl_return_{rank_col}"]
        else:
            oof_df["pnl_return"] = oof_df["y_true"]

        # 保存
        if save_oof and not oof_df.empty:
            oof_dir = config.train.save_dir / "single_full"
            oof_dir.mkdir(parents=True, exist_ok=True)
            oof_path = oof_dir / "oof_predictions.parquet"
            oof_df.to_parquet(oof_path, index=False)
            logger.info(f"OOF 已保存: {oof_path}")

        del valid_ds_meta
        gc.collect()

        return oof_df, fold_train_info


class GRUInferenceEngine:
    """
    实盘推断引擎

    大集成逻辑:
    1. 加载 Rolling 模型群 → 各自预测 → 等权平均 → Score_rolling
    2. 加载 Single_Full 模型群 → 各自预测 → 算术平均 → Score_full
    3. 加权融合: final = rolling_weight * Score_rolling + full_weight * Score_full
    """

    def __init__(self, config: Optional[GRUInferenceConfig] = None):
        self.config = config or GRUInferenceConfig()
        self.rolling_models: List[GRUModel] = []
        self.full_models: List[GRUModel] = []

    def load_models(
        self,
        device: str = "cuda",
        rolling_dir: Optional[Path] = None,
        full_dir: Optional[Path] = None,
    ):
        """加载所有模型"""
        rolling_dir = rolling_dir or self.config.rolling_models_dir
        full_dir = full_dir or self.config.full_models_dir

        # Rolling 模型
        if rolling_dir.exists():
            pth_files = sorted(rolling_dir.glob("*_best_model.pth"))
            self.rolling_models = [
                GRUModel.load(f, device=device) for f in pth_files
            ]
            logger.info(f"加载 {len(self.rolling_models)} 个 Rolling 模型")

        # Full 模型
        if full_dir.exists():
            pth_files = sorted(full_dir.glob("*_best_model_seed_*.pth"))
            self.full_models = [
                GRUModel.load(f, device=device) for f in pth_files
            ]
            logger.info(f"加载 {len(self.full_models)} 个 Single_Full 模型")

    def predict(
        self,
        loader,
    ) -> np.ndarray:
        """
        融合预测

        Returns:
            final_preds: (N, num_targets)
        """
        scores = []
        weights = []

        # Rolling 模型群
        if self.rolling_models:
            rolling_preds = [m.predict(loader) for m in self.rolling_models]
            score_rolling = np.mean(rolling_preds, axis=0)
            scores.append(score_rolling)
            weights.append(self.config.rolling_weight)

        # Full 模型群
        if self.full_models:
            full_preds = [m.predict(loader) for m in self.full_models]
            score_full = np.mean(full_preds, axis=0)
            scores.append(score_full)
            weights.append(self.config.full_weight)

        if not scores:
            raise ValueError("没有可用模型")

        # 权重归一化
        w = np.array(weights)
        w = w / w.sum()

        final = sum(s * wi for s, wi in zip(scores, w))
        return final
