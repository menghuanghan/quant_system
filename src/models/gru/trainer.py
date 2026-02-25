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

from ..config import (
    GRUConfig,
    GRUDataConfig,
    GRUNetworkConfig,
    GRUSplitConfig,
    GRUTrainConfig,
    GRUInferenceConfig,
    get_gru_feature_columns,
    get_gru_selected_features,
)
from .dataset import (
    GRUFoldInfo,
    GRUTimeSeriesSplitter,
    GRUTensorDataset,
    create_dataloader,
)
from .gru_model import GRUModel, set_seed

logger = logging.getLogger(__name__)


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

        # 预计算的共享张量（load_data 中创建）
        self.feature_tensor: Optional[torch.Tensor] = None
        self.label_tensor: Optional[torch.Tensor] = None
        self.dates_arr: Optional[np.ndarray] = None
        self.codes_arr: Optional[np.ndarray] = None

        # 供外部读取的元信息
        self.loaded_columns: List[str] = []

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

        # ---- Step 2: 确定特征列（LGB Top 50 + 宏观特征） ----
        mode = self.config.split.mode
        self.feature_cols = get_gru_selected_features(
            all_columns, mode=mode, top_n=50,
        )

        # ---- Step 3: 确定读取列 ----
        valid_target_cols = [c for c in target_cols if c in all_columns]
        if not valid_target_cols:
            raise ValueError(f"目标列在数据中均不存在: {target_cols}")
        if len(valid_target_cols) < len(target_cols):
            missing = set(target_cols) - set(valid_target_cols)
            logger.warning(f"目标列未找到: {missing}")
        self.config.data.target_cols = valid_target_cols
        target_cols = valid_target_cols

        read_cols = list(dict.fromkeys(
            ['ts_code', 'trade_date'] + self.feature_cols + target_cols
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

        # 过滤非数值特征列
        numeric_cols = df.select_dtypes(
            include=['float32', 'float64', 'Float32', 'Float64',
                     'int8', 'int16', 'int32', 'int64', 'Int32', 'Int64']
        ).columns.tolist()
        self.feature_cols = [c for c in self.feature_cols if c in numeric_cols]

        logger.info(f"特征列数: {len(self.feature_cols)}, 目标列: {target_cols}")

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
        np.nan_to_num(labels_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

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

        # ---- Step 9: 更新配置 ----
        self.config.network.num_features = len(self.feature_cols)
        self.config.network.num_targets = len(target_cols)

        feat_mb = self.feature_tensor.numel() * 4 / 1024 / 1024
        label_mb = self.label_tensor.numel() * 4 / 1024 / 1024
        logger.info(
            f"数据加载完成: rows={len(self.dates_arr):,}, "
            f"features={len(self.feature_cols)}, targets={len(target_cols)}, "
            f"tensor_mem={feat_mb + label_mb:.0f}MB "
            f"(features={feat_mb:.0f}MB, labels={label_mb:.0f}MB), "
            f"device={device}, 总耗时={time.time() - t0:.1f}s"
        )

        return self.df

    def train(
        self,
        mode: Optional[str] = None,
        save_models: bool = True,
        save_oof: bool = True,
    ) -> pd.DataFrame:
        """
        训练入口

        GRU 是多任务并行 → 不需要 for target in target_cols 循环,
        直接开始滚窗迭代与 OOF 拼装.

        Args:
            mode: 训练模式 (rolling / expanding / single_full)
            save_models: 是否持久化模型
            save_oof: 是否保存 OOF

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
            return self._train_single_full(save_models, save_oof)
        else:
            return self._train_fold_mode(mode, save_models, save_oof)

    def _train_fold_mode(
        self,
        mode: str,
        save_models: bool,
        save_oof: bool,
    ) -> pd.DataFrame:
        """Rolling / Expanding 模式训练"""
        config = self.config
        target_cols = config.data.target_cols
        seed = config.train.seed
        device = str(self.feature_tensor.device)
        set_seed(seed)

        # 1) 初始化切分器
        splitter = GRUTimeSeriesSplitter(
            df=self.df,
            target_cols=target_cols,
            config=config.split,
            seq_len=config.data.seq_len,
        )

        oof_list = []

        # 2) 遍历 Fold
        for fold_info in splitter.split(mode=mode):
            fold_idx = fold_info.fold_idx
            logger.info(f"--- Fold {fold_idx} ---")

            # 构建训练/验证 Dataset（共享张量，零拷贝）
            train_ds = GRUTensorDataset(
                features=self.feature_tensor,
                labels=self.label_tensor,
                dates=self.dates_arr,
                codes=self.codes_arr,
                indices=fold_info.train_indices,
                seq_len=config.data.seq_len,
                target_cols=target_cols,
                date_range=(fold_info.train_start, fold_info.train_end),
            )
            valid_ds = GRUTensorDataset(
                features=self.feature_tensor,
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
                num_features=len(self.feature_cols),
                config=config.train,
                device=device,
                seed=seed,
                hidden_size=config.network.hidden_size,
                num_layers=config.network.num_layers,
                dropout=config.network.dropout,
                use_attention=config.network.use_attention,
            )

            # 训练
            model.fit(train_loader, X_valid=valid_loader)

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

            # rank 通道作为主信号
            rank_col = target_cols[model.rank_channel_idx]
            oof_df["y_pred"] = oof_df[f"y_pred_{rank_col}"]
            oof_df["y_true"] = oof_df[f"y_true_{rank_col}"]

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

        return all_oof

    def _train_single_full(
        self,
        save_models: bool,
        save_oof: bool,
    ) -> pd.DataFrame:
        """
        Single_Full 模式训练

        多种子融合: 用 multi_seeds 中的每个种子分别训练，
        实盘预测时取所有模型的算术平均。
        OOF 中的预测也取多种子平均。
        """
        config = self.config
        target_cols = config.data.target_cols
        seeds = config.train.multi_seeds
        device = str(self.feature_tensor.device)

        # 1) 获取唯一 Fold（single_full 只产出1个）
        splitter = GRUTimeSeriesSplitter(
            df=self.df,
            target_cols=target_cols,
            config=config.split,
            seq_len=config.data.seq_len,
        )
        fold_info = next(splitter.split(mode="single_full"))

        logger.info(f"Single_Full: 使用 {len(seeds)} 个种子 {seeds}")

        seed_oof_preds = []  # list of (N_valid, num_targets) arrays

        for seed_idx, seed in enumerate(seeds):
            logger.info(f"=== 种子 {seed_idx + 1}/{len(seeds)}: seed={seed} ===")
            set_seed(seed)

            # 构建 Dataset（共享张量，零拷贝）
            train_ds = GRUTensorDataset(
                features=self.feature_tensor,
                labels=self.label_tensor,
                dates=self.dates_arr,
                codes=self.codes_arr,
                indices=fold_info.train_indices,
                seq_len=config.data.seq_len,
                target_cols=target_cols,
                date_range=(fold_info.train_start, fold_info.train_end),
            )
            valid_ds = GRUTensorDataset(
                features=self.feature_tensor,
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
                num_features=len(self.feature_cols),
                config=config.train,
                device=device,
                seed=seed,
                hidden_size=config.network.hidden_size,
                num_layers=config.network.num_layers,
                dropout=config.network.dropout,
                use_attention=config.network.use_attention,
            )

            # 训练
            model.fit(train_loader, X_valid=valid_loader)

            # 验证集预测
            preds = model.predict(valid_loader)
            seed_oof_preds.append(preds)

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
            features=self.feature_tensor,
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

        # 主信号
        rank_idx = 0
        for i, col in enumerate(target_cols):
            if col.startswith("rank"):
                rank_idx = i
                break
        rank_col = target_cols[rank_idx]
        oof_df["y_pred"] = oof_df[f"y_pred_{rank_col}"]
        oof_df["y_true"] = oof_df[f"y_true_{rank_col}"]

        # 保存
        if save_oof and not oof_df.empty:
            oof_dir = config.train.save_dir / "single_full"
            oof_dir.mkdir(parents=True, exist_ok=True)
            oof_path = oof_dir / "oof_predictions.parquet"
            oof_df.to_parquet(oof_path, index=False)
            logger.info(f"OOF 已保存: {oof_path}")

        del valid_ds_meta
        gc.collect()

        return oof_df


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
