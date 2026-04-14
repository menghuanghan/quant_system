"""
训练调度主脑（Trainer）

核心职责：
- 编排数据与模型的交互
- 多标签循环与动态过滤
- 滚窗迭代与 OOF 拼装
- 模型持久化

支持使用 cuDF 进行 GPU 加速数据处理
"""

import gc
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import (
    DEFAULT_TRAIN_CONFIG,
    FeatureConfig,
    InferenceConfig,
    LabelConfig,
    LGBConfig,
    SplitConfig,
    SplitMode,
    TargetType,
    TrainConfig,
)
from ..metrics import infer_pnl_return_col
from .dataset import FoldInfo, TimeSeriesSplitter
from .lgb_model import LGBQuantModel

logger = logging.getLogger(__name__)

# 尝试导入 cuDF
try:
    import cudf
    HAS_CUDF = True
    logger.info("cuDF available for GPU-accelerated data processing")
except ImportError:
    HAS_CUDF = False
    logger.info("cuDF not available, using pandas")


class LGBTrainer:
    """
    LightGBM 训练调度器
    
    负责：
    1. 加载数据（支持 cuDF GPU 加速）
    2. 多标签循环训练
    3. 动态标签过滤与标准化
    4. 时序切分与 OOF 组装
    5. 模型持久化
    
    Example:
        >>> trainer = LGBTrainer()
        >>> trainer.train(target_cols=["rank_ret_5d", "excess_ret_10d"])
        >>> oof_df = trainer.get_oof_predictions()
    """
    
    def __init__(
        self,
        config: Optional[TrainConfig] = None,
        use_gpu_df: bool = True,
    ):
        """
        初始化训练器
        
        Args:
            config: 训练配置
            use_gpu_df: 是否使用 cuDF 加速（需要 RAPIDS 环境）
        """
        self.config = config or DEFAULT_TRAIN_CONFIG
        self.use_gpu_df = use_gpu_df and HAS_CUDF and self.config.use_gpu_dataframe
        
        # 数据
        self.df: Optional[pd.DataFrame] = None
        self.df_gpu = None  # cuDF DataFrame
        
        # 训练结果
        self.oof_results: Dict[str, List[pd.DataFrame]] = {}  # target_col -> list of fold oof
        self.models: Dict[str, List[LGBQuantModel]] = {}      # target_col -> list of fold models
        self.feature_importance: Dict[str, pd.DataFrame] = {} # target_col -> importance df
        self.fold_info_dict: Dict[str, List[Dict[str, Any]]] = {}  # target_col -> fold info list
        self.model_train_info: Dict[str, List[Dict[str, Any]]] = {}  # target_col -> train info list
        
        logger.info(f"LGBTrainer initialized (use_gpu_df={self.use_gpu_df})")
    
    def load_data(self, path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        加载训练数据
        
        Args:
            path: 数据文件路径（默认从 config 读取）
            
        Returns:
            df: 加载的 DataFrame
        """
        path = Path(path) if path else self.config.train_data_path
        
        logger.info(f"Loading data from {path}...")
        
        if self.use_gpu_df:
            # 使用 cuDF 加载
            self.df_gpu = cudf.read_parquet(str(path))
            self.df = self.df_gpu.to_pandas()
            logger.info(f"Data loaded to GPU: shape={self.df.shape}")
        else:
            self.df = pd.read_parquet(path)
            logger.info(f"Data loaded: shape={self.df.shape}")
        
        # 按日期过滤数据（使用 config 中的日期范围）
        self._filter_by_date_range()
        
        # 预处理：修复类别特征负值（CUDA 模式需要）
        self._fix_categorical_negative_values()
        
        return self.df
    
    def _filter_by_date_range(self) -> None:
        """
        按 config 中的日期范围过滤数据
        """
        date_col = "trade_date"
        start_date = self.config.split_config.data_start_date
        end_date = self.config.split_config.data_end_date
        
        original_count = len(self.df)
        self.df = self.df[
            (self.df[date_col] >= start_date)
            & (self.df[date_col] <= end_date)
        ].copy()
        filtered_count = len(self.df)
        
        actual_min = self.df[date_col].min()
        actual_max = self.df[date_col].max()
        
        logger.info(
            f"Date filter: {original_count:,} -> {filtered_count:,} records "
            f"(range: {start_date} ~ {end_date}, actual: {actual_min} ~ {actual_max})"
        )
    
    def _fix_categorical_negative_values(self) -> None:
        """
        修复类别特征中的负值
        
        LightGBM CUDA 模式不支持负值类别特征，将负值映射为 max+offset
        """
        cat_cols = self.config.feature_config.category_columns
        fixed_cols = []
        
        for col in cat_cols:
            if col not in self.df.columns:
                continue
            
            neg_mask = self.df[col] < 0
            neg_count = neg_mask.sum()
            
            if neg_count > 0:
                max_val = self.df[col].max()
                # 将负值映射：-1 -> max+1, -2 -> max+2, etc.
                self.df.loc[neg_mask, col] = self.df.loc[neg_mask, col].abs() + max_val
                fixed_cols.append(f"{col}({neg_count:,})")
        
        if fixed_cols:
            logger.info(f"Fixed negative values in categorical features: {', '.join(fixed_cols)}")
    
    def _detect_target_type(self, target_col: str) -> TargetType:
        """
        根据标签列名检测标签类型
        
        Args:
            target_col: 标签列名
            
        Returns:
            target_type: 标签类型枚举
        """
        col_lower = target_col.lower()
        
        if "rank" in col_lower:
            return TargetType.RANK
        elif "bin" in col_lower:
            return TargetType.CLASSIFICATION
        else:
            # 默认回归（绝对/超额收益、夏普等）
            return TargetType.REGRESSION

    def _filter_valid_samples(
        self, 
        df: pd.DataFrame, 
        target_col: str,
    ) -> pd.DataFrame:
        """
        过滤掉标签为 NaN 的样本
        
        Args:
            df: 输入 DataFrame
            target_col: 标签列名
            
        Returns:
            filtered_df: 过滤后的 DataFrame
        """
        valid_mask = df[target_col].notna()
        filtered_df = df[valid_mask].reset_index(drop=True)
        
        n_dropped = len(df) - len(filtered_df)
        if n_dropped > 0:
            logger.info(f"Filtered {n_dropped} samples with NaN in {target_col}")
        
        return filtered_df
    
    def _get_model_name(
        self, 
        target_col: str, 
        mode: SplitMode, 
        fold_idx: int,
        fold_info: Optional[FoldInfo] = None,
    ) -> str:
        """
        生成模型命名
        
        格式：lgb_{target}_{mode}_fold{idx}_{train_start}_{train_end}
        
        Args:
            target_col: 标签列名
            mode: 切分模式
            fold_idx: Fold 序号
            fold_info: Fold 信息
            
        Returns:
            name: 模型名称
        """
        name = f"lgb_{target_col}_{mode.value}_fold{fold_idx}"
        
        if fold_info:
            train_start = fold_info.train_start.strftime("%Y%m%d")
            train_end = fold_info.train_end.strftime("%Y%m%d")
            name += f"_{train_start}_{train_end}"
        
        return name
    
    def train_single_target(
        self,
        target_col: str,
        mode: Optional[SplitMode] = None,
        save_models: bool = True,
    ) -> Tuple[List[pd.DataFrame], List[LGBQuantModel], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        训练单个标签的所有 Fold
        
        Args:
            target_col: 标签列名
            mode: 切分模式
            save_models: 是否保存模型
            
        Returns:
            (oof_list, models, fold_info_list, train_info_list): OOF 预测、模型、Fold信息、训练信息
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        mode = mode or self.config.split_config.mode
        
        logger.info(f"=" * 60)
        logger.info(f"Training target: {target_col} (mode={mode.value})")
        logger.info(f"=" * 60)
        
        # 检测标签类型
        target_type = self._detect_target_type(target_col)
        logger.info(f"Detected target type: {target_type.value}")
        
        # 过滤有效样本
        df_filtered = self._filter_valid_samples(self.df, target_col)

        # 准备标签（按用户要求：不做 target 标准化/截断）
        df_work = df_filtered
        actual_target_col = target_col

        pnl_return_col = infer_pnl_return_col(target_col, available_cols=df_work.columns)
        if pnl_return_col is None:
            pnl_return_col = target_col
            logger.warning(
                "Cannot infer dedicated pnl return column for %s, fallback to target column",
                target_col,
            )
        else:
            logger.info("PnL return column for %s: %s", target_col, pnl_return_col)
        
        # 初始化时序切分器
        splitter = TimeSeriesSplitter(
            df=df_work,
            target_col=target_col,
            date_col="trade_date",
            config=self.config.split_config,
        )
        
        oof_list = []
        models = []
        feature_importance_list = []
        fold_info_list = []  # 收集 Fold 信息
        train_info_list = []  # 收集训练信息
        
        # 遍历所有 Fold
        for fold_info in splitter.split(mode=mode):
            fold_idx = fold_info.fold_idx
            
            logger.info(f"--- Fold {fold_idx} ---")
            
            # 提取训练/验证数据
            train_df = df_work.iloc[fold_info.train_indices]
            valid_df = df_work.iloc[fold_info.valid_indices]
            
            X_train = train_df
            y_train = train_df[actual_target_col]
            X_valid = valid_df
            y_valid = valid_df[actual_target_col]
            
            # 创建模型
            model_name = self._get_model_name(target_col, mode, fold_idx, fold_info)
            model = LGBQuantModel(
                name=model_name,
                lgb_config=self.config.lgb_config,
                feature_config=self.config.feature_config,
                target_type=target_type,
            )
            
            # 训练
            model.fit(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                target_col=actual_target_col,
            )
            
            # 验证集预测
            pred = model.predict(X_valid)
            
            # 组装 OOF DataFrame
            pnl_return_values = (
                valid_df[pnl_return_col].values
                if pnl_return_col in valid_df.columns
                else valid_df[target_col].values
            )

            oof_df = pd.DataFrame({
                "trade_date": valid_df["trade_date"].values,
                "ts_code": valid_df["ts_code"].values,
                "y_true": y_valid.values,
                "y_pred": pred,
                "pnl_return": pnl_return_values,
                f"pnl_return_{target_col}": pnl_return_values,
                "fold": fold_idx,
            })
            
            oof_list.append(oof_df)
            models.append(model)
            
            # 特征重要性
            importance = model.get_feature_importance()
            importance["fold"] = fold_idx
            feature_importance_list.append(importance)
            
            # 收集 Fold 信息
            fold_info_list.append({
                "fold_idx": fold_idx,
                "train_start": fold_info.train_start.strftime("%Y-%m-%d"),
                "train_end": fold_info.train_end.strftime("%Y-%m-%d"),
                "valid_start": fold_info.valid_start.strftime("%Y-%m-%d"),
                "valid_end": fold_info.valid_end.strftime("%Y-%m-%d"),
                "train_samples": len(fold_info.train_indices),
                "valid_samples": len(fold_info.valid_indices),
                "gap_days": fold_info.gap_days,
            })
            train_info_list.append(model.train_info.copy())
            
            # 保存模型
            if save_models:
                model_dir = self.config.model_save_dir / mode.value
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f"{model_name}.pkl"
                model.save(model_path)
            
            # 清理内存
            del train_df, valid_df, X_train, y_train, X_valid, y_valid
            gc.collect()
        
        # 汇总特征重要性（取所有 Fold 的平均）
        if feature_importance_list:
            all_importance = pd.concat(feature_importance_list, ignore_index=True)
            avg_importance = all_importance.groupby("feature")["importance"].mean() \
                .sort_values(ascending=False).reset_index()
            avg_importance["importance_pct"] = avg_importance["importance"] / avg_importance["importance"].sum() * 100
            self.feature_importance[target_col] = avg_importance
        
        return oof_list, models, fold_info_list, train_info_list
    
    def train(
        self,
        target_cols: Optional[List[str]] = None,
        mode: Optional[SplitMode] = None,
        save_models: bool = True,
        save_oof: bool = True,
        generate_report: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        训练多个标签
        
        Args:
            target_cols: 标签列表（默认从 config 读取）
            mode: 切分模式
            save_models: 是否保存模型
            save_oof: 是否保存 OOF 预测
            generate_report: 是否生成训练报告
            
        Returns:
            oof_dict: {target_col: oof_df} 字典
        """
        if self.df is None:
            self.load_data()
        
        target_cols = target_cols or self.config.feature_config.default_target_cols
        mode = mode or self.config.split_config.mode
        
        logger.info(f"Starting training for {len(target_cols)} targets: {target_cols}")
        logger.info(f"Split mode: {mode.value}")
        
        oof_dict = {}
        
        for target_col in target_cols:
            # 检查标签列是否存在
            if target_col not in self.df.columns:
                logger.warning(f"Target column '{target_col}' not found in data, skipping...")
                continue
            
            # 训练
            oof_list, models, fold_info_list, train_info_list = self.train_single_target(
                target_col=target_col,
                mode=mode,
                save_models=save_models,
            )
            
            # 存储结果
            self.oof_results[target_col] = oof_list
            self.models[target_col] = models
            self.fold_info_dict[target_col] = fold_info_list
            self.model_train_info[target_col] = train_info_list
            
            # 合并 OOF
            if oof_list:
                oof_df = pd.concat(oof_list, ignore_index=True)
                oof_df = oof_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
                oof_df["target"] = target_col
                oof_dict[target_col] = oof_df
        
        # 保存 OOF（按模式保存到对应子目录，防止不同模式互相覆盖）
        if save_oof and oof_dict:
            all_oof = pd.concat(oof_dict.values(), ignore_index=True)
            # 【修复】保存到模式对应的子目录: models/lgb/{mode}/oof_predictions.parquet
            mode_dir = self.config.model_save_dir / mode.value
            mode_dir.mkdir(parents=True, exist_ok=True)
            oof_path = mode_dir / "oof_predictions.parquet"
            all_oof.to_parquet(oof_path, index=False)
            logger.info(f"OOF predictions saved to {oof_path}")
        
        # 保存特征重要性（按模式保存到对应子目录）
        if self.feature_importance:
            # 【修复】保存到模式对应的子目录: models/lgb/{mode}/feature_importance.parquet
            mode_dir = self.config.model_save_dir / mode.value
            mode_dir.mkdir(parents=True, exist_ok=True)
            importance_path = mode_dir / "feature_importance.parquet"
            all_importance = pd.concat([
                df.assign(target=target) 
                for target, df in self.feature_importance.items()
            ], ignore_index=True)
            all_importance.to_parquet(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
        
        # 生成训练报告
        if generate_report and oof_dict:
            try:
                from .report_generator import TrainingReportGenerator
                
                report_generator = TrainingReportGenerator(
                    config=self.config,
                    oof_dict=oof_dict,
                    feature_importance=self.feature_importance,
                    fold_info_dict=self.fold_info_dict,
                    model_train_info=self.model_train_info,
                )
                report_path = report_generator.generate_report()
                logger.info(f"Training report generated: {report_path}")
            except Exception as e:
                logger.warning(f"Failed to generate training report: {e}")
        
        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"Trained targets: {list(oof_dict.keys())}")
        logger.info("=" * 60)
        
        return oof_dict
    
    def get_oof_predictions(self) -> pd.DataFrame:
        """
        获取所有标签的 OOF 预测
        
        Returns:
            oof_df: 合并后的 OOF DataFrame
        """
        if not self.oof_results:
            raise ValueError("No OOF results. Run train() first.")
        
        all_oof = []
        for target_col, oof_list in self.oof_results.items():
            oof_df = pd.concat(oof_list, ignore_index=True)
            oof_df["target"] = target_col
            all_oof.append(oof_df)
        
        return pd.concat(all_oof, ignore_index=True)
    
    def get_feature_importance(self, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            target_col: 标签列名（None 返回所有标签的平均）
            
        Returns:
            importance_df: 特征重要性 DataFrame
            
        Raises:
            ValueError: 如果指定的 target_col 不存在
        """
        if not self.feature_importance:
            raise ValueError("No feature importance. Run train() first.")
        
        if target_col:
            # 【修复】添加存在性检查，避免返回 None
            if target_col not in self.feature_importance:
                available = list(self.feature_importance.keys())
                raise ValueError(
                    f"Feature importance for '{target_col}' not found. "
                    f"Available targets: {available}"
                )
            return self.feature_importance[target_col]
        
        # 所有标签的平均
        all_df = pd.concat(self.feature_importance.values(), ignore_index=True)
        avg_df = all_df.groupby("feature")["importance"].mean() \
            .sort_values(ascending=False).reset_index()
        avg_df["importance_pct"] = avg_df["importance"] / avg_df["importance"].sum() * 100
        return avg_df


class InferenceEngine:
    """
    实盘推断引擎

    能力：
    - 多目标模型加载（Rolling + Single_Full 两层）
    - Rolling 层支持时间递近加权（linear_recency）
    - 层内与层间可解释权重拆解
    - 兼容旧接口：load_models(target_col) + predict(X)
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        rolling_models_dir: Optional[Path] = None,
        full_models_dir: Optional[Path] = None,
        full_model_pattern: Optional[str] = None,
        full_model_path: Optional[Path] = None,
        rolling_weight: Optional[float] = None,
        full_weight: Optional[float] = None,
        rolling_weight_strategy: Optional[str] = None,
        target_cols: Optional[List[str]] = None,
    ):
        """
        初始化推断引擎

        Args:
            config: 推断配置对象
            rolling_models_dir: Rolling 模型目录（可覆盖 config）
            full_models_dir: Single_Full 模型目录（可覆盖 config）
            full_model_pattern: full 模型匹配模式（可覆盖 config）
            full_model_path: 兼容旧单模型路径（fallback）
            rolling_weight: Rolling 层权重
            full_weight: Single_Full 层权重
            rolling_weight_strategy: Rolling 层内权重策略（uniform/linear_recency）
            target_cols: 默认推断目标列表
        """
        self.config = config or InferenceConfig()

        self.rolling_models_dir = Path(rolling_models_dir or self.config.rolling_models_dir)
        self.full_models_dir = Path(full_models_dir or self.config.full_models_dir)
        self.full_model_pattern = full_model_pattern or self.config.full_model_pattern
        self.full_model_path = Path(full_model_path or self.config.full_model_path)

        self.rolling_weight = float(
            self.config.rolling_weight if rolling_weight is None else rolling_weight
        )
        self.full_weight = float(
            self.config.full_weight if full_weight is None else full_weight
        )
        self.rolling_weight_strategy = str(
            (self.config.rolling_weight_strategy if rolling_weight_strategy is None else rolling_weight_strategy)
        ).strip().lower()
        self.target_cols = list(target_cols or self.config.target_cols)

        # 目标级模型缓存
        self.rolling_models_by_target: Dict[str, List[LGBQuantModel]] = {}
        self.full_models_by_target: Dict[str, List[LGBQuantModel]] = {}
        self.rolling_model_files_by_target: Dict[str, List[Path]] = {}
        self.full_model_files_by_target: Dict[str, List[Path]] = {}

        # 兼容旧接口（单目标）
        self.rolling_models: List[LGBQuantModel] = []
        self.full_model: Optional[LGBQuantModel] = None

    @staticmethod
    def _normalize_weights(weights: np.ndarray) -> np.ndarray:
        """归一化权重，若和为 0 则回退等权。"""
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size == 0:
            raise ValueError("weights must be a non-empty 1D array")

        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0.0:
            w = np.ones_like(w, dtype=np.float64)
            s = float(np.sum(w))
        return w / s

    @staticmethod
    def _extract_fold_idx(path: Path) -> int:
        """从 rolling 模型文件名提取 fold 序号，提取失败返回 -1。"""
        match = re.search(r"_fold(\d+)", path.stem)
        return int(match.group(1)) if match else -1

    def _sort_rolling_model_files(self, model_files: List[Path]) -> List[Path]:
        """按 fold_idx 升序排序，确保时间递近加权语义一致。"""
        return sorted(
            model_files,
            key=lambda p: (self._extract_fold_idx(p), p.name),
        )

    def _build_rolling_model_weights(self, model_files: List[Path]) -> np.ndarray:
        """构建 rolling 层内权重。"""
        n_models = len(model_files)
        if n_models == 0:
            return np.array([], dtype=np.float64)

        strategy = self.rolling_weight_strategy
        if strategy == "uniform":
            raw = np.ones(n_models, dtype=np.float64)
        elif strategy == "linear_recency":
            # model_files 已按 fold 递增排序：后续模型权重更大
            raw = np.arange(1, n_models + 1, dtype=np.float64)
        else:
            raise ValueError(
                f"Unsupported rolling_weight_strategy='{strategy}'. "
                "Use one of ['uniform', 'linear_recency']."
            )

        return self._normalize_weights(raw)

    def _predict_model_group(
        self,
        X: pd.DataFrame,
        models: List[LGBQuantModel],
        model_weights: np.ndarray,
    ) -> np.ndarray:
        """同一层内模型按给定权重融合。"""
        if not models:
            raise ValueError("models is empty")

        if len(models) != len(model_weights):
            raise ValueError(
                f"Model count {len(models)} != weight count {len(model_weights)}"
            )

        model_weights = self._normalize_weights(model_weights)
        preds = [np.asarray(m.predict(X), dtype=np.float64) for m in models]

        fused = np.zeros_like(preds[0], dtype=np.float64)
        for pred, w in zip(preds, model_weights):
            fused += pred * w
        return fused

    def load_models_for_targets(
        self,
        target_cols: Optional[List[str]] = None,
        rolling_dir: Optional[Path] = None,
        full_dir: Optional[Path] = None,
    ) -> None:
        """
        批量加载多个目标的模型群。

        Args:
            target_cols: 目标列表
            rolling_dir: rolling 模型目录
            full_dir: single_full 模型目录
        """
        targets = list(target_cols or self.target_cols)
        if not targets:
            raise ValueError("target_cols is empty")

        rolling_base = Path(rolling_dir or self.rolling_models_dir)
        full_base = Path(full_dir or self.full_models_dir)

        self.target_cols = targets
        self.rolling_models_by_target.clear()
        self.full_models_by_target.clear()
        self.rolling_model_files_by_target.clear()
        self.full_model_files_by_target.clear()

        for target_col in targets:
            # Rolling 模型群
            rolling_files: List[Path] = []
            if rolling_base.exists():
                rolling_files = list(rolling_base.glob(f"lgb_{target_col}_rolling_*.pkl"))
                rolling_files = self._sort_rolling_model_files(rolling_files)

            rolling_models = [LGBQuantModel.load(f) for f in rolling_files]
            self.rolling_model_files_by_target[target_col] = rolling_files
            self.rolling_models_by_target[target_col] = rolling_models

            # Single_Full 模型群（按 target 匹配）
            full_files: List[Path] = []
            if full_base.exists():
                pattern = self.full_model_pattern.format(target_col=target_col)
                full_files = sorted(full_base.glob(pattern))

            # 兼容旧单模型路径（仅当前 target 未匹配到时回退）
            if not full_files and self.full_model_path.exists():
                logger.warning(
                    "No target-specific single_full model matched for %s in %s; "
                    "fallback to legacy full_model_path=%s",
                    target_col,
                    full_base,
                    self.full_model_path,
                )
                full_files = [self.full_model_path]

            full_models = [LGBQuantModel.load(f) for f in full_files]
            self.full_model_files_by_target[target_col] = full_files
            self.full_models_by_target[target_col] = full_models

            logger.info(
                "Loaded models for %s: rolling=%d, single_full=%d",
                target_col,
                len(rolling_models),
                len(full_models),
            )

    def load_models(self, target_col: str) -> None:
        """
        加载指定标签的模型（兼容旧接口）。

        Args:
            target_col: 标签列名
        """
        self.load_models_for_targets([target_col])
        self.rolling_models = self.rolling_models_by_target.get(target_col, [])
        full_models = self.full_models_by_target.get(target_col, [])
        self.full_model = full_models[0] if full_models else None

    def _fuse_target_predictions(
        self,
        X: pd.DataFrame,
        target_col: str,
        with_breakdown: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """单目标两层融合，按需返回可解释拆解。"""
        rolling_models = self.rolling_models_by_target.get(target_col, [])
        full_models = self.full_models_by_target.get(target_col, [])

        layer_names: List[str] = []
        layer_scores: List[np.ndarray] = []
        layer_weights: List[float] = []
        breakdown: Dict[str, Any] = {
            "target_col": target_col,
            "rolling": {},
            "single_full": {},
            "layer_weights": {},
        }

        # Rolling 层
        if rolling_models:
            rolling_files = self.rolling_model_files_by_target.get(target_col, [])
            if rolling_files:
                rolling_model_weights = self._build_rolling_model_weights(rolling_files)
            else:
                rolling_model_weights = self._normalize_weights(
                    np.ones(len(rolling_models), dtype=np.float64)
                )

            score_rolling = self._predict_model_group(
                X=X,
                models=rolling_models,
                model_weights=rolling_model_weights,
            )
            layer_names.append("rolling")
            layer_scores.append(score_rolling)
            layer_weights.append(self.rolling_weight)

            if with_breakdown:
                breakdown["rolling"] = {
                    "enabled": True,
                    "model_count": len(rolling_models),
                    "model_files": [str(p) for p in rolling_files],
                    "model_weights": rolling_model_weights.tolist(),
                    "pred_mean": float(np.mean(score_rolling)),
                    "pred_std": float(np.std(score_rolling)),
                }
        elif with_breakdown:
            breakdown["rolling"] = {
                "enabled": False,
                "model_count": 0,
                "model_files": [],
                "model_weights": [],
            }

        # Single_Full 层
        if full_models:
            full_model_weights = self._normalize_weights(
                np.ones(len(full_models), dtype=np.float64)
            )
            score_full = self._predict_model_group(
                X=X,
                models=full_models,
                model_weights=full_model_weights,
            )
            layer_names.append("single_full")
            layer_scores.append(score_full)
            layer_weights.append(self.full_weight)

            if with_breakdown:
                full_files = self.full_model_files_by_target.get(target_col, [])
                breakdown["single_full"] = {
                    "enabled": True,
                    "model_count": len(full_models),
                    "model_files": [str(p) for p in full_files],
                    "model_weights": full_model_weights.tolist(),
                    "pred_mean": float(np.mean(score_full)),
                    "pred_std": float(np.std(score_full)),
                }
        elif with_breakdown:
            breakdown["single_full"] = {
                "enabled": False,
                "model_count": 0,
                "model_files": [],
                "model_weights": [],
            }

        if not layer_scores:
            raise ValueError(f"No models loaded for target '{target_col}'")

        layer_weights_arr = self._normalize_weights(np.asarray(layer_weights, dtype=np.float64))
        final_score = np.zeros_like(layer_scores[0], dtype=np.float64)
        for s, w in zip(layer_scores, layer_weights_arr):
            final_score += s * w

        if with_breakdown:
            breakdown["layer_weights"] = {
                name: float(w) for name, w in zip(layer_names, layer_weights_arr)
            }
            breakdown["configured_layer_weights"] = {
                "rolling": float(self.rolling_weight),
                "single_full": float(self.full_weight),
            }
            breakdown["final"] = {
                "pred_mean": float(np.mean(final_score)),
                "pred_std": float(np.std(final_score)),
            }
            return final_score, breakdown

        return final_score

    def predict_target(
        self,
        X: pd.DataFrame,
        target_col: str,
        with_breakdown: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """单目标融合预测。"""
        loaded = bool(
            self.rolling_models_by_target.get(target_col)
            or self.full_models_by_target.get(target_col)
        )
        if not loaded:
            self.load_models(target_col)

        return self._fuse_target_predictions(
            X=X,
            target_col=target_col,
            with_breakdown=with_breakdown,
        )

    def predict_multi(
        self,
        X: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
        with_breakdown: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """多目标融合预测。"""
        targets = list(target_cols or self.target_cols)
        if not targets:
            raise ValueError("target_cols is empty")

        # 按请求目标批量加载
        self.load_models_for_targets(targets)

        preds_df = pd.DataFrame(index=X.index)
        breakdown_by_target: Dict[str, Any] = {}

        for target_col in targets:
            if with_breakdown:
                pred, breakdown = self.predict_target(
                    X=X,
                    target_col=target_col,
                    with_breakdown=True,
                )
                breakdown_by_target[target_col] = breakdown
            else:
                pred = self.predict_target(
                    X=X,
                    target_col=target_col,
                    with_breakdown=False,
                )

            preds_df[f"y_pred_{target_col}"] = pred

        if with_breakdown:
            return preds_df, {
                "targets": targets,
                "rolling_weight_strategy": self.rolling_weight_strategy,
                "layer_weight_config": {
                    "rolling": self.rolling_weight,
                    "single_full": self.full_weight,
                },
                "by_target": breakdown_by_target,
            }

        return preds_df

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        融合预测（兼容旧单目标接口）。

        说明：
        - 若仅加载了一个目标，返回该目标预测
        - 若已加载多个目标，请使用 predict_multi()
        """
        loaded_targets = [
            t for t in set(self.rolling_models_by_target.keys()) | set(self.full_models_by_target.keys())
            if self.rolling_models_by_target.get(t) or self.full_models_by_target.get(t)
        ]

        # 兼容旧路径：外部仅调用了 load_models(target_col)
        if not loaded_targets and (self.rolling_models or self.full_model is not None):
            layer_scores: List[np.ndarray] = []
            layer_weights: List[float] = []

            if self.rolling_models:
                rw = self._normalize_weights(np.ones(len(self.rolling_models), dtype=np.float64))
                layer_scores.append(self._predict_model_group(X, self.rolling_models, rw))
                layer_weights.append(self.rolling_weight)

            if self.full_model is not None:
                layer_scores.append(np.asarray(self.full_model.predict(X), dtype=np.float64))
                layer_weights.append(self.full_weight)

            if not layer_scores:
                raise ValueError("No models loaded")

            lw = self._normalize_weights(np.asarray(layer_weights, dtype=np.float64))
            out = np.zeros_like(layer_scores[0], dtype=np.float64)
            for s, w in zip(layer_scores, lw):
                out += s * w
            return out

        if not loaded_targets:
            if len(self.target_cols) == 1:
                return self.predict_target(X=X, target_col=self.target_cols[0], with_breakdown=False)
            raise ValueError(
                "No models loaded. Call load_models(target_col) or "
                "load_models_for_targets([...]) first."
            )

        if len(loaded_targets) > 1:
            raise ValueError(
                "Multiple targets are loaded. Use predict_multi() for multi-target inference."
            )

        return self.predict_target(X=X, target_col=loaded_targets[0], with_breakdown=False)