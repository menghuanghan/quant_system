"""
训练调度主脑（Trainer）

核心职责：
- 编排数据与模型的交互
- 多标签循环与动态过滤
- 截面 Z-Score 标准化（连续型标签）
- 去极值保护（Clip）
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
    LabelConfig,
    LGBConfig,
    SplitConfig,
    SplitMode,
    TargetType,
    TrainConfig,
)
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
    
    def _should_normalize(self, target_col: str) -> bool:
        """
        判断是否需要对标签做截面标准化
        
        规则：
        - rank_* 或 label_bin_* 不标准化
        - ret_*, excess_ret_*, sharpe_* 需要标准化
        
        Args:
            target_col: 标签列名
            
        Returns:
            should_normalize: 是否需要标准化
        """
        label_config = self.config.label_config
        
        # 检查是否在跳过列表中
        for prefix in label_config.skip_normalize_prefixes:
            if target_col.startswith(prefix):
                return False
        
        # 检查是否在需要标准化列表中
        for prefix in label_config.zscore_prefixes:
            if target_col.startswith(prefix):
                return True
        
        return False
    
    def _normalize_target(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        date_col: str = "trade_date",
    ) -> pd.Series:
        """
        对标签做截面 Z-Score 标准化 + 去极值
        
        公式：Target_norm = clip((Target - Mean) / Std, -3, 3)
        
        注意：必须基于传入的 df 参数计算，而非 self.df_gpu，
        因为 df 可能已经被过滤（如 _filter_valid_samples）
        
        Args:
            df: 输入 DataFrame（可能是过滤后的子集）
            target_col: 标签列名
            date_col: 日期列名
            
        Returns:
            normalized: 标准化后的标签 Series（与 df 行数一致）
        """
        label_config = self.config.label_config
        
        if self.use_gpu_df and HAS_CUDF:
            # GPU 加速版本 - 【修复】使用传入的 df 而非 self.df_gpu
            df_gpu = cudf.DataFrame.from_pandas(df[[date_col, target_col]])
            
            # 【修复】添加行索引，确保 merge 后能恢复原顺序
            df_gpu["_row_id"] = cudf.Series(range(len(df_gpu)))
            
            # 按日期分组计算截面均值和标准差
            stats = df_gpu.groupby(date_col)[target_col].agg(["mean", "std"]).reset_index()
            stats.columns = [date_col, "mean", "std"]
            
            # 合并回原数据
            merged = df_gpu.merge(stats, on=date_col, how="left")
            
            # 【修复】恢复原始行顺序
            merged = merged.sort_values("_row_id").reset_index(drop=True)
            
            # Z-Score
            normalized = (merged[target_col] - merged["mean"]) / merged["std"].replace(0, 1)
            
            # Clip
            normalized = normalized.clip(label_config.clip_min, label_config.clip_max)
            
            return normalized.to_pandas()
        else:
            # CPU 版本
            target = df[target_col].copy()
            
            # 按日期分组计算截面统计
            date_groups = df.groupby(date_col)[target_col]
            mean_map = date_groups.transform("mean")
            std_map = date_groups.transform("std").replace(0, 1)  # 避免除零
            
            # Z-Score
            normalized = (target - mean_map) / std_map
            
            # Clip 去极值
            normalized = normalized.clip(label_config.clip_min, label_config.clip_max)
            
            return normalized
    
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
        
        # 准备标签
        if self._should_normalize(target_col):
            logger.info(f"Applying cross-sectional Z-Score normalization to {target_col}")
            y_normalized = self._normalize_target(df_filtered, target_col)
            df_work = df_filtered.copy()
            df_work[f"{target_col}_norm"] = y_normalized
            actual_target_col = f"{target_col}_norm"
        else:
            logger.info(f"Skipping normalization for {target_col}")
            df_work = df_filtered
            actual_target_col = target_col
        
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
            oof_df = pd.DataFrame({
                "trade_date": valid_df["trade_date"].values,
                "ts_code": valid_df["ts_code"].values,
                "y_true": y_valid.values,
                "y_pred": pred,
                "fold": fold_idx,
            })
            
            # 如果做了标准化，也存原始标签
            if actual_target_col != target_col:
                oof_df["y_true_raw"] = valid_df[target_col].values
            
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
    
    支持：
    - 加载 Rolling 模型群 + Single Full 模型
    - 加权融合预测
    - 多目标融合
    """
    
    def __init__(
        self,
        rolling_models_dir: Optional[Path] = None,
        full_model_path: Optional[Path] = None,
        rolling_weight: float = 0.4,
        full_weight: float = 0.6,
    ):
        """
        初始化推断引擎
        
        Args:
            rolling_models_dir: Rolling 模型目录
            full_model_path: Single Full 模型路径
            rolling_weight: Rolling 模型群权重
            full_weight: Single Full 模型权重
        """
        self.rolling_models_dir = rolling_models_dir
        self.full_model_path = full_model_path
        self.rolling_weight = rolling_weight
        self.full_weight = full_weight
        
        self.rolling_models: List[LGBQuantModel] = []
        self.full_model: Optional[LGBQuantModel] = None
    
    def load_models(self, target_col: str) -> None:
        """
        加载指定标签的模型
        
        Args:
            target_col: 标签列名
        """
        # 加载 Rolling 模型群
        if self.rolling_models_dir and self.rolling_models_dir.exists():
            pattern = f"lgb_{target_col}_rolling_*.pkl"
            model_files = sorted(self.rolling_models_dir.glob(pattern))
            
            self.rolling_models = [
                LGBQuantModel.load(f) for f in model_files
            ]
            logger.info(f"Loaded {len(self.rolling_models)} rolling models for {target_col}")
        
        # 加载 Single Full 模型
        if self.full_model_path and self.full_model_path.exists():
            self.full_model = LGBQuantModel.load(self.full_model_path)
            logger.info(f"Loaded full model from {self.full_model_path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        融合预测
        
        公式：0.4 * Score_rolling + 0.6 * Score_full
        其中 Score_rolling 是所有 Rolling 模型的平均
        
        Args:
            X: 输入特征
            
        Returns:
            predictions: 融合后的预测分数
        """
        scores = []
        weights = []
        
        # Rolling 模型群预测（等权平均）
        if self.rolling_models:
            rolling_preds = [m.predict(X) for m in self.rolling_models]
            score_rolling = np.mean(rolling_preds, axis=0)
            scores.append(score_rolling)
            weights.append(self.rolling_weight)
        
        # Full 模型预测
        if self.full_model:
            score_full = self.full_model.predict(X)
            scores.append(score_full)
            weights.append(self.full_weight)
        
        if not scores:
            raise ValueError("No models loaded")
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 加权融合
        final_score = sum(s * w for s, w in zip(scores, weights))
        return final_score