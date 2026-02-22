"""
LightGBM 模型生命周期封装（LGBQuantModel）

核心职责：
- 封装 LightGBM 的 fit/predict/save/load 接口
- 支持 Early Stopping
- 支持类别特征直接传入底层
- 导出 gain 重要性排名
- GPU 加速支持
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..base_model import BaseModel
from ..config import (
    DEFAULT_FEATURE_CONFIG, 
    DEFAULT_LGB_CONFIG,
    FeatureConfig,
    LGBConfig, 
    TargetType,
)

logger = logging.getLogger(__name__)


class LGBQuantModel(BaseModel):
    """
    LightGBM 量化模型封装
    
    特点：
    - 自动识别并排除非特征列（主键、标签列等）
    - 类别特征直接传给 C++ 底层做直方图类别分裂
    - 支持 Early Stopping
    - 特征重要性强制使用 gain（信息增益）
    - GPU 加速
    
    Example:
        >>> model = LGBQuantModel(name="lgb_rank_ret_5d_fold0")
        >>> model.fit(X_train, y_train, X_valid, y_valid)
        >>> predictions = model.predict(X_test)
        >>> importance = model.get_feature_importance()
        >>> model.save("models/lgb/lgb_rank_ret_5d_rolling_fold0.pkl")
    """
    
    def __init__(
        self,
        name: str = "LGBQuantModel",
        lgb_config: Optional[LGBConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
        target_type: TargetType = TargetType.REGRESSION,
    ):
        """
        初始化 LightGBM 模型
        
        Args:
            name: 模型名称标识
            lgb_config: LightGBM 参数配置
            feature_config: 特征配置
            target_type: 标签类型（影响目标函数选择）
        """
        self.lgb_config = lgb_config or DEFAULT_LGB_CONFIG
        self.feature_config = feature_config or DEFAULT_FEATURE_CONFIG
        self.target_type = target_type
        
        super().__init__(name=name, seed=self.lgb_config.seed)
        
        # LightGBM 原生参数
        self.params = self.lgb_config.to_lgb_params(target_type)
        
        # 训练信息
        self.best_iteration: int = 0
        self.best_score: float = float("inf")
        self.evals_result: Dict[str, Any] = {}
        
    def _prepare_features(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        准备特征矩阵，剔除非特征列
        
        Args:
            X: 输入数据
            target_col: 当前训练的目标列（需要排除）
            
        Returns:
            (X_clean, feature_names): 清洗后的特征矩阵和特征名列表
        """
        if isinstance(X, np.ndarray):
            # ndarray 直接返回
            return pd.DataFrame(X), list(range(X.shape[1]))
        
        # 需要排除的列
        exclude_cols = set(self.feature_config.key_columns)  # 主键列
        
        # 排除所有标签相关列
        for col in X.columns:
            for prefix in self.feature_config.label_prefixes:
                if col.startswith(prefix):
                    exclude_cols.add(col)
        
        # 【关键】排除纯宏观特征（无截面方差，会降维打击）
        for col in X.columns:
            for prefix in self.feature_config.drop_macro_prefixes:
                if col.startswith(prefix):
                    exclude_cols.add(col)
        
        # 如果指定了当前目标列，也需要排除
        if target_col:
            exclude_cols.add(target_col)
        
        # 筛选特征列
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        
        return X[feature_cols], feature_cols
    
    def _get_categorical_indices(self, feature_names: List[str]) -> List[int]:
        """
        获取类别特征在特征列表中的索引
        
        Args:
            feature_names: 特征名列表
            
        Returns:
            indices: 类别特征的索引列表
        """
        cat_cols = set(self.feature_config.category_columns)
        indices = [i for i, name in enumerate(feature_names) if name in cat_cols]
        return indices
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_valid: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_valid: Optional[Union[pd.Series, np.ndarray]] = None,
        target_col: Optional[str] = None,
        num_boost_round: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose_eval: Optional[int] = None,
        **kwargs,
    ) -> "LGBQuantModel":
        """
        训练 LightGBM 模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_valid: 验证特征（用于 Early Stopping）
            y_valid: 验证标签
            target_col: 当前训练的目标列名（用于排除非特征列）
            num_boost_round: 最大迭代轮数
            early_stopping_rounds: 早停轮数
            verbose_eval: 日志打印频率
            **kwargs: 传递给 lgb.train 的其他参数
            
        Returns:
            self: 训练后的模型实例
        """
        num_boost_round = num_boost_round or self.lgb_config.num_boost_round
        early_stopping_rounds = early_stopping_rounds or self.lgb_config.early_stopping_rounds
        verbose_eval = verbose_eval or self.lgb_config.verbose_eval
        
        # 准备特征
        X_train_clean, feature_names = self._prepare_features(X_train, target_col)
        self.feature_names = feature_names
        
        # 获取类别特征索引
        cat_indices = self._get_categorical_indices(feature_names)
        categorical_feature = cat_indices if cat_indices else "auto"
        
        logger.info(
            f"Training {self.name}: "
            f"features={len(feature_names)}, categorical={len(cat_indices)}, "
            f"train_samples={len(y_train)}"
        )
        
        # 构建 LightGBM Dataset
        dtrain = lgb.Dataset(
            X_train_clean,
            label=y_train,
            feature_name=feature_names,
            categorical_feature=categorical_feature,
            free_raw_data=False,
        )
        
        # 验证集
        valid_sets = [dtrain]
        valid_names = ["train"]
        
        if X_valid is not None and y_valid is not None and len(y_valid) > 0:
            X_valid_clean, _ = self._prepare_features(X_valid, target_col)
            dvalid = lgb.Dataset(
                X_valid_clean,
                label=y_valid,
                reference=dtrain,
                feature_name=feature_names,
                categorical_feature=categorical_feature,
                free_raw_data=False,
            )
            valid_sets.append(dvalid)
            valid_names.append("valid")
            logger.info(f"Validation set: {len(y_valid)} samples")
        else:
            # 【修复】无验证集时使用 warning 级别，提醒可能过拟合
            logger.warning(
                "No validation set provided - early stopping will be disabled! "
                "Model may overfit. This is expected for single_full mode."
            )
        
        # 训练回调
        callbacks = [
            lgb.log_evaluation(period=verbose_eval),
        ]
        
        if early_stopping_rounds and len(valid_sets) > 1:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    first_metric_only=True,
                    verbose=True,
                )
            )
        
        # 训练
        self.evals_result = {}
        self.model = lgb.train(
            params=self.params,
            train_set=dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
            **kwargs,
        )
        
        # 记录训练信息
        self.best_iteration = self.model.best_iteration if self.model.best_iteration else num_boost_round
        if len(valid_sets) > 1 and self.model.best_score:
            self.best_score = list(self.model.best_score.get("valid", {}).values())[0] \
                if self.model.best_score.get("valid") else float("inf")
        
        self.is_fitted = True
        self.train_info = {
            "num_features": len(feature_names),
            "num_train_samples": len(y_train),
            "num_valid_samples": len(y_valid) if y_valid is not None else 0,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "target_type": self.target_type.value,
        }
        
        logger.info(
            f"Training completed: best_iteration={self.best_iteration}, "
            f"best_score={self.best_score:.6f}"
        )
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        num_iteration: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        模型预测
        
        Args:
            X: 输入特征
            num_iteration: 使用的迭代次数（默认 best_iteration）
            **kwargs: 传递给 predict 的其他参数
            
        Returns:
            predictions: 预测结果数组
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        # 准备特征（需要保持与训练时相同的列顺序）
        if isinstance(X, pd.DataFrame):
            # 确保列顺序与训练时一致
            missing_cols = set(self.feature_names) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing features: {missing_cols}")
            X_pred = X[self.feature_names]
        else:
            X_pred = X
        
        num_iteration = num_iteration or self.best_iteration
        
        return self.model.predict(
            X_pred,
            num_iteration=num_iteration,
            **kwargs,
        )
    
    def get_feature_importance(
        self, 
        importance_type: str = "gain",
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        获取特征重要性
        
        强制使用 gain（信息增益）以获取更真实的因子有效性排名
        
        Args:
            importance_type: 重要性类型（"gain" 或 "split"）
            top_n: 返回 Top N 特征（None 返回全部）
            
        Returns:
            importance_df: 包含 feature, importance 列的 DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        # 强制使用 gain
        if importance_type != "gain":
            logger.warning(f"Using gain importance instead of {importance_type}")
            importance_type = "gain"
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        
        # 添加百分比
        total = importance_df["importance"].sum()
        importance_df["importance_pct"] = importance_df["importance"] / total * 100
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def save(self, path: Union[str, Path]) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "name": self.name,
            "seed": self.seed,
            "model": self.model,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "train_info": self.train_info,
            "lgb_config": self.lgb_config,
            "feature_config": self.feature_config,
            "target_type": self.target_type,
            "params": self.params,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "LGBQuantModel":
        """
        加载模型
        
        Args:
            path: 模型文件路径
            
        Returns:
            model: 加载的模型实例
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        
        instance = cls(
            name=save_dict["name"],
            lgb_config=save_dict.get("lgb_config"),
            feature_config=save_dict.get("feature_config"),
            target_type=save_dict.get("target_type", TargetType.REGRESSION),
        )
        
        instance.model = save_dict["model"]
        instance.is_fitted = save_dict["is_fitted"]
        instance.feature_names = save_dict["feature_names"]
        instance.train_info = save_dict.get("train_info", {})
        instance.params = save_dict.get("params", {})
        instance.best_iteration = save_dict.get("best_iteration", 0)
        instance.best_score = save_dict.get("best_score", float("inf"))
        instance.seed = save_dict.get("seed", 42)
        
        logger.info(f"Model loaded from {path}")
        return instance
    
    def __repr__(self) -> str:
        return (
            f"LGBQuantModel(name={self.name}, fitted={self.is_fitted}, "
            f"features={len(self.feature_names)}, best_iter={self.best_iteration})"
        )