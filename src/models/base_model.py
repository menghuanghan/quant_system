"""
统一的模型接口基类 (fit, predict, save, load)
所有量化模型必须继承此基类，遵循统一的API规范
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pickle
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    量化模型统一基类
    
    所有子类模型（LightGBM/GRU等）必须实现以下接口：
    - fit(): 训练模型
    - predict(): 预测
    - save(): 序列化模型
    - load(): 反序列化模型
    """
    
    def __init__(self, name: str = "BaseModel", seed: int = 42):
        """
        初始化基类
        
        Args:
            name: 模型名称标识
            seed: 全局随机种子
        """
        self.name = name
        self.seed = seed
        self.model = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.train_info: Dict[str, Any] = {}
        
    @abstractmethod
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_valid: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_valid: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> "BaseModel":
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_valid: 验证特征（用于早停）
            y_valid: 验证标签
            **kwargs: 额外训练参数
            
        Returns:
            self: 返回训练后的模型实例
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """
        模型预测
        
        Args:
            X: 输入特征
            **kwargs: 额外预测参数
            
        Returns:
            predictions: 预测结果数组
        """
        pass
    
    def save(self, path: Union[str, Path]) -> None:
        """
        序列化保存模型
        
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
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseModel":
        """
        反序列化加载模型
        
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
        
        instance = cls.__new__(cls)
        instance.name = save_dict["name"]
        instance.seed = save_dict["seed"]
        instance.model = save_dict["model"]
        instance.is_fitted = save_dict["is_fitted"]
        instance.feature_names = save_dict["feature_names"]
        instance.train_info = save_dict.get("train_info", {})
        
        logger.info(f"Model loaded from {path}")
        return instance
    
    @abstractmethod
    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            importance_type: 重要性类型 ("gain" / "split")
            
        Returns:
            importance_df: 包含 feature, importance 列的 DataFrame
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, fitted={self.is_fitted})"


class ModelEnsemble:
    """
    模型集成器
    
    支持多模型加权融合，用于实盘推断时的集成预测
    """
    
    def __init__(self, models: Optional[List[BaseModel]] = None, weights: Optional[List[float]] = None):
        """
        Args:
            models: 模型列表
            weights: 权重列表（与模型一一对应，默认等权）
        """
        self.models = models or []
        self.weights = weights
        
        if self.weights is None and self.models:
            # 默认等权
            self.weights = [1.0 / len(self.models)] * len(self.models)
    
    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """添加模型到集成"""
        self.models.append(model)
        if self.weights is None:
            self.weights = [weight]
        else:
            self.weights.append(weight)
        # 重新归一化权重
        self._normalize_weights()
    
    def _normalize_weights(self) -> None:
        """归一化权重使其和为1"""
        if self.weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        集成预测（加权平均）
        
        Args:
            X: 输入特征
            
        Returns:
            predictions: 加权平均后的预测结果
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    @classmethod
    def load_from_dir(
        cls, 
        model_dir: Union[str, Path], 
        pattern: str = "*.pkl",
        model_class: type = None
    ) -> "ModelEnsemble":
        """
        从目录加载所有模型
        
        Args:
            model_dir: 模型目录
            pattern: 文件匹配模式
            model_class: 模型类（用于反序列化）
            
        Returns:
            ensemble: 模型集成实例
        """
        model_dir = Path(model_dir)
        model_files = sorted(model_dir.glob(pattern))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir} with pattern {pattern}")
        
        models = []
        for f in model_files:
            if model_class:
                model = model_class.load(f)
            else:
                # 尝试通用加载
                with open(f, "rb") as fp:
                    save_dict = pickle.load(fp)
                model = save_dict.get("model")
            models.append(model)
        
        logger.info(f"Loaded {len(models)} models from {model_dir}")
        return cls(models=models)