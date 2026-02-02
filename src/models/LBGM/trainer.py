"""
LightGBM 训练循环

包含:
- lgb.Dataset 构建
- Early Stopping
- 模型保存
"""

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np

from .config import LGBMConfig
from .metrics import ic_eval, rank_ic_eval

logger = logging.getLogger(__name__)


class LGBMTrainer:
    """
    LightGBM 训练器
    
    特点:
    - GPU 加速训练
    - 自定义 IC 评价函数
    - Early Stopping 防止过拟合
    - 模型持久化
    """
    
    def __init__(self, config: LGBMConfig):
        """
        初始化训练器
        
        Args:
            config: LGBMConfig 配置对象
        """
        self.config = config
        self.model: Optional[lgb.Booster] = None
        self._train_history: Dict[str, List[float]] = {}
        self._best_iteration: int = 0
        self._training_time: float = 0.0
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        feature_names: List[str],
        categorical_features: Optional[List[str]] = None,
    ) -> lgb.Booster:
        """
        训练 LightGBM 模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_valid: 验证特征
            y_valid: 验证标签
            feature_names: 特征名列表
            categorical_features: 类别特征名列表
            
        Returns:
            训练好的 Booster 对象
        """
        logger.info("=" * 60)
        logger.info("🚀 开始训练 LightGBM 模型")
        logger.info("=" * 60)
        
        params = self.config.params.to_dict()
        train_config = self.config.train
        
        # 打印配置
        logger.info("  📋 模型配置:")
        logger.info(f"     设备: {params['device']}")
        logger.info(f"     叶子数: {params['num_leaves']}")
        logger.info(f"     树深度: {params['max_depth']}")
        logger.info(f"     学习率: {params['learning_rate']}")
        logger.info(f"     特征采样: {params['feature_fraction']}")
        logger.info(f"     样本采样: {params['bagging_fraction']}")
        logger.info(f"     L1 正则: {params['lambda_l1']}")
        logger.info(f"     L2 正则: {params['lambda_l2']}")
        
        logger.info(f"  📋 训练配置:")
        logger.info(f"     最大轮数: {train_config.num_boost_round}")
        logger.info(f"     早停轮数: {train_config.early_stopping_rounds}")
        
        # 处理类别特征索引
        cat_feature_indices = None
        if categorical_features:
            cat_feature_indices = [
                feature_names.index(f) for f in categorical_features 
                if f in feature_names
            ]
            logger.info(f"  📋 类别特征: {len(cat_feature_indices)} 个")
        
        # 构建 Dataset
        # lgb.Dataset 会将数据转为二进制直方图格式
        logger.info("  📊 构建 LightGBM Dataset...")
        
        train_set = lgb.Dataset(
            X_train, 
            label=y_train,
            feature_name=feature_names,
            categorical_feature=cat_feature_indices,
            free_raw_data=True,  # 释放原始数据节省内存
        )
        
        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=train_set,  # 共享特征直方图
            free_raw_data=True,
        )
        
        logger.info("  ✓ Dataset 构建完成")
        
        # 训练记录
        evals_result = {}
        
        # 训练回调
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=train_config.early_stopping_rounds,
                first_metric_only=True,
                verbose=True,
            ),
            lgb.log_evaluation(period=train_config.verbose_eval),
            lgb.record_evaluation(evals_result),  # 记录评估结果
        ]
        
        # 开始训练
        logger.info("  🏋️ 开始迭代训练...")
        start_time = time.time()
        
        self.model = lgb.train(
            params=params,
            train_set=train_set,
            num_boost_round=train_config.num_boost_round,
            valid_sets=[train_set, valid_set],
            valid_names=['train', 'valid'],
            feval=ic_eval,  # 使用自定义 IC 评价函数
            callbacks=callbacks,
        )
        
        self._training_time = time.time() - start_time
        self._best_iteration = self.model.best_iteration
        
        # 记录训练历史 (从 record_evaluation 回调中获取)
        self._train_history = {
            'train_IC': evals_result.get('train', {}).get('IC', []),
            'valid_IC': evals_result.get('valid', {}).get('IC', []),
        }
        
        logger.info("=" * 60)
        logger.info("  ✅ 训练完成")
        logger.info(f"     最佳轮次: {self._best_iteration}")
        logger.info(f"     训练耗时: {self._training_time:.2f} 秒")
        
        if self._train_history['valid_IC']:
            best_ic = max(self._train_history['valid_IC'])
            logger.info(f"     最佳验证 IC: {best_ic:.4f}")
        
        logger.info("=" * 60)
        
        # 释放内存
        del train_set, valid_set
        gc.collect()
        
        # 保存模型
        if train_config.save_model:
            self.save_model()
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测值数组
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train()")
        
        return self.model.predict(X, num_iteration=self._best_iteration)
    
    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        保存模型
        
        Args:
            path: 保存路径 (可选)
            
        Returns:
            实际保存路径
        """
        if self.model is None:
            raise RuntimeError("模型未训练，无法保存")
        
        output_dir = self.config.data.output_dir
        model_name = self.config.train.model_name
        
        if path is None:
            path = output_dir / f"{model_name}.pkl"
        
        # 使用 joblib 保存 (支持压缩)
        joblib.dump(self.model, path)
        logger.info(f"  💾 模型已保存: {path}")
        
        # 同时保存为 LightGBM 原生格式
        txt_path = path.with_suffix('.txt')
        self.model.save_model(str(txt_path))
        logger.info(f"  💾 模型已保存: {txt_path}")
        
        return path
    
    def load_model(self, path: Path) -> lgb.Booster:
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            Booster 对象
        """
        if path.suffix == '.pkl':
            self.model = joblib.load(path)
        else:
            self.model = lgb.Booster(model_file=str(path))
        
        logger.info(f"  📖 模型已加载: {path}")
        return self.model
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> Dict[str, float]:
        """
        获取特征重要性
        
        Args:
            importance_type: 'gain' (增益) 或 'split' (分裂次数)
            
        Returns:
            特征名 -> 重要性 的字典
        """
        if self.model is None:
            raise RuntimeError("模型未训练")
        
        feature_names = self.model.feature_name()
        importance = self.model.feature_importance(importance_type=importance_type)
        
        return dict(zip(feature_names, importance))
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """获取训练历史"""
        return self._train_history
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            "best_iteration": self._best_iteration,
            "training_time_seconds": self._training_time,
            "final_train_IC": self._train_history.get('train_IC', [None])[-1],
            "final_valid_IC": self._train_history.get('valid_IC', [None])[-1],
            "best_valid_IC": max(self._train_history.get('valid_IC', [0])),
        }
