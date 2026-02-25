"""
GRU 多任务模型（gru_model.py）

核心职责:
- 多任务联合损失函数 (MultiTaskLoss / UncertaintyWeightedLoss)
- 封装网络 / 训练循环 / 多任务损失
- 混合精度 (AMP) + 梯度裁剪
- AdamW + OneCycleLR（余弦退火）
- 以 Rank IC 为核心的早停策略
- 继承 BaseModel 基类，遵循统一 API 规范
- 模型评估复用 QuantEvaluator

改造说明（2026.02）:
- 继承 BaseModel，统一 fit/predict/save/load 接口
- 多任务损失从 loss.py 合并至此
- IC/RankIC 计算改用 src/models/metrics/evaluator.py
"""

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..base_model import BaseModel
from ..config import GRUTrainConfig
from ..metrics import QuantEvaluator
from .network import MultiTaskGRUNetwork

logger = logging.getLogger(__name__)


# ============================================================================
# 多任务损失函数
# ============================================================================

class MultiTaskLoss(nn.Module):
    """
    多任务联合损失（静态权重版本）

    对每个标签分别计算损失，按预设权重加权求和:
    Total = sum(weight_i * loss_i(pred_i, true_i))

    Args:
        target_cols: 多目标列名列表
        loss_weights: {col: weight} 各任务权重
        loss_types: {col: 'mse'|'huber'} 各任务损失类型
    """

    def __init__(
        self,
        target_cols: List[str],
        loss_weights: Dict[str, float],
        loss_types: Dict[str, str],
    ):
        super().__init__()
        self.target_cols = target_cols
        self.n_targets = len(target_cols)

        # 权重（归一化）
        raw_weights = [loss_weights.get(col, 1.0 / self.n_targets) for col in target_cols]
        total_w = sum(raw_weights)
        self.weights = [w / total_w for w in raw_weights]

        # 各任务的损失函数
        self.loss_fns = nn.ModuleList()
        for col in target_cols:
            lt = loss_types.get(col, "mse")
            if lt == "huber":
                self.loss_fns.append(nn.SmoothL1Loss())
            else:
                self.loss_fns.append(nn.MSELoss())

        info = ", ".join(
            f"{col}({loss_types.get(col, 'mse')}, w={w:.2f})"
            for col, w in zip(target_cols, self.weights)
        )
        logger.info(f"MultiTaskLoss: {info}")

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            preds: (batch, n_targets)
            targets: (batch, n_targets)

        Returns:
            total_loss: scalar
        """
        total = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
        for i, (fn, w) in enumerate(zip(self.loss_fns, self.weights)):
            loss_i = fn(preds[:, i], targets[:, i])
            total = total + w * loss_i
        return total


class UncertaintyWeightedLoss(nn.Module):
    """
    动态不确定性加权损失

    基于 "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al.)
    每个任务有可学习参数 log_sigma_i:
    L_total = sum( 1/(2*sigma_i^2) * L_i + log(sigma_i) )
    """

    def __init__(
        self,
        target_cols: List[str],
        loss_types: Dict[str, str],
    ):
        super().__init__()
        self.target_cols = target_cols
        self.n_targets = len(target_cols)

        # 可学习的 log_sigma
        self.log_sigmas = nn.Parameter(torch.zeros(self.n_targets))

        # 各任务基础损失
        self.loss_fns = nn.ModuleList()
        for col in target_cols:
            lt = loss_types.get(col, "mse")
            if lt == "huber":
                self.loss_fns.append(nn.SmoothL1Loss())
            else:
                self.loss_fns.append(nn.MSELoss())

        logger.info(f"UncertaintyWeightedLoss: n_targets={self.n_targets}")

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
        for i, fn in enumerate(self.loss_fns):
            loss_i = fn(preds[:, i], targets[:, i])
            precision = torch.exp(-2 * self.log_sigmas[i])
            total = total + precision * loss_i + self.log_sigmas[i]
        return total


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int):
    """固定全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")


# ============================================================================
# GRU 多任务模型（继承 BaseModel）
# ============================================================================

class GRUModel(BaseModel):
    """
    GRU 多任务训练模型

    继承 BaseModel，遵循统一 API 规范：
    - fit(): 训练（接受 DataLoader）
    - predict(): 预测（接受 DataLoader）
    - save() / load(): 序列化（torch.save / torch.load）
    - get_feature_importance(): 特征重要性（GRU 返回空 DataFrame）

    Args:
        target_cols: 多目标标签列名列表
        num_features: 特征数
        config: 训练配置
        device: 训练设备
        seed: 随机种子
        hidden_size: GRU 隐层大小
        num_layers: GRU 层数
        dropout: dropout 率
        use_attention: 是否使用时间注意力
        name: 模型名称标识
    """

    def __init__(
        self,
        target_cols: List[str],
        num_features: int,
        config: GRUTrainConfig,
        device: str = "cuda",
        seed: int = 42,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        name: str = "GRUModel",
    ):
        super().__init__(name=name, seed=seed)

        self.target_cols = target_cols
        self.num_features = num_features
        self.config = config
        self.device = device

        # 固定种子
        set_seed(seed)

        # ---- 构建网络 ----
        self.network = MultiTaskGRUNetwork(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_targets=len(target_cols),
            use_attention=use_attention,
        ).to(device)

        # BaseModel 的 model 属性指向 network
        self.model = self.network

        # ---- 构建损失函数 ----
        if config.use_uncertainty_weighting:
            self.criterion = UncertaintyWeightedLoss(
                target_cols=target_cols,
                loss_types=config.loss_types,
            ).to(device)
        else:
            self.criterion = MultiTaskLoss(
                target_cols=target_cols,
                loss_weights=config.loss_weights,
                loss_types=config.loss_types,
            ).to(device)

        # ---- 优化器 (AdamW) ----
        params = list(self.network.parameters())
        if isinstance(self.criterion, UncertaintyWeightedLoss):
            params += list(self.criterion.parameters())

        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 调度器在 fit() 里根据实际 steps 创建
        self.scheduler = None

        # ---- 混合精度 ----
        self.scaler = GradScaler("cuda") if config.use_amp and device == "cuda" else None

        # ---- 训练状态 ----
        self.best_rank_ic = -float('inf')
        self.patience_counter = 0
        self.best_state_dict = None

        # ---- 训练历史 ----
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'valid_loss': [],
            'valid_rank_ic': [],
            'lr': [],
        }

        # ---- 找到 rank 通道索引（用于早停） ----
        self.rank_channel_idx = self._find_rank_channel()

        # ---- 评估器（复用统一 QuantEvaluator） ----
        self._evaluator = QuantEvaluator()

        logger.info(
            f"GRUModel 初始化: targets={target_cols}, "
            f"features={num_features}, seed={seed}, "
            f"rank_channel={self.rank_channel_idx}"
        )

    def _find_rank_channel(self) -> int:
        """找到 rank_ret_* / rank_* 通道的索引"""
        for i, col in enumerate(self.target_cols):
            if col.startswith("rank"):
                return i
        logger.warning("未找到 rank 通道，使用第 0 个目标作为早停指标")
        return 0

    def fit(
        self,
        X_train: Union[DataLoader, pd.DataFrame, np.ndarray],
        y_train: Union[None, pd.Series, np.ndarray] = None,
        X_valid: Optional[Union[DataLoader, pd.DataFrame, np.ndarray]] = None,
        y_valid: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs,
    ) -> "GRUModel":
        """
        训练模型

        GRU 模式下 X_train/X_valid 为 DataLoader，y_train/y_valid 忽略
        （标签已包含在 DataLoader 中）。

        Args:
            X_train: 训练 DataLoader
            y_train: 忽略（兼容 BaseModel 签名）
            X_valid: 验证 DataLoader
            y_valid: 忽略（兼容 BaseModel 签名）
            **kwargs: 额外参数

        Returns:
            self: 训练后的模型实例
        """
        train_loader = X_train
        valid_loader = X_valid
        config = self.config

        # 创建 OneCycleLR（需要知道总 steps）
        total_steps = config.epochs * len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.max_lr,
            total_steps=total_steps,
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor,
        )

        logger.info("=" * 60)
        logger.info("开始训练 GRU 多任务模型")
        logger.info(f"  Epochs: {config.epochs}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Valid batches: {len(valid_loader)}")
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(config.epochs):
            epoch_start = time.time()

            # ---- 训练 ----
            train_loss = self._train_epoch(train_loader)

            # ---- 验证 ----
            valid_loss, valid_rank_ic, valid_metrics = self._validate(valid_loader)

            # ---- 记录 ----
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)
            self.history['valid_rank_ic'].append(valid_rank_ic)
            self.history['lr'].append(current_lr)

            epoch_time = time.time() - epoch_start

            # ---- 日志 ----
            metrics_str = ", ".join(
                f"{col}:IC={v['ic']:.4f}/RkIC={v['rank_ic']:.4f}"
                for col, v in valid_metrics.items()
            )
            logger.info(
                f"Epoch {epoch + 1:3d}/{config.epochs} | "
                f"TrainLoss={train_loss:.4f} | "
                f"ValidLoss={valid_loss:.4f} | "
                f"RankIC={valid_rank_ic:.4f} | "
                f"LR={current_lr:.2e} | "
                f"Time={epoch_time:.1f}s"
            )
            if epoch % 10 == 0 or epoch == config.epochs - 1:
                logger.info(f"  Details: {metrics_str}")

            # ---- 早停（以 Rank IC 为准） ----
            if valid_rank_ic > self.best_rank_ic:
                self.best_rank_ic = valid_rank_ic
                self.patience_counter = 0
                # 保存最佳权重到内存
                self.best_state_dict = {
                    k: v.cpu().clone() for k, v in self.network.state_dict().items()
                }
                logger.info(f"  >>> 新最佳模型 (RankIC={valid_rank_ic:.4f})")
            else:
                self.patience_counter += 1
                if epoch >= config.min_epochs and self.patience_counter >= config.patience:
                    logger.info(
                        f"早停: {config.patience} 轮无提升 "
                        f"(best RankIC={self.best_rank_ic:.4f})"
                    )
                    break

        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(
            f"训练完成: 耗时={total_time:.1f}s, "
            f"最佳 RankIC={self.best_rank_ic:.4f}"
        )
        logger.info("=" * 60)

        # 恢复最佳权重
        if self.best_state_dict is not None:
            self.network.load_state_dict(self.best_state_dict)
            self.network.to(self.device)

        self.is_fitted = True
        self.train_info = {
            "best_rank_ic": self.best_rank_ic,
            "epochs_trained": len(self.history['train_loss']),
            "target_cols": self.target_cols,
            "num_features": self.num_features,
        }

        return self

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个 epoch"""
        self.network.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (X, Y) in enumerate(train_loader):
            if X.device != torch.device(self.device):
                X = X.to(self.device, non_blocking=True)
                Y = Y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast("cuda"):
                    preds = self.network(X)       # (batch, num_targets)
                    loss = self.criterion(preds, Y)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), max_norm=self.config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.network(X)
                loss = self.criterion(preds, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), max_norm=self.config.grad_clip
                )
                self.optimizer.step()

            # OneCycleLR 每个 step 都要调度
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.debug(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss={total_loss / n_batches:.4f}"
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(
        self, valid_loader: DataLoader,
    ) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
        """
        验证（使用 QuantEvaluator 计算 IC / RankIC）

        Returns:
            (valid_loss, rank_ic_of_rank_channel, per_target_metrics)
        """
        self.network.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        for X, Y in valid_loader:
            if X.device != torch.device(self.device):
                X = X.to(self.device, non_blocking=True)
                Y = Y.to(self.device, non_blocking=True)

            if self.config.use_amp and self.device == "cuda":
                with autocast("cuda"):
                    preds = self.network(X)
                    loss = self.criterion(preds, Y)
            else:
                preds = self.network(X)
                loss = self.criterion(preds, Y)

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(preds.cpu())
            all_targets.append(Y.cpu())

        all_preds_np = torch.cat(all_preds, dim=0).numpy()      # (N, num_targets)
        all_targets_np = torch.cat(all_targets, dim=0).numpy()

        # 使用 QuantEvaluator 计算每个目标的 IC / Rank IC
        metrics = {}
        for i, col in enumerate(self.target_cols):
            ic = self._evaluator._calc_ic(all_targets_np[:, i], all_preds_np[:, i])
            ric = self._evaluator._calc_rank_ic(all_targets_np[:, i], all_preds_np[:, i])
            metrics[col] = {
                "ic": ic if not np.isnan(ic) else 0.0,
                "rank_ic": ric if not np.isnan(ric) else 0.0,
            }

        # Rank 通道的 Rank IC 作为早停指标
        rank_col = self.target_cols[self.rank_channel_idx]
        rank_ic = metrics[rank_col]["rank_ic"]

        return total_loss / max(n_batches, 1), rank_ic, metrics

    @torch.no_grad()
    def predict(
        self,
        X: Union[DataLoader, pd.DataFrame, np.ndarray],
        **kwargs,
    ) -> np.ndarray:
        """
        模型预测

        Args:
            X: DataLoader（GRU 模式）

        Returns:
            preds: (N, num_targets) numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        self.network.eval()
        all_preds = []
        loader = X

        for batch_X, _ in loader:
            if batch_X.device != torch.device(self.device):
                batch_X = batch_X.to(self.device, non_blocking=True)

            if self.config.use_amp and self.device == "cuda":
                with autocast("cuda"):
                    preds = self.network(batch_X)
            else:
                preds = self.network(batch_X)

            all_preds.append(preds.cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    def save(self, path: Union[str, Path]) -> None:
        """保存模型（网络权重 + 配置元信息）"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'name': self.name,
            'seed': self.seed,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'train_info': self.train_info,
            'network_state_dict': self.network.state_dict(),
            'target_cols': self.target_cols,
            'num_features': self.num_features,
            'best_rank_ic': self.best_rank_ic,
            'history': self.history,
            'network_config': {
                'num_features': self.network.num_features,
                'hidden_size': self.network.hidden_size,
                'num_layers': self.network.num_layers,
                'num_targets': self.network.num_targets,
                'use_attention': self.network.use_attention,
            },
        }
        torch.save(save_dict, path)
        logger.info(f"模型已保存: {path}")

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cuda") -> "GRUModel":
        """
        加载模型

        Args:
            path: 模型文件路径
            device: 推断设备

        Returns:
            model: 加载好的 GRUModel 实例
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        nc = checkpoint['network_config']
        target_cols = checkpoint['target_cols']
        num_features = checkpoint['num_features']
        seed = checkpoint.get('seed', 42)
        name = checkpoint.get('name', 'GRUModel')

        # 创建一个轻量 config（推断不需要训练参数）
        config = GRUTrainConfig()

        model = cls(
            target_cols=target_cols,
            num_features=num_features,
            config=config,
            device=device,
            seed=seed,
            hidden_size=nc['hidden_size'],
            num_layers=nc['num_layers'],
            dropout=0.0,  # 推断时关闭 dropout
            use_attention=nc['use_attention'],
            name=name,
        )

        model.network.load_state_dict(checkpoint['network_state_dict'])
        model.network.eval()
        model.is_fitted = checkpoint.get('is_fitted', True)
        model.feature_names = checkpoint.get('feature_names', [])
        model.train_info = checkpoint.get('train_info', {})
        model.best_rank_ic = checkpoint.get('best_rank_ic', 0.0)
        model.history = checkpoint.get('history', {})

        logger.info(
            f"模型加载成功: {path}, "
            f"targets={target_cols}, "
            f"best_rank_ic={model.best_rank_ic:.4f}"
        )
        return model

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        获取特征重要性

        GRU 模型不具备传统树模型特征重要性，返回空 DataFrame。
        可通过外部工具（如 SHAP、Captum）获取归因分析。
        """
        logger.warning("GRU 模型不支持内置特征重要性，返回空 DataFrame")
        return pd.DataFrame(columns=["feature", "importance"])

    def __repr__(self) -> str:
        return (
            f"GRUModel(name={self.name}, fitted={self.is_fitted}, "
            f"features={self.num_features}, targets={len(self.target_cols)}, "
            f"best_rank_ic={self.best_rank_ic:.4f})"
        )
