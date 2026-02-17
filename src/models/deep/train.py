"""
GRU 训练模块

核心功能:
1. 混合精度训练 (torch.amp)
2. AdamW 优化器 + CosineAnnealingLR
3. 早停 (监控 Validation RankIC)
4. RTX 5070 优化 (大 Batch Size)
5. 可复现性 (固定随机种子)

改造说明（2026.02）:
- 适配新配置系统
- 支持可配置的梯度裁剪
- 完善日志输出
- 添加 seed_everything 函数确保可复现性
"""

import logging
import time
import gc
import os
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from .model import GRUModel, ModelConfig, create_model
from .loss import CombinedLoss, ICLoss, compute_ic, compute_rank_ic
from .dataset import StockDataset
from .config import GRUTrainConfig, GRUConfig

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42):
    """
    固定所有随机源，确保训练可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 以下两行会让训练变慢，但能保证完全一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🎲 随机种子已固定: {seed}")


@dataclass
class TrainConfig:
    """训练配置（保留兼容性）"""
    
    # 基础配置
    epochs: int = 100
    batch_size: int = 2048  # RTX 5070 12GB 可以跑 2048-4096
    num_workers: int = 4    # WSL2 下不要太大
    
    # 优化器
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # cosine / step / none
    lr_min: float = 1e-5
    
    # 损失函数类型
    loss_type: str = "combined"  # combined / ic_only / mse
    # 当 loss_type="combined" 时使用以下权重
    mse_weight: float = 0.5
    ic_weight: float = 0.5
    
    # 早停
    patience: int = 15  # 增加耐心
    min_epochs: int = 5  # 最少训练轮数
    
    # 混合精度
    use_amp: bool = True
    
    # 梯度裁剪
    grad_clip: float = 1.0
    
    # 保存
    save_dir: Path = Path("/home/menghuanghan/quant_system/models/gru")
    save_best: bool = True
    model_name: str = "gru_excess_ret_5d"
    best_model_prefix: str = "best_model"  # 最佳模型文件名前缀（支持多种子模式）
    
    # 日志
    log_interval: int = 100  # 每多少 batch 打印一次


class GRUTrainer:
    """
    GRU 训练器
    
    实现:
    1. 混合精度训练 (AMP)
    2. 梯度裁剪
    3. 早停机制
    4. 模型保存与加载
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        device: str = "cuda",
    ):
        """
        Args:
            model: GRU 模型
            config: 训练配置
            device: 设备 (cuda / cpu)
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 损失函数 - 根据 loss_type 选择
        if config.loss_type == "ic_only":
            self.criterion = ICLoss()
            logger.info("📊 使用纯 IC Loss")
        else:
            self.criterion = CombinedLoss(
                alpha=config.mse_weight,
                beta=config.ic_weight,
            )
            logger.info(f"📊 使用 Combined Loss (MSE={config.mse_weight}, IC={config.ic_weight})")
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度
        self.scaler = GradScaler("cuda") if config.use_amp and device == "cuda" else None
        
        # 训练状态
        self.current_epoch = 0
        self.best_valid_ic = -float('inf')
        self.patience_counter = 0
        
        # 训练历史
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'valid_loss': [],
            'train_ic': [],
            'valid_ic': [],
            'valid_rank_ic': [],
            'lr': [],
        }
        
        # 创建保存目录
        config.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🤖 GRU Trainer 初始化完成")
        logger.info(f"  设备: {device}")
        logger.info(f"  混合精度: {config.use_amp}")
        logger.info(f"  Batch Size: {config.batch_size}")
        logger.info(f"  学习率: {config.learning_rate}")
    
    def _create_scheduler(self) -> Optional[Any]:
        """创建学习率调度器"""
        config = self.config
        
        if config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
                eta_min=config.lr_min,
            )
        elif config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        else:
            return None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Tuple[float, float]:
        """
        训练一个 epoch
        
        支持两种数据格式:
        - (X, y): 纯数值特征
        - (X, cat_dict, y): 数值特征 + 类别特征字典
        
        Returns:
            (epoch_loss, epoch_ic)
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        n_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # 解析 batch 数据 (支持两种格式)
            if len(batch_data) == 2:
                X, y = batch_data
                cat_features = None
            else:
                X, cat_features, y = batch_data
            
            # 如果数据不在目标设备上，才做设备转移
            if X.device != torch.device(self.device):
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if cat_features is not None:
                    cat_features = {k: v.to(self.device, non_blocking=True) for k, v in cat_features.items()}
            
            self.optimizer.zero_grad()
            
            # 混合精度前向
            if self.scaler is not None:
                with autocast("cuda"):
                    # 根据模型类型调用前向传播
                    if cat_features is not None and hasattr(self.model, 'embeddings'):
                        pred = self.model(X, cat_features=cat_features)
                    else:
                        pred = self.model(X)
                    loss = self.criterion(pred, y)
                
                # 混合精度反向
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪（使用配置参数）
                self.scaler.unscale_(self.optimizer)
                grad_clip = getattr(self.config, 'grad_clip', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 根据模型类型调用前向传播
                if cat_features is not None and hasattr(self.model, 'embeddings'):
                    pred = self.model(X, cat_features=cat_features)
                else:
                    pred = self.model(X)
                loss = self.criterion(pred, y)
                loss.backward()
                grad_clip = getattr(self.config, 'grad_clip', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            # 收集预测值用于计算 IC
            all_preds.append(pred.detach())
            all_labels.append(y.detach())
            
            # 日志
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / n_batches
                logger.debug(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
        
        # 计算 epoch IC
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        epoch_ic = compute_ic(all_preds, all_labels)
        epoch_loss = total_loss / n_batches
        
        return epoch_loss, epoch_ic
    
    @torch.no_grad()
    def validate(
        self,
        valid_loader: DataLoader,
    ) -> Tuple[float, float, float]:
        """
        验证
        
        支持两种数据格式:
        - (X, y): 纯数值特征
        - (X, cat_dict, y): 数值特征 + 类别特征字典
        
        Returns:
            (loss, ic, rank_ic)
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        n_batches = 0
        
        for batch_data in valid_loader:
            # 解析 batch 数据 (支持两种格式)
            if len(batch_data) == 2:
                X, y = batch_data
                cat_features = None
            else:
                X, cat_features, y = batch_data
            
            # 如果数据不在目标设备上，才做设备转移
            if X.device != torch.device(self.device):
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if cat_features is not None:
                    cat_features = {k: v.to(self.device, non_blocking=True) for k, v in cat_features.items()}
            
            if self.config.use_amp and self.device == "cuda":
                with autocast("cuda"):
                    if cat_features is not None and hasattr(self.model, 'embeddings'):
                        pred = self.model(X, cat_features=cat_features)
                    else:
                        pred = self.model(X)
                    loss = self.criterion(pred, y)
            else:
                if cat_features is not None and hasattr(self.model, 'embeddings'):
                    pred = self.model(X, cat_features=cat_features)
                else:
                    pred = self.model(X)
                loss = self.criterion(pred, y)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(pred)
            all_labels.append(y)
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        ic = compute_ic(all_preds, all_labels)
        rank_ic = compute_rank_ic(all_preds, all_labels)
        avg_loss = total_loss / n_batches
        
        return avg_loss, ic, rank_ic
    
    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Returns:
            训练历史
        """
        logger.info("=" * 60)
        logger.info("🚀 开始训练 GRU 模型")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # 训练
            train_loss, train_ic = self.train_epoch(train_loader)
            
            # 验证
            valid_loss, valid_ic, valid_rank_ic = self.validate(valid_loader)
            
            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)
            self.history['train_ic'].append(train_ic)
            self.history['valid_ic'].append(valid_ic)
            self.history['valid_rank_ic'].append(valid_rank_ic)
            self.history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # 打印日志
            logger.info(
                f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f}, IC: {train_ic:.4f} | "
                f"Valid Loss: {valid_loss:.4f}, IC: {valid_ic:.4f}, RankIC: {valid_rank_ic:.4f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )
            
            # 早停检查
            if valid_rank_ic > self.best_valid_ic:
                self.best_valid_ic = valid_rank_ic
                self.patience_counter = 0
                
                # 保存最佳模型（使用 best_model_prefix 支持多种子模式）
                if self.config.save_best:
                    best_model_filename = f"{self.config.best_model_prefix}.pt"
                    self.save_model(best_model_filename)
                    logger.info(f"  💾 保存最佳模型 (RankIC: {valid_rank_ic:.4f})")
            else:
                self.patience_counter += 1
                # 只有超过 min_epochs 才检查早停
                if epoch >= self.config.min_epochs and self.patience_counter >= self.config.patience:
                    logger.info(f"⏹️ 早停: {self.config.patience} epoch 无提升 (min_epochs={self.config.min_epochs})")
                    break
        
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("✅ 训练完成")
        logger.info(f"  总耗时: {total_time:.1f}s")
        logger.info(f"  最佳验证 RankIC: {self.best_valid_ic:.4f}")
        logger.info("=" * 60)
        
        return self.history
    
    def save_model(self, filename: str):
        """保存模型"""
        path = self.config.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_valid_ic': self.best_valid_ic,
            'history': self.history,
        }, path)
    
    def load_model(self, filename: str):
        """加载模型"""
        path = self.config.save_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_valid_ic = checkpoint['best_valid_ic']
        self.history = checkpoint['history']
        
        logger.info(f"✓ 加载模型: {path}")
    
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        """
        预测
        
        支持两种数据格式:
        - (X, y): 纯数值特征
        - (X, cat_dict, y): 数值特征 + 类别特征字典
        
        Returns:
            预测值数组
        """
        self.model.eval()
        
        all_preds = []
        for batch_data in loader:
            # 解析 batch 数据 (支持两种格式)
            if len(batch_data) == 2:
                X, _ = batch_data
                cat_features = None
            else:
                X, cat_features, _ = batch_data
            
            X = X.to(self.device)
            if cat_features is not None:
                cat_features = {k: v.to(self.device) for k, v in cat_features.items()}
            
            if self.config.use_amp and self.device == "cuda":
                with autocast("cuda"):
                    if cat_features is not None and hasattr(self.model, 'embeddings'):
                        pred = self.model(X, cat_features=cat_features)
                    else:
                        pred = self.model(X)
            else:
                if cat_features is not None and hasattr(self.model, 'embeddings'):
                    pred = self.model(X, cat_features=cat_features)
                else:
                    pred = self.model(X)
            
            all_preds.append(pred.cpu().numpy())
        
        return np.concatenate(all_preds)


def create_dataloader(
    dataset: StockDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,  # GPU 数据集不需要多进程
) -> DataLoader:
    """
    创建 DataLoader
    
    注意: 
    - 如果数据已在 GPU 上，num_workers 必须为 0
    - GPU 数据集不需要 pin_memory
    """
    # 检查数据是否在 GPU 上
    is_gpu_dataset = hasattr(dataset, 'device') and 'cuda' in str(dataset.device)
    
    if is_gpu_dataset:
        # GPU 数据集: 单进程, 无需 pin_memory
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # GPU 数据必须用单进程
            pin_memory=False,
            drop_last=shuffle,
        )
    else:
        # CPU 数据集: 多进程 + pin_memory 加速传输
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=shuffle,
        )


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型
    model = create_model(input_dim=30, hidden_dim=64)
    
    # 创建训练器
    config = TrainConfig(epochs=5, batch_size=32)
    trainer = GRUTrainer(model, config, device="cpu")
    
    # 模拟数据
    from torch.utils.data import TensorDataset
    X_train = torch.randn(1000, 20, 30)
    y_train = torch.randn(1000)
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    X_valid = torch.randn(200, 20, 30)
    y_valid = torch.randn(200)
    valid_ds = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_ds, batch_size=32)
    
    # 训练
    history = trainer.train(train_loader, valid_loader)
    print(f"最佳验证 IC: {trainer.best_valid_ic:.4f}")
