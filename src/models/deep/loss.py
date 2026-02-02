"""
损失函数模块

核心: IC Loss (1 - Pearson Correlation)

设计考虑:
1. MSE 关注绝对误差, 但量化投资关注 "排序"
2. IC Loss 直接优化预测与真实值的相关性
3. 混合损失: 0.7 * MSE + 0.3 * IC_Loss 效果最好
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ICLoss(nn.Module):
    """
    IC Loss (Information Coefficient Loss)
    
    定义: Loss = 1 - Correlation(Y_pred, Y_true)
    
    注意事项:
    1. Batch Size 必须足够大 (> 1024), 否则相关系数波动太大
    2. 需要处理标准差为 0 的极端情况 (加 epsilon)
    """
    
    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: 防止除零的小常数
        """
        super().__init__()
        self.eps = eps
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        计算 IC Loss
        
        Args:
            y_pred: (N,) 预测值
            y_true: (N,) 真实值
            
        Returns:
            标量损失值
        """
        # 去中心化
        pred_mean = y_pred.mean()
        true_mean = y_true.mean()
        
        pred_centered = y_pred - pred_mean
        true_centered = y_true - true_mean
        
        # 计算协方差
        covariance = (pred_centered * true_centered).mean()
        
        # 计算标准差
        pred_std = pred_centered.pow(2).mean().sqrt()
        true_std = true_centered.pow(2).mean().sqrt()
        
        # 计算相关系数 (加 epsilon 防止除零)
        correlation = covariance / (pred_std * true_std + self.eps)
        
        # IC Loss = 1 - correlation
        # 目标是最大化相关性, 所以最小化 1 - correlation
        loss = 1.0 - correlation
        
        return loss


class RankICLoss(nn.Module):
    """
    Rank IC Loss (Spearman 相关系数)
    
    比 IC Loss 更鲁棒, 因为使用排名而非原始值
    但计算排名操作不可微, 需要用可微近似
    """
    
    def __init__(self, eps: float = 1e-8, temperature: float = 1.0):
        """
        Args:
            eps: 防止除零的小常数
            temperature: 软排名的温度参数
        """
        super().__init__()
        self.eps = eps
        self.temperature = temperature
    
    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        可微的软排名近似
        
        使用 softmax 对比较矩阵进行软化
        """
        n = x.size(0)
        
        # 构建比较矩阵: x[i] > x[j]
        # (n, n)
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # x[i] - x[j]
        
        # 软化的比较: sigmoid((x[i] - x[j]) / temperature)
        soft_compare = torch.sigmoid(diff / self.temperature)
        
        # 排名 ≈ 比自己小的元素数量 + 1
        ranks = soft_compare.sum(dim=1)
        
        return ranks
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        计算 Rank IC Loss
        """
        # 获取软排名
        pred_rank = self._soft_rank(y_pred)
        true_rank = self._soft_rank(y_true)
        
        # 计算 Pearson 相关系数 (在排名上)
        pred_mean = pred_rank.mean()
        true_mean = true_rank.mean()
        
        pred_centered = pred_rank - pred_mean
        true_centered = true_rank - true_mean
        
        covariance = (pred_centered * true_centered).mean()
        pred_std = pred_centered.pow(2).mean().sqrt()
        true_std = true_centered.pow(2).mean().sqrt()
        
        correlation = covariance / (pred_std * true_std + self.eps)
        
        return 1.0 - correlation


class CombinedLoss(nn.Module):
    """
    混合损失函数
    
    Loss = alpha * MSE + beta * IC_Loss
    
    推荐比例: alpha=0.7, beta=0.3
    - MSE 保证数值不离谱
    - IC Loss 保证排序正确
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        eps: float = 1e-8,
    ):
        """
        Args:
            alpha: MSE 权重
            beta: IC Loss 权重
            eps: 防止除零的小常数
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.ic_loss = ICLoss(eps=eps)
        
        logger.info(f"✓ CombinedLoss 初始化:")
        logger.info(f"  MSE 权重: {alpha}")
        logger.info(f"  IC Loss 权重: {beta}")
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        计算混合损失
        
        Args:
            y_pred: (N,) 预测值
            y_true: (N,) 真实值
            
        Returns:
            标量损失值
        """
        mse_loss = self.mse(y_pred, y_true)
        ic_loss = self.ic_loss(y_pred, y_true)
        
        combined = self.alpha * mse_loss + self.beta * ic_loss
        
        return combined


def compute_ic(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> float:
    """
    计算 IC (不用于反向传播, 仅用于评估)
    
    Returns:
        IC 值 (标量)
    """
    with torch.no_grad():
        pred_mean = y_pred.mean()
        true_mean = y_true.mean()
        
        pred_centered = y_pred - pred_mean
        true_centered = y_true - true_mean
        
        covariance = (pred_centered * true_centered).mean()
        pred_std = pred_centered.pow(2).mean().sqrt()
        true_std = true_centered.pow(2).mean().sqrt()
        
        ic = covariance / (pred_std * true_std + eps)
        
        return ic.item()


def compute_rank_ic(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    计算 Rank IC (Spearman 相关系数)
    
    使用精确排名计算, 仅用于评估
    """
    with torch.no_grad():
        # 转为 numpy 计算排名
        pred_np = y_pred.cpu().numpy()
        true_np = y_true.cpu().numpy()
        
        # 获取排名
        pred_rank = pred_np.argsort().argsort().astype(float)
        true_rank = true_np.argsort().argsort().astype(float)
        
        # Pearson on ranks
        pred_mean = pred_rank.mean()
        true_mean = true_rank.mean()
        
        pred_centered = pred_rank - pred_mean
        true_centered = true_rank - true_mean
        
        covariance = (pred_centered * true_centered).mean()
        pred_std = pred_centered.std()
        true_std = true_centered.std()
        
        if pred_std < 1e-8 or true_std < 1e-8:
            return 0.0
        
        rank_ic = covariance / (pred_std * true_std)
        
        return float(rank_ic)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建损失函数
    loss_fn = CombinedLoss(alpha=0.7, beta=0.3)
    
    # 模拟数据
    y_pred = torch.randn(1024, requires_grad=True)
    y_true = torch.randn(1024) + 0.3 * y_pred.detach()
    
    # 计算损失
    loss = loss_fn(y_pred, y_true)
    print(f"Combined Loss: {loss.item():.4f}")
    
    # 计算 IC
    ic = compute_ic(y_pred, y_true)
    rank_ic = compute_rank_ic(y_pred, y_true)
    print(f"IC: {ic:.4f}")
    print(f"Rank IC: {rank_ic:.4f}")
    
    # 测试反向传播
    loss.backward()
    print(f"Gradient norm: {y_pred.grad.norm().item():.4f}")
