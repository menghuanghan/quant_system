"""
多任务 GRU 神经网络（network.py）

架构设计:
    Input (batch, seq_len, num_features)
        ↓
    nn.LayerNorm(num_features)               # 时间维度分布对齐
        ↓
    nn.GRU(hidden_size=64, num_layers=1~2)   # 低层 GRU，防过拟合
        ↓
    Linear Attention (可选)                   # 对 seq_len 个时间步加权求和
        ↓
    context_vector (batch, hidden_size)
        ↓
    Multi-Head Output                         # 并行多个 nn.Linear(hidden_size, 1)
        ↓
    Output (batch, num_targets)
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """
    简单的线性注意力层

    对 seq_len 个时间步的 Hidden States 进行加权求和，
    让网络自己决定过去 seq_len 天里哪一天的形态最重要。

    score_t = w^T * tanh(W * h_t + b)
    alpha   = softmax(scores)
    context = sum(alpha_t * h_t)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, gru_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gru_output: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size)
        """
        # (batch, seq_len, hidden_size)
        energy = torch.tanh(self.attn_proj(gru_output))
        # (batch, seq_len, 1)
        scores = self.attn_score(energy)
        # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)
        # (batch, hidden_size)
        context = (gru_output * weights).sum(dim=1)
        return context


class MultiTaskGRUNetwork(nn.Module):
    """
    多任务 GRU 网络

    Args:
        num_features: 输入特征维度
        hidden_size: GRU 隐层大小 (推荐 64)
        num_layers: GRU 层数 (推荐 1~2)
        dropout: GRU dropout
        num_targets: 输出目标数 (len(target_cols))
        use_attention: 是否使用时间注意力
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_targets: int = 3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.use_attention = use_attention

        # ---- 1. LayerNorm: 拉平时间维度上的分布偏移 ----
        self.layer_norm = nn.LayerNorm(num_features)

        # ---- 2. GRU ----
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ---- 3. 时间注意力（可选） ----
        if use_attention:
            self.attention = TemporalAttention(hidden_size)

        # ---- 4. 多头输出层 ----
        # 每个目标各自一个 Linear(hidden_size, 1)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_targets)
        ])

        # 初始化权重
        self._init_weights()
        self._log_info()

    def _init_weights(self):
        # GRU
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        # 多头
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def _log_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"MultiTaskGRUNetwork: "
            f"features={self.num_features}, hidden={self.hidden_size}, "
            f"layers={self.num_layers}, targets={self.num_targets}, "
            f"attention={self.use_attention}, "
            f"params={total:,} (trainable={trainable:,})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, seq_len, num_features)

        Returns:
            preds: (batch, num_targets)
        """
        # 1. LayerNorm
        x = self.layer_norm(x)

        # 2. GRU
        # gru_out: (batch, seq_len, hidden_size)
        gru_out, _ = self.gru(x)

        # 3. 提取时序浓缩表征
        if self.use_attention:
            context = self.attention(gru_out)        # (batch, hidden_size)
        else:
            context = gru_out[:, -1, :]              # (batch, hidden_size)

        # 4. 多头输出
        head_outputs = [head(context) for head in self.heads]  # list of (batch, 1)
        preds = torch.cat(head_outputs, dim=-1)                # (batch, num_targets)

        return preds
