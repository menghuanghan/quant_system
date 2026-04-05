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


class FeatureSpatialDropout1D(nn.Module):
    """
    Spatial Dropout 1D（通道级）

    对输入 x: (batch, seq_len, features) 按特征通道采样 dropout mask，
    mask 形状为 (batch, 1, features)，沿时间维共享，
    从而避免普通元素级 dropout 对时序结构造成噪声破坏。
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout 概率 p 必须在 [0,1]，当前={p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        if self.p >= 1.0:
            return torch.zeros_like(x)

        keep_prob = 1.0 - self.p
        # (batch, 1, features): 每个样本每个特征通道共享同一时间维 mask
        mask = x.new_empty((x.size(0), 1, x.size(2))).bernoulli_(keep_prob)
        mask = mask.div_(keep_prob)
        return x * mask


class MultiTaskGRUNetwork(nn.Module):
    """
    多任务 GRU 网络

    Args:
        num_features: 输入特征维度
        hidden_size: GRU 隐层大小 (默认 32)
        num_layers: GRU 层数 (默认 1)
        dropout: GRU dropout
        num_targets: 输出目标数 (len(target_cols))
        use_attention: 是否使用时间注意力
    """

    def __init__(
        self,
        num_features: int,
        num_cont_features: Optional[int] = None,
        num_cat_features: int = 0,
        cat_cardinalities: Optional[List[int]] = None,
        cat_embedding_dims: Optional[List[int]] = None,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        num_targets: int = 3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.num_cont_features = num_cont_features if num_cont_features is not None else num_features
        self.num_cat_features = num_cat_features
        self.cat_cardinalities = list(cat_cardinalities or [])

        if self.num_cat_features != len(self.cat_cardinalities):
            if self.num_cat_features == 0:
                self.cat_cardinalities = []
            else:
                raise ValueError(
                    f"num_cat_features={self.num_cat_features} 与 cat_cardinalities={len(self.cat_cardinalities)} 不一致"
                )

        if cat_embedding_dims is None:
            self.cat_embedding_dims = [
                max(4, min(32, int((card + 1) // 2)))
                for card in self.cat_cardinalities
            ]
        else:
            self.cat_embedding_dims = list(cat_embedding_dims)

        if len(self.cat_embedding_dims) != self.num_cat_features:
            raise ValueError(
                f"cat_embedding_dims 数量({len(self.cat_embedding_dims)})与 num_cat_features({self.num_cat_features}) 不一致"
            )

        self.num_features = self.num_cont_features + int(sum(self.cat_embedding_dims))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.use_attention = use_attention

        # ---- 0. 类别 Embedding ----
        self.cat_embeddings = nn.ModuleList()
        for card, emb_dim in zip(self.cat_cardinalities, self.cat_embedding_dims):
            self.cat_embeddings.append(nn.Embedding(num_embeddings=card, embedding_dim=emb_dim))

        # ---- 1. 前融合层 ----
        self.fuse_linear = nn.Linear(self.num_features, self.num_features)
        self.fuse_dropout = FeatureSpatialDropout1D(dropout)
        self.layer_norm = nn.LayerNorm(self.num_features)

        # ---- 1.5 进入 GRU 前的单头时间注意力 + 残差 ----
        self.pre_attn_proj = nn.Linear(self.num_features, self.num_features, bias=True)
        self.pre_attn_score = nn.Linear(self.num_features, 1, bias=False)

        # ---- 2. GRU ----
        self.gru = nn.GRU(
            input_size=self.num_features,
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
        # embedding
        for emb in self.cat_embeddings:
            nn.init.uniform_(emb.weight, a=-0.05, b=0.05)

        # fusion + pre-attn
        nn.init.xavier_uniform_(self.fuse_linear.weight)
        nn.init.zeros_(self.fuse_linear.bias)
        nn.init.xavier_uniform_(self.pre_attn_proj.weight)
        nn.init.zeros_(self.pre_attn_proj.bias)
        nn.init.xavier_uniform_(self.pre_attn_score.weight)

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
            f"cont={self.num_cont_features}, cat={self.num_cat_features}, "
            f"fused_features={self.num_features}, hidden={self.hidden_size}, "
            f"layers={self.num_layers}, targets={self.num_targets}, "
            f"attention={self.use_attention}, "
            f"params={total:,} (trainable={trainable:,})"
        )

    def _fuse_inputs(
        self,
        x_cont: torch.Tensor,
        x_cat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.num_cat_features <= 0:
            x = x_cont
        else:
            if x_cat is None:
                raise ValueError("网络配置包含类别特征，但 forward 未提供 x_cat")
            if x_cat.size(-1) != self.num_cat_features:
                raise ValueError(
                    f"x_cat 最后一维={x_cat.size(-1)} 与 num_cat_features={self.num_cat_features} 不一致"
                )

            embs = []
            for i, emb_layer in enumerate(self.cat_embeddings):
                idx = x_cat[..., i].long()
                idx = torch.clamp(idx, min=0, max=emb_layer.num_embeddings - 1)
                embs.append(emb_layer(idx))

            x = torch.cat([x_cont] + embs, dim=-1)

        x = self.fuse_linear(x)
        x = self.fuse_dropout(x)
        x = self.layer_norm(x)
        return x

    def _pre_gru_attention_residual(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, fused_features)
        energy = torch.tanh(self.pre_attn_proj(x))
        scores = self.pre_attn_score(energy)             # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)
        context = (x * weights).sum(dim=1, keepdim=True) # (batch, 1, fused_features)
        return x + context

    def forward(self, x_cont: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x_cont: (batch, seq_len, n_cont)
            x_cat: (batch, seq_len, n_cat)

        Returns:
            preds: (batch, num_targets)
        """
        # 1. 连续+类别融合
        x = self._fuse_inputs(x_cont=x_cont, x_cat=x_cat)

        # 1.5 进入 GRU 前增加单头时间注意力 + 残差
        x = self._pre_gru_attention_residual(x)

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
