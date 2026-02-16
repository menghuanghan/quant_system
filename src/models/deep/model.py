"""
GRU 模型架构

设计思路:
1. 输入: (N, L, F) - N批次, L窗口长度, F特征数
2. GRU: 2 层, Hidden=64/128, Dropout=0.2
3. 提取: 只取最后时间步的隐状态
4. 输出: MLP 头 -> 单值预测

改造说明（2026.02）:
- 支持动态 input_dim（从数据自动获取）
- 增加隐层大小适应更多特征
- 添加新配置支持
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from .config import GRUModelConfig

logger = logging.getLogger(__name__)


# ModelConfig 保留兼容性别名
@dataclass
class ModelConfig:
    """模型配置（兼容旧接口）"""
    
    # 输入维度 (特征数, 运行时确定)
    input_dim: int = 256
    
    # GRU 配置
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    
    # MLP 头配置
    mlp_hidden: int = 64
    mlp_dropout: float = 0.3
    
    # 是否双向
    bidirectional: bool = False


class GRUModel(nn.Module):
    """
    GRU 时序预测模型
    
    架构:
        Input (N, L, F)
            ↓
        GRU Layers (2层, dropout=0.2)
            ↓
        取最后时间步 (N, hidden_dim)
            ↓
        MLP Head: Linear -> ReLU -> Dropout -> Linear
            ↓
        Output (N, 1)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # GRU 核心层
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,  # 输入格式 (N, L, F)
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )
        
        # 双向时隐层翻倍
        gru_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # MLP 预测头
        self.head = nn.Sequential(
            nn.Linear(gru_output_dim, config.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(config.mlp_dropout),
            nn.Linear(config.mlp_hidden, 1),
        )
        
        # 初始化权重
        self._init_weights()
        
        # 打印模型信息
        self._log_model_info()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _log_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"🤖 GRU 模型初始化:")
        logger.info(f"  输入维度: {self.config.input_dim}")
        logger.info(f"  隐层大小: {self.config.hidden_dim}")
        logger.info(f"  GRU 层数: {self.config.num_layers}")
        logger.info(f"  Dropout: {self.config.dropout}")
        logger.info(f"  双向: {self.config.bidirectional}")
        logger.info(f"  总参数量: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
    
    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (N, L, F) 输入张量
            h0: 初始隐状态 (可选)
            
        Returns:
            (N, 1) 预测值
        """
        # GRU 前向
        # output: (N, L, hidden_dim)
        # hidden: (num_layers, N, hidden_dim)
        output, hidden = self.gru(x, h0)
        
        # 取最后一个时间步的输出
        # (N, hidden_dim)
        last_hidden = output[:, -1, :]
        
        # MLP 头预测
        # (N, 1)
        pred = self.head(last_hidden)
        
        return pred.squeeze(-1)  # (N,)


class AttentionGRUModel(nn.Module):
    """
    带注意力机制的 GRU 模型 (可选)
    
    在 GRU 输出后加入自注意力，让模型学习哪些时间步更重要
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # GRU
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )
        
        gru_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # 时间注意力
        self.attention = nn.Sequential(
            nn.Linear(gru_output_dim, gru_output_dim // 2),
            nn.Tanh(),
            nn.Linear(gru_output_dim // 2, 1),
        )
        
        # MLP 头
        self.head = nn.Sequential(
            nn.Linear(gru_output_dim, config.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(config.mlp_dropout),
            nn.Linear(config.mlp_hidden, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for module in [*self.attention, *self.head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        # GRU
        output, hidden = self.gru(x, h0)  # output: (N, L, hidden)
        
        # 注意力权重
        attn_scores = self.attention(output)  # (N, L, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (N, L, 1)
        
        # 加权求和
        context = (output * attn_weights).sum(dim=1)  # (N, hidden)
        
        # 预测
        pred = self.head(context)
        
        return pred.squeeze(-1)


class LSTMSkipModel(nn.Module):
    """
    LSTM + Skip Connection 模型
    
    架构:
        Input (N, L, F)
            ↓
        Input Projection: F -> hidden_dim
            ↓
        LSTM Layers (2层, dropout)
            ↓
        取最后时间步 + Skip Connection (输入最后时间步投影)
            ↓
        MLP Head
            ↓
        Output (N, 1)
    
    Skip Connection 作用：
    1. 保留原始特征信息，避免 LSTM 遗忘
    2. 提供梯度短路，加速收敛
    3. 让模型同时学习时序模式和静态特征
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 输入投影: 将特征维度映射到 hidden_dim
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # LSTM 核心层
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,  # 投影后的维度
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )
        
        # 双向时隐层翻倍
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # Skip connection 后的维度: LSTM输出 + 投影输入
        skip_dim = lstm_output_dim + config.hidden_dim
        
        # Layer Norm 用于稳定训练
        self.layer_norm = nn.LayerNorm(skip_dim)
        
        # MLP 预测头
        self.head = nn.Sequential(
            nn.Linear(skip_dim, config.mlp_hidden),
            nn.GELU(),  # GELU 比 ReLU 更平滑
            nn.Dropout(config.mlp_dropout),
            nn.Linear(config.mlp_hidden, 1),
        )
        
        # 初始化权重
        self._init_weights()
        
        # 打印模型信息
        self._log_model_info()
    
    def _init_weights(self):
        """初始化权重"""
        # 输入投影
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        # LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # 设置遗忘门偏置为 1，帮助 LSTM 记住长期信息
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # MLP 头
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _log_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"🤖 LSTM+Skip 模型初始化:")
        logger.info(f"  输入维度: {self.config.input_dim}")
        logger.info(f"  隐层大小: {self.config.hidden_dim}")
        logger.info(f"  LSTM 层数: {self.config.num_layers}")
        logger.info(f"  Dropout: {self.config.dropout}")
        logger.info(f"  双向: {self.config.bidirectional}")
        logger.info(f"  总参数量: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (N, L, F) 输入张量
            
        Returns:
            (N,) 预测值
        """
        # Step 1: 输入投影
        # (N, L, F) -> (N, L, hidden_dim)
        proj = self.input_proj(x)
        
        # Step 2: LSTM 前向
        # output: (N, L, hidden_dim)
        lstm_out, _ = self.lstm(proj)
        
        # Step 3: 取最后时间步
        # (N, hidden_dim)
        lstm_last = lstm_out[:, -1, :]
        proj_last = proj[:, -1, :]
        
        # Step 4: Skip Connection - 拼接 LSTM 输出与投影输入
        # (N, hidden_dim + hidden_dim)
        combined = torch.cat([lstm_last, proj_last], dim=-1)
        
        # Step 5: Layer Norm
        combined = self.layer_norm(combined)
        
        # Step 6: MLP 头预测
        pred = self.head(combined)
        
        return pred.squeeze(-1)  # (N,)


class GRUWithIndustryEmbedding(nn.Module):
    """
    带行业 Embedding 的 GRU 模型
    
    解决问题：
    - industry_idx 是类别变量（1-31），直接作为数值输入会让模型误以为行业 10 比行业 1 "大"
    - 使用 Embedding 层将行业 ID 映射到连续向量空间
    
    架构：
        数值特征 (N, L, F)           行业 ID (N, L, 1)
               ↓                            ↓
            GRU Layers           nn.Embedding (num_industries, embed_dim)
               ↓                            ↓
        取最后时间步 (N, hidden)    取最后时间步 (N, embed_dim)
               ↓                            ↓
               └─────────── Concat ────────────┘
                              ↓
                      MLP Head (hidden + embed_dim -> 1)
                              ↓
                         Output (N, 1)
    """
    
    def __init__(
        self,
        config: ModelConfig,
        num_industries: int = 32,
        industry_embed_dim: int = 8,
    ):
        super().__init__()
        self.config = config
        self.num_industries = num_industries
        self.industry_embed_dim = industry_embed_dim
        
        # 行业 Embedding 层
        self.industry_embedding = nn.Embedding(
            num_embeddings=num_industries + 1,  # +1 for unknown/padding
            embedding_dim=industry_embed_dim,
            padding_idx=0,  # 0 作为 padding
        )
        
        # GRU 核心层（输入维度不含 industry_idx）
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )
        
        gru_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # MLP 预测头（输入维度 = GRU 输出 + 行业 Embedding）
        combined_dim = gru_output_dim + industry_embed_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, config.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(config.mlp_dropout),
            nn.Linear(config.mlp_hidden, 1),
        )
        
        self._init_weights()
        self._log_model_info()
    
    def _init_weights(self):
        """初始化权重"""
        # GRU 初始化
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Embedding 初始化
        nn.init.normal_(self.industry_embedding.weight, mean=0, std=0.1)
        
        # MLP 头初始化
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _log_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"🤖 GRU + Industry Embedding 模型初始化:")
        logger.info(f"  输入维度: {self.config.input_dim}")
        logger.info(f"  隐层大小: {self.config.hidden_dim}")
        logger.info(f"  GRU 层数: {self.config.num_layers}")
        logger.info(f"  Dropout: {self.config.dropout}")
        logger.info(f"  行业数量: {self.num_industries}")
        logger.info(f"  行业 Embedding 维度: {self.industry_embed_dim}")
        logger.info(f"  总参数量: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        industry_ids: Optional[torch.Tensor] = None,
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (N, L, F) 数值特征输入
            industry_ids: (N, L) 或 (N,) 行业 ID（整数 0-31）
            h0: 初始隐状态（可选）
            
        Returns:
            (N,) 预测值
        """
        # GRU 前向
        output, hidden = self.gru(x, h0)  # output: (N, L, hidden)
        last_gru_hidden = output[:, -1, :]  # (N, hidden)
        
        # 行业 Embedding
        if industry_ids is not None:
            if industry_ids.dim() == 2:
                # (N, L) -> 取最后一个时间步
                industry_ids = industry_ids[:, -1]  # (N,)
            industry_embed = self.industry_embedding(industry_ids.long())  # (N, embed_dim)
            
            # 拼接 GRU 输出和行业 Embedding
            combined = torch.cat([last_gru_hidden, industry_embed], dim=-1)  # (N, hidden + embed_dim)
        else:
            # 如果没有行业 ID，使用零向量
            batch_size = x.size(0)
            zero_embed = torch.zeros(batch_size, self.industry_embed_dim, device=x.device)
            combined = torch.cat([last_gru_hidden, zero_embed], dim=-1)
        
        # MLP 预测
        pred = self.head(combined)
        
        return pred.squeeze(-1)  # (N,)


class GRUWithMultiEmbedding(nn.Module):
    """
    支持多个类别特征 Embedding 的 GRU 模型
    
    架构：
        数值特征 (N, L, F)      类别特征 {market, sw_l1_idx, ...}
               ↓                         ↓
            GRU Layers         Multiple Embedding Layers
               ↓                         ↓
        取最后时间步 (N, H)    取最后时间步 + Concat (N, sum(embed_dims))
               ↓                         ↓
               └─────────── Concat ────────────┘
                              ↓
                      MLP Head -> Output (N, 1)
    
    Parameters:
        config: 模型配置 (GRUModelConfig)
        embedding_config: 字典 {feature_name: {'num_embeddings': int, 'embed_dim': int}}
    """
    
    def __init__(
        self,
        config: ModelConfig,
        embedding_config: dict = None,
    ):
        super().__init__()
        self.config = config
        
        # 默认 Embedding 配置
        if embedding_config is None:
            embedding_config = {
                'sw_l1_idx': {'num_embeddings': 34, 'embed_dim': 8},
            }
        self.embedding_config = embedding_config
        
        # 创建多个 Embedding 层
        self.embeddings = nn.ModuleDict()
        self.total_embed_dim = 0
        
        for name, cfg in embedding_config.items():
            num_emb = cfg.get('num_embeddings', 32)
            emb_dim = cfg.get('embed_dim', 8)
            self.embeddings[name] = nn.Embedding(
                num_embeddings=num_emb + 1,  # +1 for unknown/padding
                embedding_dim=emb_dim,
                padding_idx=0,
            )
            self.total_embed_dim += emb_dim
        
        # GRU 核心层
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )
        
        gru_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # MLP 预测头（输入维度 = GRU 输出 + 所有 Embedding）
        combined_dim = gru_output_dim + self.total_embed_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, config.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(config.mlp_dropout),
            nn.Linear(config.mlp_hidden, 1),
        )
        
        self._init_weights()
        self._log_model_info()
    
    def _init_weights(self):
        """初始化权重"""
        # GRU 初始化
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Embedding 初始化
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0, std=0.1)
        
        # MLP 头初始化
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _log_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"🤖 GRU + Multi-Embedding 模型初始化:")
        logger.info(f"  输入维度: {self.config.input_dim}")
        logger.info(f"  隐层大小: {self.config.hidden_dim}")
        logger.info(f"  GRU 层数: {self.config.num_layers}")
        logger.info(f"  Dropout: {self.config.dropout}")
        logger.info(f"  Embedding 特征: {list(self.embedding_config.keys())}")
        logger.info(f"  Embedding 总维度: {self.total_embed_dim}")
        logger.info(f"  总参数量: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        cat_features: Optional[Dict[str, torch.Tensor]] = None,
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (N, L, F) 数值特征输入
            cat_features: 类别特征字典 {feature_name: (N, L) tensor}
            h0: 初始隐状态（可选）
            
        Returns:
            (N,) 预测值
        """
        batch_size = x.size(0)
        
        # GRU 前向
        output, hidden = self.gru(x, h0)  # output: (N, L, hidden)
        last_gru_hidden = output[:, -1, :]  # (N, hidden)
        
        # 处理所有 Embedding
        embed_list = [last_gru_hidden]
        
        for name, emb_layer in self.embeddings.items():
            if cat_features is not None and name in cat_features:
                cat_ids = cat_features[name]
                if cat_ids.dim() == 2:
                    # (N, L) -> 取最后一个时间步
                    cat_ids = cat_ids[:, -1]  # (N,)
                # 确保索引为正整数
                cat_ids = cat_ids.clamp(min=0).long()
                embed = emb_layer(cat_ids)  # (N, embed_dim)
            else:
                # 使用零向量
                embed_dim = emb_layer.embedding_dim
                embed = torch.zeros(batch_size, embed_dim, device=x.device)
            embed_list.append(embed)
        
        # 拼接所有表示
        combined = torch.cat(embed_list, dim=-1)  # (N, hidden + total_embed_dim)
        
        # MLP 预测
        pred = self.head(combined)
        
        return pred.squeeze(-1)  # (N,)


def create_model(
    input_dim: int = None,
    config: Union[ModelConfig, GRUModelConfig] = None,
    hidden_dim: int = 32,
    num_layers: int = 1,
    dropout: float = 0.5,
    model_type: str = "gru",
    num_industries: int = 32,
    industry_embed_dim: int = 8,
    embedding_config: dict = None,
) -> nn.Module:
    """
    工厂函数: 创建模型
    
    Args:
        input_dim: 输入特征数（如果 config 未提供则必须指定）
        config: 模型配置对象（可选，优先级高于其他参数）
        hidden_dim: 隐层大小
        num_layers: RNN 层数
        dropout: Dropout 比例
        model_type: 模型类型 (gru / attention / lstm_skip / industry_embedding / multi_embedding)
        num_industries: 行业数量（仅 industry_embedding 模型）
        industry_embed_dim: 行业 Embedding 维度（仅 industry_embedding 模型）
        embedding_config: Embedding 配置（仅 multi_embedding 模型）
        
    Returns:
        模型实例
    """
    if config is not None:
        # 使用配置对象
        if isinstance(config, GRUModelConfig):
            model_config = ModelConfig(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                mlp_hidden=config.mlp_hidden,
                mlp_dropout=config.mlp_dropout,
                bidirectional=config.bidirectional,
            )
            # 检查是否使用新的 multi_embedding
            if hasattr(config, 'use_embedding') and config.use_embedding:
                model_type = "multi_embedding"
                embedding_config = getattr(config, 'embedding_config', None)
                # 只保留实际使用的 embedding 特征
                embedding_features = getattr(config, 'embedding_features', [])
                if embedding_config and embedding_features:
                    embedding_config = {k: v for k, v in embedding_config.items() if k in embedding_features}
            # 兼容旧的 industry_embedding 配置
            elif hasattr(config, 'use_industry_embedding') and config.use_industry_embedding:
                model_type = "industry_embedding"
                num_industries = getattr(config, 'num_industries', 32)
                industry_embed_dim = getattr(config, 'industry_embed_dim', 8)
        else:
            model_config = config
    else:
        # 使用参数构建配置
        if input_dim is None:
            raise ValueError("input_dim 必须指定（如果未提供 config）")
        model_config = ModelConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    
    logger.info(f"创建模型: type={model_type}, input_dim={model_config.input_dim}, hidden_dim={model_config.hidden_dim}")
    
    if model_type == "attention":
        return AttentionGRUModel(model_config)
    elif model_type == "lstm_skip":
        return LSTMSkipModel(model_config)
    elif model_type == "industry_embedding":
        return GRUWithIndustryEmbedding(
            config=model_config,
            num_industries=num_industries,
            industry_embed_dim=industry_embed_dim,
        )
    elif model_type == "multi_embedding":
        return GRUWithMultiEmbedding(
            config=model_config,
            embedding_config=embedding_config,
        )
    else:
        return GRUModel(model_config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型
    model = create_model(input_dim=30, hidden_dim=64)
    
    # 测试前向传播
    x = torch.randn(32, 20, 30)  # (batch, seq_len, features)
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")  # (32,)
