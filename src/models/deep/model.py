"""
GRU 模型架构

设计思路:
1. 输入: (N, L, F) - N批次, L窗口长度, F特征数
2. GRU: 2 层, Hidden=64/128, Dropout=0.2
3. 提取: 只取最后时间步的隐状态
4. 输出: MLP 头 -> 单值预测
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    
    # 输入维度 (特征数, 运行时确定)
    input_dim: int = 30
    
    # GRU 配置
    hidden_dim: int = 64  # 隐层大小 (64 或 128, 太大易过拟合)
    num_layers: int = 2   # GRU 层数
    dropout: float = 0.2  # Dropout 比例
    
    # MLP 头配置
    mlp_hidden: int = 32
    mlp_dropout: float = 0.2
    
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


def create_model(
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    model_type: str = "gru",
) -> nn.Module:
    """
    工厂函数: 创建模型
    
    Args:
        input_dim: 输入特征数
        hidden_dim: 隐层大小
        num_layers: RNN 层数
        dropout: Dropout 比例
        model_type: 模型类型 (gru / attention / lstm_skip)
        
    Returns:
        模型实例
    """
    config = ModelConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    
    if model_type == "attention":
        return AttentionGRUModel(config)
    elif model_type == "lstm_skip":
        return LSTMSkipModel(config)
    else:
        return GRUModel(config)


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
