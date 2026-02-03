"""
深度学习数据集模块

核心功能：
1. 3D 张量构建 (N, L, F) - N样本数, L窗口长度, F特征数
2. 滑动窗口索引映射 (不预展开, 内存友好)
3. cuDF + DLPack 零拷贝加速
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """数据配置"""
    
    # 数据路径
    data_path: Path = Path("/home/menghuanghan/quant_system/data/features/structured/train.parquet")
    
    # 窗口长度 (对应一个月交易日)
    window_size: int = 20
    
    # 特征筛选: 只使用 _zscore 后缀的列
    # 原因: 神经网络对输入尺度极度敏感
    zscore_only: bool = True
    
    # 目标列
    target_col: str = "ret_5d"
    
    # Purging（隔离带）: 防止标签泄露
    # 由于 ret_5d 使用未来 5 天的收益，需要在训练/验证边界预留间隔
    purge_days: int = 5  # 隔离天数 (应 >= 标签预测天数)
    
    # 排除列
    exclude_cols: List[str] = field(default_factory=lambda: [
        "ts_code", "trade_date",
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "label_1d", "label_5d", "label_10d", "label_20d",
        "lag_days", "market",
    ])
    
    # 时间切分 (Purging 后的有效边界)
    # 原始边界: train_end + purge_days = gap 结束
    train_start: str = "2021-01-01"
    train_end: str = "2023-12-25"  # 提前 5 天，留出 2023-12-26~12-31 作为隔离带
    valid_start: str = "2024-01-01"
    valid_end: str = "2024-12-25"  # 提前 5 天
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"
    
    # GPU 加速
    use_gpu: bool = True


class StockDataset(Dataset):
    """
    股票时序数据集
    
    核心思路:
    1. 将数据按 ts_code 和 trade_date 排序
    2. 记录每只股票的 start_index 和 end_index
    3. 在 __getitem__ 中动态切片取出窗口数据
    
    GPU 优化版:
    - 支持将整个特征矩阵预加载到 GPU
    - 切片操作直接在 GPU 上进行，避免 CPU<->GPU 传输瓶颈
    """
    
    def __init__(
        self,
        data: Any,  # numpy array 或 cuDF DataFrame
        indices: np.ndarray,  # 有效样本的索引
        feature_cols: List[str],
        target_col: str,
        window_size: int = 20,
        dates: Optional[np.ndarray] = None,  # 日期数组 (用于按日计算 IC)
        codes: Optional[np.ndarray] = None,  # 股票代码数组 (用于对齐)
        device: str = "cpu",  # 数据存储设备 (cpu / cuda)
    ):
        """
        Args:
            data: 原始数据 (已按 ts_code + trade_date 排序)
            indices: 每个样本对应的 "窗口最后一行" 索引
            feature_cols: 特征列名
            target_col: 目标列名
            window_size: 窗口长度
            dates: 日期数组
            codes: 股票代码数组
            device: 数据存储设备 (预加载到 GPU 可大幅提升训练速度)
        """
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.device = device
        
        # 保存元数据
        self.dates = dates
        self.codes = codes
        
        # 提取特征矩阵和标签
        if hasattr(data, 'to_arrow'):
            # cuDF DataFrame -> numpy -> tensor
            features_np = data[feature_cols].to_arrow().to_pandas().values.astype(np.float32)
            labels_np = data[target_col].to_arrow().to_pandas().values.astype(np.float32)
        else:
            features_np = data[feature_cols].values.astype(np.float32)
            labels_np = data[target_col].values.astype(np.float32)
        
        # 转为 PyTorch Tensor 并放到指定设备
        self.features = torch.from_numpy(features_np).to(device)
        self.labels = torch.from_numpy(labels_np).to(device)
        
        # 关键优化: indices 保留在 CPU (numpy)，避免 __getitem__ 中的 .item() GPU 同步开销
        # 原因: PyTorch DataLoader 每秒调用 __getitem__ 数千次，.item() 会触发 CUDA 同步
        self.indices = indices.astype(np.int64)  # numpy array, 保留在 CPU
        
        # 计算内存/显存占用
        mem_mb = (self.features.numel() * 4 + self.labels.numel() * 4) / 1024 / 1024
        
        logger.info(f"✓ StockDataset 初始化完成:")
        logger.info(f"  样本数: {len(indices):,}")
        logger.info(f"  特征数: {len(feature_cols)}")
        logger.info(f"  窗口长度: {window_size}")
        logger.info(f"  数据设备: {device}")
        logger.info(f"  {'显存' if 'cuda' in device else '内存'}占用: {mem_mb:.1f} MB")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本 (GPU 原生切片，零同步开销)
        
        Args:
            idx: 样本索引
            
        Returns:
            (X, y) 元组
            - X: (window_size, n_features) 张量 (已在 GPU 上)
            - y: 标量张量 (已在 GPU 上)
        """
        # 窗口最后一行的实际索引 (indices 在 CPU，转为 Python int)
        # 关键: 不使用 .item()，避免 GPU<->CPU 同步
        end_idx = int(self.indices[idx])
        start_idx = end_idx - self.window_size + 1
        
        # GPU 上直接切片，零拷贝
        X = self.features[start_idx:end_idx + 1]  # (window_size, n_features)
        y = self.labels[end_idx]
        
        return X, y
    
    def get_dates(self, idx: int) -> Optional[str]:
        """获取样本对应的日期"""
        if self.dates is not None:
            return self.dates[int(self.indices[idx])]
        return None
    
    def get_all_dates(self) -> Optional[np.ndarray]:
        """获取所有样本的日期数组"""
        if self.dates is not None:
            return np.array([self.dates[int(i)] for i in self.indices])
        return None
    
    def get_all_codes(self) -> Optional[np.ndarray]:
        """获取所有样本的股票代码数组"""
        if self.codes is not None:
            return np.array([self.codes[int(i)] for i in self.indices])
        return None


def build_index_map(
    df: Any,
    ts_code_col: str = "ts_code",
    window_size: int = 20
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    构建索引映射
    
    假设数据已按 ts_code + trade_date 排序
    
    Returns:
        valid_indices: 有效样本的索引 (窗口完整)
        stock_bounds: 每只股票的 (start, end) 边界
    """
    if hasattr(df, 'to_arrow'):
        ts_codes = df[ts_code_col].to_arrow().to_pandas().values
    else:
        ts_codes = df[ts_code_col].values
    
    n = len(ts_codes)
    stock_bounds = {}
    
    # 找出每只股票的起止位置
    current_code = ts_codes[0]
    start_idx = 0
    
    for i in range(1, n):
        if ts_codes[i] != current_code:
            stock_bounds[current_code] = (start_idx, i - 1)
            current_code = ts_codes[i]
            start_idx = i
    # 最后一只股票
    stock_bounds[current_code] = (start_idx, n - 1)
    
    logger.info(f"📊 股票数量: {len(stock_bounds)}")
    
    # 收集有效索引 (窗口完整的样本)
    valid_indices = []
    for code, (start, end) in stock_bounds.items():
        # 只有当股票有足够数据形成一个完整窗口时才有效
        # 窗口要求从 start + window_size - 1 开始
        for idx in range(start + window_size - 1, end + 1):
            valid_indices.append(idx)
    
    valid_indices = np.array(valid_indices, dtype=np.int64)
    logger.info(f"📊 有效样本数: {len(valid_indices):,} (窗口完整)")
    
    return valid_indices, stock_bounds


def prepare_data(
    config: DataConfig,
    device: str = "cpu",  # 数据预加载设备
) -> Tuple[StockDataset, StockDataset, StockDataset, List[str]]:
    """
    准备训练/验证/测试数据集
    
    Args:
        config: 数据配置
        device: 数据预加载设备 (cpu / cuda)
            - "cuda": 预加载到 GPU，训练速度最快
            - "cpu": 保留在 CPU，适合大数据或调试
    
    Returns:
        train_dataset, valid_dataset, test_dataset, feature_cols
    """
    logger.info("=" * 60)
    logger.info("📊 准备深度学习数据集")
    logger.info("=" * 60)
    
    # 1. 加载数据
    if config.use_gpu:
        try:
            import cudf
            df = cudf.read_parquet(str(config.data_path))
            logger.info(f"✓ cuDF 加载数据: {len(df):,} 行")
        except ImportError:
            import pandas as pd
            df = pd.read_parquet(str(config.data_path))
            logger.info(f"✓ Pandas 加载数据: {len(df):,} 行")
    else:
        import pandas as pd
        df = pd.read_parquet(str(config.data_path))
        logger.info(f"✓ Pandas 加载数据: {len(df):,} 行")
    
    # 2. 筛选特征列
    all_cols = df.columns.tolist()
    
    if config.zscore_only:
        # 只使用 _zscore 后缀的列
        feature_cols = [c for c in all_cols if c.endswith('_zscore')]
        logger.info(f"✓ Z-Score 特征筛选: {len(feature_cols)} 个特征")
    else:
        # 排除指定列
        exclude_set = set(config.exclude_cols)
        feature_cols = [c for c in all_cols if c not in exclude_set]
    
    logger.info(f"  特征列: {feature_cols[:5]}... (共 {len(feature_cols)} 个)")
    
    # 3. 按 ts_code + trade_date 排序
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    logger.info("✓ 数据已按 ts_code + trade_date 排序")
    
    # 4. 时间切分
    if hasattr(df, 'to_arrow'):
        dates = df['trade_date'].to_arrow().to_pandas().values
        codes = df['ts_code'].to_arrow().to_pandas().values
    else:
        dates = df['trade_date'].values
        codes = df['ts_code'].values
    
    # 转换为字符串比较
    dates_str = np.array([str(d)[:10] for d in dates])
    codes_str = np.array([str(c) for c in codes])
    
    train_mask = (dates_str >= config.train_start) & (dates_str <= config.train_end)
    valid_mask = (dates_str >= config.valid_start) & (dates_str <= config.valid_end)
    test_mask = (dates_str >= config.test_start) & (dates_str <= config.test_end)
    
    logger.info(f"📅 时间切分 (Purging={config.purge_days}天):")
    logger.info(f"  训练集: {config.train_start} ~ {config.train_end}")
    logger.info(f"  隔离带: {config.train_end} + {config.purge_days}天")
    logger.info(f"  验证集: {config.valid_start} ~ {config.valid_end}")
    logger.info(f"  隔离带: {config.valid_end} + {config.purge_days}天")
    logger.info(f"  测试集: {config.test_start} ~ {config.test_end}")
    
    # 5. 构建索引映射
    logger.info("📊 构建索引映射...")
    all_indices, stock_bounds = build_index_map(df, window_size=config.window_size)
    
    # 6. 按时间过滤有效索引
    train_indices = all_indices[train_mask[all_indices]]
    valid_indices = all_indices[valid_mask[all_indices]]
    test_indices = all_indices[test_mask[all_indices]]
    
    logger.info(f"📊 各数据集样本数:")
    logger.info(f"  训练集: {len(train_indices):,}")
    logger.info(f"  验证集: {len(valid_indices):,}")
    logger.info(f"  测试集: {len(test_indices):,}")
    
    # 7. 创建数据集 (预加载到指定设备)
    logger.info(f"📊 预加载数据到设备: {device}")
    
    train_dataset = StockDataset(
        data=df,
        indices=train_indices,
        feature_cols=feature_cols,
        target_col=config.target_col,
        window_size=config.window_size,
        dates=dates_str,
        codes=codes_str,
        device=device,
    )
    
    valid_dataset = StockDataset(
        data=df,
        indices=valid_indices,
        feature_cols=feature_cols,
        target_col=config.target_col,
        window_size=config.window_size,
        dates=dates_str,
        codes=codes_str,
        device=device,
    )
    
    test_dataset = StockDataset(
        data=df,
        indices=test_indices,
        feature_cols=feature_cols,
        target_col=config.target_col,
        window_size=config.window_size,
        dates=dates_str,
        codes=codes_str,
        device=device,
    )
    
    logger.info("✅ 数据集准备完成")
    
    return train_dataset, valid_dataset, test_dataset, feature_cols


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    config = DataConfig()
    train_ds, valid_ds, test_ds, feature_cols = prepare_data(config)
    
    # 测试取样
    X, y = train_ds[0]
    print(f"X shape: {X.shape}")  # (20, n_features)
    print(f"y: {y}")
