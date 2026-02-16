#!/usr/bin/env python
"""
GRU 模型训练脚本（防过拟合优化版）

优化措施：
1. 只使用 LightGBM Top 50 强特征
2. 增大 Dropout (0.5) 和 Weight Decay (1e-3)
3. 降低模型容量 (hidden_dim=32, num_layers=1)
4. 可选：使用行业 Embedding 替代 industry_idx 数值

使用方法:
    python scripts/run_gru_train.py
    python scripts/run_gru_train.py --use-feature-selection
    python scripts/run_gru_train.py --hidden-dim 32 --dropout 0.5 --weight-decay 1e-3
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.deep.config import GRUDataConfig
from src.models.deep.dataset import prepare_data
from src.models.deep.model import create_model
from src.models.deep.train import GRUTrainer, TrainConfig, create_dataloader
from src.models.deep.loss import compute_ic, compute_rank_ic


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GRU 模型训练（防过拟合优化版）")
    
    # 基础配置
    parser.add_argument(
        "--target",
        type=str,
        default="excess_ret_5d",
        help="目标列 (default: excess_ret_5d)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch Size (default: 2048)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数 (default: 100)",
    )
    
    # 模型配置（防过拟合）
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="GRU 隐层大小 (default: 32, 减小防过拟合)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="GRU 层数 (default: 1, 简化模型)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout 比例 (default: 0.5, 增大防过拟合)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="权重衰减 (default: 1e-3, 增大防过拟合)",
    )
    
    # 特征筛选
    parser.add_argument(
        "--use-feature-selection",
        action="store_true",
        help="启用特征筛选（只使用 LightGBM Top 50 特征）",
    )
    parser.add_argument(
        "--feature-json",
        type=str,
        default=str(PROJECT_ROOT / "models" / "lgbm" / "top50_features.json"),
        help="特征列表 JSON 文件路径",
    )
    
    # 其他配置
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="时间窗口大小 (default: 20)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="数据路径 (默认 data/features/structured/train_gru.parquet)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="使用 CPU 模式",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("🤖 GRU 模型训练（防过拟合优化版）")
    logger.info("=" * 60)
    
    # 设备
    if args.cpu:
        device = "cpu"
        data_device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_device = device  # 预加载到 GPU
    
    logger.info(f"设备: {device}")
    
    # 创建数据配置
    data_config = GRUDataConfig()
    data_config.target_col = args.target
    data_config.window_size = args.window_size
    
    if args.data_path:
        data_config.data_path = Path(args.data_path)
    
    # 启用特征筛选
    if args.use_feature_selection:
        data_config.use_feature_selection = True
        data_config.feature_selection_json = Path(args.feature_json)
        logger.info(f"✅ 特征筛选已启用: {args.feature_json}")
    
    logger.info(f"目标列: {args.target}")
    logger.info(f"窗口大小: {args.window_size}")
    logger.info(f"数据路径: {data_config.data_path}")
    
    # 准备数据
    train_dataset, valid_dataset, test_dataset, feature_cols = prepare_data(
        data_config,
        device=data_device,
    )
    
    logger.info(f"特征数量: {len(feature_cols)}")
    if args.use_feature_selection:
        logger.info(f"特征列表: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"特征列表: {feature_cols}")
    logger.info(f"训练集: {len(train_dataset):,} 样本")
    logger.info(f"验证集: {len(valid_dataset):,} 样本")
    logger.info(f"测试集: {len(test_dataset):,} 样本")
    
    # 创建模型（使用防过拟合配置）
    model = create_model(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    
    # 训练配置（增强正则化）
    train_config = TrainConfig()
    train_config.epochs = args.epochs
    train_config.batch_size = args.batch_size
    train_config.weight_decay = args.weight_decay  # 增大权重衰减
    train_config.model_name = f"gru_{args.target}"
    
    logger.info("=" * 60)
    logger.info("📋 防过拟合配置")
    logger.info("=" * 60)
    logger.info(f"  隐层大小: {args.hidden_dim} (原 128)")
    logger.info(f"  GRU 层数: {args.num_layers} (原 2)")
    logger.info(f"  Dropout: {args.dropout} (原 0.3)")
    logger.info(f"  权重衰减: {args.weight_decay} (原 1e-4)")
    logger.info(f"  特征筛选: {'启用' if args.use_feature_selection else '禁用'}")
    
    # 创建 DataLoader
    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = create_dataloader(valid_dataset, args.batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, args.batch_size, shuffle=False)
    
    # 训练
    trainer = GRUTrainer(model, train_config, device=device)
    history = trainer.train(train_loader, valid_loader)
    
    # 测试集评估
    logger.info("=" * 60)
    logger.info("📊 测试集评估")
    logger.info("=" * 60)
    
    test_predictions = trainer.predict(test_loader)
    
    # 获取测试集标签
    test_labels = []
    for _, y in test_loader:
        test_labels.append(y.cpu().numpy())
    import numpy as np
    test_labels = np.concatenate(test_labels)
    
    # 计算 IC
    test_ic = compute_ic(
        torch.from_numpy(test_predictions),
        torch.from_numpy(test_labels)
    )
    test_rank_ic = compute_rank_ic(
        torch.from_numpy(test_predictions),
        torch.from_numpy(test_labels)
    )
    
    logger.info(f"测试集 IC: {test_ic:.4f}")
    logger.info(f"测试集 RankIC: {test_rank_ic:.4f}")
    
    # 保存最终模型
    trainer.save_model(f"final_{args.target}.pt")
    
    # 训练总结
    import numpy as np
    logger.info("=" * 60)
    logger.info("📊 训练总结")
    logger.info("=" * 60)
    logger.info(f"  最佳验证 RankIC: {trainer.best_valid_ic:.4f}")
    
    train_ics = history.get('train_rank_ic', [])
    if train_ics:
        best_train_ic = max(train_ics)
        logger.info(f"  最大训练 RankIC: {best_train_ic:.4f}")
        logger.info(f"  过拟合差距: {best_train_ic - trainer.best_valid_ic:.4f}")
    
    logger.info(f"  测试 RankIC: {test_rank_ic:.4f}")
    
    # 过拟合诊断
    if train_ics and best_train_ic > 0.3:
        logger.warning("⚠️ 训练 IC > 0.3，可能存在严重过拟合！")
    elif train_ics and best_train_ic - trainer.best_valid_ic > 0.1:
        logger.warning("⚠️ 训练-验证差距 > 0.1，存在过拟合风险")
    else:
        logger.info("✅ 过拟合程度可接受")
    
    logger.info("=" * 60)
    logger.info("✅ 训练完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
