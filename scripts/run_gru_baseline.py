#!/usr/bin/env python
"""
GRU 基准模型训练脚本

使用方法:
    python scripts/run_gru_baseline.py [OPTIONS]

示例:
    # 默认配置训练
    python scripts/run_gru_baseline.py
    
    # 调整 batch size
    python scripts/run_gru_baseline.py --batch-size 4096
    
    # 干跑 (只加载数据)
    python scripts/run_gru_baseline.py --dry-run
"""

import argparse
import logging
import sys
import gc
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GRU 基准模型训练")
    
    # 模型参数
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="GRU 隐层大小 (默认: 64)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="GRU 层数 (默认: 2)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout 比例 (默认: 0.2)")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Batch Size (默认: 2048)")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="学习率 (默认: 1e-3)")
    parser.add_argument("--patience", type=int, default=10,
                        help="早停耐心值 (默认: 10)")
    
    # 数据参数
    parser.add_argument("--window-size", type=int, default=20,
                        help="窗口长度 (默认: 20)")
    parser.add_argument("--target", type=str, default="ret_5d",
                        help="目标列 (默认: ret_5d)")
    
    # 其他
    parser.add_argument("--no-amp", action="store_true",
                        help="禁用混合精度")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用 CPU")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader 工作进程 (默认: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑: 只加载数据, 不训练")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("🚀 GRU 基准模型训练")
    logger.info(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info(f"  📋 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"  📋 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  📋 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"  📋 Batch Size: {args.batch_size}")
    logger.info(f"  📋 Hidden Dim: {args.hidden_dim}")
    logger.info(f"  📋 Window Size: {args.window_size}")
    logger.info(f"  📋 目标列: {args.target}")
    logger.info("")
    
    # 导入模块 (延迟导入以便快速显示帮助)
    from src.models.deep.dataset import DataConfig, prepare_data
    from src.models.deep.model import create_model
    from src.models.deep.train import TrainConfig, GRUTrainer, create_dataloader
    from src.models.deep.loss import compute_rank_ic
    
    # 确定设备 (提前确定以支持 GPU 数据预加载)
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"📍 数据预加载到设备: {device}")
    
    # Step 1: 准备数据 (直接预加载到 GPU)
    data_config = DataConfig(
        window_size=args.window_size,
        target_col=args.target,
        zscore_only=True,
    )
    
    train_ds, valid_ds, test_ds, feature_cols = prepare_data(data_config, device=device)
    
    if args.dry_run:
        logger.info("")
        logger.info("🏁 Dry run 完成，跳过训练")
        
        # 测试取样
        X, y = train_ds[0]
        logger.info(f"  样本 X 形状: {X.shape}")
        logger.info(f"  样本 y: {y.item():.4f}")
        
        # 释放内存
        del train_ds, valid_ds, test_ds
        gc.collect()
        return
    
    # 创建 DataLoader (GPU 数据时自动使用 num_workers=0)
    train_loader = create_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    valid_loader = create_dataloader(
        valid_ds, batch_size=args.batch_size * 2, shuffle=False
    )
    test_loader = create_dataloader(
        test_ds, batch_size=args.batch_size * 2, shuffle=False
    )
    
    logger.info(f"📊 DataLoader 创建完成:")
    logger.info(f"  训练集: {len(train_loader)} batches")
    logger.info(f"  验证集: {len(valid_loader)} batches")
    logger.info(f"  测试集: {len(test_loader)} batches")
    
    # Step 2: 创建模型
    logger.info("")
    input_dim = len(feature_cols)
    model = create_model(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    
    # Step 3: 创建训练器
    use_amp = not args.no_amp and device == "cuda"
    
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        num_workers=0 if device == "cuda" else args.num_workers,  # GPU 数据不需要多进程
        use_amp=use_amp,
    )
    
    trainer = GRUTrainer(model, train_config, device=device)
    
    # Step 4: 训练
    logger.info("")
    history = trainer.train(train_loader, valid_loader)
    
    # Step 5: 测试集评估
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 测试集评估 (Out-of-Sample)")
    logger.info("=" * 60)
    
    # 加载最佳模型
    trainer.load_model("best_model.pt")
    
    # 预测
    test_preds = trainer.predict(test_loader)
    test_labels = np.array([test_ds[i][1].item() for i in range(len(test_ds))])
    
    # 计算指标
    test_ic = np.corrcoef(test_preds, test_labels)[0, 1]
    test_rank_ic = compute_rank_ic(
        torch.from_numpy(test_preds),
        torch.from_numpy(test_labels),
    )
    
    logger.info(f"  测试集 IC: {test_ic:.4f}")
    logger.info(f"  测试集 RankIC: {test_rank_ic:.4f}")
    
    # 计算 ICIR (需要按日聚合)
    # 简化版: 整体计算
    logger.info("")
    
    # 保存报告
    report_path = PROJECT_ROOT / "reports" / f"gru_{args.target}_analysis.md"
    with open(report_path, 'w') as f:
        f.write("# GRU 模型分析报告\n\n")
        f.write("## 1. 模型配置\n\n")
        f.write(f"| 参数 | 值 |\n")
        f.write(f"|------|-----|\n")
        f.write(f"| 输入维度 | {input_dim} |\n")
        f.write(f"| 窗口长度 | {args.window_size} |\n")
        f.write(f"| 隐层大小 | {args.hidden_dim} |\n")
        f.write(f"| GRU 层数 | {args.num_layers} |\n")
        f.write(f"| Dropout | {args.dropout} |\n")
        f.write(f"| Batch Size | {args.batch_size} |\n")
        f.write(f"| 学习率 | {args.learning_rate} |\n")
        f.write("\n## 2. 训练结果\n\n")
        f.write(f"| 指标 | 值 |\n")
        f.write(f"|------|-----|\n")
        f.write(f"| 最佳 Epoch | {history['valid_rank_ic'].index(max(history['valid_rank_ic'])) + 1} |\n")
        f.write(f"| 验证集最佳 RankIC | {trainer.best_valid_ic:.4f} |\n")
        f.write(f"| 测试集 IC | {test_ic:.4f} |\n")
        f.write(f"| 测试集 RankIC | {test_rank_ic:.4f} |\n")
    
    logger.info(f"  📝 报告已保存: {report_path}")
    
    # 总结
    logger.info("")
    logger.info("=" * 60)
    logger.info("🎉 训练完成!")
    logger.info(f"   模型: {train_config.save_dir / 'best_model.pt'}")
    logger.info(f"   报告: {report_path}")
    logger.info("=" * 60)
    
    # 释放内存
    del trainer, model, train_loader, valid_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
