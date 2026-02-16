#!/usr/bin/env python
"""
LightGBM 模型训练脚本

使用改造后的模型层训练 LightGBM 模型。
支持全域数据 train_lgb.parquet (310 列特征)。

使用方法:
    python scripts/train_lgb.py
    python scripts/train_lgb.py --target excess_ret_5d
    python scripts/train_lgb.py --cpu
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.LBGM.config import LGBMConfig, get_feature_columns, CATEGORICAL_FEATURES
from src.models.LBGM.data_loader import DataLoader
from src.models.LBGM.trainer import LGBMTrainer


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LightGBM 模型训练")
    
    parser.add_argument(
        "--target",
        type=str,
        default="excess_ret_5d",
        help="目标列 (default: excess_ret_5d)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="使用 CPU 模式 (默认 GPU)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="数据路径 (默认 data/features/structured/train_lgb.parquet)",
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
    logger.info("🌲 LightGBM 模型训练")
    logger.info("=" * 60)
    
    # 创建配置
    if args.cpu:
        config = LGBMConfig.cpu_mode()
        logger.info("使用 CPU 模式")
    else:
        config = LGBMConfig.default()
        logger.info("使用 GPU 模式")
    
    # 设置目标列
    config.data.target_col = args.target
    config.train.model_name = f"lgbm_{args.target}"
    logger.info(f"目标列: {args.target}")
    
    # 设置数据路径
    if args.data_path:
        config.data.data_path = Path(args.data_path)
    logger.info(f"数据路径: {config.data.data_path}")
    
    # 加载数据
    loader = DataLoader(config.data, use_gpu=not args.cpu)
    loader.load()
    
    # 切分数据
    X_train, y_train, X_valid, y_valid, X_test, y_test = loader.split()
    feature_names = loader.get_feature_names()
    
    logger.info(f"特征数量: {len(feature_names)}")
    logger.info(f"训练集: {X_train.shape}")
    logger.info(f"验证集: {X_valid.shape}")
    logger.info(f"测试集: {X_test.shape}")
    
    # 训练模型
    trainer = LGBMTrainer(config)
    model = trainer.train(
        X_train, y_train,
        X_valid, y_valid,
        feature_names=feature_names,
        categorical_features=config.data.categorical_features,
    )
    
    # 测试集预测
    logger.info("=" * 60)
    logger.info("📊 测试集评估")
    logger.info("=" * 60)
    
    test_pred = trainer.predict(X_test)
    
    # 计算测试集 IC
    from src.models.LBGM.metrics import pearson_corr, spearman_corr
    test_ic = pearson_corr(test_pred, y_test)
    test_rank_ic = spearman_corr(test_pred, y_test)
    
    logger.info(f"测试集 IC: {test_ic:.4f}")
    logger.info(f"测试集 RankIC: {test_rank_ic:.4f}")
    
    # 保存特征重要性
    importance = trainer.get_feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    
    logger.info("=" * 60)
    logger.info("📊 Top 20 特征重要性")
    logger.info("=" * 60)
    for i, (name, imp) in enumerate(top_features, 1):
        logger.info(f"  {i:2d}. {name}: {imp:.0f}")
    
    # 保存训练统计
    stats = trainer.get_training_stats()
    logger.info("=" * 60)
    logger.info("📊 训练统计")
    logger.info("=" * 60)
    logger.info(f"  最佳轮次: {stats['best_iteration']}")
    logger.info(f"  训练耗时: {stats['training_time_seconds']:.1f}s")
    logger.info(f"  最佳验证 IC: {stats['best_valid_IC']:.4f}")
    
    # 释放内存
    loader.cleanup()
    
    logger.info("=" * 60)
    logger.info("✅ 训练完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
