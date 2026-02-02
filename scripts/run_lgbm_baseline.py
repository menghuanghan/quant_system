#!/usr/bin/env python3
"""
LightGBM 基准模型训练脚本

使用方法:
    python scripts/run_lgbm_baseline.py                    # 使用默认配置
    python scripts/run_lgbm_baseline.py --no-gpu           # CPU 模式
    python scripts/run_lgbm_baseline.py --target ret_10d   # 预测 10 日收益
    python scripts/run_lgbm_baseline.py --dry-run          # 只加载数据，不训练

输出:
    models/lgbm/lgbm_ret5d.pkl    # 模型文件
    models/lgbm/lgbm_ret5d.txt    # LightGBM 原生格式
    reports/lgbm_analysis.md      # 分析报告
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.LBGM import (
    LGBMConfig,
    DataLoader,
    LGBMTrainer,
    ModelAnalyzer,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "lgbm_training.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="LightGBM 量化因子模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="禁用 GPU 加速",
    )
    
    parser.add_argument(
        "--target",
        type=str,
        default="ret_5d",
        choices=["ret_1d", "ret_5d", "ret_10d", "ret_20d"],
        help="预测目标 (默认: ret_5d)",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
        help="学习率 (默认: 0.03)",
    )
    
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=63,
        help="叶子数 (默认: 63)",
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="树深度 (默认: 6)",
    )
    
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=50,
        help="早停轮数 (默认: 50)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只加载数据，不训练",
    )
    
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="跳过 SHAP 分析 (较耗时)",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("🚀 LightGBM 基准模型训练")
    logger.info(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # 创建配置
    config = LGBMConfig.default()
    
    # 应用命令行参数
    if args.no_gpu:
        config.params.device = "cpu"
        logger.info("  📋 模式: CPU")
    else:
        logger.info("  📋 模式: GPU")
    
    config.data.target_col = args.target
    config.params.learning_rate = args.learning_rate
    config.params.num_leaves = args.num_leaves
    config.params.max_depth = args.max_depth
    config.train.early_stopping_rounds = args.early_stopping
    config.train.model_name = f"lgbm_{args.target}"
    
    logger.info(f"  📋 目标列: {args.target}")
    logger.info(f"  📋 学习率: {args.learning_rate}")
    logger.info(f"  📋 叶子数: {args.num_leaves}")
    logger.info(f"  📋 树深度: {args.max_depth}")
    
    # Step 1: 加载数据
    logger.info("")
    data_loader = DataLoader(config.data, use_gpu=True)  # 数据加载始终用 GPU 加速
    data_loader.load()
    
    # Step 2: 时间序列切分
    X_train, y_train, X_valid, y_valid, X_test, y_test = data_loader.split()
    feature_names = data_loader.get_feature_names()
    test_info = data_loader.get_test_info()
    
    if args.dry_run:
        logger.info("")
        logger.info("🏁 Dry run 完成，跳过训练")
        data_loader.cleanup()
        return
    
    # Step 3: 训练模型
    logger.info("")
    trainer = LGBMTrainer(config)
    
    # 获取类别特征
    cat_features = [f for f in config.data.categorical_features if f in feature_names]
    
    model = trainer.train(
        X_train, y_train,
        X_valid, y_valid,
        feature_names=feature_names,
        categorical_features=cat_features,
    )
    
    # 释放训练数据内存
    del X_train, y_train, X_valid, y_valid
    data_loader.cleanup()
    
    # Step 4: 模型分析
    logger.info("")
    analyzer = ModelAnalyzer(model, config)
    
    # 特征重要性
    fi = analyzer.analyze_feature_importance(top_n=20)
    
    # 测试集评估 (test_info['dates'] 已经是 numpy array)
    test_dates = test_info['dates']
    metrics = analyzer.evaluate_test_set(X_test, y_test, test_dates)
    
    # 每日 IC 分析
    daily_ic = analyzer.analyze_daily_ic()
    
    # SHAP 分析 (可选)
    if not args.skip_shap:
        try:
            # 取最后一天的数据做 SHAP 分析
            last_date = test_dates[-1]
            sample_mask = test_dates == last_date
            
            if sample_mask.sum() > 0:
                X_sample = X_test[sample_mask][:100]  # 最多取 100 个样本
                analyzer.explain_with_shap(
                    X_sample,
                    feature_names,
                    sample_date=last_date
                )
        except Exception as e:
            logger.warning(f"  ⚠️ SHAP 分析失败: {e}")
    
    # 生成报告
    report_path = PROJECT_ROOT / "reports" / f"lgbm_{args.target}_analysis.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    analyzer.generate_report(report_path)
    
    # 打印训练统计
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 训练统计")
    logger.info("=" * 70)
    
    train_stats = trainer.get_training_stats()
    logger.info(f"  最佳轮次: {train_stats['best_iteration']}")
    logger.info(f"  训练耗时: {train_stats['training_time_seconds']:.2f} 秒")
    logger.info(f"  最佳验证 IC: {train_stats['best_valid_IC']:.4f}")
    
    logger.info("")
    logger.info(f"  测试集 RankIC: {metrics.get('DailyRankIC_Mean', 0):.4f}")
    logger.info(f"  测试集 ICIR: {metrics.get('ICIR', 0):.4f}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎉 训练完成!")
    logger.info(f"   模型: {config.data.output_dir / config.train.model_name}.pkl")
    logger.info(f"   报告: {report_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
