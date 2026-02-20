#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightGBM 模型训练入口脚本

用法:
    python scripts/run_lgb_train.py --mode rolling --targets rank_ret_5d excess_ret_10d
    python scripts/run_lgb_train.py --mode single_full --evaluate
    python scripts/run_lgb_train.py --list-targets

支持的命令行参数:
    --mode: 训练模式 (rolling/expanding/single_full)
    --targets: 要训练的目标标签列表
    --no-save: 不保存模型
    --no-evaluate: 不进行评估
    --list-targets: 列出可用的目标标签
    --data-path: 指定训练数据路径
    --no-gpu-df: 禁用 cuDF GPU 加速
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.models import SplitMode, TrainConfig
from src.models.lgb import LGBTrainer
from src.models.metrics import QuantEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "models" / "lgb_train.log"),
    ],
)
logger = logging.getLogger(__name__)


# 可用的目标标签
AVAILABLE_TARGETS = [
    # 收益类（回归）
    "ret_1d",
    "ret_5d", 
    "ret_10d",
    "ret_20d",
    # 超额收益类（回归）
    "excess_ret_5d",
    "excess_ret_10d",
    # 排序类（秩）
    "rank_ret_5d",
    "rank_ret_10d",
    # 夏普比率类（回归）
    "sharpe_5d",
    "sharpe_10d",
    "sharpe_20d",
    # 分类标签
    "label_bin_5d",
]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="LightGBM 模型训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="rolling",
        choices=["rolling", "expanding", "single_full"],
        help="训练模式: rolling(滚动窗口), expanding(扩展窗口), single_full(单次全量)",
    )
    
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["rank_ret_5d", "excess_ret_10d", "sharpe_20d"],
        help="要训练的目标标签列表",
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="训练数据路径（默认: data/features/structured/train_lgb.parquet）",
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存模型",
    )
    
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="不进行 OOF 评估",
    )
    
    parser.add_argument(
        "--no-gpu-df",
        action="store_true",
        help="禁用 cuDF GPU 加速数据处理",
    )
    
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="列出可用的目标标签",
    )
    
    parser.add_argument(
        "--train-window",
        type=int,
        default=24,
        help="训练窗口长度（月），默认24",
    )
    
    parser.add_argument(
        "--valid-window",
        type=int,
        default=3,
        help="验证窗口长度（月），默认3",
    )
    
    parser.add_argument(
        "--step",
        type=int,
        default=3,
        help="滚动步长（月），默认3",
    )
    
    return parser.parse_args()


def list_available_targets(data_path: Path) -> None:
    """列出数据中可用的目标标签"""
    print("\n可用的目标标签：")
    print("=" * 50)
    
    if data_path.exists():
        df = pd.read_parquet(data_path)
        for target in AVAILABLE_TARGETS:
            if target in df.columns:
                non_null = df[target].notna().sum()
                print(f"  ✓ {target:<20} ({non_null:,} non-null samples)")
            else:
                print(f"  ✗ {target:<20} (not found)")
    else:
        for target in AVAILABLE_TARGETS:
            print(f"  ? {target}")
    
    print("=" * 50)


def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = TrainConfig()
    
    # 更新数据路径
    if args.data_path:
        config.train_data_path = Path(args.data_path)
    
    # 更新切分配置
    config.split_config.train_window_months = args.train_window
    config.split_config.valid_window_months = args.valid_window
    config.split_config.step_months = args.step
    
    # 列出可用标签
    if args.list_targets:
        list_available_targets(config.train_data_path)
        return
    
    # 解析训练模式
    mode_map = {
        "rolling": SplitMode.ROLLING,
        "expanding": SplitMode.EXPANDING,
        "single_full": SplitMode.SINGLE_FULL,
    }
    mode = mode_map[args.mode]
    config.split_config.mode = mode
    
    logger.info("=" * 60)
    logger.info("LightGBM 训练启动")
    logger.info("=" * 60)
    logger.info(f"模式: {args.mode}")
    logger.info(f"目标标签: {args.targets}")
    logger.info(f"训练窗口: {args.train_window} 月")
    logger.info(f"验证窗口: {args.valid_window} 月") 
    logger.info(f"滚动步长: {args.step} 月")
    logger.info(f"GPU 数据加速: {not args.no_gpu_df}")
    logger.info("=" * 60)
    
    # 创建训练器
    trainer = LGBTrainer(
        config=config,
        use_gpu_df=not args.no_gpu_df,
    )
    
    # 加载数据
    logger.info("加载训练数据...")
    trainer.load_data()
    
    # 验证目标标签
    valid_targets = [t for t in args.targets if t in trainer.df.columns]
    if not valid_targets:
        logger.error(f"没有找到有效的目标标签！请检查: {args.targets}")
        return
    
    if len(valid_targets) < len(args.targets):
        missing = set(args.targets) - set(valid_targets)
        logger.warning(f"以下目标标签未找到，将跳过: {missing}")
    
    # 训练
    logger.info(f"开始训练 {len(valid_targets)} 个目标标签...")
    oof_dict = trainer.train(
        target_cols=valid_targets,
        mode=mode,
        save_models=not args.no_save,
        save_oof=not args.no_save,
    )
    
    # 评估
    if not args.no_evaluate and oof_dict:
        logger.info("=" * 60)
        logger.info("开始 OOF 评估...")
        logger.info("=" * 60)
        
        evaluator = QuantEvaluator()
        
        for target_col, oof_df in oof_dict.items():
            logger.info(f"\n评估目标: {target_col}")
            evaluator.print_report(
                oof_df,
                y_pred_col="y_pred",
                y_true_col="y_true",
                target_name=target_col,
            )
    
    # 输出特征重要性 Top 20
    logger.info("\n特征重要性 Top 20:")
    for target_col in valid_targets:
        importance = trainer.get_feature_importance(target_col)
        if importance is not None:
            print(f"\n[{target_col}]")
            print(importance.head(20).to_string(index=False))
    
    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
