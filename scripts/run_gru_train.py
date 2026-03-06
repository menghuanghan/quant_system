#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRU 多任务模型训练入口脚本

用法:
    # Rolling 模式（默认）
    python scripts/run_gru_train.py --mode rolling

    # Expanding 模式
    python scripts/run_gru_train.py --mode expanding

    # Single_Full 模式（多种子融合）
    python scripts/run_gru_train.py --mode single_full

    # 自定义目标标签
    python scripts/run_gru_train.py --targets rank_ret_5d excess_ret_10d sharpe_20d

    # 自定义窗口参数
    python scripts/run_gru_train.py --mode rolling --train-window 24 --valid-window 3 --step 3

    # 列出可用目标标签
    python scripts/run_gru_train.py --list-targets

支持的命令行参数:
    --mode: 训练模式 (rolling/expanding/single_full)
    --targets: 多目标标签列表
    --data-path: 训练数据路径
    --train-window: 训练窗口（月）
    --valid-window: 验证窗口（月）
    --step: 滚动步长（月）
    --seq-len: 回看窗口长度
    --epochs: 最大训练轮数
    --batch-size: 批次大小
    --hidden-size: GRU 隐层大小
    --num-layers: GRU 层数
    --learning-rate: 学习率
    --no-save: 不保存模型
    --no-evaluate: 不进行 OOF 评估
    --no-gpu: 禁用 GPU
    --list-targets: 列出可用目标标签
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.models.gru import (
    GRUConfig,
    GRUTrainer,
)
from src.models.metrics import QuantEvaluator

# 配置日志
log_dir = PROJECT_ROOT / "logs" / "models"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "gru_train.log"),
    ],
)
logger = logging.getLogger(__name__)


# 可用的目标标签
AVAILABLE_TARGETS = [
    "rank_ret_5d", "rank_ret_10d",
    "excess_ret_5d", "excess_ret_10d",
    "sharpe_5d", "sharpe_10d", "sharpe_20d",
    "ret_5d", "ret_10d", "ret_20d",
]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="GRU 多任务模型训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="rolling",
        choices=["rolling", "expanding", "single_full"],
        help="训练模式 (默认: rolling)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["rank_ret_5d", "excess_ret_5d", "sharpe_5d"],
        help="多目标标签列表",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="训练数据路径 (默认: data/features/structured/train_gru.parquet)",
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=24,
        help="训练窗口长度（月），默认 24",
    )
    parser.add_argument(
        "--valid-window",
        type=int,
        default=3,
        help="验证窗口长度（月），默认 3",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=3,
        help="滚动步长（月），默认 3",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=20,
        help="回看窗口长度（交易日），默认 20",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="最大训练轮数，默认 100",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="批次大小，默认 2048",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="GRU 隐层大小，默认 64",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="GRU 层数，默认 2",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="学习率，默认 5e-4",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存模型和 OOF",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="不进行 OOF 评估",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="不生成训练报告",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="禁用 GPU",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="列出可用的目标标签",
    )
    parser.add_argument(
        "--data-start-date",
        type=str,
        default="2021-01-01",
        help="数据开始日期，默认 2021-01-01",
    )
    parser.add_argument(
        "--data-end-date",
        type=str,
        default="2025-12-31",
        help="数据结束日期，默认 2025-12-31",
    )

    return parser.parse_args()


def list_available_targets(data_path: Path) -> None:
    """列出数据中可用的目标标签"""
    print("\n可用的多目标标签:")
    print("=" * 50)

    if data_path.exists():
        df = pd.read_parquet(data_path, columns=AVAILABLE_TARGETS + ["trade_date"])
        for target in AVAILABLE_TARGETS:
            if target in df.columns:
                non_null = df[target].notna().sum()
                print(f"  OK  {target:<20} ({non_null:,} non-null)")
            else:
                print(f"  --  {target:<20} (未找到)")
    else:
        for target in AVAILABLE_TARGETS:
            print(f"  ?   {target}")

    print("=" * 50)


def main():
    """主函数"""
    args = parse_args()

    # ---- 构建配置 ----
    config = GRUConfig.default()

    # 数据
    if args.data_path:
        config.data.data_path = Path(args.data_path)
    config.data.seq_len = args.seq_len
    config.data.target_cols = args.targets
    config.data.data_start_date = args.data_start_date
    config.data.data_end_date = args.data_end_date
    config.data.use_gpu = not args.no_gpu

    # 切分
    config.split.mode = args.mode
    config.split.train_window_months = args.train_window
    config.split.valid_window_months = args.valid_window
    config.split.step_months = args.step
    # 同步日期边界到 SplitConfig（Splitter 使用逻辑边界而非物理数据边界）
    config.split.data_start_date = args.data_start_date
    config.split.data_end_date = args.data_end_date

    # 网络
    config.network.hidden_size = args.hidden_size
    config.network.num_layers = args.num_layers
    config.network.num_targets = len(args.targets)

    # 训练
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.learning_rate
    config.train.max_lr = args.learning_rate * 2  # OneCycleLR max_lr

    # 动态设置损失权重
    n_targets = len(args.targets)
    loss_weights = {}
    loss_types = {}
    for col in args.targets:
        if col.startswith("rank"):
            loss_weights[col] = 0.6
            loss_types[col] = "mse"
        elif col.startswith("excess"):
            loss_weights[col] = 0.2
            loss_types[col] = "huber"
        elif col.startswith("sharpe"):
            loss_weights[col] = 0.2
            loss_types[col] = "huber"
        else:
            loss_weights[col] = 1.0 / n_targets
            loss_types[col] = "mse"
    config.train.loss_weights = loss_weights
    config.train.loss_types = loss_types

    # ---- 列出目标 ----
    if args.list_targets:
        list_available_targets(config.data.data_path)
        return

    # ---- 日志 ----
    logger.info("=" * 60)
    logger.info("GRU 多任务模型训练启动")
    logger.info("=" * 60)
    logger.info(f"模式: {args.mode}")
    logger.info(f"目标标签: {args.targets}")
    logger.info(f"序列长度: {args.seq_len}")
    if args.mode != "single_full":
        logger.info(f"训练窗口: {args.train_window} 月")
        logger.info(f"验证窗口: {args.valid_window} 月")
        logger.info(f"滑动步长: {args.step} 月")
    else:
        logger.info(f"Single_Full: 训练用全部数据(保留最后1月做早停)")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Hidden Size: {args.hidden_size}")
    logger.info(f"Num Layers: {args.num_layers}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"GPU: {not args.no_gpu}")
    logger.info(f"数据范围(逻辑边界): {args.data_start_date} ~ {args.data_end_date}")
    logger.info(f"损失权重: {loss_weights}")
    logger.info(f"损失类型: {loss_types}")
    logger.info("=" * 60)

    # ---- 创建训练器 ----
    trainer = GRUTrainer(config=config)

    # ---- 加载数据 ----
    logger.info("加载训练数据...")
    trainer.load_data()

    # ---- 验证目标标签 ----
    valid_targets = [t for t in args.targets if t in trainer.loaded_columns]
    if not valid_targets:
        logger.error(f"没有找到有效的目标标签: {args.targets}")
        return
    if len(valid_targets) < len(args.targets):
        missing = set(args.targets) - set(valid_targets)
        logger.warning(f"以下标签未找到，将跳过: {missing}")
        config.data.target_cols = valid_targets
        config.network.num_targets = len(valid_targets)

    # ---- 训练 ----
    logger.info(f"开始 {args.mode} 模式训练...")
    oof_df = trainer.train(
        mode=args.mode,
        save_models=not args.no_save,
        save_oof=not args.no_save,
        generate_report=not args.no_report,
    )

    # ---- OOF 评估 ----
    if not args.no_evaluate and not oof_df.empty:
        logger.info("=" * 60)
        logger.info("OOF 评估")
        logger.info("=" * 60)

        evaluator = QuantEvaluator()

        # 每个目标单独评估
        for col in valid_targets:
            if f"y_pred_{col}" in oof_df.columns and f"y_true_{col}" in oof_df.columns:
                logger.info(f"\n评估目标: {col}")
                evaluator.print_report(
                    oof_df,
                    y_pred_col=f"y_pred_{col}",
                    y_true_col=f"y_true_{col}",
                    target_name=col,
                )

        # 综合评估（主信号 rank 通道）
        if "y_pred" in oof_df.columns:
            logger.info(f"\n综合评估（主信号）:")
            evaluator.print_report(
                oof_df,
                y_pred_col="y_pred",
                y_true_col="y_true",
                target_name="主信号 (rank)",
            )

    logger.info("=" * 60)
    logger.info("GRU 训练完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
