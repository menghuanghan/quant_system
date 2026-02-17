#!/usr/bin/env python
"""
GRU 模型训练脚本（防过拟合优化版 + Embedding 支持）

优化措施：
1. 只使用 LightGBM Top 50 强特征
2. 增大 Dropout (0.5) 和 Weight Decay (1e-3)
3. 降低模型容量 (hidden_dim=32, num_layers=1)
4. 可选：使用 Industry/Market Embedding 处理类别特征

自动特征筛选:
- 当启用 --use-feature-selection 时，自动检查特征文件是否存在
- 若不存在，从 LightGBM 模型自动生成 Top 50 特征
- 若启用 --use-embedding，类别特征由 Embedding 处理，不排除
- 若未启用 --use-embedding，类别特征将被排除

使用方法:
    python scripts/run_gru_train.py
    python scripts/run_gru_train.py --use-feature-selection
    python scripts/run_gru_train.py --use-feature-selection --use-embedding
    python scripts/run_gru_train.py --hidden-dim 32 --dropout 0.5 --weight-decay 1e-3
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.deep.config import GRUDataConfig, GRUModelConfig, CATEGORICAL_FEATURES
from src.models.deep.dataset import prepare_data
from src.models.deep.model import create_model
from src.models.deep.train import GRUTrainer, TrainConfig, create_dataloader, seed_everything
from src.models.deep.loss import compute_ic, compute_rank_ic


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_feature_selection(
    lgbm_model_paths: list,
    output_json_path: Path,
    exclude_categorical: bool = True,
    top_n: int = 50,
) -> list:
    """
    从 LightGBM 模型生成 Top N 特征列表（委员会投票机制）
    
    对所有 LightGBM 模型的特征重要性进行平均，选出综合排名最高的特征。
    
    Args:
        lgbm_model_paths: LightGBM 模型路径列表 (.pkl)
        output_json_path: 输出 JSON 路径
        exclude_categorical: 是否排除类别特征
        top_n: 保留的特征数量
        
    Returns:
        特征列表
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"📊 生成特征筛选文件（委员会投票）...")
    logger.info(f"  模型数量: {len(lgbm_model_paths)}")
    logger.info(f"  排除类别特征: {exclude_categorical}")
    
    from collections import defaultdict
    import numpy as np
    
    # 收集所有模型的特征重要性
    feature_importance_sum = defaultdict(float)
    model_count = 0
    
    for model_path in lgbm_model_paths:
        logger.info(f"  加载模型: {model_path.name}")
        
        # 加载 LightGBM 模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # 获取特征重要性 (使用 gain)
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        
        # 累加重要性
        for name, imp in zip(feature_names, importance):
            feature_importance_sum[name] += imp
        
        model_count += 1
    
    # 计算平均重要性
    feature_importance_avg = {
        name: imp / model_count 
        for name, imp in feature_importance_sum.items()
    }
    
    # 按平均重要性排序
    sorted_features = sorted(
        feature_importance_avg.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # 获取 Top N 特征
    top_features = [f[0] for f in sorted_features[:top_n * 2]]  # 预留更多
    
    # 是否排除类别特征
    if exclude_categorical:
        categorical = CATEGORICAL_FEATURES
        selected_features = [f for f in top_features if f not in categorical][:top_n]
        logger.info(f"  排除类别特征: {categorical}")
    else:
        selected_features = top_features[:top_n]
        logger.info(f"  保留类别特征（由 Embedding 处理）")
    
    # 保存
    output = {
        'features': selected_features,
        'count': len(selected_features),
        'exclude_categorical': exclude_categorical,
        'model_count': model_count,
        'model_paths': [str(p.name) for p in lgbm_model_paths],
        'feature_importance': {f: feature_importance_avg.get(f, 0) for f in selected_features},
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 已保存 {len(selected_features)} 个特征到 {output_json_path}")
    logger.info(f"  Top 10: {selected_features[:10]}")
    
    return selected_features


def ensure_feature_selection(
    feature_json_path: Path,
    lgbm_model_dir: Path,
    target_col: str,
    use_embedding: bool,
    force_regenerate: bool = False,
) -> None:
    """
    确保特征筛选文件存在，不存在则自动生成
    
    使用委员会投票机制：自动查找所有 LightGBM 模型文件，
    计算特征重要性平均值，选出综合排名最高的特征。
    
    Args:
        feature_json_path: 特征 JSON 文件路径
        lgbm_model_dir: LightGBM 模型目录
        target_col: 目标列名（用于匹配模型文件）
        use_embedding: 是否启用 Embedding（决定是否排除类别特征）
        force_regenerate: 强制重新生成
    """
    import glob
    logger = logging.getLogger(__name__)
    
    # 使用 glob 模式匹配所有 LightGBM 模型文件
    # 匹配模式: lgbm_{target}_seed*.pkl
    pattern = lgbm_model_dir / f"lgbm_{target_col}_seed*.pkl"
    model_files = list(lgbm_model_dir.glob(f"lgbm_{target_col}_seed*.pkl"))
    
    # 如果没有找到 seed 模型，尝试匹配默认模型文件
    if not model_files:
        default_model = lgbm_model_dir / f"lgbm_{target_col}.pkl"
        if default_model.exists():
            model_files = [default_model]
    
    # 检查是否找到模型文件
    if not model_files:
        raise FileNotFoundError(
            f"未找到 LightGBM 模型文件！\n"
            f"  已尝试匹配: {pattern}\n"
            f"  请先运行 LightGBM 训练: python scripts/run_lgb_train.py"
        )
    
    logger.info(f"📂 找到 {len(model_files)} 个 LightGBM 模型文件:")
    for f in model_files:
        logger.info(f"    - {f.name}")
    
    # 检查特征文件是否存在
    need_regenerate = force_regenerate or not feature_json_path.exists()
    
    # 检查特征文件配置是否匹配当前 Embedding 设置
    if feature_json_path.exists() and not force_regenerate:
        with open(feature_json_path, 'r') as f:
            existing = json.load(f)
        
        # 如果 Embedding 设置变化，需要重新生成
        existing_exclude = existing.get('exclude_categorical', True)
        current_exclude = not use_embedding  # Embedding 启用时不排除类别特征
        
        if existing_exclude != current_exclude:
            logger.warning(f"⚠️ 特征文件配置不匹配 (exclude_categorical: {existing_exclude} → {current_exclude})")
            need_regenerate = True
        
        # 如果模型数量变化，需要重新生成
        existing_model_count = existing.get('model_count', 1)
        if existing_model_count != len(model_files):
            logger.warning(f"⚠️ 模型数量变化 ({existing_model_count} → {len(model_files)})")
            need_regenerate = True
    
    if need_regenerate:
        generate_feature_selection(
            lgbm_model_paths=model_files,
            output_json_path=feature_json_path,
            exclude_categorical=not use_embedding,  # Embedding 启用时不排除
            top_n=50,
        )
    else:
        logger.info(f"✅ 使用已有特征文件: {feature_json_path}")


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
    parser.add_argument(
        "--regenerate-features",
        action="store_true",
        help="强制重新生成特征筛选文件",
    )
    
    # Embedding 配置
    parser.add_argument(
        "--use-embedding",
        action="store_true",
        help="启用类别特征 Embedding",
    )
    parser.add_argument(
        "--embedding-features",
        type=str,
        nargs="+",
        default=["sw_l1_idx", "market", "industry_idx", "sw_l2_idx"],
        help="Embedding 特征列表 (default: sw_l1_idx)",
    )
    
    # 训练策略
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="早停耐心值 (default: 15)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率 (default: 1e-3)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="梯度裁剪阈值 (default: 1.0)",
    )
    
    # 随机种子配置（可复现性）
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (default: 42)",
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="启用多种子训练模式（训练多个模型用于 Seed Ensemble）",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 100, 888],
        help="多种子训练时的种子列表 (default: 42 100 888)",
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
    
    # 多种子训练模式
    if args.multi_seed:
        import subprocess
        import sys
        
        logger.info("=" * 60)
        logger.info("🎲 多种子训练模式（Seed Ensemble）")
        logger.info("=" * 60)
        logger.info(f"种子列表: {args.seeds}")
        logger.info("使用 subprocess 运行每个种子，确保内存完全释放")
        
        for idx, seed in enumerate(args.seeds):
            logger.info(f"\n{'='*60}")
            logger.info(f"🌱 使用种子 {seed} 开始训练... ({idx + 1}/{len(args.seeds)})")
            logger.info("=" * 60)
            
            # 构建单种子训练命令
            cmd = [
                sys.executable, str(Path(__file__)),
                "--seed", str(seed),
                "--target", args.target,
                "--batch-size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--hidden-dim", str(args.hidden_dim),
                "--num-layers", str(args.num_layers),
                "--dropout", str(args.dropout),
                "--weight-decay", str(args.weight_decay),
                "--patience", str(args.patience),
                "--lr", str(args.lr),
                "--grad-clip", str(args.grad_clip),
                "--window-size", str(args.window_size),
                "--log-level", args.log_level,
            ]
            
            # 添加可选参数
            if args.use_feature_selection:
                cmd.append("--use-feature-selection")
                cmd.extend(["--feature-json", args.feature_json])
                # 只在第一个种子运行时重新生成特征文件
                if args.regenerate_features and idx == 0:
                    cmd.append("--regenerate-features")
            if args.use_embedding:
                cmd.append("--use-embedding")
                cmd.extend(["--embedding-features"] + args.embedding_features)
            if args.data_path:
                cmd.extend(["--data-path", args.data_path])
            if args.cpu:
                cmd.append("--cpu")
            
            # 运行子进程（继承 stdout/stderr）
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                logger.error(f"❌ 种子 {seed} 训练失败 (返回码: {result.returncode})")
            else:
                logger.info(f"✅ 种子 {seed} 训练完成")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 多种子训练完成！")
        logger.info(f"   已保存模型: gru_{args.target}_seed*.pt")
        logger.info("   请在融合脚本中启用 --seed-ensemble 选项")
        logger.info("=" * 60)
    else:
        # 单种子训练模式 - 使用带种子后缀的文件名
        train_single_model(args, seed_suffix=f"_seed{args.seed}")


def train_single_model(args, seed_suffix: str = ""):
    """
    训练单个模型
    
    Args:
        args: 命令行参数
        seed_suffix: 模型文件名后缀（多种子模式使用）
    """
    logger = logging.getLogger(__name__)
    
    # 固定随机种子（可复现性）
    seed_everything(args.seed)
    
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
        feature_json_path = Path(args.feature_json)
        lgbm_model_dir = PROJECT_ROOT / "models" / "lgbm"
        
        # 自动检查/生成特征筛选文件（委员会投票机制）
        ensure_feature_selection(
            feature_json_path=feature_json_path,
            lgbm_model_dir=lgbm_model_dir,
            target_col=args.target,
            use_embedding=args.use_embedding,
            force_regenerate=args.regenerate_features,
        )
        
        data_config.use_feature_selection = True
        data_config.feature_selection_json = feature_json_path
        logger.info(f"✅ 特征筛选已启用: {feature_json_path}")
    
    # 启用 Embedding
    if args.use_embedding:
        data_config.use_embedding = True
        data_config.embedding_features = args.embedding_features
        data_config.exclude_categorical = True  # 排除类别特征作为数值输入（由 Embedding 处理）
        logger.info(f"✅ Embedding 已启用: {args.embedding_features}")
    
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
    model_config = GRUModelConfig()
    model_config.input_dim = len(feature_cols)
    model_config.hidden_dim = args.hidden_dim
    model_config.num_layers = args.num_layers
    model_config.dropout = args.dropout
    
    # 设置 Embedding 配置
    if args.use_embedding:
        model_config.use_embedding = True
        model_config.embedding_features = args.embedding_features
    
    model = create_model(config=model_config)
    
    # 训练配置（增强正则化）
    train_config = TrainConfig()
    train_config.epochs = args.epochs
    train_config.batch_size = args.batch_size
    train_config.weight_decay = args.weight_decay  # 增大权重衰减
    train_config.patience = args.patience  # 早停耐心
    train_config.learning_rate = args.lr  # 学习率
    train_config.grad_clip = args.grad_clip  # 梯度裁剪
    train_config.model_name = f"gru_{args.target}"
    
    # 多种子模式：设置不同的 best_model 文件名
    if seed_suffix:
        train_config.best_model_prefix = f"best_model{seed_suffix}"
    
    logger.info(f"  随机种子: {args.seed}")
    
    logger.info("=" * 60)
    logger.info("📋 防过拟合配置")
    logger.info("=" * 60)
    logger.info(f"  隐层大小: {args.hidden_dim} (原 128)")
    logger.info(f"  GRU 层数: {args.num_layers} (原 2)")
    logger.info(f"  Dropout: {args.dropout} (原 0.3)")
    logger.info(f"  权重衰减: {args.weight_decay} (原 1e-4)")
    logger.info(f"  学习率: {args.lr}")
    logger.info(f"  早停耐心: {args.patience}")
    logger.info(f"  梯度裁剪: {args.grad_clip}")
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
    for batch_data in test_loader:
        if len(batch_data) == 2:
            _, y = batch_data
        else:
            _, _, y = batch_data
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
    
    # 保存最终模型（支持多种子模式）
    if seed_suffix:
        # 多种子模式：保存为 gru_{target}_seed{N}.pt
        trainer.save_model(f"gru_{args.target}{seed_suffix}.pt")
        # 同时保存最佳模型
        best_model_name = f"best_model{seed_suffix}.pt"
    else:
        trainer.save_model(f"final_{args.target}.pt")
        best_model_name = "best_model.pt"
    
    # 保存模型配置（供融合脚本使用）
    model_config_dict = {
        'input_dim': len(feature_cols),
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'mlp_hidden': model_config.mlp_hidden,
        'use_embedding': args.use_embedding,
        'embedding_features': args.embedding_features if args.use_embedding else [],
        'use_feature_selection': args.use_feature_selection,
        'feature_json_path': str(Path(args.feature_json)) if args.use_feature_selection else None,
        'target_col': args.target,
        'window_size': args.window_size,
        'seed': args.seed,  # 记录使用的种子
        # 训练配置（参考用）
        'train_config': {
            'weight_decay': args.weight_decay,
            'learning_rate': args.lr,
            'patience': args.patience,
            'epochs': args.epochs,
        },
        # 训练结果
        'best_valid_rank_ic': float(trainer.best_valid_ic),
        'test_rank_ic': float(test_rank_ic),
    }
    
    # 配置文件名（多种子模式使用不同文件名）
    if seed_suffix:
        config_save_path = train_config.save_dir / f"model_config{seed_suffix}.json"
    else:
        config_save_path = train_config.save_dir / "model_config.json"
    
    with open(config_save_path, 'w') as f:
        json.dump(model_config_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 模型配置已保存: {config_save_path}")
    
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
