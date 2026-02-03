#!/usr/bin/env python3
"""
模型融合脚本

使用方法:
    python scripts/run_ensemble.py                    # 运行完整融合流程
    python scripts/run_ensemble.py --lgb-only        # 只评估 LightGBM
    python scripts/run_ensemble.py --gru-only        # 只评估 GRU
    python scripts/run_ensemble.py --skip-train      # 跳过训练，只做融合（需要已有预测）

输出:
    reports/ensemble_analysis.md    # 分析报告
"""

import argparse
import logging
import sys
import gc
from datetime import datetime
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "ensemble.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="模型融合")
    
    parser.add_argument("--target", type=str, default="ret_5d",
                        help="预测目标 (默认: ret_5d)")
    parser.add_argument("--lgb-weight", type=float, default=0.6,
                        help="LightGBM 权重 (默认: 0.6)")
    parser.add_argument("--gru-weight", type=float, default=0.4,
                        help="GRU 权重 (默认: 0.4)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="禁用 GPU")
    parser.add_argument("--dry-run", action="store_true",
                        help="只加载数据，不训练")
    parser.add_argument("--load-models", action="store_true",
                        help="加载预训练模型而不是重新训练")
    
    return parser.parse_args()


def load_lightgbm_predictions(target_col: str = "ret_5d"):
    """加载预训练 LightGBM 模型并生成测试集预测"""
    from src.models.LBGM import LGBMConfig, DataLoader
    import pickle
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("📍 加载预训练 LightGBM 模型")
    logger.info("=" * 70)
    
    model_path = PROJECT_ROOT / "models" / "lgbm" / "lgbm_ret_5d.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"找不到预训练模型: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"  ✓ 模型已加载: {model_path}")
    
    # 加载测试数据
    config = LGBMConfig.default()
    config.data.target_col = target_col
    
    data_loader = DataLoader(config.data, use_gpu=True)
    data_loader.load()
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = data_loader.split()
    test_info = data_loader.get_test_info()
    
    # 预测
    lgb_pred = model.predict(X_test)
    
    # 清理
    del X_train, y_train, X_valid, y_valid
    data_loader.cleanup()
    gc.collect()
    
    logger.info(f"  ✓ LightGBM 预测完成: {len(lgb_pred)} 条")
    
    return lgb_pred, y_test, test_info


def load_gru_predictions(target_col: str = "ret_5d", device: str = "cuda"):
    """加载预训练 GRU 模型并生成测试集预测"""
    from src.models.deep.dataset import prepare_data, DataConfig
    from src.models.deep.model import create_model
    from src.models.deep.train import create_dataloader
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("📍 加载预训练 GRU 模型")
    logger.info("=" * 70)
    
    model_path = PROJECT_ROOT / "models" / "gru" / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"找不到预训练模型: {model_path}")
    
    # 准备数据配置
    data_config = DataConfig(
        target_col=target_col,
        window_size=20,
        use_gpu=True,
    )
    
    # 准备数据
    train_dataset, valid_dataset, test_dataset, feature_cols = prepare_data(
        config=data_config,
        device=device,
    )
    
    test_loader = create_dataloader(test_dataset, batch_size=2048, shuffle=False)
    
    # 创建模型并加载权重
    model = create_model(
        input_dim=len(feature_cols),
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"  ✓ 模型已加载: {model_path}")
    logger.info(f"    最佳验证 RankIC: {checkpoint.get('best_valid_metric', 'N/A')}")
    
    model.eval()
    
    gru_preds = []
    gru_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            pred = model(x)
            gru_preds.append(pred.cpu().numpy())
            gru_labels.append(y.cpu().numpy())
    
    gru_pred = np.concatenate(gru_preds)
    gru_label = np.concatenate(gru_labels)
    
    # 获取测试集的日期和股票代码信息
    test_dates = test_dataset.get_all_dates()
    test_codes = test_dataset.get_all_codes()
    
    logger.info(f"  ✓ GRU 预测完成: {len(gru_pred)} 条")
    
    return gru_pred, gru_label, test_dates, test_codes


def train_lightgbm(target_col: str = "ret_5d"):
    """训练 LightGBM 并返回测试集预测"""
    from src.models.LBGM import LGBMConfig, DataLoader, LGBMTrainer
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("📍 训练 LightGBM")
    logger.info("=" * 70)
    
    config = LGBMConfig.default()
    config.data.target_col = target_col
    
    data_loader = DataLoader(config.data, use_gpu=True)
    data_loader.load()
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = data_loader.split()
    feature_names = data_loader.get_feature_names()
    test_info = data_loader.get_test_info()
    
    trainer = LGBMTrainer(config)
    model = trainer.train(X_train, y_train, X_valid, y_valid, feature_names=feature_names)
    
    # 预测
    lgb_pred = model.predict(X_test)
    
    # 清理
    del X_train, y_train, X_valid, y_valid
    data_loader.cleanup()
    gc.collect()
    
    logger.info(f"  ✓ LightGBM 预测完成: {len(lgb_pred)} 条")
    
    return lgb_pred, y_test, test_info


def train_gru(target_col: str = "ret_5d", device: str = "cuda"):
    """训练 GRU 并返回测试集预测"""
    from src.models.deep.dataset import prepare_data, DataConfig
    from src.models.deep.model import create_model
    from src.models.deep.train import TrainConfig, GRUTrainer, create_dataloader
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("📍 训练 GRU")
    logger.info("=" * 70)
    
    # 准备数据配置
    data_config = DataConfig(
        target_col=target_col,
        window_size=20,
        use_gpu=True,
    )
    
    # 准备数据
    train_dataset, valid_dataset, test_dataset, feature_cols = prepare_data(
        config=data_config,
        device=device,
    )
    
    train_loader = create_dataloader(train_dataset, batch_size=2048, shuffle=True)
    valid_loader = create_dataloader(valid_dataset, batch_size=2048, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=2048, shuffle=False)
    
    # 创建模型
    model = create_model(
        input_dim=len(feature_cols),
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
    ).to(device)
    
    # 训练配置
    train_config = TrainConfig(
        epochs=30,
        batch_size=2048,
        learning_rate=1e-3,
        patience=10,
        use_amp=True,
    )
    
    trainer = GRUTrainer(model, train_config, device=device)
    trainer.train(train_loader, valid_loader)
    
    # 加载最佳模型并预测
    trainer.load_model("best_model.pt")
    model.eval()
    
    gru_preds = []
    gru_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            pred = model(x)
            gru_preds.append(pred.cpu().numpy())
            gru_labels.append(y.cpu().numpy())
    
    gru_pred = np.concatenate(gru_preds)
    gru_label = np.concatenate(gru_labels)
    
    # 获取测试集的日期和股票代码信息
    test_dates = test_dataset.get_all_dates()
    test_codes = test_dataset.get_all_codes()
    
    logger.info(f"  ✓ GRU 预测完成: {len(gru_pred)} 条")
    
    return gru_pred, gru_label, test_dates, test_codes


def main():
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("🚀 模型融合实验")
    logger.info(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    logger.info(f"  📋 设备: {device}")
    logger.info(f"  📋 目标: {args.target}")
    logger.info(f"  📋 默认权重: LGB={args.lgb_weight}, GRU={args.gru_weight}")
    
    if args.dry_run:
        logger.info("")
        logger.info("🏁 Dry run 模式，跳过训练")
        return
    
    # ========== 获取预测 ==========
    if args.load_models:
        logger.info("")
        logger.info("📦 加载预训练模型模式")
        # 1. LightGBM
        lgb_pred, lgb_y_test, lgb_test_info = load_lightgbm_predictions(args.target)
        
        # 清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. GRU
        gru_pred, gru_y_test, gru_dates, gru_codes = load_gru_predictions(args.target, device)
    else:
        # 1. LightGBM
        lgb_pred, lgb_y_test, lgb_test_info = train_lightgbm(args.target)
        
        # 清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. GRU
        gru_pred, gru_y_test, gru_dates, gru_codes = train_gru(args.target, device)
    
    # ========== 精确对齐数据 ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("📍 精确对齐预测数据 (使用 date + code)")
    logger.info("=" * 70)
    
    # 创建 DataFrame 以便精确对齐
    lgb_df = pd.DataFrame({
        'date': lgb_test_info['dates'],
        'code': lgb_test_info['codes'],
        'lgb_pred': lgb_pred,
        'label': lgb_y_test,
    })
    
    gru_df = pd.DataFrame({
        'date': gru_dates,
        'code': gru_codes,
        'gru_pred': gru_pred,
    })
    
    # 去除重复的 (date, code) 组合（保留第一个）
    lgb_df = lgb_df.drop_duplicates(subset=['date', 'code'], keep='first')
    gru_df = gru_df.drop_duplicates(subset=['date', 'code'], keep='first')
    
    logger.info(f"  LightGBM 样本数 (去重后): {len(lgb_df)}")
    logger.info(f"  GRU 样本数 (去重后): {len(gru_df)}")
    
    # 精确 merge
    merged_df = pd.merge(
        lgb_df,
        gru_df,
        on=['date', 'code'],
        how='inner'
    )
    
    logger.info(f"  ✓ 对齐后样本数: {len(merged_df)}")
    
    # 提取对齐后的数据
    lgb_pred_aligned = merged_df['lgb_pred'].values
    gru_pred_aligned = merged_df['gru_pred'].values
    y_test = merged_df['label'].values
    test_dates = merged_df['date'].values
    
    # ========== 融合 ==========
    from src.models.ensemble import ModelFusion, FusionConfig
    
    config = FusionConfig(
        lgb_weight=args.lgb_weight,
        gru_weight=args.gru_weight,
    )
    
    fusion = ModelFusion(config)
    fusion.set_predictions(
        lgb_pred=lgb_pred_aligned,
        gru_pred=gru_pred_aligned,
        labels=y_test,
        dates=test_dates,
    )
    
    # 相关性分析
    corr_results = fusion.analyze_correlation()
    
    # 对比所有策略
    results_df = fusion.compare_all()
    
    # ========== 生成报告 ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("📝 生成报告")
    logger.info("=" * 70)
    
    report_path = PROJECT_ROOT / "reports" / "ensemble_analysis.md"
    
    best_strategy = results_df.iloc[0]
    
    report_content = f"""# 模型融合分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 模型相关性

| 指标 | 值 |
|------|-----|
| Pearson 相关系数 | {corr_results['pearson_corr']:.4f} |
| Spearman 相关系数 | {corr_results['spearman_corr']:.4f} |

**结论**: {"模型互补性强，融合价值大" if abs(corr_results['pearson_corr']) < 0.7 else "模型有一定同质性"}

## 2. 融合策略对比

| 策略 | 日均 RankIC | ICIR | 胜率 |
|------|------------|------|------|
"""
    
    for _, row in results_df.iterrows():
        report_content += f"| {row['strategy']} | {row['daily_rank_ic_mean']:.4f} | {row['icir']:.4f} | {row['win_rate']:.2%} |\n"
    
    report_content += f"""

## 3. 最佳策略

- **策略**: {best_strategy['strategy']}
- **日均 RankIC**: {best_strategy['daily_rank_ic_mean']:.4f}
- **ICIR**: {best_strategy['icir']:.4f}
- **胜率**: {best_strategy['win_rate']:.2%}

## 4. 结论

{"✅ 融合后 ICIR > 0.5，策略非常稳定！" if best_strategy['icir'] > 0.5 else ""}
{"✓ 融合后 ICIR > 0.3，策略较稳定" if 0.3 < best_strategy['icir'] <= 0.5 else ""}
{"⚠️ ICIR < 0.3，需要进一步优化" if best_strategy['icir'] <= 0.3 else ""}
"""
    
    report_path.write_text(report_content, encoding="utf-8")
    logger.info(f"  ✓ 报告已保存: {report_path}")
    
    # ========== 完成 ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎉 融合实验完成!")
    logger.info(f"   最佳策略: {best_strategy['strategy']}")
    logger.info(f"   ICIR: {best_strategy['icir']:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
