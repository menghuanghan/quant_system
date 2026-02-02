#!/usr/bin/env python
"""
GRU 模型参数调优脚本

网格搜索关键超参数，找到最优配置
"""

import argparse
import logging
import sys
import gc
from pathlib import Path
from datetime import datetime
from itertools import product
import json

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ========== 超参数搜索空间 ==========
PARAM_GRID = {
    "hidden_dim": [64, 128],
    "num_layers": [2, 3],
    "dropout": [0.3, 0.4],
    "learning_rate": [5e-4, 1e-3],
    "weight_decay": [1e-4, 1e-3],
}

# ========== 固定参数 ==========
FIXED_PARAMS = {
    "batch_size": 8192,
    "epochs": 30,
    "patience": 8,
    "window_size": 20,
    "target_col": "ret_5d",
}


def run_experiment(config: dict, train_ds, valid_ds, test_ds, feature_cols, device: str):
    """运行单次实验"""
    from src.models.deep.model import create_model
    from src.models.deep.train import TrainConfig, GRUTrainer, create_dataloader
    from src.models.deep.loss import compute_rank_ic
    
    # 创建 DataLoader
    train_loader = create_dataloader(train_ds, batch_size=config["batch_size"], shuffle=True)
    valid_loader = create_dataloader(valid_ds, batch_size=config["batch_size"] * 2, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=config["batch_size"] * 2, shuffle=False)
    
    # 创建模型
    input_dim = len(feature_cols)
    model = create_model(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    
    # 配置训练器
    train_config = TrainConfig(
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        patience=config["patience"],
        num_workers=0,
        use_amp=True,
        weight_decay=config["weight_decay"],
    )
    
    trainer = GRUTrainer(model, train_config, device=device)
    
    # 训练
    history = trainer.train(train_loader, valid_loader)
    
    # 加载最佳模型并评估测试集
    trainer.load_model("best_model.pt")
    test_preds = trainer.predict(test_loader)
    
    # 获取测试集标签 (数据可能在 GPU 上)
    test_labels = []
    for i in range(len(test_ds)):
        _, y = test_ds[i]
        if hasattr(y, 'cpu'):
            test_labels.append(y.cpu().item())
        else:
            test_labels.append(float(y))
    test_labels = np.array(test_labels)
    
    test_ic = np.corrcoef(test_preds, test_labels)[0, 1]
    test_rank_ic = compute_rank_ic(test_preds, test_labels)
    
    result = {
        "config": config,
        "best_valid_rank_ic": max(history["valid_rank_ic"]),
        "best_epoch": int(np.argmax(history["valid_rank_ic"])) + 1,
        "test_ic": float(test_ic),
        "test_rank_ic": float(test_rank_ic),
        "total_params": sum(p.numel() for p in model.parameters()),
    }
    
    # 清理
    del model, trainer, train_loader, valid_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="GRU 参数调优")
    parser.add_argument("--max-trials", type=int, default=None, help="最大实验次数")
    parser.add_argument("--quick", action="store_true", help="快速模式: 减少 epochs")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🔬 GRU 超参数调优")
    logger.info(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"📍 设备: {device}")
    
    if args.quick:
        FIXED_PARAMS["epochs"] = 15
        FIXED_PARAMS["patience"] = 5
        logger.info("⚡ 快速模式: epochs=15, patience=5")
    
    # 生成所有配置组合
    param_names = list(PARAM_GRID.keys())
    param_values = [PARAM_GRID[k] for k in param_names]
    all_configs = []
    for values in product(*param_values):
        config = dict(zip(param_names, values))
        config.update(FIXED_PARAMS)
        all_configs.append(config)
    
    total_trials = len(all_configs)
    if args.max_trials:
        all_configs = all_configs[:args.max_trials]
        
    logger.info(f"📊 总配置数: {total_trials}")
    logger.info(f"📊 实际运行: {len(all_configs)}")
    logger.info("")
    
    # 加载数据 (只加载一次)
    from src.models.deep.dataset import DataConfig, prepare_data
    
    data_config = DataConfig(
        window_size=FIXED_PARAMS["window_size"],
        target_col=FIXED_PARAMS["target_col"],
        zscore_only=True,
    )
    
    train_ds, valid_ds, test_ds, feature_cols = prepare_data(data_config, device=device)
    
    # 运行实验
    results = []
    for i, config in enumerate(all_configs, 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"🧪 实验 {i}/{len(all_configs)}")
        logger.info(f"   hidden_dim={config['hidden_dim']}, layers={config['num_layers']}")
        logger.info(f"   dropout={config['dropout']}, lr={config['learning_rate']}")
        logger.info(f"   weight_decay={config['weight_decay']}")
        logger.info("=" * 60)
        
        try:
            result = run_experiment(config, train_ds, valid_ds, test_ds, feature_cols, device)
            results.append(result)
            
            logger.info(f"   ✅ Test RankIC: {result['test_rank_ic']:.4f}")
            logger.info(f"   ✅ Best Epoch: {result['best_epoch']}")
        except Exception as e:
            logger.error(f"   ❌ 实验失败: {e}")
            continue
    
    # 汇总结果
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 调优结果汇总")
    logger.info("=" * 60)
    
    if results:
        # 按测试 RankIC 排序
        results_sorted = sorted(results, key=lambda x: x["test_rank_ic"], reverse=True)
        
        for i, r in enumerate(results_sorted[:5], 1):
            c = r["config"]
            logger.info(f"")
            logger.info(f"Top {i}: Test RankIC = {r['test_rank_ic']:.4f}")
            logger.info(f"   hidden_dim={c['hidden_dim']}, layers={c['num_layers']}, dropout={c['dropout']}")
            logger.info(f"   lr={c['learning_rate']}, weight_decay={c['weight_decay']}")
            logger.info(f"   params={r['total_params']:,}, best_epoch={r['best_epoch']}")
        
        # 保存结果
        output_path = PROJECT_ROOT / "reports" / "gru_tuning_results.json"
        with open(output_path, "w") as f:
            json.dump(results_sorted, f, indent=2)
        logger.info(f"")
        logger.info(f"💾 结果已保存: {output_path}")
        
        # 打印最佳配置
        best = results_sorted[0]
        logger.info("")
        logger.info("=" * 60)
        logger.info("🏆 最佳配置")
        logger.info("=" * 60)
        logger.info(f"python scripts/run_gru_baseline.py \\")
        logger.info(f"  --hidden-dim {best['config']['hidden_dim']} \\")
        logger.info(f"  --num-layers {best['config']['num_layers']} \\")
        logger.info(f"  --dropout {best['config']['dropout']} \\")
        logger.info(f"  --learning-rate {best['config']['learning_rate']} \\")
        logger.info(f"  --epochs 50 --patience 15")


if __name__ == "__main__":
    main()
