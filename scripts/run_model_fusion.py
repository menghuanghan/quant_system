#!/usr/bin/env python
"""
模型融合脚本（适配改造后的模型层）

功能：
1. 加载 models/ 目录下已有的 LightGBM 和 GRU 模型
2. 生成测试集预测
3. 使用 src/models/ensemble/fusion.py 的 ModelFusion 类执行融合
4. 输出融合报告

使用方法:
    python scripts/run_model_fusion.py
    python scripts/run_model_fusion.py --lgb-weight 0.7 --gru-weight 0.3
"""

import argparse
import gc
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="模型融合")
    parser.add_argument("--target", type=str, default="excess_ret_5d",
                        help="预测目标 (default: excess_ret_5d)")
    parser.add_argument("--lgb-weight", type=float, default=0.7,
                        help="LightGBM 权重 (default: 0.7)")
    parser.add_argument("--gru-weight", type=float, default=0.3,
                        help="GRU 权重 (default: 0.3)")
    parser.add_argument("--retrain-lgb", action="store_true",
                        help="强制重新训练 LightGBM")
    parser.add_argument("--retrain-gru", action="store_true",
                        help="强制重新训练 GRU")
    
    # 种子融合（Seed Ensemble）参数
    parser.add_argument("--gru-seed-ensemble", action="store_true",
                        help="启用 GRU 多种子融合（加载所有 best_model_seed*.pt）")
    parser.add_argument("--lgb-seed-ensemble", action="store_true",
                        help="启用 LightGBM 多种子融合（加载所有 lgbm_*_seed*.pkl）")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 888],
                        help="种子列表，用于种子融合 (default: 42 100 888)")
    
    return parser.parse_args()


def load_or_train_lgb(target_col: str, retrain: bool = False):
    """加载或训练 LightGBM 模型"""
    from src.models.LBGM.config import LGBMConfig
    from src.models.LBGM.data_loader import DataLoader
    from src.models.LBGM.trainer import LGBMTrainer
    
    logger.info("=" * 60)
    logger.info("📊 LightGBM 模型")
    logger.info("=" * 60)
    
    model_path = PROJECT_ROOT / "models" / "lgbm" / f"lgbm_{target_col}.pkl"
    
    # 加载数据
    config = LGBMConfig.default()
    config.data.target_col = target_col
    
    loader = DataLoader(config.data, use_gpu=True)
    loader.load()
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = loader.split()
    feature_names = loader.get_feature_names()
    test_info = loader.get_test_info()
    
    # 加载或训练模型
    if model_path.exists() and not retrain:
        logger.info(f"  ✓ 加载已有模型: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        logger.info("  📍 训练新模型...")
        trainer = LGBMTrainer(config)
        model = trainer.train(X_train, y_train, X_valid, y_valid, feature_names=feature_names)
        
        # 保存模型
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"  💾 模型已保存: {model_path}")
    
    # 预测
    lgb_pred = model.predict(X_test)
    logger.info(f"  ✓ 预测完成: {len(lgb_pred):,} 条")
    
    # 清理训练数据
    del X_train, y_train, X_valid, y_valid, X_test
    loader.cleanup()
    gc.collect()
    
    return lgb_pred, y_test, test_info


def load_lgb_seed_ensemble(target_col: str, seeds: list[int], device: str = "cuda"):
    """
    加载多种子 LightGBM 模型并进行 Seed Ensemble（预测平均）
    
    Args:
        target_col: 目标列名
        seeds: 种子列表
        device: 设备（用于数据加载）
        
    Returns:
        lgb_pred: 平均后的预测值
        y_test: 测试集标签
        test_info: 测试集信息
    """
    from src.models.LBGM.config import LGBMConfig
    from src.models.LBGM.data_loader import DataLoader
    
    logger.info("=" * 60)
    logger.info("📊 LightGBM Seed Ensemble")
    logger.info("=" * 60)
    logger.info(f"  种子列表: {seeds}")
    
    model_dir = PROJECT_ROOT / "models" / "lgbm"
    
    # 加载数据（只需要一次）
    config = LGBMConfig.default()
    config.data.target_col = target_col
    
    loader = DataLoader(config.data, use_gpu=True)
    loader.load()
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = loader.split()
    test_info = loader.get_test_info()
    
    # 加载所有种子模型并预测
    all_predictions = []
    loaded_seeds = []
    
    for seed in seeds:
        model_path = model_dir / f"lgbm_{target_col}_seed{seed}.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            pred = model.predict(X_test)
            all_predictions.append(pred)
            loaded_seeds.append(seed)
            logger.info(f"  ✓ 已加载 seed={seed}: {model_path.name}")
    
    if not all_predictions:
        # 尝试加载默认模型
        default_path = model_dir / f"lgbm_{target_col}.pkl"
        if default_path.exists():
            logger.warning(f"  ⚠️ 未找到种子模型，使用默认模型: {default_path.name}")
            with open(default_path, "rb") as f:
                model = pickle.load(f)
            lgb_pred = model.predict(X_test)
        else:
            raise FileNotFoundError(f"未找到任何 LightGBM 模型文件")
    else:
        # 计算平均预测
        lgb_pred = np.mean(all_predictions, axis=0)
        logger.info(f"  ✓ Seed Ensemble 完成: {len(loaded_seeds)} 个模型")
        logger.info(f"  ✓ 使用的种子: {loaded_seeds}")
    
    logger.info(f"  ✓ 预测完成: {len(lgb_pred):,} 条")
    
    # 清理
    del X_train, y_train, X_valid, y_valid, X_test
    loader.cleanup()
    gc.collect()
    
    return lgb_pred, y_test, test_info


def load_or_train_gru(target_col: str, retrain: bool = False, device: str = "cuda"):
    """
    加载或训练 GRU 模型
    
    优先从 models/gru/model_config.json 读取模型配置，确保与训练参数一致
    """
    import json
    from src.models.deep.config import GRUDataConfig, GRUModelConfig
    from src.models.deep.dataset import prepare_data
    from src.models.deep.model import create_model
    from src.models.deep.train import TrainConfig, GRUTrainer, create_dataloader
    
    logger.info("=" * 60)
    logger.info("📊 GRU 模型")
    logger.info("=" * 60)
    
    model_dir = PROJECT_ROOT / "models" / "gru"
    model_path = model_dir / "best_model.pt"
    config_path = model_dir / "model_config.json"
    
    # 尝试加载模型配置
    if config_path.exists():
        logger.info(f"  ✓ 加载模型配置: {config_path}")
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        # 使用保存的配置
        feature_json_path = Path(saved_config.get('feature_json_path', PROJECT_ROOT / "models" / "lgbm" / "top50_features.json"))
        use_feature_selection = saved_config.get('use_feature_selection', True)
        window_size = saved_config.get('window_size', 20)
        
        # 模型配置
        hidden_dim = saved_config.get('hidden_dim', 32)
        num_layers = saved_config.get('num_layers', 1)
        dropout = saved_config.get('dropout', 0.5)
        mlp_hidden = saved_config.get('mlp_hidden', 32)
        use_embedding = saved_config.get('use_embedding', False)
        embedding_features = saved_config.get('embedding_features', [])
        
        logger.info(f"    hidden_dim: {hidden_dim}")
        logger.info(f"    dropout: {dropout}")
        logger.info(f"    mlp_hidden: {mlp_hidden}")
        logger.info(f"    use_embedding: {use_embedding}")
        if saved_config.get('best_valid_rank_ic'):
            logger.info(f"    训练时验证 RankIC: {saved_config['best_valid_rank_ic']:.4f}")
        if saved_config.get('test_rank_ic'):
            logger.info(f"    训练时测试 RankIC: {saved_config['test_rank_ic']:.4f}")
    else:
        # 使用默认配置
        logger.warning(f"  ⚠️ 未找到配置文件，使用默认配置")
        feature_json_path = PROJECT_ROOT / "models" / "lgbm" / "top50_features.json"
        use_feature_selection = True
        window_size = 20
        hidden_dim = 32
        num_layers = 1
        dropout = 0.6
        mlp_hidden = 32
        use_embedding = False
        embedding_features = []
    
    # 数据配置
    data_config = GRUDataConfig(
        target_col=target_col,
        window_size=window_size,
        use_gpu=True,
        use_feature_selection=use_feature_selection,
        feature_selection_json=feature_json_path,
        use_embedding=use_embedding,
        embedding_features=embedding_features if use_embedding else [],
    )
    
    # 准备数据
    train_dataset, valid_dataset, test_dataset, feature_cols = prepare_data(
        config=data_config,
        device=device,
    )
    
    # 创建模型配置
    model_config = GRUModelConfig(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        mlp_hidden=mlp_hidden,
        use_embedding=use_embedding,
        embedding_features=embedding_features if use_embedding else [],
    )
    model = create_model(config=model_config).to(device)
    
    # 加载或训练模型
    if model_path.exists() and not retrain:
        logger.info(f"  ✓ 加载已有模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_metric = checkpoint.get('best_valid_metric', 'N/A')
        if isinstance(best_metric, (int, float)):
            logger.info(f"    最佳验证 RankIC: {best_metric:.4f}")
        else:
            logger.info(f"    最佳验证 RankIC: {best_metric}")
    else:
        logger.info("  📍 训练新模型...")
        train_loader = create_dataloader(train_dataset, batch_size=2048, shuffle=True)
        valid_loader = create_dataloader(valid_dataset, batch_size=2048, shuffle=False)
        
        train_config = TrainConfig(
            epochs=50,
            batch_size=2048,
            learning_rate=1e-3,
            patience=10,
            use_amp=True,
        )
        
        trainer = GRUTrainer(model, train_config, device=device)
        trainer.train(train_loader, valid_loader)
        
        # 重新加载最佳模型
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    
    # 测试集预测
    test_loader = create_dataloader(test_dataset, batch_size=2048, shuffle=False)
    
    gru_preds = []
    gru_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            pred = model(x)
            gru_preds.append(pred.cpu().numpy())
            gru_labels.append(y.cpu().numpy())
    
    gru_pred = np.concatenate(gru_preds).flatten()
    gru_label = np.concatenate(gru_labels).flatten()
    
    logger.info(f"  ✓ 预测完成: {len(gru_pred):,} 条")
    
    # 获取测试集信息
    # GRU 测试集由于窗口化，日期信息在 dataset 中
    if hasattr(test_dataset, 'dates'):
        gru_dates = test_dataset.dates
        if hasattr(gru_dates, 'cpu'):  # torch tensor
            gru_dates = gru_dates.cpu().numpy()
        else:
            gru_dates = np.asarray(gru_dates)
    else:
        gru_dates = None
    
    gru_codes = test_dataset.codes if hasattr(test_dataset, 'codes') else None
    
    # 清理
    del train_dataset, valid_dataset, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return gru_pred, gru_label, {"dates": gru_dates, "codes": gru_codes}


def load_gru_seed_ensemble(target_col: str, seeds: list[int], device: str = "cuda"):
    """
    加载多种子 GRU 模型并进行 Seed Ensemble（预测平均）
    
    Args:
        target_col: 目标列名
        seeds: 种子列表
        device: 设备
        
    Returns:
        gru_pred: 平均后的预测值
        gru_label: 测试集标签
        gru_info: 测试集信息
    """
    import json
    from src.models.deep.config import GRUDataConfig, GRUModelConfig
    from src.models.deep.dataset import prepare_data
    from src.models.deep.model import create_model
    from src.models.deep.train import create_dataloader
    
    logger.info("=" * 60)
    logger.info("📊 GRU Seed Ensemble")
    logger.info("=" * 60)
    logger.info(f"  种子列表: {seeds}")
    
    model_dir = PROJECT_ROOT / "models" / "gru"
    
    # 首先从任意一个种子配置文件读取模型参数（所有种子应该使用相同配置）
    config_loaded = False
    for seed in seeds:
        config_path = model_dir / f"model_config_seed{seed}.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            config_loaded = True
            logger.info(f"  ✓ 加载配置: {config_path.name}")
            break
    
    if not config_loaded:
        # 尝试加载默认配置
        config_path = model_dir / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            logger.warning(f"  ⚠️ 使用默认配置: {config_path.name}")
        else:
            raise FileNotFoundError("未找到任何 GRU 模型配置文件")
    
    # 解析配置
    feature_json_path = Path(saved_config.get('feature_json_path', PROJECT_ROOT / "models" / "lgbm" / "top50_features.json"))
    use_feature_selection = saved_config.get('use_feature_selection', True)
    window_size = saved_config.get('window_size', 20)
    hidden_dim = saved_config.get('hidden_dim', 32)
    num_layers = saved_config.get('num_layers', 1)
    dropout = saved_config.get('dropout', 0.5)
    mlp_hidden = saved_config.get('mlp_hidden', 32)
    use_embedding = saved_config.get('use_embedding', False)
    embedding_features = saved_config.get('embedding_features', [])
    
    logger.info(f"    hidden_dim: {hidden_dim}")
    logger.info(f"    dropout: {dropout}")
    logger.info(f"    mlp_hidden: {mlp_hidden}")
    
    # 准备数据（只需要一次）
    data_config = GRUDataConfig(
        target_col=target_col,
        window_size=window_size,
        use_gpu=True,
        use_feature_selection=use_feature_selection,
        feature_selection_json=feature_json_path,
        use_embedding=use_embedding,
        embedding_features=embedding_features if use_embedding else [],
    )
    
    train_dataset, valid_dataset, test_dataset, feature_cols = prepare_data(
        config=data_config,
        device=device,
    )
    
    # 创建测试 DataLoader
    test_loader = create_dataloader(test_dataset, batch_size=2048, shuffle=False)
    
    # 加载所有种子模型并预测
    all_predictions = []
    loaded_seeds = []
    gru_labels = None
    
    for seed in seeds:
        model_path = model_dir / f"best_model_seed{seed}.pt"
        if model_path.exists():
            # 创建模型
            model_config = GRUModelConfig(
                input_dim=len(feature_cols),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                mlp_hidden=mlp_hidden,
                use_embedding=use_embedding,
                embedding_features=embedding_features if use_embedding else [],
            )
            model = create_model(config=model_config).to(device)
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            # 预测
            seed_preds = []
            seed_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch
                    pred = model(x)
                    seed_preds.append(pred.cpu().numpy())
                    seed_labels.append(y.cpu().numpy())
            
            pred = np.concatenate(seed_preds).flatten()
            all_predictions.append(pred)
            loaded_seeds.append(seed)
            
            # 保存标签（所有种子共用）
            if gru_labels is None:
                gru_labels = np.concatenate(seed_labels).flatten()
            
            logger.info(f"  ✓ 已加载 seed={seed}: {model_path.name}")
    
    if not all_predictions:
        # 尝试加载默认模型
        default_path = model_dir / "best_model.pt"
        if default_path.exists():
            logger.warning(f"  ⚠️ 未找到种子模型，使用默认模型: {default_path.name}")
            
            model_config = GRUModelConfig(
                input_dim=len(feature_cols),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                mlp_hidden=mlp_hidden,
                use_embedding=use_embedding,
                embedding_features=embedding_features if use_embedding else [],
            )
            model = create_model(config=model_config).to(device)
            
            checkpoint = torch.load(default_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            gru_preds = []
            gru_labels_list = []
            
            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch
                    pred = model(x)
                    gru_preds.append(pred.cpu().numpy())
                    gru_labels_list.append(y.cpu().numpy())
            
            gru_pred = np.concatenate(gru_preds).flatten()
            gru_labels = np.concatenate(gru_labels_list).flatten()
        else:
            raise FileNotFoundError(f"未找到任何 GRU 模型文件")
    else:
        # 计算平均预测
        gru_pred = np.mean(all_predictions, axis=0)
        logger.info(f"  ✓ Seed Ensemble 完成: {len(loaded_seeds)} 个模型")
        logger.info(f"  ✓ 使用的种子: {loaded_seeds}")
    
    logger.info(f"  ✓ 预测完成: {len(gru_pred):,} 条")
    
    # 获取测试集信息
    if hasattr(test_dataset, 'dates'):
        gru_dates = test_dataset.dates
        if hasattr(gru_dates, 'cpu'):
            gru_dates = gru_dates.cpu().numpy()
        else:
            gru_dates = np.asarray(gru_dates)
    else:
        gru_dates = None
    
    gru_codes = test_dataset.codes if hasattr(test_dataset, 'codes') else None
    
    # 清理
    del train_dataset, valid_dataset, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return gru_pred, gru_labels, {"dates": gru_dates, "codes": gru_codes}


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("🚀 模型融合流程")
    logger.info("=" * 60)
    logger.info(f"  目标: {args.target}")
    logger.info(f"  权重: LightGBM={args.lgb_weight}, GRU={args.gru_weight}")
    
    # 显示种子融合配置
    if args.lgb_seed_ensemble:
        logger.info(f"  LightGBM Seed Ensemble: 启用 (seeds={args.seeds})")
    if args.gru_seed_ensemble:
        logger.info(f"  GRU Seed Ensemble: 启用 (seeds={args.seeds})")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Step 1: LightGBM
    if args.lgb_seed_ensemble:
        lgb_pred, lgb_label, lgb_test_info = load_lgb_seed_ensemble(args.target, args.seeds, device)
    else:
        lgb_pred, lgb_label, lgb_test_info = load_or_train_lgb(args.target, args.retrain_lgb)
    lgb_label = np.asarray(lgb_label, dtype=np.float64)
    lgb_dates = lgb_test_info.get("dates")
    lgb_codes = lgb_test_info.get("codes")
    
    # Step 2: GRU
    if args.gru_seed_ensemble:
        gru_pred, gru_label, gru_test_info = load_gru_seed_ensemble(args.target, args.seeds, device)
    else:
        gru_pred, gru_label, gru_test_info = load_or_train_gru(args.target, args.retrain_gru, device)
    gru_label = np.asarray(gru_label, dtype=np.float64)
    gru_dates = gru_test_info.get("dates")
    
    # Step 3: 数据对齐
    logger.info("=" * 60)
    logger.info("📊 数据对齐")
    logger.info("=" * 60)
    logger.info(f"  LightGBM 测试集: {len(lgb_pred):,} 条")
    logger.info(f"  GRU 测试集:      {len(gru_pred):,} 条")
    
    # GRU 由于滑窗，测试集较小。取交集（按尾部对齐）
    n_lgb = len(lgb_pred)
    n_gru = len(gru_pred)
    
    if n_lgb > n_gru:
        offset = n_lgb - n_gru
        lgb_pred = lgb_pred[offset:]
        lgb_label = lgb_label[offset:]
        if lgb_dates is not None:
            lgb_dates = lgb_dates[offset:]
        if lgb_codes is not None:
            lgb_codes = lgb_codes[offset:]
        logger.info(f"  ✓ 对齐后: {len(lgb_pred):,} 条")
    elif n_gru > n_lgb:
        offset = n_gru - n_lgb
        gru_pred = gru_pred[offset:]
        gru_label = gru_label[offset:]
        if gru_dates is not None:
            gru_dates = gru_dates[offset:]
        logger.info(f"  ✓ 对齐后: {len(gru_pred):,} 条")
    
    # 使用 lgb_dates 作为日期（如果 gru_dates 不可用）
    dates = lgb_dates if lgb_dates is not None else gru_dates
    codes = lgb_codes
    
    # Step 4: 使用 ModelFusion 类进行融合
    from src.models.ensemble.fusion import FusionConfig, ModelFusion
    
    fusion_config = FusionConfig(
        lgb_weight=args.lgb_weight,
        gru_weight=args.gru_weight,
        rank_first=True,
        target_col=args.target,
    )
    
    fusion = ModelFusion(fusion_config)
    
    # 设置预测数据
    fusion.set_predictions(
        lgb_pred=lgb_pred,
        gru_pred=gru_pred,
        labels=lgb_label,  # 使用 LightGBM 的标签（两者应相同）
        dates=dates,
        codes=codes,
    )
    
    # Step 5: 相关性分析
    corr_metrics = fusion.analyze_correlation()
    
    # Step 6: 对比所有融合策略
    comparison_df = fusion.compare_all(
        static_weights=[
            (1.0, 0.0),   # 纯 LightGBM
            (0.0, 1.0),   # 纯 GRU
            (0.5, 0.5),   # 等权
            (0.6, 0.4),   # 默认
            (0.7, 0.3),   # LGB 偏重
            (0.8, 0.2),   # LGB 高权重
        ]
    )
    
    # Step 7: 生成报告
    logger.info("")
    logger.info("=" * 60)
    logger.info("💾 生成报告")
    logger.info("=" * 60)
    
    # 找出最佳策略
    best_row = comparison_df.iloc[0]
    
    report = f"""# 模型融合报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**预测目标**: {args.target}  
**测试集大小**: {len(lgb_pred):,} 条

---

## 1. 预测相关性

| 指标 | 值 |
|------|-----|
| Pearson 相关 | {corr_metrics['pearson_corr']:.4f} |
| Spearman 相关 | {corr_metrics['spearman_corr']:.4f} |

**结论**: {'✅ 相关性 < 0.7，融合价值高' if abs(corr_metrics['pearson_corr']) < 0.7 else '⚠️ 相关性 >= 0.7，融合增益有限'}

---

## 2. 融合策略对比

| 策略 | 日均 RankIC | ICIR | 胜率 |
|------|------------|------|------|
"""
    
    for _, row in comparison_df.iterrows():
        report += f"| {row['strategy']} | {row['daily_rank_ic_mean']:.4f} | {row['icir']:.4f} | {row['win_rate']:.2%} |\n"
    
    report += f"""
---

## 3. 最佳策略

**{best_row['strategy']}**

- 日均 RankIC: **{best_row['daily_rank_ic_mean']:.4f}**
- ICIR: **{best_row['icir']:.4f}**
- 胜率: **{best_row['win_rate']:.2%}**

---

## 4. 建议

"""
    
    # 添加建议
    lgb_icir = comparison_df[comparison_df['strategy'].str.contains('1.0L')].iloc[0]['icir']
    gru_icir = comparison_df[comparison_df['strategy'].str.contains('1.0G')].iloc[0]['icir']
    best_icir = best_row['icir']
    
    if best_icir > max(lgb_icir, gru_icir):
        report += f"✅ **融合有效**：最佳融合策略 ICIR ({best_icir:.4f}) 超过单模型最优 ({max(lgb_icir, gru_icir):.4f})\n"
    else:
        report += f"⚠️ **融合效果有限**：建议直接使用 {'LightGBM' if lgb_icir > gru_icir else 'GRU'} 单模型\n"
    
    if abs(corr_metrics['pearson_corr']) >= 0.8:
        report += "⚠️ 两模型预测高度相关，考虑引入其他类型模型（如 Transformer）增加多样性\n"
    
    # 保存报告
    report_path = PROJECT_ROOT / "reports" / "model_fusion_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"  ✓ 报告已保存: {report_path}")
    
    # 保存对比结果 CSV
    csv_path = PROJECT_ROOT / "reports" / "model_fusion_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"  ✓ CSV 已保存: {csv_path}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ 融合完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
