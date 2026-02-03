"""
模型融合 (Model Ensemble)

实现 LightGBM + GRU 的融合策略

核心思路:
1. LightGBM 擅长截面估值挖掘
2. GRU 擅长时序量价规律
3. 两者互补，融合能降低方差、提升 ICIR
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def pearsonr(x, y):
    """计算 Pearson 相关系数"""
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n < 2:
        return np.nan, 1.0
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)
    
    if x_std == 0 or y_std == 0:
        return np.nan, 1.0
    
    cov = np.sum((x - x_mean) * (y - y_mean)) / (n - 1)
    r = cov / (x_std * y_std)
    
    # 简化的 p 值计算
    t = r * np.sqrt((n - 2) / (1 - r**2 + 1e-10))
    p = 2 * (1 - min(0.999, abs(t) / np.sqrt(n)))  # 近似
    
    return r, p


def spearmanr(x, y):
    """计算 Spearman 相关系数"""
    x = np.asarray(x)
    y = np.asarray(y)
    
    # 计算排名
    x_ranks = np.argsort(np.argsort(x)).astype(float) + 1
    y_ranks = np.argsort(np.argsort(y)).astype(float) + 1
    
    return pearsonr(x_ranks, y_ranks)


def rankdata(x):
    """计算排名"""
    x = np.asarray(x)
    return np.argsort(np.argsort(x)).astype(float) + 1


@dataclass
class FusionConfig:
    """融合配置"""
    
    # 数据路径
    data_path: Path = field(default_factory=lambda: Path("data/features/structured/train.parquet"))
    
    # LightGBM 权重
    lgb_weight: float = 0.6
    
    # GRU 权重
    gru_weight: float = 0.4
    
    # 动态加权窗口
    dynamic_window: int = 20
    
    # 是否先 Rank 化
    rank_first: bool = True
    
    # 目标列
    target_col: str = "ret_5d"


class ModelFusion:
    """
    模型融合器
    
    支持:
    - 静态加权融合
    - 动态加权融合
    - 相关性分析
    """
    
    def __init__(self, config: FusionConfig = None):
        self.config = config or FusionConfig()
        self.lgb_predictions: Optional[np.ndarray] = None
        self.gru_predictions: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        self.test_dates: Optional[np.ndarray] = None
        self.test_codes: Optional[np.ndarray] = None
    
    def set_predictions(
        self,
        lgb_pred: np.ndarray,
        gru_pred: np.ndarray,
        labels: np.ndarray,
        dates: np.ndarray,
        codes: np.ndarray = None
    ):
        """
        设置预测结果
        
        Args:
            lgb_pred: LightGBM 预测值
            gru_pred: GRU 预测值
            labels: 真实标签
            dates: 日期
            codes: 股票代码 (可选)
        """
        assert len(lgb_pred) == len(gru_pred) == len(labels) == len(dates), \
            f"长度不匹配: lgb={len(lgb_pred)}, gru={len(gru_pred)}, labels={len(labels)}, dates={len(dates)}"
        
        self.lgb_predictions = lgb_pred
        self.gru_predictions = gru_pred
        self.test_labels = labels
        self.test_dates = dates
        self.test_codes = codes
        
        logger.info(f"✓ 预测数据已加载: {len(lgb_pred)} 条记录")
    
    def analyze_correlation(self) -> Dict[str, float]:
        """
        分析两个模型预测值的相关性
        
        Returns:
            相关性指标
        """
        if self.lgb_predictions is None or self.gru_predictions is None:
            raise ValueError("请先调用 set_predictions() 设置预测值")
        
        logger.info("=" * 60)
        logger.info("📊 模型相关性分析")
        logger.info("=" * 60)
        
        # Pearson 相关系数
        pearson_corr, pearson_p = pearsonr(self.lgb_predictions, self.gru_predictions)
        
        # Spearman 相关系数 (Rank 相关)
        spearman_corr, spearman_p = spearmanr(self.lgb_predictions, self.gru_predictions)
        
        logger.info(f"  Pearson 相关系数: {pearson_corr:.4f} (p={pearson_p:.2e})")
        logger.info(f"  Spearman 相关系数: {spearman_corr:.4f} (p={spearman_p:.2e})")
        
        # 判断融合价值
        if abs(pearson_corr) < 0.5:
            logger.info("  ✅ 相关性 < 0.5，模型高度互补，融合价值极大！")
        elif abs(pearson_corr) < 0.7:
            logger.info("  ✓ 相关性 < 0.7，模型有一定互补性，融合有价值")
        elif abs(pearson_corr) < 0.9:
            logger.info("  ⚠️ 相关性 < 0.9，模型部分同质，融合价值有限")
        else:
            logger.warning("  ❌ 相关性 > 0.9，模型高度同质，融合几乎无意义")
        
        return {
            "pearson_corr": pearson_corr,
            "pearson_pvalue": pearson_p,
            "spearman_corr": spearman_corr,
            "spearman_pvalue": spearman_p,
        }
    
    def _rank_transform(self, arr: np.ndarray, dates: np.ndarray) -> np.ndarray:
        """
        按日期分组做 Rank 归一化
        
        Args:
            arr: 原始预测值
            dates: 日期
            
        Returns:
            Rank 归一化后的值 (0-1 范围)
        """
        result = np.zeros_like(arr)
        unique_dates = np.unique(dates)
        
        for date in unique_dates:
            mask = dates == date
            values = arr[mask]
            # 计算百分位排名 (0-1)
            ranks = rankdata(values) / len(values)
            result[mask] = ranks
        
        return result
    
    def static_blend(
        self,
        lgb_weight: float = None,
        gru_weight: float = None,
        rank_first: bool = None
    ) -> np.ndarray:
        """
        静态加权融合
        
        Args:
            lgb_weight: LightGBM 权重
            gru_weight: GRU 权重
            rank_first: 是否先做 Rank 归一化
            
        Returns:
            融合后的预测值
        """
        lgb_weight = lgb_weight or self.config.lgb_weight
        gru_weight = gru_weight or self.config.gru_weight
        rank_first = rank_first if rank_first is not None else self.config.rank_first
        
        if self.lgb_predictions is None or self.gru_predictions is None:
            raise ValueError("请先调用 set_predictions() 设置预测值")
        
        logger.info("=" * 60)
        logger.info(f"📊 静态加权融合 (LGB={lgb_weight:.2f}, GRU={gru_weight:.2f})")
        logger.info("=" * 60)
        
        lgb_pred = self.lgb_predictions
        gru_pred = self.gru_predictions
        
        if rank_first:
            logger.info("  ✓ 先做 Rank 归一化...")
            lgb_pred = self._rank_transform(lgb_pred, self.test_dates)
            gru_pred = self._rank_transform(gru_pred, self.test_dates)
        
        # 加权融合
        blended = lgb_weight * lgb_pred + gru_weight * gru_pred
        
        logger.info(f"  ✓ 融合完成，输出形状: {blended.shape}")
        
        return blended
    
    def dynamic_blend(self, window: int = None) -> np.ndarray:
        """
        动态加权融合
        
        根据近 N 天的表现动态调整权重
        
        Args:
            window: 回看窗口
            
        Returns:
            融合后的预测值
        """
        window = window or self.config.dynamic_window
        
        if self.lgb_predictions is None or self.gru_predictions is None:
            raise ValueError("请先调用 set_predictions() 设置预测值")
        
        logger.info("=" * 60)
        logger.info(f"📊 动态加权融合 (window={window})")
        logger.info("=" * 60)
        
        # 先做 Rank 归一化
        lgb_ranked = self._rank_transform(self.lgb_predictions, self.test_dates)
        gru_ranked = self._rank_transform(self.gru_predictions, self.test_dates)
        
        unique_dates = np.unique(self.test_dates)
        unique_dates.sort()
        
        result = np.zeros_like(self.lgb_predictions)
        
        # 计算每日 IC
        daily_lgb_ic = {}
        daily_gru_ic = {}
        
        for date in unique_dates:
            mask = self.test_dates == date
            lgb_ic = spearmanr(lgb_ranked[mask], self.test_labels[mask])[0]
            gru_ic = spearmanr(gru_ranked[mask], self.test_labels[mask])[0]
            daily_lgb_ic[date] = lgb_ic if not np.isnan(lgb_ic) else 0
            daily_gru_ic[date] = gru_ic if not np.isnan(gru_ic) else 0
        
        # 动态加权
        for i, date in enumerate(unique_dates):
            mask = self.test_dates == date
            
            # 获取历史 IC
            start_idx = max(0, i - window)
            hist_dates = unique_dates[start_idx:i]
            
            if len(hist_dates) == 0:
                # 没有历史，使用默认权重
                lgb_w = self.config.lgb_weight
                gru_w = self.config.gru_weight
            else:
                # 根据历史 IC 计算权重
                lgb_ics = [daily_lgb_ic[d] for d in hist_dates]
                gru_ics = [daily_gru_ic[d] for d in hist_dates]
                
                avg_lgb_ic = np.mean(lgb_ics)
                avg_gru_ic = np.mean(gru_ics)
                
                # 避免负权重
                avg_lgb_ic = max(avg_lgb_ic, 0.001)
                avg_gru_ic = max(avg_gru_ic, 0.001)
                
                total = avg_lgb_ic + avg_gru_ic
                lgb_w = avg_lgb_ic / total
                gru_w = avg_gru_ic / total
            
            # 加权融合
            result[mask] = lgb_w * lgb_ranked[mask] + gru_w * gru_ranked[mask]
        
        logger.info(f"  ✓ 动态融合完成")
        
        return result
    
    def evaluate(
        self,
        predictions: np.ndarray,
        name: str = "Ensemble"
    ) -> Dict[str, float]:
        """
        评估预测结果
        
        Args:
            predictions: 预测值
            name: 模型名称
            
        Returns:
            评估指标
        """
        logger.info("=" * 60)
        logger.info(f"📊 {name} 评估结果")
        logger.info("=" * 60)
        
        unique_dates = np.unique(self.test_dates)
        unique_dates.sort()
        
        # 每日 RankIC
        daily_rank_ics = []
        for date in unique_dates:
            mask = self.test_dates == date
            if mask.sum() < 10:
                continue
            
            pred_day = predictions[mask]
            label_day = self.test_labels[mask]
            
            rank_ic, _ = spearmanr(pred_day, label_day)
            if not np.isnan(rank_ic):
                daily_rank_ics.append(rank_ic)
        
        daily_rank_ics = np.array(daily_rank_ics)
        
        # 整体 RankIC
        overall_rank_ic = spearmanr(predictions, self.test_labels)[0]
        
        # 统计指标
        mean_ic = np.mean(daily_rank_ics)
        std_ic = np.std(daily_rank_ics)
        icir = mean_ic / std_ic if std_ic > 0 else 0
        win_rate = np.mean(daily_rank_ics > 0)
        
        logger.info(f"  📋 整体 RankIC: {overall_rank_ic:.4f}")
        logger.info(f"  📋 日均 RankIC: {mean_ic:.4f} ± {std_ic:.4f}")
        logger.info(f"  📋 ICIR: {icir:.4f}")
        logger.info(f"  📋 胜率: {win_rate:.2%}")
        logger.info(f"  📋 交易日数: {len(daily_rank_ics)}")
        
        # 判断效果
        if icir > 0.5:
            logger.info("  ✅ ICIR > 0.5，策略非常稳定！")
        elif icir > 0.3:
            logger.info("  ✓ ICIR > 0.3，策略较稳定")
        else:
            logger.warning("  ⚠️ ICIR < 0.3，策略稳定性欠佳")
        
        return {
            "overall_rank_ic": overall_rank_ic,
            "daily_rank_ic_mean": mean_ic,
            "daily_rank_ic_std": std_ic,
            "icir": icir,
            "win_rate": win_rate,
            "n_days": len(daily_rank_ics),
        }
    
    def compare_all(
        self,
        static_weights: List[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        对比所有融合策略
        
        Args:
            static_weights: 静态权重列表 [(lgb_w, gru_w), ...]
            
        Returns:
            对比结果 DataFrame
        """
        if static_weights is None:
            static_weights = [
                (1.0, 0.0),  # 纯 LGB
                (0.0, 1.0),  # 纯 GRU
                (0.6, 0.4),  # 默认
                (0.5, 0.5),  # 等权
                (0.7, 0.3),  # LGB 偏重
                (0.4, 0.6),  # GRU 偏重
            ]
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 融合策略对比")
        logger.info("=" * 70)
        
        results = []
        
        # 静态融合
        for lgb_w, gru_w in static_weights:
            name = f"Static ({lgb_w:.1f}L+{gru_w:.1f}G)"
            pred = self.static_blend(lgb_w, gru_w)
            metrics = self.evaluate(pred, name)
            results.append({
                "strategy": name,
                **metrics
            })
        
        # 动态融合
        dynamic_pred = self.dynamic_blend()
        dynamic_metrics = self.evaluate(dynamic_pred, "Dynamic")
        results.append({
            "strategy": "Dynamic",
            **dynamic_metrics
        })
        
        # 汇总
        df = pd.DataFrame(results)
        df = df.sort_values("icir", ascending=False)
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 策略排名 (按 ICIR)")
        logger.info("=" * 70)
        
        for i, row in df.iterrows():
            logger.info(
                f"  {row['strategy']:<25} | "
                f"RankIC={row['daily_rank_ic_mean']:.4f} | "
                f"ICIR={row['icir']:.4f} | "
                f"胜率={row['win_rate']:.2%}"
            )
        
        return df


def run_fusion_pipeline(
    data_path: Path = None,
    target_col: str = "ret_5d"
) -> Tuple[ModelFusion, pd.DataFrame]:
    """
    运行完整的融合流程
    
    注意: 这个函数需要先训练 LightGBM 和 GRU 模型
    
    Args:
        data_path: 数据路径
        target_col: 目标列
        
    Returns:
        (fusion, results_df)
    """
    from pathlib import Path
    import sys
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(PROJECT_ROOT))
    
    logger.info("=" * 70)
    logger.info("🚀 开始模型融合流程")
    logger.info("=" * 70)
    
    # 默认数据路径
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "features" / "structured" / "train.parquet"
    
    config = FusionConfig(
        data_path=data_path,
        target_col=target_col,
    )
    
    fusion = ModelFusion(config)
    
    # ========== Step 1: 训练 LightGBM ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("📍 Step 1: 训练 LightGBM")
    logger.info("=" * 70)
    
    from src.models.LBGM import LGBMConfig, DataLoader, LGBMTrainer
    
    lgb_config = LGBMConfig.default()
    lgb_config.data.target_col = target_col
    
    data_loader = DataLoader(lgb_config.data, use_gpu=True)
    data_loader.load()
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = data_loader.split()
    feature_names = data_loader.get_feature_names()
    test_info = data_loader.get_test_info()
    
    trainer = LGBMTrainer(lgb_config)
    model_lgb = trainer.train(X_train, y_train, X_valid, y_valid, feature_names=feature_names)
    
    # LightGBM 预测
    lgb_pred = model_lgb.predict(X_test)
    
    # 释放训练数据
    del X_train, y_train, X_valid, y_valid
    data_loader.cleanup()
    
    logger.info(f"  ✓ LightGBM 预测完成: {len(lgb_pred)} 条")
    
    # ========== Step 2: 训练 GRU ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("📍 Step 2: 训练 GRU")
    logger.info("=" * 70)
    
    from src.models.deep.dataset import prepare_deep_learning_data
    from src.models.deep.model import create_model
    from src.models.deep.train import TrainConfig, GRUTrainer, create_dataloader
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 准备 GRU 数据
    (train_dataset, valid_dataset, test_dataset), feature_cols = prepare_deep_learning_data(
        data_path=data_path,
        target_col=target_col,
        window_size=20,
        device=device,
    )
    
    train_loader = create_dataloader(train_dataset, batch_size=2048, shuffle=True)
    valid_loader = create_dataloader(valid_dataset, batch_size=2048, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=2048, shuffle=False)
    
    # 创建模型
    model_gru = create_model(
        input_dim=len(feature_cols),
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
    ).to(device)
    
    # 训练
    train_config = TrainConfig(
        epochs=30,
        batch_size=2048,
        learning_rate=1e-3,
        patience=10,
        use_amp=True,
    )
    
    gru_trainer = GRUTrainer(model_gru, train_config, device=device)
    gru_trainer.train(train_loader, valid_loader)
    
    # GRU 预测
    gru_trainer.load_model("best_model.pt")
    model_gru.eval()
    
    gru_preds = []
    with torch.no_grad():
        for batch in test_loader:
            x, _ = batch
            pred = model_gru(x)
            gru_preds.append(pred.cpu().numpy())
    
    gru_pred = np.concatenate(gru_preds)
    
    logger.info(f"  ✓ GRU 预测完成: {len(gru_pred)} 条")
    
    # ========== Step 3: 对齐预测结果 ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("📍 Step 3: 对齐预测结果")
    logger.info("=" * 70)
    
    # 注意: LightGBM 和 GRU 的测试集可能不完全对齐
    # LightGBM 使用全部测试集，GRU 需要滑动窗口（前 window-1 天没有预测）
    
    # 取交集
    n_lgb = len(lgb_pred)
    n_gru = len(gru_pred)
    
    logger.info(f"  LightGBM 预测数: {n_lgb}")
    logger.info(f"  GRU 预测数: {n_gru}")
    
    # GRU 窗口造成的偏移
    # 使用 GRU 的长度作为基准，LightGBM 截取后 n_gru 条
    if n_lgb > n_gru:
        offset = n_lgb - n_gru
        lgb_pred = lgb_pred[offset:]
        y_test = y_test[offset:]
        test_dates = test_info['dates'][offset:]
        test_codes = test_info['codes'][offset:] if 'codes' in test_info else None
        logger.info(f"  ✓ 对齐后: {len(lgb_pred)} 条")
    else:
        test_dates = test_info['dates']
        test_codes = test_info.get('codes')
    
    # ========== Step 4: 融合与评估 ==========
    fusion.set_predictions(
        lgb_pred=lgb_pred,
        gru_pred=gru_pred,
        labels=y_test,
        dates=test_dates,
        codes=test_codes,
    )
    
    # 相关性分析
    fusion.analyze_correlation()
    
    # 对比所有策略
    results_df = fusion.compare_all()
    
    return fusion, results_df
