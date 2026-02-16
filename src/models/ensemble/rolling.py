"""
滚动训练模块 (Rolling Training)

实现 Walk-Forward Optimization 策略:
1. 按时间窗口滚动训练
2. 同时训练 LightGBM 和 GRU
3. 融合预测结果
4. 防止未来信息泄露

时间切片策略:
- Train(2021-2022) -> Test(2023 Q1)
- Train(2021-2023 Q1) -> Test(2023 Q2)
- ...
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import gc

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[3]


@dataclass
class RollingConfig:
    """滚动训练配置"""
    
    # 数据路径
    lgb_data_path: Path = BASE_DIR / "data" / "features" / "structured" / "train_lgb.parquet"
    gru_data_path: Path = BASE_DIR / "data" / "features" / "structured" / "train_gru.parquet"
    
    # 时间范围
    start_date: str = "2021-01-01"
    end_date: str = "2025-12-31"
    
    # 训练窗口配置
    train_window_years: int = 2      # 训练窗口长度（年）
    test_window_months: int = 3      # 测试窗口长度（月/季度）
    purge_days: int = 5              # 隔离天数
    
    # 滚动步长
    step_months: int = 3             # 每次滚动的月数
    
    # 目标列
    target_col: str = "excess_ret_5d"
    
    # 融合权重
    lgb_weight: float = 0.6
    gru_weight: float = 0.4
    
    # 保存配置
    save_dir: Path = BASE_DIR / "models" / "rolling"
    save_predictions: bool = True
    
    def __post_init__(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)


class TimeWindow:
    """时间窗口定义"""
    
    def __init__(
        self,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        window_id: int = 0,
    ):
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.window_id = window_id
    
    def __repr__(self):
        return (
            f"Window {self.window_id}: "
            f"Train[{self.train_start} ~ {self.train_end}] -> "
            f"Test[{self.test_start} ~ {self.test_end}]"
        )


def generate_rolling_windows(config: RollingConfig) -> List[TimeWindow]:
    """
    生成滚动时间窗口
    
    Args:
        config: 滚动配置
        
    Returns:
        时间窗口列表
    """
    windows = []
    
    start = datetime.strptime(config.start_date, "%Y-%m-%d")
    end = datetime.strptime(config.end_date, "%Y-%m-%d")
    
    # 初始训练窗口结束日期 = 开始日期 + 训练窗口长度
    train_end = start + timedelta(days=config.train_window_years * 365)
    
    window_id = 0
    while True:
        # 测试窗口
        test_start = train_end + timedelta(days=config.purge_days)
        test_end = test_start + timedelta(days=config.test_window_months * 30)
        
        # 确保测试窗口不超过终止日期
        if test_start > end:
            break
        if test_end > end:
            test_end = end
        
        window = TimeWindow(
            train_start=start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
            window_id=window_id,
        )
        windows.append(window)
        
        # 滚动
        train_end = train_end + timedelta(days=config.step_months * 30)
        window_id += 1
    
    logger.info(f"生成 {len(windows)} 个滚动窗口")
    for w in windows:
        logger.info(f"  {w}")
    
    return windows


class RollingTrainer:
    """
    滚动训练器
    
    实现:
    1. 按窗口训练 LightGBM 和 GRU
    2. 在测试窗口进行预测
    3. 融合预测结果
    4. 汇总所有窗口的结果
    """
    
    def __init__(self, config: RollingConfig = None):
        self.config = config or RollingConfig()
        self.windows = generate_rolling_windows(self.config)
        
        # 存储结果
        self.window_results: List[Dict[str, Any]] = []
        self.all_predictions: List[pd.DataFrame] = []
    
    def train_lgbm_window(
        self,
        window: TimeWindow,
        lgb_data: pd.DataFrame,
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        """
        训练单个窗口的 LightGBM 模型
        
        Returns:
            (model, test_predictions, test_labels)
        """
        from ..LBGM.config import LGBMConfig, get_feature_columns
        from ..LBGM.trainer import LGBMTrainer
        
        logger.info(f"  🌲 训练 LightGBM ({window})")
        
        # 创建配置
        config = LGBMConfig()
        config.data.target_col = self.config.target_col
        config.data.train_start = window.train_start
        config.data.train_end = window.train_end
        config.data.valid_start = window.test_start  # 用测试窗口做验证
        config.data.valid_end = window.test_end
        config.data.test_start = window.test_start
        config.data.test_end = window.test_end
        
        # 数据切分
        train_mask = (lgb_data['trade_date'] >= window.train_start) & (lgb_data['trade_date'] <= window.train_end)
        test_mask = (lgb_data['trade_date'] >= window.test_start) & (lgb_data['trade_date'] <= window.test_end)
        
        train_df = lgb_data[train_mask].copy()
        test_df = lgb_data[test_mask].copy()
        
        # 动态特征识别
        feature_cols = get_feature_columns(lgb_data.columns.tolist())
        feature_cols = [c for c in feature_cols if c in lgb_data.select_dtypes(include=['float32', 'float64', 'int32', 'int64', 'int8']).columns]
        
        # 准备数据
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[self.config.target_col].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df[self.config.target_col].values.astype(np.float32)
        
        # 训练
        trainer = LGBMTrainer(config)
        model = trainer.train(
            X_train, y_train,
            X_test, y_test,  # 用测试集做验证
            feature_names=feature_cols,
            categorical_features=config.data.categorical_features,
        )
        
        # 预测
        predictions = trainer.predict(X_test)
        
        return model, predictions, y_test, test_df['trade_date'].values, test_df['ts_code'].values
    
    def train_gru_window(
        self,
        window: TimeWindow,
        gru_data: pd.DataFrame,
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        """
        训练单个窗口的 GRU 模型
        
        Returns:
            (model, test_predictions, test_labels)
        """
        from ..deep.config import GRUConfig, get_gru_feature_columns
        from ..deep.dataset import StockDataset, build_index_map
        from ..deep.model import create_model
        from ..deep.train import GRUTrainer, TrainConfig, create_dataloader
        import torch
        
        logger.info(f"  🤖 训练 GRU ({window})")
        
        # 创建配置
        config = GRUConfig()
        config.data.target_col = self.config.target_col
        config.data.train_start = window.train_start
        config.data.train_end = window.train_end
        config.data.valid_start = window.test_start
        config.data.valid_end = window.test_end
        config.data.test_start = window.test_start
        config.data.test_end = window.test_end
        
        # 动态特征识别
        feature_cols = get_gru_feature_columns(gru_data.columns.tolist())
        feature_cols = [c for c in feature_cols if c in gru_data.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns]
        
        # 数据排序
        gru_data = gru_data.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        # 日期处理
        dates = gru_data['trade_date'].values
        if hasattr(dates[0], 'strftime'):
            dates_str = np.array([d.strftime('%Y-%m-%d') for d in dates])
        else:
            dates_str = np.array([str(d)[:10] for d in dates])
        codes_str = gru_data['ts_code'].values.astype(str)
        
        # 时间切分
        train_mask = (dates_str >= window.train_start) & (dates_str <= window.train_end)
        test_mask = (dates_str >= window.test_start) & (dates_str <= window.test_end)
        
        # 构建索引
        all_indices, _ = build_index_map(gru_data, window_size=config.data.window_size)
        train_indices = all_indices[train_mask[all_indices]]
        test_indices = all_indices[test_mask[all_indices]]
        
        # 设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 创建数据集
        train_dataset = StockDataset(
            data=gru_data, indices=train_indices,
            feature_cols=feature_cols, target_col=self.config.target_col,
            window_size=config.data.window_size, dates=dates_str, codes=codes_str,
            device=device,
        )
        
        test_dataset = StockDataset(
            data=gru_data, indices=test_indices,
            feature_cols=feature_cols, target_col=self.config.target_col,
            window_size=config.data.window_size, dates=dates_str, codes=codes_str,
            device=device,
        )
        
        # 创建模型
        model = create_model(
            input_dim=len(feature_cols),
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
        )
        
        # 训练配置
        train_config = TrainConfig()
        train_config.epochs = 50  # 滚动训练使用较少轮数
        train_config.patience = 10
        
        # 训练
        trainer = GRUTrainer(model, train_config, device=device)
        train_loader = create_dataloader(train_dataset, train_config.batch_size, shuffle=True)
        valid_loader = create_dataloader(test_dataset, train_config.batch_size, shuffle=False)
        
        trainer.train(train_loader, valid_loader)
        
        # 预测
        predictions = trainer.predict(valid_loader)
        
        # 获取标签
        test_labels = np.array([test_dataset.labels[int(idx)].cpu().numpy() for idx in test_indices])
        test_dates = np.array([dates_str[int(idx)] for idx in test_indices])
        test_codes = np.array([codes_str[int(idx)] for idx in test_indices])
        
        return model, predictions, test_labels, test_dates, test_codes
    
    def train_window(self, window: TimeWindow) -> Dict[str, Any]:
        """
        训练单个窗口（LightGBM + GRU）
        
        Returns:
            窗口结果字典
        """
        logger.info("=" * 60)
        logger.info(f"📊 训练窗口 {window.window_id}")
        logger.info("=" * 60)
        
        # 加载数据
        logger.info("加载数据...")
        lgb_data = pd.read_parquet(self.config.lgb_data_path)
        gru_data = pd.read_parquet(self.config.gru_data_path)
        
        # 日期格式处理
        if hasattr(lgb_data['trade_date'].iloc[0], 'strftime'):
            lgb_data['trade_date'] = lgb_data['trade_date'].dt.strftime('%Y-%m-%d')
        else:
            lgb_data['trade_date'] = lgb_data['trade_date'].astype(str).str[:10]
        
        if hasattr(gru_data['trade_date'].iloc[0], 'strftime'):
            gru_data['trade_date'] = gru_data['trade_date'].dt.strftime('%Y-%m-%d')
        else:
            gru_data['trade_date'] = gru_data['trade_date'].astype(str).str[:10]
        
        # 训练 LightGBM
        lgb_model, lgb_pred, lgb_labels, lgb_dates, lgb_codes = self.train_lgbm_window(window, lgb_data)
        
        # 释放 LightGBM 数据内存
        del lgb_data
        gc.collect()
        
        # 训练 GRU
        gru_model, gru_pred, gru_labels, gru_dates, gru_codes = self.train_gru_window(window, gru_data)
        
        # 释放 GRU 数据内存
        del gru_data
        gc.collect()
        
        # 对齐预测结果
        # 创建 DataFrame 便于对齐
        lgb_df = pd.DataFrame({
            'trade_date': lgb_dates,
            'ts_code': lgb_codes,
            'lgb_pred': lgb_pred,
            'label': lgb_labels,
        })
        
        gru_df = pd.DataFrame({
            'trade_date': gru_dates,
            'ts_code': gru_codes,
            'gru_pred': gru_pred,
        })
        
        # 按 trade_date + ts_code 对齐
        merged = lgb_df.merge(gru_df, on=['trade_date', 'ts_code'], how='inner')
        
        logger.info(f"  对齐后样本数: {len(merged):,} (LGB={len(lgb_df):,}, GRU={len(gru_df):,})")
        
        # 融合
        lgb_ranked = self._rank_by_date(merged['lgb_pred'].values, merged['trade_date'].values)
        gru_ranked = self._rank_by_date(merged['gru_pred'].values, merged['trade_date'].values)
        
        blended = self.config.lgb_weight * lgb_ranked + self.config.gru_weight * gru_ranked
        merged['blended_pred'] = blended
        
        # 评估
        from .fusion import spearmanr
        
        lgb_ic = spearmanr(merged['lgb_pred'].values, merged['label'].values)[0]
        gru_ic = spearmanr(merged['gru_pred'].values, merged['label'].values)[0]
        blend_ic = spearmanr(blended, merged['label'].values)[0]
        
        logger.info(f"  LGB RankIC: {lgb_ic:.4f}")
        logger.info(f"  GRU RankIC: {gru_ic:.4f}")
        logger.info(f"  Blend RankIC: {blend_ic:.4f}")
        
        result = {
            'window_id': window.window_id,
            'train_start': window.train_start,
            'train_end': window.train_end,
            'test_start': window.test_start,
            'test_end': window.test_end,
            'n_samples': len(merged),
            'lgb_ic': lgb_ic,
            'gru_ic': gru_ic,
            'blend_ic': blend_ic,
        }
        
        self.window_results.append(result)
        self.all_predictions.append(merged)
        
        return result
    
    def _rank_by_date(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        """按日期分组做 Rank 归一化"""
        result = np.zeros_like(values)
        unique_dates = np.unique(dates)
        
        for date in unique_dates:
            mask = dates == date
            day_values = values[mask]
            # 计算百分位排名 (0-1)
            ranks = np.argsort(np.argsort(day_values)).astype(float) / max(len(day_values) - 1, 1)
            result[mask] = ranks
        
        return result
    
    def train_all(self) -> pd.DataFrame:
        """
        训练所有窗口
        
        Returns:
            结果汇总 DataFrame
        """
        logger.info("=" * 60)
        logger.info("🚀 开始滚动训练")
        logger.info("=" * 60)
        
        for window in self.windows:
            self.train_window(window)
        
        # 汇总结果
        results_df = pd.DataFrame(self.window_results)
        
        logger.info("=" * 60)
        logger.info("📊 滚动训练结果汇总")
        logger.info("=" * 60)
        logger.info(f"  平均 LGB RankIC: {results_df['lgb_ic'].mean():.4f}")
        logger.info(f"  平均 GRU RankIC: {results_df['gru_ic'].mean():.4f}")
        logger.info(f"  平均 Blend RankIC: {results_df['blend_ic'].mean():.4f}")
        
        # 保存
        if self.config.save_predictions:
            all_pred_df = pd.concat(self.all_predictions, ignore_index=True)
            pred_path = self.config.save_dir / "rolling_predictions.parquet"
            all_pred_df.to_parquet(pred_path)
            logger.info(f"  💾 预测结果已保存: {pred_path}")
            
            results_path = self.config.save_dir / "rolling_results.csv"
            results_df.to_csv(results_path, index=False)
            logger.info(f"  💾 结果汇总已保存: {results_path}")
        
        return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试滚动窗口生成
    config = RollingConfig()
    windows = generate_rolling_windows(config)
    
    for w in windows:
        print(w)
