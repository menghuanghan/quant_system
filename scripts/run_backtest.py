#!/usr/bin/env python3
"""
回测验证脚本

使用融合模型预测进行回测，验证实际收益

使用方法:
    python scripts/run_backtest.py                   # 默认回测测试集(2025年)
    python scripts/run_backtest.py --top-k 100      # 持仓100只
    python scripts/run_backtest.py --start-date 20240101  # 指定起始日期

输出:
    reports/backtest_analysis.md    # 分析报告
    plots/nav_curve.png             # 净值曲线
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
        logging.FileHandler(PROJECT_ROOT / "logs" / "backtest.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="回测验证")
    
    parser.add_argument("--top-k", type=int, default=50,
                        help="持仓数量 (默认: 50)")
    parser.add_argument("--cost-rate", type=float, default=0.003,
                        help="双边交易成本 (默认: 0.003 千三)")
    parser.add_argument("--capital", type=float, default=10_000_000,
                        help="初始资金 (默认: 10,000,000)")
    parser.add_argument("--start-date", type=str, default="20250101",
                        help="回测起始日期 (默认: 20250101)")
    parser.add_argument("--end-date", type=str, default="20251231",
                        help="回测结束日期 (默认: 20251231)")
    parser.add_argument("--lgb-weight", type=float, default=0.5,
                        help="LightGBM 权重 (默认: 0.5)")
    parser.add_argument("--gru-weight", type=float, default=0.5,
                        help="GRU 权重 (默认: 0.5)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="禁用 GPU")
    
    return parser.parse_args()


def generate_predictions(lgb_weight: float, gru_weight: float, device: str = "cuda"):
    """
    生成融合预测
    
    Returns:
        pred_df: 包含 trade_date, ts_code, score 的 DataFrame
    """
    import pickle
    
    # 自定义 rankdata 函数（避免 scipy 导入问题）
    def rankdata(x):
        """简单的 rank 实现"""
        temp = x.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(x))
        return ranks + 1
    
    logger.info("=" * 60)
    logger.info("📊 生成融合预测")
    logger.info("=" * 60)
    
    # ========== 1. 加载 LightGBM 预测 ==========
    logger.info("  📍 加载 LightGBM 模型...")
    
    from src.models.LBGM import LGBMConfig, DataLoader
    
    model_path = PROJECT_ROOT / "models" / "lgbm" / "lgbm_ret_5d.pkl"
    with open(model_path, "rb") as f:
        lgb_model = pickle.load(f)
    
    config = LGBMConfig.default()
    data_loader = DataLoader(config.data, use_gpu=True)
    data_loader.load()
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = data_loader.split()
    test_info = data_loader.get_test_info()
    
    lgb_pred = lgb_model.predict(X_test)
    
    # 创建 LightGBM 预测 DataFrame
    lgb_df = pd.DataFrame({
        'trade_date': test_info['dates'],
        'ts_code': test_info['codes'],
        'lgb_pred': lgb_pred,
    })
    
    del X_train, y_train, X_valid, y_valid, X_test
    data_loader.cleanup()
    gc.collect()
    
    logger.info(f"    LightGBM 预测: {len(lgb_df):,} 条")
    
    # ========== 2. 加载 GRU 预测 ==========
    logger.info("  📍 加载 GRU 模型...")
    
    from src.models.deep.dataset import prepare_data, DataConfig
    from src.models.deep.model import create_model
    from src.models.deep.train import create_dataloader
    
    data_config = DataConfig(
        target_col="ret_5d",
        window_size=20,
        use_gpu=True,
    )
    
    train_dataset, valid_dataset, test_dataset, feature_cols = prepare_data(
        config=data_config,
        device=device,
    )
    
    test_loader = create_dataloader(test_dataset, batch_size=2048, shuffle=False)
    
    model = create_model(
        input_dim=len(feature_cols),
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
    ).to(device)
    
    model_path = PROJECT_ROOT / "models" / "gru" / "best_model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    gru_preds = []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            pred = model(x)
            gru_preds.append(pred.cpu().numpy())
    
    gru_pred = np.concatenate(gru_preds)
    gru_dates = test_dataset.get_all_dates()
    gru_codes = test_dataset.get_all_codes()
    
    # 创建 GRU 预测 DataFrame
    gru_df = pd.DataFrame({
        'trade_date': gru_dates,
        'ts_code': gru_codes,
        'gru_pred': gru_pred,
    })
    
    logger.info(f"    GRU 预测: {len(gru_df):,} 条")
    
    # ========== 3. 对齐并融合 ==========
    logger.info("  📍 对齐并融合预测...")
    
    # 去重
    lgb_df = lgb_df.drop_duplicates(subset=['trade_date', 'ts_code'], keep='first')
    gru_df = gru_df.drop_duplicates(subset=['trade_date', 'ts_code'], keep='first')
    
    # 合并
    merged_df = pd.merge(lgb_df, gru_df, on=['trade_date', 'ts_code'], how='inner')
    
    # Rank 归一化
    def rank_normalize(x):
        # 自定义 rank：处理 pandas Series
        arr = x.values
        temp = arr.argsort()
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(arr))
        return (ranks + 1) / len(arr)
    
    # 按日归一化
    merged_df['lgb_rank'] = merged_df.groupby('trade_date')['lgb_pred'].transform(rank_normalize)
    merged_df['gru_rank'] = merged_df.groupby('trade_date')['gru_pred'].transform(rank_normalize)
    
    # 加权融合
    merged_df['score'] = lgb_weight * merged_df['lgb_rank'] + gru_weight * merged_df['gru_rank']
    
    # 只保留需要的列
    pred_df = merged_df[['trade_date', 'ts_code', 'score']].copy()
    
    logger.info(f"    融合预测: {len(pred_df):,} 条")
    
    return pred_df


def load_market_data():
    """加载行情数据"""
    logger.info("=" * 60)
    logger.info("📊 加载行情数据")
    logger.info("=" * 60)
    
    try:
        import cudf
        use_gpu = True
        _pd = cudf
        logger.info("  🚀 使用 cuDF 加速")
    except ImportError:
        use_gpu = False
        _pd = pd
    
    dwd_path = PROJECT_ROOT / "data" / "processed" / "structured" / "dwd"
    
    # 价格数据
    logger.info("  📍 加载价格数据...")
    price_df = _pd.read_parquet(dwd_path / "dwd_stock_price.parquet")
    price_cols = ['trade_date', 'ts_code', 'close', 'return_1d', 'is_trading']
    price_df = price_df[price_cols]
    
    # 状态数据
    logger.info("  📍 加载状态数据...")
    status_df = _pd.read_parquet(dwd_path / "dwd_stock_status.parquet")
    status_cols = ['trade_date', 'ts_code', 'is_limit_up', 'is_limit_down']
    status_df = status_df[status_cols]
    
    # 合并
    logger.info("  📍 合并数据...")
    market_df = price_df.merge(status_df, on=['trade_date', 'ts_code'], how='left')
    
    # 填充缺失值
    market_df['is_limit_up'] = market_df['is_limit_up'].fillna(0)
    market_df['is_limit_down'] = market_df['is_limit_down'].fillna(0)
    market_df['is_trading'] = market_df['is_trading'].fillna(1)
    
    if use_gpu:
        market_df = market_df.to_pandas()
    
    logger.info(f"  ✓ 行情数据: {len(market_df):,} 行")
    
    return market_df


def run_backtest(
    pred_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: dict
) -> "BacktestResult":
    """运行回测"""
    from src.backtesting.engine import VectorBacktester, BacktestConfig
    
    bt_config = BacktestConfig(
        top_k=config['top_k'],
        cost_rate=config['cost_rate'],
        capital=config['capital'],
        use_gpu=not config['no_gpu'],
    )
    
    backtester = VectorBacktester(bt_config)
    
    # 筛选日期范围
    start_date = pd.to_datetime(config['start_date'], format='%Y%m%d')
    end_date = pd.to_datetime(config['end_date'], format='%Y%m%d')
    
    # 确保日期格式一致
    pred_df['trade_date'] = pd.to_datetime(pred_df['trade_date'])
    market_df['trade_date'] = pd.to_datetime(market_df['trade_date'])
    
    pred_df = pred_df[
        (pred_df['trade_date'] >= start_date) &
        (pred_df['trade_date'] <= end_date)
    ]
    
    market_df = market_df[
        (market_df['trade_date'] >= start_date) &
        (market_df['trade_date'] <= end_date)
    ]
    
    logger.info(f"  📅 回测区间: {start_date.date()} ~ {end_date.date()}")
    logger.info(f"  📊 预测数据: {len(pred_df):,} 条")
    logger.info(f"  📊 行情数据: {len(market_df):,} 条")
    
    result = backtester.run(pred_df, market_df)
    
    return result


def generate_report(result: "BacktestResult", config: dict):
    """生成回测报告"""
    logger.info("=" * 60)
    logger.info("📝 生成回测报告")
    logger.info("=" * 60)
    
    m = result.metrics
    
    report = f"""# 回测分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 回测配置

| 参数 | 值 |
|------|-----|
| 持仓数量 | {config['top_k']} |
| 交易成本 | {config['cost_rate']:.4f} (双边) |
| 初始资金 | {config['capital']:,.0f} |
| 融合权重 | LGB={config['lgb_weight']}, GRU={config['gru_weight']} |
| 回测区间 | {config['start_date']} ~ {config['end_date']} |

## 2. 收益指标

| 指标 | 值 |
|------|-----|
| 年化收益率 | {m.get('annual_return', 0):.2%} |
| 累计收益率 | {m.get('total_return', 0):.2%} |
| 最大回撤 | {m.get('max_drawdown', 0):.2%} |

## 3. 风险指标

| 指标 | 值 |
|------|-----|
| 年化波动率 | {m.get('annual_volatility', 0):.2%} |
| 夏普比率 | {m.get('sharpe_ratio', 0):.4f} |
| Calmar比率 | {m.get('calmar_ratio', 0):.4f} |

## 4. 交易统计

| 指标 | 值 |
|------|-----|
| 交易日数 | {m.get('trading_days', 0)} |
| 胜率 | {m.get('win_rate', 0):.2%} |

## 5. 结论

"""
    
    # 判断结果
    sharpe = m.get('sharpe_ratio', 0)
    if sharpe > 4:
        report += "✅ **夏普比率 > 4**，策略表现优异！与 ICIR 0.99 的预期一致。\n"
    elif sharpe > 2:
        report += "✅ **夏普比率 > 2**，策略表现良好。\n"
    elif sharpe > 1:
        report += "⚠️ **夏普比率在 1-2 之间**，策略有效但还有提升空间。\n"
    else:
        report += "❌ **夏普比率 < 1**，策略需要进一步优化。\n"
    
    # 保存报告
    report_path = PROJECT_ROOT / "reports" / "backtest_analysis.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"  ✓ 报告已保存: {report_path}")
    
    return report


def plot_nav_curve(result: "BacktestResult"):
    """绘制净值曲线"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        plt.figure(figsize=(12, 6))
        
        nav = result.nav_series
        nav.index = pd.to_datetime(nav.index)
        
        plt.plot(nav.index, nav.values / nav.values[0], label='Strategy NAV', linewidth=2)
        
        plt.xlabel('Date')
        plt.ylabel('NAV (normalized)')
        plt.title('Backtest Net Asset Value Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 格式化 x 轴
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        plot_path = PROJECT_ROOT / "reports" / "nav_curve.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        logger.info(f"  ✓ 净值曲线已保存: {plot_path}")
        
    except ImportError:
        logger.warning("  ⚠️ matplotlib 未安装，跳过绘图")


def main():
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("🚀 回测验证")
    logger.info(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    logger.info(f"  📋 设备: {device}")
    logger.info(f"  📋 持仓: {args.top_k}")
    logger.info(f"  📋 成本: {args.cost_rate}")
    logger.info(f"  📋 融合: LGB={args.lgb_weight}, GRU={args.gru_weight}")
    
    config = {
        'top_k': args.top_k,
        'cost_rate': args.cost_rate,
        'capital': args.capital,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'lgb_weight': args.lgb_weight,
        'gru_weight': args.gru_weight,
        'no_gpu': args.no_gpu,
    }
    
    # 1. 生成预测
    pred_df = generate_predictions(args.lgb_weight, args.gru_weight, device)
    
    # 清理显存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. 加载行情数据
    market_df = load_market_data()
    
    # 3. 运行回测
    result = run_backtest(pred_df, market_df, config)
    
    # 4. 输出结果
    print(result.summary())
    
    # 5. 生成报告
    generate_report(result, config)
    
    # 6. 绘制净值曲线
    plot_nav_curve(result)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎉 回测完成!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
