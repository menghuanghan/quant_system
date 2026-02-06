#!/usr/bin/env python
"""
高级标签生成模块测试

验证以下标签生成功能：
1. 超额收益标签 (excess_ret_5d, excess_ret_10d)
2. 截面排名标签 (rank_ret_5d, rank_ret_10d)
3. 夏普标签 (sharpe_5d, sharpe_10d)
4. 分位数分类标签 (label_bin_5d)
"""

import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_benchmark_data():
    """加载基准指数数据"""
    index_dir = Path("/home/menghuanghan/quant_system/data/raw/structured/market_data/index_daily")
    
    # 加载沪深300
    hs300_file = index_dir / "000300_SH.parquet"
    if hs300_file.exists():
        hs300 = pd.read_parquet(hs300_file)
        logger.info(f"  ✓ 加载沪深300: {len(hs300):,} 行")
        return hs300
    else:
        logger.warning(f"  ⚠️ 找不到沪深300数据: {hs300_file}")
        return None


def load_sample_stock_data(n_stocks: int = 50, n_days: int = 500):
    """加载样例股票数据"""
    stock_dir = Path("/home/menghuanghan/quant_system/data/raw/structured/market_data/stock_daily")
    
    stock_files = list(stock_dir.glob("*.parquet"))[:n_stocks]
    
    dfs = []
    for f in stock_files:
        df = pd.read_parquet(f)
        # 取最近 n_days 天
        df = df.sort_values('trade_date').tail(n_days)
        dfs.append(df)
    
    if not dfs:
        logger.warning("  ⚠️ 未找到股票数据")
        return None
    
    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"  ✓ 加载 {len(stock_files)} 只股票, 共 {len(result):,} 行")
    
    return result


def test_advanced_labels():
    """测试高级标签生成"""
    logger.info("=" * 60)
    logger.info("🧪 高级标签生成测试")
    logger.info("=" * 60)
    
    # 1. 加载数据
    logger.info("\n📊 Step 1: 加载数据")
    
    benchmark = load_benchmark_data()
    stocks = load_sample_stock_data(n_stocks=30, n_days=300)
    
    if stocks is None:
        logger.error("无法加载股票数据，测试终止")
        return False
    
    # 2. 准备基础标签（模拟 LabelGenerator 已生成的基础标签）
    logger.info("\n📊 Step 2: 准备基础收益率标签")
    
    stocks = stocks.sort_values(['ts_code', 'trade_date'])
    
    # 确定价格列
    price_col = 'close_hfq' if 'close_hfq' in stocks.columns else 'close'
    logger.info(f"  使用价格列: {price_col}")
    
    # 生成基础收益率标签
    for days in [5, 10, 20]:
        stocks[f'ret_{days}d'] = stocks.groupby('ts_code')[price_col].transform(
            lambda x: x.shift(-days) / x - 1
        )
    
    # 添加日收益率（用于夏普标签）
    stocks['return_1d'] = stocks.groupby('ts_code')[price_col].transform(
        lambda x: x.pct_change()
    )
    
    logger.info(f"  ✓ 生成 ret_5d, ret_10d, ret_20d, return_1d")
    
    # 3. 测试 AdvancedLabelGenerator
    logger.info("\n📊 Step 3: 测试 AdvancedLabelGenerator")
    
    from src.feature_engineering.structured.labels import AdvancedLabelGenerator
    from dataclasses import dataclass, field
    from typing import List
    
    @dataclass
    class MockLabelConfig:
        # 超额收益
        generate_excess_return: bool = True
        benchmark_code: str = "000300.SH"
        excess_return_days: List[int] = field(default_factory=lambda: [5, 10])
        # 截面排名
        generate_rank_labels: bool = True
        rank_label_days: List[int] = field(default_factory=lambda: [5, 10])
        # 夏普标签
        generate_sharpe_labels: bool = True
        sharpe_label_days: List[int] = field(default_factory=lambda: [5, 10])
        # 分位数分类
        generate_bin_labels: bool = True
        bin_label_days: List[int] = field(default_factory=lambda: [5])
        bin_top_pct: float = 0.30
        bin_bottom_pct: float = 0.30
    
    config = MockLabelConfig()
    
    # 准备参考数据
    ref_data = {}
    if benchmark is not None:
        ref_data['benchmark'] = benchmark
    
    # 初始化生成器
    gen = AdvancedLabelGenerator(
        config=config,
        ref_data=ref_data,
        use_gpu=False  # 测试时用 pandas
    )
    
    # 生成高级标签
    result = gen.generate_advanced_labels(stocks)
    
    # 4. 验证结果
    logger.info("\n📊 Step 4: 验证结果")
    
    success = True
    
    # 检查超额收益标签
    for days in [5, 10]:
        col = f'excess_ret_{days}d'
        if col in result.columns:
            valid = result[col].notna().sum()
            mean = result[col].mean()
            logger.info(f"  ✓ {col}: 有效 {valid:,} 行, 均值 {mean:.4f}")
        else:
            logger.warning(f"  ⚠️ 缺少 {col}")
            success = False
    
    # 检查截面排名标签
    for days in [5, 10]:
        col = f'rank_ret_{days}d'
        if col in result.columns:
            valid = result[col].notna().sum()
            mean = result[col].mean()
            # 排名均值应接近 0.5
            logger.info(f"  ✓ {col}: 有效 {valid:,} 行, 均值 {mean:.4f} (期望约0.5)")
            if abs(mean - 0.5) > 0.1:
                logger.warning(f"    ⚠️ 均值偏离 0.5 较大")
        else:
            logger.warning(f"  ⚠️ 缺少 {col}")
            success = False
    
    # 检查夏普标签
    for days in [5, 10]:
        col = f'sharpe_{days}d'
        if col in result.columns:
            valid = result[col].notna().sum()
            mean = result[col].mean()
            logger.info(f"  ✓ {col}: 有效 {valid:,} 行, 均值 {mean:.4f}")
        else:
            logger.warning(f"  ⚠️ 缺少 {col}")
            success = False
    
    # 检查分位数分类标签
    col = 'label_bin_5d'
    if col in result.columns:
        valid = result[col].notna()
        dist = result.loc[valid, col].value_counts().sort_index()
        total = dist.sum()
        logger.info(f"  ✓ {col} 分布:")
        for label, count in dist.items():
            label_name = {0: "做空", 1: "观望", 2: "做多"}.get(int(label), str(label))
            logger.info(f"    - {label_name}: {count:,} ({count/total:.1%})")
    else:
        logger.warning(f"  ⚠️ 缺少 {col}")
        success = False
    
    # 5. 输出样例数据
    logger.info("\n📊 Step 5: 样例数据")
    
    label_cols = [
        'ts_code', 'trade_date', 'ret_5d',
        'excess_ret_5d', 'rank_ret_5d', 'sharpe_5d', 'label_bin_5d'
    ]
    label_cols = [c for c in label_cols if c in result.columns]
    
    sample = result[label_cols].dropna().head(10)
    logger.info(f"\n{sample.to_string()}")
    
    # 6. 总结
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✅ 所有高级标签生成测试通过")
    else:
        logger.warning("⚠️ 部分测试未通过")
    logger.info("=" * 60)
    
    return success


if __name__ == "__main__":
    success = test_advanced_labels()
    sys.exit(0 if success else 1)
