"""
重新采集修复后的数据类型
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data_pipeline.collectors.structured.expectations import (
    EarningsForecastCollector,
    ConsensusForecastCollector,
    ForecastRevisionCollector,
)

from data_pipeline.collectors.structured.deep_risk_quality import (
    StockGoodwillCollector,
    BreakNetStockCollector,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据保存路径
RAW_DATA_DIR = project_root / 'data' / 'raw' / 'structured'

# 时间范围：2024年全年
START_DATE = '20240101'
END_DATE = '20241231'

# 样本股票：选取5只测试
TEST_STOCKS = [
    '600519.SH',  # 贵州茅台
    '000858.SZ',  # 五粮液
    '300750.SZ',  # 宁德时代
    '600036.SH',  # 招商银行
    '601318.SH',  # 中国平安
]


def test_fixed_collectors():
    """测试修复后的采集器"""
    
    print("=" * 80)
    print("测试修复后的采集器")
    print("=" * 80)
    
    # 1. 测试业绩预告
    print("\n【1】测试业绩预告采集器...")
    try:
        collector = EarningsForecastCollector()
        df = collector.collect(start_date=START_DATE, end_date=END_DATE)
        print(f"  采集到 {len(df)} 条数据")
        if not df.empty:
            print(f"  列：{list(df.columns)}")
            null_ratios = df.isnull().sum() / len(df)
            high_null_cols = null_ratios[null_ratios > 0.9]
            if not high_null_cols.empty:
                print(f"  警告：高空列 ({len(high_null_cols)}/{len(df.columns)}): {list(high_null_cols.index)}")
            else:
                print("  ✓ 无高空列")
            print(f"  样例数据:\n{df.head(2)}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 2. 测试一致预期（一个股票）
    print("\n【2】测试一致预期采集器...")
    try:
        collector = ConsensusForecastCollector()
        df = collector.collect(ts_code=TEST_STOCKS[0], year='2024')
        print(f"  采集到 {len(df)} 条数据")
        if not df.empty:
            print(f"  列：{list(df.columns)}")
            null_ratios = df.isnull().sum() / len(df)
            high_null_cols = null_ratios[null_ratios > 0.9]
            if not high_null_cols.empty:
                print(f"  警告：高空列 ({len(high_null_cols)}/{len(df.columns)}): {list(high_null_cols.index)}")
            else:
                print("  ✓ 无高空列")
            print(f"  样例数据:\n{df.head(2)}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 3. 测试预测修正（一个股票）
    print("\n【3】测试预测修正采集器...")
    try:
        collector = ForecastRevisionCollector()
        df = collector.collect(ts_code=TEST_STOCKS[0])
        print(f"  采集到 {len(df)} 条数据")
        if not df.empty:
            print(f"  列：{list(df.columns)}")
            null_ratios = df.isnull().sum() / len(df)
            high_null_cols = null_ratios[null_ratios > 0.9]
            if not high_null_cols.empty:
                print(f"  警告：高空列 ({len(high_null_cols)}/{len(df.columns)}): {list(high_null_cols.index)}")
            else:
                print("  ✓ 无高空列")
            print(f"  样例数据:\n{df.head(2)}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 4. 测试商誉明细
    print("\n【4】测试商誉明细采集器...")
    try:
        collector = StockGoodwillCollector()
        df = collector.collect(ts_code=TEST_STOCKS[0], period='20241231')
        print(f"  采集到 {len(df)} 条数据")
        if not df.empty:
            print(f"  列：{list(df.columns)}")
            null_ratios = df.isnull().sum() / len(df)
            high_null_cols = null_ratios[null_ratios > 0.9]
            if not high_null_cols.empty:
                print(f"  警告：高空列 ({len(high_null_cols)}/{len(df.columns)}): {list(high_null_cols.index)}")
            else:
                print("  ✓ 无高空列")
            print(f"  样例数据:\n{df.head(2)}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 5. 测试破净股统计
    print("\n【5】测试破净股统计采集器...")
    try:
        collector = BreakNetStockCollector()
        df = collector.collect(date=END_DATE)
        print(f"  采集到 {len(df)} 条数据")
        if not df.empty:
            print(f"  列：{list(df.columns)}")
            null_ratios = df.isnull().sum() / len(df)
            high_null_cols = null_ratios[null_ratios > 0.9]
            if not high_null_cols.empty:
                print(f"  警告：高空列 ({len(high_null_cols)}/{len(df.columns)}): {list(high_null_cols.index)}")
            else:
                print("  ✓ 无高空列")
            print(f"  样例数据:\n{df.head(2)}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    print("\n" + "=" * 80)
    print("测试完成！请检查上述结果，确认修复是否成功。")
    print("=" * 80)


if __name__ == '__main__':
    test_fixed_collectors()
