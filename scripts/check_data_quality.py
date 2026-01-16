"""
检查采集数据的质量
"""
import pandas as pd
from pathlib import Path

# 数据路径
base_dir = Path('data/raw/structured')

# 检查预期与预测分析域
print("=" * 80)
print("预期与预测分析域 - 数据质量检查")
print("=" * 80)

files = [
    'earnings_forecast.csv',
    'consensus_forecast.csv',
    'forecast_revision.csv',
]

for filename in files:
    filepath = base_dir / 'expectations_forecasts' / filename
    if filepath.exists():
        print(f"\n【{filename}】")
        df = pd.read_csv(filepath)
        print(f"Shape: {df.shape}")
        print(f"\nColumns with >90% null:")
        null_ratios = df.isnull().sum() / len(df)
        high_null_cols = null_ratios[null_ratios > 0.9]
        for col, ratio in high_null_cols.items():
            print(f"  {col}: {ratio:.1%}")

# 检查深度风险与质量因子域
print("\n" + "=" * 80)
print("深度风险与质量因子域 - 数据质量检查")
print("=" * 80)

files = [
    'esg_refinitiv.csv',
]

for filename in files:
    filepath = base_dir / 'deep_risk_quality' / filename
    if filepath.exists():
        print(f"\n【{filename}】")
        df = pd.read_csv(filepath)
        print(f"Shape: {df.shape}")
        print(f"\nColumns with >90% null:")
        null_ratios = df.isnull().sum() / len(df)
        high_null_cols = null_ratios[null_ratios > 0.9]
        for col, ratio in high_null_cols.items():
            print(f"  {col}: {ratio:.1%}")
        print(f"\nSample data (first 5 rows):")
        print(df.head())
