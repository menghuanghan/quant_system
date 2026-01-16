"""
测试AkShare数据接口，查看实际返回的字段
"""
import akshare as ak
import pandas as pd

print("=" * 80)
print("测试 AkShare 数据接口")
print("=" * 80)

# 1. 业绩预告
print("\n1. 业绩预告 (stock_yjyg_em)")
try:
    df = ak.stock_yjyg_em(date='20241231')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n样例数据:")
    print(df.head(3))
except Exception as e:
    print(f"Error: {e}")

# 2. 一致预期
print("\n\n2. 一致预期 (stock_analyst_rank_em)")
try:
    df = ak.stock_analyst_rank_em(symbol="600519")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n样例数据:")
    print(df.head(3))
except Exception as e:
    print(f"Error: {e}")

# 3. 盈利预测修正
print("\n\n3. 盈利预测修正 (stock_profit_forecast_em)")
try:
    df = ak.stock_profit_forecast_em(symbol="贵州茅台")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n样例数据:")
    print(df.head(3))
except Exception as e:
    print(f"Error: {e}")

# 4. 商誉明细
print("\n\n4. 商誉明细 (stock_sy_em)")
try:
    df = ak.stock_sy_em(date="20241231")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n样例数据:")
    print(df.head(3))
except Exception as e:
    print(f"Error: {e}")

# 5. 破净股
print("\n\n5. 破净股 (stock_zh_a_hist_min_em)")
try:
    df = ak.stock_pb_em()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n样例数据:")
    print(df.head(3))
except Exception as e:
    print(f"Error: {e}")
