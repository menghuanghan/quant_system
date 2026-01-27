import pandas as pd
import os
import random

fund_adj_dir = 'data/raw/structured/derivatives/fund_adj'
all_files = os.listdir(fund_adj_dir)

print(f"基金复权因子文件总数: {len(all_files)}\n")

# 随机抽取10个文件检查
sample_files = random.sample(all_files, min(10, len(all_files)))

print("随机抽样检查10个基金:")
print("=" * 80)

for f in sample_files:
    df = pd.read_parquet(os.path.join(fund_adj_dir, f))
    unique_values = df['adj_factor'].unique()
    
    print(f"\n{f}")
    print(f"  数据形状: {df.shape}")
    print(f"  adj_factor唯一值数量: {len(unique_values)}")
    print(f"  adj_factor最小值: {df['adj_factor'].min():.4f}")
    print(f"  adj_factor最大值: {df['adj_factor'].max():.4f}")
    print(f"  前5个值: {sorted(unique_values)[:5]}")

# 统计所有基金的情况
print("\n" + "=" * 80)
print("整体统计:")
print("=" * 80)

all_only_one = 0
all_has_changes = 0

for f in all_files:
    df = pd.read_parquet(os.path.join(fund_adj_dir, f))
    if df['adj_factor'].nunique() == 1:
        all_only_one += 1
    else:
        all_has_changes += 1

print(f"复权因子始终为1的基金数: {all_only_one}")
print(f"复权因子有变化的基金数: {all_has_changes}")
print(f"比例: {all_has_changes/len(all_files)*100:.1f}% 的基金有复权因子变化")
