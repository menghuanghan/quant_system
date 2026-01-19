"""
政策数据质量分析报告

分析近一年采集的政策数据
"""

import pandas as pd
from pathlib import Path

# 读取数据
data_file = Path("data/raw/unstructured/policy/csv/policy_yearly_20260119.csv")
df = pd.read_csv(data_file)

print("="*60)
print("政策数据质量分析报告")
print("="*60)
print(f"\n数据文件: {data_file}")
print(f"总记录数: {len(df)}")

print("\n" + "="*60)
print("字段信息")
print("="*60)
print(df.info())

print("\n" + "="*60)
print("数据统计")
print("="*60)

# 标题长度分析
title_len = df['title'].str.len()
print(f"\n标题长度分布:")
print(f"  平均: {title_len.mean():.1f} 字符")
print(f"  中位数: {title_len.median():.1f} 字符")
print(f"  最短: {title_len.min()} 字符")
print(f"  最长: {title_len.max()} 字符")

# URL完整性
url_valid = df['url'].notna() & (df['url'] != '')
print(f"\nURL完整性:")
print(f"  有效URL: {url_valid.sum()} 条 ({url_valid.sum()/len(df)*100:.1f}%)")

# 发文字号提取
doc_no_valid = df['doc_no'].notna() & (df['doc_no'] != '')
print(f"\n发文字号提取:")
print(f"  已提取: {doc_no_valid.sum()} 条 ({doc_no_valid.sum()/len(df)*100:.1f}%)")
print(f"  未提取: {(~doc_no_valid).sum()} 条")

# 按来源统计
print(f"\n按来源统计:")
source_stats = df.groupby(['source', 'source_dept']).size().sort_values(ascending=False)
for (source, dept), count in source_stats.items():
    print(f"  - {dept} ({source}): {count} 条")

# 重复检查
duplicates = df.duplicated(subset=['title']).sum()
print(f"\n重复标题: {duplicates} 条")

# 各来源样例
print("\n" + "="*60)
print("政策样例")
print("="*60)

for source in df['source'].unique():
    source_df = df[df['source'] == source]
    dept = source_df['source_dept'].iloc[0]
    print(f"\n{dept} ({source}) - 共 {len(source_df)} 条:")
    for i, row in source_df.head(5).iterrows():
        print(f"  {i+1}. {row['title'][:60]}...")
        if row['doc_no']:
            print(f"     发文字号: {row['doc_no']}")

print("\n" + "="*60)
print("数据文件保存位置")
print("="*60)
print(f"CSV: {data_file}")
print(f"JSONL: {data_file.parent.parent / 'meta' / 'policy_yearly_20260119.jsonl'}")

print("\n分析完成！")
