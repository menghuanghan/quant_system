"""
修复政策数据中的发文字号提取

从标题中提取发文字号并更新CSV文件
"""

import re
import pandas as pd
from pathlib import Path

def extract_doc_no(text: str) -> str:
    """增强的发文字号提取"""
    if not text:
        return ''
    
    patterns = [
        # 括号中的发文字号：(发改环资〔2025〕1751号)
        r'\(([^()]+〔\d{4}〕\d+号)\)',
        r'\(([^()]+\[\d{4}\]\d+号)\)',
        # 直接的发文字号：证监发〔2024〕1号
        r'(证监[发办函]\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        r'(国[发办函]\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        r'([^()]+[发办函]\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
        # 公告格式
        r'(\d{4}\s*年\s*第?\s*\d+\s*号\s*公告)',
        # 令格式
        r'([\u4e00-\u9fa5]+令\s*第?\s*\d+\s*号)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            doc_no = match.group(1)
            # 清理空格
            doc_no = re.sub(r'\s+', '', doc_no)
            return doc_no
    
    return ''

# 读取数据
data_file = Path("data/raw/unstructured/policy/csv/policy_yearly_20260119.csv")
df = pd.read_csv(data_file)

print("="*60)
print("修复发文字号提取")
print("="*60)
print(f"原始记录数: {len(df)}")

# 提取发文字号
print("\n提取发文字号...")
df['doc_no'] = df['title'].apply(extract_doc_no)

# 统计
doc_no_valid = df['doc_no'].notna() & (df['doc_no'] != '')
print(f"已提取发文字号: {doc_no_valid.sum()} 条 ({doc_no_valid.sum()/len(df)*100:.1f}%)")

# 显示样例
print("\n发文字号样例:")
samples = df[doc_no_valid][['source_dept', 'doc_no', 'title']].head(20)
for _, row in samples.iterrows():
    print(f"  [{row['source_dept']}] {row['doc_no']}")
    print(f"    {row['title'][:60]}...")

# 保存更新后的数据
df.to_csv(data_file, index=False, encoding='utf-8-sig')
print(f"\n数据已更新: {data_file}")

# 更新JSONL
jsonl_file = data_file.parent.parent / 'meta' / 'policy_yearly_20260119.jsonl'
with open(jsonl_file, 'w', encoding='utf-8') as f:
    import json
    for record in df.to_dict('records'):
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"元数据已更新: {jsonl_file}")

# 按来源统计发文字号提取率
print("\n" + "="*60)
print("各来源发文字号提取率")
print("="*60)
for source in df['source'].unique():
    source_df = df[df['source'] == source]
    has_doc_no = (source_df['doc_no'] != '') & source_df['doc_no'].notna()
    dept = source_df['source_dept'].iloc[0]
    print(f"{dept} ({source}): {has_doc_no.sum()}/{len(source_df)} ({has_doc_no.sum()/len(source_df)*100:.1f}%)")

print("\n修复完成！")
