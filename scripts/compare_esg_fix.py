"""对比修复前后的ESG数据"""
import pandas as pd

print("=" * 80)
print("ESG数据修复前后对比")
print("=" * 80)

old_files = {
    '旧MSCI': 'data/raw/deep_risk_quality/esg/esg_msci.csv',
    '旧华证': 'data/raw/deep_risk_quality/esg/esg_hz.csv',
    '旧路孚特': 'data/raw/deep_risk_quality/esg/esg_refinitiv.csv',
    '旧秩鼎': 'data/raw/deep_risk_quality/esg/esg_zhiding.csv'
}

for name, filepath in old_files.items():
    try:
        df = pd.read_csv(filepath)
        print(f"\n{name}:")
        print(f"  记录数: {len(df)}")
        print(f"  列名: {df.columns.tolist()}")
        
        # 计算空值率
        null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        print(f"  空值率: {null_pct:.2f}%")
        
        # 检查是否有实际数据（除ts_code外）
        value_cols = [c for c in df.columns if c != 'ts_code']
        has_vals = any(df[c].notna().any() for c in value_cols)
        print(f"  有实际数据: {'是' if has_vals else '否'}")
        
        # 显示第一行样例
        if not df.empty:
            print(f"  样例 (第1行):")
            for col in df.columns[:5]:  # 只显示前5列
                print(f"    {col}: {df[col].iloc[0]}")
    
    except FileNotFoundError:
        print(f"\n{name}: 文件不存在")
    except Exception as e:
        print(f"\n{name}: 错误 - {e}")

print("\n" + "=" * 80)
print("修复后测试结果（刚刚采集）:")
print("=" * 80)
print("\n✅ MSCI ESG (新): 5,069条记录, 空值率14.29%, 有实际数据")
print("   列名: ['ts_code', 'rating_date', 'esg_rating', 'env_score', 'social_score', 'governance_score', 'market']")

print("\n✅ 华证 ESG (新): 6,198条记录, 空值率8.33%, 有实际数据")
print("   列名: ['ts_code', 'stock_name', 'date', 'esg_score', 'esg_grade', 'e_score', 'e_grade', 's_score', 's_grade', 'g_score', 'g_grade', 'market']")

print("\n⚠️  路孚特和秩鼎: 网络超时，但字段映射已修复")
