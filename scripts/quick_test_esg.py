"""
验证ESG采集器修复 - 快速验证脚本
只采集少量数据验证字段映射是否正确
"""
import sys
import os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_pipeline.collectors.structured.deep_risk_quality.esg_ratings import (
    get_esg_msci,
    get_esg_hz,
    get_esg_refinitiv,
    get_esg_zhiding
)

def quick_test_esg_collectors():
    """快速测试ESG采集器"""
    
    print("=" * 80)
    print("ESG采集器快速验证测试")
    print("=" * 80)
    
    collectors = {
        'MSCI ESG': get_esg_msci,
        '华证 ESG': get_esg_hz,
        '路孚特 ESG': get_esg_refinitiv,
        '秩鼎 ESG': get_esg_zhiding,
    }
    
    results = {}
    
    for name, collector_func in collectors.items():
        print(f"\n测试 {name} 采集器...")
        try:
            df = collector_func()
            
            if df.empty:
                print(f"  ⚠️  未采集到数据")
                results[name] = {'status': 'empty', 'records': 0, 'columns': []}
                continue
            
            # 取前3条记录
            sample = df.head(3)
            
            # 检查空字段
            null_counts = sample.isnull().sum()
            total_cells = len(sample) * len(sample.columns)
            null_cells = null_counts.sum()
            null_percentage = (null_cells / total_cells) * 100
            
            # 检查是否有实际数据 (排除 ts_code)
            value_columns = [col for col in sample.columns if col != 'ts_code']
            has_values = False
            for col in value_columns:
                if sample[col].notna().any():
                    has_values = True
                    break
            
            print(f"  ✅ 成功采集: {len(df)} 条记录")
            print(f"  📊 列名: {sample.columns.tolist()}")
            print(f"  📈 空值比例: {null_percentage:.2f}%")
            print(f"  🔍 是否有实际数据: {'是' if has_values else '否'}")
            
            # 显示样例数据
            print(f"\n  样例数据 (前3条):")
            print("  " + "=" * 76)
            for idx, row in sample.iterrows():
                print(f"  记录 {idx + 1}:")
                for col in sample.columns:
                    val = row[col]
                    if pd.isna(val):
                        val_str = "NULL"
                    else:
                        val_str = str(val)[:30]  # 截断长字符串
                    print(f"    {col:20s}: {val_str}")
                print()
            
            results[name] = {
                'status': 'success',
                'records': len(df),
                'columns': sample.columns.tolist(),
                'null_percentage': null_percentage,
                'has_values': has_values
            }
            
        except Exception as e:
            print(f"  ❌ 采集失败: {str(e)}")
            results[name] = {'status': 'error', 'error': str(e)}
    
    # 汇总报告
    print("\n" + "=" * 80)
    print("验证结果汇总")
    print("=" * 80)
    
    for name, result in results.items():
        status_icon = {
            'success': '✅',
            'empty': '⚠️ ',
            'error': '❌'
        }.get(result['status'], '❓')
        
        print(f"\n{status_icon} {name}:")
        print(f"  状态: {result['status']}")
        
        if result['status'] == 'success':
            print(f"  记录数: {result['records']}")
            print(f"  字段数: {len(result['columns'])}")
            print(f"  空值率: {result['null_percentage']:.2f}%")
            print(f"  有效数据: {'是' if result['has_values'] else '否'}")
            
            # 关键判断
            if result['has_values'] and result['null_percentage'] < 90:
                print(f"  ✅ 字段映射修复成功!")
            elif not result['has_values']:
                print(f"  ❌ 字段映射仍有问题 - 数据全为空")
            else:
                print(f"  ⚠️  部分字段可能映射错误")
        
        elif result['status'] == 'error':
            print(f"  错误: {result.get('error', 'Unknown')}")


if __name__ == '__main__':
    quick_test_esg_collectors()
