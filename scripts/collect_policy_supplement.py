"""
补采缺失的政策数据（证监会、工信部）并下载PDF附件

任务：
1. 采集证监会政策（证监会令、公告、要闻、征求意见）
2. 采集工信部政策（政策文件、部门规章）
3. 下载所有PDF/Word附件到files目录
4. 合并到现有数据文件中
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.collectors.unstructured.policy.csrc import CSRCCollector
from src.data_pipeline.collectors.unstructured.policy.miit import MIITCollector

def main():
    # 日期范围：近一年
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    print("="*60)
    print("补采缺失政策数据（证监会、工信部）")
    print("="*60)
    print(f"时间范围: {start_str} - {end_str}")
    print(f"下载附件: 是")
    print()
    
    all_data = []
    
    # 1. 采集证监会数据
    print("1. 采集证监会政策...")
    print("-" * 60)
    try:
        csrc_collector = CSRCCollector()
        csrc_df = csrc_collector.collect(
            start_date=start_str,
            end_date=end_str,
            categories=['order', 'announcement'],  # 主要类别（减少类别以提高成功率）
            max_pages=5,  # 减少页数，避免超时
            download_files=True  # 下载附件
        )
        print(f"证监会: 采集到 {len(csrc_df)} 条记录")
        if len(csrc_df) > 0:
            all_data.append(csrc_df)
            # 显示样本
            print("\n样本数据:")
            for idx, row in csrc_df.head(3).iterrows():
                print(f"  - {row['title'][:50]}...")
                if row.get('local_path'):
                    print(f"    附件: {row['local_path']}")
    except KeyboardInterrupt:
        print("用户中断采集")
        raise
    except Exception as e:
        print(f"证监会采集失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # 2. 采集工信部数据
    print("2. 采集工信部政策...")
    print("-" * 60)
    try:
        miit_collector = MIITCollector()
        miit_df = miit_collector.collect(
            start_date=start_str,
            end_date=end_str,
            categories=['policy'],  # 单个类别测试
            max_pages=5,  # 减少页数
            download_files=True  # 下载附件
        )
        print(f"工信部: 采集到 {len(miit_df)} 条记录")
        if len(miit_df) > 0:
            all_data.append(miit_df)
            # 显示样本
            print("\n样本数据:")
            for idx, row in miit_df.head(3).iterrows():
                print(f"  - {row['title'][:50]}...")
                if row.get('local_path'):
                    print(f"    附件: {row['local_path']}")
    except KeyboardInterrupt:
        print("用户中断采集")
        raise
    except Exception as e:
        print(f"工信部采集失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # 3. 合并数据
    if not all_data:
        print("未采集到任何新数据")
        return
    
    new_df = pd.concat(all_data, ignore_index=True)
    
    print("="*60)
    print("数据合并与保存")
    print("="*60)
    print(f"新采集数据: {len(new_df)} 条")
    
    # 读取现有数据
    existing_file = Path("data/raw/unstructured/policy/csv/policy_yearly_20260119.csv")
    if existing_file.exists():
        existing_df = pd.read_csv(existing_file)
        print(f"现有数据: {len(existing_df)} 条")
        
        # 合并（去重基于ID）
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['id'], keep='first')
        
        print(f"合并后数据: {len(combined_df)} 条")
        print(f"新增数据: {len(combined_df) - len(existing_df)} 条")
    else:
        combined_df = new_df
    
    # 保存CSV
    output_csv = Path("data/raw/unstructured/policy/csv") / f"policy_complete_{end_date.strftime('%Y%m%d')}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nCSV已保存: {output_csv}")
    
    # 保存JSONL
    output_jsonl = Path("data/raw/unstructured/policy/meta") / f"policy_complete_{end_date.strftime('%Y%m%d')}.jsonl"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        import json
        for record in combined_df.to_dict('records'):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"JSONL已保存: {output_jsonl}")
    
    # 统计
    print("\n" + "="*60)
    print("数据统计")
    print("="*60)
    
    # 按来源统计
    print("\n各来源数据量:")
    source_stats = combined_df.groupby('source_dept').size().sort_values(ascending=False)
    for dept, count in source_stats.items():
        print(f"  {dept}: {count} 条")
    
    # 附件统计
    has_files = combined_df['local_path'].notna() & (combined_df['local_path'] != '')
    print(f"\n已下载附件: {has_files.sum()} 个 ({has_files.sum()/len(combined_df)*100:.1f}%)")
    
    # 发文字号统计
    has_doc_no = combined_df['doc_no'].notna() & (combined_df['doc_no'] != '')
    print(f"已提取发文字号: {has_doc_no.sum()} 个 ({has_doc_no.sum()/len(combined_df)*100:.1f}%)")
    
    print("\n补采完成！")

if __name__ == "__main__":
    main()
