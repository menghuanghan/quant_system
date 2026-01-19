"""
采集2025年1-3月事件驱动数据

采集范围：
- 时间：2025年1月1日 - 2025年3月31日
- 事件类型：
  1. 并购重组
  2. 违规处罚
  3. 实控人变更
  4. 重大合同

存储路径：data/raw/unstructured/events/
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.collectors.unstructured.events import (
    get_cninfo_events,
    EventType
)


def main():
    # 时间范围：2025年1-3月
    start_date = '20250101'
    end_date = '20250331'
    
    print("="*70)
    print("采集2025年1-3月事件驱动数据")
    print("="*70)
    print(f"时间范围: {start_date} ~ {end_date}")
    print("存储路径: data/raw/unstructured/events/")
    print()
    
    # 事件类型配置
    event_configs = [
        {
            'type': EventType.MERGER.value,
            'name': '并购重组',
            'max_pages': 50
        },
        {
            'type': EventType.PENALTY.value,
            'name': '违规处罚',
            'max_pages': 50
        },
        {
            'type': EventType.CONTROL_CHANGE.value,
            'name': '实控人变更',
            'max_pages': 50
        },
        {
            'type': EventType.MAJOR_CONTRACT.value,
            'name': '重大合同',
            'max_pages': 50
        },
    ]
    
    all_events = []
    success_count = 0
    
    for config in event_configs:
        event_type = config['type']
        event_name = config['name']
        max_pages = config['max_pages']
        
        print("="*70)
        print(f"采集{event_name}事件")
        print("="*70)
        
        try:
            df = get_cninfo_events(
                start_date=start_date,
                end_date=end_date,
                event_types=[event_type],
                download_pdf=True,  # 下载PDF原文
                max_pages=max_pages
            )
            
            print(f"✓ {event_name}: {len(df)} 条")
            
            if len(df) > 0:
                all_events.append(df)
                success_count += 1
                
                # 显示样本
                print("\n样本数据:")
                for _, row in df.head(5).iterrows():
                    print(f"  [{row['ts_code']}] {row['title'][:50]}...")
                    if row.get('local_path'):
                        print(f"    PDF: {row['local_path']}")
                
                # 统计子类型
                if 'event_subtype' in df.columns:
                    subtype_counts = df['event_subtype'].value_counts()
                    print(f"\n子类型分布:")
                    for subtype, count in subtype_counts.items():
                        print(f"    {subtype}: {count}")
            else:
                print(f"⚠ 未采集到数据")
            
            print()
            
        except Exception as e:
            print(f"✗ 采集失败: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # 汇总统计
    print("="*70)
    print("采集汇总")
    print("="*70)
    
    if not all_events:
        print("⚠ 未采集到任何数据")
        return
    
    combined_df = pd.concat(all_events, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['id'])
    
    print(f"\n总记录数: {len(combined_df)}")
    print(f"成功采集事件类型: {success_count}/{len(event_configs)}")
    
    # 按事件类型统计
    print("\n按事件类型统计:")
    event_stats = combined_df.groupby('event_type').size().sort_values(ascending=False)
    for event_type, count in event_stats.items():
        pct = count / len(combined_df) * 100
        print(f"  {event_type:20s}: {count:4d} ({pct:5.1f}%)")
    
    # 按月份统计
    print("\n按月份统计:")
    combined_df['month'] = pd.to_datetime(combined_df['ann_date']).dt.to_period('M')
    month_stats = combined_df.groupby('month').size()
    for month, count in month_stats.items():
        print(f"  {month}: {count}")
    
    # 统计PDF下载情况
    has_pdf = combined_df['local_path'].notna() & (combined_df['local_path'] != '')
    print(f"\n已下载PDF: {has_pdf.sum()} 个 ({has_pdf.sum()/len(combined_df)*100:.1f}%)")
    
    # 保存汇总CSV
    output_dir = Path("data/raw/unstructured/events")
    output_csv = output_dir / "events_2025_Q1.csv"
    combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n汇总CSV已保存: {output_csv}")
    
    # 统计存储目录
    print("\n存储目录结构:")
    for event_dir in ['merger_acquisition', 'penalty', 'control_change', 'contract', 'meta']:
        dir_path = output_dir / event_dir
        if dir_path.exists():
            pdf_count = len(list(dir_path.rglob('*.pdf')))
            jsonl_count = len(list(dir_path.rglob('*.jsonl')))
            print(f"  {event_dir}/")
            if pdf_count > 0:
                print(f"    PDF: {pdf_count} 个")
            if jsonl_count > 0:
                print(f"    JSONL: {jsonl_count} 个")
    
    print("\n" + "="*70)
    print("✓ 采集完成！")
    print("="*70)


if __name__ == "__main__":
    main()
