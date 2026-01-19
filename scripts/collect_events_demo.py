"""
事件驱动数据采集演示

采集近一年的高价值事件数据：
1. 并购重组事件（含PDF下载）
2. 违规处罚事件
3. 实控人变更
4. 重大合同

输出：
- PDF文件：data/raw/events/{event_type}/{year}/
- 元数据：data/raw/events/meta/
- 汇总CSV：data/raw/events/events_summary.csv
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.collectors.unstructured.events import (
    CninfoEventCollector,
    EastMoneyEventCollector,
    EventType,
    get_cninfo_events
)


def main():
    # 时间范围：最近30天（演示用，正式采集可改为365天）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    print("="*70)
    print("事件驱动数据采集")
    print("="*70)
    print(f"时间范围: {start_str} ~ {end_str}")
    print()
    
    all_events = []
    
    # 1. 采集并购重组事件
    print("="*70)
    print("1. 采集并购重组事件")
    print("="*70)
    try:
        merger_df = get_cninfo_events(
            start_date=start_str,
            end_date=end_str,
            event_types=[EventType.MERGER.value],
            download_pdf=False,  # 演示不下载，正式采集改为True
            max_pages=10
        )
        print(f"并购重组: {len(merger_df)} 条")
        if len(merger_df) > 0:
            all_events.append(merger_df)
            print("\n样本:")
            for _, row in merger_df.head(3).iterrows():
                print(f"  [{row['ts_code']}] {row['title'][:45]}...")
    except Exception as e:
        print(f"采集失败: {e}")
    
    # 2. 采集违规处罚事件
    print("\n" + "="*70)
    print("2. 采集违规处罚事件")
    print("="*70)
    try:
        penalty_df = get_cninfo_events(
            start_date=start_str,
            end_date=end_str,
            event_types=[EventType.PENALTY.value],
            download_pdf=False,
            max_pages=10
        )
        print(f"违规处罚: {len(penalty_df)} 条")
        if len(penalty_df) > 0:
            all_events.append(penalty_df)
            print("\n样本:")
            for _, row in penalty_df.head(3).iterrows():
                print(f"  [{row['ts_code']}] {row['title'][:45]}...")
    except Exception as e:
        print(f"采集失败: {e}")
    
    # 3. 采集实控人变更事件
    print("\n" + "="*70)
    print("3. 采集实控人变更事件")
    print("="*70)
    try:
        control_df = get_cninfo_events(
            start_date=start_str,
            end_date=end_str,
            event_types=[EventType.CONTROL_CHANGE.value],
            download_pdf=False,
            max_pages=10
        )
        print(f"实控人变更: {len(control_df)} 条")
        if len(control_df) > 0:
            all_events.append(control_df)
            print("\n样本:")
            for _, row in control_df.head(3).iterrows():
                print(f"  [{row['ts_code']}] {row['title'][:45]}...")
    except Exception as e:
        print(f"采集失败: {e}")
    
    # 4. 采集重大合同事件
    print("\n" + "="*70)
    print("4. 采集重大合同事件")
    print("="*70)
    try:
        contract_df = get_cninfo_events(
            start_date=start_str,
            end_date=end_str,
            event_types=[EventType.MAJOR_CONTRACT.value],
            download_pdf=False,
            max_pages=10
        )
        print(f"重大合同: {len(contract_df)} 条")
        if len(contract_df) > 0:
            all_events.append(contract_df)
            print("\n样本:")
            for _, row in contract_df.head(3).iterrows():
                print(f"  [{row['ts_code']}] {row['title'][:45]}...")
    except Exception as e:
        print(f"采集失败: {e}")
    
    # 汇总
    print("\n" + "="*70)
    print("采集汇总")
    print("="*70)
    
    if not all_events:
        print("未采集到任何数据")
        return
    
    combined_df = pd.concat(all_events, ignore_index=True)
    
    # 去重
    combined_df = combined_df.drop_duplicates(subset=['id'])
    
    print(f"\n总记录数: {len(combined_df)}")
    
    # 按事件类型统计
    print("\n按事件类型统计:")
    event_stats = combined_df.groupby('event_type').size().sort_values(ascending=False)
    for event_type, count in event_stats.items():
        print(f"  {event_type}: {count}")
    
    # 按子类型统计
    print("\n按子类型统计:")
    subtype_stats = combined_df.groupby('event_subtype').size().sort_values(ascending=False)
    for subtype, count in subtype_stats.head(10).items():
        print(f"  {subtype}: {count}")
    
    # 保存汇总CSV
    output_dir = Path("data/raw/events")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = output_dir / f"events_summary_{end_date.strftime('%Y%m%d')}.csv"
    combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n汇总CSV已保存: {output_csv}")
    
    # 统计数据目录
    print("\n数据目录结构:")
    for event_dir in ['merger_acquisition', 'penalty', 'control_change', 'contract', 'meta']:
        dir_path = output_dir / event_dir
        if dir_path.exists():
            file_count = len(list(dir_path.rglob('*.*')))
            print(f"  {event_dir}/: {file_count} 个文件")
    
    print("\n" + "="*70)
    print("采集完成！")
    print("="*70)


if __name__ == "__main__":
    main()
