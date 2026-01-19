"""
采集2024年10-12月事件驱动数据

采集范围：
- 时间：2024年10月1日 - 2024年12月31日
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
import traceback

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.collectors.unstructured.events import (
    get_cninfo_events
)


def main():
    # 时间范围：2024年10-12月
    start_date = '20241001'
    end_date = '20241231'
    
    print("="*70)
    print("采集2024年10-12月事件驱动数据")
    print("="*70)
    print(f"时间范围: {start_date} ~ {end_date}")
    print("存储路径: data/raw/unstructured/events/")
    print()
    
    # 配置采集参数
    event_configs = {
        '并购重组': {
            'type': 'merger',
            'max_pages': 30,
        },
        '违规处罚': {
            'type': 'penalty',
            'max_pages': 50,
        },
        '实控人变更': {
            'type': 'control_change',
            'max_pages': 30,
        },
        '重大合同': {
            'type': 'contract',
            'max_pages': 30,
        }
    }
    
    all_events = []
    stats = {}
    
    # 逐个采集
    for name, config in event_configs.items():
        print("="*70)
        print(f"采集{name}事件")
        print("="*70)
        
        try:
            result = get_cninfo_events(
                start_date=start_date,
                end_date=end_date,
                event_types=[config['type']],
                max_pages=config['max_pages'],
                download_pdf=True,  # 下载PDF
            )
            
            if result is not None and len(result) > 0:
                print(f"✓ {name}: {len(result)} 条")
                all_events.append(result)
                stats[name] = len(result)
                
                # 打印样本
                print("\n样本数据:")
                for idx in range(min(5, len(result))):
                    row = result.iloc[idx]
                    title = row.get('title', '')[:60] + '...' if len(row.get('title', '')) > 60 else row.get('title', '')
                    print(f"  [{row.get('ts_code', 'N/A')}] {title}")
            else:
                print(f"✓ {name}: 0 条")
                print("⚠ 未采集到数据")
                stats[name] = 0
            
            print()
        
        except Exception as e:
            print(f"✗ 采集{name}失败")
            print(f"错误: {e}")
            traceback.print_exc()
            stats[name] = 0
            print()
    
    # 汇总统计
    print("="*70)
    print("采集汇总")
    print("="*70)
    
    if all_events:
        # 合并所有DataFrame
        df_all = pd.concat(all_events, ignore_index=True)
        
        print(f"\n总记录数: {len(df_all)}")
        print(f"成功采集事件类型: {sum(1 for v in stats.values() if v > 0)}/{len(event_configs)}")
        
        # 按事件类型统计
        print("\n按事件类型统计:")
        for name, count in stats.items():
            pct = count / len(df_all) * 100 if len(df_all) > 0 else 0
            event_type_key = name.replace('并购重组', 'merger').replace('违规处罚', 'penalty') \
                                .replace('实控人变更', 'control_change').replace('重大合同', 'contract')
            print(f"  {event_type_key:20s}: {count:4d} ({pct:5.1f}%)")
        
        # 按月份统计
        print("\n按月份统计:")
        months = {}
        for _, row in df_all.iterrows():
            if row.get('ann_date'):
                month = row['ann_date'][:7]  # YYYY-MM
                months[month] = months.get(month, 0) + 1
        
        for month in sorted(months.keys()):
            print(f"  {month}: {months[month]}")
        
        # PDF下载情况
        pdf_count = sum(1 for _, row in df_all.iterrows() 
                       if row.get('local_path') and Path(row['local_path']).exists())
        pdf_pct = pdf_count / len(df_all) * 100 if len(df_all) > 0 else 0
        print(f"\n已下载PDF: {pdf_count} 个 ({pdf_pct:.1f}%)")
        
        # 保存汇总CSV
        output_path = Path('data/raw/unstructured/events/events_2024_Q4.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n汇总CSV已保存: {output_path}")
        
        # 显示存储结构
        print("\n存储目录结构:")
        base_dir = Path('data/raw/unstructured/events')
        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                print(f"  {subdir.name}/")
                if subdir.name == 'meta':
                    jsonl_files = list(subdir.glob('*.jsonl'))
                    print(f"    JSONL: {len(jsonl_files)} 个")
    else:
        print("\n⚠ 未采集到任何数据")
    
    print("\n" + "="*70)
    print("✓ 采集完成！")
    print("="*70)


if __name__ == '__main__':
    main()
