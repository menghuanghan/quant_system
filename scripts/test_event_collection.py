"""
事件驱动数据采集模块测试

测试内容：
1. 巨潮事件采集器（并购重组、违规处罚）
2. 东财结构化数据采集
3. 数据对齐功能
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))


def test_imports():
    """测试模块导入"""
    print("="*60)
    print("测试1: 模块导入")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.events import (
            EventType,
            EventSource,
            CninfoEventCollector,
            EastMoneyEventCollector,
            get_cninfo_events,
            get_eastmoney_events,
            EVENT_CATEGORIES
        )
        print("✓ 所有模块导入成功")
        
        # 显示事件类型
        print("\n事件类型枚举:")
        for et in EventType:
            print(f"  - {et.name}: {et.value}")
        
        # 显示数据源
        print("\n数据源枚举:")
        for es in EventSource:
            print(f"  - {es.name}: {es.value}")
        
        # 显示巨潮分类配置
        print("\n巨潮事件分类配置:")
        for event_type, config in EVENT_CATEGORIES.items():
            print(f"  - {config['name']} ({event_type})")
            print(f"    ID: {config['id'][:30]}...")
            if config['keywords']:
                print(f"    关键词: {', '.join(config['keywords'][:3])}...")
        
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cninfo_collector():
    """测试巨潮事件采集器"""
    print("\n" + "="*60)
    print("测试2: 巨潮事件采集器")
    print("="*60)
    
    from src.data_pipeline.collectors.unstructured.events import (
        CninfoEventCollector,
        EventType
    )
    
    try:
        collector = CninfoEventCollector()
        print("✓ 采集器实例化成功")
        
        # 测试采集（最近7天，只采集并购重组，不下载PDF）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"\n采集时间范围: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        print("事件类型: 并购重组 (merger)")
        print("下载PDF: 否")
        
        df = collector.collect(
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            event_types=[EventType.MERGER.value],
            download_pdf=False,  # 测试不下载
            max_pages=3  # 限制页数
        )
        
        print(f"\n采集结果: {len(df)} 条记录")
        
        if len(df) > 0:
            print("\n样本数据:")
            for _, row in df.head(3).iterrows():
                print(f"  [{row['ts_code']}] {row['title'][:40]}...")
                print(f"    日期: {row['ann_date']}, 子类型: {row['event_subtype']}")
            
            print(f"\n列名: {df.columns.tolist()}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eastmoney_collector():
    """测试东财事件采集器"""
    print("\n" + "="*60)
    print("测试3: 东财事件采集器")
    print("="*60)
    
    from src.data_pipeline.collectors.unstructured.events import (
        EastMoneyEventCollector
    )
    
    try:
        collector = EastMoneyEventCollector()
        print("✓ 采集器实例化成功")
        
        # 测试采集（最近30天）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"\n采集时间范围: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        print("事件类型: 违规处罚 (penalty)")
        
        df = collector.collect(
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            event_types=['penalty'],
            max_pages=2  # 限制页数
        )
        
        print(f"\n采集结果: {len(df)} 条记录")
        
        if len(df) > 0:
            print("\n样本数据（含结构化标签）:")
            for _, row in df.head(3).iterrows():
                print(f"  [{row['ts_code']}] {row['stock_name']}")
                print(f"    摘要: {row['summary'][:50]}...")
                if row.get('labels'):
                    print(f"    标签: {row['labels']}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_storage():
    """测试数据存储路径"""
    print("\n" + "="*60)
    print("测试4: 数据存储结构")
    print("="*60)
    
    from src.data_pipeline.collectors.unstructured.events.base_event import (
        BaseEventCollector
    )
    
    try:
        data_dir = Path("data/raw/events")
        
        # 检查目录结构
        print(f"\n数据根目录: {data_dir.absolute()}")
        
        expected_dirs = [
            'merger_acquisition',
            'penalty',
            'control_change',
            'contract',
            'meta'
        ]
        
        for dir_name in expected_dirs:
            dir_path = data_dir / dir_name
            exists = dir_path.exists()
            status = "✓" if exists else "○"
            print(f"  {status} {dir_name}/")
        
        # 检查元数据文件
        meta_dir = data_dir / 'meta'
        if meta_dir.exists():
            jsonl_files = list(meta_dir.glob('*.jsonl'))
            if jsonl_files:
                print(f"\n元数据文件:")
                for f in jsonl_files[:5]:
                    print(f"  - {f.name}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_quick_collection():
    """快速采集测试（实际采集数据）"""
    print("\n" + "="*60)
    print("测试5: 快速采集（实际数据）")
    print("="*60)
    
    from src.data_pipeline.collectors.unstructured.events import (
        get_cninfo_events,
        EventType
    )
    
    try:
        # 采集最近7天的处罚公告（通常数量较少）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"采集处罚公告: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        df = get_cninfo_events(
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            event_types=[EventType.PENALTY.value],
            download_pdf=False,
            max_pages=2
        )
        
        print(f"\n采集结果: {len(df)} 条违规处罚记录")
        
        if len(df) > 0:
            # 统计
            print("\n按子类型统计:")
            subtype_counts = df['event_subtype'].value_counts()
            for subtype, count in subtype_counts.items():
                print(f"  {subtype}: {count}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("事件驱动数据采集模块测试")
    print("="*60)
    
    results = []
    
    # 1. 测试导入
    results.append(("模块导入", test_imports()))
    
    # 2. 测试巨潮采集器
    results.append(("巨潮采集器", test_cninfo_collector()))
    
    # 3. 测试东财采集器
    results.append(("东财采集器", test_eastmoney_collector()))
    
    # 4. 测试存储结构
    results.append(("存储结构", test_data_storage()))
    
    # 5. 快速采集测试
    results.append(("快速采集", test_quick_collection()))
    
    # 汇总
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
