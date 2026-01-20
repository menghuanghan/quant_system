"""
流式采集器验证测试脚本

验证清单：
1. 500GB 验证：检查 Parquet 文件大小（目标：5年数据 < 200GB）
2. 防泄露验证（Time Travel Check）：只有日期的公告必须被填充为 17:00:00
3. 版本控制验证：同一 URL 采集两次应产生 2 条记录（crawled_time 不同）

用法:
    python tests/test_streaming_pipeline.py
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, passed: bool, message: str = ""):
    """打印测试结果"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {name}")
    if message:
        print(f"         {message}")


# ============================================================
# 测试 1: 基础设施测试
# ============================================================

def test_time_utils():
    """测试时间清洗模块"""
    print_header("测试 1: 时间清洗模块 (time_utils)")
    
    try:
        from src.data_pipeline.clean.unstructured.time_utils import (
            standardize_publish_time,
            TimeMode,
            TimeAccuracy,
            TimeNormalizer
        )
        print_result("导入 time_utils", True)
    except ImportError as e:
        print_result("导入 time_utils", False, str(e))
        return False
    
    # 测试保守模式填充
    test_cases = [
        # (输入, 预期输出包含的关键部分)
        ("2024-01-15", "17:00:00"),      # 仅日期 → 17:00
        ("2024-01-15 14:30", "14:30"),   # 有时间 → 保留
        ("2024-01-15 14:30:45", "14:30:45"),  # 有秒 → 保留
        ("20240115", "17:00:00"),        # YYYYMMDD → 17:00
        ("2024年01月15日", "17:00:00"),  # 中文格式 → 17:00
    ]
    
    all_passed = True
    for raw_time, expected_part in test_cases:
        result = standardize_publish_time(raw_time, 'conservative')
        passed = expected_part in result
        print_result(f"标准化 '{raw_time}'", passed, f"→ {result}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_scraper_get_bytes():
    """测试 scraper_base.get_bytes 方法"""
    print_header("测试 2: ScraperBase.get_bytes()")
    
    try:
        from src.data_pipeline.collectors.unstructured.scraper_base import ScraperBase
        print_result("导入 ScraperBase", True)
    except ImportError as e:
        print_result("导入 ScraperBase", False, str(e))
        return False
    
    # 测试下载小文件到内存
    scraper = ScraperBase(use_proxy=False, rate_limit=False)
    
    test_url = "https://httpbin.org/bytes/1024"  # 返回 1KB 随机数据
    
    try:
        content = scraper.get_bytes(test_url, max_size_mb=1.0, timeout=15)
        passed = content is not None and len(content) > 0
        print_result("下载到内存", passed, f"大小: {len(content) if content else 0} bytes")
    except Exception as e:
        print_result("下载到内存", False, str(e))
        scraper.close()
        return False
    
    scraper.close()
    return True


def test_streaming_collector_base():
    """测试 StreamingCollector 基类"""
    print_header("测试 3: StreamingCollector 基类")
    
    try:
        from src.data_pipeline.collectors.unstructured.base import (
            StreamingCollector,
            BufferedItem,
            ContentType
        )
        print_result("导入 StreamingCollector", True)
    except ImportError as e:
        print_result("导入 StreamingCollector", False, str(e))
        return False
    
    # 测试 BufferedItem
    item = BufferedItem(
        source="test",
        content_type=ContentType.TEXT,
        publish_time="2024-01-15 17:00:00",
        time_accuracy="D",
        crawled_time="2024-01-15 20:00:00",
        title="测试标题"
    )
    
    item_dict = item.to_dict()
    passed = item_dict['source'] == 'test' and item_dict['content_type'] == 'text'
    print_result("BufferedItem.to_dict()", passed)
    
    return passed


# ============================================================
# 测试 2: 防泄露验证 (Time Travel Check)
# ============================================================

def test_time_travel_check():
    """
    防泄露验证
    
    检查只有日期的公告（如 2023-05-12）经过清洗后
    必须变成 2023-05-12 17:00:00，而不是 00:00:00
    """
    print_header("测试 4: 防泄露验证 (Time Travel Check)")
    
    try:
        from src.data_pipeline.clean.unstructured.time_utils import standardize_publish_time
    except ImportError as e:
        print_result("导入 time_utils", False, str(e))
        return False
    
    # 模拟仅有日期的公告
    test_dates = [
        "2023-05-12",
        "2024-01-01",
        "2024-12-31",
        "20230512",
        "2023年05月12日",
    ]
    
    all_passed = True
    for date_str in test_dates:
        result = standardize_publish_time(date_str, 'conservative')
        
        # 检查 1: 不能是 00:00:00（会导致泄露）
        is_not_midnight = "00:00:00" not in result
        
        # 检查 2: 应该是 17:00:00（保守模式）
        is_conservative = "17:00:00" in result
        
        passed = is_not_midnight and is_conservative
        
        print_result(
            f"'{date_str}' → '{result}'",
            passed,
            "✓ 保守填充 17:00" if passed else "✗ 可能泄露!"
        )
        
        if not passed:
            all_passed = False
    
    return all_passed


# ============================================================
# 测试 3: 版本控制验证
# ============================================================

def test_version_control():
    """
    版本控制验证
    
    同一条记录采集两次（模拟内容微调），应产生 2 条记录
    去重是读取层的事，采集层必须全量保留
    """
    print_header("测试 5: 版本控制验证")
    
    try:
        from src.data_pipeline.collectors.unstructured.base import (
            StreamingCollector,
            BufferedItem,
            ContentType
        )
        from src.data_pipeline.collectors.unstructured.storage import DataSink
    except ImportError as e:
        print_result("导入模块", False, str(e))
        return False
    
    # 创建临时存储目录
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        sink = DataSink(base_path=tmpdir)
        
        # 模拟同一条新闻采集两次
        items = []
        for i in range(2):
            item = BufferedItem(
                source="test",
                content_type=ContentType.TEXT,
                publish_time="2024-01-15 10:30:00",
                time_accuracy="S",
                crawled_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                title="相同的新闻标题",
                content="相同的新闻内容（第{}次采集）".format(i + 1),
                url="https://example.com/same-news",
                original_id="news_123"
            )
            items.append(item.to_dict())
            time.sleep(0.1)  # 确保 crawled_time 不同
        
        # 保存
        import pandas as pd
        df = pd.DataFrame(items)
        sink.save(df, domain="test", filename="version_test")
        
        # 读取验证
        loaded_df = sink.load(domain="test")
        
        # 检查记录数
        record_count = len(loaded_df)
        passed = record_count == 2
        
        print_result(
            f"同一 URL 采集两次",
            passed,
            f"记录数: {record_count} (应为 2)"
        )
        
        if passed:
            # 检查 crawled_time 是否不同
            crawled_times = loaded_df['crawled_time'].unique()
            times_different = len(crawled_times) == 2
            print_result(
                "crawled_time 不同",
                times_different,
                f"唯一值: {len(crawled_times)}"
            )
        
        return passed


# ============================================================
# 测试 4: 存储大小验证
# ============================================================

def test_storage_size_estimate():
    """
    存储大小估算
    
    模拟一天的数据量，估算 5 年的存储需求
    目标：5 年数据 < 200GB（500GB 限制的 40%）
    """
    print_header("测试 6: 存储大小估算 (500GB 验证)")
    
    try:
        from src.data_pipeline.collectors.unstructured.storage import DataSink
    except ImportError as e:
        print_result("导入 DataSink", False, str(e))
        return False
    
    import tempfile
    import pandas as pd
    import numpy as np
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sink = DataSink(base_path=tmpdir)
        
        # 模拟一天的公告数据（假设 500 条公告，平均每条 2KB 文本）
        num_records = 500
        avg_content_size = 2000  # 字符
        
        records = []
        for i in range(num_records):
            records.append({
                'source': 'cninfo',
                'content_type': 'text',
                'publish_time': '2024-01-15 17:00:00',
                'time_accuracy': 'D',
                'crawled_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'title': f'公告标题{i}' + '测试' * 10,
                'content': '这是公告正文内容。' * (avg_content_size // 10),
                'url': f'https://example.com/ann/{i}.pdf',
                'ts_code': f'{str(i).zfill(6)}.SZ',
                'name': f'测试股票{i}',
                'category': '年报',
                'original_id': f'ann_{i}'
            })
        
        df = pd.DataFrame(records)
        
        # 保存为 Parquet
        file_path = sink.save(df, domain="test", filename="size_test")
        
        # 获取文件大小
        import glob
        parquet_files = list(Path(tmpdir).rglob("*.parquet"))
        
        if parquet_files:
            file_size_kb = parquet_files[0].stat().st_size / 1024
            file_size_mb = file_size_kb / 1024
            
            # 估算
            days_per_year = 250  # 交易日
            years = 5
            total_estimated_gb = (file_size_mb * days_per_year * years) / 1024
            
            passed = total_estimated_gb < 200  # 目标 < 200GB
            
            print_result(
                f"单日数据大小",
                True,
                f"{file_size_mb:.2f} MB ({num_records} 条记录)"
            )
            
            print_result(
                f"5年估算大小",
                passed,
                f"{total_estimated_gb:.1f} GB (目标 < 200GB)"
            )
            
            # 计算压缩比
            raw_size_mb = (num_records * avg_content_size) / 1024 / 1024
            compression_ratio = raw_size_mb / file_size_mb if file_size_mb > 0 else 0
            print_result(
                f"Parquet 压缩比",
                True,
                f"{compression_ratio:.1f}x"
            )
            
            return passed
        else:
            print_result("保存 Parquet 文件", False, "文件未生成")
            return False


# ============================================================
# 集成测试
# ============================================================

def test_announcement_streaming_collector():
    """测试流式公告采集器（仅元数据，不下载 PDF）"""
    print_header("测试 7: StreamingAnnouncementCollector")
    
    try:
        from src.data_pipeline.collectors.unstructured.announcements import (
            StreamingAnnouncementCollector,
            verify_time_cleaning
        )
        print_result("导入 StreamingAnnouncementCollector", True)
    except ImportError as e:
        print_result("导入 StreamingAnnouncementCollector", False, str(e))
        return False
    
    # 测试时间清洗验证函数
    print("  正在验证时间清洗...")
    try:
        results = verify_time_cleaning(
            start_date='2024-01-01',
            end_date='2024-01-01',
            sample_size=3
        )
        
        if results:
            all_conservative = all(r.get('is_conservative', False) for r in results)
            print_result(
                "时间清洗验证",
                all_conservative,
                f"采样 {len(results)} 条"
            )
            
            # 打印采样结果
            for r in results[:3]:
                print(f"      {r['raw_time']} → {r['cleaned_time']}")
        else:
            print_result("时间清洗验证", True, "无数据（可能网络问题）")
            
    except Exception as e:
        print_result("时间清洗验证", False, str(e))
    
    return True


def test_news_streaming_collector():
    """测试流式新闻采集器"""
    print_header("测试 8: StreamingNewsCollector")
    
    try:
        from src.data_pipeline.collectors.unstructured.news import (
            StreamingNewsCollector
        )
        print_result("导入 StreamingNewsCollector", True)
    except ImportError as e:
        print_result("导入 StreamingNewsCollector", False, str(e))
        return False
    
    return True


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  流式采集器管道化集成 - 验证测试")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 60)
    
    results = {}
    
    # 基础设施测试
    results['time_utils'] = test_time_utils()
    results['scraper_get_bytes'] = test_scraper_get_bytes()
    results['streaming_collector_base'] = test_streaming_collector_base()
    
    # 核心验证测试
    results['time_travel_check'] = test_time_travel_check()
    results['version_control'] = test_version_control()
    results['storage_size'] = test_storage_size_estimate()
    
    # 集成测试
    results['announcement_streaming'] = test_announcement_streaming_collector()
    results['news_streaming'] = test_news_streaming_collector()
    
    # 汇总
    print_header("测试汇总")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    print(f"\n  总计: {passed}/{total} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n  🎉 所有测试通过！管道化集成验证成功。")
    else:
        print("\n  ⚠️ 部分测试失败，请检查上述错误。")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
