#!/usr/bin/env python3
"""
非结构化数据处理调度器测试脚本

测试内容：
1. 模块导入
2. Exchange数据处理（GPU加速规则打分）
3. PDF数据处理（LLM流水线）
4. 数据发现和状态
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_import():
    """测试模块导入"""
    print("=" * 60)
    print("测试 1: 模块导入")
    print("=" * 60)
    
    try:
        from src.data_pipeline.processors.unstructured.scheduler import (
            UnstructuredScheduler,
            ProcessingConfig,
            ProcessingResult,
            BatchProcessingResult,
            DataCategory,
            ProcessingStatus,
            PDFPipeline,
            ExchangePipeline,
            process_month,
            process_year,
            process_all,
        )
        print("✓ 所有调度器模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_discovery():
    """测试数据发现"""
    print("\n" + "=" * 60)
    print("测试 2: 数据发现")
    print("=" * 60)
    
    from src.data_pipeline.processors.unstructured.scheduler import (
        UnstructuredScheduler, ProcessingConfig
    )
    
    config = ProcessingConfig()
    scheduler = UnstructuredScheduler(config)
    
    print("\n发现的原始数据:")
    discovery = scheduler.discover_data()
    
    for category, files in discovery.items():
        print(f"\n  {category}:")
        for year, month in files:
            print(f"    - {year}/{month:02d}")
    
    print(f"\n总计: {sum(len(f) for f in discovery.values())} 个数据文件")
    return True


def test_exchange_processing():
    """测试Exchange数据处理（GPU加速）"""
    print("\n" + "=" * 60)
    print("测试 3: Exchange数据处理 (GPU加速)")
    print("=" * 60)
    
    from src.data_pipeline.processors.unstructured.scheduler import (
        UnstructuredScheduler, ProcessingConfig, DataCategory
    )
    
    config = ProcessingConfig(
        use_gpu=True,
        skip_existing=False,  # 强制重新处理
    )
    scheduler = UnstructuredScheduler(config)
    
    # 检查是否有数据
    raw_path = config.get_raw_path(DataCategory.EXCHANGE, 2021, 1)
    if not raw_path.exists():
        print(f"⚠ 测试数据不存在: {raw_path}")
        return True
    
    print(f"处理数据: {raw_path}")
    
    start = time.time()
    result = scheduler.process_month(DataCategory.EXCHANGE, 2021, 1, force=True)
    elapsed = time.time() - start
    
    print(f"\n处理结果:")
    print(f"  总数: {result.total}")
    print(f"  成功: {result.success_count}")
    print(f"  失败: {result.failed_count}")
    print(f"  耗时: {elapsed:.2f}s ({result.total/elapsed:.1f} 条/秒)" if result.total > 0 else "")
    print(f"  输出: {result.output_path}")
    
    if result.success_count > 0:
        print(f"\n统计:")
        print(f"  平均分: {result.avg_score:.1f}")
        print(f"  利好数: {result.bullish_count}")
        print(f"  利空数: {result.bearish_count}")
        print(f"  中性数: {result.neutral_count}")
        
        # 读取输出验证
        import pandas as pd
        if result.output_path:
            df = pd.read_parquet(result.output_path)
            print(f"\n输出数据预览:")
            print(df.head().to_string())
    
    return result.success_count > 0


def test_pdf_processing_sample():
    """测试PDF处理流水线（小样本）"""
    print("\n" + "=" * 60)
    print("测试 4: PDF流水线测试（小样本）")
    print("=" * 60)
    
    from src.data_pipeline.processors.unstructured.scheduler import (
        PDFPipeline, ProcessingConfig, DataCategory
    )
    import pandas as pd
    
    config = ProcessingConfig(
        llm_timeout=60.0,
        request_delay=0.5,
    )
    
    # 检查是否有数据
    raw_path = config.get_raw_path(DataCategory.ANNOUNCEMENTS, 2021, 1)
    if not raw_path.exists():
        print(f"⚠ 测试数据不存在: {raw_path}")
        return True
    
    # 读取原始数据
    df = pd.read_parquet(raw_path)
    print(f"原始数据: {len(df)} 条")
    
    # 只取前3条测试
    sample_df = df.head(3)
    records = sample_df.to_dict('records')
    
    print(f"\n测试样本 ({len(records)} 条):")
    for i, r in enumerate(records):
        print(f"  [{i+1}] {r.get('title', '')[:50]}...")
    
    # 初始化流水线
    pipeline = PDFPipeline(config)
    
    print("\n开始处理（需要LLM调用，可能较慢）...")
    start = time.time()
    results = pipeline.process_batch(records, DataCategory.ANNOUNCEMENTS)
    elapsed = time.time() - start
    
    print(f"\n处理结果 (耗时 {elapsed:.1f}s):")
    for i, r in enumerate(results):
        status = "✓" if r.success else "✗"
        print(f"  [{i+1}] {status} ID={r.record_id}")
        if r.success:
            print(f"      分数: {r.score}, 理由: {r.reason[:50] if r.reason else 'N/A'}...")
        else:
            print(f"      错误: {r.error_message}")
    
    success = sum(1 for r in results if r.success)
    print(f"\n成功率: {success}/{len(results)}")
    
    return True


def test_scheduler_status():
    """测试调度器状态"""
    print("\n" + "=" * 60)
    print("测试 5: 调度器状态")
    print("=" * 60)
    
    from src.data_pipeline.processors.unstructured.scheduler import (
        UnstructuredScheduler, ProcessingConfig
    )
    
    config = ProcessingConfig()
    scheduler = UnstructuredScheduler(config)
    
    status = scheduler.get_status()
    
    print("\n配置信息:")
    for k, v in status['config'].items():
        print(f"  {k}: {v}")
    
    print(f"\n已加载的流水线: {status['pipelines']}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("非结构化数据调度器测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_import),
        ("数据发现", test_data_discovery),
        ("Exchange处理", test_exchange_processing),
        ("PDF流水线样本", test_pdf_processing_sample),
        ("调度器状态", test_scheduler_status),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ 测试 '{name}' 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
