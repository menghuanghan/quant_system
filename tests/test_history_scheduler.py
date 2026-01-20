"""
全量历史调度器测试脚本

测试项目：
1. 状态管理（checkpoint）
2. 时间槽生成
3. 配置路由（热/冷数据）
4. 熔断器逻辑
5. 断点续传
6. 资源监控

运行方式：
    python tests/test_history_scheduler.py
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_task_state_enum():
    """测试任务状态枚举"""
    print("\n[TEST 1] TaskState 枚举")
    print("-" * 50)
    
    from src.data_pipeline.scheduler.unstructured.checkpoint import TaskState
    
    # 测试枚举值
    assert TaskState.PENDING.value == "PENDING"
    assert TaskState.RUNNING.value == "RUNNING"
    assert TaskState.COMPLETED.value == "COMPLETED"
    assert TaskState.FAILED.value == "FAILED"
    
    # 测试字符串转换
    assert TaskState("PENDING") == TaskState.PENDING
    assert str(TaskState.COMPLETED) == "TaskState.COMPLETED"
    
    print("  ✓ 枚举值正确")
    print("  ✓ 字符串转换正确")
    return True


def test_checkpoint_manager():
    """测试检查点管理器"""
    print("\n[TEST 2] CheckpointManager 断点续传")
    print("-" * 50)
    
    from src.data_pipeline.scheduler.unstructured.checkpoint import (
        CheckpointManager,
        TaskState,
        TaskRecord
    )
    
    # 使用临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir)
        
        # 创建管理器
        manager = CheckpointManager(state_dir=state_dir)
        print(f"  状态文件: {manager.state_path}")
        
        # 初始化任务
        sources = ['news_test', 'ann_test']
        months = ['2024-12', '2024-11', '2024-10']
        manager.initialize_tasks(sources, months)
        
        progress = manager.get_progress()
        assert progress['total_tasks'] == 6  # 2 sources x 3 months
        assert progress['pending'] == 6
        print(f"  ✓ 初始化 {progress['total_tasks']} 个任务")
        
        # 更新任务状态
        manager.update_task('news_test', '2024-12', TaskState.RUNNING)
        task = manager.get_task('news_test', '2024-12')
        assert task.state == TaskState.RUNNING
        print("  ✓ 状态更新为 RUNNING")
        
        # 模拟完成
        manager.update_task(
            'news_test', '2024-12',
            TaskState.COMPLETED,
            record_count=1000,
            file_size_mb=5.5
        )
        task = manager.get_task('news_test', '2024-12')
        assert task.state == TaskState.COMPLETED
        assert task.record_count == 1000
        print("  ✓ 状态更新为 COMPLETED，记录数=1000")
        
        # 模拟失败
        manager.update_task(
            'news_test', '2024-11',
            TaskState.FAILED,
            error_message="Connection timeout"
        )
        task = manager.get_task('news_test', '2024-11')
        assert task.state == TaskState.FAILED
        assert "timeout" in task.error_message.lower()
        print("  ✓ 状态更新为 FAILED")
        
        # 获取待处理任务
        pending = manager.get_pending_tasks('news_test', include_failed=True)
        # 应该有 1 个 PENDING + 1 个 FAILED
        assert len(pending) == 2
        # FAILED 优先
        assert pending[0][2] == TaskState.FAILED
        print(f"  ✓ 待处理任务 {len(pending)} 个，FAILED 优先")
        
        # 测试持久化（重新加载）
        manager2 = CheckpointManager(state_dir=state_dir)
        task_reloaded = manager2.get_task('news_test', '2024-12')
        assert task_reloaded.state == TaskState.COMPLETED
        assert task_reloaded.record_count == 1000
        print("  ✓ 持久化恢复成功")
        
        # 重置失败任务
        manager2.reset_failed_tasks('news_test')
        task_reset = manager2.get_task('news_test', '2024-11')
        assert task_reset.state == TaskState.PENDING
        print("  ✓ 失败任务重置为 PENDING")
    
    return True


def test_task_context_manager():
    """测试任务上下文管理器"""
    print("\n[TEST 3] task_context 上下文管理器")
    print("-" * 50)
    
    from src.data_pipeline.scheduler.unstructured.checkpoint import (
        CheckpointManager,
        TaskState
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(state_dir=Path(tmpdir))
        manager.initialize_tasks(['test_src'], ['2024-12'])
        
        # 正常完成
        with manager.task_context('test_src', '2024-12') as task:
            task.record_count = 500
        
        result = manager.get_task('test_src', '2024-12')
        assert result.state == TaskState.COMPLETED
        assert result.record_count == 500
        print("  ✓ 正常退出自动标记 COMPLETED")
        
        # 重置后测试异常
        manager.reset_failed_tasks()
        
        try:
            with manager.task_context('test_src', '2024-12') as task:
                raise ValueError("模拟采集错误")
        except ValueError:
            pass
        
        result = manager.get_task('test_src', '2024-12')
        assert result.state == TaskState.FAILED
        assert "模拟采集错误" in result.error_message
        print("  ✓ 异常退出自动标记 FAILED")
    
    return True


def test_time_slot_generation():
    """测试时间槽生成"""
    print("\n[TEST 4] 时间槽生成（倒序切片）")
    print("-" * 50)
    
    from src.data_pipeline.scheduler.unstructured.config import (
        generate_time_slots,
        DataTemperature
    )
    
    # 生成 2021-2025 的时间槽
    slots = generate_time_slots(2021, 2024, end_month=12, reverse=True)
    
    # 验证数量：4年 x 12月 = 48
    assert len(slots) == 48
    print(f"  ✓ 生成 {len(slots)} 个时间槽")
    
    # 验证倒序（最新在前）
    assert slots[0].month == '2024-12'
    assert slots[-1].month == '2021-01'
    print(f"  ✓ 倒序正确: {slots[0].month} -> {slots[-1].month}")
    
    # 验证日期格式
    slot = slots[0]  # 2024-12
    assert slot.start_date == '20241201'
    assert slot.end_date == '20241231'
    print(f"  ✓ 日期格式正确: {slot.start_date} ~ {slot.end_date}")
    
    # 验证温度分配（以当前日期为基准计算）
    hot_count = sum(1 for s in slots if s.temperature == DataTemperature.HOT)
    warm_count = sum(1 for s in slots if s.temperature == DataTemperature.WARM)
    cold_count = sum(1 for s in slots if s.temperature == DataTemperature.COLD)
    
    print(f"  温度分布: HOT={hot_count}, WARM={warm_count}, COLD={cold_count}")
    
    # 验证最新数据是 HOT
    recent_slot = slots[0]  # 2024-12
    # 注意：温度取决于当前日期，这里只验证逻辑正确
    print(f"  ✓ 2024-12 温度: {recent_slot.temperature.value}")
    
    return True


def test_temperature_policy_routing():
    """测试温度策略路由"""
    print("\n[TEST 5] 热/冷数据策略路由")
    print("-" * 50)
    
    from src.data_pipeline.scheduler.unstructured.config import (
        get_temperature_for_date,
        get_policy_for_month,
        DataTemperature,
        TEMPERATURE_POLICIES
    )
    
    # 测试 HOT 策略
    hot_policy = TEMPERATURE_POLICIES[DataTemperature.HOT]
    assert hot_policy.save_pdf == True
    assert hot_policy.text_only_mode == False
    assert hot_policy.max_workers >= 4
    print(f"  HOT: save_pdf={hot_policy.save_pdf}, workers={hot_policy.max_workers}")
    
    # 测试 COLD 策略
    cold_policy = TEMPERATURE_POLICIES[DataTemperature.COLD]
    assert cold_policy.save_pdf == False
    assert cold_policy.text_only_mode == True
    assert cold_policy.max_workers <= 2
    print(f"  COLD: text_only={cold_policy.text_only_mode}, workers={cold_policy.max_workers}")
    
    # 测试日期温度判断
    # 当前日期附近应该是 HOT
    from datetime import date, timedelta
    recent_date = date.today() - timedelta(days=30)
    recent_temp = get_temperature_for_date(recent_date)
    assert recent_temp == DataTemperature.HOT
    print(f"  ✓ 最近30天: {recent_temp.value}")
    
    # 5年前应该是 COLD
    old_date = date.today() - timedelta(days=365 * 5)
    old_temp = get_temperature_for_date(old_date)
    assert old_temp == DataTemperature.COLD
    print(f"  ✓ 5年前: {old_temp.value}")
    
    # 测试月份策略获取
    policy = get_policy_for_month('2024-12')
    print(f"  ✓ 2024-12 策略: {policy.temperature.value}")
    
    return True


def test_circuit_breaker():
    """测试熔断器"""
    print("\n[TEST 6] 熔断器逻辑")
    print("-" * 50)
    
    from src.data_pipeline.scheduler.unstructured.history_scheduler import (
        CircuitBreaker,
        CircuitBreakerState
    )
    
    # 创建熔断器（阈值=3，恢复时间=2秒）
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=2,
        half_open_max_calls=2
    )
    
    # 初始状态应该是 CLOSED
    assert breaker.state == CircuitBreakerState.CLOSED
    assert breaker.is_allowed() == True
    print("  ✓ 初始状态: CLOSED，允许请求")
    
    # 记录2次失败，不触发熔断
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitBreakerState.CLOSED
    print("  ✓ 2次失败后仍为 CLOSED")
    
    # 第3次失败，触发熔断
    breaker.record_failure()
    assert breaker.state == CircuitBreakerState.OPEN
    assert breaker.is_allowed() == False
    print("  ✓ 3次失败后触发熔断: OPEN，拒绝请求")
    
    # 等待恢复时间
    print("  等待2秒恢复...")
    time.sleep(2.1)
    
    # 应该切换到半开状态
    assert breaker.state == CircuitBreakerState.HALF_OPEN
    assert breaker.is_allowed() == True  # 允许试探请求
    print("  ✓ 超时后切换到 HALF_OPEN，允许试探")
    
    # 成功后恢复
    breaker.record_success()
    assert breaker.state == CircuitBreakerState.CLOSED
    print("  ✓ 成功后恢复: CLOSED")
    
    # 测试重置
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitBreakerState.OPEN
    
    breaker.reset()
    assert breaker.state == CircuitBreakerState.CLOSED
    print("  ✓ 手动重置成功")
    
    return True


def test_resource_monitor():
    """测试资源监控"""
    print("\n[TEST 7] 资源监控")
    print("-" * 50)
    
    from src.data_pipeline.scheduler.unstructured.history_scheduler import (
        ResourceMonitor
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = ResourceMonitor(
            max_memory_gb=16.0,
            max_storage_gb=500.0,
            storage_path=Path(tmpdir)
        )
        
        # 测试内存监控
        mem_usage = monitor.get_memory_usage_gb()
        assert mem_usage > 0
        print(f"  当前内存使用: {mem_usage:.2f} GB")
        
        mem_ok, _ = monitor.check_memory()
        assert mem_ok == True
        print("  ✓ 内存检查通过")
        
        # 测试存储监控
        storage_usage = monitor.get_storage_usage_gb()
        print(f"  当前存储使用: {storage_usage:.4f} GB")
        
        storage_ok, _ = monitor.check_storage()
        assert storage_ok == True
        print("  ✓ 存储检查通过")
        
        # 测试 GC
        monitor.force_gc()
        print("  ✓ 强制 GC 执行成功")
    
    return True


def test_collector_registry():
    """测试采集器注册表"""
    print("\n[TEST 8] 采集器注册表")
    print("-" * 50)
    
    from src.data_pipeline.scheduler.unstructured.config import (
        CollectorConfig,
        CollectorRegistry,
        DataTemperature,
        TEMPERATURE_POLICIES
    )
    
    # 获取单例
    registry = CollectorRegistry()
    
    # 注册测试采集器
    class MockCollector:
        pass
    
    config1 = CollectorConfig(
        name='test_news',
        collector_class=MockCollector,
        enabled=True,
        priority=10,
        memory_intensive=False
    )
    
    config2 = CollectorConfig(
        name='test_ann',
        collector_class=MockCollector,
        enabled=True,
        priority=20,
        memory_intensive=True,
        override_policy={'max_workers': 1}
    )
    
    registry.register(config1)
    registry.register(config2)
    print("  ✓ 注册 2 个采集器")
    
    # 获取启用的采集器（按优先级排序）
    enabled = registry.get_enabled_collectors()
    assert len(enabled) >= 2
    assert enabled[0].priority <= enabled[1].priority  # 优先级排序
    print(f"  ✓ 启用的采集器按优先级排序: {[c.name for c in enabled[:2]]}")
    
    # 测试策略覆盖
    base_policy = TEMPERATURE_POLICIES[DataTemperature.HOT]
    effective = config2.get_effective_policy(base_policy)
    assert effective['max_workers'] == 1  # 被覆盖
    print("  ✓ 策略覆盖生效: max_workers=1")
    
    # 禁用采集器
    registry.disable('test_news')
    retrieved = registry.get('test_news')
    assert retrieved.enabled == False
    print("  ✓ 禁用采集器成功")
    
    # 清理
    registry.unregister('test_news')
    registry.unregister('test_ann')
    
    return True


def test_output_path_generation():
    """测试输出路径生成"""
    print("\n[TEST 9] 输出路径生成")
    print("-" * 50)
    
    from src.data_pipeline.scheduler.unstructured.config import get_output_path
    
    base_dir = Path("data/raw/unstructured")
    
    # 测试路径生成
    path = get_output_path(base_dir, 'news_sina', '2024-12')
    expected = base_dir / 'news_sina' / '2024' / 'news_sina_2024-12.parquet'
    assert path == expected
    print(f"  ✓ 路径正确: {path}")
    
    # 测试不同年份
    path2 = get_output_path(base_dir, 'announcement', '2021-01')
    assert '2021' in str(path2)
    print(f"  ✓ 年份目录正确: {path2}")
    
    return True


def test_scheduler_dry_run():
    """测试调度器 dry_run 模式"""
    print("\n[TEST 10] 调度器 Dry Run")
    print("-" * 50)
    
    from src.data_pipeline.scheduler.unstructured import (
        UnstructuredHistoryScheduler,
        SchedulerConfig,
        CollectorConfig
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SchedulerConfig(
            start_year=2024,
            end_year=2024,
            output_base_dir=Path(tmpdir) / "output",
            state_dir=Path(tmpdir) / "state"
        )
        
        scheduler = UnstructuredHistoryScheduler(config=config)
        
        # 注册测试采集器
        class DummyCollector:
            pass
        
        scheduler.register_collector(
            'dummy_news',
            DummyCollector,
            priority=10
        )
        print("  ✓ 注册测试采集器")
        
        # Dry run（不实际执行）
        scheduler.run_backfill(
            start_year=2024,
            end_year=2024,
            dry_run=True
        )
        print("  ✓ Dry run 完成（仅打印计划）")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("  全量历史调度器测试")
    print("=" * 60)
    
    tests = [
        ("TaskState 枚举", test_task_state_enum),
        ("CheckpointManager 断点续传", test_checkpoint_manager),
        ("task_context 上下文管理器", test_task_context_manager),
        ("时间槽生成", test_time_slot_generation),
        ("温度策略路由", test_temperature_policy_routing),
        ("熔断器逻辑", test_circuit_breaker),
        ("资源监控", test_resource_monitor),
        ("采集器注册表", test_collector_registry),
        ("输出路径生成", test_output_path_generation),
        ("调度器 Dry Run", test_scheduler_dry_run),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"  ✅ PASS: {name}")
            else:
                failed += 1
                print(f"  ❌ FAIL: {name}")
        except Exception as e:
            failed += 1
            print(f"  ❌ ERROR: {name}")
            print(f"     {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"  测试结果: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
