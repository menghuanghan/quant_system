"""
全量历史调度器 (Full History Scheduler)

企业级调度器，支持：
- 断点续传：中断后自动恢复
- 资源编排：内存监控、并发控制
- 策略路由：热/冷数据差异化处理
- 异常熔断：403/封禁时全局暂停
- 输出验证：Parquet 完整性检查

设计目标：无人值守运行数周，完成 5 年历史数据回填

Usage:
    >>> scheduler = UnstructuredHistoryScheduler()
    >>> scheduler.run_backfill(start_year=2021, end_year=2025)
"""

import gc
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Event, Lock
from typing import Dict, List, Optional, Any, Callable, Type
import psutil

from .checkpoint import (
    CheckpointManager,
    TaskState,
    TaskRecord,
    get_checkpoint_manager
)
from .config import (
    SchedulerConfig,
    TimeSlot,
    DataTemperature,
    TemperaturePolicy,
    CollectorConfig,
    CollectorRegistry,
    generate_time_slots,
    get_policy_for_month,
    get_output_path,
    validate_parquet_file,
    estimate_storage_usage
)

logger = logging.getLogger(__name__)


# ============== 异常定义 ==============

class SchedulerError(Exception):
    """调度器基础异常"""
    pass


class CircuitBreakerError(SchedulerError):
    """熔断异常"""
    pass


class RateLimitError(SchedulerError):
    """频率限制异常"""
    pass


class StorageLimitError(SchedulerError):
    """存储限制异常"""
    pass


class DependencyError(SchedulerError):
    """依赖缺失异常"""
    pass


# ============== 熔断器 ==============

class CircuitBreakerState(str, Enum):
    """熔断器状态"""
    CLOSED = "closed"       # 正常运行
    OPEN = "open"           # 熔断打开
    HALF_OPEN = "half_open" # 半开（尝试恢复）


class CircuitBreaker:
    """
    熔断器
    
    当连续失败次数超过阈值时，自动熔断，防止持续请求被封禁。
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 300,
        half_open_max_calls: int = 3
    ):
        """
        Args:
            failure_threshold: 触发熔断的连续失败次数
            recovery_timeout: 熔断恢复超时（秒）
            half_open_max_calls: 半开状态最大尝试次数
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = Lock()
    
    @property
    def state(self) -> CircuitBreakerState:
        """获取当前状态"""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                # 检查是否可以切换到半开状态
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitBreakerState.HALF_OPEN
                        self._half_open_calls = 0
                        logger.info("熔断器切换到半开状态，尝试恢复")
            return self._state
    
    def is_allowed(self) -> bool:
        """检查是否允许请求"""
        state = self.state
        if state == CircuitBreakerState.CLOSED:
            return True
        elif state == CircuitBreakerState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        else:  # OPEN
            return False
    
    def record_success(self):
        """记录成功"""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
                logger.info("熔断器恢复正常")
    
    def record_failure(self, error: Optional[Exception] = None):
        """记录失败"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            # 检查是否需要触发熔断
            if self._failure_count >= self.failure_threshold:
                if self._state != CircuitBreakerState.OPEN:
                    self._state = CircuitBreakerState.OPEN
                    logger.warning(
                        f"熔断器触发！连续失败 {self._failure_count} 次，"
                        f"将在 {self.recovery_timeout} 秒后尝试恢复"
                    )
            
            # 半开状态失败，立即回到打开状态
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                logger.warning("半开状态失败，熔断器重新打开")
    
    def reset(self):
        """重置熔断器"""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


# ============== 资源监控 ==============

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(
        self,
        max_memory_gb: float = 8.0,
        max_storage_gb: float = 500.0,
        storage_path: Optional[Path] = None
    ):
        self.max_memory_gb = max_memory_gb
        self.max_storage_gb = max_storage_gb
        self.storage_path = storage_path
    
    def get_memory_usage_gb(self) -> float:
        """获取当前内存使用（GB）"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    
    def get_storage_usage_gb(self) -> float:
        """获取存储使用（GB）"""
        if self.storage_path and self.storage_path.exists():
            total_size = sum(
                f.stat().st_size for f in self.storage_path.rglob('*') if f.is_file()
            )
            return total_size / (1024 ** 3)
        return 0.0
    
    def check_memory(self) -> tuple:
        """检查内存是否超限"""
        usage = self.get_memory_usage_gb()
        is_ok = usage < self.max_memory_gb
        return is_ok, usage
    
    def check_storage(self) -> tuple:
        """检查存储是否超限"""
        usage = self.get_storage_usage_gb()
        is_ok = usage < self.max_storage_gb
        return is_ok, usage
    
    def force_gc(self):
        """强制垃圾回收"""
        gc.collect()
        logger.debug("执行强制垃圾回收")


# ============== 任务执行器 ==============

@dataclass
class TaskResult:
    """任务执行结果"""
    source: str
    month: str
    success: bool
    record_count: int = 0
    file_size_mb: float = 0.0
    error_message: Optional[str] = None
    execution_time: float = 0.0


class TaskExecutor:
    """
    任务执行器
    
    封装单个任务的执行逻辑，包括：
    - 采集器初始化
    - 数据采集
    - 输出验证
    - 错误处理
    """
    
    def __init__(
        self,
        collector_config: CollectorConfig,
        scheduler_config: SchedulerConfig,
        circuit_breaker: CircuitBreaker,
        resource_monitor: ResourceMonitor
    ):
        self.collector_config = collector_config
        self.scheduler_config = scheduler_config
        self.circuit_breaker = circuit_breaker
        self.resource_monitor = resource_monitor
        
        self._collector_instance = None
    
    def _get_collector(self):
        """获取采集器实例"""
        if self._collector_instance is None:
            if self.collector_config.collector_class is None:
                raise SchedulerError(f"采集器类未设置: {self.collector_config.name}")
            self._collector_instance = self.collector_config.collector_class()
        return self._collector_instance
    
    def execute(self, time_slot: TimeSlot) -> TaskResult:
        """
        执行采集任务
        
        Args:
            time_slot: 时间槽
        
        Returns:
            TaskResult 执行结果
        """
        start_time = time.time()
        source_name = self.collector_config.name
        month = time_slot.month
        
        logger.info(f"开始采集: {source_name} / {month}")
        
        try:
            # 检查熔断器
            if not self.circuit_breaker.is_allowed():
                raise CircuitBreakerError(f"熔断器已打开，跳过任务: {source_name}/{month}")
            
            # 检查内存
            mem_ok, mem_usage = self.resource_monitor.check_memory()
            if not mem_ok:
                logger.warning(f"内存使用过高: {mem_usage:.2f}GB，执行GC")
                self.resource_monitor.force_gc()
            
            # 获取有效策略配置
            effective_policy = self.collector_config.get_effective_policy(time_slot.policy)
            
            # 获取采集器
            collector = self._get_collector()
            
            # 构建采集参数
            collect_kwargs = {
                'start_date': time_slot.start_date,
                'end_date': time_slot.end_date,
                'text_only_mode': effective_policy.get('text_only_mode', False),
                'save_pdf': effective_policy.get('save_pdf', False),
                'timeout': effective_policy.get('timeout', 30)
            }
            
            # 执行采集
            result_df = collector.collect(**collect_kwargs)
            
            # 保存结果
            output_path = get_output_path(
                self.scheduler_config.output_base_dir,
                source_name,
                month
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if result_df is not None and len(result_df) > 0:
                result_df.to_parquet(output_path, index=False)
                file_size_mb = output_path.stat().st_size / (1024 ** 2)
                record_count = len(result_df)
            else:
                file_size_mb = 0.0
                record_count = 0
            
            # 验证输出
            if self.scheduler_config.validate_output and output_path.exists():
                is_valid, error_msg = validate_parquet_file(
                    output_path,
                    min_records=1  # 宽松检查，某些月份可能数据较少
                )
                if not is_valid:
                    logger.warning(f"输出验证失败: {error_msg}")
            
            # 记录成功
            self.circuit_breaker.record_success()
            
            execution_time = time.time() - start_time
            logger.info(
                f"采集完成: {source_name}/{month}, "
                f"记录数: {record_count}, "
                f"文件大小: {file_size_mb:.2f}MB, "
                f"耗时: {execution_time:.1f}s"
            )
            
            return TaskResult(
                source=source_name,
                month=month,
                success=True,
                record_count=record_count,
                file_size_mb=file_size_mb,
                execution_time=execution_time
            )
            
        except CircuitBreakerError as e:
            return TaskResult(
                source=source_name,
                month=month,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"采集失败: {source_name}/{month} - {error_msg}")
            
            # 记录失败
            self.circuit_breaker.record_failure(e)
            
            # 检查是否是频率限制错误
            if '403' in str(e) or 'banned' in str(e).lower() or 'rate limit' in str(e).lower():
                logger.error("检测到可能的封禁/频率限制，熔断器将触发")
            
            return TaskResult(
                source=source_name,
                month=month,
                success=False,
                error_message=error_msg,
                execution_time=time.time() - start_time
            )


# ============== 主调度器 ==============

class UnstructuredHistoryScheduler:
    """
    非结构化数据全量历史调度器
    
    核心功能：
    1. 断点续传：基于 CheckpointManager 管理任务状态
    2. 资源编排：内存监控、并发控制、存储限制
    3. 策略路由：根据数据温度差异化配置
    4. 异常熔断：403/封禁时自动暂停
    
    Example:
        >>> # 基础使用
        >>> scheduler = UnstructuredHistoryScheduler()
        >>> scheduler.run_backfill(start_year=2021, end_year=2025)
        >>> 
        >>> # 自定义配置
        >>> config = SchedulerConfig(
        ...     start_year=2023,
        ...     end_year=2025,
        ...     global_max_workers=4
        ... )
        >>> scheduler = UnstructuredHistoryScheduler(config=config)
        >>> scheduler.run_backfill()
        >>> 
        >>> # 仅运行特定数据源
        >>> scheduler.run_backfill(sources=['news_sina'])
    """
    
    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """
        初始化调度器
        
        Args:
            config: 调度器配置
            checkpoint_manager: 检查点管理器（默认使用全局单例）
        """
        self.config = config or SchedulerConfig()
        self.checkpoint = checkpoint_manager or get_checkpoint_manager()
        
        # 初始化组件
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_break_threshold,
            recovery_timeout=self.config.circuit_break_duration
        )
        self.resource_monitor = ResourceMonitor(
            max_memory_gb=self.config.max_memory_gb,
            max_storage_gb=self.config.max_storage_gb,
            storage_path=self.config.output_base_dir
        )
        self.collector_registry = CollectorRegistry()
        
        # 停止信号
        self._stop_event = Event()
        
        # 统计信息
        self._stats = {
            'total_tasks': 0,
            'completed': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }
        
        # 配置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # 配置根日志
        logger.setLevel(log_level)
        
        # 添加文件处理器
        if self.config.log_to_file:
            file_handler = logging.FileHandler(
                self.config.log_file,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)
    
    def register_collector(
        self,
        name: str,
        collector_class: type,
        **kwargs
    ):
        """
        注册采集器
        
        Args:
            name: 采集器名称
            collector_class: 采集器类
            **kwargs: 其他配置（priority, memory_intensive 等）
        """
        config = CollectorConfig(
            name=name,
            collector_class=collector_class,
            **kwargs
        )
        self.collector_registry.register(config)
        logger.info(f"注册采集器: {name}")
    
    def _check_dependencies(self, collector_config: CollectorConfig) -> bool:
        """检查采集器依赖"""
        if collector_config.requires_market_data:
            market_data_path = self.config.market_data_path
            if not market_data_path.exists():
                logger.warning(
                    f"采集器 {collector_config.name} 需要市场数据，"
                    f"但 {market_data_path} 不存在，跳过"
                )
                return False
        return True
    
    def _execute_task(
        self,
        collector_config: CollectorConfig,
        time_slot: TimeSlot
    ) -> TaskResult:
        """执行单个任务"""
        executor = TaskExecutor(
            collector_config=collector_config,
            scheduler_config=self.config,
            circuit_breaker=self.circuit_breaker,
            resource_monitor=self.resource_monitor
        )
        return executor.execute(time_slot)
    
    def _process_result(self, result: TaskResult):
        """处理任务结果"""
        if result.success:
            self._stats['completed'] += 1
            self.checkpoint.update_task(
                source=result.source,
                month=result.month,
                state=TaskState.COMPLETED,
                record_count=result.record_count,
                file_size_mb=result.file_size_mb
            )
        else:
            self._stats['failed'] += 1
            self.checkpoint.update_task(
                source=result.source,
                month=result.month,
                state=TaskState.FAILED,
                error_message=result.error_message
            )
    
    def run_backfill(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        sources: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
        dry_run: bool = False
    ):
        """
        运行全量回填
        
        Args:
            start_year: 开始年份（默认使用配置）
            end_year: 结束年份（默认使用配置）
            sources: 要运行的数据源（默认所有启用的）
            max_workers: 最大并发数（默认使用配置）
            dry_run: 仅打印计划，不实际执行
        
        Example:
            >>> scheduler.run_backfill()  # 全量回填
            >>> scheduler.run_backfill(start_year=2024, end_year=2025)  # 仅最近2年
            >>> scheduler.run_backfill(sources=['news_sina'])  # 仅新闻
        """
        self._stop_event.clear()
        self._stats['start_time'] = datetime.now()
        
        # 参数处理
        start_year = start_year or self.config.start_year
        end_year = end_year or self.config.end_year
        max_workers = max_workers or self.config.global_max_workers
        
        # 生成时间槽（倒序，优先最新数据）
        time_slots = generate_time_slots(start_year, end_year, reverse=True)
        
        # 获取采集器
        if sources:
            collectors = [
                self.collector_registry.get(name) 
                for name in sources 
                if self.collector_registry.get(name)
            ]
        else:
            collectors = self.collector_registry.get_enabled_collectors()
        
        if not collectors:
            logger.error("没有可用的采集器，请先注册采集器")
            return
        
        # 过滤掉依赖不满足的采集器
        collectors = [c for c in collectors if self._check_dependencies(c)]
        
        # 计算任务总数
        total_tasks = len(collectors) * len(time_slots)
        self._stats['total_tasks'] = total_tasks
        
        logger.info(f"=" * 60)
        logger.info(f"全量回填计划")
        logger.info(f"=" * 60)
        logger.info(f"  时间范围: {start_year}-01 ~ {end_year}-12")
        logger.info(f"  时间槽数: {len(time_slots)}")
        logger.info(f"  采集器数: {len(collectors)}")
        logger.info(f"  总任务数: {total_tasks}")
        logger.info(f"  最大并发: {max_workers}")
        logger.info(f"=" * 60)
        
        if dry_run:
            logger.info("Dry run 模式，不执行实际采集")
            self._print_plan(collectors, time_slots)
            return
        
        # 初始化任务
        self.checkpoint.initialize_tasks(
            sources=[c.name for c in collectors],
            months=[slot.month for slot in time_slots],
            skip_completed=True
        )
        
        # 检查存储空间
        storage_ok, storage_usage = self.resource_monitor.check_storage()
        if not storage_ok:
            raise StorageLimitError(
                f"存储空间不足: {storage_usage:.1f}GB / {self.config.max_storage_gb}GB"
            )
        
        # 开始执行
        try:
            self._run_with_executor(collectors, time_slots, max_workers)
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在优雅停止...")
            self._stop_event.set()
        finally:
            self._stats['end_time'] = datetime.now()
            self._print_summary()
    
    def _run_with_executor(
        self,
        collectors: List[CollectorConfig],
        time_slots: List[TimeSlot],
        max_workers: int
    ):
        """使用线程池执行任务"""
        
        # 按数据源分组执行（避免同一数据源过高并发）
        for collector in collectors:
            if self._stop_event.is_set():
                break
            
            logger.info(f"\n{'='*60}")
            logger.info(f"开始采集数据源: {collector.name}")
            logger.info(f"{'='*60}")
            
            # 获取该数据源的有效并发数
            effective_workers = min(
                max_workers,
                collector.get_effective_policy(
                    time_slots[0].policy if time_slots else TemperaturePolicy(DataTemperature.WARM)
                ).get('max_workers', max_workers)
            )
            
            # 内存密集型任务降低并发
            if collector.memory_intensive:
                effective_workers = min(effective_workers, 2)
                logger.info(f"内存密集型任务，并发数降为: {effective_workers}")
            
            # 获取待处理任务
            pending_tasks = self.checkpoint.get_pending_tasks(
                source=collector.name,
                include_failed=True,
                max_retries=3
            )
            
            if not pending_tasks:
                logger.info(f"数据源 {collector.name} 无待处理任务")
                continue
            
            logger.info(f"待处理任务数: {len(pending_tasks)}")
            
            # 使用线程池执行
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures: Dict[Future, tuple] = {}
                
                for source, month, _ in pending_tasks:
                    if self._stop_event.is_set():
                        break
                    
                    # 检查熔断器
                    if not self.circuit_breaker.is_allowed():
                        logger.warning("熔断器已打开，等待恢复...")
                        time.sleep(self.config.circuit_break_duration)
                        continue
                    
                    # 查找对应的时间槽
                    time_slot = next(
                        (s for s in time_slots if s.month == month),
                        None
                    )
                    if not time_slot:
                        continue
                    
                    # 标记为运行中
                    self.checkpoint.update_task(source, month, TaskState.RUNNING)
                    
                    # 提交任务
                    future = executor.submit(
                        self._execute_task,
                        collector,
                        time_slot
                    )
                    futures[future] = (source, month)
                
                # 收集结果
                for future in as_completed(futures):
                    if self._stop_event.is_set():
                        break
                    
                    try:
                        result = future.result(timeout=300)  # 5分钟超时
                        self._process_result(result)
                    except Exception as e:
                        source, month = futures[future]
                        logger.error(f"任务异常: {source}/{month} - {e}")
                        self.checkpoint.update_task(
                            source, month,
                            TaskState.FAILED,
                            error_message=str(e)
                        )
                        self._stats['failed'] += 1
            
            # 数据源间休息
            if collector.memory_intensive:
                logger.info("内存密集型任务完成，执行GC")
                self.resource_monitor.force_gc()
                time.sleep(5)
    
    def _print_plan(self, collectors: List[CollectorConfig], time_slots: List[TimeSlot]):
        """打印执行计划"""
        print("\n执行计划:")
        print("-" * 60)
        
        for collector in collectors:
            print(f"\n  [{collector.name}] 优先级: {collector.priority}")
            print(f"    内存密集: {collector.memory_intensive}")
            print(f"    需要市场数据: {collector.requires_market_data}")
            
            # 按温度分组统计
            temp_counts = {DataTemperature.HOT: 0, DataTemperature.WARM: 0, DataTemperature.COLD: 0}
            for slot in time_slots:
                temp_counts[slot.temperature] += 1
            
            print(f"    时间槽分布:")
            print(f"      HOT:  {temp_counts[DataTemperature.HOT]} 个月")
            print(f"      WARM: {temp_counts[DataTemperature.WARM]} 个月")
            print(f"      COLD: {temp_counts[DataTemperature.COLD]} 个月")
    
    def _print_summary(self):
        """打印执行摘要"""
        duration = (
            (self._stats['end_time'] - self._stats['start_time']).total_seconds()
            if self._stats['start_time'] and self._stats['end_time']
            else 0
        )
        
        print("\n" + "=" * 60)
        print("  执行摘要")
        print("=" * 60)
        print(f"  总任务数:   {self._stats['total_tasks']}")
        print(f"  已完成:     {self._stats['completed']}")
        print(f"  失败:       {self._stats['failed']}")
        print(f"  跳过:       {self._stats['skipped']}")
        print(f"  执行时间:   {duration:.1f} 秒")
        print("=" * 60)
        
        # 显示检查点进度
        self.checkpoint.print_summary()
        
        # 显示存储使用
        storage_stats = estimate_storage_usage(self.config.output_base_dir)
        print(f"\n存储使用: {storage_stats['total_size_gb']:.2f} GB")
        for source, size in storage_stats.get('by_source', {}).items():
            print(f"  - {source}: {size:.2f} GB")
    
    def stop(self):
        """停止调度器"""
        logger.info("停止调度器...")
        self._stop_event.set()
    
    def reset_circuit_breaker(self):
        """重置熔断器"""
        self.circuit_breaker.reset()
        logger.info("熔断器已重置")
    
    def get_progress(self) -> Dict[str, Any]:
        """获取当前进度"""
        checkpoint_progress = self.checkpoint.get_progress()
        return {
            **checkpoint_progress,
            'circuit_breaker_state': self.circuit_breaker.state.value,
            'memory_usage_gb': self.resource_monitor.get_memory_usage_gb(),
            'storage_usage_gb': self.resource_monitor.get_storage_usage_gb()
        }


# ============== 便捷函数 ==============

def run_backfill(
    start_year: int = 2021,
    end_year: int = 2025,
    sources: Optional[List[str]] = None,
    **kwargs
):
    """
    快速运行回填
    
    Example:
        >>> from src.data_pipeline.scheduler.unstructured import run_backfill
        >>> run_backfill(2024, 2025)
    """
    scheduler = UnstructuredHistoryScheduler()
    scheduler.run_backfill(
        start_year=start_year,
        end_year=end_year,
        sources=sources,
        **kwargs
    )


def get_backfill_progress() -> Dict[str, Any]:
    """获取回填进度"""
    checkpoint = get_checkpoint_manager()
    return checkpoint.get_progress()


def reset_failed_tasks(source: Optional[str] = None):
    """重置失败任务"""
    checkpoint = get_checkpoint_manager()
    checkpoint.reset_failed_tasks(source)
