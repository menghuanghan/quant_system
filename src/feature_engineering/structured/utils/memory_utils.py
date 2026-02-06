"""
内存管理工具

提供 CPU 和 GPU 内存管理功能，防止 OOM。
"""

import gc
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def get_gpu_memory_usage() -> dict:
    """
    获取 GPU 显存使用情况
    
    Returns:
        {'used_gb': float, 'total_gb': float, 'free_gb': float}
    """
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()
        
        return {
            'used_gb': used_bytes / (1024 ** 3),
            'total_gb': total_bytes / (1024 ** 3),
            'free_gb': (total_bytes - used_bytes) / (1024 ** 3)
        }
    except ImportError:
        return {'used_gb': 0, 'total_gb': 0, 'free_gb': 0}
    except Exception:
        return {'used_gb': 0, 'total_gb': 0, 'free_gb': 0}


def get_cpu_memory_usage() -> dict:
    """
    获取 CPU 内存使用情况
    
    Returns:
        {'used_gb': float, 'total_gb': float, 'free_gb': float, 'percent': float}
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'used_gb': (mem.total - mem.available) / (1024 ** 3),
            'total_gb': mem.total / (1024 ** 3),
            'free_gb': mem.available / (1024 ** 3),
            'percent': mem.percent
        }
    except ImportError:
        return {'used_gb': 0, 'total_gb': 0, 'free_gb': 0, 'percent': 0}


def force_gc(release_gpu: bool = True, verbose: bool = False):
    """
    强制垃圾回收，释放 CPU 和 GPU 内存
    
    Args:
        release_gpu: 是否释放 GPU 内存池
        verbose: 是否输出日志
    """
    # CPU GC
    gc.collect()
    
    # GPU 内存释放
    if release_gpu:
        try:
            import cupy as cp
            
            # 释放 cupy 内存池
            mempool = cp.get_default_memory_pool()
            before_bytes = mempool.used_bytes()
            mempool.free_all_blocks()
            
            # 释放 pinned memory
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
            
            after_bytes = mempool.used_bytes()
            
            if verbose and before_bytes > after_bytes:
                freed_mb = (before_bytes - after_bytes) / (1024 ** 2)
                logger.info(f"  💾 释放 GPU 显存: {freed_mb:.1f} MB")
        except ImportError:
            pass
        except Exception as e:
            if verbose:
                logger.debug(f"GPU 内存释放失败: {e}")


def log_memory_status(stage: str = ""):
    """
    记录当前内存状态
    
    Args:
        stage: 阶段名称
    """
    cpu_mem = get_cpu_memory_usage()
    gpu_mem = get_gpu_memory_usage()
    
    prefix = f"[{stage}] " if stage else ""
    
    logger.info(
        f"{prefix}内存状态 - "
        f"CPU: {cpu_mem['used_gb']:.1f}/{cpu_mem['total_gb']:.1f}GB ({cpu_mem['percent']:.0f}%)"
    )
    
    if gpu_mem['total_gb'] > 0:
        logger.info(
            f"{prefix}         GPU: {gpu_mem['used_gb']:.2f}/{gpu_mem['total_gb']:.2f}GB"
        )


def estimate_dataframe_memory(df, detailed: bool = False) -> dict:
    """
    估算 DataFrame 内存占用
    
    Args:
        df: DataFrame (pandas 或 cudf)
        detailed: 是否返回详细信息
        
    Returns:
        {'total_mb': float, 'per_col_mb': dict (if detailed)}
    """
    result = {'total_mb': 0}
    
    try:
        # 检测是否是 cuDF DataFrame
        is_cudf = hasattr(df, '_column_names')  # cuDF 特有属性
        
        if is_cudf:
            # cuDF 的内存估算
            mem_bytes = df.memory_usage(deep=True).sum()
        else:
            # pandas 内存估算
            mem_bytes = df.memory_usage(deep=True).sum()
        
        result['total_mb'] = mem_bytes / (1024 ** 2)
        result['total_gb'] = mem_bytes / (1024 ** 3)
        result['rows'] = len(df)
        result['cols'] = len(df.columns)
        
        if detailed:
            result['per_col_mb'] = {}
            for col in df.columns:
                col_bytes = df[[col]].memory_usage(deep=True).sum()
                result['per_col_mb'][col] = col_bytes / (1024 ** 2)
        
        return result
    except Exception as e:
        logger.debug(f"内存估算失败: {e}")
        return result


class MemoryMonitor:
    """
    内存监控器，用于流水线阶段性监控
    
    Usage:
        monitor = MemoryMonitor(threshold_gb=12)
        
        with monitor.track("Step 1"):
            # do something
            
        monitor.report()
    """
    
    def __init__(self, threshold_gb: float = 12.0, auto_gc: bool = True):
        """
        初始化
        
        Args:
            threshold_gb: 内存警告阈值 (GB)
            auto_gc: 是否在阈值超限时自动 GC
        """
        self.threshold_gb = threshold_gb
        self.auto_gc = auto_gc
        self.stages = []
        self._current_stage = None
    
    def track(self, stage_name: str):
        """上下文管理器，追踪阶段内存使用"""
        return _StageContext(self, stage_name)
    
    def _record_stage(self, stage_name: str, before: dict, after: dict, elapsed: float):
        """记录阶段内存使用"""
        self.stages.append({
            'stage': stage_name,
            'before_cpu_gb': before['cpu']['used_gb'],
            'after_cpu_gb': after['cpu']['used_gb'],
            'delta_cpu_gb': after['cpu']['used_gb'] - before['cpu']['used_gb'],
            'before_gpu_gb': before['gpu']['used_gb'],
            'after_gpu_gb': after['gpu']['used_gb'],
            'delta_gpu_gb': after['gpu']['used_gb'] - before['gpu']['used_gb'],
            'elapsed_s': elapsed
        })
        
        # 检查是否超限
        if after['cpu']['used_gb'] > self.threshold_gb:
            logger.warning(f"⚠️ [{stage_name}] CPU 内存超限: {after['cpu']['used_gb']:.1f} GB > {self.threshold_gb} GB")
            if self.auto_gc:
                force_gc(release_gpu=True, verbose=True)
    
    def report(self):
        """输出内存使用报告"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 内存使用报告")
        logger.info("=" * 60)
        
        for s in self.stages:
            delta_cpu = s['delta_cpu_gb']
            delta_gpu = s['delta_gpu_gb']
            sign_cpu = "+" if delta_cpu >= 0 else ""
            sign_gpu = "+" if delta_gpu >= 0 else ""
            
            logger.info(
                f"  {s['stage']}: CPU {sign_cpu}{delta_cpu:.2f}GB, "
                f"GPU {sign_gpu}{delta_gpu:.2f}GB, "
                f"耗时 {s['elapsed_s']:.1f}s"
            )
        
        logger.info("=" * 60)
    
    @staticmethod
    def get_current_memory():
        """获取当前内存状态"""
        return {
            'cpu': get_cpu_memory_usage(),
            'gpu': get_gpu_memory_usage()
        }


class _StageContext:
    """阶段上下文管理器"""
    
    def __init__(self, monitor: MemoryMonitor, stage_name: str):
        self.monitor = monitor
        self.stage_name = stage_name
        self.before = None
        self.start_time = None
    
    def __enter__(self):
        import time
        self.before = MemoryMonitor.get_current_memory()
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        after = MemoryMonitor.get_current_memory()
        elapsed = time.time() - self.start_time
        self.monitor._record_stage(self.stage_name, self.before, after, elapsed)
        return False
