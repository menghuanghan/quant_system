"""
结构化数据合并器 - GPU加速版本

使用cuDF实现GPU加速的增量数据与全量数据合并功能
支持：
- 单文件合并
- 目录结构合并（按实体拆分的文件）
- 日期格式统一
- 排序和去重
- 完整的GPU加速处理流程
"""

import os
import re
import time
import logging
import traceback
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

# GPU加速核心库
import cudf
import cupy as cp

# pandas用于最终保存
import pandas as pd
import pyarrow.parquet as pq

from .config import (
    MergeConfig,
    MergeMode,
    DateSortOrder,
    MergeTask,
    MERGE_TASKS_BY_DOMAIN,
    get_merge_tasks_by_domain,
    get_all_merge_tasks,
    get_merge_task_by_name,
)

# 配置日志
logger = logging.getLogger(__name__)


class MergeStatus(Enum):
    """合并状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class MergeResult:
    """合并结果"""
    task_name: str
    domain: str
    status: MergeStatus
    file_path: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    records_before: int = 0          # 合并前全量数据记录数
    records_increment: int = 0       # 增量数据记录数
    records_after: int = 0           # 合并后记录数
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    gpu_accelerated: bool = True     # 是否使用了GPU加速
    
    @property
    def duration_seconds(self) -> float:
        """执行耗时（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_name": self.task_name,
            "domain": self.domain,
            "status": self.status.value,
            "file_path": self.file_path,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "records_before": self.records_before,
            "records_increment": self.records_increment,
            "records_after": self.records_after,
            "gpu_accelerated": self.gpu_accelerated,
            "error_message": self.error_message,
        }


@dataclass
class MergeProgress:
    """合并进度"""
    total_tasks: int = 0
    completed_tasks: int = 0
    success_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    current_task: Optional[str] = None
    current_domain: Optional[str] = None
    start_time: Optional[datetime] = None
    results: List[MergeResult] = field(default_factory=list)
    
    @property
    def progress_percent(self) -> float:
        """完成百分比"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    def add_result(self, result: MergeResult):
        """添加任务结果"""
        self.results.append(result)
        self.completed_tasks += 1
        if result.status == MergeStatus.SUCCESS:
            self.success_tasks += 1
        elif result.status == MergeStatus.FAILED:
            self.failed_tasks += 1
        elif result.status == MergeStatus.SKIPPED:
            self.skipped_tasks += 1


class GPUDataMerger:
    """
    GPU加速数据合并器
    
    使用cuDF将增量数据合并到全量数据中，充分利用GPU加速
    
    使用示例：
        merger = GPUDataMerger(
            inc_data_dir="data/raw/inc_structured",
            full_data_dir="data/raw/structured",
        )
        
        # 合并全部数据域
        merger.merge_all()
        
        # 合并指定数据域
        merger.merge_domain("metadata")
        
        # 合并指定任务
        merger.merge_task("metadata", "trade_calendar")
    """
    
    def __init__(
        self,
        inc_data_dir: str = "data/raw/inc_structured",
        full_data_dir: str = "data/raw/structured",
        output_dir: str = None,
        backup_enabled: bool = False,
        backup_dir: str = "data/raw/structured_backup",
        dry_run: bool = False,
    ):
        """
        初始化GPU数据合并器
        
        Args:
            inc_data_dir: 增量数据目录
            full_data_dir: 全量数据目录
            output_dir: 输出目录（如果指定，则输出到该目录而不是覆盖full_data_dir）
            backup_enabled: 是否启用备份
            backup_dir: 备份目录
            dry_run: 是否干运行（不实际写入）
        """
        self.inc_data_dir = Path(inc_data_dir)
        self.full_data_dir = Path(full_data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.full_data_dir
        self.backup_enabled = backup_enabled
        self.backup_dir = Path(backup_dir)
        self.dry_run = dry_run
        
        # 合并进度
        self.progress = MergeProgress()
        
        # 验证GPU可用性
        self._verify_gpu()
        
        logger.info(
            f"GPU数据合并器初始化完成: "
            f"增量目录={inc_data_dir}, "
            f"全量目录={full_data_dir}, "
            f"输出目录={self.output_dir}, "
            f"GPU加速=启用"
        )
    
    def _verify_gpu(self):
        """验证GPU可用性"""
        try:
            # 检查CUDA是否可用
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                raise RuntimeError("未检测到CUDA设备")
            
            # 获取GPU信息
            device = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name']
            total_memory = props['totalGlobalMem'] / (1024**3)  # GB
            
            logger.info(f"GPU: {gpu_name}, 显存: {total_memory:.1f} GB")
            
        except Exception as e:
            logger.error(f"GPU验证失败: {e}")
            raise RuntimeError(f"GPU不可用: {e}")
    
    def _clear_gpu_memory(self):
        """清理GPU显存"""
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    
    def _read_parquet_gpu(self, file_path: Path) -> cudf.DataFrame:
        """
        使用cuDF读取parquet文件（GPU加速）
        
        Args:
            file_path: 文件路径
            
        Returns:
            cudf.DataFrame
        """
        return cudf.read_parquet(str(file_path))
    
    def _save_parquet_gpu(self, gdf: cudf.DataFrame, file_path: Path):
        """
        保存cudf.DataFrame为parquet文件
        
        Args:
            gdf: cudf DataFrame
            file_path: 输出文件路径
        """
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 直接使用cuDF保存parquet
        gdf.to_parquet(str(file_path), compression='snappy', index=False)
    
    def _detect_date_format(self, date_value: Any) -> str:
        """
        检测日期格式
        
        Args:
            date_value: 日期值样本
            
        Returns:
            格式标识: 'YYYYMMDD', 'YYYY-MM-DD', 或 'unknown'
        """
        if date_value is None:
            return 'unknown'
        
        date_str = str(date_value)
        
        if re.match(r'^\d{8}$', date_str):
            return 'YYYYMMDD'
        elif re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
            return 'YYYY-MM-DD'
        else:
            return 'unknown'
    
    def _normalize_date_column_gpu(
        self, 
        gdf: cudf.DataFrame, 
        date_field: str,
        target_format: str
    ) -> cudf.DataFrame:
        """
        在GPU上统一日期格式
        
        Args:
            gdf: cudf DataFrame
            date_field: 日期字段名
            target_format: 目标格式 ('YYYYMMDD' 或 'YYYY-MM-DD')
            
        Returns:
            处理后的cudf DataFrame
        """
        if date_field not in gdf.columns or len(gdf) == 0:
            return gdf
        
        try:
            # 获取当前格式
            first_val = gdf[date_field].iloc[0]
            if first_val is None:
                return gdf
            
            current_format = self._detect_date_format(first_val)
            
            # 如果格式相同，无需转换
            if current_format == target_format:
                return gdf
            
            if current_format == 'unknown' or target_format == 'unknown':
                return gdf
            
            # 执行格式转换
            gdf = gdf.copy()
            
            if current_format == 'YYYY-MM-DD' and target_format == 'YYYYMMDD':
                # YYYY-MM-DD -> YYYYMMDD: 使用字符串操作
                gdf[date_field] = gdf[date_field].astype(str).str.replace('-', '', regex=False)
                
            elif current_format == 'YYYYMMDD' and target_format == 'YYYY-MM-DD':
                # YYYYMMDD -> YYYY-MM-DD: 先转为datetime再格式化
                date_col = gdf[date_field].astype(str)
                # 构建YYYY-MM-DD格式
                gdf[date_field] = (
                    date_col.str.slice(0, 4) + '-' + 
                    date_col.str.slice(4, 6) + '-' + 
                    date_col.str.slice(6, 8)
                )
            
            logger.debug(f"日期格式转换完成: {current_format} -> {target_format}")
            return gdf
            
        except Exception as e:
            logger.warning(f"日期格式转换失败: {e}，保持原格式")
            return gdf
    
    def _merge_dataframes_gpu(
        self,
        full_gdf: cudf.DataFrame,
        inc_gdf: cudf.DataFrame,
        task: MergeTask,
    ) -> cudf.DataFrame:
        """
        在GPU上合并两个DataFrame
        
        Args:
            full_gdf: 全量数据 cudf DataFrame
            inc_gdf: 增量数据 cudf DataFrame
            task: 合并任务配置
            
        Returns:
            合并后的 cudf DataFrame
        """
        # 统一日期格式（以全量数据为准）
        if task.date_field and task.date_field in full_gdf.columns and len(full_gdf) > 0:
            first_val = full_gdf[task.date_field].iloc[0]
            if first_val is not None:
                target_format = self._detect_date_format(first_val)
                inc_gdf = self._normalize_date_column_gpu(inc_gdf, task.date_field, target_format)
        
        # 确保列对齐
        all_columns = list(set(full_gdf.columns.tolist() + inc_gdf.columns.tolist()))
        
        # 为缺失列添加空值
        for col in all_columns:
            if col not in full_gdf.columns:
                full_gdf[col] = None
            if col not in inc_gdf.columns:
                inc_gdf[col] = None
        
        # 统一列顺序
        column_order = full_gdf.columns.tolist()
        inc_gdf = inc_gdf[column_order]
        
        # 合并数据 (GPU加速的concat)
        # 根据排序方式决定合并顺序：
        # - 升序(ASC): 增量数据是更早的数据，应该放在前面 [inc, full]
        # - 降序(DESC): 增量数据是更早的数据，应该放在后面 [full, inc]
        if task.sort_order == DateSortOrder.DESC:
            # 降序：新数据在前，旧数据在后 -> [全量(新), 增量(旧)]
            merged_gdf = cudf.concat([full_gdf, inc_gdf], ignore_index=True)
        else:
            # 升序或无序：旧数据在前，新数据在后 -> [增量(旧), 全量(新)]
            merged_gdf = cudf.concat([inc_gdf, full_gdf], ignore_index=True)
        
        # 去重 (GPU加速)
        if task.dedup_keys:
            valid_keys = [k for k in task.dedup_keys if k in merged_gdf.columns]
            if valid_keys:
                merged_gdf = merged_gdf.drop_duplicates(subset=valid_keys, keep='last')
        
        # 排序 (GPU加速) - 确保最终顺序正确
        if task.date_field and task.sort_order != DateSortOrder.NONE:
            if task.date_field in merged_gdf.columns:
                ascending = task.sort_order == DateSortOrder.ASC
                merged_gdf = merged_gdf.sort_values(by=task.date_field, ascending=ascending)
        
        # 重置索引
        merged_gdf = merged_gdf.reset_index(drop=True)
        
        return merged_gdf
    
    def _backup_file(self, file_path: Path):
        """备份文件"""
        if not self.backup_enabled:
            return
        
        if not file_path.exists():
            return
        
        # 计算备份路径
        rel_path = file_path.relative_to(self.full_data_dir)
        backup_path = self.backup_dir / rel_path
        
        # 确保备份目录存在
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        import shutil
        shutil.copy2(file_path, backup_path)
        logger.debug(f"已备份: {file_path} -> {backup_path}")
    
    def merge_single_file(
        self,
        task: MergeTask,
    ) -> MergeResult:
        """
        合并单个文件（GPU加速）
        
        Args:
            task: 合并任务
            
        Returns:
            合并结果
        """
        result = MergeResult(
            task_name=task.name,
            domain=task.domain,
            status=MergeStatus.RUNNING,
            start_time=datetime.now(),
            gpu_accelerated=True,
        )
        
        try:
            # 构建文件路径
            full_file = self.full_data_dir / task.domain / task.output_file
            inc_file = self.inc_data_dir / task.domain / task.output_file
            output_file = self.output_dir / task.domain / task.output_file
            
            result.file_path = str(output_file)
            
            # 检查增量文件是否存在
            if not inc_file.exists():
                logger.info(f"增量文件不存在，跳过: {inc_file}")
                result.status = MergeStatus.SKIPPED
                result.end_time = datetime.now()
                return result
            
            # 使用GPU读取增量数据
            inc_gdf = self._read_parquet_gpu(inc_file)
            result.records_increment = len(inc_gdf)
            
            if full_file.exists():
                # 使用GPU读取全量数据
                full_gdf = self._read_parquet_gpu(full_file)
                result.records_before = len(full_gdf)
            else:
                # 全量文件不存在，创建空DataFrame
                full_gdf = cudf.DataFrame()
                result.records_before = 0
            
            # GPU加速合并
            if len(full_gdf) > 0:
                merged_gdf = self._merge_dataframes_gpu(full_gdf, inc_gdf, task)
            else:
                # 如果全量数据为空，直接使用增量数据
                merged_gdf = inc_gdf
            
            result.records_after = len(merged_gdf)
            
            # 保存结果
            if not self.dry_run:
                self._backup_file(output_file)
                self._save_parquet_gpu(merged_gdf, output_file)
                logger.info(
                    f"[GPU] 合并完成 [{task.domain}/{task.name}]: "
                    f"{result.records_before} + {result.records_increment} -> {result.records_after} 条记录"
                )
            else:
                logger.info(
                    f"[GPU][干运行] 合并完成 [{task.domain}/{task.name}]: "
                    f"{result.records_before} + {result.records_increment} -> {result.records_after} 条记录"
                )
            
            result.status = MergeStatus.SUCCESS
            
            # 清理GPU显存
            del inc_gdf, full_gdf, merged_gdf
            self._clear_gpu_memory()
            
        except Exception as e:
            logger.error(f"合并失败 [{task.domain}/{task.name}]: {e}")
            logger.error(traceback.format_exc())
            result.status = MergeStatus.FAILED
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()
            self._clear_gpu_memory()
        
        result.end_time = datetime.now()
        return result
    
    def merge_directory(
        self,
        task: MergeTask,
    ) -> MergeResult:
        """
        合并目录结构（包含多个按实体拆分的文件，GPU加速）
        
        Args:
            task: 合并任务
            
        Returns:
            合并结果
        """
        result = MergeResult(
            task_name=task.name,
            domain=task.domain,
            status=MergeStatus.RUNNING,
            start_time=datetime.now(),
            gpu_accelerated=True,
        )
        
        try:
            # 构建目录路径
            full_dir = self.full_data_dir / task.domain / task.output_file
            inc_dir = self.inc_data_dir / task.domain / task.output_file
            output_dir = self.output_dir / task.domain / task.output_file
            
            result.file_path = str(output_dir)
            
            # 检查增量目录是否存在
            if not inc_dir.exists():
                logger.info(f"增量目录不存在，跳过: {inc_dir}")
                result.status = MergeStatus.SKIPPED
                result.end_time = datetime.now()
                return result
            
            # 获取增量文件列表
            inc_files = list(inc_dir.glob("*.parquet"))
            if not inc_files:
                logger.info(f"增量目录为空，跳过: {inc_dir}")
                result.status = MergeStatus.SKIPPED
                result.end_time = datetime.now()
                return result
            
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 统计
            total_before = 0
            total_increment = 0
            total_after = 0
            files_merged = 0
            files_failed = 0
            
            # 逐个文件合并
            for i, inc_file in enumerate(inc_files):
                try:
                    file_name = inc_file.name
                    full_file = full_dir / file_name
                    output_file = output_dir / file_name
                    
                    # 使用GPU读取增量数据
                    inc_gdf = self._read_parquet_gpu(inc_file)
                    total_increment += len(inc_gdf)
                    
                    if full_file.exists():
                        # 使用GPU读取全量数据
                        full_gdf = self._read_parquet_gpu(full_file)
                        total_before += len(full_gdf)
                        
                        # GPU加速合并
                        merged_gdf = self._merge_dataframes_gpu(full_gdf, inc_gdf, task)
                        del full_gdf
                    else:
                        # 全量文件不存在，直接使用增量数据
                        merged_gdf = inc_gdf
                    
                    total_after += len(merged_gdf)
                    
                    # 保存
                    if not self.dry_run:
                        self._backup_file(output_file)
                        self._save_parquet_gpu(merged_gdf, output_file)
                    
                    files_merged += 1
                    
                    # 清理
                    del inc_gdf, merged_gdf
                    
                    # 每处理一定数量的文件后清理GPU显存
                    if (i + 1) % 100 == 0:
                        self._clear_gpu_memory()
                        logger.debug(f"已处理 {i + 1}/{len(inc_files)} 个文件")
                    
                except Exception as e:
                    logger.warning(f"合并文件失败 {inc_file}: {e}")
                    files_failed += 1
            
            result.records_before = total_before
            result.records_increment = total_increment
            result.records_after = total_after
            
            if files_failed == 0:
                result.status = MergeStatus.SUCCESS
                logger.info(
                    f"[GPU] 目录合并完成 [{task.domain}/{task.name}]: "
                    f"{files_merged} 个文件, "
                    f"{total_before} + {total_increment} -> {total_after} 条记录"
                )
            else:
                result.status = MergeStatus.SUCCESS  # 部分成功也算成功
                result.error_message = f"{files_failed} 个文件合并失败"
                logger.warning(
                    f"[GPU] 目录合并部分完成 [{task.domain}/{task.name}]: "
                    f"{files_merged}/{files_merged + files_failed} 个文件成功"
                )
            
            # 清理GPU显存
            self._clear_gpu_memory()
            
        except Exception as e:
            logger.error(f"目录合并失败 [{task.domain}/{task.name}]: {e}")
            logger.error(traceback.format_exc())
            result.status = MergeStatus.FAILED
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()
            self._clear_gpu_memory()
        
        result.end_time = datetime.now()
        return result
    
    def merge_task(
        self,
        domain: str,
        task_name: str,
    ) -> MergeResult:
        """
        执行指定的合并任务
        
        Args:
            domain: 数据域名称
            task_name: 任务名称
            
        Returns:
            合并结果
        """
        task = get_merge_task_by_name(domain, task_name)
        if task is None:
            logger.error(f"任务不存在: {domain}/{task_name}")
            return MergeResult(
                task_name=task_name,
                domain=domain,
                status=MergeStatus.FAILED,
                error_message=f"任务不存在: {domain}/{task_name}",
            )
        
        if not task.enabled:
            logger.info(f"任务已禁用，跳过: {domain}/{task_name}")
            return MergeResult(
                task_name=task_name,
                domain=domain,
                status=MergeStatus.SKIPPED,
                error_message="任务已禁用",
            )
        
        self.progress.current_task = task_name
        self.progress.current_domain = domain
        
        if task.is_directory:
            result = self.merge_directory(task)
        else:
            result = self.merge_single_file(task)
        
        self.progress.add_result(result)
        return result
    
    def merge_domain(
        self,
        domain: str,
        task_names: Optional[List[str]] = None,
    ) -> List[MergeResult]:
        """
        执行指定数据域的所有合并任务
        
        Args:
            domain: 数据域名称
            task_names: 指定要执行的任务名称列表，None表示执行全部
            
        Returns:
            合并结果列表
        """
        tasks = get_merge_tasks_by_domain(domain)
        if not tasks:
            logger.warning(f"数据域没有合并任务: {domain}")
            return []
        
        # 筛选任务
        if task_names:
            tasks = [t for t in tasks if t.name in task_names]
        
        # 只执行启用的任务
        tasks = [t for t in tasks if t.enabled]
        
        logger.info(f"[GPU] 开始合并数据域 [{domain}]: {len(tasks)} 个任务")
        
        results = []
        for task in tasks:
            result = self.merge_task(domain, task.name)
            results.append(result)
        
        # 统计
        success = sum(1 for r in results if r.status == MergeStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == MergeStatus.FAILED)
        skipped = sum(1 for r in results if r.status == MergeStatus.SKIPPED)
        
        logger.info(
            f"数据域 [{domain}] 合并完成: "
            f"成功 {success}, 失败 {failed}, 跳过 {skipped}"
        )
        
        return results
    
    def merge_all(
        self,
        domains: Optional[List[str]] = None,
    ) -> List[MergeResult]:
        """
        执行所有数据域的合并任务
        
        Args:
            domains: 指定要执行的数据域列表，None表示执行全部
            
        Returns:
            合并结果列表
        """
        self.progress = MergeProgress()
        self.progress.start_time = datetime.now()
        
        # 确定要处理的数据域
        if domains is None:
            domains = list(MERGE_TASKS_BY_DOMAIN.keys())
        
        # 计算总任务数
        total_tasks = 0
        for domain in domains:
            tasks = get_merge_tasks_by_domain(domain)
            total_tasks += len([t for t in tasks if t.enabled])
        
        self.progress.total_tasks = total_tasks
        
        logger.info(f"[GPU] 开始合并所有数据域: {len(domains)} 个域, {total_tasks} 个任务")
        
        results = []
        for domain in domains:
            domain_results = self.merge_domain(domain)
            results.extend(domain_results)
        
        # 最终统计
        success = sum(1 for r in results if r.status == MergeStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == MergeStatus.FAILED)
        skipped = sum(1 for r in results if r.status == MergeStatus.SKIPPED)
        
        total_records_merged = sum(r.records_increment for r in results if r.status == MergeStatus.SUCCESS)
        
        logger.info(
            f"[GPU] 全部合并完成: "
            f"成功 {success}, 失败 {failed}, 跳过 {skipped}, "
            f"共合并 {total_records_merged:,} 条记录"
        )
        
        return results
    
    def generate_report(self, results: List[MergeResult]) -> pd.DataFrame:
        """
        生成合并报告
        
        Args:
            results: 合并结果列表
            
        Returns:
            报告DataFrame
        """
        data = [r.to_dict() for r in results]
        return pd.DataFrame(data)


# 保持向后兼容的别名
DataMerger = GPUDataMerger


__all__ = [
    "GPUDataMerger",
    "DataMerger",
    "MergeResult",
    "MergeProgress",
    "MergeStatus",
]
