"""
增量数据采集调度器

提供IncrementCollectionScheduler类，继承并扩展FullCollectionScheduler
支持按采集模式（全部/仅时间相关/仅时间无关）进行增量数据采集
"""

import os
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from .config import (
    CollectionTask,
    DataCategory,
    StockScope,
    CollectionFrequency,
    CollectionMode,
    ALL_TASKS,
    TASKS_BY_DOMAIN,
    DOMAIN_NAMES,
    DEFAULT_INCREMENT_OUTPUT_DIR,
    get_enabled_tasks,
    get_tasks_sorted_by_priority,
    get_tasks_by_domain,
    get_tasks_by_mode,
    get_tasks_by_mode_and_domain,
)

# 从全量调度器导入基础类
from src.data_pipeline.scheduler.structured.full.collection.scheduler import (
    FullCollectionScheduler,
    TaskStatus,
    TaskResult,
    CollectionProgress,
)

# 配置日志
logger = logging.getLogger(__name__)


class IncrementCollectionScheduler(FullCollectionScheduler):
    """
    增量数据采集调度器
    
    继承自FullCollectionScheduler，额外支持：
    - 按采集模式筛选任务（全部/仅时间相关/仅时间无关）
    - 默认输出到增量数据目录
    
    使用示例：
        # 采集所有结构化非实时数据
        scheduler = IncrementCollectionScheduler(
            start_date='20190101',
            end_date='20201231',
            mode=CollectionMode.ALL,
        )
        scheduler.run_all()
        
        # 只采集时间相关数据
        scheduler = IncrementCollectionScheduler(
            start_date='20190101',
            end_date='20201231',
            mode=CollectionMode.TIME_DEPENDENT,
        )
        scheduler.run_all()
        
        # 只采集时间无关数据
        scheduler = IncrementCollectionScheduler(
            start_date='20190101',
            end_date='20201231',
            mode=CollectionMode.TIME_INDEPENDENT,
        )
        scheduler.run_all()
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        output_dir: str = DEFAULT_INCREMENT_OUTPUT_DIR,
        mode: CollectionMode = CollectionMode.ALL,
        max_workers: int = 1,
        retry_failed: bool = True,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        skip_existing: bool = False,
        progress_file: Optional[str] = None,
        full_data_dir: str = "data/raw/structured",
    ):
        """
        初始化增量采集调度器
        
        Args:
            start_date: 数据采集开始日期（YYYYMMDD格式）
            end_date: 数据采集结束日期（YYYYMMDD格式），默认为今天
            output_dir: 数据输出目录，默认为增量数据目录
            mode: 采集模式
                - ALL: 采集所有非实时数据
                - TIME_DEPENDENT: 只采集时间相关非实时数据
                - TIME_INDEPENDENT: 只采集时间无关非实时数据
            max_workers: 最大并发数（建议保持1，避免API限流）
            retry_failed: 是否重试失败的任务
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            skip_existing: 是否跳过已存在的文件（断点续采）
            progress_file: 进度文件路径（用于断点续采）
            full_data_dir: 全量数据目录（用于获取股票列表等元数据）
        """
        # 调用父类初始化
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            max_workers=max_workers,
            retry_failed=retry_failed,
            max_retries=max_retries,
            retry_delay=retry_delay,
            skip_existing=skip_existing,
            progress_file=progress_file,
        )
        
        self.mode = mode
        self.full_data_dir = Path(full_data_dir)
        
        logger.info(
            f"增量采集调度器初始化完成: "
            f"日期范围={start_date}~{self.end_date}, "
            f"输出目录={output_dir}, "
            f"采集模式={mode.value}"
        )
    
    def _get_stock_list(self) -> List[str]:
        """
        获取全A股股票列表
        
        优先从全量数据目录读取，然后从增量数据目录读取，最后使用API获取
        """
        if self._stock_list_cache is not None:
            return self._stock_list_cache['ts_code'].tolist()
        
        logger.info("获取全A股股票列表...")
        
        # 首先尝试从全量数据目录读取
        stock_list_path = self.full_data_dir / "metadata" / "stock_list_a.parquet"
        if stock_list_path.exists():
            try:
                self._stock_list_cache = pd.read_parquet(stock_list_path)
                logger.info(f"从全量数据目录加载股票列表: {len(self._stock_list_cache)} 只")
                return self._stock_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.warning(f"读取全量数据目录股票列表失败: {e}")
        
        # 然后尝试从增量数据目录读取
        stock_list_path = self.output_dir / "metadata" / "stock_list_a.parquet"
        if stock_list_path.exists():
            try:
                self._stock_list_cache = pd.read_parquet(stock_list_path)
                logger.info(f"从增量数据目录加载股票列表: {len(self._stock_list_cache)} 只")
                return self._stock_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.warning(f"读取增量数据目录股票列表失败: {e}")
        
        # 使用采集器获取
        if "get_stock_list_a" in self._collector_funcs:
            try:
                self._stock_list_cache = self._collector_funcs["get_stock_list_a"]()
                if not self._stock_list_cache.empty:
                    logger.info(f"从API获取股票列表: {len(self._stock_list_cache)} 只")
                    return self._stock_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.error(f"获取股票列表失败: {e}")
        
        logger.error("无法获取股票列表")
        return []

    def _get_fund_list(self) -> List[str]:
        """获取全量基金列表，优先从全量数据目录读取"""
        if self._fund_list_cache is not None:
            return self._fund_list_cache['ts_code'].tolist()
        
        logger.info("获取全量基金列表...")
        
        # 首先尝试从全量数据目录读取
        fund_list_path = self.full_data_dir / "derivatives" / "fund_basic.parquet"
        if fund_list_path.exists():
            try:
                self._fund_list_cache = pd.read_parquet(fund_list_path)
                logger.info(f"从全量数据目录加载基金列表: {len(self._fund_list_cache)} 只")
                return self._fund_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.warning(f"读取全量数据目录基金列表失败: {e}")
        
        # 调用父类方法
        return super()._get_fund_list()

    def _get_index_list(self) -> List[str]:
        """获取全量指数列表，优先从全量数据目录读取"""
        if self._index_list_cache is not None:
            return self._index_list_cache['ts_code'].tolist()
        
        logger.info("获取全量指数列表...")
        
        # 首先尝试从全量数据目录读取
        index_list_path = self.full_data_dir / "index_benchmark" / "index_basic.parquet"
        if index_list_path.exists():
            try:
                self._index_list_cache = pd.read_parquet(index_list_path)
                logger.info(f"从全量数据目录加载指数列表: {len(self._index_list_cache)} 只")
                return self._index_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.warning(f"读取全量数据目录指数列表失败: {e}")
        
        # 调用父类方法
        return super()._get_index_list()

    def _get_option_list(self) -> List[str]:
        """获取全量期权列表，优先从全量数据目录读取"""
        if self._option_list_cache is not None:
            return self._option_list_cache['ts_code'].tolist()
        
        logger.info("获取全量期权列表...")
        
        # 首先尝试从全量数据目录读取
        opt_list_path = self.full_data_dir / "derivatives" / "opt_basic.parquet"
        if opt_list_path.exists():
            try:
                self._option_list_cache = pd.read_parquet(opt_list_path)
                logger.info(f"从全量数据目录加载期权列表: {len(self._option_list_cache)} 只")
                return self._option_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.warning(f"读取全量数据目录期权列表失败: {e}")
        
        # 调用父类方法
        return super()._get_option_list()

    def _get_bond_list(self) -> List[str]:
        """获取全量可转债列表，优先从全量数据目录读取"""
        if self._bond_list_cache is not None:
            return self._bond_list_cache['ts_code'].tolist()
        
        logger.info("获取全量可转债列表...")
        
        # 首先尝试从全量数据目录读取
        bond_list_path = self.full_data_dir / "derivatives" / "cb_basic.parquet"
        if bond_list_path.exists():
            try:
                self._bond_list_cache = pd.read_parquet(bond_list_path)
                logger.info(f"从全量数据目录加载转债列表: {len(self._bond_list_cache)} 只")
                return self._bond_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.warning(f"读取全量数据目录转债列表失败: {e}")
        
        # 调用父类方法
        return super()._get_bond_list()
    
    def _filter_tasks_by_mode(self, tasks: List[CollectionTask]) -> List[CollectionTask]:
        """
        根据采集模式筛选任务
        
        Args:
            tasks: 原始任务列表
        
        Returns:
            符合采集模式的任务列表
        """
        if self.mode == CollectionMode.ALL:
            return tasks
        elif self.mode == CollectionMode.TIME_DEPENDENT:
            return [t for t in tasks if t.category == DataCategory.TIME_DEPENDENT]
        elif self.mode == CollectionMode.TIME_INDEPENDENT:
            return [t for t in tasks if t.category == DataCategory.TIME_INDEPENDENT]
        else:
            return tasks
    
    def run_domain(
        self, 
        domain: str, 
        task_names: Optional[List[str]] = None
    ) -> List[TaskResult]:
        """
        执行指定数据域的采集任务
        
        覆盖父类方法，增加按采集模式筛选任务的逻辑
        
        Args:
            domain: 数据域名称
            task_names: 指定要执行的任务名称列表，None表示执行全部
        
        Returns:
            任务结果列表
        """
        self._load_collector_funcs()
        
        domain_tasks = get_tasks_by_domain(domain)
        if not domain_tasks:
            logger.warning(f"数据域 {domain} 没有配置采集任务")
            return []
        
        # 过滤任务
        if task_names:
            domain_tasks = [t for t in domain_tasks if t.name in task_names]
        
        # 过滤已禁用和实时任务
        domain_tasks = [t for t in domain_tasks if t.enabled and not t.realtime]
        
        # 按采集模式筛选任务
        domain_tasks = self._filter_tasks_by_mode(domain_tasks)
        
        if not domain_tasks:
            logger.info(
                f"数据域 [{DOMAIN_NAMES.get(domain, domain)}] "
                f"在采集模式 [{self.mode.value}] 下没有匹配的任务"
            )
            return []
        
        # 按优先级排序
        domain_tasks = sorted(domain_tasks, key=lambda x: x.priority)
        
        mode_desc = {
            CollectionMode.ALL: "全部",
            CollectionMode.TIME_DEPENDENT: "时间相关",
            CollectionMode.TIME_INDEPENDENT: "时间无关",
        }
        
        logger.info(
            f"开始采集数据域 [{DOMAIN_NAMES.get(domain, domain)}] "
            f"(模式: {mode_desc.get(self.mode, self.mode.value)}), "
            f"共 {len(domain_tasks)} 个任务"
        )
        
        all_results = []
        for i, task in enumerate(domain_tasks):
            logger.info(
                f"[{i+1}/{len(domain_tasks)}] 执行任务: {task.name} - {task.description}"
            )
            self.progress.current_task = task.name
            self.progress.current_domain = domain
            
            results = self.run_task(task)
            all_results.extend(results)
            
            for result in results:
                self.progress.add_result(result)
        
        # 统计结果
        success_count = sum(1 for r in all_results if r.status == TaskStatus.SUCCESS)
        failed_count = sum(1 for r in all_results if r.status == TaskStatus.FAILED)
        
        logger.info(
            f"数据域 [{DOMAIN_NAMES.get(domain, domain)}] 采集完成: "
            f"成功={success_count}, 失败={failed_count}"
        )
        
        return all_results
    
    def run_all(
        self,
        domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        task_names: Optional[List[str]] = None,
    ) -> CollectionProgress:
        """
        执行全部数据域的增量采集任务
        
        覆盖父类方法，增加按采集模式统计任务数的逻辑
        
        Args:
            domains: 指定要执行的数据域列表，None表示执行全部
            exclude_domains: 要排除的数据域列表
            task_names: 指定要执行的任务名称列表 (可选)
        
        Returns:
            采集进度对象
        """
        self._load_collector_funcs()
        
        # 确定要执行的数据域
        all_domains = list(TASKS_BY_DOMAIN.keys())
        if domains:
            all_domains = [d for d in domains if d in TASKS_BY_DOMAIN]
        if exclude_domains:
            all_domains = [d for d in all_domains if d not in exclude_domains]
        
        # 计算总任务数（考虑采集模式）
        total_tasks = 0
        for d in all_domains:
            tasks = [t for t in TASKS_BY_DOMAIN[d] if t.enabled and not t.realtime]
            tasks = self._filter_tasks_by_mode(tasks)
            if task_names:
                tasks = [t for t in tasks if t.name in task_names]
            total_tasks += len(tasks)
        
        self.progress = CollectionProgress(
            total_tasks=total_tasks,
            start_time=datetime.now()
        )
        
        mode_desc = {
            CollectionMode.ALL: "全部非实时数据",
            CollectionMode.TIME_DEPENDENT: "时间相关非实时数据",
            CollectionMode.TIME_INDEPENDENT: "时间无关非实时数据",
        }
        
        logger.info("=" * 60)
        logger.info(f"增量数据采集开始")
        logger.info(f"日期范围: {self.start_date} ~ {self.end_date}")
        logger.info(f"采集模式: {mode_desc.get(self.mode, self.mode.value)}")
        logger.info(f"数据域: {', '.join(all_domains)}")
        logger.info(f"预计任务数: {total_tasks}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)
        
        # 按数据域依次执行
        for domain in all_domains:
            logger.info("-" * 40)
            self.run_domain(domain, task_names=task_names)
        
        # 保存采集报告
        self._save_collection_report()
        
        logger.info("=" * 60)
        logger.info(f"增量数据采集完成")
        logger.info(
            f"总计: {self.progress.completed_tasks} 个任务, "
            f"成功: {self.progress.success_tasks}, "
            f"失败: {self.progress.failed_tasks}, "
            f"跳过: {self.progress.skipped_tasks}"
        )
        logger.info("=" * 60)
        
        return self.progress
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """获取采集汇总信息（包含采集模式）"""
        summary = super().get_collection_summary()
        summary["mode"] = self.mode.value
        summary["full_data_dir"] = str(self.full_data_dir)
        return summary
