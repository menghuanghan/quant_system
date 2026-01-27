"""
非结构化数据全量采集调度器

统一调度五类非结构化数据的全量采集：
- announcements: 上市公司公告
- events: 事件驱动型数据
- news: 新闻
- policy: 政策
- reports: 研报

支持特性：
- 按月分区滚动采集
- 股票分组批量采集
- 断点续采
- 采集进度跟踪
"""

import os
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from dateutil.relativedelta import relativedelta
import importlib

import pandas as pd

from .config import (
    CollectionTask,
    DataType,
    StockScope,
    StoragePattern,
    ALL_TASKS,
    TASKS_BY_TYPE,
    TYPE_NAMES,
    get_enabled_tasks,
    get_tasks_by_type,
    get_tasks_sorted_by_priority,
)

# 配置日志
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"  # 部分成功


@dataclass
class TaskResult:
    """任务执行结果"""
    task_name: str
    data_type: str
    status: TaskStatus
    year_month: Optional[str] = None
    stock_code: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    records_count: int = 0
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

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
            "data_type": self.data_type,
            "status": self.status.value,
            "year_month": self.year_month,
            "stock_code": self.stock_code,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "records_count": self.records_count,
            "output_path": self.output_path,
            "error_message": self.error_message,
        }


@dataclass
class CollectionProgress:
    """采集进度"""
    total_months: int = 0
    completed_months: int = 0
    total_stocks: int = 0
    processed_stocks: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    total_records: int = 0
    current_task: Optional[str] = None
    current_month: Optional[str] = None
    current_stock: Optional[str] = None
    start_time: Optional[datetime] = None
    results: List[TaskResult] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        """完成百分比"""
        if self.total_months == 0:
            return 0.0
        return (self.completed_months / self.total_months) * 100

    def add_result(self, result: TaskResult):
        """添加任务结果"""
        self.results.append(result)
        self.total_records += result.records_count
        if result.status == TaskStatus.SUCCESS:
            self.success_count += 1
        elif result.status == TaskStatus.FAILED:
            self.failed_count += 1
        elif result.status == TaskStatus.SKIPPED:
            self.skipped_count += 1


@dataclass
class CheckpointData:
    """断点续采检查点数据"""
    start_date: str
    end_date: str
    data_types: List[str]
    last_completed_month: Optional[str] = None
    last_completed_task: Optional[str] = None
    last_completed_stock: Optional[str] = None
    completed_months: List[str] = field(default_factory=list)
    completed_tasks: Dict[str, List[str]] = field(default_factory=dict)  # task -> [year_month]
    completed_stocks: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)  # task -> {year_month -> [stock]}
    updated_at: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckpointData':
        return cls(**data)


class UnstructuredFullCollectionScheduler:
    """
    非结构化数据全量采集调度器
    
    使用示例：
        scheduler = UnstructuredFullCollectionScheduler(
            start_date='2021-01-01',
            end_date='2025-12-31',
            output_dir='data/raw/unstructured'
        )
        
        # 采集全部类型
        scheduler.run_all()
        
        # 仅采集指定类型
        scheduler.run_by_types(['announcements', 'events'])
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = "data/raw/unstructured",
        stock_list_path: str = "data/raw/structured/metadata/stock_list_a.parquet",
        checkpoint_dir: str = "data/checkpoints/unstructured",
        skip_existing: bool = True,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        batch_size: int = 30,
        enable_checkpoint: bool = True,
        stock_limit: Optional[int] = None,
    ):
        """
        初始化调度器
        
        Args:
            start_date: 开始日期（YYYY-MM-DD 或 YYYYMMDD格式）
            end_date: 结束日期
            output_dir: 数据输出目录
            stock_list_path: A股股票列表文件路径
            checkpoint_dir: 检查点目录（用于断点续采）
            skip_existing: 是否跳过已存在的文件
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            batch_size: 每批股票数量
            enable_checkpoint: 是否启用断点续采
            stock_limit: 限制股票数量（None=全部，100=前100只，500=前500只等）
        """
        # 日期标准化
        self.start_date = self._normalize_date(start_date)
        self.end_date = self._normalize_date(end_date)
        self.output_dir = Path(output_dir)
        self.stock_list_path = Path(stock_list_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.skip_existing = skip_existing
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.enable_checkpoint = enable_checkpoint
        self.stock_limit = stock_limit
        
        # 进度跟踪
        self.progress = CollectionProgress()
        
        # 采集器实例缓存
        self._collectors: Dict[str, Any] = {}
        
        # 股票列表缓存
        self._stock_list: Optional[pd.DataFrame] = None
        
        # 检查点数据
        self._checkpoint: Optional[CheckpointData] = None
        
        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"非结构化数据全量采集调度器初始化完成:\n"
            f"  日期范围: {self.start_date} ~ {self.end_date}\n"
            f"  输出目录: {self.output_dir}\n"
            f"  股票范围: {'前' + str(stock_limit) + '只' if stock_limit else '全部'}\n"
            f"  断点续采: {'启用' if enable_checkpoint else '禁用'}\n"
            f"  跳过已存在: {skip_existing}"
        )

    def _normalize_date(self, date_str: str) -> str:
        """标准化日期格式为 YYYY-MM-DD"""
        date_str = date_str.replace('/', '-').replace('.', '-')
        if len(date_str) == 8 and '-' not in date_str:
            # YYYYMMDD -> YYYY-MM-DD
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str

    def _get_month_range(self) -> List[Tuple[str, str]]:
        """
        获取日期范围内的所有月份
        
        Returns:
            List of (year_month, start_date, end_date) tuples
        """
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        months = []
        current = start.replace(day=1)
        
        while current <= end:
            year_month = current.strftime('%Y-%m')
            month_start = max(current, start).strftime('%Y-%m-%d')
            
            # 计算月末
            next_month = current + relativedelta(months=1)
            month_end = min(next_month - timedelta(days=1), end).strftime('%Y-%m-%d')
            
            months.append((year_month, month_start, month_end))
            current = next_month
        
        return months

    def _load_stock_list(self) -> List[str]:
        """加载A股股票列表"""
        if self._stock_list is not None:
            stock_codes = self._stock_list['ts_code'].tolist()
            # 应用股票数量限制
            if self.stock_limit and self.stock_limit < len(stock_codes):
                stock_codes = stock_codes[:self.stock_limit]
            return stock_codes
        
        if not self.stock_list_path.exists():
            logger.warning(f"股票列表文件不存在: {self.stock_list_path}")
            return []
        
        try:
            self._stock_list = pd.read_parquet(self.stock_list_path)
            stock_codes = self._stock_list['ts_code'].tolist()
            
            # 应用股票数量限制
            if self.stock_limit and self.stock_limit < len(stock_codes):
                stock_codes = stock_codes[:self.stock_limit]
                logger.info(f"加载股票列表: {len(stock_codes)} 只股票（限制前{self.stock_limit}只）")
            else:
                logger.info(f"加载股票列表: {len(stock_codes)} 只股票")
            
            return stock_codes
        except Exception as e:
            logger.error(f"加载股票列表失败: {e}")
            return []

    def _get_stock_batches(self, stock_codes: List[str], batch_size: int) -> List[List[str]]:
        """将股票列表分批"""
        return [stock_codes[i:i + batch_size] for i in range(0, len(stock_codes), batch_size)]

    def _get_collector(self, task: CollectionTask) -> Any:
        """获取或创建采集器实例"""
        cache_key = f"{task.collector_module}.{task.collector_class}"
        
        if cache_key not in self._collectors:
            try:
                module = importlib.import_module(task.collector_module)
                collector_class = getattr(module, task.collector_class)
                self._collectors[cache_key] = collector_class()
                logger.debug(f"创建采集器实例: {cache_key}")
            except Exception as e:
                logger.error(f"创建采集器失败 [{cache_key}]: {e}")
                raise
        
        return self._collectors[cache_key]

    def _get_output_path(
        self, 
        task: CollectionTask, 
        year_month: str, 
        stock_code: Optional[str] = None
    ) -> Path:
        """
        获取输出文件路径
        
        存储结构：
        - BY_STOCK: data_type/[subdir]/year/month/stock_code.parquet
        - BY_DATE: data_type/[subdir]/year/month.parquet
        """
        base_dir = self.output_dir / task.data_type.value
        if task.output_subdir:
            base_dir = base_dir / task.output_subdir
        
        year, month = year_month.split('-')
        
        if task.storage_pattern == StoragePattern.BY_STOCK and stock_code:
            # 按股票代码存储
            return base_dir / year / month / f"{stock_code.replace('.', '_')}.parquet"
        elif task.storage_pattern == StoragePattern.BY_DATE:
            # 按日期存储
            return base_dir / year / f"{month}.parquet"
        else:
            # 单文件存储
            return base_dir / f"{task.name}.parquet"

    def _save_dataframe(
        self, 
        df: pd.DataFrame, 
        output_path: Path, 
        append: bool = False
    ) -> bool:
        """保存DataFrame到parquet文件"""
        if df.empty:
            logger.debug(f"跳过保存空数据: {output_path}")
            return True
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if append and output_path.exists():
                existing_df = pd.read_parquet(output_path)
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates()
            
            df.to_parquet(output_path, index=False, compression='snappy')
            logger.debug(f"保存数据: {len(df)} 条 -> {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存parquet失败 [{output_path}]: {e}")
            return False

    def _load_checkpoint(self, data_types: List[str]) -> Optional[CheckpointData]:
        """加载检查点"""
        if not self.enable_checkpoint:
            return None
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.start_date}_{self.end_date}.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            checkpoint = CheckpointData.from_dict(data)
            
            # 检查是否匹配当前采集配置
            if checkpoint.start_date != self.start_date or checkpoint.end_date != self.end_date:
                logger.warning("检查点日期范围不匹配，将重新开始")
                return None
            
            logger.info(
                f"加载检查点:\n"
                f"  已完成月份: {len(checkpoint.completed_months)}\n"
                f"  上次任务: {checkpoint.last_completed_task}\n"
                f"  上次月份: {checkpoint.last_completed_month}"
            )
            return checkpoint
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}")
            return None

    def _save_checkpoint(self, checkpoint: CheckpointData):
        """保存检查点"""
        if not self.enable_checkpoint:
            return
        
        checkpoint.updated_at = datetime.now().isoformat()
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.start_date}_{self.end_date}.json"
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)
            logger.debug(f"保存检查点: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"保存检查点失败: {e}")

    def _is_completed(
        self, 
        checkpoint: Optional[CheckpointData], 
        task_name: str, 
        year_month: str,
        stock_code: Optional[str] = None
    ) -> bool:
        """检查任务是否已完成"""
        if not checkpoint:
            return False
        
        # 检查月份级别
        task_months = checkpoint.completed_tasks.get(task_name, [])
        if year_month in task_months:
            if stock_code:
                # 需要进一步检查股票级别
                task_stocks = checkpoint.completed_stocks.get(task_name, {})
                month_stocks = task_stocks.get(year_month, [])
                return stock_code in month_stocks
            return True
        
        return False

    def _mark_completed(
        self,
        checkpoint: CheckpointData,
        task_name: str,
        year_month: str,
        stock_code: Optional[str] = None
    ):
        """标记任务完成"""
        if stock_code:
            # 标记股票完成
            if task_name not in checkpoint.completed_stocks:
                checkpoint.completed_stocks[task_name] = {}
            if year_month not in checkpoint.completed_stocks[task_name]:
                checkpoint.completed_stocks[task_name][year_month] = []
            if stock_code not in checkpoint.completed_stocks[task_name][year_month]:
                checkpoint.completed_stocks[task_name][year_month].append(stock_code)
        else:
            # 标记整月完成
            if task_name not in checkpoint.completed_tasks:
                checkpoint.completed_tasks[task_name] = []
            if year_month not in checkpoint.completed_tasks[task_name]:
                checkpoint.completed_tasks[task_name].append(year_month)
        
        checkpoint.last_completed_task = task_name
        checkpoint.last_completed_month = year_month
        if stock_code:
            checkpoint.last_completed_stock = stock_code

    def _execute_task_for_month(
        self,
        task: CollectionTask,
        year_month: str,
        month_start: str,
        month_end: str,
        checkpoint: CheckpointData,
        stock_codes: Optional[List[str]] = None,
    ) -> List[TaskResult]:
        """
        执行单个任务的月度采集
        
        Args:
            task: 采集任务配置
            year_month: 年月 (YYYY-MM)
            month_start: 月初日期
            month_end: 月末日期
            checkpoint: 检查点数据
            stock_codes: 股票代码列表（用于股票级别采集）
        
        Returns:
            任务结果列表
        """
        results = []
        
        try:
            collector = self._get_collector(task)
            collect_func = getattr(collector, task.collector_func)
        except Exception as e:
            logger.error(f"获取采集器失败 [{task.name}]: {e}")
            return [TaskResult(
                task_name=task.name,
                data_type=task.data_type.value,
                status=TaskStatus.FAILED,
                year_month=year_month,
                error_message=str(e),
            )]
        
        # 根据股票范围决定采集策略
        if task.stock_scope == StockScope.ALL_A and stock_codes:
            # 按股票分批采集
            batches = self._get_stock_batches(stock_codes, task.batch_size)
            total_batches = len(batches)
            
            for batch_idx, batch in enumerate(batches):
                for stock_code in batch:
                    # 检查是否已完成
                    if self._is_completed(checkpoint, task.name, year_month, stock_code):
                        logger.debug(f"跳过已完成: {task.name}/{year_month}/{stock_code}")
                        results.append(TaskResult(
                            task_name=task.name,
                            data_type=task.data_type.value,
                            status=TaskStatus.SKIPPED,
                            year_month=year_month,
                            stock_code=stock_code,
                        ))
                        continue
                    
                    # 检查文件是否已存在
                    output_path = self._get_output_path(task, year_month, stock_code)
                    if self.skip_existing and output_path.exists():
                        logger.debug(f"跳过已存在文件: {output_path}")
                        self._mark_completed(checkpoint, task.name, year_month, stock_code)
                        results.append(TaskResult(
                            task_name=task.name,
                            data_type=task.data_type.value,
                            status=TaskStatus.SKIPPED,
                            year_month=year_month,
                            stock_code=stock_code,
                            output_path=str(output_path),
                        ))
                        continue
                    
                    # 执行采集
                    result = self._collect_with_retry(
                        task=task,
                        collect_func=collect_func,
                        year_month=year_month,
                        month_start=month_start,
                        month_end=month_end,
                        stock_code=stock_code,
                    )
                    results.append(result)
                    
                    # 更新检查点
                    if result.status == TaskStatus.SUCCESS:
                        self._mark_completed(checkpoint, task.name, year_month, stock_code)
                        self._save_checkpoint(checkpoint)
                    
                    # 速率限制
                    time.sleep(task.rate_limit_delay)
                
                # 批次完成日志
                logger.info(
                    f"[{task.name}] {year_month} 批次 {batch_idx + 1}/{total_batches} 完成"
                )
        else:
            # 非股票级别采集（整月一次）
            if self._is_completed(checkpoint, task.name, year_month):
                logger.debug(f"跳过已完成: {task.name}/{year_month}")
                return [TaskResult(
                    task_name=task.name,
                    data_type=task.data_type.value,
                    status=TaskStatus.SKIPPED,
                    year_month=year_month,
                )]
            
            output_path = self._get_output_path(task, year_month)
            if self.skip_existing and output_path.exists():
                logger.debug(f"跳过已存在文件: {output_path}")
                self._mark_completed(checkpoint, task.name, year_month)
                return [TaskResult(
                    task_name=task.name,
                    data_type=task.data_type.value,
                    status=TaskStatus.SKIPPED,
                    year_month=year_month,
                    output_path=str(output_path),
                )]
            
            result = self._collect_with_retry(
                task=task,
                collect_func=collect_func,
                year_month=year_month,
                month_start=month_start,
                month_end=month_end,
                stock_codes=stock_codes,  # 传入股票列表（如果有的话）
            )
            results.append(result)
            
            if result.status == TaskStatus.SUCCESS:
                self._mark_completed(checkpoint, task.name, year_month)
                self._save_checkpoint(checkpoint)
        
        return results

    def _collect_with_retry(
        self,
        task: CollectionTask,
        collect_func: Callable,
        year_month: str,
        month_start: str,
        month_end: str,
        stock_code: Optional[str] = None,
        stock_codes: Optional[List[str]] = None,
    ) -> TaskResult:
        """带重试的采集执行"""
        result = TaskResult(
            task_name=task.name,
            data_type=task.data_type.value,
            status=TaskStatus.RUNNING,
            year_month=year_month,
            stock_code=stock_code,
            start_time=datetime.now(),
        )
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # 构建采集参数
                params = dict(task.params)
                
                # 根据不同任务类型构建参数
                params = self._build_collector_params(
                    task=task,
                    month_start=month_start,
                    month_end=month_end,
                    stock_code=stock_code,
                    stock_codes=stock_codes,
                    base_params=params,
                )
                
                # 执行采集
                df = collect_func(**params)
                
                if df is None:
                    df = pd.DataFrame()
                
                # 日期过滤（确保数据在范围内）
                df = self._filter_by_date(df, month_start, month_end)
                
                # 保存数据
                output_path = self._get_output_path(task, year_month, stock_code)
                
                if not df.empty:
                    if self._save_dataframe(df, output_path):
                        result.status = TaskStatus.SUCCESS
                        result.records_count = len(df)
                        result.output_path = str(output_path)
                        
                        log_msg = f"[{task.name}] {year_month}"
                        if stock_code:
                            log_msg += f" {stock_code}"
                        elif stock_codes:
                            log_msg += f" ({len(stock_codes)}只股票)"
                        log_msg += f": {len(df)} 条记录"
                        logger.info(log_msg)
                    else:
                        raise Exception(f"保存数据失败: {output_path}")
                else:
                    # 空数据也算成功
                    result.status = TaskStatus.SUCCESS
                    result.records_count = 0
                    logger.debug(f"[{task.name}] {year_month} {stock_code or ''}: 无数据")
                
                break  # 成功，退出重试循环
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"[{task.name}] {year_month} {stock_code or ''} "
                        f"失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}"
                    )
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    result.status = TaskStatus.FAILED
                    result.error_message = str(e)
                    result.error_traceback = traceback.format_exc()
                    logger.error(
                        f"[{task.name}] {year_month} {stock_code or ''} 最终失败: {e}"
                    )
        
        result.end_time = datetime.now()
        return result

    def _build_collector_params(
        self,
        task: CollectionTask,
        month_start: str,
        month_end: str,
        stock_code: Optional[str],
        stock_codes: Optional[List[str]],
        base_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        根据任务类型构建采集器参数
        
        不同采集器有不同的参数格式要求：
        - 公告采集器: start_date/end_date (YYYY-MM-DD), ts_codes (列表)
        - 事件采集器: start_date/end_date (YYYYMMDD), stock_codes (列表，可选)
        - 新闻采集器: start_date/end_date (YYYY-MM-DD 或 YYYYMMDD)
        - 政策采集器: start_date/end_date (YYYYMMDD)
        - 研报采集器: start_date/end_date, stock_codes (列表)
        """
        params = dict(base_params)
        
        # 日期格式
        date_yyyymmdd = month_start.replace('-', '')
        date_end_yyyymmdd = month_end.replace('-', '')
        
        # 根据任务名类型确定参数格式
        if task.name == 'cninfo_announcements':
            # 公告采集器使用 YYYY-MM-DD 格式和 ts_codes
            params['start_date'] = month_start
            params['end_date'] = month_end
            if stock_codes:
                # 传入整个股票列表
                params['ts_codes'] = stock_codes
            elif stock_code:
                params['ts_codes'] = [stock_code]
                
        elif task.name == 'cninfo_events':
            # 事件采集器使用 YYYYMMDD 格式
            params['start_date'] = date_yyyymmdd
            params['end_date'] = date_end_yyyymmdd
            # 事件采集器的stock_codes是可选的，不传则采集全市场
            if stock_codes:
                # 转换为纯代码格式
                params['stock_codes'] = [c.split('.')[0] for c in stock_codes]
            elif stock_code:
                params['stock_codes'] = [stock_code.split('.')[0]]
                
        elif task.name in ['cctv_news', 'exchange_news']:
            # 新闻采集器使用 YYYY-MM-DD 格式
            params['start_date'] = month_start
            params['end_date'] = month_end
            
        elif task.name in ['gov_council_policy', 'ndrc_policy']:
            # 政策采集器使用 YYYYMMDD 格式
            params['start_date'] = date_yyyymmdd
            params['end_date'] = date_end_yyyymmdd
            
        elif task.name == 'eastmoney_reports':
            # 研报采集器使用 YYYY-MM-DD 格式和 stock_codes
            params['start_date'] = month_start
            params['end_date'] = month_end
            if stock_codes:
                # 转换为纯代码格式
                params['stock_codes'] = [c.split('.')[0] for c in stock_codes]
            elif stock_code:
                params['stock_codes'] = [stock_code.split('.')[0]]
        else:
            # 默认使用 YYYYMMDD 格式
            params['start_date'] = date_yyyymmdd
            params['end_date'] = date_end_yyyymmdd
            if stock_codes:
                params['stock_codes'] = stock_codes
            elif stock_code:
                params['stock_codes'] = [stock_code]
        
        return params

    def _filter_by_date(
        self, 
        df: pd.DataFrame, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """过滤日期范围内的数据"""
        if df.empty:
            return df
        
        # 尝试常见的日期列名
        date_columns = ['date', 'pub_date', 'publish_date', 'ann_date', 'trade_date']
        
        for col in date_columns:
            if col in df.columns:
                try:
                    # 标准化日期格式
                    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                    df = df[(df[col] >= start_date) & (df[col] <= end_date)].copy()
                    return df
                except Exception:
                    pass
        
        return df

    def run_all(
        self,
        data_types: Optional[List[str]] = None,
    ) -> CollectionProgress:
        """
        执行全量采集
        
        Args:
            data_types: 要采集的数据类型列表，None表示全部
                       可选: ['announcements', 'events', 'news', 'policy', 'reports']
        
        Returns:
            采集进度对象
        """
        # 确定要采集的数据类型
        if data_types:
            types_to_collect = [DataType(t) for t in data_types if t in [dt.value for dt in DataType]]
        else:
            types_to_collect = list(DataType)
        
        type_names = [TYPE_NAMES.get(t, t.value) for t in types_to_collect]
        
        logger.info("=" * 60)
        logger.info("非结构化数据全量采集开始")
        logger.info("=" * 60)
        logger.info(f"日期范围: {self.start_date} ~ {self.end_date}")
        logger.info(f"数据类型: {', '.join(type_names)}")
        logger.info("=" * 60)
        
        # 初始化进度
        self.progress = CollectionProgress(start_time=datetime.now())
        
        # 获取月份范围
        months = self._get_month_range()
        self.progress.total_months = len(months)
        logger.info(f"共 {len(months)} 个月需要采集")
        
        # 加载股票列表
        stock_codes = self._load_stock_list()
        self.progress.total_stocks = len(stock_codes)
        
        # 加载检查点
        checkpoint = self._load_checkpoint([t.value for t in types_to_collect])
        if not checkpoint:
            checkpoint = CheckpointData(
                start_date=self.start_date,
                end_date=self.end_date,
                data_types=[t.value for t in types_to_collect],
            )
        self._checkpoint = checkpoint
        
        # 按数据类型执行采集
        for data_type in types_to_collect:
            tasks = get_tasks_by_type(data_type)
            if not tasks:
                logger.warning(f"数据类型 {data_type.value} 没有配置任务")
                continue
            
            logger.info(f"\n{'=' * 40}")
            logger.info(f"开始采集: {TYPE_NAMES.get(data_type, data_type.value)}")
            logger.info(f"{'=' * 40}")
            
            for task in tasks:
                if not task.enabled:
                    logger.info(f"跳过禁用的任务: {task.name}")
                    continue
                
                self.progress.current_task = task.name
                logger.info(f"\n[任务] {task.name}: {task.description}")
                
                # 按月执行
                for year_month, month_start, month_end in months:
                    self.progress.current_month = year_month
                    
                    # 确定是否需要股票列表
                    # ALL_A范围或needs_stock_list标记的任务需要股票列表
                    task_stock_codes = None
                    if task.stock_scope == StockScope.ALL_A or task.needs_stock_list:
                        task_stock_codes = stock_codes
                    
                    results = self._execute_task_for_month(
                        task=task,
                        year_month=year_month,
                        month_start=month_start,
                        month_end=month_end,
                        checkpoint=checkpoint,
                        stock_codes=task_stock_codes,
                    )
                    
                    for result in results:
                        self.progress.add_result(result)
                    
                    self.progress.completed_months += 1
        
        # 输出汇总
        self._print_summary()
        
        return self.progress

    def run_by_types(self, data_types: List[str]) -> CollectionProgress:
        """按指定数据类型执行采集"""
        return self.run_all(data_types=data_types)

    def _print_summary(self):
        """打印采集汇总"""
        elapsed = datetime.now() - self.progress.start_time if self.progress.start_time else timedelta(0)
        
        logger.info("\n" + "=" * 60)
        logger.info("采集完成汇总")
        logger.info("=" * 60)
        logger.info(f"日期范围: {self.start_date} ~ {self.end_date}")
        logger.info(f"总耗时: {elapsed}")
        logger.info(f"进度: {self.progress.progress_percent:.1f}%")
        logger.info(
            f"结果: 成功={self.progress.success_count}, "
            f"失败={self.progress.failed_count}, "
            f"跳过={self.progress.skipped_count}"
        )
        logger.info(f"总记录数: {self.progress.total_records}")
        
        # 按数据类型统计
        type_stats = {}
        for result in self.progress.results:
            if result.data_type not in type_stats:
                type_stats[result.data_type] = {
                    'success': 0, 'failed': 0, 'skipped': 0, 'records': 0
                }
            if result.status == TaskStatus.SUCCESS:
                type_stats[result.data_type]['success'] += 1
            elif result.status == TaskStatus.FAILED:
                type_stats[result.data_type]['failed'] += 1
            elif result.status == TaskStatus.SKIPPED:
                type_stats[result.data_type]['skipped'] += 1
            type_stats[result.data_type]['records'] += result.records_count
        
        logger.info("\n各数据类型统计:")
        for dtype, stats in type_stats.items():
            logger.info(
                f"  {dtype}: 成功={stats['success']}, "
                f"失败={stats['failed']}, 跳过={stats['skipped']}, "
                f"记录数={stats['records']}"
            )
        
        # 输出失败任务
        failed_results = [r for r in self.progress.results if r.status == TaskStatus.FAILED]
        if failed_results:
            logger.warning(f"\n失败任务 ({len(failed_results)} 个):")
            for r in failed_results[:20]:
                logger.warning(f"  - {r.task_name}/{r.year_month}/{r.stock_code or ''}: {r.error_message}")
            if len(failed_results) > 20:
                logger.warning(f"  ... 还有 {len(failed_results) - 20} 个失败任务")
        
        logger.info("=" * 60)

    def get_collection_summary(self) -> Dict[str, Any]:
        """获取采集汇总"""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'output_dir': str(self.output_dir),
            'progress': {
                'total_months': self.progress.total_months,
                'completed_months': self.progress.completed_months,
                'progress_percent': f"{self.progress.progress_percent:.1f}%",
                'success_count': self.progress.success_count,
                'failed_count': self.progress.failed_count,
                'skipped_count': self.progress.skipped_count,
                'total_records': self.progress.total_records,
            },
        }
