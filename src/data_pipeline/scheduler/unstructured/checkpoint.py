"""
状态管理模块 (Checkpoint Manager)

提供断点续传能力，支持：
- 任务状态持久化（JSON 文件）
- 自动恢复：跳过 COMPLETED，优先重试 FAILED
- 原子性更新：防止写入中断导致状态丢失

状态文件位置：data/state/backfill_status.json
"""

import json
import logging
import os
import shutil
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from threading import Lock
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    """任务状态枚举"""
    PENDING = "PENDING"         # 待处理
    RUNNING = "RUNNING"         # 运行中
    COMPLETED = "COMPLETED"     # 已完成
    FAILED = "FAILED"           # 失败
    SKIPPED = "SKIPPED"         # 跳过（如无数据）
    CIRCUIT_BREAK = "CIRCUIT_BREAK"  # 熔断（被封禁等）


@dataclass
class TaskRecord:
    """
    单个任务的执行记录
    
    记录某个数据源在某个月份的采集状态
    """
    state: TaskState = TaskState.PENDING
    start_time: Optional[str] = None      # 开始时间
    end_time: Optional[str] = None        # 结束时间
    retry_count: int = 0                  # 重试次数
    error_message: Optional[str] = None   # 错误信息
    record_count: int = 0                 # 采集记录数
    file_size_mb: float = 0.0             # 生成文件大小
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['state'] = self.state.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskRecord':
        if 'state' in data:
            data['state'] = TaskState(data['state'])
        return cls(**data)


@dataclass
class BackfillStatus:
    """
    全量回填状态
    
    结构：
    {
        "version": "1.0",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-15T10:30:00",
        "tasks": {
            "news_sina": {
                "2025-12": {"state": "COMPLETED", ...},
                "2025-11": {"state": "FAILED", ...}
            },
            "announcement_cninfo": { ... }
        },
        "global_stats": {
            "total_tasks": 360,
            "completed": 120,
            "failed": 5,
            "pending": 235
        }
    }
    """
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tasks: Dict[str, Dict[str, Dict]] = field(default_factory=dict)
    global_stats: Dict[str, int] = field(default_factory=lambda: {
        'total_tasks': 0,
        'completed': 0,
        'failed': 0,
        'pending': 0,
        'running': 0,
        'skipped': 0
    })
    
    def get_task(self, source: str, month: str) -> TaskRecord:
        """获取任务记录"""
        if source not in self.tasks:
            self.tasks[source] = {}
        if month not in self.tasks[source]:
            self.tasks[source][month] = TaskRecord().to_dict()
        return TaskRecord.from_dict(self.tasks[source][month])
    
    def set_task(self, source: str, month: str, record: TaskRecord):
        """设置任务记录"""
        if source not in self.tasks:
            self.tasks[source] = {}
        self.tasks[source][month] = record.to_dict()
        self.updated_at = datetime.now().isoformat()
    
    def update_stats(self):
        """更新全局统计"""
        stats = {
            'total_tasks': 0,
            'completed': 0,
            'failed': 0,
            'pending': 0,
            'running': 0,
            'skipped': 0,
            'circuit_break': 0
        }
        
        for source, months in self.tasks.items():
            for month, record_dict in months.items():
                stats['total_tasks'] += 1
                state = TaskState(record_dict.get('state', 'PENDING'))
                
                if state == TaskState.COMPLETED:
                    stats['completed'] += 1
                elif state == TaskState.FAILED:
                    stats['failed'] += 1
                elif state == TaskState.PENDING:
                    stats['pending'] += 1
                elif state == TaskState.RUNNING:
                    stats['running'] += 1
                elif state == TaskState.SKIPPED:
                    stats['skipped'] += 1
                elif state == TaskState.CIRCUIT_BREAK:
                    stats['circuit_break'] += 1
        
        self.global_stats = stats
    
    def to_dict(self) -> Dict:
        self.update_stats()
        return {
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'tasks': self.tasks,
            'global_stats': self.global_stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BackfillStatus':
        return cls(
            version=data.get('version', '1.0'),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            tasks=data.get('tasks', {}),
            global_stats=data.get('global_stats', {})
        )


class CheckpointManager:
    """
    检查点管理器
    
    负责状态的持久化和恢复，提供断点续传能力。
    
    Features:
    - 原子性写入（先写临时文件再重命名）
    - 线程安全（锁保护）
    - 自动备份
    
    Example:
        >>> manager = CheckpointManager()
        >>> 
        >>> # 更新任务状态
        >>> with manager.task_context('news_sina', '2024-12') as task:
        ...     # 执行采集
        ...     task.record_count = 1000
        ...     task.state = TaskState.COMPLETED
        >>> 
        >>> # 获取待处理任务
        >>> pending = manager.get_pending_tasks('news_sina')
    """
    
    DEFAULT_STATE_DIR = Path("data/state")
    DEFAULT_STATE_FILE = "backfill_status.json"
    
    def __init__(
        self,
        state_dir: Optional[Path] = None,
        state_file: Optional[str] = None,
        auto_backup: bool = True,
        backup_interval: int = 10  # 每 N 次更新备份一次
    ):
        """
        初始化检查点管理器
        
        Args:
            state_dir: 状态文件目录
            state_file: 状态文件名
            auto_backup: 是否自动备份
            backup_interval: 备份间隔（更新次数）
        """
        self.state_dir = state_dir or self.DEFAULT_STATE_DIR
        self.state_file = state_file or self.DEFAULT_STATE_FILE
        self.state_path = self.state_dir / self.state_file
        self.auto_backup = auto_backup
        self.backup_interval = backup_interval
        
        self._lock = Lock()
        self._update_count = 0
        self._status: Optional[BackfillStatus] = None
        
        # 确保目录存在
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载或创建状态
        self._load_or_create()
    
    def _load_or_create(self):
        """加载或创建状态文件"""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._status = BackfillStatus.from_dict(data)
                logger.info(f"已加载状态文件: {self.state_path}")
                logger.info(f"  总任务: {self._status.global_stats.get('total_tasks', 0)}, "
                           f"已完成: {self._status.global_stats.get('completed', 0)}, "
                           f"失败: {self._status.global_stats.get('failed', 0)}")
            except Exception as e:
                logger.warning(f"加载状态文件失败: {e}, 创建新文件")
                self._status = BackfillStatus()
                self._save()
        else:
            logger.info(f"创建新状态文件: {self.state_path}")
            self._status = BackfillStatus()
            self._save()
    
    def _save(self):
        """保存状态文件（原子性写入）"""
        temp_path = self.state_path.with_suffix('.tmp')
        
        try:
            # 先写入临时文件
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self._status.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 原子性重命名
            shutil.move(str(temp_path), str(self.state_path))
            
            # 自动备份
            self._update_count += 1
            if self.auto_backup and self._update_count % self.backup_interval == 0:
                self._create_backup()
                
        except Exception as e:
            logger.error(f"保存状态文件失败: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def _create_backup(self):
        """创建备份"""
        backup_dir = self.state_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f"backfill_status_{timestamp}.json"
        
        shutil.copy2(self.state_path, backup_path)
        logger.debug(f"已创建备份: {backup_path}")
        
        # 保留最近 10 个备份
        backups = sorted(backup_dir.glob("backfill_status_*.json"))
        if len(backups) > 10:
            for old_backup in backups[:-10]:
                old_backup.unlink()
    
    @property
    def status(self) -> BackfillStatus:
        """获取当前状态"""
        return self._status
    
    def get_task(self, source: str, month: str) -> TaskRecord:
        """获取任务记录"""
        with self._lock:
            return self._status.get_task(source, month)
    
    def update_task(
        self,
        source: str,
        month: str,
        state: TaskState,
        **kwargs
    ):
        """
        更新任务状态
        
        Args:
            source: 数据源名称
            month: 月份 (YYYY-MM)
            state: 新状态
            **kwargs: 其他字段（record_count, error_message 等）
        """
        with self._lock:
            record = self._status.get_task(source, month)
            record.state = state
            
            if state == TaskState.RUNNING:
                record.start_time = datetime.now().isoformat()
            elif state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.SKIPPED):
                record.end_time = datetime.now().isoformat()
            
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            
            self._status.set_task(source, month, record)
            self._save()
    
    @contextmanager
    def task_context(self, source: str, month: str):
        """
        任务上下文管理器
        
        自动处理状态转换：
        - 进入时标记为 RUNNING
        - 正常退出时标记为 COMPLETED（如果 task.state 未被修改）
        - 异常退出时标记为 FAILED
        
        Example:
            >>> with manager.task_context('news_sina', '2024-12') as task:
            ...     # 执行采集
            ...     task.record_count = 1000
        """
        record = self.get_task(source, month)
        record.state = TaskState.RUNNING
        record.start_time = datetime.now().isoformat()
        record.retry_count += 1
        
        try:
            yield record
            
            # 如果子任务没有显式设置状态，默认为 COMPLETED
            if record.state == TaskState.RUNNING:
                record.state = TaskState.COMPLETED
            record.end_time = datetime.now().isoformat()
            record.error_message = None
            
        except Exception as e:
            record.state = TaskState.FAILED
            record.end_time = datetime.now().isoformat()
            record.error_message = str(e)
            raise
            
        finally:
            with self._lock:
                self._status.set_task(source, month, record)
                self._save()
    
    def get_pending_tasks(
        self,
        source: Optional[str] = None,
        include_failed: bool = True,
        max_retries: int = 3
    ) -> List[tuple]:
        """
        获取待处理任务列表
        
        优先级：FAILED (重试) > PENDING
        
        Args:
            source: 数据源名称（None 表示所有）
            include_failed: 是否包含失败任务
            max_retries: 最大重试次数
        
        Returns:
            [(source, month, state), ...] 按优先级排序
        """
        pending = []
        failed = []
        
        with self._lock:
            sources = [source] if source else list(self._status.tasks.keys())
            
            for src in sources:
                if src not in self._status.tasks:
                    continue
                    
                for month, record_dict in self._status.tasks[src].items():
                    record = TaskRecord.from_dict(record_dict)
                    
                    if record.state == TaskState.PENDING:
                        pending.append((src, month, record.state))
                        
                    elif record.state == TaskState.FAILED and include_failed:
                        if record.retry_count < max_retries:
                            failed.append((src, month, record.state))
        
        # 优先重试失败任务
        return failed + pending
    
    def get_completed_tasks(self, source: Optional[str] = None) -> List[tuple]:
        """获取已完成任务列表"""
        completed = []
        
        with self._lock:
            sources = [source] if source else list(self._status.tasks.keys())
            
            for src in sources:
                if src not in self._status.tasks:
                    continue
                    
                for month, record_dict in self._status.tasks[src].items():
                    if TaskState(record_dict.get('state')) == TaskState.COMPLETED:
                        completed.append((src, month))
        
        return completed
    
    def initialize_tasks(
        self,
        sources: List[str],
        months: List[str],
        skip_completed: bool = True
    ):
        """
        批量初始化任务
        
        用于首次运行时创建所有任务槽位
        
        Args:
            sources: 数据源列表
            months: 月份列表 (YYYY-MM)
            skip_completed: 是否跳过已完成的任务
        """
        with self._lock:
            for source in sources:
                if source not in self._status.tasks:
                    self._status.tasks[source] = {}
                
                for month in months:
                    if month in self._status.tasks[source]:
                        existing = TaskRecord.from_dict(self._status.tasks[source][month])
                        if skip_completed and existing.state == TaskState.COMPLETED:
                            continue
                        if existing.state == TaskState.RUNNING:
                            # 上次运行中断，重置为待处理
                            existing.state = TaskState.PENDING
                            self._status.tasks[source][month] = existing.to_dict()
                    else:
                        self._status.tasks[source][month] = TaskRecord().to_dict()
            
            self._save()
            logger.info(f"已初始化任务: {len(sources)} 数据源 x {len(months)} 月份")
    
    def reset_failed_tasks(self, source: Optional[str] = None):
        """重置所有失败任务为待处理"""
        with self._lock:
            sources = [source] if source else list(self._status.tasks.keys())
            reset_count = 0
            
            for src in sources:
                if src not in self._status.tasks:
                    continue
                    
                for month, record_dict in self._status.tasks[src].items():
                    if TaskState(record_dict.get('state')) == TaskState.FAILED:
                        record = TaskRecord.from_dict(record_dict)
                        record.state = TaskState.PENDING
                        record.error_message = None
                        self._status.tasks[src][month] = record.to_dict()
                        reset_count += 1
            
            if reset_count > 0:
                self._save()
                logger.info(f"已重置 {reset_count} 个失败任务")
    
    def get_progress(self) -> Dict[str, Any]:
        """获取整体进度"""
        with self._lock:
            self._status.update_stats()
            stats = self._status.global_stats.copy()
            
            total = stats.get('total_tasks', 0)
            completed = stats.get('completed', 0)
            
            stats['progress_pct'] = (completed / total * 100) if total > 0 else 0
            stats['updated_at'] = self._status.updated_at
            
            return stats
    
    def print_summary(self):
        """打印进度摘要"""
        progress = self.get_progress()
        
        print("\n" + "=" * 50)
        print("  回填进度摘要")
        print("=" * 50)
        print(f"  总任务数: {progress.get('total_tasks', 0)}")
        print(f"  已完成:   {progress.get('completed', 0)} ({progress.get('progress_pct', 0):.1f}%)")
        print(f"  失败:     {progress.get('failed', 0)}")
        print(f"  待处理:   {progress.get('pending', 0)}")
        print(f"  运行中:   {progress.get('running', 0)}")
        print(f"  最后更新: {progress.get('updated_at', 'N/A')}")
        print("=" * 50)


# ============== 全局单例 ==============

_global_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """获取全局检查点管理器"""
    global _global_checkpoint_manager
    if _global_checkpoint_manager is None:
        _global_checkpoint_manager = CheckpointManager()
    return _global_checkpoint_manager


def set_checkpoint_manager(manager: CheckpointManager):
    """设置全局检查点管理器"""
    global _global_checkpoint_manager
    _global_checkpoint_manager = manager
