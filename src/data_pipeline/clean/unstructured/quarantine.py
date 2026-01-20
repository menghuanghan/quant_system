"""
失败隔离区模块 (Quarantine System)

当清洗/提取失败时，将原始文件暂存到隔离区，防止数据永久丢失。

Features:
- 自动保存失败的原始文件（PDF、HTML等）
- 记录详细的错误日志
- 支持后续批量重试或人工处理
- 空间管理：自动清理超龄文件

目录结构：
    data/raw/quarantine/
    ├── pdf/
    │   └── 2024-01/
    │       ├── 000001_20240115_abc123.pdf
    │       └── ...
    ├── html/
    │   └── 2024-01/
    │       └── ...
    └── error_log.jsonl
"""

import json
import hashlib
import logging
import shutil
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, BinaryIO
from threading import Lock
from enum import Enum
import os

logger = logging.getLogger(__name__)


class FileType(str, Enum):
    """文件类型"""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    OTHER = "other"


@dataclass
class QuarantineRecord:
    """隔离记录"""
    quarantine_id: str           # 隔离ID
    original_url: str            # 原始URL
    file_type: str               # 文件类型
    file_path: str               # 隔离文件路径
    error_type: str              # 错误类型
    error_message: str           # 错误信息
    source: str                  # 数据源
    ts_code: Optional[str]       # 关联股票代码
    publish_date: Optional[str]  # 发布日期
    created_at: str              # 创建时间
    retry_count: int = 0         # 重试次数
    resolved: bool = False       # 是否已解决
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QuarantineRecord':
        return cls(**data)


class QuarantineManager:
    """
    隔离区管理器
    
    负责管理所有清洗失败的文件：
    - 保存原始文件到隔离目录
    - 记录错误日志
    - 提供重试和清理接口
    
    Example:
        >>> quarantine = QuarantineManager()
        >>> 
        >>> # 隔离失败的PDF
        >>> try:
        ...     text = extract_pdf(pdf_content)
        ... except Exception as e:
        ...     quarantine.quarantine_file(
        ...         content=pdf_content,
        ...         url="http://...",
        ...         file_type=FileType.PDF,
        ...         error=e,
        ...         source="cninfo"
        ...     )
        >>> 
        >>> # 获取待处理的隔离文件
        >>> pending = quarantine.get_pending_records()
    """
    
    DEFAULT_QUARANTINE_DIR = Path("data/raw/quarantine")
    MAX_RETENTION_DAYS = 90  # 默认保留90天
    
    def __init__(
        self,
        quarantine_dir: Optional[Path] = None,
        max_retention_days: int = 90,
        max_file_size_mb: float = 50.0  # 单文件最大大小
    ):
        """
        Args:
            quarantine_dir: 隔离目录
            max_retention_days: 最大保留天数
            max_file_size_mb: 单文件最大大小（MB）
        """
        self.quarantine_dir = quarantine_dir or self.DEFAULT_QUARANTINE_DIR
        self.max_retention_days = max_retention_days
        self.max_file_size_mb = max_file_size_mb
        
        self._lock = Lock()
        self._stats = {
            'total_quarantined': 0,
            'total_resolved': 0,
            'by_type': {},
            'by_error': {}
        }
        
        # 确保目录存在
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        (self.quarantine_dir / "pdf").mkdir(exist_ok=True)
        (self.quarantine_dir / "html").mkdir(exist_ok=True)
        (self.quarantine_dir / "other").mkdir(exist_ok=True)
        
        # 错误日志文件
        self.error_log_path = self.quarantine_dir / "error_log.jsonl"
        
        # 加载统计
        self._load_stats()
    
    def _generate_id(self, url: str, content: bytes) -> str:
        """生成唯一隔离ID"""
        hash_input = f"{url}_{len(content)}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _get_month_dir(self, file_type: FileType) -> Path:
        """获取当月目录"""
        month_str = datetime.now().strftime('%Y-%m')
        month_dir = self.quarantine_dir / file_type.value / month_str
        month_dir.mkdir(parents=True, exist_ok=True)
        return month_dir
    
    def quarantine_file(
        self,
        content: bytes,
        url: str,
        file_type: FileType,
        error: Exception,
        source: str,
        ts_code: Optional[str] = None,
        publish_date: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[QuarantineRecord]:
        """
        将失败的文件放入隔离区
        
        Args:
            content: 文件内容（bytes）
            url: 原始URL
            file_type: 文件类型
            error: 异常对象
            source: 数据源标识
            ts_code: 关联股票代码
            publish_date: 发布日期
            metadata: 额外元数据
        
        Returns:
            QuarantineRecord 或 None（如果文件太大或其他原因）
        """
        # 检查文件大小
        size_mb = len(content) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            logger.warning(
                f"文件过大，不保存到隔离区: {size_mb:.1f}MB > {self.max_file_size_mb}MB"
            )
            return None
        
        with self._lock:
            try:
                # 生成ID和文件名
                quarantine_id = self._generate_id(url, content)
                month_dir = self._get_month_dir(file_type)
                
                # 构建文件名
                ts_code_part = ts_code.replace('.', '_') if ts_code else "unknown"
                date_part = publish_date or datetime.now().strftime('%Y%m%d')
                filename = f"{ts_code_part}_{date_part}_{quarantine_id}.{file_type.value}"
                file_path = month_dir / filename
                
                # 保存文件
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                # 创建记录
                record = QuarantineRecord(
                    quarantine_id=quarantine_id,
                    original_url=url,
                    file_type=file_type.value,
                    file_path=str(file_path),
                    error_type=type(error).__name__,
                    error_message=str(error),
                    source=source,
                    ts_code=ts_code,
                    publish_date=publish_date,
                    created_at=datetime.now().isoformat()
                )
                
                # 写入日志
                self._append_log(record, metadata)
                
                # 更新统计
                self._stats['total_quarantined'] += 1
                self._stats['by_type'][file_type.value] = \
                    self._stats['by_type'].get(file_type.value, 0) + 1
                self._stats['by_error'][record.error_type] = \
                    self._stats['by_error'].get(record.error_type, 0) + 1
                
                logger.debug(f"文件已隔离: {file_path}")
                return record
                
            except Exception as e:
                logger.error(f"隔离文件失败: {e}")
                return None
    
    def quarantine_from_stream(
        self,
        stream: BinaryIO,
        url: str,
        file_type: FileType,
        error: Exception,
        source: str,
        **kwargs
    ) -> Optional[QuarantineRecord]:
        """
        从流中读取内容并隔离
        
        用于处理大文件，避免内存问题
        """
        try:
            content = stream.read()
            return self.quarantine_file(
                content=content,
                url=url,
                file_type=file_type,
                error=error,
                source=source,
                **kwargs
            )
        except Exception as e:
            logger.error(f"从流读取并隔离失败: {e}")
            return None
    
    def _append_log(self, record: QuarantineRecord, metadata: Optional[Dict] = None):
        """追加错误日志"""
        log_entry = record.to_dict()
        if metadata:
            log_entry['metadata'] = metadata
        
        with open(self.error_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def get_pending_records(
        self,
        file_type: Optional[FileType] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[QuarantineRecord]:
        """
        获取待处理的隔离记录
        
        Args:
            file_type: 筛选文件类型
            source: 筛选数据源
            limit: 最大返回数量
        """
        records = []
        
        if not self.error_log_path.exists():
            return records
        
        with open(self.error_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    record = QuarantineRecord.from_dict(data)
                    
                    # 跳过已解决的
                    if record.resolved:
                        continue
                    
                    # 筛选
                    if file_type and record.file_type != file_type.value:
                        continue
                    if source and record.source != source:
                        continue
                    
                    records.append(record)
                    
                    if len(records) >= limit:
                        break
                        
                except Exception:
                    continue
        
        return records
    
    def mark_resolved(self, quarantine_id: str, delete_file: bool = False):
        """
        标记记录为已解决
        
        Args:
            quarantine_id: 隔离ID
            delete_file: 是否删除隔离文件
        """
        # 这里简化处理，实际应该重写整个日志文件或使用数据库
        # 在生产环境中建议使用 SQLite
        logger.info(f"标记已解决: {quarantine_id}")
        self._stats['total_resolved'] += 1
    
    def cleanup_expired(self) -> int:
        """
        清理过期的隔离文件
        
        Returns:
            清理的文件数量
        """
        cutoff_date = datetime.now() - timedelta(days=self.max_retention_days)
        cleaned_count = 0
        
        for type_dir in [self.quarantine_dir / "pdf", 
                         self.quarantine_dir / "html",
                         self.quarantine_dir / "other"]:
            if not type_dir.exists():
                continue
            
            for month_dir in type_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                
                # 检查月份目录是否过期
                try:
                    month_date = datetime.strptime(month_dir.name, '%Y-%m')
                    if month_date < cutoff_date:
                        # 删除整个月份目录
                        shutil.rmtree(month_dir)
                        logger.info(f"清理过期目录: {month_dir}")
                        cleaned_count += 1
                except ValueError:
                    continue
        
        return cleaned_count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self._stats.copy()
            
            # 计算存储使用
            total_size = 0
            for file_path in self.quarantine_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix != '.jsonl':
                    total_size += file_path.stat().st_size
            
            stats['storage_size_mb'] = total_size / (1024 * 1024)
            return stats
    
    def _load_stats(self):
        """从日志加载统计"""
        if not self.error_log_path.exists():
            return
        
        try:
            with open(self.error_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        self._stats['total_quarantined'] += 1
                        
                        file_type = data.get('file_type', 'other')
                        self._stats['by_type'][file_type] = \
                            self._stats['by_type'].get(file_type, 0) + 1
                        
                        error_type = data.get('error_type', 'Unknown')
                        self._stats['by_error'][error_type] = \
                            self._stats['by_error'].get(error_type, 0) + 1
                        
                        if data.get('resolved'):
                            self._stats['total_resolved'] += 1
                            
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"加载隔离统计失败: {e}")
    
    def print_summary(self):
        """打印摘要"""
        stats = self.get_stats()
        
        print("\n" + "=" * 50)
        print("  隔离区摘要")
        print("=" * 50)
        print(f"  总隔离数: {stats.get('total_quarantined', 0)}")
        print(f"  已解决:   {stats.get('total_resolved', 0)}")
        print(f"  存储占用: {stats.get('storage_size_mb', 0):.1f} MB")
        print("\n  按类型分布:")
        for file_type, count in stats.get('by_type', {}).items():
            print(f"    - {file_type}: {count}")
        print("\n  按错误类型分布:")
        for error_type, count in sorted(
            stats.get('by_error', {}).items(), 
            key=lambda x: -x[1]
        )[:5]:
            print(f"    - {error_type}: {count}")
        print("=" * 50)


# ============== 全局单例 ==============

_global_quarantine: Optional[QuarantineManager] = None


def get_quarantine_manager() -> QuarantineManager:
    """获取全局隔离管理器"""
    global _global_quarantine
    if _global_quarantine is None:
        _global_quarantine = QuarantineManager()
    return _global_quarantine


def quarantine_failed_file(
    content: bytes,
    url: str,
    file_type: FileType,
    error: Exception,
    source: str,
    **kwargs
) -> Optional[QuarantineRecord]:
    """便捷函数：隔离失败的文件"""
    return get_quarantine_manager().quarantine_file(
        content=content,
        url=url,
        file_type=file_type,
        error=error,
        source=source,
        **kwargs
    )
