"""
统一数据存储管理器 (DataSink/StorageManager)

提供统一的数据落地接口，支持多种存储格式：
- Parquet (默认，高效压缩)
- JSONL (流式备份)
- CSV (兼容性)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import json
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StorageFormat(Enum):
    """存储格式枚举"""
    PARQUET = "parquet"
    JSONL = "jsonl"
    CSV = "csv"
    

class CompressionType(Enum):
    """压缩类型"""
    SNAPPY = "snappy"  # Parquet默认
    GZIP = "gzip"
    BROTLI = "brotli"
    NONE = None


class DataSink:
    """
    数据落地管理器
    
    Features:
    - 统一的存储接口
    - 自动格式转换
    - 智能压缩
    - 增量追加
    - 分区存储（按日期/类型）
    
    Example:
        >>> sink = DataSink(base_path="data/raw/unstructured")
        >>> sink.save(
        ...     data=df,
        ...     domain="events",
        ...     format=StorageFormat.PARQUET,
        ...     partition_by="ann_date"
        ... )
    """
    
    def __init__(
        self,
        base_path: Union[str, Path],
        default_format: StorageFormat = StorageFormat.PARQUET,
        compression: CompressionType = CompressionType.SNAPPY,
        enable_backup: bool = True
    ):
        """
        Args:
            base_path: 数据根目录
            default_format: 默认存储格式
            compression: 压缩类型
            enable_backup: 是否启用JSONL备份
        """
        self.base_path = Path(base_path)
        self.default_format = default_format
        self.compression = compression
        self.enable_backup = enable_backup
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        domain: str,
        sub_domain: Optional[str] = None,
        format: Optional[StorageFormat] = None,
        partition_by: Optional[str] = None,
        filename: Optional[str] = None,
        mode: str = "overwrite"  # "overwrite" or "append"
    ) -> Path:
        """
        保存数据
        
        Args:
            data: 数据（DataFrame或字典列表）
            domain: 数据域（如 events, news, announcements）
            sub_domain: 子域（如 penalty, merger）
            format: 存储格式
            partition_by: 分区字段（如 ann_date）
            filename: 文件名（不含扩展名）
            mode: 写入模式（overwrite/append）
            
        Returns:
            保存的文件路径
        """
        # 数据类型转换
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        if df.empty:
            logger.warning("Empty data, skipping save")
            return None
            
        # 确定存储格式
        fmt = format or self.default_format
        
        # 构建存储路径
        storage_path = self.base_path / domain
        if sub_domain:
            storage_path = storage_path / sub_domain
            
        # 处理分区
        if partition_by and partition_by in df.columns:
            # 按日期分区（年/月）
            if 'date' in partition_by.lower():
                df[partition_by] = pd.to_datetime(df[partition_by])
                df['_year'] = df[partition_by].dt.year
                df['_month'] = df[partition_by].dt.month
                
                # 按年月分区保存
                for (year, month), group in df.groupby(['_year', '_month']):
                    partition_path = storage_path / f"year={year}" / f"month={month:02d}"
                    partition_path.mkdir(parents=True, exist_ok=True)
                    
                    partition_filename = filename or f"{domain}_{year}{month:02d}"
                    file_path = self._save_format(
                        group.drop(['_year', '_month'], axis=1),
                        partition_path,
                        partition_filename,
                        fmt,
                        mode
                    )
                    
                return storage_path  # 返回根分区路径
            else:
                # 按其他字段分区
                for partition_val, group in df.groupby(partition_by):
                    partition_path = storage_path / f"{partition_by}={partition_val}"
                    partition_path.mkdir(parents=True, exist_ok=True)
                    
                    partition_filename = filename or f"{domain}_{partition_val}"
                    self._save_format(group, partition_path, partition_filename, fmt, mode)
                    
                return storage_path
        else:
            # 不分区，直接保存
            storage_path.mkdir(parents=True, exist_ok=True)
            default_filename = filename or f"{domain}_{datetime.now().strftime('%Y%m%d')}"
            file_path = self._save_format(df, storage_path, default_filename, fmt, mode)
            
            # JSONL备份
            if self.enable_backup and fmt != StorageFormat.JSONL:
                backup_path = storage_path / "backup"
                backup_path.mkdir(exist_ok=True)
                self._save_jsonl(df, backup_path / f"{default_filename}.jsonl", mode)
                
            return file_path
    
    def _save_format(
        self,
        df: pd.DataFrame,
        path: Path,
        filename: str,
        format: StorageFormat,
        mode: str
    ) -> Path:
        """根据格式保存数据"""
        if format == StorageFormat.PARQUET:
            return self._save_parquet(df, path / f"{filename}.parquet", mode)
        elif format == StorageFormat.JSONL:
            return self._save_jsonl(df, path / f"{filename}.jsonl", mode)
        elif format == StorageFormat.CSV:
            return self._save_csv(df, path / f"{filename}.csv", mode)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_parquet(self, df: pd.DataFrame, file_path: Path, mode: str) -> Path:
        """保存为Parquet格式"""
        compression_map = {
            CompressionType.SNAPPY: "snappy",
            CompressionType.GZIP: "gzip",
            CompressionType.BROTLI: "brotli",
            CompressionType.NONE: None
        }
        
        if mode == "append" and file_path.exists():
            # 追加模式：先读取再合并
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            
        df.to_parquet(
            file_path,
            engine="pyarrow",
            compression=compression_map[self.compression],
            index=False
        )
        
        file_size = file_path.stat().st_size / 1024 / 1024  # MB
        logger.info(f"Saved Parquet: {file_path} ({file_size:.2f} MB)")
        return file_path
    
    def _save_jsonl(self, df: pd.DataFrame, file_path: Path, mode: str) -> Path:
        """保存为JSONL格式（流式）"""
        write_mode = 'a' if mode == "append" else 'w'
        
        with open(file_path, write_mode, encoding='utf-8') as f:
            for _, row in df.iterrows():
                json.dump(row.to_dict(), f, ensure_ascii=False)
                f.write('\n')
                
        file_size = file_path.stat().st_size / 1024 / 1024
        logger.info(f"Saved JSONL: {file_path} ({file_size:.2f} MB)")
        return file_path
    
    def _save_csv(self, df: pd.DataFrame, file_path: Path, mode: str) -> Path:
        """保存为CSV格式"""
        write_mode = 'a' if mode == "append" else 'w'
        header = write_mode == 'w' or not file_path.exists()
        
        df.to_csv(
            file_path,
            mode=write_mode,
            header=header,
            index=False,
            encoding='utf-8-sig'
        )
        
        file_size = file_path.stat().st_size / 1024 / 1024
        logger.info(f"Saved CSV: {file_path} ({file_size:.2f} MB)")
        return file_path
    
    def load(
        self,
        domain: str,
        sub_domain: Optional[str] = None,
        format: Optional[StorageFormat] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            domain: 数据域
            sub_domain: 子域
            format: 存储格式
            filters: 过滤条件（如 {'year': 2024, 'month': 12}）
            
        Returns:
            DataFrame
        """
        fmt = format or self.default_format
        storage_path = self.base_path / domain
        if sub_domain:
            storage_path = storage_path / sub_domain
            
        if not storage_path.exists():
            logger.warning(f"Path not exists: {storage_path}")
            return pd.DataFrame()
            
        # 查找文件
        if fmt == StorageFormat.PARQUET:
            pattern = "**/*.parquet"
        elif fmt == StorageFormat.JSONL:
            pattern = "**/*.jsonl"
        elif fmt == StorageFormat.CSV:
            pattern = "**/*.csv"
        else:
            raise ValueError(f"Unsupported format: {fmt}")
            
        files = list(storage_path.glob(pattern))
        
        if not files:
            logger.warning(f"No files found: {storage_path}/{pattern}")
            return pd.DataFrame()
            
        # 应用分区过滤
        if filters:
            filtered_files = []
            for file in files:
                match = True
                for key, value in filters.items():
                    if f"{key}={value}" not in str(file):
                        match = False
                        break
                if match:
                    filtered_files.append(file)
            files = filtered_files
            
        # 加载数据
        dfs = []
        for file in files:
            if fmt == StorageFormat.PARQUET:
                dfs.append(pd.read_parquet(file))
            elif fmt == StorageFormat.JSONL:
                dfs.append(pd.read_json(file, lines=True))
            elif fmt == StorageFormat.CSV:
                dfs.append(pd.read_csv(file))
                
        if not dfs:
            return pd.DataFrame()
            
        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(result)} records from {len(files)} files")
        return result
    
    def get_storage_stats(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Args:
            domain: 指定数据域（None表示全部）
            
        Returns:
            统计信息字典
        """
        target_path = self.base_path / domain if domain else self.base_path
        
        if not target_path.exists():
            return {}
            
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'by_format': {},
            'by_domain': {}
        }
        
        for file in target_path.rglob('*'):
            if file.is_file():
                stats['total_files'] += 1
                size_mb = file.stat().st_size / 1024 / 1024
                stats['total_size_mb'] += size_mb
                
                # 按格式统计
                ext = file.suffix
                if ext not in stats['by_format']:
                    stats['by_format'][ext] = {'count': 0, 'size_mb': 0}
                stats['by_format'][ext]['count'] += 1
                stats['by_format'][ext]['size_mb'] += size_mb
                
                # 按域统计
                domain_name = file.relative_to(self.base_path).parts[0]
                if domain_name not in stats['by_domain']:
                    stats['by_domain'][domain_name] = {'count': 0, 'size_mb': 0}
                stats['by_domain'][domain_name]['count'] += 1
                stats['by_domain'][domain_name]['size_mb'] += size_mb
                
        return stats


# 全局单例
_global_sink: Optional[DataSink] = None


def get_data_sink(base_path: Optional[Union[str, Path]] = None) -> DataSink:
    """获取全局DataSink实例"""
    global _global_sink
    
    if _global_sink is None:
        if base_path is None:
            base_path = Path("data/raw/unstructured")
        _global_sink = DataSink(base_path=base_path)
        
    return _global_sink


def set_data_sink(sink: DataSink):
    """设置全局DataSink实例"""
    global _global_sink
    _global_sink = sink
