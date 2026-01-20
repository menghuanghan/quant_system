"""
Schema 统一与 Hive 分区存储模块

功能：
1. 定义标准化 Schema，确保所有 Parquet 文件列一致
2. 支持 Hive 风格分区存储 (year=YYYY/month=MM/)
3. 提供 Schema 验证和转换工具

标准 Schema 字段：
- event_id: String (唯一ID)
- publish_time: Timestamp (发布时间，已时区标准化)
- ticker: String (关联标的)
- title: String (标题)
- content: String (清洗后文本)
- source_url: String (来源URL)
- source_type: String (数据源类型)
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

logger = logging.getLogger(__name__)


# ============== Schema 定义 ==============

# 标准字段定义
STANDARD_FIELDS = {
    # 核心字段
    'event_id': pa.string(),           # 唯一事件ID
    'publish_time': pa.timestamp('ms'), # 发布时间（毫秒精度）
    'collect_time': pa.timestamp('ms'), # 采集时间
    
    # 内容字段
    'title': pa.string(),              # 标题
    'content': pa.string(),            # 清洗后的正文
    'summary': pa.string(),            # 摘要（如有）
    'raw_content': pa.string(),        # 原始内容（可选）
    
    # 关联字段
    'ticker': pa.string(),             # 主关联股票（单个）
    'related_securities': pa.list_(pa.string()),  # 关联股票列表
    
    # 来源字段
    'source_url': pa.string(),         # 来源URL
    'source_type': pa.string(),        # 数据源类型
    'source_id': pa.string(),          # 数据源内部ID
    
    # 分类字段
    'category': pa.string(),           # 分类
    'tags': pa.list_(pa.string()),     # 标签列表
    
    # 元数据
    'author': pa.string(),             # 作者
    'file_path': pa.string(),          # 关联文件路径（如PDF）
    'version': pa.int32(),             # 版本号
    
    # 分区字段（用于 Hive 分区）
    'year': pa.int32(),
    'month': pa.int32(),
}


# 各数据类型的必需字段
REQUIRED_FIELDS = {
    'announcement': ['event_id', 'publish_time', 'title', 'content', 'source_url', 'source_type'],
    'news': ['event_id', 'publish_time', 'title', 'content', 'source_url', 'source_type'],
    'report': ['event_id', 'publish_time', 'title', 'source_url', 'source_type'],
    'sentiment': ['event_id', 'publish_time', 'content', 'ticker', 'source_type'],
    'event': ['event_id', 'publish_time', 'title', 'ticker', 'category'],
}


def get_standard_schema(
    data_type: str = 'announcement',
    include_optional: bool = False
) -> pa.Schema:
    """
    获取标准 Schema
    
    Args:
        data_type: 数据类型 ('announcement', 'news', 'report', 'sentiment', 'event')
        include_optional: 是否包含可选字段
    
    Returns:
        PyArrow Schema
    """
    required = REQUIRED_FIELDS.get(data_type, REQUIRED_FIELDS['announcement'])
    
    if include_optional:
        fields = [(name, dtype) for name, dtype in STANDARD_FIELDS.items()]
    else:
        fields = [(name, STANDARD_FIELDS[name]) for name in required if name in STANDARD_FIELDS]
        # 添加分区字段
        fields.append(('year', pa.int32()))
        fields.append(('month', pa.int32()))
    
    return pa.schema(fields)


# ============== ID 生成 ==============

def generate_event_id(
    source_type: str,
    source_id: Optional[str] = None,
    url: Optional[str] = None,
    title: Optional[str] = None,
    publish_time: Optional[datetime] = None
) -> str:
    """
    生成唯一事件ID
    
    ID 格式: {source_type}_{hash}_{timestamp}
    
    Args:
        source_type: 数据源类型
        source_id: 数据源内部ID（如有）
        url: URL
        title: 标题
        publish_time: 发布时间
    
    Returns:
        唯一事件ID
    """
    if source_id:
        # 优先使用数据源ID
        hash_input = f"{source_type}_{source_id}"
    elif url:
        # 使用URL哈希
        hash_input = url
    else:
        # 使用标题+时间
        time_str = publish_time.isoformat() if publish_time else datetime.now().isoformat()
        hash_input = f"{title}_{time_str}"
    
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    if publish_time:
        date_str = publish_time.strftime('%Y%m%d')
    else:
        date_str = datetime.now().strftime('%Y%m%d')
    
    return f"{source_type}_{date_str}_{hash_value}"


# ============== Schema 验证 ==============

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


def validate_dataframe(
    df: pd.DataFrame,
    data_type: str = 'announcement'
) -> ValidationResult:
    """
    验证 DataFrame 是否符合 Schema
    
    Args:
        df: 待验证的 DataFrame
        data_type: 数据类型
    
    Returns:
        ValidationResult
    """
    errors = []
    warnings = []
    
    required = REQUIRED_FIELDS.get(data_type, REQUIRED_FIELDS['announcement'])
    
    # 检查必需字段
    for field in required:
        if field not in df.columns:
            errors.append(f"缺少必需字段: {field}")
    
    # 检查字段类型
    if 'publish_time' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['publish_time']):
            warnings.append("publish_time 应为 datetime 类型")
    
    if 'related_securities' in df.columns:
        if not df['related_securities'].apply(lambda x: isinstance(x, (list, type(None)))).all():
            warnings.append("related_securities 应为列表类型")
    
    # 检查空值
    if 'event_id' in df.columns:
        null_count = df['event_id'].isna().sum()
        if null_count > 0:
            errors.append(f"event_id 存在 {null_count} 个空值")
    
    if 'publish_time' in df.columns:
        null_count = df['publish_time'].isna().sum()
        if null_count > 0:
            warnings.append(f"publish_time 存在 {null_count} 个空值")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def normalize_dataframe(
    df: pd.DataFrame,
    data_type: str = 'announcement',
    source_type: str = 'unknown'
) -> pd.DataFrame:
    """
    标准化 DataFrame
    
    - 添加缺失的必需字段
    - 生成 event_id
    - 添加分区字段
    
    Args:
        df: 输入 DataFrame
        data_type: 数据类型
        source_type: 数据源类型
    
    Returns:
        标准化后的 DataFrame
    """
    df = df.copy()
    
    # 生成 event_id
    if 'event_id' not in df.columns or df['event_id'].isna().any():
        df['event_id'] = df.apply(
            lambda row: generate_event_id(
                source_type=source_type,
                source_id=row.get('source_id'),
                url=row.get('source_url'),
                title=row.get('title'),
                publish_time=row.get('publish_time')
            ),
            axis=1
        )
    
    # 确保 publish_time 是 datetime
    if 'publish_time' in df.columns:
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    
    # 添加分区字段
    if 'publish_time' in df.columns:
        df['year'] = df['publish_time'].dt.year.astype('Int32')
        df['month'] = df['publish_time'].dt.month.astype('Int32')
    else:
        df['year'] = datetime.now().year
        df['month'] = datetime.now().month
    
    # 添加采集时间
    if 'collect_time' not in df.columns:
        df['collect_time'] = datetime.now()
    
    # 确保 source_type 存在
    if 'source_type' not in df.columns:
        df['source_type'] = source_type
    
    # 添加缺失的可选字段
    optional_fields = {
        'summary': None,
        'raw_content': None,
        'ticker': None,
        'related_securities': lambda: [],
        'category': None,
        'tags': lambda: [],
        'author': None,
        'file_path': None,
        'version': 1,
    }
    
    for field, default in optional_fields.items():
        if field not in df.columns:
            if callable(default):
                df[field] = df.apply(lambda _: default(), axis=1)
            else:
                df[field] = default
    
    return df


# ============== Hive 分区存储 ==============

class HivePartitionWriter:
    """
    Hive 风格分区存储写入器
    
    自动按 year/month 分区存储数据，支持：
    - 自动创建分区目录
    - 追加写入模式
    - Schema 一致性检查
    
    目录结构：
        base_dir/
        └── year=2024/
            └── month=01/
                └── part-0001.snappy.parquet
    
    Example:
        >>> writer = HivePartitionWriter("data/processed/announcements")
        >>> writer.write(df, partition_cols=['year', 'month'])
    """
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        data_type: str = 'announcement',
        compression: str = 'snappy'
    ):
        """
        Args:
            base_dir: 基础目录
            data_type: 数据类型
            compression: 压缩方式 ('snappy', 'gzip', 'lz4', 'zstd')
        """
        self.base_dir = Path(base_dir)
        self.data_type = data_type
        self.compression = compression
        
        # 确保目录存在
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取 Schema
        self.schema = get_standard_schema(data_type, include_optional=True)
    
    def write(
        self,
        df: pd.DataFrame,
        partition_cols: List[str] = ['year', 'month'],
        mode: str = 'append'
    ):
        """
        写入数据
        
        Args:
            df: 数据
            partition_cols: 分区列
            mode: 写入模式 ('append', 'overwrite')
        """
        if df.empty:
            logger.warning("空 DataFrame，跳过写入")
            return
        
        # 标准化
        df = normalize_dataframe(df, self.data_type)
        
        # 验证
        result = validate_dataframe(df, self.data_type)
        if not result.is_valid:
            for error in result.errors:
                logger.error(f"Schema 验证失败: {error}")
            raise ValueError(f"Schema 验证失败: {result.errors}")
        
        for warning in result.warnings:
            logger.warning(f"Schema 警告: {warning}")
        
        # 使用 PyArrow 写入分区
        table = pa.Table.from_pandas(df)
        
        pq.write_to_dataset(
            table,
            root_path=str(self.base_dir),
            partition_cols=partition_cols,
            compression=self.compression,
            existing_data_behavior='overwrite_or_ignore' if mode == 'append' else 'delete_matching'
        )
        
        logger.info(f"写入 {len(df)} 条记录到 {self.base_dir}")
    
    def write_month(
        self,
        df: pd.DataFrame,
        year: int,
        month: int,
        part_num: int = 1
    ):
        """
        写入单月数据
        
        Args:
            df: 数据
            year: 年份
            month: 月份
            part_num: 分片编号
        """
        if df.empty:
            return
        
        # 添加分区字段
        df = df.copy()
        df['year'] = year
        df['month'] = month
        
        # 标准化
        df = normalize_dataframe(df, self.data_type)
        
        # 创建分区目录
        partition_dir = self.base_dir / f"year={year}" / f"month={month:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        filename = f"part-{part_num:04d}.{self.compression}.parquet"
        file_path = partition_dir / filename
        
        # 写入
        df.to_parquet(file_path, compression=self.compression, index=False)
        logger.info(f"写入: {file_path} ({len(df)} 条)")


class HivePartitionReader:
    """
    Hive 风格分区存储读取器
    
    支持高效的分区裁剪，仅读取需要的数据
    
    Example:
        >>> reader = HivePartitionReader("data/processed/announcements")
        >>> df = reader.read(year=2024, month=1)
        >>> df = reader.read(year_range=(2023, 2024))
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
    
    def read(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
        year_range: Optional[tuple] = None,
        columns: Optional[List[str]] = None,
        filters: Optional[List] = None
    ) -> pd.DataFrame:
        """
        读取数据
        
        Args:
            year: 指定年份
            month: 指定月份
            year_range: 年份范围 (start, end)
            columns: 要读取的列
            filters: PyArrow 过滤条件
        
        Returns:
            DataFrame
        """
        # 构建过滤条件
        partition_filters = []
        
        if year:
            partition_filters.append(('year', '==', year))
        elif year_range:
            partition_filters.append(('year', '>=', year_range[0]))
            partition_filters.append(('year', '<=', year_range[1]))
        
        if month:
            partition_filters.append(('month', '==', month))
        
        combined_filters = partition_filters + (filters or [])
        
        try:
            df = pd.read_parquet(
                self.base_dir,
                columns=columns,
                filters=combined_filters if combined_filters else None
            )
            return df
        except Exception as e:
            logger.error(f"读取失败: {e}")
            return pd.DataFrame()
    
    def list_partitions(self) -> List[Dict[str, int]]:
        """列出所有分区"""
        partitions = []
        
        for year_dir in self.base_dir.glob("year=*"):
            year = int(year_dir.name.split('=')[1])
            
            for month_dir in year_dir.glob("month=*"):
                month = int(month_dir.name.split('=')[1])
                partitions.append({'year': year, 'month': month})
        
        return sorted(partitions, key=lambda x: (x['year'], x['month']))
    
    def get_partition_stats(self) -> Dict[str, Any]:
        """获取分区统计"""
        stats = {
            'total_partitions': 0,
            'total_files': 0,
            'total_size_mb': 0,
            'partitions': []
        }
        
        for partition in self.list_partitions():
            year, month = partition['year'], partition['month']
            partition_dir = self.base_dir / f"year={year}" / f"month={month:02d}"
            
            files = list(partition_dir.glob("*.parquet"))
            size = sum(f.stat().st_size for f in files)
            
            stats['partitions'].append({
                'year': year,
                'month': month,
                'files': len(files),
                'size_mb': size / (1024 * 1024)
            })
            
            stats['total_partitions'] += 1
            stats['total_files'] += len(files)
            stats['total_size_mb'] += size / (1024 * 1024)
        
        return stats


# ============== 便捷函数 ==============

def save_with_partition(
    df: pd.DataFrame,
    base_dir: Union[str, Path],
    data_type: str = 'announcement'
):
    """
    便捷函数：保存数据（自动分区）
    """
    writer = HivePartitionWriter(base_dir, data_type)
    writer.write(df)


def load_partition(
    base_dir: Union[str, Path],
    year: Optional[int] = None,
    month: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    便捷函数：加载分区数据
    """
    reader = HivePartitionReader(base_dir)
    return reader.read(year=year, month=month, **kwargs)
