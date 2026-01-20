"""
非结构化数据采集器基类模块

提供非结构化数据（公告、新闻等）采集的基础设施：
- 统一元数据字段定义
- 公告类型枚举
- 日期时间处理工具
- 版本管理支持
- 流式缓冲池（管道化集成）
- 时间清洗与防泄露机制
"""

import os
import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
from dotenv import load_dotenv

# 导入时间清洗模块（防止未来函数）
try:
    from ...clean.unstructured.time_utils import (
        standardize_publish_time,
        TimeMode,
        TimeAccuracy,
        TimeNormalizer
    )
    HAS_TIME_UTILS = True
except ImportError:
    HAS_TIME_UTILS = False
    
# 导入存储管理器
from .storage import DataSink, StorageFormat, CompressionType, get_data_sink

# 加载环境变量
load_dotenv()

# 配置日志
logger = logging.getLogger(__name__)


class AnnouncementCategory(Enum):
    """公告类型枚举"""
    # 定期报告
    PERIODIC_ANNUAL = "年报"
    PERIODIC_SEMI = "中报"
    PERIODIC_Q1 = "一季报"
    PERIODIC_Q3 = "三季报"
    
    # 临时公告
    MERGER_ACQUISITION = "并购重组"
    EQUITY_INCREASE = "增持"
    EQUITY_DECREASE = "减持"
    EQUITY_CHANGE = "股权变动"
    MAJOR_CONTRACT = "重大合同"
    LITIGATION = "诉讼仲裁"
    REGULATORY_PENALTY = "监管处罚"
    RELATED_TRANSACTION = "关联交易"
    
    # 其他关键文本
    EARNINGS_FORECAST = "业绩预告"
    EARNINGS_EXPRESS = "业绩快报"
    MAJOR_EVENT = "重大事项"
    DIVIDEND = "分红派息"
    ISSUANCE = "发行上市"
    
    # 更正类
    CORRECTION = "更正公告"
    SUPPLEMENT = "补充公告"
    
    # 其他
    OTHER = "其他"
    
    @classmethod
    def from_string(cls, category_str: str) -> 'AnnouncementCategory':
        """从字符串转换为枚举值"""
        for cat in cls:
            if cat.value in category_str or category_str in cat.value:
                return cat
        return cls.OTHER
    
    @classmethod
    def get_periodic_categories(cls) -> List['AnnouncementCategory']:
        """获取定期报告类型列表"""
        return [cls.PERIODIC_ANNUAL, cls.PERIODIC_SEMI, 
                cls.PERIODIC_Q1, cls.PERIODIC_Q3]
    
    @classmethod
    def get_temporary_categories(cls) -> List['AnnouncementCategory']:
        """获取临时公告类型列表"""
        return [
            cls.MERGER_ACQUISITION, cls.EQUITY_INCREASE, cls.EQUITY_DECREASE,
            cls.EQUITY_CHANGE, cls.MAJOR_CONTRACT, cls.LITIGATION,
            cls.REGULATORY_PENALTY, cls.RELATED_TRANSACTION
        ]


class DataSourceType(Enum):
    """数据源类型枚举"""
    TUSHARE = "tushare"
    AKSHARE = "akshare"
    CNINFO = "cninfo"


@dataclass
class AnnouncementMetadata:
    """公告元数据标准字段"""
    ts_code: str                    # 股票代码（如 000001.SZ）
    name: str                       # 股票名称
    title: str                      # 公告标题
    ann_date: str                   # 公告日期（YYYY-MM-DD）
    category: str                   # 公告类型
    url: str                        # 公告原文URL
    source: str                     # 数据源（tushare/akshare/cninfo）
    
    # 可选字段
    ann_time: Optional[str] = None          # 公告时间（HH:MM:SS）
    is_correction: bool = False             # 是否为更正公告
    correction_of: Optional[str] = None     # 原公告ID（仅更正公告有值）
    list_status: str = 'L'                  # 公司上市状态（L/D/P）
    original_id: Optional[str] = None       # 原始公告ID
    file_path: Optional[str] = None         # 本地文件路径（如已下载）
    
    # 版本管理字段
    version: int = 1                        # 版本号
    is_latest: bool = True                  # 是否最新版本
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnnouncementMetadata':
        """从字典创建实例"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CollectionProgress:
    """采集进度跟踪（支持断点续传）"""
    task_id: str                            # 任务唯一ID
    start_date: str                         # 采集开始日期
    end_date: str                           # 采集结束日期
    current_date: Optional[str] = None      # 当前处理到的日期
    current_stock: Optional[str] = None     # 当前处理到的股票
    total_stocks: int = 0                   # 总股票数
    processed_stocks: int = 0               # 已处理股票数
    total_records: int = 0                  # 已采集记录数
    failed_records: List[str] = field(default_factory=list)  # 失败记录列表
    status: str = "pending"                 # pending/running/completed/failed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def update(self, **kwargs):
        """更新进度"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class UnstructuredCollector(ABC):
    """非结构化数据采集器基类"""
    
    # 标准输出字段
    STANDARD_FIELDS = [
        'ts_code', 'name', 'title', 'ann_date', 'ann_time',
        'category', 'url', 'source', 'is_correction', 
        'correction_of', 'list_status', 'original_id'
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def collect(
        self,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集数据的抽象方法
        
        Args:
            start_date: 开始日期（YYYY-MM-DD 格式）
            end_date: 结束日期（YYYY-MM-DD 格式）
            **kwargs: 其他参数
        
        Returns:
            DataFrame with standardized columns
        """
        pass
    
    def collect_incremental(
        self,
        since: Optional[str] = None,
        days: int = 1
    ) -> pd.DataFrame:
        """
        增量采集（用于调度器定期更新）
        
        Args:
            since: 从指定日期开始（为空则使用days参数）
            days: 采集最近N天数据（默认1天）
        
        Returns:
            DataFrame with new announcements
        """
        if since:
            start_date = since
        else:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        return self.collect(start_date=start_date, end_date=end_date)
    
    def _standardize_date(self, date_str: str) -> str:
        """
        标准化日期格式为 YYYY-MM-DD
        
        支持输入格式：YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD
        """
        if not date_str:
            return ""
        
        date_str = str(date_str).strip()
        
        # 已经是标准格式
        if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
            return date_str
        
        # YYYYMMDD 格式
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # YYYY/MM/DD 格式
        if '/' in date_str:
            return date_str.replace('/', '-')
        
        return date_str
    
    def _standardize_dataframe(
        self,
        df: pd.DataFrame,
        source: str
    ) -> pd.DataFrame:
        """
        标准化DataFrame输出
        
        Args:
            df: 原始DataFrame
            source: 数据源名称
        
        Returns:
            标准化后的DataFrame
        """
        if df.empty:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        # 添加数据源标识
        df['source'] = source
        
        # 标准化日期字段
        if 'ann_date' in df.columns:
            df['ann_date'] = df['ann_date'].apply(self._standardize_date)
        
        # 确保包含所有标准字段
        for col in self.STANDARD_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        # 设置默认值
        if 'is_correction' not in df.columns or df['is_correction'].isna().all():
            df['is_correction'] = False
        
        if 'list_status' not in df.columns or df['list_status'].isna().all():
            df['list_status'] = 'L'
        
        return df[self.STANDARD_FIELDS]
    
    # ============== 存储方法 ==============
    
    def _save_to_jsonl(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        append: bool = True,
        include_timestamp: bool = True
    ) -> bool:
        """
        保存数据到 JSONL 格式
        
        适用于：
        - 增量数据追加
        - 流式写入
        - 人类可读的文本格式
        
        Args:
            df: 要保存的 DataFrame
            file_path: 文件路径
            append: 是否追加模式
            include_timestamp: 是否包含采集时间戳
        
        Returns:
            是否保存成功
        """
        if df.empty:
            self.logger.warning("DataFrame 为空，跳过保存")
            return False
        
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 添加采集时间戳
            if include_timestamp:
                df = df.copy()
                df['_collected_at'] = datetime.now().isoformat()
            
            mode = 'a' if append else 'w'
            with open(file_path, mode, encoding='utf-8') as f:
                for _, row in df.iterrows():
                    record = row.to_dict()
                    # 处理 NaN 值
                    record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            self.logger.info(f"已保存 {len(df)} 条记录到 {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存 JSONL 失败: {e}")
            return False
    
    def _save_to_parquet(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        partition_cols: Optional[List[str]] = None,
        compression: str = 'snappy',
        append: bool = False
    ) -> bool:
        """
        保存数据到 Parquet 格式
        
        适用于：
        - 大规模历史数据存储（舆情数据）
        - 需要高压缩比的场景
        - 后续 Pandas 大规模读取分析
        
        优势：
        - 列式存储，压缩率高（通常 5-10x）
        - 读取速度快（支持列裁剪）
        - 保留数据类型信息
        
        Args:
            df: 要保存的 DataFrame
            file_path: 文件路径
            partition_cols: 分区列（如 ['trade_date', 'ts_code']）
            compression: 压缩算法 ('snappy', 'gzip', 'brotli', 'zstd')
            append: 是否追加到现有数据集
        
        Returns:
            是否保存成功
        """
        if df.empty:
            self.logger.warning("DataFrame 为空，跳过保存")
            return False
        
        try:
            file_path = Path(file_path)
            
            # 确保目录存在
            if partition_cols:
                file_path.mkdir(parents=True, exist_ok=True)
            else:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 添加采集时间戳
            df = df.copy()
            df['_collected_at'] = datetime.now().isoformat()
            
            # 标准化日期列为字符串（避免时区问题）
            for col in df.columns:
                if 'date' in col.lower():
                    df[col] = df[col].astype(str)
            
            if append and file_path.exists():
                # 追加模式：读取现有数据并合并
                if partition_cols:
                    # 分区数据集使用 pyarrow.parquet.write_to_dataset
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    
                    table = pa.Table.from_pandas(df)
                    pq.write_to_dataset(
                        table,
                        root_path=str(file_path),
                        partition_cols=partition_cols,
                        compression=compression,
                        existing_data_behavior='overwrite_or_ignore'
                    )
                else:
                    # 单文件追加
                    existing_df = pd.read_parquet(file_path)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_parquet(
                        file_path,
                        engine='pyarrow',
                        compression=compression,
                        index=False
                    )
            else:
                # 新建模式
                if partition_cols:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    
                    table = pa.Table.from_pandas(df)
                    pq.write_to_dataset(
                        table,
                        root_path=str(file_path),
                        partition_cols=partition_cols,
                        compression=compression
                    )
                else:
                    df.to_parquet(
                        file_path,
                        engine='pyarrow',
                        compression=compression,
                        index=False
                    )
            
            self.logger.info(
                f"已保存 {len(df)} 条记录到 {file_path} "
                f"(Parquet, {compression})"
            )
            return True
            
        except ImportError:
            self.logger.error(
                "Parquet 保存需要 pyarrow 库，请安装: pip install pyarrow"
            )
            return False
        except Exception as e:
            self.logger.error(f"保存 Parquet 失败: {e}")
            return False
    
    def _read_parquet(
        self,
        file_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None
    ) -> pd.DataFrame:
        """
        读取 Parquet 文件
        
        Args:
            file_path: 文件路径
            columns: 要读取的列（列裁剪优化）
            filters: 行过滤条件（谓词下推优化）
                     格式: [('column', 'operator', value), ...]
                     例如: [('trade_date', '>=', '2024-01-01')]
        
        Returns:
            DataFrame
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.warning(f"文件不存在: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_parquet(
                file_path,
                columns=columns,
                filters=filters,
                engine='pyarrow'
            )
            
            self.logger.info(f"已读取 {len(df)} 条记录从 {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"读取 Parquet 失败: {e}")
            return pd.DataFrame()
    
    def _save_auto(
        self,
        df: pd.DataFrame,
        base_path: Union[str, Path],
        name: str,
        use_parquet: bool = True,
        partition_by_date: bool = True
    ) -> bool:
        """
        自动选择存储格式
        
        规则：
        - 小数据量 (<10000行): JSONL
        - 大数据量 (>=10000行): Parquet
        - 可通过 use_parquet 强制指定
        
        Args:
            df: 要保存的 DataFrame
            base_path: 基础路径
            name: 文件名（不含扩展名）
            use_parquet: 强制使用 Parquet
            partition_by_date: 是否按日期分区（仅 Parquet）
        
        Returns:
            是否保存成功
        """
        if df.empty:
            return False
        
        base_path = Path(base_path)
        
        # 自动选择格式
        should_use_parquet = use_parquet or len(df) >= 10000
        
        if should_use_parquet:
            partition_cols = ['trade_date'] if partition_by_date and 'trade_date' in df.columns else None
            return self._save_to_parquet(
                df,
                base_path / f"{name}.parquet" if not partition_cols else base_path / name,
                partition_cols=partition_cols
            )
        else:
            return self._save_to_jsonl(
                df,
                base_path / f"{name}.jsonl"
            )
    
    def _detect_correction(self, title: str) -> bool:
        """
        检测是否为更正公告
        
        通过标题关键词判断
        """
        correction_keywords = [
            '更正', '补充更正', '更正公告', '补充公告',
            '修订', '修正', '勘误', '更改', '补正'
        ]
        return any(keyword in title for keyword in correction_keywords)
    
    def _extract_original_ref(self, title: str) -> Optional[str]:
        """
        从更正公告标题提取原公告引用
        
        例如：《关于2023年年度报告的更正公告》-> 2023年年度报告
        """
        import re
        
        # 尝试匹配《xxx》格式
        match = re.search(r'《(.+?)》', title)
        if match:
            return match.group(1)
        
        # 尝试匹配"关于xxx的更正"格式
        match = re.search(r'关于(.+?)的(?:更正|补充|修订)', title)
        if match:
            return match.group(1)
        
        return None
    
    def _deduplicate(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        数据去重
        
        Args:
            df: 原始DataFrame
            subset: 用于判断重复的列（默认使用 ts_code + title + ann_date）
        
        Returns:
            去重后的DataFrame
        """
        if df.empty:
            return df
        
        subset = subset or ['ts_code', 'title', 'ann_date']
        subset = [col for col in subset if col in df.columns]
        
        if not subset:
            return df
        
        original_len = len(df)
        df = df.drop_duplicates(subset=subset, keep='first')
        
        if len(df) < original_len:
            self.logger.info(f"去重: {original_len} -> {len(df)} 条记录")
        
        return df


# ============================================================================
# 流式采集器基类（管道化集成）
# ============================================================================

class ContentType(str, Enum):
    """内容类型枚举"""
    TEXT = 'text'           # 纯文本内容（清洗后）
    SCANNED_PDF = 'scanned' # 扫描件 PDF（需后续 OCR）
    FULL_PDF = 'full_pdf'   # 完整 PDF（高价值保留原件）
    HTML = 'html'           # HTML 内容（清洗后）


@dataclass 
class BufferedItem:
    """
    缓冲池条目
    
    存储已清洗但尚未落盘的数据。
    注意：content 字段仅存储清洗后的文本，原始 PDF/HTML 不进入缓冲池。
    """
    # 核心字段
    source: str                          # 数据源（cninfo/eastmoney/sina等）
    content_type: ContentType            # 内容类型
    
    # 时间字段（核心防泄露）
    publish_time: str                    # 标准化发布时间 (YYYY-MM-DD HH:MM:SS)
    time_accuracy: str                   # 时间精度 (Y/M/D/H/Mi/S)
    crawled_time: str                    # 采集时间（审计用）
    
    # 内容字段
    title: str                           # 标题
    content: Optional[str] = None        # 清洗后的文本内容
    url: Optional[str] = None            # 原始 URL
    file_path: Optional[str] = None      # 落盘文件路径（仅扫描件/高价值PDF）
    
    # 元数据
    ts_code: Optional[str] = None        # 股票代码
    name: Optional[str] = None           # 股票名称
    category: Optional[str] = None       # 分类
    original_id: Optional[str] = None    # 原始 ID
    
    # 扩展字段（子类可添加）
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为扁平字典（用于 DataFrame）"""
        d = {
            'source': self.source,
            'content_type': self.content_type.value if isinstance(self.content_type, ContentType) else self.content_type,
            'publish_time': self.publish_time,
            'time_accuracy': self.time_accuracy,
            'crawled_time': self.crawled_time,
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'file_path': self.file_path,
            'ts_code': self.ts_code,
            'name': self.name,
            'category': self.category,
            'original_id': self.original_id,
        }
        # 合并扩展字段
        d.update(self.extra)
        return d


class StreamingCollector(ABC):
    """
    流式采集器基类
    
    核心设计理念：
    1. 即时清洗（Extract-on-the-fly）：下载后立即清洗，不存储原始文件
    2. 缓冲池机制：批量落盘，减少 IO 次数
    3. 时间清洗（防泄露）：所有 publish_time 必须经过标准化
    4. 版本控制：保留所有采集记录（含 crawled_time），去重是读取层的事
    
    使用模式：
    ```python
    class MyCollector(StreamingCollector):
        def _collect_items(self, start_date, end_date, **kwargs):
            for item in fetch_data_from_api(...):
                yield {
                    'title': item['title'],
                    'publish_time': item['date'],  # 原始时间，将被清洗
                    'content': clean_text(item['body']),
                    ...
                }
    
    # 使用
    with MyCollector() as collector:
        collector.collect('2024-01-01', '2024-01-31')
    # 退出时自动 flush
    ```
    """
    
    # 子类必须定义的属性
    SOURCE_NAME: str = "base"           # 数据源名称
    DOMAIN: str = "unstructured"        # 数据域
    SUB_DOMAIN: Optional[str] = None    # 子域（如 announcements, news）
    
    # 配置参数
    DEFAULT_BUFFER_SIZE = 1000          # 默认缓冲区大小
    DEFAULT_TIME_MODE = 'conservative'  # 默认时间填充模式
    
    # 高价值公告类型（强制保留 PDF 原件）
    HIGH_VALUE_CATEGORIES = [
        '并购重组', '重大合同', '重大资产重组',
        '股权激励', '回购', '分红派息'
    ]
    
    def __init__(
        self,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        time_mode: Literal['conservative', 'aggressive', 'ultra_conservative'] = 'conservative',
        data_sink: Optional[DataSink] = None,
        base_path: Optional[Union[str, Path]] = None,
        enable_backup: bool = True,
        scanned_pdf_dir: Optional[Union[str, Path]] = None,
        high_value_pdf_dir: Optional[Union[str, Path]] = None
    ):
        """
        初始化流式采集器
        
        Args:
            buffer_size: 缓冲区大小（条），达到后自动 flush
            time_mode: 时间填充模式
                - 'conservative': 17:00（默认，最安全）
                - 'aggressive': 08:00（风险：可能泄露盘中数据）
                - 'ultra_conservative': T+1 09:30（最保守）
            data_sink: 数据落地管理器（为空则使用全局单例）
            base_path: 数据根目录
            enable_backup: 是否启用 JSONL 备份
            scanned_pdf_dir: 扫描件 PDF 存储目录
            high_value_pdf_dir: 高价值 PDF 存储目录
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.buffer_size = buffer_size
        self.time_mode = time_mode
        self.enable_backup = enable_backup
        
        # 缓冲池
        self._buffer: List[BufferedItem] = []
        self._current_month: Optional[str] = None  # 用于文件分片
        self._batch_id: int = 0
        
        # 统计信息
        self._stats = {
            'total_items': 0,
            'cleaned_items': 0,
            'dropped_items': 0,
            'scanned_pdfs': 0,
            'high_value_pdfs': 0,
            'flush_count': 0,
            'time_clean_failures': 0
        }
        
        # 数据落地
        if data_sink is not None:
            self._sink = data_sink
        elif base_path is not None:
            self._sink = DataSink(
                base_path=Path(base_path),
                enable_backup=enable_backup
            )
        else:
            self._sink = get_data_sink()
        
        # PDF 存储目录
        self.scanned_pdf_dir = Path(scanned_pdf_dir) if scanned_pdf_dir else \
            Path("data/raw/unstructured") / "scanned_pdfs"
        self.high_value_pdf_dir = Path(high_value_pdf_dir) if high_value_pdf_dir else \
            Path("data/raw/unstructured") / "high_value_pdfs"
    
    # ============== 核心流程方法 ==============
    
    @abstractmethod
    def _collect_items(
        self,
        start_date: str,
        end_date: str,
        **kwargs
    ):
        """
        采集数据的生成器（子类实现）
        
        每次 yield 一个字典，包含：
        - title: 标题
        - publish_time: 原始发布时间（将被清洗）
        - content: 清洗后的文本（可选，HTML/PDF 类采集器需提供）
        - url: 原始 URL（可选）
        - ts_code: 股票代码（可选）
        - category: 分类（可选）
        - ... 其他字段
        
        Yields:
            Dict[str, Any]: 数据字典
        """
        pass
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        主采集入口
        
        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            **kwargs: 其他参数
        
        Returns:
            采集统计信息
        """
        self.logger.info(f"开始采集 [{self.SOURCE_NAME}] {start_date} ~ {end_date}")
        
        # 重置统计
        self._stats = {
            'total_items': 0,
            'cleaned_items': 0,
            'dropped_items': 0,
            'scanned_pdfs': 0,
            'high_value_pdfs': 0,
            'flush_count': 0,
            'time_clean_failures': 0
        }
        
        try:
            # 迭代采集
            for raw_item in self._collect_items(start_date, end_date, **kwargs):
                self._stats['total_items'] += 1
                self._buffer_item(raw_item)
            
            # 最终 flush（不要丢失最后一批）
            if self._buffer:
                self._flush_buffer()
            
            self.logger.info(
                f"采集完成 [{self.SOURCE_NAME}]: "
                f"总计 {self._stats['total_items']}，"
                f"入库 {self._stats['cleaned_items']}，"
                f"丢弃 {self._stats['dropped_items']}"
            )
            
        except Exception as e:
            self.logger.error(f"采集异常: {e}")
            # 尝试保存已有数据
            if self._buffer:
                self.logger.warning(f"尝试保存缓冲区中的 {len(self._buffer)} 条数据...")
                try:
                    self._flush_buffer()
                except:
                    pass
            raise
        
        return self._stats.copy()
    
    def _buffer_item(self, raw_item: Dict[str, Any]):
        """
        缓冲单条数据（核心清洗关卡）
        
        流程：
        1. 时间清洗（防泄露）
        2. 添加审计时间
        3. 放入缓冲池
        4. 水位检测，触发 flush
        
        Args:
            raw_item: 原始数据字典
        """
        # ============ 第一步：时间清洗（防泄露关卡）============
        raw_time = raw_item.get('publish_time', '')
        
        if HAS_TIME_UTILS:
            try:
                cleaned_time, accuracy = TimeNormalizer.standardize_publish_time(
                    raw_time,
                    default_time_mode=self.time_mode,
                    return_accuracy=True
                )
            except Exception as e:
                self.logger.warning(f"时间清洗失败: {raw_time} - {e}")
                self._stats['time_clean_failures'] += 1
                self._stats['dropped_items'] += 1
                return  # 丢弃脏时间数据
        else:
            # 无 time_utils 时的降级处理
            cleaned_time = self._fallback_time_clean(raw_time)
            accuracy = 'D'  # 默认按天处理
        
        # 验证清洗结果
        if not cleaned_time or cleaned_time == 'INVALID':
            self.logger.debug(f"无效时间，丢弃: {raw_time}")
            self._stats['dropped_items'] += 1
            return
        
        # ============ 第二步：添加审计时间 ============
        crawled_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # ============ 第三步：构造缓冲条目 ============
        buffered_item = BufferedItem(
            source=self.SOURCE_NAME,
            content_type=ContentType(raw_item.get('content_type', 'text')),
            publish_time=cleaned_time,
            time_accuracy=accuracy.value if hasattr(accuracy, 'value') else str(accuracy),
            crawled_time=crawled_time,
            title=raw_item.get('title', ''),
            content=raw_item.get('content'),
            url=raw_item.get('url'),
            file_path=raw_item.get('file_path'),
            ts_code=raw_item.get('ts_code'),
            name=raw_item.get('name'),
            category=raw_item.get('category'),
            original_id=raw_item.get('original_id'),
            extra={k: v for k, v in raw_item.items() 
                   if k not in ('source', 'content_type', 'publish_time', 'crawled_time',
                               'title', 'content', 'url', 'file_path', 'ts_code',
                               'name', 'category', 'original_id', 'time_accuracy')}
        )
        
        # ============ 第四步：放入缓冲池 ============
        self._buffer.append(buffered_item)
        self._stats['cleaned_items'] += 1
        
        # ============ 第五步：水位检测 ============
        if len(self._buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _fallback_time_clean(self, raw_time: str) -> str:
        """
        时间清洗的降级实现（无 time_utils 时使用）
        
        保守策略：仅日期时，填充 17:00:00
        """
        if not raw_time:
            return 'INVALID'
        
        raw_time = str(raw_time).strip()
        
        # 尝试常见格式
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d',
            '%Y%m%d',
        ]
        
        dt = None
        has_time = ':' in raw_time
        
        for fmt in formats:
            try:
                dt = datetime.strptime(raw_time, fmt)
                break
            except ValueError:
                continue
        
        if dt is None:
            return 'INVALID'
        
        # 保守填充
        if not has_time:
            dt = dt.replace(hour=17, minute=0, second=0)
        
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    def _flush_buffer(self):
        """
        刷新缓冲区到磁盘
        
        策略：
        - 使用 Parquet (Snappy 压缩) 存储
        - 文件名格式：{source}_{year}{month}_{batch_id}.parquet
        - Append 模式，不覆盖旧文件
        """
        if not self._buffer:
            return
        
        self._stats['flush_count'] += 1
        self._batch_id += 1
        
        # 转为 DataFrame
        records = [item.to_dict() for item in self._buffer]
        df = pd.DataFrame(records)
        
        # 确定月份（用于文件分片）
        if 'publish_time' in df.columns and not df.empty:
            try:
                first_time = pd.to_datetime(df['publish_time'].iloc[0])
                self._current_month = first_time.strftime('%Y%m')
            except:
                self._current_month = datetime.now().strftime('%Y%m')
        else:
            self._current_month = datetime.now().strftime('%Y%m')
        
        # 文件名
        filename = f"{self.SOURCE_NAME}_{self._current_month}_{self._batch_id:04d}"
        
        # 使用 DataSink 落盘
        try:
            self._sink.save(
                data=df,
                domain=self.DOMAIN,
                sub_domain=self.SUB_DOMAIN,
                format=StorageFormat.PARQUET,
                filename=filename,
                mode="overwrite"  # 新文件，不追加
            )
            
            self.logger.info(
                f"Flush #{self._stats['flush_count']}: "
                f"{len(self._buffer)} 条 -> {filename}.parquet"
            )
            
        except Exception as e:
            self.logger.error(f"Flush 失败: {e}")
            # 尝试 JSONL 降级
            try:
                self._sink.save(
                    data=df,
                    domain=self.DOMAIN,
                    sub_domain=self.SUB_DOMAIN,
                    format=StorageFormat.JSONL,
                    filename=filename,
                    mode="overwrite"
                )
                self.logger.warning(f"降级保存为 JSONL: {filename}.jsonl")
            except Exception as e2:
                self.logger.error(f"JSONL 降级也失败: {e2}")
                raise
        
        # 清空缓冲区
        self._buffer.clear()
    
    # ============== PDF 分流处理 ==============
    
    def _save_scanned_pdf(
        self,
        pdf_bytes: bytes,
        ts_code: str,
        ann_date: str,
        title: str
    ) -> str:
        """
        保存扫描件 PDF（需后续 OCR）
        
        Args:
            pdf_bytes: PDF 二进制内容
            ts_code: 股票代码
            ann_date: 公告日期
            title: 标题（用于生成文件名）
        
        Returns:
            保存的文件路径
        """
        # 清理文件名
        safe_title = self._sanitize_filename(title)[:50]
        filename = f"{ts_code}_{ann_date}_{safe_title}.pdf"
        
        # 按年月分目录
        year_month = ann_date[:7].replace('-', '')
        save_dir = self.scanned_pdf_dir / year_month
        save_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = save_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(pdf_bytes)
        
        self._stats['scanned_pdfs'] += 1
        self.logger.debug(f"扫描件 PDF 已保存: {file_path}")
        
        return str(file_path)
    
    def _save_high_value_pdf(
        self,
        pdf_bytes: bytes,
        ts_code: str,
        ann_date: str,
        title: str,
        category: str
    ) -> str:
        """
        保存高价值 PDF（无论是否提取成功，均保留原件）
        
        Args:
            pdf_bytes: PDF 二进制内容
            ts_code: 股票代码
            ann_date: 公告日期
            title: 标题
            category: 分类
        
        Returns:
            保存的文件路径
        """
        safe_title = self._sanitize_filename(title)[:50]
        safe_category = self._sanitize_filename(category)
        filename = f"{ts_code}_{ann_date}_{safe_category}_{safe_title}.pdf"
        
        # 按分类和年月分目录
        year_month = ann_date[:7].replace('-', '')
        save_dir = self.high_value_pdf_dir / safe_category / year_month
        save_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = save_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(pdf_bytes)
        
        self._stats['high_value_pdfs'] += 1
        self.logger.debug(f"高价值 PDF 已保存: {file_path}")
        
        return str(file_path)
    
    def _is_high_value_category(self, category: str) -> bool:
        """判断是否为高价值类型"""
        if not category:
            return False
        return any(hv in category for hv in self.HIGH_VALUE_CATEGORIES)
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """清理文件名（移除非法字符）"""
        if not name:
            return "unknown"
        # 移除常见非法字符
        illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r']
        for char in illegal_chars:
            name = name.replace(char, '_')
        return name.strip()
    
    # ============== 上下文管理 ==============
    
    def close(self):
        """关闭采集器，强制 flush 剩余数据"""
        if self._buffer:
            self.logger.info(f"关闭前 flush 剩余 {len(self._buffer)} 条数据...")
            self._flush_buffer()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取采集统计"""
        return self._stats.copy()


class DateRangeIterator:
    """
    日期范围迭代器
    
    用于大规模历史数据的分批采集
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        chunk_days: int = 30
    ):
        """
        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            chunk_days: 每批处理的天数
        """
        self.start = datetime.strptime(start_date, '%Y-%m-%d')
        self.end = datetime.strptime(end_date, '%Y-%m-%d')
        self.chunk_days = chunk_days
        self.current = self.start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        
        chunk_start = self.current
        chunk_end = min(
            self.current + timedelta(days=self.chunk_days - 1),
            self.end
        )
        
        self.current = chunk_end + timedelta(days=1)
        
        return (
            chunk_start.strftime('%Y-%m-%d'),
            chunk_end.strftime('%Y-%m-%d')
        )
    
    def total_chunks(self) -> int:
        """计算总批次数"""
        total_days = (self.end - self.start).days + 1
        return (total_days + self.chunk_days - 1) // self.chunk_days


# 工具函数
def generate_task_id() -> str:
    """生成唯一任务ID"""
    import uuid
    return f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"


def parse_date_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: int = 5
) -> tuple:
    """
    解析日期范围
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        years: 默认年数（当start_date为空时使用）
    
    Returns:
        (start_date, end_date) 元组
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_dt = datetime.now() - timedelta(days=years*365)
        start_date = start_dt.strftime('%Y-%m-%d')
    
    return start_date, end_date
