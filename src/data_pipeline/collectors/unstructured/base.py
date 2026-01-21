"""
非结构化数据采集器基类模块

提供非结构化数据（公告、新闻等）采集的基础设施：
- 统一元数据字段定义
- 公告类型枚举
- 日期时间处理工具
- 版本管理支持
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

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
