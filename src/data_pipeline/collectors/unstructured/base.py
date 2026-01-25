"""
非结构化数据采集器基类模块（简化版）

提供非结构化数据（公告、新闻等）采集的基础设施：
- 统一数据字段定义
- 公告类型枚举
- 日期时间处理工具
"""

import logging
import hashlib
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict

import pandas as pd

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
    AKSHARE = "akshare"
    CNINFO = "cninfo"
    EASTMONEY = "eastmoney"


@dataclass
class AnnouncementMetadata:
    """公告元数据结构"""
    ts_code: str
    name: str
    title: str
    date: str
    content: str
    category: str
    url: str
    source: str
    
    is_correction: bool = False
    correction_of: Optional[str] = None
    list_status: str = 'L'
    original_id: Optional[str] = None
    file_path: Optional[str] = None
    version: int = 1
    is_latest: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnnouncementMetadata':
        """从字典创建实例"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CollectionProgress:
    """采集进度跟踪"""
    task_id: str
    start_date: str
    end_date: str
    current_date: Optional[str] = None
    current_stock: Optional[str] = None
    total_stocks: int = 0
    processed_stocks: int = 0
    total_records: int = 0
    failed_records: List[str] = field(default_factory=list)
    status: str = "pending"
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
    """
    非结构化数据采集器基类
    
    职责：
    - 采集数据并返回DataFrame
    - 不负责数据存储、清洗、PDF下载等
    """
    
    STANDARD_FIELDS = [
        'ts_code', 'name', 'title', 'date', 'content',
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
        采集数据（核心方法）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
            
        Returns:
            包含采集数据的DataFrame
        """
        pass
    
    def collect_incremental(
        self,
        since: Optional[str] = None,
        days: int = 1
    ) -> pd.DataFrame:
        """增量采集"""
        if since:
            start_date = since
        else:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        return self.collect(start_date=start_date, end_date=end_date)
    
    def _standardize_date(self, date_str: str) -> str:
        """标准化日期格式为 YYYY-MM-DD"""
        if not date_str:
            return ""
        
        date_str = str(date_str).strip()
        
        if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
            return date_str
        
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        if '/' in date_str:
            return date_str.replace('/', '-')
        
        return date_str
    
    def _standardize_dataframe(
        self,
        df: pd.DataFrame,
        source: str
    ) -> pd.DataFrame:
        """标准化DataFrame输出"""
        if df.empty:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        df = df.copy()
        
        # 添加数据源标识
        if 'source' not in df.columns:
            df['source'] = source
        
        # 标准化日期
        if 'date' in df.columns:
            df['date'] = df['date'].apply(self._standardize_date)
        
        # 确保有必要字段
        for field in self.STANDARD_FIELDS:
            if field not in df.columns:
                df[field] = None
        
        return df


class DateRangeIterator:
    """日期范围迭代器"""
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        chunk_days: int = 30
    ):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.chunk_days = chunk_days
        self.current = self.start_date
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current > self.end_date:
            raise StopIteration
        
        chunk_start = self.current
        chunk_end = min(
            self.current + timedelta(days=self.chunk_days - 1),
            self.end_date
        )
        
        self.current = chunk_end + timedelta(days=1)
        
        return (
            chunk_start.strftime('%Y-%m-%d'),
            chunk_end.strftime('%Y-%m-%d')
        )


def generate_task_id(prefix: str = "task") -> str:
    """生成任务ID"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    hash_part = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
    return f"{prefix}_{timestamp}_{hash_part}"


def parse_date_range(date_str: str) -> tuple:
    """解析日期范围字符串"""
    if '~' in date_str:
        parts = date_str.split('~')
        return parts[0].strip(), parts[1].strip()
    return date_str, date_str
