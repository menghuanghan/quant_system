"""
调度器基类和数据结构

定义处理配置、结果数据结构、数据类别枚举等
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    """数据类别枚举"""
    ANNOUNCEMENTS = "announcements"   # 公告 (PDF)
    REPORTS = "reports"               # 研报 (PDF)
    EVENTS = "events"                 # 事件 (PDF)
    EXCHANGE = "news/exchange"        # 交易所公告 (title only)
    CCTV = "news/cctv"                # CCTV新闻 (已有content, 生成Beta因子)
    POLICY_GOV = "policy/gov"         # 国务院政策 (HTML解析, 行业映射)
    POLICY_NDRC = "policy/ndrc"       # 发改委政策 (HTML解析, 行业映射)
    
    @property
    def requires_pdf(self) -> bool:
        """是否需要PDF处理"""
        return self in (
            DataCategory.ANNOUNCEMENTS,
            DataCategory.REPORTS,
            DataCategory.EVENTS
        )
    
    @property
    def requires_html(self) -> bool:
        """是否需要HTML解析"""
        return self in (
            DataCategory.POLICY_GOV,
            DataCategory.POLICY_NDRC
        )
    
    @property
    def requires_llm(self) -> bool:
        """是否需要LLM处理"""
        return self in (
            DataCategory.ANNOUNCEMENTS,
            DataCategory.REPORTS,
            DataCategory.EVENTS,
            DataCategory.CCTV,
            DataCategory.POLICY_GOV,
            DataCategory.POLICY_NDRC
        )
    
    @property
    def has_reason(self) -> bool:
        """输出是否包含reason字段"""
        return self.requires_pdf
    
    @property
    def is_cctv(self) -> bool:
        """是否是CCTV新闻"""
        return self == DataCategory.CCTV
    
    @property
    def is_policy(self) -> bool:
        """是否是政策数据"""
        return self in (DataCategory.POLICY_GOV, DataCategory.POLICY_NDRC)
    
    def get_id_column(self) -> str:
        """获取ID列名"""
        id_columns = {
            DataCategory.ANNOUNCEMENTS: "original_id",
            DataCategory.REPORTS: "report_id",
            DataCategory.EVENTS: "id",
            DataCategory.EXCHANGE: "news_id",
            DataCategory.CCTV: "news_id",
            DataCategory.POLICY_GOV: "id",
            DataCategory.POLICY_NDRC: "id",
        }
        return id_columns.get(self, "id")
    
    def get_code_column(self) -> str:
        """获取股票代码列名"""
        code_columns = {
            DataCategory.ANNOUNCEMENTS: "ts_code",
            DataCategory.REPORTS: "stock_code",
            DataCategory.EVENTS: "ts_code",
            DataCategory.EXCHANGE: "stock_code",
            DataCategory.CCTV: None,  # CCTV新闻是市场级别，没有股票代码
            DataCategory.POLICY_GOV: None,  # 政策是行业级别
            DataCategory.POLICY_NDRC: None,
        }
        return code_columns.get(self, "ts_code")
    
    def get_date_column(self) -> str:
        """获取日期列名"""
        return "date"
    
    def get_url_column(self) -> str:
        """获取URL列名"""
        url_columns = {
            DataCategory.ANNOUNCEMENTS: "url",
            DataCategory.REPORTS: "pdf_url",
            DataCategory.EVENTS: "pdf_url",
            DataCategory.EXCHANGE: None,  # 不需要URL
            DataCategory.CCTV: None,  # 已有content，不需要URL
            DataCategory.POLICY_GOV: "url",  # 需要下载HTML
            DataCategory.POLICY_NDRC: "url",
        }
        return url_columns.get(self)
    
    def get_title_column(self) -> str:
        """获取标题列名"""
        return "title"
    
    def get_content_column(self) -> str:
        """获取内容列名（如果已有）"""
        content_columns = {
            DataCategory.CCTV: "content",  # CCTV已有content
        }
        return content_columns.get(self)


class ProcessingStatus(Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingConfig:
    """处理配置"""
    
    # 路径配置
    raw_data_dir: str = "data/raw/unstructured"
    processed_data_dir: str = "data/processed/unstructured"
    checkpoint_dir: str = "data/checkpoints/unstructured_scheduler"
    
    # LLM配置
    model_name: str = "qwen2.5:7b-instruct"
    ollama_host: str = "http://localhost:11434"
    llm_timeout: float = 60.0
    llm_max_retries: int = 3
    
    # 处理配置
    batch_size: int = 10           # 每批处理数量
    max_workers: int = 4           # 最大并发数
    use_gpu: bool = True           # 是否使用GPU加速
    skip_existing: bool = True     # 跳过已处理的文件
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 日志配置
    log_level: str = "INFO"
    log_file: str = "logs/unstructured_scheduler.log"
    
    # 限流配置（防止API限流）
    requests_per_minute: int = 30
    request_delay: float = 0.5     # 请求间隔
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建目录
        for dir_path in [self.processed_data_dir, self.checkpoint_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_raw_path(self, category: DataCategory, year: int, month: int) -> Path:
        """获取原始数据路径"""
        return Path(self.raw_data_dir) / category.value / str(year) / f"{month:02d}.parquet"
    
    def get_processed_path(self, category: DataCategory, year: int, month: int) -> Path:
        """获取处理后数据路径"""
        return Path(self.processed_data_dir) / category.value / str(year) / f"{month:02d}.parquet"
    
    def get_checkpoint_path(self, category: DataCategory, year: int, month: int) -> Path:
        """获取检查点路径"""
        return Path(self.checkpoint_dir) / category.value / str(year) / f"{month:02d}.json"


@dataclass
class ProcessingResult:
    """单条处理结果"""
    success: bool
    record_id: str
    ts_code: str
    date: str
    score: Optional[int] = None
    reason: Optional[str] = None
    error_message: Optional[str] = None
    elapsed_time: float = 0.0
    
    # 中间结果（调试用）
    content_extracted: bool = False
    content_length: int = 0
    summary_generated: bool = False
    summary_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'record_id': self.record_id,
            'ts_code': self.ts_code,
            'date': self.date,
            'score': self.score,
            'reason': self.reason,
            'error_message': self.error_message,
            'elapsed_time': self.elapsed_time,
        }


@dataclass
class BatchProcessingResult:
    """批量处理结果"""
    category: DataCategory
    year: int
    month: int
    total: int
    success_count: int
    failed_count: int
    skipped_count: int
    results: List[ProcessingResult]
    elapsed_time_seconds: float
    output_path: Optional[str] = None
    
    # 统计
    avg_score: float = 0.0
    bullish_count: int = 0     # 利好数量
    bearish_count: int = 0     # 利空数量
    neutral_count: int = 0     # 中性数量
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success_count / self.total if self.total > 0 else 0.0
    
    @property
    def throughput(self) -> float:
        """吞吐量（每秒处理数）"""
        return self.total / self.elapsed_time_seconds if self.elapsed_time_seconds > 0 else 0.0
    
    def summary(self) -> str:
        """生成摘要"""
        return (
            f"{self.category.value}/{self.year}/{self.month:02d}: "
            f"总计 {self.total}, 成功 {self.success_count}, "
            f"失败 {self.failed_count}, 跳过 {self.skipped_count}, "
            f"成功率 {self.success_rate:.1%}, "
            f"耗时 {self.elapsed_time_seconds:.1f}s"
        )


@dataclass
class CheckpointData:
    """检查点数据"""
    category: str
    year: int
    month: int
    processed_ids: List[str] = field(default_factory=list)
    last_processed_index: int = 0
    total_records: int = 0
    success_count: int = 0
    failed_count: int = 0
    last_update_time: str = ""
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'category': self.category,
            'year': self.year,
            'month': self.month,
            'processed_ids': self.processed_ids,
            'last_processed_index': self.last_processed_index,
            'total_records': self.total_records,
            'success_count': self.success_count,
            'failed_count': self.failed_count,
            'last_update_time': self.last_update_time,
            'status': self.status,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointData':
        """从字典创建"""
        return cls(
            category=data.get('category', ''),
            year=data.get('year', 0),
            month=data.get('month', 0),
            processed_ids=data.get('processed_ids', []),
            last_processed_index=data.get('last_processed_index', 0),
            total_records=data.get('total_records', 0),
            success_count=data.get('success_count', 0),
            failed_count=data.get('failed_count', 0),
            last_update_time=data.get('last_update_time', ''),
            status=data.get('status', 'pending'),
        )


# ========== 输出字段配置 ==========

# PDF类数据输出字段（需要reason）
PDF_OUTPUT_COLUMNS = ['id', 'ts_code', 'date', 'score', 'reason']

# Exchange数据输出字段（不需要reason）
EXCHANGE_OUTPUT_COLUMNS = ['id', 'ts_code', 'date', 'score']

# CCTV新闻输出字段（市场情绪指数 + Beta信号）
CCTV_OUTPUT_COLUMNS = ['date', 'id', 'market_sentiment', 'beta_signal', 'keywords', 'tone_analysis']

# 政策数据输出字段（行业映射 + 行业打分）
POLICY_OUTPUT_COLUMNS = ['date', 'id', 'summary', 'benefited_industries', 'harmed_industries', 'industry_scores']

# 各类别的原始字段映射
FIELD_MAPPING = {
    DataCategory.ANNOUNCEMENTS: {
        'id': 'original_id',
        'ts_code': 'ts_code',
        'date': 'date',
        'url': 'url',
        'title': 'title',
    },
    DataCategory.REPORTS: {
        'id': 'report_id',
        'ts_code': 'stock_code',
        'date': 'date',
        'url': 'pdf_url',
        'title': 'title',
    },
    DataCategory.EVENTS: {
        'id': 'id',
        'ts_code': 'ts_code',
        'date': 'date',
        'url': 'pdf_url',
        'title': 'title',
    },
    DataCategory.EXCHANGE: {
        'id': 'news_id',
        'ts_code': 'stock_code',
        'date': 'date',
        'url': None,
        'title': 'title',
    },
    DataCategory.CCTV: {
        'id': 'news_id',
        'date': 'date',
        'title': 'title',
        'content': 'content',  # CCTV已有content
        'url': None,
    },
    DataCategory.POLICY_GOV: {
        'id': 'id',
        'date': 'date',
        'title': 'title',
        'url': 'url',
    },
    DataCategory.POLICY_NDRC: {
        'id': 'id',
        'date': 'date',
        'title': 'title',
        'url': 'url',
    },
}
