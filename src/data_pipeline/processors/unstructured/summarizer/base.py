"""
摘要生成器基类模块

定义摘要生成的统一接口和数据结构
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class DataType(Enum):
    """数据类型枚举 - 用于选择对应的Prompt模板"""
    
    # 公告类
    ANNOUNCEMENT = "announcement"           # 通用公告
    ANNOUNCEMENT_FINANCIAL = "announcement_financial"   # 财务公告（业绩报告等）
    ANNOUNCEMENT_MAJOR = "announcement_major"           # 重大事项公告（并购、重组等）
    ANNOUNCEMENT_MANAGEMENT = "announcement_management" # 管理层变动公告
    
    # 研报类
    REPORT = "report"                       # 通用研报
    REPORT_COMPANY = "report_company"       # 个股研报
    REPORT_INDUSTRY = "report_industry"     # 行业研报
    REPORT_STRATEGY = "report_strategy"     # 策略研报
    
    # 政策类
    POLICY = "policy"                       # 通用政策
    POLICY_FISCAL = "policy_fiscal"         # 财政政策
    POLICY_MONETARY = "policy_monetary"     # 货币政策
    POLICY_INDUSTRY = "policy_industry"     # 产业政策
    
    # 新闻类
    NEWS = "news"                           # 通用新闻
    NEWS_MARKET = "news_market"             # 市场新闻
    NEWS_COMPANY = "news_company"           # 公司新闻
    
    # 事件类
    EVENT = "event"                         # 事件公告
    
    # 通用
    GENERIC = "generic"                     # 通用文本
    
    @classmethod
    def from_domain(cls, domain: str) -> 'DataType':
        """根据数据域推断数据类型"""
        domain_mapping = {
            'announcements': cls.ANNOUNCEMENT,
            'reports': cls.REPORT,
            'policy': cls.POLICY,
            'policy/gov': cls.POLICY,
            'policy/ndrc': cls.POLICY_INDUSTRY,
            'news': cls.NEWS,
            'news/cctv': cls.NEWS_MARKET,
            'news/exchange': cls.NEWS_MARKET,
            'events': cls.EVENT,
        }
        return domain_mapping.get(domain, cls.GENERIC)


@dataclass
class SummarizerConfig:
    """摘要生成器配置"""
    
    # LLM配置
    model_name: str = "qwen2.5:7b-instruct"  # Ollama模型名称
    ollama_host: str = "http://localhost:11434"
    
    # 生成参数
    temperature: float = 0.3                  # 较低的温度确保输出稳定
    max_tokens: int = 256                     # 摘要最大token数
    top_p: float = 0.9
    top_k: int = 40
    
    # 摘要长度控制
    min_summary_chars: int = 50               # 摘要最小字数
    max_summary_chars: int = 150              # 摘要最大字数
    target_summary_chars: int = 100           # 目标摘要字数
    
    # 输入预处理
    max_input_chars: int = 8000               # 输入文本最大字数（超过则截断/分块）
    chunk_size: int = 4000                    # 分块大小
    chunk_overlap: int = 200                  # 分块重叠
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0                  # 重试间隔（秒）
    timeout: float = 60.0                     # 单次请求超时（秒）
    
    # 批量处理配置
    batch_size: int = 10                      # 批量处理大小
    max_concurrent: int = 4                   # 最大并发数
    
    # GPU配置
    use_gpu: bool = True                      # 是否使用GPU（由Ollama自动管理）
    
    def to_ollama_options(self) -> Dict[str, Any]:
        """转换为Ollama API选项"""
        return {
            'temperature': self.temperature,
            'num_predict': self.max_tokens,
            'top_p': self.top_p,
            'top_k': self.top_k,
        }


@dataclass
class SummaryResult:
    """摘要结果数据结构"""
    
    # 成功标志
    success: bool
    
    # 生成的摘要
    content: Optional[str] = None
    
    # 原始输入
    original_text: str = ""
    original_char_count: int = 0
    
    # 数据类型
    data_type: DataType = DataType.GENERIC
    
    # 使用的Prompt
    prompt_used: Optional[str] = None
    
    # 处理信息
    model_name: str = ""
    process_time_ms: float = 0.0
    input_tokens: Optional[int] = None        # 输入token数
    output_tokens: Optional[int] = None       # 输出token数
    
    # 摘要统计
    summary_char_count: Optional[int] = None
    compression_ratio: Optional[float] = None  # 压缩比（原文长度/摘要长度）
    
    # 错误信息
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后处理：计算统计信息"""
        if self.original_text:
            self.original_char_count = len(self.original_text)
        if self.content:
            self.summary_char_count = len(self.content)
        if self.original_char_count and self.summary_char_count:
            self.compression_ratio = self.original_char_count / self.summary_char_count
    
    @staticmethod
    def failure(
        original_text: str,
        error_message: str,
        error_code: str = "SUMMARIZE_ERROR",
        data_type: DataType = DataType.GENERIC
    ) -> 'SummaryResult':
        """创建失败结果的便捷方法"""
        return SummaryResult(
            success=False,
            original_text=original_text,
            data_type=data_type,
            error_message=error_message,
            error_code=error_code,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'content': self.content,
            'original_char_count': self.original_char_count,
            'data_type': self.data_type.value,
            'model_name': self.model_name,
            'process_time_ms': self.process_time_ms,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'summary_char_count': self.summary_char_count,
            'compression_ratio': self.compression_ratio,
            'error_message': self.error_message,
            'error_code': self.error_code,
            'metadata': self.metadata,
        }


class BaseSummarizer(ABC):
    """摘要生成器基类"""
    
    def __init__(self, config: Optional[SummarizerConfig] = None):
        """
        初始化摘要生成器
        
        Args:
            config: 配置对象，为None时使用默认配置
        """
        self.config = config or SummarizerConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def summarize(
        self,
        content_text: str,
        data_type: DataType = DataType.GENERIC,
        **kwargs
    ) -> SummaryResult:
        """
        生成摘要
        
        Args:
            content_text: 输入的原始文本（来自content_extractor）
            data_type: 数据类型，用于选择Prompt
            **kwargs: 额外参数
            
        Returns:
            SummaryResult: 摘要结果
        """
        pass
    
    @abstractmethod
    def summarize_batch(
        self,
        texts: List[str],
        data_types: Optional[List[DataType]] = None,
        **kwargs
    ) -> List[SummaryResult]:
        """
        批量生成摘要
        
        Args:
            texts: 文本列表
            data_types: 数据类型列表，长度需与texts一致
            **kwargs: 额外参数
            
        Returns:
            List[SummaryResult]: 摘要结果列表
        """
        pass


class TextCleaningMixin:
    """文本清洗混入类 - 支持GPU加速"""
    
    _cudf_available = None
    
    @classmethod
    def _check_cudf(cls) -> bool:
        """检查cuDF是否可用"""
        if cls._cudf_available is None:
            try:
                import cudf
                cls._cudf_available = True
            except ImportError:
                cls._cudf_available = False
        return cls._cudf_available
    
    @staticmethod
    def clean_text_basic(text: str) -> str:
        """基础文本清洗（CPU）"""
        import re
        
        if not text:
            return ""
        
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 移除重复的标点
        text = re.sub(r'([。！？，、；：])\1+', r'\1', text)
        
        # 统一引号
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    @classmethod
    def clean_text_batch_gpu(cls, texts: List[str]) -> List[str]:
        """GPU加速的批量文本清洗"""
        if not cls._check_cudf() or len(texts) < 10:
            # 数量少时使用CPU
            return [cls.clean_text_basic(t) for t in texts]
        
        try:
            import cudf
            
            # 转换为cuDF Series
            gs = cudf.Series(texts)
            
            # GPU上的字符串操作
            gs = gs.str.normalize_spaces()
            gs = gs.str.strip()
            
            # 转回Python列表
            result = gs.to_pandas().tolist()
            
            # 进一步的清洗（cuDF不支持的操作）
            return [cls.clean_text_basic(t) if t else "" for t in result]
            
        except Exception as e:
            logger.warning(f"GPU批量清洗失败，回退到CPU: {e}")
            return [cls.clean_text_basic(t) for t in texts]
