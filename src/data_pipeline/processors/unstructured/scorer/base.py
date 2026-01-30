"""
量化打分器基类模块

定义打分的统一接口和数据结构
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class ScoreLevel(Enum):
    """分数等级枚举"""
    EXTREMELY_BULLISH = "extremely_bullish"    # 极度利好 (75-100)
    BULLISH = "bullish"                        # 利好 (25-75)
    SLIGHTLY_BULLISH = "slightly_bullish"      # 轻微利好 (5-25)
    NEUTRAL = "neutral"                        # 中性 (-5 到 5)
    SLIGHTLY_BEARISH = "slightly_bearish"      # 轻微利空 (-25 到 -5)
    BEARISH = "bearish"                        # 利空 (-75 到 -25)
    EXTREMELY_BEARISH = "extremely_bearish"    # 极度利空 (-100 到 -75)
    
    @classmethod
    def from_score(cls, score: float) -> 'ScoreLevel':
        """根据分数返回对应等级"""
        if score >= 75:
            return cls.EXTREMELY_BULLISH
        elif score >= 25:
            return cls.BULLISH
        elif score >= 5:
            return cls.SLIGHTLY_BULLISH
        elif score >= -5:
            return cls.NEUTRAL
        elif score >= -25:
            return cls.SLIGHTLY_BEARISH
        elif score >= -75:
            return cls.BEARISH
        else:
            return cls.EXTREMELY_BEARISH
    
    @property
    def label_cn(self) -> str:
        """中文标签"""
        labels = {
            self.EXTREMELY_BULLISH: "极度利好",
            self.BULLISH: "利好",
            self.SLIGHTLY_BULLISH: "轻微利好",
            self.NEUTRAL: "中性",
            self.SLIGHTLY_BEARISH: "轻微利空",
            self.BEARISH: "利空",
            self.EXTREMELY_BEARISH: "极度利空",
        }
        return labels.get(self, "未知")


class ScoringMethod(Enum):
    """打分方法枚举"""
    LLM = "llm"           # 基于LLM的语义理解打分
    RULE = "rule"         # 基于规则的正则匹配打分
    HYBRID = "hybrid"     # 混合方法（规则优先，LLM兜底）
    AUTO = "auto"         # 自动选择（根据数据类型）


@dataclass
class ScorerConfig:
    """打分器配置"""
    
    # 默认打分方法
    default_method: ScoringMethod = ScoringMethod.AUTO
    
    # LLM配置
    model_name: str = "qwen2.5:7b-instruct"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.1        # 更低的温度确保输出稳定
    max_tokens: int = 256
    
    # 分数范围
    min_score: int = -100
    max_score: int = 100
    
    # 规则打分配置
    rule_confidence_threshold: float = 0.8  # 规则匹配置信度阈值
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    
    # 批量处理
    batch_size: int = 20
    max_concurrent: int = 4
    
    # 数据类型 -> 打分方法映射
    # 对于这些类型使用规则打分（更快更准）
    RULE_BASED_TYPES = {
        'news/exchange',
        'news_exchange',
        'exchange_news',
    }
    
    def get_method_for_type(self, data_type: str) -> ScoringMethod:
        """根据数据类型获取打分方法"""
        if self.default_method != ScoringMethod.AUTO:
            return self.default_method
        
        # 交易所公告使用规则打分
        if data_type and data_type.lower() in self.RULE_BASED_TYPES:
            return ScoringMethod.RULE
        
        return ScoringMethod.LLM


@dataclass
class ScoreResult:
    """打分结果数据结构"""
    
    # 成功标志
    success: bool
    
    # 分数 (-100 到 100)
    score: Optional[int] = None
    
    # 归一化分数 (-1.0 到 1.0)
    normalized_score: Optional[float] = None
    
    # 分数等级
    level: Optional[ScoreLevel] = None
    
    # 打分理由（LLM生成）
    reason: Optional[str] = None
    
    # 原始输入
    content: str = ""
    data_type: str = ""
    
    # 打分方法
    method: ScoringMethod = ScoringMethod.LLM
    
    # 规则匹配信息（如果使用规则打分）
    matched_rule: Optional[str] = None
    rule_confidence: Optional[float] = None
    
    # 处理信息
    model_name: str = ""
    process_time_ms: float = 0.0
    
    # 错误信息
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后处理：计算归一化分数和等级"""
        if self.score is not None:
            # 确保分数在范围内
            self.score = max(-100, min(100, self.score))
            # 归一化到 [-1, 1]
            self.normalized_score = self.score / 100.0
            # 计算等级
            self.level = ScoreLevel.from_score(self.score)
    
    @staticmethod
    def failure(
        content: str,
        error_message: str,
        error_code: str = "SCORE_ERROR",
        data_type: str = ""
    ) -> 'ScoreResult':
        """创建失败结果的便捷方法"""
        return ScoreResult(
            success=False,
            content=content,
            data_type=data_type,
            error_message=error_message,
            error_code=error_code,
        )
    
    @staticmethod
    def neutral(
        content: str,
        reason: str = "无法判断情感倾向",
        data_type: str = ""
    ) -> 'ScoreResult':
        """创建中性结果的便捷方法"""
        return ScoreResult(
            success=True,
            score=0,
            content=content,
            data_type=data_type,
            reason=reason,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'score': self.score,
            'normalized_score': self.normalized_score,
            'level': self.level.value if self.level else None,
            'level_cn': self.level.label_cn if self.level else None,
            'reason': self.reason,
            'content': self.content[:100] + "..." if len(self.content) > 100 else self.content,
            'data_type': self.data_type,
            'method': self.method.value,
            'matched_rule': self.matched_rule,
            'process_time_ms': self.process_time_ms,
            'error_message': self.error_message,
        }
    
    @property
    def is_bullish(self) -> bool:
        """是否利好"""
        return self.score is not None and self.score > 5
    
    @property
    def is_bearish(self) -> bool:
        """是否利空"""
        return self.score is not None and self.score < -5
    
    @property
    def is_neutral(self) -> bool:
        """是否中性"""
        return self.score is not None and -5 <= self.score <= 5


class BaseScorer(ABC):
    """打分器基类"""
    
    def __init__(self, config: Optional[ScorerConfig] = None):
        """
        初始化打分器
        
        Args:
            config: 配置对象，为None时使用默认配置
        """
        self.config = config or ScorerConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def score(
        self,
        content: str,
        data_type: str = "",
        **kwargs
    ) -> ScoreResult:
        """
        对文本进行打分
        
        Args:
            content: 输入文本（摘要或标题）
            data_type: 数据类型
            **kwargs: 额外参数
            
        Returns:
            ScoreResult: 打分结果
        """
        pass
    
    @abstractmethod
    def score_batch(
        self,
        contents: List[str],
        data_types: Optional[List[str]] = None,
        **kwargs
    ) -> List[ScoreResult]:
        """
        批量打分
        
        Args:
            contents: 文本列表
            data_types: 数据类型列表
            **kwargs: 额外参数
            
        Returns:
            List[ScoreResult]: 打分结果列表
        """
        pass


# 分数参考标准
SCORE_GUIDELINES = """
分数标准（-100 到 100）：

极度利好 (75-100):
- 业绩大幅超预期（如净利润增长50%以上）
- 重大利好政策（直接利好行业）
- 重大并购/重组成功
- 获得重大订单/合同
- 行业龙头地位确认

利好 (25-75):
- 业绩正增长
- 新产品/技术突破
- 获得资质/认证
- 股东增持
- 积极的行业政策

轻微利好 (5-25):
- 经营稳定
- 小幅业绩改善
- 一般性利好消息

中性 (-5 到 5):
- 日常经营公告
- 例行信息披露
- 无明显影响的消息

轻微利空 (-25 到 -5):
- 小幅业绩下滑
- 一般性风险提示
- 行业小幅调整

利空 (-75 到 -25):
- 业绩下滑
- 收到监管函/警示函
- 重要人员离职
- 不利的行业政策
- 诉讼/仲裁

极度利空 (-100 到 -75):
- 业绩爆雷（如亏损、大幅下滑）
- 被立案调查
- 退市风险警示
- 重大违规/造假
- 核心业务重大损失
"""
