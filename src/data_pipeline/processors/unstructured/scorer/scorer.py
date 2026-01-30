"""
量化打分器主模块

提供统一的API接口，支持：
- 自动选择打分策略（规则/LLM）
- 单条/批量打分
- DataFrame行级打分
- 混合打分（规则优先，LLM兜底）

这是scorer模块的主入口
"""

import logging
import time
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .base import (
    ScoreResult,
    ScorerConfig,
    ScoreLevel,
    ScoringMethod,
    BaseScorer,
)
from .rule_scorer import RuleScorer
from .llm_scorer import LLMScorer

logger = logging.getLogger(__name__)


@dataclass
class BatchScoreResult:
    """批量打分结果"""
    total: int
    success_count: int
    failed_count: int
    results: List[ScoreResult]
    elapsed_time_seconds: float
    
    # 统计
    bullish_count: int = 0      # 利好数量
    bearish_count: int = 0      # 利空数量
    neutral_count: int = 0      # 中性数量
    
    avg_score: float = 0.0      # 平均分数
    rule_scored: int = 0        # 规则打分数量
    llm_scored: int = 0         # LLM打分数量
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success_count / self.total if self.total > 0 else 0.0
    
    @property
    def throughput(self) -> float:
        """吞吐量（每秒处理数）"""
        return self.total / self.elapsed_time_seconds if self.elapsed_time_seconds > 0 else 0.0
    
    def get_bullish_results(self) -> List[ScoreResult]:
        """获取利好结果"""
        return [r for r in self.results if r.is_bullish]
    
    def get_bearish_results(self) -> List[ScoreResult]:
        """获取利空结果"""
        return [r for r in self.results if r.is_bearish]
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        records = [r.to_dict() for r in self.results]
        return pd.DataFrame(records)


class Scorer(BaseScorer):
    """
    量化打分器
    
    将摘要/标题文本转换为情感分数(-100到100)，用于量化分析。
    
    特点：
    1. 自动策略选择：交易所公告用规则打分，其他用LLM
    2. 混合打分：规则优先，LLM兜底
    3. 支持批量处理和DataFrame操作
    4. GPU加速的规则批量匹配
    
    使用示例：
    ```python
    from src.data_pipeline.processors.unstructured.scorer import Scorer
    
    scorer = Scorer()
    
    # 单条打分
    result = scorer.score(
        content="公司净利润同比增长80%",
        data_type="announcement"
    )
    print(result.score)   # 75
    print(result.reason)  # "业绩大幅增长"
    print(result.level)   # ScoreLevel.BULLISH
    
    # 交易所公告（自动使用规则打分）
    result = scorer.score(
        content="关于对XXX公司采取出具警示函的决定",
        data_type="news/exchange"
    )
    print(result.score)   # -55
    print(result.method)  # ScoringMethod.RULE
    
    # 批量打分
    batch_result = scorer.score_batch(
        contents=["文本1", "文本2"],
        data_types=["announcement", "news/exchange"]
    )
    print(f"平均分: {batch_result.avg_score:.1f}")
    ```
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        config: Optional[ScorerConfig] = None,
        rule_scorer: Optional[RuleScorer] = None,
        llm_scorer: Optional[LLMScorer] = None
    ):
        """
        初始化打分器
        
        Args:
            config: 配置对象
            rule_scorer: 规则打分器（可选，用于注入自定义实例）
            llm_scorer: LLM打分器（可选）
        """
        super().__init__(config)
        
        # 初始化子打分器
        self.rule_scorer = rule_scorer or RuleScorer(self.config)
        
        # LLM打分器延迟初始化（可能不需要）
        self._llm_scorer = llm_scorer
        self._llm_scorer_initialized = llm_scorer is not None
        
        self.logger.info(
            f"打分器初始化完成 - 默认方法: {self.config.default_method.value}"
        )
    
    @property
    def llm_scorer(self) -> LLMScorer:
        """延迟初始化LLM打分器"""
        if not self._llm_scorer_initialized:
            self._llm_scorer = LLMScorer(self.config)
            self._llm_scorer_initialized = True
        return self._llm_scorer
    
    def score(
        self,
        content: str,
        data_type: str = "",
        method: Optional[ScoringMethod] = None,
        **kwargs
    ) -> ScoreResult:
        """
        对文本进行打分
        
        Args:
            content: 输入文本（摘要或标题）
            data_type: 数据类型（用于自动选择打分方法）
            method: 指定打分方法（覆盖自动选择）
            **kwargs: 额外参数
            
        Returns:
            ScoreResult: 打分结果
        """
        start_time = time.time()
        
        if not content or not content.strip():
            return ScoreResult.failure(
                content="",
                error_message="输入文本为空",
                error_code="EMPTY_INPUT",
                data_type=data_type
            )
        
        # 确定打分方法
        if method is None:
            method = self.config.get_method_for_type(data_type)
        
        # 执行打分
        if method == ScoringMethod.RULE:
            result = self.rule_scorer.score(content, data_type, **kwargs)
        elif method == ScoringMethod.LLM:
            result = self.llm_scorer.score(content, data_type, **kwargs)
        elif method == ScoringMethod.HYBRID:
            result = self._score_hybrid(content, data_type, **kwargs)
        else:  # AUTO
            # 先尝试规则打分
            result = self.rule_scorer.score(content, data_type, **kwargs)
            # 如果规则匹配置信度低，使用LLM
            if result.rule_confidence and result.rule_confidence < self.config.rule_confidence_threshold:
                result = self.llm_scorer.score(content, data_type, **kwargs)
        
        return result
    
    def _score_hybrid(
        self,
        content: str,
        data_type: str,
        **kwargs
    ) -> ScoreResult:
        """
        混合打分：规则优先，LLM兜底
        
        1. 先尝试规则匹配
        2. 如果规则命中且置信度高，直接返回
        3. 否则使用LLM打分
        """
        # 规则打分
        rule_result = self.rule_scorer.score(content, data_type, **kwargs)
        
        # 如果规则命中且置信度高
        if (rule_result.matched_rule and 
            rule_result.rule_confidence >= self.config.rule_confidence_threshold):
            return rule_result
        
        # 使用LLM打分
        return self.llm_scorer.score(content, data_type, **kwargs)
    
    def score_batch(
        self,
        contents: List[str],
        data_types: Optional[List[str]] = None,
        method: Optional[ScoringMethod] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> BatchScoreResult:
        """
        批量打分
        
        Args:
            contents: 文本列表
            data_types: 数据类型列表
            method: 指定打分方法（覆盖自动选择）
            progress_callback: 进度回调函数
            **kwargs: 额外参数
            
        Returns:
            BatchScoreResult: 批量结果
        """
        start_time = time.time()
        total = len(contents)
        
        if total == 0:
            return BatchScoreResult(
                total=0, success_count=0, failed_count=0,
                results=[], elapsed_time_seconds=0.0
            )
        
        # 准备数据类型
        if data_types is None:
            data_types = [""] * total
        elif len(data_types) != total:
            data_types = data_types + [""] * (total - len(data_types))
        
        # 分组处理：规则打分 vs LLM打分
        rule_indices = []
        llm_indices = []
        
        for i, dtype in enumerate(data_types):
            if method == ScoringMethod.RULE:
                rule_indices.append(i)
            elif method == ScoringMethod.LLM:
                llm_indices.append(i)
            else:
                # AUTO模式：根据数据类型选择
                selected = self.config.get_method_for_type(dtype)
                if selected == ScoringMethod.RULE:
                    rule_indices.append(i)
                else:
                    llm_indices.append(i)
        
        # 初始化结果列表
        results = [None] * total
        
        # 规则批量打分（快速）
        if rule_indices:
            rule_contents = [contents[i] for i in rule_indices]
            rule_dtypes = [data_types[i] for i in rule_indices]
            rule_results = self.rule_scorer.score_batch(rule_contents, rule_dtypes, **kwargs)
            for idx, result in zip(rule_indices, rule_results):
                results[idx] = result
        
        # LLM打分（较慢，串行）
        completed = len(rule_indices)
        for i, idx in enumerate(llm_indices):
            result = self.llm_scorer.score(contents[idx], data_types[idx], **kwargs)
            results[idx] = result
            completed += 1
            
            if progress_callback:
                progress_callback(completed, total)
        
        elapsed = time.time() - start_time
        
        # 计算统计
        success_count = sum(1 for r in results if r.success)
        bullish_count = sum(1 for r in results if r.is_bullish)
        bearish_count = sum(1 for r in results if r.is_bearish)
        neutral_count = sum(1 for r in results if r.is_neutral)
        
        scores = [r.score for r in results if r.success and r.score is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        rule_scored = sum(1 for r in results if r.method == ScoringMethod.RULE)
        llm_scored = sum(1 for r in results if r.method == ScoringMethod.LLM)
        
        return BatchScoreResult(
            total=total,
            success_count=success_count,
            failed_count=total - success_count,
            results=results,
            elapsed_time_seconds=elapsed,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            avg_score=avg_score,
            rule_scored=rule_scored,
            llm_scored=llm_scored,
        )
    
    def score_dataframe(
        self,
        df: pd.DataFrame,
        content_column: str = 'content',
        output_column: str = 'score',
        type_column: Optional[str] = None,
        domain_column: Optional[str] = None,
        method: Optional[ScoringMethod] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        对DataFrame中的文本列进行打分
        
        Args:
            df: 输入DataFrame
            content_column: 内容列名
            output_column: 输出分数列名
            type_column: 数据类型列名
            domain_column: 数据域列名（用于推断数据类型）
            method: 打分方法
            progress_callback: 进度回调
            **kwargs: 额外参数
            
        Returns:
            pd.DataFrame: 添加了分数列的DataFrame
        """
        df = df.copy()
        
        # 提取内容
        contents = df[content_column].fillna('').tolist()
        
        # 确定数据类型
        if type_column and type_column in df.columns:
            data_types = df[type_column].fillna('').tolist()
        elif domain_column and domain_column in df.columns:
            data_types = df[domain_column].fillna('').tolist()
        else:
            data_types = None
        
        # 批量打分
        batch_result = self.score_batch(
            contents=contents,
            data_types=data_types,
            method=method,
            progress_callback=progress_callback,
            **kwargs
        )
        
        # 添加结果列
        df[output_column] = [r.score if r.success else None for r in batch_result.results]
        df[f'{output_column}_normalized'] = [r.normalized_score for r in batch_result.results]
        df[f'{output_column}_level'] = [r.level.value if r.level else None for r in batch_result.results]
        df[f'{output_column}_reason'] = [r.reason for r in batch_result.results]
        df[f'{output_column}_method'] = [r.method.value for r in batch_result.results]
        df[f'{output_column}_success'] = [r.success for r in batch_result.results]
        
        self.logger.info(
            f"DataFrame打分完成 - 总数: {batch_result.total}, "
            f"成功: {batch_result.success_count}, "
            f"利好: {batch_result.bullish_count}, "
            f"利空: {batch_result.bearish_count}, "
            f"平均分: {batch_result.avg_score:.1f}"
        )
        
        return df
    
    def is_available(self) -> bool:
        """检查服务是否可用"""
        # 规则打分器始终可用
        return True
    
    def is_llm_available(self) -> bool:
        """检查LLM服务是否可用"""
        try:
            return self.llm_scorer.is_available()
        except Exception:
            return False


# ==================== 便捷函数 ====================


def score(
    content: str,
    data_type: str = "",
    method: Optional[ScoringMethod] = None,
    **kwargs
) -> ScoreResult:
    """
    便捷函数：打分
    
    Args:
        content: 输入文本
        data_type: 数据类型
        method: 打分方法
        **kwargs: 额外参数
        
    Returns:
        ScoreResult: 打分结果
    """
    scorer = Scorer()
    return scorer.score(content, data_type, method, **kwargs)


def score_batch(
    contents: List[str],
    data_types: Optional[List[str]] = None,
    method: Optional[ScoringMethod] = None,
    **kwargs
) -> BatchScoreResult:
    """
    便捷函数：批量打分
    """
    scorer = Scorer()
    return scorer.score_batch(contents, data_types, method, **kwargs)


def score_with_rule(content: str, data_type: str = "") -> ScoreResult:
    """便捷函数：仅使用规则打分"""
    scorer = RuleScorer()
    return scorer.score(content, data_type)


def score_with_llm(content: str, data_type: str = "") -> ScoreResult:
    """便捷函数：仅使用LLM打分"""
    scorer = LLMScorer()
    return scorer.score(content, data_type)
