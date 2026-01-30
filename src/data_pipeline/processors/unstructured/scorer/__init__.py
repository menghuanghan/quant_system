"""
量化打分器模块 (Scorer)

将摘要/标题文本转换为情感分数，用于量化分析。

主要组件：
- Scorer: 主接口类，自动选择打分策略
- LLMScorer: 基于LLM的打分器（理解语义，输出分数+理由）
- RuleScorer: 基于规则的打分器（正则匹配，快速准确）

使用示例：
```python
from src.data_pipeline.processors.unstructured.scorer import Scorer

scorer = Scorer()

# 单条打分
result = scorer.score(
    content="公司净利润同比增长80%，超预期",
    data_type="announcement"
)
print(result.score)   # 75 (利好)
print(result.reason)  # "业绩大幅超预期，利好股价"

# 批量打分
results = scorer.score_batch(
    contents=["文本1", "文本2"],
    data_types=["announcement", "news"]
)
```
"""

from .base import (
    ScoreResult,
    ScorerConfig,
    ScoreLevel,
    ScoringMethod,
)
from .rule_scorer import RuleScorer
from .llm_scorer import LLMScorer
from .scorer import Scorer, BatchScoreResult, score, score_batch

__all__ = [
    # 主接口
    'Scorer',
    'BatchScoreResult',
    
    # 具体打分器
    'LLMScorer',
    'RuleScorer',
    
    # 数据结构
    'ScoreResult',
    'ScorerConfig',
    'ScoreLevel',
    'ScoringMethod',
    
    # 便捷函数
    'score',
    'score_batch',
]

__version__ = "1.0.0"
