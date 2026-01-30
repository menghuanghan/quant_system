"""
基于规则的打分器

使用正则表达式匹配关键词进行快速打分。
特别适用于交易所公告等格式化程度高的文本。

优点：
- 速度快（毫秒级）
- 准确率高（针对特定模式）
- 无需LLM调用
- 支持GPU加速批量处理
"""

import re
import logging
import time
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass

from .base import (
    ScoreResult,
    ScorerConfig,
    ScoringMethod,
    BaseScorer,
)

logger = logging.getLogger(__name__)


@dataclass
class ScoringRule:
    """打分规则"""
    pattern: str            # 正则表达式模式
    score: int              # 匹配后的分数
    category: str           # 规则类别
    description: str        # 规则描述
    confidence: float = 1.0 # 置信度
    priority: int = 0       # 优先级（高优先级先匹配）
    
    _compiled: re.Pattern = None
    
    def __post_init__(self):
        """编译正则表达式"""
        try:
            self._compiled = re.compile(self.pattern, re.IGNORECASE)
        except re.error as e:
            logger.error(f"正则表达式编译失败: {self.pattern}, 错误: {e}")
            self._compiled = None
    
    def match(self, text: str) -> Optional[re.Match]:
        """匹配文本"""
        if self._compiled is None:
            return None
        return self._compiled.search(text)


class RuleScorer(BaseScorer):
    """
    基于规则的打分器
    
    使用预定义的正则规则对文本进行快速打分。
    主要用于交易所公告、监管信息等格式化文本。
    
    使用示例：
    ```python
    scorer = RuleScorer()
    
    # 打分
    result = scorer.score("关于对XXX公司采取出具警示函监管措施的决定")
    print(result.score)  # -50
    print(result.matched_rule)  # "警示函"
    
    # 批量打分
    results = scorer.score_batch(["标题1", "标题2"])
    ```
    """
    
    # ==================== 利空规则 ====================
    
    # 极度利空 (-100 到 -75)
    EXTREMELY_BEARISH_RULES = [
        ScoringRule(
            pattern=r"(立案|调查|侦查)",
            score=-90,
            category="监管处罚",
            description="被立案调查",
            priority=100
        ),
        ScoringRule(
            pattern=r"退市|终止上市|摘牌",
            score=-100,
            category="退市风险",
            description="退市相关",
            priority=100
        ),
        ScoringRule(
            pattern=r"(ST|\\*ST|退市风险警示)",
            score=-85,
            category="风险警示",
            description="ST或退市风险警示",
            priority=95
        ),
        ScoringRule(
            pattern=r"(财务造假|虚假陈述|信息披露违法)",
            score=-95,
            category="违规违法",
            description="财务造假或虚假陈述",
            priority=100
        ),
        ScoringRule(
            pattern=r"(暂停上市|停止交易)",
            score=-80,
            category="交易暂停",
            description="暂停上市或停止交易",
            priority=90
        ),
        ScoringRule(
            pattern=r"(被认定为不适当人选|市场禁入)",
            score=-80,
            category="人员处罚",
            description="人员被认定不适当或市场禁入",
            priority=85
        ),
    ]
    
    # 利空 (-75 到 -25)
    BEARISH_RULES = [
        ScoringRule(
            pattern=r"(监管函|关注函|问询函)",
            score=-50,
            category="监管关注",
            description="收到监管函/关注函",
            priority=70
        ),
        ScoringRule(
            pattern=r"(警示函|责令改正)",
            score=-55,
            category="监管处罚",
            description="收到警示函或责令改正",
            priority=75
        ),
        ScoringRule(
            pattern=r"(通报批评|公开谴责)",
            score=-60,
            category="监管处罚",
            description="通报批评或公开谴责",
            priority=75
        ),
        ScoringRule(
            pattern=r"(行政处罚|罚款|没收)",
            score=-65,
            category="行政处罚",
            description="行政处罚相关",
            priority=80
        ),
        ScoringRule(
            pattern=r"(诉讼|仲裁|起诉|被诉)",
            score=-40,
            category="法律纠纷",
            description="诉讼仲裁相关",
            priority=60
        ),
        ScoringRule(
            pattern=r"(业绩预亏|预计亏损|首次亏损)",
            score=-60,
            category="业绩风险",
            description="业绩预亏",
            priority=70
        ),
        ScoringRule(
            pattern=r"(业绩下滑|业绩下降|净利润下降|净利润下滑)",
            score=-40,
            category="业绩风险",
            description="业绩下滑",
            priority=60
        ),
        ScoringRule(
            pattern=r"(减持|股东减持|大股东减持|高管减持)",
            score=-35,
            category="股东减持",
            description="股东/高管减持",
            priority=55
        ),
        ScoringRule(
            pattern=r"(质押|股权质押|高比例质押)",
            score=-30,
            category="质押风险",
            description="股权质押",
            priority=50
        ),
        ScoringRule(
            pattern=r"(违规|违反规定|违反法规)",
            score=-45,
            category="违规",
            description="违规行为",
            priority=65
        ),
        ScoringRule(
            pattern=r"(风险提示|重大风险)",
            score=-25,
            category="风险提示",
            description="风险提示",
            priority=45
        ),
        ScoringRule(
            pattern=r"(解聘|免职|辞职).*(董事|总经理|高管|财务总监)",
            score=-35,
            category="人事变动",
            description="高管离职/解聘",
            priority=55
        ),
        ScoringRule(
            pattern=r"(延期披露|无法按时|信披违规)",
            score=-40,
            category="信披问题",
            description="信息披露问题",
            priority=60
        ),
        ScoringRule(
            pattern=r"(监管措施|行政监管|纪律处分)",
            score=-55,
            category="监管处罚",
            description="受到监管措施",
            priority=72
        ),
    ]
    
    # 轻微利空 (-25 到 -5)
    SLIGHTLY_BEARISH_RULES = [
        ScoringRule(
            pattern=r"(终止|取消).*(重组|并购|收购|合作)",
            score=-20,
            category="事项终止",
            description="重大事项终止",
            priority=40
        ),
        ScoringRule(
            pattern=r"(审计意见|保留意见|否定意见)",
            score=-15,
            category="审计问题",
            description="审计意见问题",
            priority=35
        ),
        ScoringRule(
            pattern=r"(会计差错|差错更正|更正公告)",
            score=-12,
            category="信披问题",
            description="会计差错更正",
            priority=32
        ),
        ScoringRule(
            pattern=r"(辞职|离任)",
            score=-10,
            category="人事变动",
            description="高管辞职",
            priority=28
        ),
    ]
    
    # ==================== 利好规则 ====================
    
    # 极度利好 (75-100)
    EXTREMELY_BULLISH_RULES = [
        ScoringRule(
            pattern=r"(业绩预增|净利润预增|业绩大幅增长).*(50%|60%|70%|80%|90%|100%|翻番|翻倍)",
            score=85,
            category="业绩利好",
            description="业绩大幅预增",
            priority=90
        ),
        ScoringRule(
            pattern=r"(中标|签约|签订).*(重大|重要|大额|战略)",
            score=75,
            category="合同订单",
            description="重大合同中标",
            priority=85
        ),
        ScoringRule(
            pattern=r"(摘帽|撤销ST|撤销退市风险)",
            score=80,
            category="风险解除",
            description="摘帽或撤销风险警示",
            priority=90
        ),
        ScoringRule(
            pattern=r"(重组|并购|收购).*(完成|成功|获批)",
            score=75,
            category="重组并购",
            description="重大重组并购完成",
            priority=85
        ),
    ]
    
    # 利好 (25-75)
    BULLISH_RULES = [
        ScoringRule(
            pattern=r"(业绩预增|净利润增长|营收增长)",
            score=50,
            category="业绩利好",
            description="业绩增长",
            priority=70
        ),
        ScoringRule(
            pattern=r"(股东增持|大股东增持|高管增持|回购)",
            score=45,
            category="增持回购",
            description="股东增持或回购",
            priority=65
        ),
        ScoringRule(
            pattern=r"(获得|取得).*(资质|认证|许可|批准|批件)",
            score=40,
            category="资质获取",
            description="获得资质认证",
            priority=60
        ),
        ScoringRule(
            pattern=r"(中标|签约|签订|合同|订单)",
            score=35,
            category="合同订单",
            description="合同订单",
            priority=55
        ),
        ScoringRule(
            pattern=r"(分红|派息|现金分红|送股|转增)",
            score=30,
            category="股东回报",
            description="分红送股",
            priority=50
        ),
        ScoringRule(
            pattern=r"(新产品|新技术|专利|研发成功)",
            score=35,
            category="研发创新",
            description="研发创新成果",
            priority=55
        ),
        ScoringRule(
            pattern=r"(产能扩张|扩产|投产|新建)",
            score=30,
            category="产能扩张",
            description="产能扩张投产",
            priority=50
        ),
        ScoringRule(
            pattern=r"(战略合作|深度合作|强强联合)",
            score=35,
            category="战略合作",
            description="战略合作",
            priority=55
        ),
        ScoringRule(
            pattern=r"(核准|批复|获批|通过审核|审核通过).*(重组|定增|发行|上市)",
            score=70,
            category="审批通过",
            description="重大事项获批",
            priority=82
        ),
        ScoringRule(
            pattern=r"(重组|定增|发行|上市).*(核准|批复|获批|通过)",
            score=70,
            category="审批通过",
            description="重大事项获批（后置）",
            priority=80
        ),
    ]
    
    # 轻微利好 (5-25)
    SLIGHTLY_BULLISH_RULES = [
        ScoringRule(
            pattern=r"(正常经营|运营稳定|经营稳健)",
            score=10,
            category="经营稳定",
            description="经营稳定",
            priority=30
        ),
        ScoringRule(
            pattern=r"(股权激励|员工持股)",
            score=20,
            category="激励计划",
            description="股权激励",
            priority=40
        ),
        ScoringRule(
            pattern=r"(政府补助|财政补贴|税收返还)",
            score=15,
            category="政府补助",
            description="政府补助相关",
            priority=38
        ),
    ]
    
    # ==================== 中性规则 ====================
    
    NEUTRAL_RULES = [
        ScoringRule(
            pattern=r"(董事会决议|股东大会|临时公告)",
            score=0,
            category="例行公告",
            description="例行公告",
            priority=20
        ),
        ScoringRule(
            pattern=r"(章程修订|制度修订)",
            score=0,
            category="制度变更",
            description="制度修订",
            priority=20
        ),
        ScoringRule(
            pattern=r"(信息披露|定期报告|年报|半年报|季报|季度报告)",
            score=0,
            category="定期披露",
            description="定期报告披露",
            priority=15
        ),
    ]
    
    def __init__(self, config: Optional[ScorerConfig] = None):
        """初始化规则打分器"""
        super().__init__(config)
        
        # 按优先级排序所有规则
        self._all_rules = self._build_rules_list()
        self._all_rules.sort(key=lambda r: r.priority, reverse=True)
        
        # 检查cuDF可用性
        self._cudf_available = self._check_cudf()
        
        self.logger.info(
            f"规则打分器初始化完成 - 规则数量: {len(self._all_rules)}, "
            f"GPU加速: {'可用' if self._cudf_available else '不可用'}"
        )
    
    def _build_rules_list(self) -> List[ScoringRule]:
        """构建完整规则列表"""
        return (
            self.EXTREMELY_BEARISH_RULES +
            self.BEARISH_RULES +
            self.SLIGHTLY_BEARISH_RULES +
            self.NEUTRAL_RULES +
            self.SLIGHTLY_BULLISH_RULES +
            self.BULLISH_RULES +
            self.EXTREMELY_BULLISH_RULES
        )
    
    def _check_cudf(self) -> bool:
        """检查cuDF是否可用"""
        try:
            import cudf
            return True
        except ImportError:
            return False
    
    def score(
        self,
        content: str,
        data_type: str = "",
        **kwargs
    ) -> ScoreResult:
        """
        对文本进行规则打分
        
        Args:
            content: 输入文本（标题或摘要）
            data_type: 数据类型
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
        
        # 遍历规则进行匹配
        for rule in self._all_rules:
            match = rule.match(content)
            if match:
                elapsed_ms = (time.time() - start_time) * 1000
                
                return ScoreResult(
                    success=True,
                    score=rule.score,
                    reason=f"{rule.description}: {match.group()}",
                    content=content,
                    data_type=data_type,
                    method=ScoringMethod.RULE,
                    matched_rule=rule.pattern,
                    rule_confidence=rule.confidence,
                    process_time_ms=elapsed_ms,
                    metadata={
                        'rule_category': rule.category,
                        'matched_text': match.group(),
                    }
                )
        
        # 没有匹配的规则，返回中性
        elapsed_ms = (time.time() - start_time) * 1000
        return ScoreResult(
            success=True,
            score=0,
            reason="无匹配规则，默认中性",
            content=content,
            data_type=data_type,
            method=ScoringMethod.RULE,
            matched_rule=None,
            rule_confidence=0.5,
            process_time_ms=elapsed_ms,
        )
    
    def score_batch(
        self,
        contents: List[str],
        data_types: Optional[List[str]] = None,
        use_gpu: bool = True,
        **kwargs
    ) -> List[ScoreResult]:
        """
        批量规则打分
        
        Args:
            contents: 文本列表
            data_types: 数据类型列表
            use_gpu: 是否使用GPU加速
            **kwargs: 额外参数
            
        Returns:
            List[ScoreResult]: 打分结果列表
        """
        if not contents:
            return []
        
        # 准备数据类型
        if data_types is None:
            data_types = [""] * len(contents)
        elif len(data_types) != len(contents):
            data_types = data_types + [""] * (len(contents) - len(data_types))
        
        # GPU加速批量处理
        if use_gpu and self._cudf_available and len(contents) >= 100:
            return self._score_batch_gpu(contents, data_types)
        
        # CPU串行处理
        return [
            self.score(content, dtype)
            for content, dtype in zip(contents, data_types)
        ]
    
    def _score_batch_gpu(
        self,
        contents: List[str],
        data_types: List[str]
    ) -> List[ScoreResult]:
        """GPU加速的批量打分"""
        import cudf
        start_time = time.time()
        
        try:
            # 转换为cuDF Series
            gs = cudf.Series(contents)
            
            # 初始化结果
            scores = [0] * len(contents)
            reasons = ["无匹配规则，默认中性"] * len(contents)
            matched_rules = [None] * len(contents)
            confidences = [0.5] * len(contents)
            
            # 按优先级处理规则
            matched_mask = cudf.Series([False] * len(contents))
            
            for rule in self._all_rules:
                if rule._compiled is None:
                    continue
                
                # 使用cuDF的字符串匹配
                try:
                    matches = gs.str.contains(rule.pattern, regex=True, flags=re.IGNORECASE)
                    # 只更新尚未匹配的
                    new_matches = matches & ~matched_mask
                    
                    if new_matches.any():
                        indices = new_matches[new_matches].index.to_pandas().tolist()
                        for idx in indices:
                            scores[idx] = rule.score
                            reasons[idx] = rule.description
                            matched_rules[idx] = rule.pattern
                            confidences[idx] = rule.confidence
                        
                        matched_mask = matched_mask | new_matches
                        
                except Exception as e:
                    self.logger.warning(f"GPU规则匹配失败: {rule.pattern}, 错误: {e}")
                    continue
            
            elapsed_ms = (time.time() - start_time) * 1000
            avg_time = elapsed_ms / len(contents)
            
            # 构建结果
            results = []
            for i, (content, dtype, score, reason, rule, conf) in enumerate(
                zip(contents, data_types, scores, reasons, matched_rules, confidences)
            ):
                results.append(ScoreResult(
                    success=True,
                    score=score,
                    reason=reason,
                    content=content,
                    data_type=dtype,
                    method=ScoringMethod.RULE,
                    matched_rule=rule,
                    rule_confidence=conf,
                    process_time_ms=avg_time,
                ))
            
            self.logger.info(
                f"GPU批量规则打分完成 - 数量: {len(contents)}, "
                f"总耗时: {elapsed_ms:.0f}ms, 平均: {avg_time:.2f}ms/条"
            )
            
            return results
            
        except Exception as e:
            self.logger.warning(f"GPU批量打分失败，回退到CPU: {e}")
            return [
                self.score(content, dtype)
                for content, dtype in zip(contents, data_types)
            ]
    
    def add_rule(self, rule: ScoringRule) -> None:
        """添加自定义规则"""
        self._all_rules.append(rule)
        self._all_rules.sort(key=lambda r: r.priority, reverse=True)
    
    def get_rules_by_category(self, category: str) -> List[ScoringRule]:
        """获取指定类别的规则"""
        return [r for r in self._all_rules if r.category == category]
    
    def get_all_categories(self) -> List[str]:
        """获取所有规则类别"""
        return list(set(r.category for r in self._all_rules))


# 预定义的交易所公告专用规则集
EXCHANGE_NEWS_RULES = RuleScorer.EXTREMELY_BEARISH_RULES + RuleScorer.BEARISH_RULES
