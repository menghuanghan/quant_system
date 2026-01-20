"""
动态存储路由器 (Storage Router)

根据内容特征动态决定存储策略：
- 重要公告（并购重组、立案调查等）：保留 PDF + 文本
- 普通公告：仅保留文本
- 批量新闻：压缩存储

节省存储空间的同时，确保关键数据不丢失
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class StorageType(str, Enum):
    """存储类型"""
    TEXT_ONLY = "text_only"       # 仅存储文本
    BOTH = "both"                 # 存储文本 + 原始文件
    COMPRESSED = "compressed"    # 压缩存储
    SKIP = "skip"                 # 跳过（不存储）


@dataclass
class StorageDecision:
    """存储决策"""
    storage_type: StorageType
    reason: str                    # 决策原因
    priority: int                  # 优先级（用于冲突处理）
    save_pdf: bool = False         # 是否保存PDF
    compress: bool = False         # 是否压缩
    metadata_only: bool = False    # 是否仅保存元数据


class StorageRouter:
    """
    动态存储路由器
    
    根据标题、内容、来源等特征决定存储策略。
    
    规则优先级（从高到低）：
    1. 关键词规则（并购重组等必须保留原文）
    2. 来源规则（交易所公告比新闻重要）
    3. 时间规则（近期数据更重要）
    4. 默认规则
    
    Example:
        >>> router = StorageRouter()
        >>> 
        >>> decision = router.decide(
        ...     title="关于重大资产重组的公告",
        ...     category="重大事项",
        ...     source="cninfo"
        ... )
        >>> print(decision.storage_type)  # StorageType.BOTH
        >>> print(decision.save_pdf)  # True
    """
    
    # 必须保留原文的关键词
    CRITICAL_KEYWORDS = {
        # 重大事项
        "并购", "重组", "资产重组", "重大资产", "合并", "分立",
        "要约收购", "私有化", "借壳", "资产注入", "资产置换",
        
        # 监管相关
        "立案调查", "立案", "调查", "处罚", "警示函", "监管函",
        "问询函", "关注函", "责令改正", "行政处罚",
        
        # 风险事件
        "违规", "诉讼", "仲裁", "担保", "质押", "冻结",
        "暂停上市", "终止上市", "退市", "风险警示",
        
        # 重大变更
        "控制权变更", "实际控制人变更", "董事长辞职", "总经理辞职",
        "董事会换届", "股权转让", "股份转让",
        
        # 财务相关
        "业绩预告", "业绩快报", "业绩修正", "审计报告",
        "非标意见", "保留意见", "无法表示意见",
    }
    
    # 高优先级来源
    HIGH_PRIORITY_SOURCES = {
        "cninfo",      # 巨潮
        "sse",         # 上交所
        "szse",        # 深交所
        "csrc",        # 证监会
        "exchange",    # 交易所
    }
    
    # 低优先级来源（可以仅文本）
    LOW_PRIORITY_SOURCES = {
        "sina",        # 新浪
        "eastmoney",   # 东方财富
        "stcn",        # 证券时报
        "cctv",        # 央视
    }
    
    def __init__(
        self,
        custom_keywords: Optional[Set[str]] = None,
        default_type: StorageType = StorageType.TEXT_ONLY
    ):
        """
        Args:
            custom_keywords: 自定义关键词（会与默认关键词合并）
            default_type: 默认存储类型
        """
        self.critical_keywords = self.CRITICAL_KEYWORDS.copy()
        if custom_keywords:
            self.critical_keywords.update(custom_keywords)
        
        self.default_type = default_type
        
        # 自定义规则
        self._custom_rules: List[Callable] = []
    
    def add_rule(self, rule_func: Callable):
        """
        添加自定义规则
        
        规则函数签名: (title, content, category, source, **kwargs) -> Optional[StorageDecision]
        返回 None 表示该规则不适用
        """
        self._custom_rules.append(rule_func)
    
    def _check_keywords(self, text: str) -> Optional[str]:
        """检查文本中的关键词"""
        if not text:
            return None
        
        for keyword in self.critical_keywords:
            if keyword in text:
                return keyword
        
        return None
    
    def _apply_keyword_rule(
        self,
        title: Optional[str],
        content: Optional[str],
        category: Optional[str]
    ) -> Optional[StorageDecision]:
        """应用关键词规则"""
        # 检查标题
        matched = self._check_keywords(title)
        if matched:
            return StorageDecision(
                storage_type=StorageType.BOTH,
                reason=f"标题包含关键词: {matched}",
                priority=100,
                save_pdf=True
            )
        
        # 检查分类
        matched = self._check_keywords(category)
        if matched:
            return StorageDecision(
                storage_type=StorageType.BOTH,
                reason=f"分类包含关键词: {matched}",
                priority=90,
                save_pdf=True
            )
        
        # 检查内容（仅检查前500字符，避免性能问题）
        if content:
            matched = self._check_keywords(content[:500])
            if matched:
                return StorageDecision(
                    storage_type=StorageType.BOTH,
                    reason=f"内容包含关键词: {matched}",
                    priority=80,
                    save_pdf=True
                )
        
        return None
    
    def _apply_source_rule(self, source: Optional[str]) -> Optional[StorageDecision]:
        """应用来源规则"""
        if not source:
            return None
        
        source_lower = source.lower()
        
        if source_lower in self.HIGH_PRIORITY_SOURCES:
            return StorageDecision(
                storage_type=StorageType.BOTH,
                reason=f"高优先级来源: {source}",
                priority=50,
                save_pdf=True
            )
        
        if source_lower in self.LOW_PRIORITY_SOURCES:
            return StorageDecision(
                storage_type=StorageType.TEXT_ONLY,
                reason=f"普通来源: {source}",
                priority=10,
                save_pdf=False
            )
        
        return None
    
    def _apply_category_rule(self, category: Optional[str]) -> Optional[StorageDecision]:
        """应用分类规则"""
        if not category:
            return None
        
        # 定期报告（年报、半年报、季报）
        if re.search(r'(年度报告|年报|半年度报告|季度报告)', category):
            return StorageDecision(
                storage_type=StorageType.BOTH,
                reason=f"定期报告: {category}",
                priority=70,
                save_pdf=True
            )
        
        # 临时公告
        if '临时' in category or '临' in category:
            return StorageDecision(
                storage_type=StorageType.TEXT_ONLY,
                reason=f"临时公告: {category}",
                priority=20,
                save_pdf=False
            )
        
        return None
    
    def decide(
        self,
        title: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
        file_size_mb: Optional[float] = None,
        **kwargs
    ) -> StorageDecision:
        """
        决定存储策略
        
        Args:
            title: 标题
            content: 内容
            category: 分类
            source: 来源
            file_size_mb: 文件大小（MB）
            **kwargs: 其他属性
        
        Returns:
            StorageDecision 存储决策
        """
        decisions = []
        
        # 应用自定义规则
        for rule in self._custom_rules:
            try:
                result = rule(title, content, category, source, **kwargs)
                if result:
                    decisions.append(result)
            except Exception as e:
                logger.warning(f"自定义规则执行失败: {e}")
        
        # 应用关键词规则
        keyword_decision = self._apply_keyword_rule(title, content, category)
        if keyword_decision:
            decisions.append(keyword_decision)
        
        # 应用分类规则
        category_decision = self._apply_category_rule(category)
        if category_decision:
            decisions.append(category_decision)
        
        # 应用来源规则
        source_decision = self._apply_source_rule(source)
        if source_decision:
            decisions.append(source_decision)
        
        # 大文件特殊处理
        if file_size_mb and file_size_mb > 10:
            # 超过10MB的文件，仅在关键词匹配时保留
            if not keyword_decision:
                decisions.append(StorageDecision(
                    storage_type=StorageType.TEXT_ONLY,
                    reason=f"大文件仅保留文本: {file_size_mb:.1f}MB",
                    priority=60,
                    save_pdf=False
                ))
        
        # 选择最高优先级的决策
        if decisions:
            decisions.sort(key=lambda d: -d.priority)
            return decisions[0]
        
        # 默认决策
        return StorageDecision(
            storage_type=self.default_type,
            reason="默认规则",
            priority=0,
            save_pdf=self.default_type == StorageType.BOTH
        )
    
    def decide_batch(
        self,
        items: List[Dict],
        title_key: str = 'title',
        category_key: str = 'category',
        source_key: str = 'source'
    ) -> List[StorageDecision]:
        """
        批量决策
        
        Args:
            items: 待决策的项目列表
            title_key: 标题字段名
            category_key: 分类字段名
            source_key: 来源字段名
        
        Returns:
            决策列表
        """
        decisions = []
        for item in items:
            decision = self.decide(
                title=item.get(title_key),
                category=item.get(category_key),
                source=item.get(source_key)
            )
            decisions.append(decision)
        return decisions


class AnnouncementStorageRouter(StorageRouter):
    """
    公告专用存储路由器
    
    针对公告数据的存储策略优化
    """
    
    # 公告专用关键词
    ANNOUNCEMENT_KEYWORDS = {
        # IPO 相关
        "首次公开发行", "IPO", "招股说明书", "上市公告书",
        
        # 增发配股
        "增发", "配股", "可转债", "定向增发", "非公开发行",
        
        # 分红派息
        "分红", "派息", "送股", "转增", "利润分配",
        
        # 回购注销
        "回购", "注销", "股份回购",
        
        # 关联交易
        "关联交易", "关联方",
    }
    
    def __init__(self):
        super().__init__()
        self.critical_keywords.update(self.ANNOUNCEMENT_KEYWORDS)


class NewsStorageRouter(StorageRouter):
    """
    新闻专用存储路由器
    
    新闻数据通常只需要文本
    """
    
    def __init__(self):
        super().__init__(default_type=StorageType.TEXT_ONLY)
        
        # 新闻中的重要关键词（需要额外关注）
        self.critical_keywords = {
            "央行", "证监会", "银保监会", "国务院",
            "政策", "监管", "改革", "宏观调控",
            "利率", "降准", "降息", "加息",
        }


# ============== 全局实例 ==============

_global_router: Optional[StorageRouter] = None
_announcement_router: Optional[AnnouncementStorageRouter] = None
_news_router: Optional[NewsStorageRouter] = None


def get_storage_router() -> StorageRouter:
    """获取全局存储路由器"""
    global _global_router
    if _global_router is None:
        _global_router = StorageRouter()
    return _global_router


def get_announcement_router() -> AnnouncementStorageRouter:
    """获取公告存储路由器"""
    global _announcement_router
    if _announcement_router is None:
        _announcement_router = AnnouncementStorageRouter()
    return _announcement_router


def get_news_router() -> NewsStorageRouter:
    """获取新闻存储路由器"""
    global _news_router
    if _news_router is None:
        _news_router = NewsStorageRouter()
    return _news_router


def decide_storage(
    title: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    content_type: str = "announcement"
) -> StorageDecision:
    """
    便捷函数：决定存储策略
    
    Args:
        title: 标题
        category: 分类
        source: 来源
        content_type: 内容类型 ('announcement', 'news', 'report')
    """
    if content_type == "announcement":
        router = get_announcement_router()
    elif content_type == "news":
        router = get_news_router()
    else:
        router = get_storage_router()
    
    return router.decide(title=title, category=category, source=source)
