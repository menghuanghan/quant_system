"""
统一公告采集接口

多数据源聚合采集器，提供标准化的公告采集API：
- 自动数据源切换
- 增量更新支持
- 批量历史数据采集
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd

from ..base import (
    UnstructuredCollector,
    AnnouncementCategory,
    DataSourceType,
    DateRangeIterator,
    CollectionProgress,
    generate_task_id,
    parse_date_range
)
from .akshare_collector import AKShareAnnouncementCollector
from .cninfo_crawler import CninfoAnnouncementCrawler

logger = logging.getLogger(__name__)


class AnnouncementCollector:
    """
    统一公告采集接口
    
    聚合多个数据源，提供标准化的公告采集API
    
    数据源优先级：
    1. AKShare - 免费，数据可能不完整
    2. Cninfo - 最完整，需要爬取
    
    使用示例:
        >>> collector = AnnouncementCollector()
        >>> df = collector.collect_announcements(
        ...     start_date='2024-01-01',
        ...     end_date='2024-12-31',
        ...     categories=['年报', '业绩预告']
        ... )
    """
    
    # 标准输出字段
    STANDARD_FIELDS = [
        'ts_code',          # 股票代码
        'name',             # 股票名称
        'title',            # 公告标题
        'ann_date',         # 公告日期
        'ann_time',         # 公告时间
        'category',         # 公告类型
        'url',              # 公告链接
        'source',           # 数据源
        'is_correction',    # 是否更正公告
        'correction_of',    # 原公告ID
        'list_status',      # 上市状态
        'original_id',      # 原始ID
    ]
    
    def __init__(
        self,
        source_priority: Optional[List[str]] = None,
        use_proxy: bool = False,
        enable_fallback: bool = True
    ):
        """
        Args:
            source_priority: 数据源优先级列表
            use_proxy: 是否为爬虫使用代理
            enable_fallback: 是否启用数据源降级
        """
        self.source_priority = source_priority or ['cninfo', 'akshare']
        self.use_proxy = use_proxy
        self.enable_fallback = enable_fallback
        
        # 初始化采集器
        self._collectors = {}
        self._init_collectors()
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _init_collectors(self):
        """初始化各数据源采集器"""
        
        try:
            self._collectors['akshare'] = AKShareAnnouncementCollector()
        except Exception as e:
            self.logger.warning(f"AKShare 初始化失败: {e}")
        
        try:
            self._collectors['cninfo'] = CninfoAnnouncementCrawler(
                use_proxy=self.use_proxy
            )
        except Exception as e:
            self.logger.warning(f"Cninfo 初始化失败: {e}")
    
    def collect_announcements(
        self,
        start_date: str,
        end_date: str,
        ts_codes: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        include_delisted: bool = True,
        preferred_source: Optional[str] = None
    ) -> pd.DataFrame:
        """
        采集公告数据（主接口）
        
        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            ts_codes: 股票代码列表（如 ['000001.SZ', '600000.SH']）
            categories: 公告类型列表（如 ['年报', '业绩预告']）
            include_delisted: 是否包含退市公司
            preferred_source: 指定使用的数据源
        
        Returns:
            标准化的公告数据DataFrame
            
        输出字段:
            - ts_code: 股票代码
            - name: 股票名称
            - title: 公告标题
            - ann_date: 公告日期
            - category: 公告类型
            - url: 公告链接
            - source: 数据源
            - is_correction: 是否更正公告
            - correction_of: 原公告ID
            - list_status: 上市状态
        
        Example:
            >>> df = collector.collect_announcements(
            ...     start_date='2024-01-01',
            ...     end_date='2024-12-31',
            ...     categories=['年报']
            ... )
        """
        # 确定数据源顺序
        if preferred_source:
            sources = [preferred_source]
            if self.enable_fallback:
                sources.extend([s for s in self.source_priority if s != preferred_source])
        else:
            sources = self.source_priority
        
        # 尝试各数据源
        for source in sources:
            if source not in self._collectors:
                continue
            
            collector = self._collectors[source]
            
            try:
                self.logger.info(f"使用 {source} 采集公告...")
                
                if source == 'akshare':
                    # 转换代码格式
                    symbols = None
                    if ts_codes:
                        symbols = [c.split('.')[0] for c in ts_codes]
                    df = collector.collect(
                        start_date=start_date,
                        end_date=end_date,
                        symbols=symbols
                    )
                elif source == 'cninfo':
                    # 转换代码格式
                    stock_codes = None
                    if ts_codes:
                        stock_codes = [c.split('.')[0] for c in ts_codes]
                    df = collector.collect(
                        start_date=start_date,
                        end_date=end_date,
                        stock_codes=stock_codes,
                        categories=categories,
                        include_delisted=include_delisted
                    )
                else:
                    continue
                
                if not df.empty:
                    # 按类别过滤（如果指定了类别且数据源未支持）
                    if categories and 'category' in df.columns:
                        df = df[df['category'].isin(categories)]
                    
                    self.logger.info(
                        f"从 {source} 成功采集 {len(df)} 条公告"
                    )
                    return self._ensure_columns(df)
                
            except Exception as e:
                self.logger.warning(f"{source} 采集失败: {e}")
                if not self.enable_fallback:
                    raise
                continue
        
        self.logger.warning("所有数据源均无法采集公告")
        return pd.DataFrame(columns=self.STANDARD_FIELDS)
    
    def collect_incremental(
        self,
        since: Optional[str] = None,
        days: int = 1,
        **kwargs
    ) -> pd.DataFrame:
        """
        增量采集（用于调度器定期更新）
        
        Args:
            since: 从指定日期开始（为空则使用days参数）
            days: 采集最近N天数据
            **kwargs: 传递给 collect_announcements 的其他参数
        
        Returns:
            新增公告DataFrame
        
        Example:
            >>> # 采集最近1天的公告
            >>> df = collector.collect_incremental(days=1)
            >>> # 采集从指定日期以来的公告
            >>> df = collector.collect_incremental(since='2024-01-15')
        """
        if since:
            start_date = since
        else:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        return self.collect_announcements(
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
    
    def collect_full_history(
        self,
        years: int = 5,
        categories: Optional[List[str]] = None,
        include_delisted: bool = True,
        chunk_months: int = 3,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        采集全市场历史公告
        
        Args:
            years: 采集最近N年数据
            categories: 公告类型列表
            include_delisted: 是否包含退市公司
            chunk_months: 分批采集的月份数
            progress_callback: 进度回调函数
        
        Returns:
            全量公告DataFrame
        
        Example:
            >>> df = collector.collect_full_history(
            ...     years=3,
            ...     categories=['年报', '中报']
            ... )
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        all_data = []
        total_chunks = (years * 12) // chunk_months + 1
        current_chunk = 0
        
        for chunk_start, chunk_end in DateRangeIterator(
            start_str, end_str, chunk_days=chunk_months * 30
        ):
            current_chunk += 1
            
            self.logger.info(
                f"采集历史公告 [{current_chunk}/{total_chunks}]: "
                f"{chunk_start} ~ {chunk_end}"
            )
            
            df = self.collect_announcements(
                start_date=chunk_start,
                end_date=chunk_end,
                categories=categories,
                include_delisted=include_delisted
            )
            
            if not df.empty:
                all_data.append(df)
            
            if progress_callback:
                progress_callback(current_chunk, total_chunks, len(df))
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(
            subset=['ts_code', 'title', 'ann_date'],
            keep='first'
        )
        
        self.logger.info(f"历史公告采集完成，共 {len(result)} 条")
        return result
    
    def get_correction_announcements(
        self,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        获取更正公告及其关联的原始公告
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他采集参数
        
        Returns:
            更正公告DataFrame（包含原公告关联）
        """
        df = self.collect_announcements(
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        
        if df.empty or 'is_correction' not in df.columns:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        # 筛选更正公告
        corrections = df[df['is_correction'] == True].copy()
        
        return corrections
    
    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保DataFrame包含所有标准列"""
        for col in self.STANDARD_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.STANDARD_FIELDS]
    
    def get_available_sources(self) -> List[str]:
        """获取可用的数据源列表"""
        return list(self._collectors.keys())


# ============= 便捷函数接口 =============

def get_announcements(
    start_date: str,
    end_date: str,
    ts_codes: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    include_delisted: bool = True
) -> pd.DataFrame:
    """
    获取上市公司公告数据（主入口）
    
    Args:
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
        ts_codes: 股票代码列表
        categories: 公告类型列表
        include_delisted: 是否包含退市公司
    
    Returns:
        公告数据DataFrame
    
    Example:
        >>> df = get_announcements(
        ...     start_date='2024-01-01',
        ...     end_date='2024-12-31',
        ...     ts_codes=['000001.SZ'],
        ...     categories=['年报', '业绩预告']
        ... )
    """
    collector = AnnouncementCollector()
    return collector.collect_announcements(
        start_date=start_date,
        end_date=end_date,
        ts_codes=ts_codes,
        categories=categories,
        include_delisted=include_delisted
    )


def get_announcement_by_date(ann_date: str) -> pd.DataFrame:
    """
    获取指定日期的全市场公告
    
    Args:
        ann_date: 公告日期（YYYY-MM-DD）
    
    Returns:
        公告数据DataFrame
    
    Example:
        >>> df = get_announcement_by_date('2024-01-15')
    """
    collector = AnnouncementCollector()
    return collector.collect_announcements(
        start_date=ann_date,
        end_date=ann_date
    )


def get_announcements_incremental(
    days: int = 1,
    since: Optional[str] = None
) -> pd.DataFrame:
    """
    增量获取公告（用于调度器）
    
    Args:
        days: 获取最近N天的公告
        since: 从指定日期开始
    
    Returns:
        公告数据DataFrame
    
    Example:
        >>> # 获取最近1天的公告
        >>> df = get_announcements_incremental(days=1)
        >>> # 获取从指定日期以来的公告
        >>> df = get_announcements_incremental(since='2024-01-15')
    """
    collector = AnnouncementCollector()
    return collector.collect_incremental(days=days, since=since)


def get_correction_announcements(
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    获取更正公告及其关联的原始公告
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        更正公告DataFrame
    
    Example:
        >>> df = get_correction_announcements('2024-01-01', '2024-12-31')
    """
    collector = AnnouncementCollector()
    return collector.get_correction_announcements(
        start_date=start_date,
        end_date=end_date
    )


def get_full_market_history(
    years: int = 5,
    categories: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    获取全市场历史公告
    
    Args:
        years: 采集最近N年
        categories: 公告类型列表
    
    Returns:
        全量公告DataFrame
    
    Example:
        >>> df = get_full_market_history(years=3, categories=['年报'])
    """
    collector = AnnouncementCollector()
    return collector.collect_full_history(
        years=years,
        categories=categories
    )
