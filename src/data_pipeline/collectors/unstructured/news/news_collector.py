"""
统一新闻采集接口

多数据源聚合采集器，提供标准化的新闻采集API
"""

import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd

from ..base import UnstructuredCollector, DateRangeIterator
from .cctv_collector import CCTVNewsCollector, NewsCategory
from .eastmoney_collector import EastMoneyNewsCollector
from .sina_crawler import SinaFinanceCrawler
from .stcn_crawler import STCNCrawler
from .exchange_news_crawler import ExchangeNewsCrawler

logger = logging.getLogger(__name__)


class NewsCollector:
    """
    统一新闻采集接口
    
    聚合多个数据源，提供标准化的新闻采集API
    
    数据源：
    1. CCTV新闻联播 - 权威政策信息
    2. 东方财富 - 个股新闻、市场快讯
    3. 新浪财经 - 财经要闻
    4. 证券时报 - 专业分析
    5. 交易所公告解读 - 公告分析
    
    使用示例:
        >>> collector = NewsCollector()
        >>> df = collector.collect_news(
        ...     start_date='2025-01-01',
        ...     end_date='2025-01-31'
        ... )
    """
    
    STANDARD_FIELDS = [
        'news_id',
        'title',
        'content',
        'summary',
        'pub_time',
        'pub_date',
        'source',
        'category',
        'url',
        'related_stocks',
        'keywords',
    ]
    
    # 数据源配置
    SOURCE_CONFIGS = {
        'cctv': {
            'name': '央视新闻联播',
            'collector': CCTVNewsCollector,
            'enabled': True,
        },
        'eastmoney': {
            'name': '东方财富',
            'collector': EastMoneyNewsCollector,
            'enabled': True,
        },
        'sina': {
            'name': '新浪财经',
            'collector': SinaFinanceCrawler,
            'enabled': True,
        },
        'stcn': {
            'name': '证券时报',
            'collector': STCNCrawler,
            'enabled': True,
        },
        'exchange': {
            'name': '交易所公告解读',
            'collector': ExchangeNewsCrawler,
            'enabled': True,
        },
    }
    
    def __init__(
        self,
        sources: Optional[List[str]] = None,
        enable_fallback: bool = True
    ):
        """
        Args:
            sources: 启用的数据源列表（为空则启用全部）
            enable_fallback: 是否启用错误降级
        """
        self.sources = sources or list(self.SOURCE_CONFIGS.keys())
        self.enable_fallback = enable_fallback
        self._collectors = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_collector(self, source: str):
        """获取或创建采集器实例"""
        if source not in self._collectors:
            config = self.SOURCE_CONFIGS.get(source)
            if config and config.get('enabled'):
                try:
                    self._collectors[source] = config['collector']()
                except Exception as e:
                    self.logger.warning(f"初始化 {source} 采集器失败: {e}")
                    return None
        return self._collectors.get(source)
    
    def collect_news(
        self,
        start_date: str,
        end_date: str,
        sources: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集新闻数据（主接口）
        
        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            sources: 数据源列表
            symbols: 股票代码列表（用于个股新闻）
            categories: 新闻类别过滤
        
        Returns:
            标准化的新闻数据DataFrame
        
        Example:
            >>> df = collector.collect_news(
            ...     start_date='2025-01-01',
            ...     end_date='2025-01-31',
            ...     sources=['cctv', 'eastmoney']
            ... )
        """
        all_data = []
        sources = sources or self.sources
        
        for source in sources:
            if source not in self.SOURCE_CONFIGS:
                continue
            
            collector = self._get_collector(source)
            if collector is None:
                continue
            
            try:
                self.logger.info(f"采集 {self.SOURCE_CONFIGS[source]['name']} 新闻...")
                
                # 根据数据源调用不同参数
                if source == 'eastmoney' and symbols:
                    df = collector.collect(
                        start_date=start_date,
                        end_date=end_date,
                        symbols=symbols
                    )
                else:
                    df = collector.collect(
                        start_date=start_date,
                        end_date=end_date
                    )
                
                if not df.empty:
                    self.logger.info(
                        f"  {self.SOURCE_CONFIGS[source]['name']}: {len(df)} 条"
                    )
                    all_data.append(df)
                    
            except Exception as e:
                self.logger.warning(f"采集 {source} 失败: {e}")
                if not self.enable_fallback:
                    raise
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        
        # 去重
        result = result.drop_duplicates(subset=['news_id'], keep='first')
        
        # 类别过滤
        if categories and 'category' in result.columns:
            result = result[result['category'].isin(categories)]
        
        return self._ensure_columns(result)
    
    def collect_cctv(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        仅采集央视新闻联播
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            新闻数据DataFrame
        """
        return self.collect_news(
            start_date=start_date,
            end_date=end_date,
            sources=['cctv']
        )
    
    def collect_stock_news(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        采集个股新闻
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
        
        Returns:
            新闻数据DataFrame
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        return self.collect_news(
            start_date=start_date,
            end_date=end_date,
            sources=['eastmoney'],
            symbols=symbols
        )
    
    def collect_incremental(
        self,
        days: int = 1,
        sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        增量采集（用于调度器）
        
        Args:
            days: 采集最近N天
            sources: 数据源列表
        
        Returns:
            新闻数据DataFrame
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        return self.collect_news(
            start_date=start_date,
            end_date=end_date,
            sources=sources
        )
    
    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保所有标准字段存在"""
        for col in self.STANDARD_FIELDS:
            if col not in df.columns:
                df[col] = ''
        
        return df[self.STANDARD_FIELDS]
    
    def get_available_sources(self) -> List[str]:
        """获取可用的数据源列表"""
        return [s for s, c in self.SOURCE_CONFIGS.items() if c.get('enabled')]


# ============= 便捷函数接口 =============

def get_news(
    start_date: str,
    end_date: str,
    sources: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    获取新闻数据（主入口）
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        sources: 数据源列表
    
    Returns:
        新闻数据DataFrame
    """
    collector = NewsCollector()
    return collector.collect_news(
        start_date=start_date,
        end_date=end_date,
        sources=sources
    )


def get_news_by_date(news_date: str) -> pd.DataFrame:
    """
    获取指定日期的新闻
    
    Args:
        news_date: 日期
    
    Returns:
        新闻数据DataFrame
    """
    return get_news(news_date, news_date)


def get_news_incremental(days: int = 1) -> pd.DataFrame:
    """
    增量获取新闻（调度器接口）
    
    Args:
        days: 天数
    
    Returns:
        新闻数据DataFrame
    """
    collector = NewsCollector()
    return collector.collect_incremental(days=days)


def get_stock_related_news(symbols: List[str]) -> pd.DataFrame:
    """
    获取个股相关新闻
    
    Args:
        symbols: 股票代码列表
    
    Returns:
        新闻数据DataFrame
    """
    collector = NewsCollector()
    return collector.collect_stock_news(symbols=symbols)
