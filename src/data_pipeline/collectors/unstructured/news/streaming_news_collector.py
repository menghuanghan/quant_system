"""
流式新闻采集器 (Streaming News Collector)

基于 StreamingCollector 实现的新闻采集器，支持：
- 即时清洗（Extract-on-the-fly）：HTML 下载后立即提取正文
- 瘦身存储：丢弃原始 HTML，仅保留清洗后的文本
- 时间清洗：防止未来函数泄露
- 批量落盘：Parquet 压缩存储

用法：
    >>> with StreamingNewsCollector() as collector:
    ...     stats = collector.collect('2024-01-01', '2024-01-31')
    ...     print(stats)
"""

import logging
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime, timedelta

from ..base import StreamingCollector, ContentType
from ..scraper_base import ScraperBase, get_scraper

# 尝试导入 HTML 解析器
try:
    from ....clean.unstructured.html_parser import HTMLParser, HTMLCleanConfig
    HAS_HTML_PARSER = True
except ImportError:
    HAS_HTML_PARSER = False
    HTMLParser = None
    HTMLCleanConfig = None

# 导入现有新闻采集器（用于元数据获取）
from .cctv_collector import CCTVNewsCollector, NewsCategory
from .eastmoney_collector import EastMoneyNewsCollector
from .sina_crawler import SinaFinanceCrawler
from .stcn_crawler import STCNCrawler
from .exchange_news_crawler import ExchangeNewsCrawler

logger = logging.getLogger(__name__)


class StreamingNewsCollector(StreamingCollector):
    """
    流式新闻采集器
    
    聚合多个新闻源，采用流式处理：
    1. 获取新闻元数据（标题、URL、日期等）
    2. 下载 HTML 到内存
    3. 即时清洗：HTMLParser.extract_article_content()
    4. 瘦身存储：仅保留 clean_text、title、url、publish_time
    
    核心设计原则：
    - 严格丢弃原始 HTML
    - 时间精确到秒（防泄露）
    - 版本控制（同一 URL 可多次采集）
    """
    
    # 数据源配置
    SOURCE_NAME = "news"
    DOMAIN = "unstructured"
    SUB_DOMAIN = "news"
    
    # 支持的数据源
    SUPPORTED_SOURCES = {
        'cctv': CCTVNewsCollector,
        'eastmoney': EastMoneyNewsCollector,
        'sina': SinaFinanceCrawler,
        'stcn': STCNCrawler,
        'exchange': ExchangeNewsCrawler,
    }
    
    def __init__(
        self,
        sources: Optional[List[str]] = None,
        extract_content: bool = True,
        use_proxy: bool = False,
        **kwargs
    ):
        """
        初始化流式新闻采集器
        
        Args:
            sources: 启用的数据源列表（None 启用全部）
            extract_content: 是否下载并提取正文（False 则仅元数据）
            use_proxy: 是否使用代理
            **kwargs: StreamingCollector 参数
        """
        super().__init__(**kwargs)
        
        self.sources = sources or list(self.SUPPORTED_SOURCES.keys())
        self.extract_content = extract_content
        self.use_proxy = use_proxy
        
        # 元数据采集器实例缓存
        self._collectors: Dict[str, Any] = {}
        
        # 爬虫工具
        self._scraper = ScraperBase(
            use_proxy=use_proxy,
            rate_limit=True,
            timeout=30
        )
        
        # HTML 解析器
        if HAS_HTML_PARSER:
            self._html_parser = HTMLParser()
        else:
            self._html_parser = None
            self.logger.warning(
                "HTMLParser 未导入，将仅采集元数据（无正文提取）"
            )
    
    def _get_collector(self, source: str):
        """获取或创建元数据采集器"""
        if source not in self._collectors:
            collector_class = self.SUPPORTED_SOURCES.get(source)
            if collector_class:
                try:
                    self._collectors[source] = collector_class()
                except Exception as e:
                    self.logger.warning(f"初始化 {source} 采集器失败: {e}")
                    return None
        return self._collectors.get(source)
    
    def _collect_items(
        self,
        start_date: str,
        end_date: str,
        sources: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        采集新闻数据（生成器）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            sources: 数据源列表
            symbols: 股票代码列表（用于个股新闻）
        
        Yields:
            处理后的数据字典
        """
        sources = sources or self.sources
        
        for source in sources:
            if source not in self.SUPPORTED_SOURCES:
                continue
            
            collector = self._get_collector(source)
            if collector is None:
                continue
            
            self.logger.info(f"采集 {source} 新闻...")
            
            try:
                # 获取元数据
                if source == 'eastmoney' and symbols:
                    metadata_df = collector.collect(
                        start_date=start_date,
                        end_date=end_date,
                        symbols=symbols
                    )
                else:
                    metadata_df = collector.collect(
                        start_date=start_date,
                        end_date=end_date
                    )
                
                if metadata_df.empty:
                    self.logger.debug(f"{source} 无数据")
                    continue
                
                self.logger.info(f"  {source}: {len(metadata_df)} 条元数据")
                
                # 逐条处理
                for _, row in metadata_df.iterrows():
                    item = self._process_single_news(row, source)
                    if item is not None:
                        yield item
                        
            except Exception as e:
                self.logger.warning(f"采集 {source} 失败: {e}")
                continue
    
    def _process_single_news(
        self,
        row: Dict[str, Any],
        source: str
    ) -> Optional[Dict[str, Any]]:
        """
        处理单条新闻
        
        流程：
        1. 提取元数据
        2. 如果有 URL 且启用提取 → 下载 HTML → 清洗正文
        3. 严格丢弃原始 HTML
        
        Args:
            row: 元数据行
            source: 数据源名称
        
        Returns:
            处理后的数据字典
        """
        # 提取基础字段
        title = row.get('title', '')
        url = row.get('url', '')
        content = row.get('content', '')  # 部分数据源直接提供内容
        pub_time = row.get('pub_time', '') or row.get('pub_date', '')
        
        # 基础数据
        base_item = {
            'source': source,
            'title': title,
            'publish_time': pub_time,  # 原始时间，将被 StreamingCollector 清洗
            'url': url,
            'content_type': 'html',
            'original_id': row.get('news_id', ''),
            'category': row.get('category', ''),
            'related_stocks': row.get('related_stocks', ''),
            'keywords': row.get('keywords', ''),
        }
        
        # 如果数据源直接提供了内容（如 CCTV）
        if content:
            base_item['content'] = self._clean_content(content)
            return base_item
        
        # 无 URL 或不需要提取内容
        if not url or not self.extract_content or self._html_parser is None:
            return base_item
        
        # 下载 HTML 到内存
        try:
            html_bytes = self._scraper.get_bytes(
                url,
                max_size_mb=5.0,  # 新闻页面通常较小
                timeout=30
            )
        except Exception as e:
            self.logger.debug(f"HTML 下载失败: {url} - {e}")
            return base_item
        
        if html_bytes is None:
            return base_item
        
        # 解码 HTML
        try:
            # 尝试常见编码
            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                try:
                    html_text = html_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                html_text = html_bytes.decode('utf-8', errors='ignore')
        except Exception as e:
            self.logger.debug(f"HTML 解码失败: {url} - {e}")
            return base_item
        
        # 提取正文
        try:
            clean_text = self._html_parser.extract_article_content(html_text)
            if clean_text:
                base_item['content'] = clean_text
        except Exception as e:
            self.logger.debug(f"HTML 解析失败: {url} - {e}")
        
        # 严格丢弃原始 HTML（内存中已释放）
        # html_text = None  # 显式释放
        
        return base_item
    
    def _clean_content(self, content: str) -> str:
        """
        清洗内容文本
        
        即使数据源直接提供内容，也可能包含 HTML 标签或多余空白
        """
        if not content:
            return ''
        
        # 简单清洗
        import re
        # 移除 HTML 标签
        content = re.sub(r'<[^>]+>', '', content)
        # 规范化空白
        content = re.sub(r'\s+', ' ', content)
        # 去除首尾空白
        content = content.strip()
        
        return content
    
    def close(self):
        """关闭资源"""
        super().close()
        if self._scraper:
            self._scraper.close()


# ============== 便捷函数 ==============

def collect_news_streaming(
    start_date: str,
    end_date: str,
    sources: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    extract_content: bool = True,
    use_proxy: bool = False,
    buffer_size: int = 500,
    time_mode: str = 'conservative'
) -> Dict[str, Any]:
    """
    流式采集新闻数据（推荐接口）
    
    采用即时清洗策略：
    - HTML 下载后立即提取正文
    - 严格丢弃原始 HTML
    - 时间清洗防止未来函数
    - 批量落盘到 Parquet
    
    Args:
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
        sources: 数据源列表（cctv/eastmoney/sina/stcn/exchange）
        symbols: 股票代码列表（用于个股新闻）
        extract_content: 是否提取正文
        use_proxy: 是否使用代理
        buffer_size: 缓冲区大小
        time_mode: 时间填充模式
    
    Returns:
        采集统计信息
    
    Example:
        >>> stats = collect_news_streaming(
        ...     '2024-01-01', '2024-01-31',
        ...     sources=['cctv', 'eastmoney']
        ... )
        >>> print(f"采集 {stats['cleaned_items']} 条")
    """
    with StreamingNewsCollector(
        sources=sources,
        extract_content=extract_content,
        use_proxy=use_proxy,
        buffer_size=buffer_size,
        time_mode=time_mode
    ) as collector:
        return collector.collect(
            start_date=start_date,
            end_date=end_date,
            sources=sources,
            symbols=symbols
        )


def collect_stock_news_streaming(
    symbols: List[str],
    days: int = 30,
    extract_content: bool = True,
    time_mode: str = 'conservative'
) -> Dict[str, Any]:
    """
    流式采集个股新闻
    
    Args:
        symbols: 股票代码列表
        days: 采集最近 N 天
        extract_content: 是否提取正文
        time_mode: 时间填充模式
    
    Returns:
        采集统计信息
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    return collect_news_streaming(
        start_date=start_date,
        end_date=end_date,
        sources=['eastmoney'],  # 个股新闻主要来源
        symbols=symbols,
        extract_content=extract_content,
        time_mode=time_mode
    )
