"""
东方财富新闻采集器

多接口采集东方财富财经新闻：
- AKShare stock_news_em: 个股新闻
- 自建爬虫: 财经要闻频道
"""

import logging
import hashlib
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd

from ..base import UnstructuredCollector, DataSourceType, DateRangeIterator
from ..request_utils import safe_request, RequestDisguiser
from .cctv_collector import NewsCategory

logger = logging.getLogger(__name__)


class EastMoneyNewsCollector(UnstructuredCollector):
    """
    东方财富新闻采集器
    
    数据来源：
    1. AKShare stock_news_em - 个股新闻
    2. 东方财富财经频道 - 财经要闻
    """
    
    # 东方财富新闻API
    NEWS_API = "https://newsapi.eastmoney.com/kuaixun/v1/getlist_102_ajaxResult_50_{page}_.html"
    STOCK_NEWS_API = "https://searchapi.eastmoney.com/api/suggest/get"
    
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
    
    def __init__(self):
        super().__init__()
        self._ak = None
        self._disguiser = RequestDisguiser()
    
    @property
    def ak(self):
        """懒加载AKShare"""
        if self._ak is None:
            import akshare as ak
            self._ak = ak
        return self._ak
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        include_market_news: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集东方财富新闻
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            symbols: 股票代码列表（用于个股新闻）
            include_market_news: 是否包含市场要闻
        
        Returns:
            新闻数据DataFrame
        """
        all_data = []
        
        # 采集个股新闻
        if symbols:
            for symbol in symbols:
                df = self._collect_stock_news(symbol)
                if not df.empty:
                    df = self._filter_by_date(df, start_date, end_date)
                    if not df.empty:
                        all_data.append(df)
        
        # 采集市场要闻
        if include_market_news:
            df = self._collect_market_news()
            if not df.empty:
                df = self._filter_by_date(df, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['news_id'], keep='first')
        
        return self._standardize_output(result)
    
    def _collect_stock_news(self, symbol: str) -> pd.DataFrame:
        """
        采集个股新闻
        
        Args:
            symbol: 股票代码（纯数字）
        
        Returns:
            新闻DataFrame
        """
        try:
            df = self.ak.stock_news_em(symbol=symbol)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 映射字段
            result = pd.DataFrame()
            result['title'] = df.get('新闻标题', df.get('title', ''))
            result['content'] = df.get('新闻内容', df.get('content', ''))
            result['pub_time'] = df.get('发布时间', df.get('pub_time', ''))
            result['url'] = df.get('新闻链接', df.get('url', ''))
            result['source_name'] = df.get('文章来源', 'eastmoney')
            
            # 生成其他字段
            result['news_id'] = result['title'].apply(
                lambda x: self._generate_id(x, 'eastmoney')
            )
            result['summary'] = result['content'].apply(
                lambda x: str(x)[:200] + '...' if x and len(str(x)) > 200 else str(x)
            )
            result['source'] = 'eastmoney'
            result['category'] = NewsCategory.STOCK
            result['related_stocks'] = symbol
            result['keywords'] = ''
            
            # 提取日期
            result['pub_date'] = result['pub_time'].apply(self._extract_date)
            
            return result
            
        except Exception as e:
            logger.warning(f"采集 {symbol} 个股新闻失败: {e}")
            return pd.DataFrame()
    
    def _collect_market_news(self, max_pages: int = 5) -> pd.DataFrame:
        """
        采集市场要闻（快讯）
        
        Args:
            max_pages: 最大页数
        
        Returns:
            新闻DataFrame
        """
        all_records = []
        
        for page in range(1, max_pages + 1):
            url = self.NEWS_API.format(page=page)
            
            try:
                response = safe_request(
                    url,
                    headers=self._disguiser.get_json_headers(),
                    rate_limit=True
                )
                
                if response is None:
                    break
                
                # 解析JSONP格式
                text = response.text
                if text.startswith('var'):
                    # 提取JSON部分
                    import re
                    import json
                    match = re.search(r'\{.*\}', text, re.DOTALL)
                    if match:
                        data = json.loads(match.group())
                        news_list = data.get('LivesList', [])
                        
                        for item in news_list:
                            record = {
                                'news_id': self._generate_id(
                                    item.get('title', ''), 'eastmoney'
                                ),
                                'title': item.get('title', ''),
                                'content': item.get('digest', ''),
                                'summary': item.get('digest', ''),
                                'pub_time': item.get('showtime', ''),
                                'url': item.get('url', ''),
                                'source': 'eastmoney',
                                'category': NewsCategory.MARKET,
                                'related_stocks': '',
                                'keywords': '',
                            }
                            all_records.append(record)
                
            except Exception as e:
                logger.debug(f"采集东方财富快讯第{page}页失败: {e}")
                break
        
        if not all_records:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        df['pub_date'] = df['pub_time'].apply(self._extract_date)
        
        return df
    
    def _generate_id(self, title: str, source: str) -> str:
        """生成新闻唯一ID"""
        content = f"{source}_{title}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _extract_date(self, time_str: str) -> str:
        """从时间字符串提取日期"""
        if not time_str:
            return ''
        try:
            # 尝试多种格式
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d %H:%M:%S']:
                try:
                    dt = datetime.strptime(str(time_str)[:19], fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            return str(time_str)[:10]
        except Exception:
            return ''
    
    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """按日期过滤"""
        if df.empty or 'pub_date' not in df.columns:
            return df
        
        mask = (df['pub_date'] >= start_date) & (df['pub_date'] <= end_date)
        return df[mask].copy()
    
    def _standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化输出"""
        for col in self.STANDARD_FIELDS:
            if col not in df.columns:
                df[col] = ''
        
        return df[self.STANDARD_FIELDS]


# 便捷函数

def get_eastmoney_news(
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    获取东方财富新闻
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        symbols: 股票代码列表（个股新闻）
    
    Returns:
        新闻数据DataFrame
    """
    collector = EastMoneyNewsCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols
    )


def get_stock_news(symbol: str) -> pd.DataFrame:
    """
    获取个股新闻
    
    Args:
        symbol: 股票代码
    
    Returns:
        新闻数据DataFrame
    """
    collector = EastMoneyNewsCollector()
    return collector._collect_stock_news(symbol)
