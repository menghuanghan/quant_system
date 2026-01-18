"""
新浪财经新闻爬虫

从新浪财经网站采集财经新闻
"""

import logging
import hashlib
import re
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd

from ..base import UnstructuredCollector
from ..request_utils import safe_request, RequestDisguiser
from .cctv_collector import NewsCategory

logger = logging.getLogger(__name__)


class SinaFinanceCrawler(UnstructuredCollector):
    """
    新浪财经新闻爬虫
    
    目标：finance.sina.com.cn
    """
    
    # 新浪财经API
    ROLL_NEWS_API = "https://feed.mix.sina.com.cn/api/roll/get"
    
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
    
    # 新闻分类ID (更新后的有效ID)
    CATEGORY_IDS = {
        'stock': '2516',     # 股票
        'fund': '2517',      # 基金
        'futures': '2518',   # 期货
        'forex': '2519',     # 外汇
        'finance': '2512',   # 财经要闻
        'market': '2511',    # 市场动态
    }
    
    def __init__(self):
        super().__init__()
        self._disguiser = RequestDisguiser()
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        categories: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集新浪财经新闻
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            categories: 分类列表（stock/fund/futures/forex/finance/market）
        
        Returns:
            新闻数据DataFrame
        """
        all_data = []
        categories = categories or ['stock', 'finance', 'market']
        
        for category in categories:
            if category not in self.CATEGORY_IDS:
                continue
            
            df = self._collect_by_category(category)
            if not df.empty:
                df = self._filter_by_date(df, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['news_id'], keep='first')
        
        return self._standardize_output(result)
    
    def _collect_by_category(
        self,
        category: str,
        max_pages: int = 10
    ) -> pd.DataFrame:
        """
        按分类采集新闻
        
        Args:
            category: 分类名称
            max_pages: 最大页数
        
        Returns:
            新闻DataFrame
        """
        all_records = []
        lid = self.CATEGORY_IDS.get(category, '2512')
        
        for page in range(1, max_pages + 1):
            params = {
                'pageid': '153',
                'lid': lid,
                'num': '50',
                'page': str(page),
                'k': '',
            }
            
            try:
                response = safe_request(
                    self.ROLL_NEWS_API,
                    params=params,
                    headers=self._disguiser.get_json_headers({
                        'Referer': 'https://finance.sina.com.cn/',
                    }),
                    rate_limit=True
                )
                
                if response is None:
                    break
                
                data = response.json()
                result = data.get('result', {})
                news_list = result.get('data', [])
                
                if not news_list:
                    break
                
                for item in news_list:
                    record = self._parse_news_item(item, category)
                    if record:
                        all_records.append(record)
                
                logger.debug(f"采集新浪财经 {category} 第{page}页: {len(news_list)} 条")
                
            except Exception as e:
                logger.warning(f"采集新浪财经 {category} 第{page}页失败: {e}")
                break
        
        if not all_records:
            return pd.DataFrame()
        
        return pd.DataFrame(all_records)
    
    def _parse_news_item(self, item: Dict, category: str) -> Optional[Dict]:
        """解析单条新闻"""
        try:
            title = item.get('title', '')
            if not title:
                return None
            
            pub_time = item.get('ctime', '')
            
            # 转换时间戳（可能是字符串或数字）
            if pub_time:
                try:
                    ts = int(pub_time) if isinstance(pub_time, str) else pub_time
                    pub_time = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError, OSError):
                    pub_time = str(pub_time)
            
            pub_date = pub_time[:10] if pub_time and len(pub_time) >= 10 else ''
            
            return {
                'news_id': self._generate_id(title, 'sina'),
                'title': title,
                'content': item.get('intro', ''),
                'summary': item.get('intro', '')[:200] if item.get('intro') else '',
                'pub_time': pub_time,
                'pub_date': pub_date,
                'source': 'sina',
                'category': self._map_category(category),
                'url': item.get('url', ''),
                'related_stocks': '',
                'keywords': item.get('keywords', ''),
            }
            
        except Exception as e:
            logger.debug(f"解析新闻失败: {e}")
            return None
    
    def _map_category(self, category: str) -> str:
        """映射分类"""
        mapping = {
            'stock': NewsCategory.STOCK,
            'fund': NewsCategory.MARKET,
            'futures': NewsCategory.MARKET,
            'forex': NewsCategory.MARKET,
            'finance': NewsCategory.MACRO,
        }
        return mapping.get(category, NewsCategory.OTHER)
    
    def _generate_id(self, title: str, source: str) -> str:
        """生成唯一ID"""
        content = f"{source}_{title}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
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

def get_sina_news(
    start_date: str,
    end_date: str,
    categories: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    获取新浪财经新闻
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        categories: 分类列表
    
    Returns:
        新闻数据DataFrame
    """
    crawler = SinaFinanceCrawler()
    return crawler.collect(
        start_date=start_date,
        end_date=end_date,
        categories=categories
    )
