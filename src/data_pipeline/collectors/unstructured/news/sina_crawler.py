"""
新浪财经新闻爬虫

从新浪财经网站采集财经新闻

支持多种API接口：
1. 滚动新闻 API（主推荐）
2. 财经频道 API
3. 新浪新闻搜索 API（备选）
"""

import logging
import hashlib
import re
import time
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd

from ..base import UnstructuredCollector
from ..request_utils import safe_request, RequestDisguiser
from .cctv_collector import NewsCategory

logger = logging.getLogger(__name__)


class SinaFinanceCrawler(UnstructuredCollector):
    """
    新浪财经新闻爬虫（增强版）
    
    目标：finance.sina.com.cn
    
    API优先级：
    1. 滚动新闻 API (feed.mix.sina.com.cn)
    2. 财经频道 API (interface.sina.cn)
    3. 新浪新闻搜索 (search.sina.com.cn)
    """
    
    # 主要 API
    ROLL_NEWS_API = "https://feed.mix.sina.com.cn/api/roll/get"
    
    # 备用 API - 财经频道
    CHANNEL_API = "https://interface.sina.cn/news/wap/fymap2020_getNews.d.json"
    
    # 备用 API - 实时新闻
    REALTIME_API = "https://zhibo.sina.com.cn/api/zhibo/feed"
    
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
    
    # 新闻分类ID - 更新后的有效ID（2024年验证）
    CATEGORY_IDS = {
        'stock': '252',       # 股票（新ID）
        'fund': '253',        # 基金
        'futures': '254',     # 期货
        'forex': '255',       # 外汇
        'finance': '250',     # 财经要闻
        'market': '251',      # 市场动态
        'hk': '260',          # 港股
        'us': '261',          # 美股
    }
    
    # 旧版分类ID（某些接口仍使用）
    LEGACY_CATEGORY_IDS = {
        'stock': '2516',
        'fund': '2517',
        'futures': '2518',
        'forex': '2519',
        'finance': '2512',
        'market': '2511',
    }
    
    # 财经频道映射
    CHANNEL_MAPPING = {
        'stock': 'finance_stock',
        'fund': 'finance_fund',
        'finance': 'finance',
        'market': 'finance',
    }
    
    def __init__(self):
        super().__init__()
        self._disguiser = RequestDisguiser()
        self._api_failed = set()  # 记录失败的API
    
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
            df = pd.DataFrame()
            
            # 策略1：滚动新闻 API（新版）
            if 'roll_api' not in self._api_failed:
                df = self._collect_roll_news(category)
                if df.empty:
                    self._api_failed.add('roll_api')
            
            # 策略2：滚动新闻 API（旧版ID）
            if df.empty and 'roll_api_legacy' not in self._api_failed:
                df = self._collect_roll_news_legacy(category)
                if df.empty:
                    self._api_failed.add('roll_api_legacy')
            
            # 策略3：财经频道 API
            if df.empty and 'channel_api' not in self._api_failed:
                df = self._collect_channel_news(category)
                if df.empty:
                    self._api_failed.add('channel_api')
            
            # 策略4：实时新闻 API
            if df.empty and 'realtime_api' not in self._api_failed:
                df = self._collect_realtime_news(category)
            
            if not df.empty:
                df = self._filter_by_date(df, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"采集新浪财经 {category}: {len(df)} 条")
        
        if not all_data:
            logger.warning("新浪财经所有API接口均未返回数据")
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['news_id'], keep='first')
        
        return self._standardize_output(result)
    
    def _collect_roll_news(
        self,
        category: str,
        max_pages: int = 10
    ) -> pd.DataFrame:
        """使用新版分类ID采集滚动新闻"""
        lid = self.CATEGORY_IDS.get(category, '250')
        return self._collect_by_category_internal(category, lid, max_pages)
    
    def _collect_roll_news_legacy(
        self,
        category: str,
        max_pages: int = 10
    ) -> pd.DataFrame:
        """使用旧版分类ID采集滚动新闻"""
        lid = self.LEGACY_CATEGORY_IDS.get(category, '2512')
        return self._collect_by_category_internal(category, lid, max_pages)
    
    def _collect_by_category_internal(
        self,
        category: str,
        lid: str,
        max_pages: int = 10
    ) -> pd.DataFrame:
        """
        按分类采集新闻（内部方法）
        
        Args:
            category: 分类名称
            lid: 分类ID
            max_pages: 最大页数
        
        Returns:
            新闻DataFrame
        """
        all_records = []
        
        for page in range(1, max_pages + 1):
            params = {
                'pageid': '153',
                'lid': lid,
                'num': '50',
                'page': str(page),
                'k': '',
                'callback': '',  # 不使用 JSONP
            }
            
            try:
                response = safe_request(
                    self.ROLL_NEWS_API,
                    params=params,
                    headers=self._disguiser.get_json_headers({
                        'Referer': 'https://finance.sina.com.cn/',
                        'Origin': 'https://finance.sina.com.cn',
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
                
                logger.debug(f"采集新浪财经 {category} (lid={lid}) 第{page}页: {len(news_list)} 条")
                
                time.sleep(0.1)  # 请求间隔
                
            except Exception as e:
                logger.warning(f"采集新浪财经 {category} 第{page}页失败: {e}")
                break
        
        if not all_records:
            return pd.DataFrame()
        
        return pd.DataFrame(all_records)
    
    def _collect_channel_news(
        self,
        category: str,
        max_pages: int = 5
    ) -> pd.DataFrame:
        """
        采集财经频道新闻（备用方案）
        
        使用 interface.sina.cn 接口
        """
        all_records = []
        channel = self.CHANNEL_MAPPING.get(category, 'finance')
        
        for page in range(1, max_pages + 1):
            params = {
                'callback': '',
                'cat': channel,
                'num': '40',
                'page': str(page),
            }
            
            try:
                response = safe_request(
                    self.CHANNEL_API,
                    params=params,
                    headers=self._disguiser.get_json_headers({
                        'Referer': 'https://finance.sina.com.cn/',
                    }),
                    rate_limit=True
                )
                
                if response is None:
                    break
                
                data = response.json()
                news_list = data.get('data', [])
                
                if not news_list:
                    break
                
                for item in news_list:
                    record = self._parse_channel_item(item, category)
                    if record:
                        all_records.append(record)
                
                logger.debug(f"采集新浪财经频道 {channel} 第{page}页: {len(news_list)} 条")
                
                time.sleep(0.3)
                
            except Exception as e:
                logger.warning(f"采集新浪财经频道 {channel} 失败: {e}")
                break
        
        if not all_records:
            return pd.DataFrame()
        
        return pd.DataFrame(all_records)
    
    def _collect_realtime_news(
        self,
        category: str,
        max_pages: int = 5
    ) -> pd.DataFrame:
        """
        采集实时滚动新闻（备用方案）
        
        使用财经直播接口
        """
        all_records = []
        
        # 财经频道ID
        channel_ids = {
            'stock': 152,
            'finance': 151,
            'market': 152,
        }
        channel_id = channel_ids.get(category, 152)
        
        for page in range(1, max_pages + 1):
            params = {
                'zhibo_id': channel_id,
                'page': str(page),
                'page_size': '50',
                'tag_id': '0',
            }
            
            try:
                response = safe_request(
                    self.REALTIME_API,
                    params=params,
                    headers=self._disguiser.get_json_headers({
                        'Referer': 'https://zhibo.sina.com.cn/',
                    }),
                    rate_limit=True
                )
                
                if response is None:
                    break
                
                data = response.json()
                result = data.get('result', {})
                feeds = result.get('data', {}).get('feed', {}).get('list', [])
                
                if not feeds:
                    break
                
                for item in feeds:
                    record = self._parse_realtime_item(item, category)
                    if record:
                        all_records.append(record)
                
                logger.debug(f"采集新浪财经实时 第{page}页: {len(feeds)} 条")
                
                time.sleep(0.3)
                
            except Exception as e:
                logger.warning(f"采集新浪财经实时失败: {e}")
                break
        
        if not all_records:
            return pd.DataFrame()
        
        return pd.DataFrame(all_records)
    
    def _collect_by_category(
        self,
        category: str,
        max_pages: int = 10
    ) -> pd.DataFrame:
        """
        按分类采集新闻（兼容旧接口）
        
        Args:
            category: 分类名称
            max_pages: 最大页数
        
        Returns:
            新闻DataFrame
        """
        all_records = []
        lid = self.CATEGORY_IDS.get(category, '250')
        
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
        """解析滚动新闻单条"""
        try:
            title = item.get('title', '')
            if not title:
                return None
            
            pub_time = item.get('ctime', '') or item.get('create_time', '')
            
            # 转换时间戳（可能是字符串或数字）
            if pub_time:
                try:
                    ts = int(pub_time) if isinstance(pub_time, str) and pub_time.isdigit() else pub_time
                    if isinstance(ts, int):
                        pub_time = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError, OSError):
                    pub_time = str(pub_time)
            
            pub_date = pub_time[:10] if pub_time and len(pub_time) >= 10 else ''
            
            return {
                'news_id': self._generate_id(title, 'sina'),
                'title': title,
                'content': item.get('intro', '') or item.get('summary', ''),
                'summary': (item.get('intro', '') or item.get('summary', ''))[:200],
                'pub_time': pub_time,
                'pub_date': pub_date,
                'source': 'sina',
                'category': self._map_category(category),
                'url': item.get('url', '') or item.get('wapurl', ''),
                'related_stocks': '',
                'keywords': item.get('keywords', '') or item.get('tags', ''),
            }
            
        except Exception as e:
            logger.debug(f"解析新闻失败: {e}")
            return None
    
    def _parse_channel_item(self, item: Dict, category: str) -> Optional[Dict]:
        """解析财经频道新闻"""
        try:
            title = item.get('title', '')
            if not title:
                return None
            
            pub_time = item.get('ctime', '') or item.get('createtime', '')
            pub_date = pub_time[:10] if pub_time and len(pub_time) >= 10 else ''
            
            return {
                'news_id': self._generate_id(title, 'sina'),
                'title': title,
                'content': item.get('intro', '') or item.get('content', ''),
                'summary': (item.get('intro', '') or '')[:200],
                'pub_time': pub_time,
                'pub_date': pub_date,
                'source': 'sina',
                'category': self._map_category(category),
                'url': item.get('url', '') or item.get('link', ''),
                'related_stocks': '',
                'keywords': item.get('keywords', ''),
            }
            
        except Exception as e:
            logger.debug(f"解析频道新闻失败: {e}")
            return None
    
    def _parse_realtime_item(self, item: Dict, category: str) -> Optional[Dict]:
        """解析实时新闻"""
        try:
            title = item.get('rich_text', '') or item.get('text', '')
            if not title:
                return None
            
            # 实时新闻标题可能很长，截取前100字符作为标题
            title_short = title[:100] if len(title) > 100 else title
            
            create_time = item.get('create_time', 0)
            if isinstance(create_time, int) and create_time > 0:
                pub_time = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S')
            else:
                pub_time = str(create_time) if create_time else ''
            
            pub_date = pub_time[:10] if pub_time and len(pub_time) >= 10 else ''
            
            return {
                'news_id': self._generate_id(title_short, 'sina'),
                'title': title_short,
                'content': title,
                'summary': title[:200] if len(title) > 200 else title,
                'pub_time': pub_time,
                'pub_date': pub_date,
                'source': 'sina',
                'category': self._map_category(category),
                'url': item.get('url', ''),
                'related_stocks': '',
                'keywords': '',
            }
            
        except Exception as e:
            logger.debug(f"解析实时新闻失败: {e}")
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
        """按日期过滤
        
        Args:
            df: 数据DataFrame
            start_date: 开始日期 (YYYYMMDD 或 YYYY-MM-DD)
            end_date: 结束日期 (YYYYMMDD 或 YYYY-MM-DD)
        """
        if df.empty or 'pub_date' not in df.columns:
            return df
        
        # 统一日期格式为 YYYY-MM-DD
        if len(start_date) == 8 and '-' not in start_date:
            start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        if len(end_date) == 8 and '-' not in end_date:
            end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        
        # 确保 pub_date 是字符串格式
        df['pub_date'] = df['pub_date'].astype(str)
        
        # 过滤有效日期的记录
        valid_mask = df['pub_date'].str.len() >= 10
        df_valid = df[valid_mask].copy()
        df_invalid = df[~valid_mask].copy()
        
        if not df_valid.empty:
            mask = (df_valid['pub_date'] >= start_date) & (df_valid['pub_date'] <= end_date)
            df_valid = df_valid[mask]
        
        # 合并（保留无日期的记录）
        return pd.concat([df_valid, df_invalid], ignore_index=True)
    
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
