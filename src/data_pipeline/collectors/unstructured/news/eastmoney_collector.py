"""
东方财富新闻采集器（增强版）

多接口采集东方财富财经新闻：
- AKShare stock_news_em: 个股新闻
- 东方财富财经快讯 API
- Playwright 动态渲染（备选）
"""

import logging
import hashlib
import re
import json
import time
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd

from ..base import UnstructuredCollector, DataSourceType, DateRangeIterator
from ..request_utils import safe_request, RequestDisguiser
from ..scraper_base import ScraperBase, PlaywrightDriver
from .cctv_collector import NewsCategory

logger = logging.getLogger(__name__)


class EastMoneyNewsCollector(UnstructuredCollector):
    """
    东方财富新闻采集器（增强版）
    
    数据来源优先级：
    1. AKShare stock_news_em - 个股新闻
    2. 东方财富快讯 API - 财经要闻
    3. 7x24小时直播 API - 实时资讯
    4. Playwright 渲染 - 动态内容备选
    """
    
    # 东方财富新闻API（更新后的有效接口 - 2024年验证）
    # 7x24快讯 - 新版API
    NEWS_7X24_API = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    # 快讯列表
    NEWS_KUAIXUN_API = "https://push2.eastmoney.com/api/qt/clist/get"
    # 财经要闻 - 通过wap接口
    NEWS_CJYW_API = "https://np-cnotice-stock.eastmoney.com/api/content/ann"
    # 个股资讯
    STOCK_NEWS_API = "https://searchapi.eastmoney.com/api/search/search"
    # 要闻直接接口
    NEWS_DIRECT_API = "https://finance.eastmoney.com/a/ccjyw.html"
    
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
    
    # 频道ID映射
    COLUMN_IDS = {
        'cjyw': '350',      # 财经要闻
        'stock': '358',     # 股票
        'fund': '359',      # 基金
        'finance': '352',   # 金融
        'macro': '353',     # 宏观
        'industry': '354',  # 产业
        'company': '355',   # 公司
        'global': '356',    # 全球
    }
    
    def __init__(self, use_playwright: bool = False):
        super().__init__()
        self._ak = None
        self._scraper = ScraperBase(rate_limit=True)
        self._disguiser = RequestDisguiser()
        self._use_playwright = use_playwright
        self._browser: Optional[PlaywrightDriver] = None
    
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
        channels: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集东方财富新闻
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            symbols: 股票代码列表（用于个股新闻）
            include_market_news: 是否包含市场要闻
            channels: 频道列表（cjyw/stock/fund/finance/macro/industry/company/global）
        
        Returns:
            新闻数据DataFrame
        """
        all_data = []
        
        # 1. 采集个股新闻
        if symbols:
            for symbol in symbols:
                df = self._collect_stock_news(symbol)
                if not df.empty:
                    df = self._filter_by_date(df, start_date, end_date)
                    if not df.empty:
                        all_data.append(df)
        
        # 2. 采集市场要闻（7x24快讯）
        if include_market_news:
            df = self._collect_7x24_news(start_date, end_date)
            if not df.empty:
                all_data.append(df)
        
        # 3. 采集各频道新闻
        if channels:
            for channel in channels:
                if channel in self.COLUMN_IDS:
                    df = self._collect_channel_news(channel, start_date, end_date)
                    if not df.empty:
                        all_data.append(df)
        
        if not all_data:
            # 备选：使用 Playwright 采集
            if self._use_playwright:
                df = self._collect_with_playwright()
                if not df.empty:
                    all_data.append(df)
            else:
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
            # 尝试 AkShare
            df = self.ak.stock_news_em(symbol=symbol)
            
            if df is not None and not df.empty:
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
                
                logger.info(f"AkShare 采集 {symbol} 个股新闻: {len(result)} 条")
                return result
                
        except Exception as e:
            logger.debug(f"AkShare 采集 {symbol} 新闻失败: {e}")
        
        # 备选：使用搜索 API
        return self._search_stock_news(symbol)
    
    def _search_stock_news(self, symbol: str) -> pd.DataFrame:
        """使用搜索 API 获取个股新闻"""
        all_records = []
        
        try:
            # 东方财富搜索 API
            params = {
                'keyword': symbol,
                'type': 'news',
                'pageindex': 1,
                'pagesize': 50,
            }
            
            response = self._scraper.get(
                self.STOCK_NEWS_API,
                params=params,
                referer='https://so.eastmoney.com/'
            )
            
            if response and response.status_code == 200:
                data = response.json()
                news_list = data.get('data', {}).get('news', {}).get('list', [])
                
                for item in news_list:
                    record = {
                        'news_id': self._generate_id(item.get('title', ''), 'eastmoney'),
                        'title': item.get('title', ''),
                        'content': item.get('content', ''),
                        'summary': item.get('digest', ''),
                        'pub_time': item.get('date', ''),
                        'url': item.get('url', ''),
                        'source': 'eastmoney',
                        'category': NewsCategory.STOCK,
                        'related_stocks': symbol,
                        'keywords': '',
                    }
                    all_records.append(record)
                    
        except Exception as e:
            logger.debug(f"搜索 API 采集失败: {e}")
        
        if all_records:
            df = pd.DataFrame(all_records)
            df['pub_date'] = df['pub_time'].apply(self._extract_date)
            return df
        
        return pd.DataFrame()
    
    def _collect_7x24_news(
        self,
        start_date: str,
        end_date: str,
        max_count: int = 500
    ) -> pd.DataFrame:
        """
        采集7x24小时快讯
        
        策略：
        1. 尝试AkShare财经快讯接口
        2. 备用：直接解析财经要闻页面
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            max_count: 最大条数
        
        Returns:
            新闻DataFrame
        """
        all_records = []
        
        # 策略1：AkShare财经新闻
        try:
            df = self.ak.stock_zh_a_alerts_cls()
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    title = str(row.get('标题', row.get('title', '')))
                    content = str(row.get('内容', row.get('content', '')))
                    pub_time = str(row.get('发布时间', row.get('time', '')))
                    
                    if not title:
                        continue
                    
                    record = {
                        'news_id': self._generate_id(title, 'eastmoney_cls'),
                        'title': title,
                        'content': content,
                        'summary': content[:200] if content else '',
                        'pub_time': pub_time,
                        'pub_date': pub_time[:10] if pub_time and len(pub_time) >= 10 else '',
                        'url': '',
                        'source': 'eastmoney_cls',
                        'category': NewsCategory.MARKET,
                        'related_stocks': '',
                        'keywords': '',
                    }
                    all_records.append(record)
                
                logger.info(f"AkShare快讯采集完成: {len(all_records)} 条")
        except Exception as e:
            logger.debug(f"AkShare快讯采集失败: {e}")
        
        # 策略2：东方财富快讯页面
        if not all_records:
            try:
                records = self._scrape_eastmoney_kuaixun()
                all_records.extend(records)
            except Exception as e:
                logger.debug(f"东方财富快讯页面采集失败: {e}")
        
        # 策略3：AkShare股票新闻
        if not all_records:
            try:
                df = self.ak.stock_news_em(symbol='股票')
                if df is not None and not df.empty:
                    for _, row in df.head(max_count).iterrows():
                        title = str(row.get('新闻标题', row.get('title', '')))
                        content = str(row.get('新闻内容', row.get('content', '')))
                        pub_time = str(row.get('发布时间', row.get('pub_time', '')))
                        
                        if not title:
                            continue
                        
                        record = {
                            'news_id': self._generate_id(title, 'eastmoney'),
                            'title': title,
                            'content': content,
                            'summary': content[:200] if content else '',
                            'pub_time': pub_time,
                            'pub_date': pub_time[:10] if pub_time and len(pub_time) >= 10 else '',
                            'url': str(row.get('新闻链接', '')),
                            'source': 'eastmoney',
                            'category': NewsCategory.MARKET,
                            'related_stocks': '',
                            'keywords': '',
                        }
                        all_records.append(record)
                    
                    logger.info(f"AkShare股票新闻采集完成: {len(all_records)} 条")
            except Exception as e:
                logger.debug(f"AkShare股票新闻采集失败: {e}")
        
        logger.info(f"7x24快讯采集完成: {len(all_records)} 条")
        
        if all_records:
            df = pd.DataFrame(all_records)
            # 日期过滤 - 统一格式
            if 'pub_date' in df.columns:
                # 转换日期格式
                start_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}" if len(start_date) == 8 and '-' not in start_date else start_date
                end_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}" if len(end_date) == 8 and '-' not in end_date else end_date
                mask = (df['pub_date'] >= start_fmt) & (df['pub_date'] <= end_fmt)
                df = df[mask | (df['pub_date'] == '') | (df['pub_date'].str.len() < 10)]
            return df
        
        return pd.DataFrame()
    
    def _scrape_eastmoney_kuaixun(self) -> List[Dict]:
        """爬取东方财富快讯页面"""
        records = []
        
        try:
            # 快讯页面
            url = "https://kuaixun.eastmoney.com/"
            
            response = self._scraper.get(
                url,
                referer="https://www.eastmoney.com/"
            )
            
            if response and response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找新闻列表
                for item in soup.select('.news_item, .item, .kuaixun-item, li.news'):
                    try:
                        title_elem = item.select_one('a.title, a, h2 a, .news-title')
                        if not title_elem:
                            continue
                        
                        title = title_elem.get_text(strip=True)
                        if not title or len(title) < 5:
                            continue
                        
                        href = title_elem.get('href', '')
                        
                        time_elem = item.select_one('.time, .date, .pub-time, span.time')
                        pub_time = time_elem.get_text(strip=True) if time_elem else ''
                        
                        content_elem = item.select_one('.content, .desc, .summary')
                        content = content_elem.get_text(strip=True) if content_elem else ''
                        
                        records.append({
                            'news_id': self._generate_id(title, 'eastmoney_kuaixun'),
                            'title': title,
                            'content': content or title,
                            'summary': (content or title)[:200],
                            'pub_time': pub_time,
                            'pub_date': '',
                            'url': href if href.startswith('http') else f"https://kuaixun.eastmoney.com{href}",
                            'source': 'eastmoney_kuaixun',
                            'category': NewsCategory.MARKET,
                            'related_stocks': '',
                            'keywords': '',
                        })
                        
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.debug(f"爬取快讯页面失败: {e}")
        
        return records
    
    def _collect_channel_news(
        self,
        channel: str,
        start_date: str,
        end_date: str,
        max_pages: int = 10
    ) -> pd.DataFrame:
        """
        采集指定频道新闻
        
        策略：
        1. 尝试API接口
        2. 备用：爬取频道页面
        
        Args:
            channel: 频道名称
            start_date: 开始日期
            end_date: 结束日期
            max_pages: 最大页数
        
        Returns:
            新闻DataFrame
        """
        all_records = []
        
        # 策略1：尝试API
        records = self._collect_channel_api(channel, max_pages)
        if records:
            all_records.extend(records)
        
        # 策略2：爬取频道页面
        if not all_records:
            records = self._scrape_channel_page(channel)
            if records:
                all_records.extend(records)
        
        logger.info(f"{channel} 频道新闻采集完成: {len(all_records)} 条")
        
        if all_records:
            df = pd.DataFrame(all_records)
            # 日期过滤
            if 'pub_date' in df.columns:
                mask = (df['pub_date'] >= start_date) & (df['pub_date'] <= end_date) | (df['pub_date'] == '')
                df = df[mask]
            return df
        
        return pd.DataFrame()
    
    def _collect_channel_api(self, channel: str, max_pages: int = 5) -> List[Dict]:
        """通过API采集频道新闻"""
        records = []
        column_id = self.COLUMN_IDS.get(channel, '350')
        
        try:
            for page in range(1, max_pages + 1):
                params = {
                    'column': column_id,
                    'client': 'web',
                    'pageNo': page,
                    'pageSize': 50,
                }
                
                response = self._scraper.get(
                    self.NEWS_CJYW_API,
                    params=params,
                    referer='https://finance.eastmoney.com/'
                )
                
                if not response or response.status_code != 200:
                    break
                
                try:
                    data = response.json()
                    news_list = data.get('data', {}).get('list', [])
                    
                    if not news_list:
                        break
                    
                    for item in news_list:
                        pub_time = item.get('showTime', item.get('date', ''))
                        pub_date = pub_time[:10] if pub_time and len(pub_time) >= 10 else ''
                        
                        record = {
                            'news_id': self._generate_id(item.get('title', ''), f'eastmoney_{channel}'),
                            'title': item.get('title', ''),
                            'content': item.get('digest', item.get('content', '')),
                            'summary': item.get('digest', '')[:200] if item.get('digest') else '',
                            'pub_time': pub_time,
                            'pub_date': pub_date,
                            'url': item.get('url', ''),
                            'source': f'eastmoney_{channel}',
                            'category': self._map_channel_category(channel),
                            'related_stocks': '',
                            'keywords': '',
                        }
                        records.append(record)
                    
                except Exception as e:
                    logger.debug(f"解析 {channel} API响应失败: {e}")
                    break
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.debug(f"{channel} API采集失败: {e}")
        
        return records
    
    def _scrape_channel_page(self, channel: str) -> List[Dict]:
        """爬取频道页面"""
        records = []
        
        # 频道页面URL映射
        channel_urls = {
            'cjyw': 'https://finance.eastmoney.com/a/ccjyw.html',
            'stock': 'https://stock.eastmoney.com/a/cgsgg.html',
            'fund': 'https://fund.eastmoney.com/a/cfgjj.html',
            'macro': 'https://finance.eastmoney.com/a/chgjj.html',
            'company': 'https://stock.eastmoney.com/a/cgsxw.html',
        }
        
        url = channel_urls.get(channel)
        if not url:
            return records
        
        try:
            response = self._scraper.get(
                url,
                referer='https://www.eastmoney.com/'
            )
            
            if response and response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找新闻列表
                for item in soup.select('.news_item a, .article a, .list a, ul.news_list li a, .title a'):
                    try:
                        title = item.get_text(strip=True)
                        if not title or len(title) < 8:
                            continue
                        
                        href = item.get('href', '')
                        if not href or not re.search(r'/\d{14}\.html|/a/\d+\.html', href):
                            continue
                        
                        # 从URL提取日期
                        pub_date = ''
                        date_match = re.search(r'/(\d{4})(\d{2})(\d{2})', href)
                        if date_match:
                            pub_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                        
                        records.append({
                            'news_id': self._generate_id(title, f'eastmoney_{channel}'),
                            'title': title,
                            'content': '',
                            'summary': title,
                            'pub_time': pub_date,
                            'pub_date': pub_date,
                            'url': href if href.startswith('http') else f"https://finance.eastmoney.com{href}",
                            'source': f'eastmoney_{channel}',
                            'category': self._map_channel_category(channel),
                            'related_stocks': '',
                            'keywords': '',
                        })
                        
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.debug(f"爬取 {channel} 页面失败: {e}")
        
        return records
    
    def _collect_with_playwright(self) -> pd.DataFrame:
        """使用 Playwright 采集动态渲染的新闻"""
        all_records = []
        
        try:
            self._browser = PlaywrightDriver(headless=True)
            
            # 财经要闻页面
            html = self._browser.get_with_scroll(
                "https://finance.eastmoney.com/",
                scroll_times=3,
                wait_time=1.0
            )
            
            if html:
                records = self._parse_finance_page(html)
                all_records.extend(records)
            
            logger.info(f"Playwright 采集新闻: {len(all_records)} 条")
            
        except Exception as e:
            logger.warning(f"Playwright 采集失败: {e}")
        finally:
            if self._browser:
                self._browser.close()
                self._browser = None
        
        if all_records:
            return pd.DataFrame(all_records)
        
        return pd.DataFrame()
    
    def _parse_finance_page(self, html: str) -> List[Dict]:
        """解析财经页面HTML"""
        records = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # 查找新闻列表
            for item in soup.select('.news_item, .article_item, .list_item a'):
                try:
                    title = item.get_text(strip=True)
                    if not title or len(title) < 10:
                        continue
                    
                    href = item.get('href', '')
                    if not href:
                        link = item.find('a')
                        href = link.get('href', '') if link else ''
                    
                    if not href.startswith('http'):
                        href = f"https://finance.eastmoney.com{href}" if href.startswith('/') else ''
                    
                    record = {
                        'news_id': self._generate_id(title, 'eastmoney'),
                        'title': title,
                        'content': '',
                        'summary': title[:100] if len(title) > 100 else title,
                        'pub_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'pub_date': datetime.now().strftime('%Y-%m-%d'),
                        'url': href,
                        'source': 'eastmoney',
                        'category': NewsCategory.MARKET,
                        'related_stocks': '',
                        'keywords': '',
                    }
                    records.append(record)
                    
                except Exception:
                    continue
                    
        except ImportError:
            logger.warning("BeautifulSoup 未安装")
        except Exception as e:
            logger.debug(f"解析页面失败: {e}")
        
        return records
    
    def _map_channel_category(self, channel: str) -> str:
        """映射频道到分类"""
        mapping = {
            'cjyw': NewsCategory.MARKET,
            'stock': NewsCategory.STOCK,
            'fund': NewsCategory.MARKET,
            'finance': NewsCategory.MARKET,
            'macro': NewsCategory.MACRO,
            'industry': NewsCategory.INDUSTRY,
            'company': NewsCategory.COMPANY,
            'global': NewsCategory.OTHER,
        }
        return mapping.get(channel, NewsCategory.OTHER)
    
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
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d']:
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
        
        # 返回有效日期的记录 + 无日期的记录（保留）
        return pd.concat([df_valid, df_invalid], ignore_index=True)
    
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
    symbols: Optional[List[str]] = None,
    channels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    获取东方财富新闻
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        symbols: 股票代码列表（个股新闻）
        channels: 频道列表（cjyw/stock/fund/finance/macro/industry/company/global）
    
    Returns:
        新闻数据DataFrame
    """
    collector = EastMoneyNewsCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        channels=channels or ['cjyw', 'stock']
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


def get_7x24_news(
    start_date: str,
    end_date: str,
    max_count: int = 500
) -> pd.DataFrame:
    """
    获取7x24小时快讯
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        max_count: 最大条数
    
    Returns:
        新闻数据DataFrame
    """
    collector = EastMoneyNewsCollector()
    return collector._collect_7x24_news(start_date, end_date, max_count)
