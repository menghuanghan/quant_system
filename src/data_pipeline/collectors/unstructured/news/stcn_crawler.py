"""
证券时报新闻爬虫

从证券时报网站采集专业证券新闻
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


class STCNCrawler(UnstructuredCollector):
    """
    证券时报新闻爬虫
    
    目标：news.stcn.com
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
    
    # 频道配置（更新后的URL）
    CHANNEL_URLS = {
        'sd': 'https://news.stcn.com/sd/index.html',         # 深度
        'djjd': 'https://news.stcn.com/djjd/index.html',     # 独家解读
        'gsxw': 'https://company.stcn.com/gsxw/index.html',  # 公司新闻
        'kj': 'https://news.stcn.com/kj/index.html',         # 科技
        'cj': 'https://news.stcn.com/cj/index.html',         # 财经
    }
    
    def __init__(self):
        super().__init__()
        self._disguiser = RequestDisguiser()
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        channels: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集证券时报新闻
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            channels: 频道列表
        
        Returns:
            新闻数据DataFrame
        """
        all_data = []
        channels = channels or list(self.CHANNEL_URLS.keys())
        
        for channel in channels:
            if channel not in self.CHANNEL_URLS:
                continue
            
            df = self._collect_by_channel(channel)
            if not df.empty:
                # 不做日期过滤，因为页面上可能没有日期
                all_data.append(df)
                logger.info(f"采集证券时报 {channel}: {len(df)} 条")
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['news_id'], keep='first')
        
        return self._standardize_output(result)
    
    def _collect_by_channel(
        self,
        channel: str,
        max_pages: int = 3
    ) -> pd.DataFrame:
        """
        按频道采集新闻
        """
        all_records = []
        base_url = self.CHANNEL_URLS.get(channel, '')
        
        if not base_url:
            return pd.DataFrame()
        
        for page in range(1, max_pages + 1):
            # 构建页面URL
            if page == 1:
                url = base_url
            else:
                url = base_url.replace('index.html', f'index_{page}.html')
            
            try:
                response = safe_request(
                    url,
                    headers=self._disguiser.get_headers({
                        'Referer': 'https://www.stcn.com/',
                    }),
                    rate_limit=True
                )
                
                if response is None:
                    break
                
                # 使用BeautifulSoup解析
                records = self._parse_news_page(response.text, channel)
                
                if not records:
                    break
                
                all_records.extend(records)
                logger.debug(f"采集证券时报 {channel} 第{page}页: {len(records)} 条")
                
            except Exception as e:
                logger.warning(f"采集证券时报 {channel} 第{page}页失败: {e}")
                break
        
        if not all_records:
            return pd.DataFrame()
        
        return pd.DataFrame(all_records)
    
    def _parse_news_page(self, html: str, channel: str) -> List[Dict]:
        """使用BeautifulSoup解析新闻页面"""
        records = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # 查找所有新闻链接
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link.get('href', '')
                title = link.get_text().strip()
                
                # 过滤条件
                if not title or len(title) < 10:
                    continue
                if not href:
                    continue
                    
                # 检查是否是新闻链接（包含日期格式的路径）
                if not re.search(r'/\d{6}/t\d{8}_\d+\.html', href):
                    continue
                
                # 从URL提取日期
                date_match = re.search(r'/(\d{6})/t(\d{8})_', href)
                pub_date = ''
                if date_match:
                    date_str = date_match.group(2)  # YYYYMMDD
                    try:
                        pub_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    except:
                        pass
                
                # 构建完整URL
                if href.startswith('./'):
                    href = href[2:]
                if href.startswith('/'):
                    href = f"https://news.stcn.com{href}"
                elif not href.startswith('http'):
                    href = f"https://news.stcn.com/{href}"
                
                record = {
                    'news_id': self._generate_id(title, 'stcn'),
                    'title': title,
                    'content': '',
                    'summary': title[:100] if len(title) > 100 else title,
                    'pub_time': pub_date,
                    'pub_date': pub_date,
                    'source': 'stcn',
                    'category': self._map_category(channel),
                    'url': href,
                    'related_stocks': '',
                    'keywords': '',
                }
                records.append(record)
            
        except ImportError:
            logger.warning("BeautifulSoup未安装，使用正则解析")
            records = self._parse_with_regex(html, channel)
        except Exception as e:
            logger.debug(f"解析HTML失败: {e}")
        
        # 去重
        seen = set()
        unique_records = []
        for r in records:
            if r['news_id'] not in seen:
                seen.add(r['news_id'])
                unique_records.append(r)
        
        return unique_records
    
    def _parse_with_regex(self, html: str, channel: str) -> List[Dict]:
        """使用正则表达式解析（备选方案）"""
        records = []
        
        # 匹配新闻链接模式
        pattern = r'href="([^"]*?/\d{6}/t\d{8}_\d+\.html)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, html)
        
        for href, title in matches:
            title = title.strip()
            if not title or len(title) < 10:
                continue
            
            # 提取日期
            date_match = re.search(r'/(\d{6})/t(\d{8})_', href)
            pub_date = ''
            if date_match:
                date_str = date_match.group(2)
                try:
                    pub_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                except:
                    pass
            
            # 完整URL
            if not href.startswith('http'):
                href = f"https://news.stcn.com{href}" if href.startswith('/') else f"https://news.stcn.com/{href}"
            
            record = {
                'news_id': self._generate_id(title, 'stcn'),
                'title': title,
                'content': '',
                'summary': '',
                'pub_time': pub_date,
                'pub_date': pub_date,
                'source': 'stcn',
                'category': self._map_category(channel),
                'url': href,
                'related_stocks': '',
                'keywords': '',
            }
            records.append(record)
        
        return records
    
    def _map_category(self, channel: str) -> str:
        """映射分类"""
        mapping = {
            'sd': NewsCategory.STOCK,
            'djjd': NewsCategory.COMPANY,
            'gsxw': NewsCategory.COMPANY,
            'kj': NewsCategory.INDUSTRY,
            'cj': NewsCategory.MACRO,
        }
        return mapping.get(channel, NewsCategory.OTHER)
    
    def _generate_id(self, title: str, source: str) -> str:
        """生成唯一ID"""
        content = f"{source}_{title}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化输出"""
        for col in self.STANDARD_FIELDS:
            if col not in df.columns:
                df[col] = ''
        
        return df[self.STANDARD_FIELDS]


# 便捷函数

def get_stcn_news(
    start_date: str,
    end_date: str,
    channels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    获取证券时报新闻
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        channels: 频道列表
    
    Returns:
        新闻数据DataFrame
    """
    crawler = STCNCrawler()
    return crawler.collect(
        start_date=start_date,
        end_date=end_date,
        channels=channels
    )
