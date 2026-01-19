"""
证券时报新闻爬虫（增强版）

从证券时报网站采集专业证券新闻

支持：
1. 传统HTTP请求采集
2. Playwright动态页面采集（用于JS渲染页面）
3. API接口采集（如有）
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


class STCNCrawler(UnstructuredCollector):
    """
    证券时报新闻爬虫（增强版）
    
    目标：news.stcn.com, www.stcn.com
    
    采集策略：
    1. 优先HTTP请求采集静态页面
    2. 失败时使用Playwright处理动态加载
    3. 支持API接口采集（如可用）
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
    
    # 频道配置（更新后的URL）- 2024年验证
    CHANNEL_URLS = {
        'sd': 'https://news.stcn.com/sd/',           # 深度
        'djjd': 'https://news.stcn.com/djjd/',       # 独家解读
        'gsxw': 'https://company.stcn.com/',         # 公司新闻
        'kj': 'https://news.stcn.com/kj/',           # 科技
        'cj': 'https://news.stcn.com/cj/',           # 财经
        'yw': 'https://www.stcn.com/xw/yw/',         # 要闻
        'sc': 'https://www.stcn.com/xw/sc/',         # 市场
        'kx': 'https://kuaixun.stcn.com/',           # 快讯
    }
    
    # 新闻列表API（如有）
    NEWS_API = "https://www.stcn.com/article/list.json"
    
    def __init__(self):
        super().__init__()
        self._disguiser = RequestDisguiser()
        self._browser = None
        self._use_playwright = False  # 是否使用Playwright
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        channels: Optional[List[str]] = None,
        use_playwright: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集证券时报新闻
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            channels: 频道列表
            use_playwright: 是否强制使用Playwright（用于处理动态页面）
        
        Returns:
            新闻数据DataFrame
        """
        all_data = []
        self._use_playwright = use_playwright
        channels = channels or ['sd', 'djjd', 'gsxw', 'yw', 'sc']
        
        for channel in channels:
            if channel not in self.CHANNEL_URLS:
                continue
            
            # 尝试HTTP采集
            df = self._collect_by_channel_http(channel)
            
            # 如果HTTP失败或强制使用Playwright，尝试Playwright采集
            if (df.empty or self._use_playwright) and self._check_playwright_available():
                logger.info(f"使用Playwright采集 {channel}")
                df_pw = self._collect_by_channel_playwright(channel)
                if not df_pw.empty:
                    df = df_pw
            
            if not df.empty:
                # 日期过滤
                df = self._filter_by_date(df, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"采集证券时报 {channel}: {len(df)} 条")
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['news_id'], keep='first')
        
        return self._standardize_output(result)
    
    def _check_playwright_available(self) -> bool:
        """检查Playwright是否可用"""
        try:
            from playwright.sync_api import sync_playwright
            return True
        except ImportError:
            return False
    
    def _collect_by_channel_http(
        self,
        channel: str,
        max_pages: int = 5
    ) -> pd.DataFrame:
        """使用HTTP请求采集"""
        all_records = []
        base_url = self.CHANNEL_URLS.get(channel, '')
        
        if not base_url:
            return pd.DataFrame()
        
        for page in range(1, max_pages + 1):
            # 构建页面URL
            if page == 1:
                url = base_url
            else:
                # 尝试不同的分页格式
                if base_url.endswith('/'):
                    url = f"{base_url}index_{page}.html"
                else:
                    url = base_url.replace('.html', f'_{page}.html')
                    if url == base_url:
                        url = f"{base_url}?page={page}"
            
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
                
                # 使用BeautifulSoup解析，传递base_url
                records = self._parse_news_page(response.text, channel, base_url)
                
                if not records:
                    break
                
                all_records.extend(records)
                logger.debug(f"采集证券时报 {channel} 第{page}页: {len(records)} 条")
                
                time.sleep(0.3)
                
            except Exception as e:
                logger.warning(f"采集证券时报 {channel} 第{page}页失败: {e}")
                break
        
        if not all_records:
            return pd.DataFrame()
        
        return pd.DataFrame(all_records)
    
    def _collect_by_channel_playwright(
        self,
        channel: str,
        max_pages: int = 3
    ) -> pd.DataFrame:
        """
        使用Playwright采集动态页面
        
        用于JS渲染的页面或反爬严格的情况
        """
        all_records = []
        base_url = self.CHANNEL_URLS.get(channel, '')
        
        if not base_url:
            return pd.DataFrame()
        
        try:
            from ..scraper_base import PlaywrightDriver
            
            # 创建浏览器实例
            driver = PlaywrightDriver(headless=True)
            
            for page in range(1, max_pages + 1):
                if page == 1:
                    url = base_url
                else:
                    if base_url.endswith('/'):
                        url = f"{base_url}index_{page}.html"
                    else:
                        url = f"{base_url}?page={page}"
                
                try:
                    # 使用Playwright获取页面
                    html = driver.get_with_scroll(url, scroll_times=2, wait_time=1.0)
                    
                    if html:
                        records = self._parse_news_page(html, channel)
                        if records:
                            all_records.extend(records)
                            logger.debug(f"Playwright采集 {channel} 第{page}页: {len(records)} 条")
                    
                    time.sleep(1.0)  # Playwright需要更长间隔
                    
                except Exception as e:
                    logger.warning(f"Playwright采集 {channel} 第{page}页失败: {e}")
                    break
            
            driver.close()
            
        except ImportError:
            logger.warning("Playwright未安装，跳过动态采集")
        except Exception as e:
            logger.error(f"Playwright采集失败: {e}")
        
        if not all_records:
            return pd.DataFrame()
        
        return pd.DataFrame(all_records)
    
    def _parse_news_page(self, html: str, channel: str, base_url: str = '') -> List[Dict]:
        """使用BeautifulSoup解析新闻页面"""
        records = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # 收集所有链接
            found_links = set()
            
            # 遍历所有链接
            for elem in soup.find_all('a', href=True):
                href = elem.get('href', '')
                title = elem.get_text(strip=True)
                
                if not title or len(title) < 8:
                    continue
                if not href:
                    continue
                
                # 避免重复
                if href in found_links:
                    continue
                
                # 检查是否是新闻链接
                if not self._is_news_link(href):
                    continue
                
                found_links.add(href)
                
                # 从URL提取日期
                pub_date = self._extract_date_from_url(href)
                
                # 构建完整URL
                full_url = self._build_full_url(href, base_url)
                
                record = {
                    'news_id': self._generate_id(title, 'stcn'),
                    'title': title,
                    'content': '',
                    'summary': title[:100] if len(title) > 100 else title,
                    'pub_time': pub_date,
                    'pub_date': pub_date,
                    'source': 'stcn',
                    'category': self._map_category(channel),
                    'url': full_url,
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
    
    def _is_news_link(self, href: str) -> bool:
        """判断是否是新闻链接"""
        if not href:
            return False
        
        # 排除非新闻链接
        exclude_patterns = [
            'javascript:',
            '#',
            'mailto:',
            '/tag/',
            '/author/',
            '/page/',
        ]
        href_lower = href.lower()
        for pattern in exclude_patterns:
            if pattern in href_lower:
                return False
        
        # 证券时报常见的新闻链接模式
        # 模式1: ./cj/202210/t20221001_xxx.html 或 ./202210/t20221001_xxx.html
        # 匹配带子频道或不带子频道的相对路径
        if re.search(r'\.?/?(\w+/)?(\d{6})/t\d{8}_\d+\.html', href):
            return True
        # 模式2: https://news.stcn.com/sd/202210/t20221001_xxx.html
        if re.search(r'stcn\.com/\w+/\d{6}/t\d{8}_\d+\.html', href):
            return True
        # 模式3: /article/xxx
        if re.search(r'/article/\d+', href):
            return True
        # 模式4: /news/xxx
        if re.search(r'/news/\d+', href):
            return True
        # 模式5: /YYYY/MM/DD/xxx
        if re.search(r'/\d{4}/\d{2}/\d{2}/', href):
            return True
        
        return False
    
    def _extract_date_from_url(self, href: str) -> str:
        """从URL提取日期"""
        # 模式1: ./cj/202210/t20221001_xxx.html 或 /sd/202210/t20221001_xxx.html
        # 匹配 t后面的8位日期
        match = re.search(r'/t(\d{8})_\d+\.html', href)
        if match:
            date_str = match.group(1)
            year = int(date_str[:4])
            if 2020 <= year <= 2030:
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # 模式2: /YYYY/MM/DD/
        match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        
        # 模式3: YYYYMMDD在URL中
        match = re.search(r'(\d{4})(\d{2})(\d{2})', href)
        if match:
            year = match.group(1)
            month = match.group(2)
            day = match.group(3)
            if 2020 <= int(year) <= 2030 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
                return f"{year}-{month}-{day}"
        
        return ''
    
    def _build_full_url(self, href: str, base_url: str = '') -> str:
        """构建完整URL"""
        if href.startswith('http'):
            return href
        
        # 处理相对路径 ./xxx
        if href.startswith('./'):
            href = href[2:]
            if base_url:
                # 根据base_url构建完整路径
                # 例如: base_url = https://kuaixun.stcn.com/
                #       href = cj/202210/t20221001_xxx.html
                # 结果: https://kuaixun.stcn.com/cj/202210/t20221001_xxx.html
                if base_url.endswith('/'):
                    return f"{base_url}{href}"
                else:
                    base = base_url.rsplit('/', 1)[0]
                    return f"{base}/{href}"
        
        if href.startswith('//'):
            return f"https:{href}"
        
        if href.startswith('/'):
            return f"https://www.stcn.com{href}"
        
        # 对于没有./的相对路径，也需要处理
        if base_url:
            if base_url.endswith('/'):
                return f"{base_url}{href}"
            else:
                base = base_url.rsplit('/', 1)[0]
                return f"{base}/{href}"
        
        # 默认用news.stcn.com
        return f"https://news.stcn.com/{href}"
    
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
        if len(start_date) == 8:
            start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        if len(end_date) == 8:
            end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        
        # 如果没有日期信息，保留所有记录
        df_with_date = df[df['pub_date'].astype(str).str.len() >= 10].copy()
        df_without_date = df[df['pub_date'].astype(str).str.len() < 10].copy()
        
        if not df_with_date.empty:
            mask = (df_with_date['pub_date'] >= start_date) & (df_with_date['pub_date'] <= end_date)
            df_with_date = df_with_date[mask]
        
        # 合并有日期和无日期的记录
        return pd.concat([df_with_date, df_without_date], ignore_index=True)
    
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
