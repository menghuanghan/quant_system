"""
投资者舆情文本采集器

采集投资者舆情数据：
1. 互动易问答 (Tushare cninfo_interaction) - 高质量官方归档
2. 股吧/东方财富评论 - 高噪声但覆盖广
3. 雪球讨论 - 专业投资者聚集

核心特性：
- 混合采集策略 (API + 爬虫)
- 事件驱动回溯：只爬取股价波动>3%或成交量异常的日期
- Cookie轮询支持（雪球登录态）
"""

import os
import re
import time
import hashlib
import logging
from typing import Optional, List, Dict, Any, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from ..base import UnstructuredCollector, DateRangeIterator
from ..scraper_base import ScraperBase, CookieManager, get_cookie_manager

load_dotenv()

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """舆情数据来源"""
    CNINFO_INTERACTION = "cninfo_interaction"  # 互动易（Tushare）
    GUBA = "guba"                              # 东方财富股吧
    XUEQIU = "xueqiu"                          # 雪球
    ALL = "all"                                # 所有来源


@dataclass
class EventFilter:
    """事件过滤器配置"""
    # 股价波动阈值（百分比）
    price_change_threshold: float = 3.0
    # 成交量放大倍数阈值
    volume_ratio_threshold: float = 2.0
    # 是否使用涨跌停过滤
    include_limit_up: bool = True
    include_limit_down: bool = True


@dataclass
class SentimentConfig:
    """舆情采集配置"""
    # 每只股票每页评论数
    page_size: int = 20
    # 最大页数
    max_pages: int = 50
    # 请求间隔（秒）
    request_interval: float = 1.0
    # 是否启用事件驱动回溯
    event_driven: bool = True
    # 事件过滤器
    event_filter: EventFilter = field(default_factory=EventFilter)


class InvestorSentimentCollector(UnstructuredCollector):
    """
    投资者舆情文本采集器
    
    混合采集策略：
    1. 互动易（High Quality）: Tushare cninfo_interaction
    2. 股吧/雪球（High Noise）: 自建爬虫
    
    事件驱动回溯：
    - 输入股价波动表，只爬取异常交易日的评论
    - 避免全量爬取5年数据
    
    输出字段:
        - comment_id: 评论唯一ID（MD5）
        - ts_code: 股票代码
        - name: 股票名称
        - trade_date: 关联交易日
        - pub_time: 发布时间
        - author: 作者
        - title: 标题（帖子）
        - content: 内容
        - reply_count: 回复数
        - like_count: 点赞数
        - source: 数据来源
        - url: 原文链接
    """
    
    STANDARD_FIELDS = [
        'comment_id',
        'ts_code',
        'name',
        'trade_date',
        'pub_time',
        'author',
        'title',
        'content',
        'reply_count',
        'like_count',
        'source',
        'url',
    ]
    
    def __init__(self, config: Optional[SentimentConfig] = None):
        super().__init__()
        self.config = config or SentimentConfig()
        self._ts = None
        self._ak = None
        self._scraper: Optional[ScraperBase] = None
        self._tushare_token = os.getenv('TUSHARE_TOKEN')
    
    @property
    def ts(self):
        """懒加载 Tushare"""
        if self._ts is None and self._tushare_token:
            import tushare as ts
            ts.set_token(self._tushare_token)
            self._ts = ts.pro_api()
        return self._ts
    
    @property
    def ak(self):
        """懒加载 AkShare"""
        if self._ak is None:
            import akshare as ak
            self._ak = ak
        return self._ak
    
    @property
    def scraper(self) -> ScraperBase:
        """懒加载 Scraper"""
        if self._scraper is None:
            self._scraper = ScraperBase(
                use_proxy=False,
                rate_limit=True,
                timeout=30
            )
        return self._scraper
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        ts_codes: Optional[List[str]] = None,
        source: SentimentSource = SentimentSource.ALL,
        price_volatility_df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集投资者舆情数据
        
        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            ts_codes: 股票代码列表
            source: 数据来源
            price_volatility_df: 股价波动表（事件驱动用）
                                 需包含列: ts_code, trade_date, pct_chg, volume_ratio
        
        Returns:
            标准化的舆情数据 DataFrame
        """
        all_data = []
        
        # 确定要采集的股票和日期
        if self.config.event_driven and price_volatility_df is not None:
            # 事件驱动模式：筛选异常日期
            target_dates = self._filter_event_dates(price_volatility_df, ts_codes)
            self.logger.info(f"事件驱动模式：筛选出 {len(target_dates)} 个目标日期")
        else:
            target_dates = None
        
        # 1. 互动易数据（最优先）
        if source in (SentimentSource.CNINFO_INTERACTION, SentimentSource.ALL):
            df = self._collect_cninfo_interaction(start_date, end_date, ts_codes)
            if not df.empty:
                all_data.append(df)
        
        # 2. 股吧数据
        if source in (SentimentSource.GUBA, SentimentSource.ALL):
            df = self._collect_guba_comments(
                start_date, end_date, ts_codes, target_dates
            )
            if not df.empty:
                all_data.append(df)
        
        # 3. 雪球数据
        if source in (SentimentSource.XUEQIU, SentimentSource.ALL):
            df = self._collect_xueqiu_comments(
                start_date, end_date, ts_codes, target_dates
            )
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        # 合并去重
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['comment_id'], keep='first')
        
        return self._ensure_fields(result)
    
    # ==================== 互动易采集 ====================
    
    def _collect_cninfo_interaction(
        self,
        start_date: str,
        end_date: str,
        ts_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        采集互动易问答数据
        
        采集策略（优先级）：
        1. Tushare irm_qa 接口（需要5000+积分）
        2. 巨潮互动易网站爬虫（备选方案）
        
        数据源: Tushare cninfo_interaction / irm_qa / 网页爬虫
        """
        all_data = []
        
        # 策略1：尝试 Tushare
        if self.ts:
            df_tushare = self._collect_cninfo_tushare(start_date, end_date, ts_codes)
            if not df_tushare.empty:
                all_data.append(df_tushare)
                self.logger.info(f"Tushare互动易采集成功: {len(df_tushare)} 条")
        
        # 策略2：如果 Tushare 失败或无数据，使用爬虫
        if not all_data:
            self.logger.info("Tushare互动易接口不可用，尝试爬虫采集")
            df_scrape = self._scrape_cninfo_interaction(start_date, end_date, ts_codes)
            if not df_scrape.empty:
                all_data.append(df_scrape)
                self.logger.info(f"互动易爬虫采集成功: {len(df_scrape)} 条")
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            return result.drop_duplicates(subset=['comment_id'], keep='first')
        
        return pd.DataFrame()
    
    def _collect_cninfo_tushare(self, start_date: str, end_date: str, ts_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """使用 Tushare 采集互动易数据"""
        if not self.ts:
            return pd.DataFrame()
        
        all_data = []
        start_date_ts = start_date.replace('-', '')
        end_date_ts = end_date.replace('-', '')
        
        try:
            # ✅ 改进: 尝试多个接口
            methods = [
                ('irm_qa', {'start_date': start_date_ts, 'end_date': end_date_ts}),
                ('cninfo_interaction', {'start_date': start_date_ts, 'end_date': end_date_ts}),
            ]
            
            for method_name, base_params in methods:
                if not hasattr(self.ts, method_name):
                    continue
                    
                method = getattr(self.ts, method_name)
                
                if ts_codes:
                    # 按股票采集
                    for ts_code in ts_codes[:10]:  # 限制数量避免超时
                        try:
                            params = base_params.copy()
                            params['ts_code'] = ts_code
                            df = method(**params)
                            
                            if df is not None and not df.empty:
                                df['source'] = SentimentSource.CNINFO_INTERACTION.value
                                all_data.append(df)
                                self.logger.debug(f"{method_name} 采集 {ts_code}: {len(df)} 条")
                                
                        except Exception as e:
                            self.logger.debug(f"{method_name} 采集 {ts_code} 失败: {e}")
                            continue
                else:
                    # 全市场采集（按日期）
                    try:
                        df = method(**base_params)
                        if df is not None and not df.empty:
                            df['source'] = SentimentSource.CNINFO_INTERACTION.value
                            all_data.append(df)
                            self.logger.info(f"{method_name} 全市场采集: {len(df)} 条")
                    except Exception as e:
                        self.logger.debug(f"{method_name} 全市场采集失败: {e}")
                
                # 如果已有数据，停止尝试其他接口
                if all_data:
                    break
            
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                return self._standardize_cninfo(result)
                
        except Exception as e:
            self.logger.warning(f"Tushare互动易采集失败: {e}")
        
        return pd.DataFrame()
    
    def _scrape_cninfo_interaction(
        self,
        start_date: str,
        end_date: str,
        ts_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        爬取巨潮互动易网站
        
        URL: http://irm.cninfo.com.cn/
        
        采集策略：
        1. 如果指定股票代码，按个股采集
        2. 否则采集最新问答列表
        """
        all_data = []
        
        # 互动易API
        api_url = "http://irm.cninfo.com.cn/ircs/interaction/lastRask498"
        search_api = "http://irm.cninfo.com.cn/ircs/interaction/query"
        
        try:
            if ts_codes:
                # 按股票采集
                for ts_code in ts_codes:
                    records = self._scrape_cninfo_by_stock(ts_code, start_date, end_date)
                    all_data.extend(records)
                    time.sleep(0.5)
            else:
                # 采集最新问答
                records = self._scrape_cninfo_latest(start_date, end_date)
                all_data.extend(records)
            
        except Exception as e:
            self.logger.warning(f"互动易爬虫失败: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            return self._standardize_cninfo_scraped(df)
        
        return pd.DataFrame()
    
    def _scrape_cninfo_by_stock(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        max_pages: int = 5
    ) -> List[Dict]:
        """按股票爬取互动易问答"""
        records = []
        
        # 提取股票代码
        symbol = ts_code.split('.')[0]
        
        # 互动易搜索API
        search_url = "http://irm.cninfo.com.cn/ircs/interaction/query"
        
        try:
            for page in range(1, max_pages + 1):
                params = {
                    'stockCode': symbol,
                    'pageNum': page,
                    'pageSize': 30,
                }
                
                response = self.scraper.get(
                    search_url,
                    params=params,
                    headers={
                        'Accept': 'application/json',
                        'Referer': 'http://irm.cninfo.com.cn/',
                    }
                )
                
                if not response or response.status_code != 200:
                    break
                
                try:
                    data = response.json()
                    items = data.get('results', []) or data.get('data', [])
                    
                    if not items:
                        break
                    
                    for item in items:
                        q_date = item.get('intDate', '') or item.get('questionDate', '')
                        a_date = item.get('replyDate', '') or item.get('answerDate', '')
                        
                        # 日期过滤
                        check_date = q_date[:10] if q_date else ''
                        if check_date and (check_date < start_date or check_date > end_date):
                            continue
                        
                        record = {
                            'ts_code': ts_code,
                            'name': item.get('sname', '') or item.get('stockName', ''),
                            'q_date': q_date,
                            'a_date': a_date,
                            'question': item.get('question', ''),
                            'answer': item.get('reply', '') or item.get('answer', ''),
                            'source': SentimentSource.CNINFO_INTERACTION.value,
                        }
                        records.append(record)
                    
                except Exception as e:
                    self.logger.debug(f"解析互动易响应失败: {e}")
                    break
                    
        except Exception as e:
            self.logger.debug(f"按股票爬取互动易失败 {ts_code}: {e}")
        
        return records
    
    def _scrape_cninfo_latest(
        self,
        start_date: str,
        end_date: str,
        max_pages: int = 10
    ) -> List[Dict]:
        """爬取最新互动易问答"""
        records = []
        
        # 最新问答API
        api_url = "http://irm.cninfo.com.cn/ircs/interaction/lastRaskList.do"
        
        try:
            for page in range(1, max_pages + 1):
                params = {
                    'pageNum': page,
                    'pageSize': 50,
                }
                
                response = self.scraper.get(
                    api_url,
                    params=params,
                    headers={
                        'Accept': 'application/json',
                        'Referer': 'http://irm.cninfo.com.cn/',
                    }
                )
                
                if not response or response.status_code != 200:
                    # 尝试备用接口
                    response = self.scraper.post(
                        "http://irm.cninfo.com.cn/ircs/interaction/lastRaskList.do",
                        data=params,
                        headers={
                            'Content-Type': 'application/x-www-form-urlencoded',
                            'Referer': 'http://irm.cninfo.com.cn/',
                        }
                    )
                
                if not response or response.status_code != 200:
                    break
                
                try:
                    data = response.json()
                    items = data.get('results', []) or data.get('data', []) or data.get('list', [])
                    
                    if not items:
                        break
                    
                    for item in items:
                        q_date = item.get('intDate', '') or item.get('questionDate', '')
                        a_date = item.get('replyDate', '') or item.get('answerDate', '')
                        
                        # 日期过滤
                        check_date = q_date[:10] if q_date else ''
                        if check_date and (check_date < start_date or check_date > end_date):
                            continue
                        
                        # 提取股票代码
                        stock_code = item.get('stockCode', '') or item.get('secCode', '')
                        market = item.get('market', '')
                        
                        # 转换为 ts_code 格式
                        if stock_code:
                            if stock_code.startswith('6'):
                                ts_code = f"{stock_code}.SH"
                            elif stock_code.startswith(('0', '3')):
                                ts_code = f"{stock_code}.SZ"
                            else:
                                ts_code = stock_code
                        else:
                            ts_code = ''
                        
                        record = {
                            'ts_code': ts_code,
                            'name': item.get('sname', '') or item.get('stockName', ''),
                            'q_date': q_date,
                            'a_date': a_date,
                            'question': item.get('question', ''),
                            'answer': item.get('reply', '') or item.get('answer', ''),
                            'source': SentimentSource.CNINFO_INTERACTION.value,
                        }
                        records.append(record)
                    
                except Exception as e:
                    self.logger.debug(f"解析互动易响应失败: {e}")
                    break
                
                time.sleep(0.3)
                    
        except Exception as e:
            self.logger.debug(f"爬取最新互动易失败: {e}")
        
        return records
    
    def _standardize_cninfo_scraped(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化爬虫获取的互动易数据"""
        if df.empty:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        # 字段映射
        column_map = {
            'q_date': 'pub_time',
            'a_date': 'trade_date',
            'question': 'content',
            'answer': 'title',
        }
        
        df = df.rename(columns=column_map)
        
        # 生成唯一ID
        df['comment_id'] = df.apply(
            lambda x: hashlib.md5(
                f"cninfo_{x.get('ts_code', '')}{x.get('pub_time', '')}{str(x.get('content', ''))[:50]}".encode()
            ).hexdigest(),
            axis=1
        )
        
        df['author'] = '投资者'
        df['reply_count'] = 1
        df['like_count'] = None
        df['url'] = df.apply(
            lambda x: f"http://irm.cninfo.com.cn/ircs/company/{x.get('ts_code', '').split('.')[0]}" if x.get('ts_code') else '',
            axis=1
        )
        
        return df
        
        return pd.DataFrame()
    
    def _standardize_cninfo(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化互动易数据"""
        if df.empty:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        # 字段映射
        column_map = {
            'q_date': 'pub_time',
            'a_date': 'trade_date',
            'question': 'content',
            'answer': 'title',  # 回答作为标题
        }
        
        df = df.rename(columns=column_map)
        
        # 生成唯一ID
        df['comment_id'] = df.apply(
            lambda x: hashlib.md5(
                f"{x.get('ts_code', '')}{x.get('pub_time', '')}{x.get('content', '')[:50]}".encode()
            ).hexdigest(),
            axis=1
        )
        
        # 标准化日期
        if 'pub_time' in df.columns:
            df['pub_time'] = pd.to_datetime(df['pub_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        
        df['author'] = '投资者'
        df['reply_count'] = 1
        df['like_count'] = None
        df['url'] = None
        
        return df
    
    # ==================== 股吧采集 ====================
    
    def _collect_guba_comments(
        self,
        start_date: str,
        end_date: str,
        ts_codes: Optional[List[str]] = None,
        target_dates: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        采集东方财富股吧评论
        
        策略：
        1. 优先使用 AkShare stock_comment_em（如果可用）
        2. 备选：自建爬虫
        """
        all_data = []
        
        # 优先使用爬虫获取真实评论内容（AkShare接口返回的是机构参与度数据而非评论）
        if ts_codes:
            for ts_code in ts_codes:
                # 首先尝试爬虫获取带内容的评论
                try:
                    dates = target_dates.get(ts_code) if target_dates else None
                    df = self._scrape_guba_comments(ts_code, dates, fetch_content=True)
                    if not df.empty:
                        all_data.append(df)
                        self.logger.info(f"股吧爬虫采集 {ts_code}: {len(df)} 条")
                        continue
                except Exception as e:
                    self.logger.debug(f"股吧爬虫采集 {ts_code} 失败: {e}")
                
                # 备选：AkShare（注意：返回的是机构参与度数据）
                try:
                    df = self._collect_guba_akshare(ts_code, start_date, end_date)
                    if not df.empty:
                        all_data.append(df)
                except Exception as e:
                    self.logger.warning(f"AkShare 股吧采集 {ts_code} 失败: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        
        return pd.DataFrame()
    
    def _collect_guba_akshare(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """使用 AkShare 采集股吧数据"""
        try:
            # 尝试股吧热帖接口
            symbol = ts_code.split('.')[0]
            df = self.ak.stock_comment_detail_zlkp_jgcyd_em(symbol=symbol)
            
            if df is not None and not df.empty:
                df['ts_code'] = ts_code
                df['source'] = SentimentSource.GUBA.value
                df['comment_id'] = df.apply(
                    lambda x: hashlib.md5(
                        f"{ts_code}{x.get('发布时间', '')}{str(x.values)[:50]}".encode()
                    ).hexdigest(),
                    axis=1
                )
                return self._standardize_guba(df)
                
        except Exception as e:
            self.logger.debug(f"AkShare 股吧接口不可用: {e}")
        
        return pd.DataFrame()
    
    def _scrape_guba_comments(
        self,
        ts_code: str,
        target_dates: Optional[List[str]] = None,
        fetch_content: bool = True
    ) -> pd.DataFrame:
        """
        爬取股吧评论（增强版 - 获取完整内容）
        
        URL格式: https://guba.eastmoney.com/list,{code}.html
        
        策略：
        1. 获取帖子列表
        2. 对每个帖子获取详情页内容（可选）
        
        Args:
            ts_code: 股票代码
            target_dates: 目标日期过滤
            fetch_content: 是否获取帖子详细内容
        """
        all_data = []
        symbol = ts_code.split('.')[0]
        
        # 股吧列表页API
        list_api = "https://guba.eastmoney.com/interface/GetData.aspx"
        
        try:
            for page in range(1, min(self.config.max_pages + 1, 6)):  # 限制页数以加速
                # 方法1：使用API获取列表
                posts = self._fetch_guba_list_api(symbol, page)
                
                # 方法2：如果API失败，使用HTML解析
                if not posts:
                    posts = self._fetch_guba_list_html(symbol, page)
                
                if not posts:
                    break
                
                for post in posts:
                    post_id = post.get('post_id', '')
                    title = post.get('title', '') or post.get('post_title', '')
                    pub_time = post.get('pub_time', '') or post.get('post_publish_time', '')
                    
                    # 日期过滤
                    if target_dates:
                        pub_date = pub_time[:10] if pub_time else ''
                        if pub_date and pub_date not in target_dates:
                            continue
                    
                    # 获取帖子内容
                    content = post.get('content', '') or post.get('post_content', '')
                    
                    # 如果内容为空且启用了内容获取，则获取详情页
                    if fetch_content and not content and post_id:
                        content = self._fetch_guba_post_content(symbol, post_id)
                        time.sleep(0.05)  # 控制请求频率
                    
                    record = {
                        'ts_code': ts_code,
                        'pub_time': pub_time,
                        'author': post.get('author', '') or post.get('user_nickname', ''),
                        'title': title,
                        'content': content,
                        'reply_count': post.get('reply_count', 0),
                        'like_count': post.get('click_count', 0) or post.get('post_click_count', 0),
                        'source': SentimentSource.GUBA.value,
                        'url': f"https://guba.eastmoney.com/news,{symbol},{post_id}.html" if post_id else ''
                    }
                    
                    record['comment_id'] = hashlib.md5(
                        f"{ts_code}{record['pub_time']}{title[:30]}".encode()
                    ).hexdigest()
                    
                    all_data.append(record)
                
                self.logger.debug(f"股吧采集 {ts_code} 第{page}页: {len(posts)} 条")
                time.sleep(0.1)
            
        except Exception as e:
            self.logger.warning(f"股吧爬虫失败 {ts_code}: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            df['trade_date'] = df['pub_time'].str[:10]
            return df
        
        return pd.DataFrame()
    
    def _fetch_guba_list_api(self, symbol: str, page: int) -> List[Dict]:
        """使用API获取股吧帖子列表"""
        try:
            # ✅ 改进: 增加更完整的请求头
            api_url = "https://guba.eastmoney.com/interface/GetData.aspx"
            params = {
                'path': 'reply/api/ArtReplyList',
                'param': f'postid={symbol},sorttype=1,p={page},ps={self.config.page_size}',
                '_': int(time.time() * 1000)
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': f'https://guba.eastmoney.com/list,{symbol}.html',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'X-Requested-With': 'XMLHttpRequest',
            }
            
            response = self.scraper.get(
                api_url,
                params=params,
                headers=headers,
                timeout=15
            )
            
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    return data.get('re', [])
                except:
                    # 如果不是JSON，尝试解析HTML
                    self.logger.debug("股吧API返回非 JSON 数据")
        except Exception as e:
            self.logger.debug(f"股吧API获取列表失败: {e}")
        
        return []
    
    def _fetch_guba_list_html(self, symbol: str, page: int) -> List[Dict]:
        """解析股吧HTML页面获取帖子列表"""
        posts = []
        
        try:
            # 股吧列表页URL
            if page == 1:
                url = f"https://guba.eastmoney.com/list,{symbol}.html"
            else:
                url = f"https://guba.eastmoney.com/list,{symbol},f_{page}.html"
            
            response = self.scraper.get(
                url,
                referer="https://guba.eastmoney.com/"
            )
            
            if not response or response.status_code != 200:
                return []
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找帖子列表
            for item in soup.select('.listitem, .articleh'):
                try:
                    # 标题和链接
                    title_elem = item.select_one('.l3 a, .title a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    href = title_elem.get('href', '')
                    
                    # 提取帖子ID
                    post_id_match = re.search(r',(\d+)\.html', href)
                    post_id = post_id_match.group(1) if post_id_match else ''
                    
                    # 作者
                    author_elem = item.select_one('.l4 a, .author a')
                    author = author_elem.get_text(strip=True) if author_elem else ''
                    
                    # 时间
                    time_elem = item.select_one('.l5, .update')
                    pub_time = time_elem.get_text(strip=True) if time_elem else ''
                    
                    # 标准化时间格式
                    if pub_time and len(pub_time) <= 5:  # 如 "12:30"
                        today = datetime.now().strftime('%Y-%m-%d')
                        pub_time = f"{today} {pub_time}"
                    
                    # 阅读数
                    read_elem = item.select_one('.l1, .read')
                    read_count = read_elem.get_text(strip=True) if read_elem else '0'
                    try:
                        read_count = int(read_count.replace(',', ''))
                    except:
                        read_count = 0
                    
                    # 回复数
                    reply_elem = item.select_one('.l2, .reply')
                    reply_count = reply_elem.get_text(strip=True) if reply_elem else '0'
                    try:
                        reply_count = int(reply_count.replace(',', ''))
                    except:
                        reply_count = 0
                    
                    posts.append({
                        'post_id': post_id,
                        'title': title,
                        'author': author,
                        'pub_time': pub_time,
                        'click_count': read_count,
                        'reply_count': reply_count,
                        'content': ''  # 稍后获取
                    })
                    
                except Exception as e:
                    continue
                    
        except ImportError:
            self.logger.warning("BeautifulSoup未安装")
        except Exception as e:
            self.logger.debug(f"解析股吧HTML失败: {e}")
        
        return posts
    
    def _fetch_guba_post_content(self, symbol: str, post_id: str) -> str:
        """获取股吧帖子详细内容"""
        try:
            url = f"https://guba.eastmoney.com/news,{symbol},{post_id}.html"
            
            response = self.scraper.get(
                url,
                referer=f"https://guba.eastmoney.com/list,{symbol}.html"
            )
            
            if response and response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 多种选择器尝试获取正文
                content_selectors = [
                    '.stockcodec',
                    '.post-content',
                    '.newstext',
                    '#zwconbody',
                    '.article-content',
                ]
                
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        # 清理HTML标签
                        content = content_elem.get_text(strip=True)
                        if content and len(content) > 10:
                            return content
            
        except Exception as e:
            self.logger.debug(f"获取帖子内容失败 {post_id}: {e}")
        
        return ''
    
    def _standardize_guba(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化股吧数据"""
        if df.empty:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        # 尝试映射常见字段名
        column_map = {
            '发布时间': 'pub_time',
            '作者': 'author',
            '标题': 'title',
            '内容': 'content',
            '回复数': 'reply_count',
            '点击数': 'like_count',
        }
        
        for old_col, new_col in column_map.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        if 'pub_time' in df.columns:
            df['trade_date'] = pd.to_datetime(df['pub_time']).dt.strftime('%Y-%m-%d')
        
        return df
    
    # ==================== 雪球采集 ====================
    
    def _collect_xueqiu_comments(
        self,
        start_date: str,
        end_date: str,
        ts_codes: Optional[List[str]] = None,
        target_dates: Optional[Dict[str, List[str]]] = None,
        use_two_phase: bool = True
    ) -> pd.DataFrame:
        """
        采集雪球评论（增强版）
        
        采集策略（优先级）：
        1. AkShare stock_comment_xq（如果可用）
        2. 两段式采集：Playwright获取Cookie + HTTP高频请求
        3. 传统Cookie爬虫（需要手动配置Cookie）
        
        Args:
            use_two_phase: 是否使用两段式采集（Playwright自动获取Cookie）
        """
        all_data = []
        
        if ts_codes:
            for ts_code in ts_codes:
                # 策略1：AkShare
                try:
                    df = self._collect_xueqiu_akshare(ts_code)
                    if not df.empty:
                        all_data.append(df)
                        self.logger.info(f"AkShare雪球采集 {ts_code}: {len(df)} 条")
                        continue
                except Exception as e:
                    self.logger.debug(f"AkShare雪球采集 {ts_code} 失败: {e}")
                
                # 策略2：两段式采集
                if use_two_phase:
                    try:
                        dates = target_dates.get(ts_code) if target_dates else None
                        df = self._scrape_xueqiu_two_phase(ts_code, dates)
                        if not df.empty:
                            all_data.append(df)
                            self.logger.info(f"两段式雪球采集 {ts_code}: {len(df)} 条")
                            continue
                    except Exception as e:
                        self.logger.debug(f"两段式雪球采集 {ts_code} 失败: {e}")
                
                # 策略3：传统Cookie爬虫
                try:
                    dates = target_dates.get(ts_code) if target_dates else None
                    df = self._scrape_xueqiu_comments(ts_code, dates)
                    if not df.empty:
                        all_data.append(df)
                        self.logger.info(f"传统雪球爬虫采集 {ts_code}: {len(df)} 条")
                except Exception as e:
                    self.logger.warning(f"雪球爬虫采集 {ts_code} 失败: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        
        return pd.DataFrame()
    
    def _scrape_xueqiu_two_phase(
        self,
        ts_code: str,
        target_dates: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        两段式雪球采集
        
        阶段1：使用Playwright访问雪球首页，自动获取访客Cookie
        阶段2：使用获取的Cookie进行HTTP高频请求采集
        """
        try:
            from ..scraper_base import XueqiuCollector
            
            # 使用专用的雪球采集器
            collector = XueqiuCollector(use_proxy=False, rate_limit=True)
            
            # 确保有Cookie（自动或交互式获取）
            if not collector.ensure_cookies(interactive=False):
                self.logger.warning("无法自动获取雪球Cookie")
                return pd.DataFrame()
            
            # 转换为雪球股票代码格式
            symbol = self._to_xueqiu_symbol(ts_code)
            
            # 获取评论
            posts = collector.get_stock_comments(
                symbol=symbol,
                count=self.config.page_size,
                max_pages=min(self.config.max_pages, 5)
            )
            
            if not posts:
                return pd.DataFrame()
            
            # 转换为DataFrame
            all_data = []
            for post in posts:
                created_at = post.get('created_at', 0)
                pub_time = datetime.fromtimestamp(created_at / 1000).strftime('%Y-%m-%d %H:%M:%S') if created_at else ''
                
                # 日期过滤
                if target_dates:
                    pub_date = pub_time[:10] if pub_time else ''
                    if pub_date and pub_date not in target_dates:
                        continue
                
                record = {
                    'ts_code': ts_code,
                    'pub_time': pub_time,
                    'author': post.get('user', {}).get('screen_name', ''),
                    'title': post.get('title', ''),
                    'content': self._clean_html(post.get('text', '')),
                    'reply_count': post.get('reply_count', 0),
                    'like_count': post.get('like_count', 0),
                    'source': SentimentSource.XUEQIU.value,
                    'url': f"https://xueqiu.com/{post.get('user_id', '')}/{post.get('id', '')}"
                }
                
                record['comment_id'] = hashlib.md5(
                    f"{ts_code}xq{post.get('id', '')}".encode()
                ).hexdigest()
                
                all_data.append(record)
            
            collector.close()
            
            if all_data:
                df = pd.DataFrame(all_data)
                df['trade_date'] = df['pub_time'].str[:10]
                return df
            
        except ImportError:
            self.logger.debug("XueqiuCollector不可用，跳过两段式采集")
        except Exception as e:
            self.logger.warning(f"两段式雪球采集失败 {ts_code}: {e}")
        
        return pd.DataFrame()
    
    def _collect_xueqiu_akshare(self, ts_code: str) -> pd.DataFrame:
        """使用 AkShare 采集雪球数据"""
        try:
            # 雪球股票讨论
            symbol = self._to_xueqiu_symbol(ts_code)
            
            # 尝试雪球热帖
            df = self.ak.stock_hot_follow_xq(symbol=symbol)
            
            if df is not None and not df.empty:
                df['ts_code'] = ts_code
                df['source'] = SentimentSource.XUEQIU.value
                df['comment_id'] = df.apply(
                    lambda x: hashlib.md5(
                        f"{ts_code}xq{str(x.values)[:50]}".encode()
                    ).hexdigest(),
                    axis=1
                )
                return self._standardize_xueqiu(df)
                
        except Exception as e:
            self.logger.debug(f"AkShare 雪球接口不可用: {e}")
        
        return pd.DataFrame()
    
    def _scrape_xueqiu_comments(
        self,
        ts_code: str,
        target_dates: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        爬取雪球评论（需要Cookie）
        
        API: https://xueqiu.com/statuses/stock_timeline.json
        """
        all_data = []
        symbol = self._to_xueqiu_symbol(ts_code)
        
        # 检查是否有雪球Cookie
        cookie_str = self.scraper.cookie_manager.get_cookie_string('xueqiu.com')
        if not cookie_str:
            self.logger.warning("雪球Cookie未配置，跳过爬虫采集")
            return pd.DataFrame()
        
        api_url = "https://xueqiu.com/statuses/stock_timeline.json"
        
        try:
            max_id = None
            
            for page in range(1, min(self.config.max_pages + 1, 6)):  # 雪球限制更多
                params = {
                    'symbol': symbol,
                    'count': self.config.page_size,
                    'source': 'all'
                }
                
                if max_id:
                    params['max_id'] = max_id
                
                headers = {
                    'Cookie': cookie_str,
                    'X-Requested-With': 'XMLHttpRequest',
                }
                
                response = self.scraper.get(
                    api_url,
                    params=params,
                    headers=headers,
                    referer=f"https://xueqiu.com/S/{symbol}"
                )
                
                if not response or response.status_code != 200:
                    if response and response.status_code == 400:
                        self.logger.warning("雪球Cookie可能已失效")
                        self.scraper.cookie_manager.mark_invalid(
                            'xueqiu.com',
                            self.scraper.cookie_manager.get_cookies('xueqiu.com') or {}
                        )
                    break
                
                try:
                    data = response.json()
                    posts = data.get('list', [])
                    
                    if not posts:
                        break
                    
                    for post in posts:
                        created_at = post.get('created_at', 0)
                        pub_time = datetime.fromtimestamp(created_at / 1000).strftime('%Y-%m-%d %H:%M:%S') if created_at else ''
                        
                        record = {
                            'ts_code': ts_code,
                            'pub_time': pub_time,
                            'author': post.get('user', {}).get('screen_name', ''),
                            'title': post.get('title', ''),
                            'content': self._clean_html(post.get('text', '')),
                            'reply_count': post.get('reply_count', 0),
                            'like_count': post.get('like_count', 0),
                            'source': SentimentSource.XUEQIU.value,
                            'url': f"https://xueqiu.com/{post.get('user_id', '')}/{post.get('id', '')}"
                        }
                        
                        # 日期过滤
                        if target_dates:
                            pub_date = pub_time[:10] if pub_time else ''
                            if pub_date and pub_date not in target_dates:
                                continue
                        
                        record['comment_id'] = hashlib.md5(
                            f"{ts_code}xq{post.get('id', '')}".encode()
                        ).hexdigest()
                        
                        all_data.append(record)
                        
                        # 记录最后的ID用于分页
                        max_id = post.get('id')
                    
                except Exception as e:
                    self.logger.debug(f"解析雪球响应失败: {e}")
                    break
                
                time.sleep(self.config.request_interval * 0.5)  # 减少延迟
            
        except Exception as e:
            self.logger.warning(f"雪球爬虫失败 {ts_code}: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            df['trade_date'] = df['pub_time'].str[:10]
            return df
        
        return pd.DataFrame()
    
    def _standardize_xueqiu(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化雪球数据"""
        if df.empty:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        column_map = {
            '发布时间': 'pub_time',
            '作者': 'author',
            '标题': 'title',
            '内容': 'content',
        }
        
        for old_col, new_col in column_map.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        if 'pub_time' in df.columns:
            df['trade_date'] = pd.to_datetime(df['pub_time']).dt.strftime('%Y-%m-%d')
        
        return df
    
    # ==================== 事件驱动回溯 ====================
    
    def _filter_event_dates(
        self,
        volatility_df: pd.DataFrame,
        ts_codes: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        从股价波动表筛选目标日期
        
        Args:
            volatility_df: 必须包含 ts_code, trade_date, pct_chg 列
                          可选包含 volume_ratio, turnover_rate_f 列
        
        Returns:
            {ts_code: [date1, date2, ...]} 字典
        """
        result = {}
        config = self.config.event_filter
        
        # 确保必要的列存在
        required_cols = ['ts_code', 'trade_date', 'pct_chg']
        if not all(col in volatility_df.columns for col in required_cols):
            self.logger.warning(f"波动表缺少必要列: {required_cols}")
            return {}
        
        # 筛选股票
        if ts_codes:
            volatility_df = volatility_df[volatility_df['ts_code'].isin(ts_codes)]
        
        # 筛选条件
        mask = (
            (volatility_df['pct_chg'].abs() >= config.price_change_threshold)
        )
        
        # 成交量放大条件（如果有该列）
        if 'volume_ratio' in volatility_df.columns:
            volume_mask = volatility_df['volume_ratio'] >= config.volume_ratio_threshold
            mask = mask | volume_mask
        
        # 涨跌停条件
        if config.include_limit_up:
            mask = mask | (volatility_df['pct_chg'] >= 9.9)
        if config.include_limit_down:
            mask = mask | (volatility_df['pct_chg'] <= -9.9)
        
        filtered = volatility_df[mask]
        
        # 按股票分组
        for ts_code, group in filtered.groupby('ts_code'):
            dates = group['trade_date'].astype(str).tolist()
            # 标准化日期格式
            dates = [self._standardize_date(d) for d in dates]
            result[ts_code] = dates
        
        return result
    
    # ==================== 工具方法 ====================
    
    def _to_xueqiu_symbol(self, ts_code: str) -> str:
        """转换为雪球股票代码格式（SH600000 / SZ000001）"""
        code, market = ts_code.split('.')
        return f"{market}{code}"
    
    def _clean_html(self, text: str) -> str:
        """清理HTML标签"""
        if not text:
            return ''
        # 简单清理
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _ensure_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保输出包含所有标准字段"""
        for field in self.STANDARD_FIELDS:
            if field not in df.columns:
                df[field] = None
        
        return df[self.STANDARD_FIELDS]


# ==================== 便捷函数 ====================

def get_investor_sentiment(
    ts_codes: List[str],
    start_date: str,
    end_date: str,
    source: str = 'all',
    event_driven: bool = False,
    volatility_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    获取投资者舆情数据（便捷函数）
    
    Args:
        ts_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        source: 数据来源 ('cninfo_interaction', 'guba', 'xueqiu', 'all')
        event_driven: 是否启用事件驱动回溯
        volatility_df: 股价波动表（事件驱动模式必需）
    
    Returns:
        舆情数据 DataFrame
    """
    config = SentimentConfig(event_driven=event_driven)
    collector = InvestorSentimentCollector(config=config)
    
    source_enum = SentimentSource(source) if isinstance(source, str) else source
    
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        ts_codes=ts_codes,
        source=source_enum,
        price_volatility_df=volatility_df
    )


def get_cninfo_interaction(
    ts_codes: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取互动易问答数据（便捷函数）
    
    Args:
        ts_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        互动易 DataFrame
    """
    collector = InvestorSentimentCollector()
    
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    return collector._collect_cninfo_interaction(start_date, end_date, ts_codes)


def get_guba_comments(
    ts_codes: List[str],
    max_pages: int = 10
) -> pd.DataFrame:
    """
    获取股吧评论（便捷函数）
    
    Args:
        ts_codes: 股票代码列表
        max_pages: 每只股票最大页数
    
    Returns:
        股吧评论 DataFrame
    """
    config = SentimentConfig(max_pages=max_pages, event_driven=False)
    collector = InvestorSentimentCollector(config=config)
    
    return collector._collect_guba_comments(
        start_date=datetime.now().strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d'),
        ts_codes=ts_codes
    )


def get_xueqiu_comments(
    ts_codes: List[str],
    max_pages: int = 5
) -> pd.DataFrame:
    """
    获取雪球评论（便捷函数）
    
    需要配置 XUEQIU_COOKIE 环境变量
    
    Args:
        ts_codes: 股票代码列表
        max_pages: 每只股票最大页数
    
    Returns:
        雪球评论 DataFrame
    """
    config = SentimentConfig(max_pages=max_pages, event_driven=False)
    collector = InvestorSentimentCollector(config=config)
    
    return collector._collect_xueqiu_comments(
        start_date=datetime.now().strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d'),
        ts_codes=ts_codes
    )


def get_event_driven_sentiment(
    ts_codes: List[str],
    volatility_df: pd.DataFrame,
    price_threshold: float = 3.0,
    volume_threshold: float = 2.0
) -> pd.DataFrame:
    """
    事件驱动舆情采集（便捷函数）
    
    只采集股价波动显著日期的舆情数据
    
    Args:
        ts_codes: 股票代码列表
        volatility_df: 股价波动表
                      必须包含: ts_code, trade_date, pct_chg
                      可选包含: volume_ratio
        price_threshold: 股价波动阈值（百分比）
        volume_threshold: 成交量放大阈值
    
    Returns:
        舆情数据 DataFrame
    """
    config = SentimentConfig(
        event_driven=True,
        event_filter=EventFilter(
            price_change_threshold=price_threshold,
            volume_ratio_threshold=volume_threshold
        )
    )
    
    collector = InvestorSentimentCollector(config=config)
    
    # 从波动表获取日期范围
    dates = pd.to_datetime(volatility_df['trade_date'])
    start_date = dates.min().strftime('%Y-%m-%d')
    end_date = dates.max().strftime('%Y-%m-%d')
    
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        ts_codes=ts_codes,
        source=SentimentSource.ALL,
        price_volatility_df=volatility_df
    )
