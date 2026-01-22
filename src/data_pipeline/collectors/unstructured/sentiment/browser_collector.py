"""
浏览器模拟舆情采集器

使用 Playwright 无头浏览器绕过反爬虫机制采集：
1. 股吧评论
2. 雪球讨论
3. 东方财富互动问答

特性：
- 完整浏览器指纹伪造
- Cookie自动管理
- 智能请求限速
- 失败自动重试
"""

import os
import re
import time
import json
import random
import hashlib
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

import pandas as pd

from .anti_crawler_config import (
    GUBA_CONFIG, XUEQIU_CONFIG, EASTMONEY_CONFIG,
    HeaderTemplates, STEALTH_JS, get_random_fingerprint,
    CrawlerStrategy
)

logger = logging.getLogger(__name__)


@dataclass
class BrowserCollectorConfig:
    """浏览器采集器配置"""
    headless: bool = True                # 无头模式
    slow_mo: int = 50                    # 操作延迟(ms)
    max_pages: int = 10                  # 最大采集页数
    page_size: int = 20                  # 每页数量
    timeout: int = 30000                 # 超时时间(ms)
    retry_times: int = 3                 # 重试次数
    screenshot_on_error: bool = True     # 错误时截图
    block_media: bool = True             # 屏蔽媒体资源


class BrowserSentimentCollector:
    """
    浏览器模拟舆情采集器
    
    使用 Playwright 采集需要绕过反爬虫的舆情数据
    """
    
    STANDARD_FIELDS = [
        'comment_id', 'ts_code', 'name', 'trade_date', 'pub_time',
        'author', 'title', 'content', 'reply_count', 'like_count',
        'source', 'url'
    ]
    
    def __init__(self, config: Optional[BrowserCollectorConfig] = None):
        self.config = config or BrowserCollectorConfig()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self.logger = logger
        
        # Cookie 存储
        self._cookies_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '..', '..', 
            'data', 'cookies'
        )
        os.makedirs(self._cookies_dir, exist_ok=True)
    
    def _ensure_browser(self):
        """确保浏览器已启动"""
        if self._browser is None:
            try:
                from playwright.sync_api import sync_playwright
                
                self._playwright = sync_playwright().start()
                
                # 获取随机指纹
                fingerprint = get_random_fingerprint()
                
                # 启动浏览器
                self._browser = self._playwright.chromium.launch(
                    headless=self.config.headless,
                    slow_mo=self.config.slow_mo,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--no-sandbox',
                        '--disable-web-security',
                        '--disable-features=IsolateOrigins,site-per-process',
                    ]
                )
                
                # 创建上下文
                self._context = self._browser.new_context(
                    viewport=fingerprint['viewport'],
                    user_agent=fingerprint['user_agent'],
                    locale=fingerprint.get('locale', 'zh-CN'),
                    timezone_id=fingerprint.get('timezone', 'Asia/Shanghai'),
                    color_scheme='light',
                    has_touch=False,
                    is_mobile=False,
                    extra_http_headers={
                        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    }
                )
                
                # 注入反检测脚本
                self._context.add_init_script(STEALTH_JS)
                
                # 创建页面
                self._page = self._context.new_page()
                
                # 屏蔽媒体资源
                if self.config.block_media:
                    self._page.route(
                        "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,mp4,mp3,ico}",
                        lambda route: route.abort()
                    )
                
                self.logger.info(f"Playwright 浏览器已启动 (headless={self.config.headless})")
                
            except ImportError:
                raise ImportError(
                    "请安装 playwright: pip install playwright && playwright install chromium"
                )
    
    def _load_cookies(self, domain: str) -> bool:
        """加载保存的Cookie"""
        cookie_file = os.path.join(self._cookies_dir, f"{domain.replace('.', '_')}_cookies.json")
        
        if os.path.exists(cookie_file):
            try:
                with open(cookie_file, 'r', encoding='utf-8') as f:
                    cookies = json.load(f)
                
                if cookies and self._context:
                    self._context.add_cookies(cookies)
                    self.logger.info(f"已加载 {domain} 的 {len(cookies)} 个Cookie")
                    return True
            except Exception as e:
                self.logger.warning(f"加载Cookie失败: {e}")
        
        return False
    
    def _save_cookies(self, domain: str):
        """保存当前Cookie"""
        if self._context:
            try:
                cookies = self._context.cookies()
                # 过滤出指定域名的Cookie
                domain_cookies = [c for c in cookies if domain in c.get('domain', '')]
                
                if domain_cookies:
                    cookie_file = os.path.join(
                        self._cookies_dir, 
                        f"{domain.replace('.', '_')}_cookies.json"
                    )
                    with open(cookie_file, 'w', encoding='utf-8') as f:
                        json.dump(domain_cookies, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"已保存 {domain} 的 {len(domain_cookies)} 个Cookie")
            except Exception as e:
                self.logger.warning(f"保存Cookie失败: {e}")
    
    def close(self):
        """关闭浏览器"""
        if self._page:
            self._page.close()
            self._page = None
        if self._context:
            self._context.close()
            self._context = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # ==================== 股吧采集 ====================
    
    def collect_guba(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_pages: Optional[int] = None,
        use_historical: bool = False
    ) -> pd.DataFrame:
        """
        采集股吧评论
        
        Args:
            ts_code: 股票代码 (如 600519.SH)
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            max_pages: 最大页数
            use_historical: 是否启用历史数据回溯（利用静态分页）
        
        Returns:
            评论DataFrame
        """
        self._ensure_browser()
        all_data = []
        
        symbol = ts_code.split('.')[0]
        max_pages = max_pages or self.config.max_pages
        
        # 加载Cookie
        self._load_cookies('eastmoney.com')
        
        try:
            # 如果启用历史回溯且指定了开始日期
            if use_historical and start_date:
                self.logger.info(f"启用历史回溯模式，定位 {start_date} 对应的页码...")
                start_page = self._binary_search_date_page(symbol, start_date)
                self.logger.info(f"定位完成：从第 {start_page} 页开始采集")
            else:
                start_page = 1
            
            for page in range(start_page, start_page + max_pages):
                try:
                    posts = self._fetch_guba_page(symbol, page)
                    
                    if not posts:
                        self.logger.info(f"股吧 {ts_code} 第{page}页无数据，停止")
                        break
                    
                    # 日期过滤
                    filtered_posts = []
                    for post in posts:
                        pub_date = post.get('pub_time', '')[:10]
                        
                        # 如果有结束日期且当前帖子超过结束日期，停止采集
                        if end_date and pub_date > end_date:
                            continue
                        
                        # 如果有开始日期且当前帖子早于开始日期，停止采集
                        if start_date and pub_date < start_date:
                            self.logger.info(f"已到达开始日期边界 ({start_date})，停止采集")
                            break
                        
                        post['ts_code'] = ts_code
                        post['source'] = 'guba'
                        post['comment_id'] = hashlib.md5(
                            f"guba_{ts_code}_{post.get('pub_time', '')}_{post.get('title', '')[:30]}".encode()
                        ).hexdigest()
                        filtered_posts.append(post)
                    
                    all_data.extend(filtered_posts)
                    
                    self.logger.info(f"股吧 {ts_code} 第{page}页: {len(filtered_posts)} 条 (原始{len(posts)}条)")
                    
                    # 如果这一页有帖子早于开始日期，停止采集
                    if start_date and any(p.get('pub_time', '')[:10] < start_date for p in posts):
                        break
                    
                    # 随机延迟
                    time.sleep(GUBA_CONFIG.get_delay())
                    
                except Exception as e:
                    self.logger.warning(f"股吧第{page}页采集失败: {e}")
                    if self.config.screenshot_on_error and self._page:
                        self._page.screenshot(path=f"error_guba_{symbol}_{page}.png")
                    continue
            
            # 保存Cookie
            self._save_cookies('eastmoney.com')
            
        except Exception as e:
            self.logger.error(f"股吧采集失败 {ts_code}: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            df['trade_date'] = df['pub_time'].str[:10]
            return self._ensure_fields(df)
        
        return pd.DataFrame(columns=self.STANDARD_FIELDS)
    
    def _fetch_guba_page(self, symbol: str, page: int) -> List[Dict]:
        """获取股吧单页数据"""
        posts = []
        
        # 构建URL
        if page == 1:
            url = f"https://guba.eastmoney.com/list,{symbol}.html"
        else:
            url = f"https://guba.eastmoney.com/list,{symbol},f_{page}.html"
        
        try:
            # 访问页面
            self._page.goto(url, timeout=self.config.timeout, wait_until='domcontentloaded')
            
            # 等待内容加载
            try:
                self._page.wait_for_selector('tr.listitem', timeout=10000)
            except:
                pass
            
            # 解析帖子列表 - 股吧使用表格布局，每行是一个帖子
            items = self._page.query_selector_all('tr.listitem')
            
            for item in items:
                try:
                    # 表格列：阅读数 | 评论数 | 标题 | 作者 | 最后发表
                    tds = item.query_selector_all('td')
                    if len(tds) < 5:
                        continue
                    
                    # 阅读数 (第1列)
                    read_count = 0
                    try:
                        read_text = tds[0].inner_text().strip()
                        # 处理 "3.3万" 这种格式
                        if '万' in read_text:
                            read_count = int(float(read_text.replace('万', '')) * 10000)
                        else:
                            read_count = int(read_text.replace(',', ''))
                    except:
                        pass
                    
                    # 评论数 (第2列)
                    reply_count = 0
                    try:
                        reply_text = tds[1].inner_text().strip()
                        reply_count = int(reply_text.replace(',', ''))
                    except:
                        pass
                    
                    # 标题和链接 (第3列)
                    title_td = tds[2]
                    title_elem = title_td.query_selector('a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.inner_text().strip()
                    href = title_elem.get_attribute('href') or ''
                    
                    # 提取帖子ID
                    post_id_match = re.search(r',(\d+)\.html', href)
                    post_id = post_id_match.group(1) if post_id_match else ''
                    
                    # 作者 (第4列)
                    author = ''
                    try:
                        author_elem = tds[3].query_selector('a')
                        if author_elem:
                            author = author_elem.inner_text().strip()
                    except:
                        pass
                    
                    # 时间 (第5列)
                    pub_time = ''
                    try:
                        pub_time = tds[4].inner_text().strip()
                        # 标准化时间格式
                        if pub_time:
                            pub_time = self._normalize_guba_time(pub_time)
                    except:
                        pass
                    
                    post = {
                        'title': title,
                        'content': '',  # 内容需要进入详情页获取
                        'author': author,
                        'pub_time': pub_time,
                        'reply_count': reply_count,
                        'like_count': read_count,
                        'url': f"https://guba.eastmoney.com{href}" if href.startswith('/') else href,
                        'post_id': post_id,
                    }
                    posts.append(post)
                    
                except Exception as e:
                    self.logger.debug(f"解析单条帖子失败: {e}")
                    continue
            
        except Exception as e:
            self.logger.debug(f"股吧页面解析失败: {e}")
        
        return posts
    
    def _normalize_guba_time(self, time_str: str) -> str:
        """标准化股吧时间格式"""
        now = datetime.now()
        
        # 处理 "今天 12:30" 格式
        if '今天' in time_str:
            time_part = time_str.replace('今天', '').strip()
            return f"{now.strftime('%Y-%m-%d')} {time_part}"
        
        # 处理 "昨天 12:30" 格式
        if '昨天' in time_str:
            yesterday = now - timedelta(days=1)
            time_part = time_str.replace('昨天', '').strip()
            return f"{yesterday.strftime('%Y-%m-%d')} {time_part}"
        
        # 处理 "12:30" 格式 (今天)
        if re.match(r'^\d{1,2}:\d{2}$', time_str):
            return f"{now.strftime('%Y-%m-%d')} {time_str}"
        
        # 处理 "01-15 12:30" 格式
        if re.match(r'^\d{2}-\d{2}\s+\d{2}:\d{2}$', time_str):
            month_day = time_str.split()[0]  # "01-15"
            time_part = time_str.split()[1]  # "12:30"
            month, day = map(int, month_day.split('-'))
            
            # 智能判断年份：如果月-日比当前日期晚，说明是去年
            current_month_day = (now.month, now.day)
            post_month_day = (month, day)
            
            if post_month_day > current_month_day:
                # 帖子的月-日比今天晚，肯定是去年的
                year = now.year - 1
            else:
                # 帖子的月-日比今天早或相等，是今年的
                year = now.year
            
            return f"{year}-{month:02d}-{day:02d} {time_part}"
        
        # 处理 "2024-01-15 12:30" 格式
        if re.match(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}', time_str):
            return time_str
        
        return time_str
    
    def _get_page_date(self, symbol: str, page: int) -> Optional[str]:
        """
        获取指定页最早的帖子日期
        
        Args:
            symbol: 股票代码
            page: 页码
        
        Returns:
            日期字符串 (YYYY-MM-DD)，失败返回None
        """
        self._ensure_browser()  # 确保浏览器已启动
        
        try:
            posts = self._fetch_guba_page(symbol, page)
            if not posts:
                return None
            
            # 获取这一页最早的日期（通常是列表最后一条）
            dates = [p.get('pub_time', '')[:10] for p in posts if p.get('pub_time')]
            if dates:
                return min(dates)  # 返回最早的日期
            
        except Exception as e:
            self.logger.debug(f"获取第{page}页日期失败: {e}")
        
        return None
    
    def _binary_search_date_page(
        self, 
        symbol: str, 
        target_date: str,
        min_page: int = 1,
        max_page: int = 10000,
        tolerance_days: int = 7
    ) -> int:
        """
        二分查找定位目标日期对应的页码
        
        利用股吧静态分页结构，通过二分法快速定位历史数据所在页码
        
        Args:
            symbol: 股票代码
            target_date: 目标日期 (YYYY-MM-DD)
            min_page: 最小页码
            max_page: 最大页码（猜测的最大值）
            tolerance_days: 容忍的日期偏差（天）
        
        Returns:
            目标日期对应的页码
        """
        self.logger.info(f"开始二分查找 {target_date} 对应的页码...")
        
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        
        # 首先检查边界
        first_page_date = self._get_page_date(symbol, min_page)
        if first_page_date:
            first_dt = datetime.strptime(first_page_date, '%Y-%m-%d')
            if first_dt <= target_dt:
                self.logger.info(f"目标日期在第1页或更晚，从第1页开始")
                return min_page
        
        # 检查最大页是否存在数据
        last_page_date = self._get_page_date(symbol, max_page)
        if not last_page_date:
            # 如果max_page没有数据，先找到有数据的最大页
            self.logger.info(f"第{max_page}页无数据，寻找有效最大页...")
            max_page = max_page // 2
            while max_page > min_page:
                if self._get_page_date(symbol, max_page):
                    break
                max_page = max_page // 2
            self.logger.info(f"有效最大页: {max_page}")
        
        # 二分查找
        left, right = min_page, max_page
        best_page = min_page
        
        iteration = 0
        max_iterations = 20  # 防止无限循环
        
        while left <= right and iteration < max_iterations:
            mid = (left + right) // 2
            iteration += 1
            
            mid_date = self._get_page_date(symbol, mid)
            
            if not mid_date:
                self.logger.debug(f"第{mid}页无数据，调整范围")
                right = mid - 1
                continue
            
            mid_dt = datetime.strptime(mid_date, '%Y-%m-%d')
            diff_days = (target_dt - mid_dt).days
            
            self.logger.info(f"迭代{iteration}: 第{mid}页日期={mid_date}, 目标={target_date}, 偏差={diff_days}天")
            
            # 如果偏差在容忍范围内，找到了
            if abs(diff_days) <= tolerance_days:
                self.logger.info(f"找到匹配页码: 第{mid}页 (偏差{diff_days}天)")
                return mid
            
            if mid_dt > target_dt:
                # 中间页日期太新，目标在更后面的页
                left = mid + 1
                best_page = mid
            else:
                # 中间页日期太旧，目标在更前面的页
                right = mid - 1
            
            # 添加小延迟避免请求过快
            time.sleep(0.5)
        
        self.logger.info(f"二分查找完成，返回第{best_page}页")
        return best_page
    
    # ==================== 雪球采集 ====================
    
    def collect_xueqiu(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_pages: Optional[int] = None
    ) -> pd.DataFrame:
        """
        采集雪球讨论
        
        Args:
            ts_code: 股票代码 (如 600519.SH)
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            max_pages: 最大页数
        
        Returns:
            讨论DataFrame
        """
        self._ensure_browser()
        all_data = []
        
        # 转换为雪球格式的股票代码
        xueqiu_symbol = self._to_xueqiu_symbol(ts_code)
        max_pages = max_pages or self.config.max_pages
        
        # 加载Cookie
        self._load_cookies('xueqiu.com')
        
        try:
            # 首先访问主页获取Cookie
            self._page.goto('https://xueqiu.com/', timeout=self.config.timeout)
            time.sleep(2)
            
            # 访问股票页面
            stock_url = f"https://xueqiu.com/S/{xueqiu_symbol}"
            self._page.goto(stock_url, timeout=self.config.timeout)
            time.sleep(2)
            
            for page in range(1, max_pages + 1):
                try:
                    posts = self._fetch_xueqiu_page(xueqiu_symbol, page)
                    
                    if not posts:
                        self.logger.info(f"雪球 {ts_code} 第{page}页无数据，停止")
                        break
                    
                    # 日期过滤
                    for post in posts:
                        pub_date = post.get('pub_time', '')[:10]
                        if start_date and pub_date < start_date:
                            continue
                        if end_date and pub_date > end_date:
                            continue
                        
                        post['ts_code'] = ts_code
                        post['source'] = 'xueqiu'
                        post['comment_id'] = hashlib.md5(
                            f"xueqiu_{ts_code}_{post.get('pub_time', '')}_{post.get('title', '')[:30]}".encode()
                        ).hexdigest()
                        all_data.append(post)
                    
                    self.logger.info(f"雪球 {ts_code} 第{page}页: {len(posts)} 条")
                    
                    # 随机延迟
                    time.sleep(XUEQIU_CONFIG.get_delay())
                    
                    # 滚动加载更多
                    self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.warning(f"雪球第{page}页采集失败: {e}")
                    if self.config.screenshot_on_error and self._page:
                        self._page.screenshot(path=f"error_xueqiu_{xueqiu_symbol}_{page}.png")
                    continue
            
            # 保存Cookie
            self._save_cookies('xueqiu.com')
            
        except Exception as e:
            self.logger.error(f"雪球采集失败 {ts_code}: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            df['trade_date'] = df['pub_time'].str[:10]
            return self._ensure_fields(df)
        
        return pd.DataFrame(columns=self.STANDARD_FIELDS)
    
    def _to_xueqiu_symbol(self, ts_code: str) -> str:
        """转换为雪球股票代码格式"""
        code, market = ts_code.split('.')
        if market == 'SH':
            return f"SH{code}"
        elif market == 'SZ':
            return f"SZ{code}"
        else:
            return ts_code
    
    def _fetch_xueqiu_page(self, symbol: str, page: int) -> List[Dict]:
        """获取雪球单页讨论数据"""
        posts = []
        
        try:
            # 雪球API地址
            full_url = f"https://xueqiu.com/statuses/stock_timeline.json?symbol={symbol}&count=20&source=all&page={page}"
            
            self.logger.debug(f"雪球API URL: {full_url}")
            
            # 使用 fetch API 获取数据
            response = self._page.evaluate(f"""
                async () => {{
                    try {{
                        const response = await fetch('{full_url}', {{
                            credentials: 'include',
                            headers: {{
                                'Accept': 'application/json',
                                'X-Requested-With': 'XMLHttpRequest',
                            }}
                        }});
                        if (!response.ok) {{
                            return {{ error: 'HTTP ' + response.status }};
                        }}
                        return await response.json();
                    }} catch(e) {{
                        return {{ error: e.message }};
                    }}
                }}
            """)
            
            # 检查错误
            if response and 'error' in response:
                self.logger.debug(f"雪球API返回错误: {response['error']}")
                # 可能需要登录
                if '400' in str(response['error']) or '401' in str(response['error']):
                    self.logger.warning("雪球需要登录才能获取数据，请运行 get_cookies.py 获取Cookie")
            
            if response and 'list' in response:
                for item in response['list']:
                    created_at = item.get('created_at', 0)
                    if created_at:
                        pub_time = datetime.fromtimestamp(created_at / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        pub_time = ''
                    
                    # 清理HTML标签
                    text = item.get('text', '')
                    text = re.sub(r'<[^>]+>', '', text)
                    
                    post = {
                        'title': item.get('title', '') or text[:50],
                        'content': text,
                        'author': item.get('user', {}).get('screen_name', ''),
                        'pub_time': pub_time,
                        'reply_count': item.get('reply_count', 0),
                        'like_count': item.get('like_count', 0),
                        'url': f"https://xueqiu.com{item.get('target', '')}",
                    }
                    posts.append(post)
                    
                self.logger.debug(f"雪球API返回 {len(posts)} 条数据")
            
        except Exception as e:
            self.logger.debug(f"雪球API获取失败: {e}")
            
            # 备选：解析页面
            try:
                items = self._page.query_selector_all('.timeline__item, .status-content')
                
                for item in items:
                    try:
                        # 内容
                        content_elem = item.query_selector('.status-content, .detail')
                        content = content_elem.inner_text().strip() if content_elem else ''
                        
                        # 作者
                        author_elem = item.query_selector('.status-name, .user-name')
                        author = author_elem.inner_text().strip() if author_elem else ''
                        
                        # 时间
                        time_elem = item.query_selector('.status-time, .time')
                        pub_time = time_elem.inner_text().strip() if time_elem else ''
                        
                        if content:
                            post = {
                                'title': content[:50],
                                'content': content,
                                'author': author,
                                'pub_time': pub_time,
                                'reply_count': 0,
                                'like_count': 0,
                                'url': '',
                            }
                            posts.append(post)
                            
                    except Exception:
                        continue
                        
            except Exception as e:
                self.logger.debug(f"雪球页面解析失败: {e}")
        
        return posts
    
    # ==================== 互动易采集 ====================
    
    def collect_cninfo(
        self,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_pages: Optional[int] = None
    ) -> pd.DataFrame:
        """
        采集互动易问答
        
        Args:
            ts_code: 股票代码 (可选，不传则采集全市场)
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            max_pages: 最大页数
        
        Returns:
            问答DataFrame
        """
        self._ensure_browser()
        all_data = []
        
        max_pages = max_pages or self.config.max_pages
        symbol = ts_code.split('.')[0] if ts_code else None
        
        try:
            # 访问互动易首页
            self._page.goto('http://irm.cninfo.com.cn/', timeout=self.config.timeout)
            time.sleep(2)
            
            for page in range(1, max_pages + 1):
                try:
                    items = self._fetch_cninfo_page(symbol, page)
                    
                    if not items:
                        self.logger.info(f"互动易 第{page}页无数据，停止")
                        break
                    
                    # 日期过滤
                    for item in items:
                        q_date = item.get('q_date', '')[:10]
                        if start_date and q_date < start_date:
                            continue
                        if end_date and q_date > end_date:
                            continue
                        
                        item['source'] = 'cninfo_interaction'
                        item['comment_id'] = hashlib.md5(
                            f"cninfo_{item.get('ts_code', '')}_{item.get('q_date', '')}_{item.get('question', '')[:30]}".encode()
                        ).hexdigest()
                        all_data.append(item)
                    
                    self.logger.info(f"互动易 第{page}页: {len(items)} 条")
                    
                    # 随机延迟
                    time.sleep(random.uniform(0.5, 1.5))
                    
                except Exception as e:
                    self.logger.warning(f"互动易第{page}页采集失败: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"互动易采集失败: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            # 标准化字段
            df = df.rename(columns={
                'q_date': 'pub_time',
                'a_date': 'trade_date',
                'question': 'content',
                'answer': 'title',
            })
            df['author'] = '投资者'
            df['reply_count'] = 1
            df['like_count'] = 0
            df['url'] = ''
            return self._ensure_fields(df)
        
        return pd.DataFrame(columns=self.STANDARD_FIELDS)
    
    def _fetch_cninfo_page(self, symbol: Optional[str], page: int) -> List[Dict]:
        """获取互动易单页数据"""
        items = []
        
        try:
            # 使用requests直接访问API（互动易API不需要浏览器）
            import requests
            
            if symbol:
                api_url = f"http://irm.cninfo.com.cn/ircs/interaction/query"
                params = {
                    'stockCode': symbol,
                    'pageNum': page,
                    'pageSize': 30,
                }
            else:
                api_url = "http://irm.cninfo.com.cn/ircs/interaction/lastRaskList.do"
                params = {
                    'pageNum': page,
                    'pageSize': 30,
                }
            
            self.logger.debug(f"互动易API URL: {api_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Referer': 'http://irm.cninfo.com.cn/',
            }
            
            response = requests.get(api_url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    data_list = data.get('results', []) or data.get('data', []) or data.get('list', [])
                    
                    self.logger.debug(f"互动易API返回 {len(data_list)} 条数据")
                    
                    for item in data_list:
                        stock_code = item.get('stockCode', '') or item.get('secCode', '')
                        
                        # 转换为ts_code
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
                            'q_date': item.get('intDate', '') or item.get('questionDate', ''),
                            'a_date': item.get('replyDate', '') or item.get('answerDate', ''),
                            'question': item.get('question', ''),
                            'answer': item.get('reply', '') or item.get('answer', ''),
                        }
                        items.append(record)
                except Exception as e:
                    self.logger.debug(f"互动易API响应解析失败: {e}")
            else:
                self.logger.debug(f"互动易API返回状态码: {response.status_code}")
            
        except Exception as e:
            self.logger.debug(f"互动易API获取失败: {e}")
        
        return items
    
    # ==================== 工具方法 ====================
    
    def _ensure_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保DataFrame包含所有标准字段"""
        for col in self.STANDARD_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.STANDARD_FIELDS]
    
    def collect_all(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
        sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        批量采集多只股票的舆情数据
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            sources: 数据源列表 ['guba', 'xueqiu', 'cninfo']
        
        Returns:
            合并的舆情DataFrame
        """
        if sources is None:
            sources = ['guba', 'xueqiu']
        
        all_data = []
        
        try:
            for ts_code in ts_codes:
                self.logger.info(f"采集 {ts_code} 舆情数据...")
                
                if 'guba' in sources:
                    try:
                        df = self.collect_guba(ts_code, start_date, end_date, max_pages=5)
                        if not df.empty:
                            all_data.append(df)
                            self.logger.info(f"  股吧: {len(df)} 条")
                    except Exception as e:
                        self.logger.warning(f"  股吧采集失败: {e}")
                
                if 'xueqiu' in sources:
                    try:
                        df = self.collect_xueqiu(ts_code, start_date, end_date, max_pages=5)
                        if not df.empty:
                            all_data.append(df)
                            self.logger.info(f"  雪球: {len(df)} 条")
                    except Exception as e:
                        self.logger.warning(f"  雪球采集失败: {e}")
                
                # 股票间延迟
                time.sleep(random.uniform(2, 4))
            
            # 采集全市场互动易
            if 'cninfo' in sources:
                try:
                    df = self.collect_cninfo(
                        start_date=start_date, 
                        end_date=end_date, 
                        max_pages=10
                    )
                    if not df.empty:
                        all_data.append(df)
                        self.logger.info(f"互动易: {len(df)} 条")
                except Exception as e:
                    self.logger.warning(f"互动易采集失败: {e}")
        
        finally:
            self.close()
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.drop_duplicates(subset=['comment_id'], keep='first')
            return result
        
        return pd.DataFrame(columns=self.STANDARD_FIELDS)


def collect_sentiment_with_browser(
    ts_codes: List[str],
    start_date: str,
    end_date: str,
    sources: Optional[List[str]] = None,
    headless: bool = True
) -> pd.DataFrame:
    """
    便捷函数：使用浏览器采集舆情数据
    
    Args:
        ts_codes: 股票代码列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        sources: 数据源列表 ['guba', 'xueqiu', 'cninfo']
        headless: 是否无头模式
    
    Returns:
        舆情数据DataFrame
    """
    config = BrowserCollectorConfig(headless=headless)
    
    with BrowserSentimentCollector(config) as collector:
        return collector.collect_all(ts_codes, start_date, end_date, sources)
