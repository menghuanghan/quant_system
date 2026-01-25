"""
增强型爬虫基类模块

提供高级爬虫功能作为工具类供 Collector 调用：
- 代理池 (Proxy Pool) 集成
- User-Agent 随机轮换
- 指数退避重试 (Exponential Backoff)
- Cookie 轮询管理
- 浏览器模拟（Playwright/Selenium 预留接口）
"""

import os
import time
import random
import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from .proxy_pool import get_proxy_pool, ProxyPool
from .rate_limiter import get_rate_limiter, RateLimiter

load_dotenv()

logger = logging.getLogger(__name__)


# ============== User-Agent 管理 ==============

class UserAgentManager:
    """
    User-Agent 管理器
    
    支持：
    - 静态 UA 列表随机/轮询
    - fake-useragent 动态生成
    """
    
    # 预置的现代浏览器 User-Agent 列表
    DEFAULT_USER_AGENTS = [
        # Chrome on Windows 11
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        # Chrome on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        # Firefox on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0",
        # Edge
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
        # Safari
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    ]
    
    def __init__(
        self,
        user_agents: Optional[List[str]] = None,
        use_fake_ua: bool = True
    ):
        """
        Args:
            user_agents: 自定义 UA 列表
            use_fake_ua: 是否尝试使用 fake-useragent 库
        """
        self._user_agents = user_agents or self.DEFAULT_USER_AGENTS.copy()
        self._index = 0
        self._lock = threading.Lock()
        self._fake_ua = None
        
        # 尝试加载 fake-useragent
        if use_fake_ua:
            try:
                from fake_useragent import UserAgent
                self._fake_ua = UserAgent(browsers=['chrome', 'firefox', 'edge'])
                logger.info("已启用 fake-useragent 动态 UA 生成")
            except ImportError:
                logger.debug("fake-useragent 未安装，使用静态 UA 列表")
            except Exception as e:
                logger.warning(f"fake-useragent 初始化失败: {e}")
    
    def get_random(self) -> str:
        """随机获取 User-Agent"""
        if self._fake_ua:
            try:
                return self._fake_ua.random
            except:
                pass
        return random.choice(self._user_agents)
    
    def get_next(self) -> str:
        """轮询获取 User-Agent"""
        with self._lock:
            ua = self._user_agents[self._index % len(self._user_agents)]
            self._index += 1
            return ua
    
    def get_chrome(self) -> str:
        """获取 Chrome User-Agent"""
        if self._fake_ua:
            try:
                return self._fake_ua.chrome
            except:
                pass
        chrome_uas = [ua for ua in self._user_agents if 'Chrome' in ua and 'Edg' not in ua]
        return random.choice(chrome_uas) if chrome_uas else self.get_random()
    
    def add_custom(self, ua: str):
        """添加自定义 UA"""
        if ua not in self._user_agents:
            self._user_agents.append(ua)


# ============== Cookie 管理 ==============

@dataclass
class CookieEntry:
    """Cookie 条目"""
    cookies: Dict[str, str]     # Cookie 键值对
    domain: str                  # 域名
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    use_count: int = 0
    is_valid: bool = True
    expire_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expire_at:
            return time.time() > self.expire_at
        return False
    
    def mark_used(self):
        """标记为已使用"""
        self.use_count += 1
        self.last_used = time.time()
    
    def mark_invalid(self):
        """标记为无效"""
        self.is_valid = False


class CookieManager:
    """
    Cookie 轮询管理器
    
    用于管理多个账号的 Cookie，支持雪球等需要登录的站点
    """
    
    def __init__(self):
        self._cookies: Dict[str, List[CookieEntry]] = {}  # domain -> cookie list
        self._lock = threading.Lock()
        self._domain_index: Dict[str, int] = {}  # 每个域名的轮询索引
    
    def add_cookies(
        self,
        domain: str,
        cookies: Union[Dict[str, str], str],
        expire_hours: Optional[float] = None
    ):
        """
        添加 Cookie
        
        Args:
            domain: 域名（如 xueqiu.com）
            cookies: Cookie 字典或字符串（如 "key1=val1; key2=val2"）
            expire_hours: 过期时间（小时）
        """
        # 解析 Cookie 字符串
        if isinstance(cookies, str):
            cookies = self._parse_cookie_string(cookies)
        
        if not cookies:
            return
        
        entry = CookieEntry(
            cookies=cookies,
            domain=domain,
            expire_at=time.time() + expire_hours * 3600 if expire_hours else None
        )
        
        with self._lock:
            if domain not in self._cookies:
                self._cookies[domain] = []
                self._domain_index[domain] = 0
            
            # 避免重复
            existing_keys = [frozenset(c.cookies.items()) for c in self._cookies[domain]]
            if frozenset(cookies.items()) not in existing_keys:
                self._cookies[domain].append(entry)
                logger.info(f"添加 Cookie: {domain}, 当前共 {len(self._cookies[domain])} 个")
    
    def _parse_cookie_string(self, cookie_str: str) -> Dict[str, str]:
        """解析 Cookie 字符串"""
        cookies = {}
        for item in cookie_str.split(';'):
            item = item.strip()
            if '=' in item:
                key, value = item.split('=', 1)
                cookies[key.strip()] = value.strip()
        return cookies
    
    def get_cookies(self, domain: str) -> Optional[Dict[str, str]]:
        """
        获取指定域名的 Cookie（轮询方式）
        
        Args:
            domain: 域名
        
        Returns:
            Cookie 字典，无可用 Cookie 时返回 None
        """
        with self._lock:
            if domain not in self._cookies:
                return None
            
            # 获取有效的 Cookie 列表
            valid_cookies = [
                c for c in self._cookies[domain]
                if c.is_valid and not c.is_expired()
            ]
            
            if not valid_cookies:
                return None
            
            # 轮询选择
            idx = self._domain_index.get(domain, 0) % len(valid_cookies)
            entry = valid_cookies[idx]
            entry.mark_used()
            self._domain_index[domain] = idx + 1
            
            return entry.cookies
    
    def get_cookies_for_requests(self, domain: str) -> Optional[Dict[str, str]]:
        """获取用于 requests 库的 Cookie 字典"""
        return self.get_cookies(domain)
    
    def get_cookie_string(self, domain: str) -> Optional[str]:
        """获取 Cookie 字符串格式"""
        cookies = self.get_cookies(domain)
        if cookies:
            return '; '.join(f"{k}={v}" for k, v in cookies.items())
        return None
    
    def mark_invalid(self, domain: str, cookies: Dict[str, str]):
        """标记某个 Cookie 为无效"""
        with self._lock:
            if domain not in self._cookies:
                return
            
            cookie_key = frozenset(cookies.items())
            for entry in self._cookies[domain]:
                if frozenset(entry.cookies.items()) == cookie_key:
                    entry.mark_invalid()
                    logger.warning(f"Cookie 已标记为无效: {domain}")
                    break
    
    def remove_expired(self):
        """清理过期的 Cookie"""
        with self._lock:
            for domain in list(self._cookies.keys()):
                original_len = len(self._cookies[domain])
                self._cookies[domain] = [
                    c for c in self._cookies[domain]
                    if c.is_valid and not c.is_expired()
                ]
                removed = original_len - len(self._cookies[domain])
                if removed > 0:
                    logger.info(f"清理 {domain} 的 {removed} 个过期 Cookie")
    
    def load_from_env(self):
        """
        从环境变量加载 Cookie
        
        支持的环境变量格式：
        - XUEQIU_COOKIE: 雪球 Cookie
        - GUBA_COOKIE: 股吧 Cookie
        - EASTMONEY_COOKIE: 东方财富 Cookie
        """
        # 雪球 Cookie
        xueqiu_cookie = os.getenv('XUEQIU_COOKIE')
        if xueqiu_cookie:
            self.add_cookies('xueqiu.com', xueqiu_cookie, expire_hours=24)
            logger.info("从环境变量加载雪球 Cookie")
        
        # 股吧 Cookie
        guba_cookie = os.getenv('GUBA_COOKIE')
        if guba_cookie:
            self.add_cookies('guba.eastmoney.com', guba_cookie, expire_hours=24)
            logger.info("从环境变量加载股吧 Cookie")
        
        # 东方财富 Cookie
        eastmoney_cookie = os.getenv('EASTMONEY_COOKIE')
        if eastmoney_cookie:
            self.add_cookies('eastmoney.com', eastmoney_cookie, expire_hours=24)
            logger.info("从环境变量加载东方财富 Cookie")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {}
        with self._lock:
            for domain, entries in self._cookies.items():
                valid = sum(1 for e in entries if e.is_valid and not e.is_expired())
                stats[domain] = {
                    'total': len(entries),
                    'valid': valid,
                    'use_counts': [e.use_count for e in entries]
                }
        return stats


# ============== 指数退避重试装饰器 ==============

def exponential_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (
        requests.exceptions.RequestException,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    ),
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)
):
    """
    指数退避重试装饰器
    
    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        exponential_base: 指数基数
        jitter: 是否添加随机抖动
        retryable_exceptions: 可重试的异常类型
        retryable_status_codes: 可重试的 HTTP 状态码
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # 检查返回的 Response 对象
                    if isinstance(result, requests.Response):
                        if result.status_code in retryable_status_codes:
                            raise RetryableHTTPError(
                                f"HTTP {result.status_code}",
                                status_code=result.status_code
                            )
                    
                    return result
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"达到最大重试次数 {max_retries}: {e}")
                        raise
                    
                    # 计算延迟时间
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    # 添加随机抖动
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"请求失败 ({e})，{delay:.1f}秒后重试 "
                        f"({attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    
                except RetryableHTTPError as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"达到最大重试次数 {max_retries}: {e}")
                        raise
                    
                    # 对于 429 等状态码使用更长的延迟
                    if e.status_code == 429:
                        delay = min(
                            base_delay * (exponential_base ** (attempt + 2)),
                            max_delay
                        )
                    else:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                    
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"HTTP 错误 {e.status_code}，{delay:.1f}秒后重试 "
                        f"({attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
            
            return None
        
        return wrapper
    return decorator


class RetryableHTTPError(Exception):
    """可重试的 HTTP 错误"""
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


# ============== 浏览器模拟基类（预留接口） ==============

class BrowserDriver(ABC):
    """浏览器驱动抽象基类"""
    
    @abstractmethod
    def get(self, url: str) -> str:
        """访问 URL 并返回页面内容"""
        pass
    
    @abstractmethod
    def get_with_wait(self, url: str, wait_selector: str, timeout: int = 10) -> str:
        """访问 URL 并等待元素出现"""
        pass
    
    @abstractmethod
    def close(self):
        """关闭浏览器"""
        pass


class PlaywrightDriver(BrowserDriver):
    """
    Playwright 浏览器驱动（增强版）
    
    支持：
    - 无头/有头模式切换
    - 指纹伪造（viewport, timezone, webgl, 等）
    - 自动Cookie捕获
    - 代理池集成
    - 登录态保持
    
    需要安装：pip install playwright && playwright install
    """
    
    # 常用设备指纹配置
    FINGERPRINTS = [
        {
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "locale": "zh-CN",
            "timezone_id": "Asia/Shanghai",
        },
        {
            "viewport": {"width": 1440, "height": 900},
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "locale": "zh-CN",
            "timezone_id": "Asia/Shanghai",
        },
        {
            "viewport": {"width": 1366, "height": 768},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
            "locale": "zh-CN",
            "timezone_id": "Asia/Shanghai",
        },
    ]
    
    def __init__(
        self,
        headless: bool = True,
        proxy: Optional[Dict[str, str]] = None,
        fingerprint_index: Optional[int] = None,
        slow_mo: int = 0,
        block_media: bool = True
    ):
        """
        Args:
            headless: 是否无头模式
            proxy: 代理配置 {"server": "http://proxy:port", "username": "...", "password": "..."}
            fingerprint_index: 指定指纹配置索引，None则随机
            slow_mo: 操作延迟（毫秒），用于调试
            block_media: 是否屏蔽图片/字体/媒体资源
        """
        self.headless = headless
        self.proxy = proxy
        self.slow_mo = slow_mo
        self.block_media = block_media
        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None
        self._captured_cookies: Dict[str, List[Dict]] = {}  # domain -> cookies
        
        # 选择指纹
        if fingerprint_index is not None:
            self._fingerprint = self.FINGERPRINTS[fingerprint_index % len(self.FINGERPRINTS)]
        else:
            self._fingerprint = random.choice(self.FINGERPRINTS)
    
    def _ensure_browser(self):
        """确保浏览器已启动"""
        if self._browser is None:
            try:
                from playwright.sync_api import sync_playwright
                self._playwright = sync_playwright().start()
                
                # 浏览器启动参数
                launch_args = {
                    "headless": self.headless,
                    "slow_mo": self.slow_mo,
                    "args": [
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                    ]
                }
                
                # 代理配置
                if self.proxy:
                    launch_args["proxy"] = self.proxy
                
                self._browser = self._playwright.chromium.launch(**launch_args)
                
                # 创建带指纹的上下文
                context_args = {
                    "viewport": self._fingerprint["viewport"],
                    "user_agent": self._fingerprint["user_agent"],
                    "locale": self._fingerprint.get("locale", "zh-CN"),
                    "timezone_id": self._fingerprint.get("timezone_id", "Asia/Shanghai"),
                    "color_scheme": "light",
                    "has_touch": False,
                    "is_mobile": False,
                    # 反自动化检测
                    "extra_http_headers": {
                        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    }
                }
                
                self._context = self._browser.new_context(**context_args)
                
                # 注入反检测脚本
                self._context.add_init_script("""
                    // 隐藏 webdriver 属性
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    
                    // 伪造 plugins
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5]
                    });
                    
                    // 伪造 languages
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['zh-CN', 'zh', 'en']
                    });
                    
                    // 伪造 Chrome 属性
                    window.chrome = {runtime: {}};
                """)
                
                self._page = self._context.new_page()
                
                # 路由拦截：屏蔽图片、字体、媒体文件
                if self.block_media:
                    self._page.route(
                        "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,otf,mp4,mp3}", 
                        lambda route: route.abort()
                    )
                
                logger.info(f"Playwright 浏览器已启动 (headless={self.headless}, block_media={self.block_media})")
                
            except ImportError:
                raise ImportError(
                    "请安装 playwright: pip install playwright && playwright install chromium"
                )
    
    def get(self, url: str) -> str:
        """访问 URL"""
        self._ensure_browser()
        try:
            # 优化：使用 domcontentloaded 代替 networkidle，并设置超时
            self._page.goto(url, wait_until="domcontentloaded", timeout=20000)
        except Exception as e:
            # 超时通常意味着页面主要内容已加载，但某些资源卡住
            logger.warning(f"页面加载超时 (继续处理): {e}")
            
        self._capture_cookies()
        return self._page.content()
    
    def get_with_wait(self, url: str, wait_selector: str, timeout: int = 10) -> str:
        """访问 URL 并等待元素"""
        self._ensure_browser()
        try:
            # 优化：设置 goto 超时
            self._page.goto(url, wait_until="domcontentloaded", timeout=20000)
        except Exception as e:
            logger.warning(f"页面导航超时 (继续等待元素): {e}")
            
        try:
            # 使用 state='attached' 只等待元素存在于DOM中，不要求可见
            # 这可以避免元素在视口外导致的超时问题
            self._page.wait_for_selector(wait_selector, timeout=timeout * 1000, state='attached')
        except Exception as e:
            logger.warning(f"等待元素超时: {wait_selector}, {e}")
            
        self._capture_cookies()
        return self._page.content()
    
    def get_with_scroll(self, url: str, scroll_times: int = 3, wait_time: float = 1.0) -> str:
        """访问 URL 并滚动加载（用于无限滚动页面）"""
        self._ensure_browser()
        self._page.goto(url, wait_until="networkidle")
        
        for i in range(scroll_times):
            self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(0.3)
        
        self._capture_cookies()
        return self._page.content()
    
    def wait_for_login(self, url: str, success_indicator: str, timeout: int = 120) -> bool:
        """
        等待用户手动登录
        
        用于需要验证码或复杂登录的站点（如雪球）
        
        Args:
            url: 登录页面 URL
            success_indicator: 登录成功的标志（CSS选择器或URL包含的字符串）
            timeout: 超时时间（秒）
        
        Returns:
            是否登录成功
        """
        if self.headless:
            logger.warning("等待登录需要有头模式，正在重新初始化...")
            self.close()
            self.headless = False
        
        self._ensure_browser()
        self._page.goto(url)
        
        logger.info(f"请在 {timeout} 秒内完成登录...")
        logger.info(f"登录成功标志: {success_indicator}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # 检查 URL 是否变化
            if success_indicator.startswith('http') or success_indicator.startswith('/'):
                if success_indicator in self._page.url:
                    logger.info("检测到登录成功（URL匹配）")
                    self._capture_cookies()
                    return True
            else:
                # 检查元素是否出现
                try:
                    if self._page.query_selector(success_indicator):
                        logger.info("检测到登录成功（元素匹配）")
                        self._capture_cookies()
                        return True
                except:
                    pass
            
            time.sleep(1)
        
        logger.warning("登录超时")
        return False
    
    def _capture_cookies(self):
        """捕获当前页面的 Cookie"""
        if self._context:
            cookies = self._context.cookies()
            for cookie in cookies:
                domain = cookie.get('domain', '').lstrip('.')
                if domain not in self._captured_cookies:
                    self._captured_cookies[domain] = []
                
                # 避免重复
                existing = [c['name'] for c in self._captured_cookies[domain]]
                if cookie['name'] not in existing:
                    self._captured_cookies[domain].append(cookie)
    
    def get_captured_cookies(self, domain: Optional[str] = None) -> Dict[str, str]:
        """
        获取捕获的 Cookie（字典格式，可直接用于 requests）
        
        Args:
            domain: 指定域名，None则返回所有
        
        Returns:
            Cookie 字典
        """
        result = {}
        
        if domain:
            cookies = self._captured_cookies.get(domain, [])
            for c in cookies:
                result[c['name']] = c['value']
        else:
            for domain_cookies in self._captured_cookies.values():
                for c in domain_cookies:
                    result[c['name']] = c['value']
        
        return result
    
    def get_cookie_string(self, domain: Optional[str] = None) -> str:
        """获取 Cookie 字符串格式"""
        cookies = self.get_captured_cookies(domain)
        return '; '.join(f"{k}={v}" for k, v in cookies.items())
    
    def inject_cookies(self, cookies: List[Dict], url: str):
        """
        注入 Cookie 到浏览器
        
        Args:
            cookies: Cookie 列表，每个元素为 {"name": ..., "value": ..., "domain": ...}
            url: 当前页面 URL（用于设置 Cookie 的域）
        """
        self._ensure_browser()
        self._context.add_cookies(cookies)
        logger.info(f"已注入 {len(cookies)} 个 Cookie")
    
    def screenshot(self, path: str):
        """截图保存"""
        if self._page:
            self._page.screenshot(path=path)
            logger.info(f"截图已保存: {path}")
    
    def close(self):
        """关闭浏览器"""
        if self._context:
            self._context.close()
            self._context = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        self._page = None


class SeleniumDriver(BrowserDriver):
    """
    Selenium 浏览器驱动（预留实现）
    
    需要安装：pip install selenium webdriver-manager
    """
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._driver = None
    
    def _ensure_browser(self):
        """确保浏览器已启动"""
        if self._driver is None:
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                
                options = Options()
                if self.headless:
                    options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                
                service = Service(ChromeDriverManager().install())
                self._driver = webdriver.Chrome(service=service, options=options)
                logger.info("Selenium 浏览器已启动")
            except ImportError:
                raise ImportError(
                    "请安装 selenium: pip install selenium webdriver-manager"
                )
    
    def get(self, url: str) -> str:
        """访问 URL"""
        self._ensure_browser()
        self._driver.get(url)
        return self._driver.page_source
    
    def get_with_wait(self, url: str, wait_selector: str, timeout: int = 10) -> str:
        """访问 URL 并等待元素"""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        self._ensure_browser()
        self._driver.get(url)
        WebDriverWait(self._driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
        )
        return self._driver.page_source
    
    def close(self):
        """关闭浏览器"""
        if self._driver:
            self._driver.quit()


# ============== 增强型爬虫基类 ==============

class ScraperBase:
    """
    增强型爬虫基类
    
    作为工具类被 Collector 调用，提供：
    - 代理池集成
    - User-Agent 随机轮换
    - 指数退避重试
    - Cookie 轮询管理
    - 浏览器模拟（可选）
    
    使用示例：
    ```python
    class MyCollector(UnstructuredCollector):
        def __init__(self):
            super().__init__()
            self.scraper = ScraperBase()
            # 添加 Cookie
            self.scraper.cookie_manager.load_from_env()
        
        def collect(self, ...):
            # 普通请求
            response = self.scraper.get("https://example.com")
            
            # 带 Cookie 的请求
            response = self.scraper.get(
                "https://xueqiu.com/api/data",
                use_cookies=True
            )
            
            # 使用浏览器模拟
            html = self.scraper.get_with_browser("https://example.com")
    ```
    """
    
    def __init__(
        self,
        use_proxy: bool = False,
        rate_limit: bool = True,
        timeout: int = 30,
        max_retries: int = 5
    ):
        """
        Args:
            use_proxy: 是否使用代理池
            rate_limit: 是否启用速率限制
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.use_proxy = use_proxy
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 组件初始化
        self.ua_manager = UserAgentManager()
        self.cookie_manager = CookieManager()
        self._proxy_pool: Optional[ProxyPool] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._browser: Optional[BrowserDriver] = None
        self._session: Optional[requests.Session] = None
        
        # 从环境变量加载 Cookie
        self.cookie_manager.load_from_env()
        
        # 懒加载组件
        if use_proxy:
            self._proxy_pool = get_proxy_pool()
        if rate_limit:
            self._rate_limiter = get_rate_limiter()
    
    @property
    def session(self) -> requests.Session:
        """获取或创建 Session"""
        if self._session is None:
            self._session = self._create_session()
        return self._session
    
    def _create_session(self) -> requests.Session:
        """创建配置好的 Session"""
        session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=3,  # Session 内部重试
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_headers(
        self,
        custom_headers: Optional[Dict[str, str]] = None,
        referer: Optional[str] = None
    ) -> Dict[str, str]:
        """生成请求头"""
        headers = {
            'User-Agent': self.ua_manager.get_random(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
        }
        
        if referer:
            headers['Referer'] = referer
        
        if custom_headers:
            headers.update(custom_headers)
        
        return headers
    
    def _get_domain(self, url: str) -> str:
        """从 URL 提取域名"""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return ''
    
    def get_proxy(self) -> Optional[Dict[str, str]]:
        """
        获取代理（预留接口）
        
        子类可以重写此方法实现自定义代理获取逻辑
        """
        if self._proxy_pool:
            return self._proxy_pool.get_proxy()
        return None
    
    @exponential_backoff(
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0
    )
    def get(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        use_cookies: bool = False,
        referer: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Optional[requests.Response]:
        """
        发送 GET 请求
        
        Args:
            url: 请求 URL
            params: URL 参数
            headers: 自定义请求头
            use_cookies: 是否使用 Cookie
            referer: Referer 头
            timeout: 超时时间
        
        Returns:
            Response 对象，失败返回 None
        """
        return self._request('GET', url, params=params, headers=headers,
                           use_cookies=use_cookies, referer=referer, timeout=timeout)
    
    @exponential_backoff(
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0
    )
    def post(
        self,
        url: str,
        data: Optional[Dict] = None,
        json: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        use_cookies: bool = False,
        referer: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Optional[requests.Response]:
        """
        发送 POST 请求
        
        Args:
            url: 请求 URL
            data: 表单数据
            json: JSON 数据
            headers: 自定义请求头
            use_cookies: 是否使用 Cookie
            referer: Referer 头
            timeout: 超时时间
        
        Returns:
            Response 对象，失败返回 None
        """
        return self._request('POST', url, data=data, json=json, headers=headers,
                           use_cookies=use_cookies, referer=referer, timeout=timeout)
    
    def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        json: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        use_cookies: bool = False,
        referer: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Optional[requests.Response]:
        """内部请求方法"""
        # 速率限制
        if self._rate_limiter:
            self._rate_limiter.wait(url)
        
        # 准备请求头
        request_headers = self._get_headers(headers, referer)
        
        # 准备 Cookie
        cookies = None
        if use_cookies:
            domain = self._get_domain(url)
            cookies = self.cookie_manager.get_cookies_for_requests(domain)
        
        # 获取代理
        proxy = self.get_proxy() if self.use_proxy else None
        
        try:
            start_time = time.time()
            
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json,
                headers=request_headers,
                cookies=cookies,
                proxies=proxy,
                timeout=timeout or self.timeout
            )
            
            response_time = time.time() - start_time
            
            # 报告结果
            if self._rate_limiter:
                if response.status_code in (429, 502, 503):
                    self._rate_limiter.report_error(url, response.status_code)
                else:
                    self._rate_limiter.report_success(url)
            
            if self._proxy_pool and proxy:
                if response.status_code < 400:
                    self._proxy_pool.report_success(proxy, response_time)
                else:
                    self._proxy_pool.report_failure(proxy)
            
            # 检查 Cookie 是否有效
            if use_cookies and response.status_code in (401, 403):
                domain = self._get_domain(url)
                if cookies:
                    self.cookie_manager.mark_invalid(domain, cookies)
                    logger.warning(f"Cookie 可能已失效: {domain}")
            
            return response
            
        except Exception as e:
            if self._proxy_pool and proxy:
                self._proxy_pool.report_failure(proxy)
            raise
    
    def get_json(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        use_cookies: bool = False
    ) -> Optional[Dict]:
        """
        发送 GET 请求并解析 JSON 响应
        
        Returns:
            JSON 数据字典，失败返回 None
        """
        response = self.get(url, params=params, headers=headers, use_cookies=use_cookies)
        if response and response.status_code == 200:
            try:
                return response.json()
            except:
                logger.error(f"JSON 解析失败: {url}")
        return None
    
    # ============== 浏览器模拟 ==============
    
    def init_browser(self, driver_type: str = 'playwright', headless: bool = True):
        """
        初始化浏览器驱动
        
        Args:
            driver_type: 驱动类型 ('playwright' 或 'selenium')
            headless: 是否无头模式
        """
        if driver_type == 'playwright':
            self._browser = PlaywrightDriver(headless=headless)
        elif driver_type == 'selenium':
            self._browser = SeleniumDriver(headless=headless)
        else:
            raise ValueError(f"不支持的驱动类型: {driver_type}")
    
    def get_with_browser(
        self,
        url: str,
        wait_selector: Optional[str] = None,
        timeout: int = 10
    ) -> Optional[str]:
        """
        使用浏览器访问页面（用于动态加载内容）
        
        Args:
            url: 页面 URL
            wait_selector: 等待的 CSS 选择器
            timeout: 超时时间（秒）
        
        Returns:
            页面 HTML 内容
        """
        if self._browser is None:
            self.init_browser()
        
        try:
            if wait_selector:
                return self._browser.get_with_wait(url, wait_selector, timeout)
            else:
                return self._browser.get(url)
        except Exception as e:
            logger.error(f"浏览器访问失败: {url} - {e}")
            return None
    
    def close_browser(self):
        """关闭浏览器"""
        if self._browser:
            self._browser.close()
            self._browser = None
    
    def close(self):
        """关闭所有资源"""
        if self._session:
            self._session.close()
            self._session = None
        self.close_browser()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============== 全局实例 ==============

_global_scraper: Optional[ScraperBase] = None
_global_cookie_manager: Optional[CookieManager] = None


def get_scraper(
    use_proxy: bool = False,
    rate_limit: bool = True
) -> ScraperBase:
    """获取全局 Scraper 实例"""
    global _global_scraper
    if _global_scraper is None:
        _global_scraper = ScraperBase(use_proxy=use_proxy, rate_limit=rate_limit)
    return _global_scraper


def get_cookie_manager() -> CookieManager:
    """获取全局 Cookie 管理器"""
    global _global_cookie_manager
    if _global_cookie_manager is None:
        _global_cookie_manager = CookieManager()
        _global_cookie_manager.load_from_env()
    return _global_cookie_manager


# ============== 两段式采集器 ==============

class TwoPhaseCollector:
    """
    两段式采集器
    
    采集策略：
    1. 身份获取段：使用 Playwright 浏览器访问目标站点，手动或自动登录，捕获 Cookie
    2. 高频请求段：将 Cookie 注入到轻量级 HTTP 请求中，进行大规模数据回溯
    
    适用场景：
    - 雪球（需要登录查看评论）
    - 东方财富（高级功能需要登录）
    - 其他反爬机制严密的站点
    
    使用示例：
    ```python
    collector = TwoPhaseCollector()
    
    # 阶段1：获取 Cookie（手动登录）
    success = collector.acquire_cookies_interactive(
        url="https://xueqiu.com",
        domain="xueqiu.com",
        login_url="https://xueqiu.com/service/login",
        success_indicator=".nav__user"  # 登录成功后出现的元素
    )
    
    # 阶段2：使用 Cookie 进行高频采集
    if success:
        data = collector.fetch_with_cookies(
            "https://xueqiu.com/statuses/stock_timeline.json",
            domain="xueqiu.com",
            params={"symbol": "SH600519", "count": 20}
        )
    ```
    """
    
    def __init__(
        self,
        use_proxy: bool = False,
        rate_limit: bool = True
    ):
        self._scraper = ScraperBase(use_proxy=use_proxy, rate_limit=rate_limit)
        self._browser: Optional[PlaywrightDriver] = None
        self._acquired_cookies: Dict[str, str] = {}  # domain -> cookie_string
    
    @property
    def scraper(self) -> ScraperBase:
        return self._scraper
    
    @property
    def cookie_manager(self) -> CookieManager:
        return self._scraper.cookie_manager
    
    # ============== 阶段1：身份获取 ==============
    
    def acquire_cookies_auto(
        self,
        url: str,
        domain: str,
        wait_selector: Optional[str] = None,
        headless: bool = True
    ) -> bool:
        """
        自动获取 Cookie（无需登录的站点）
        
        只是访问页面，让服务器设置必要的 Cookie
        
        Args:
            url: 目标页面 URL
            domain: Cookie 域名
            wait_selector: 等待元素选择器
            headless: 是否无头模式
        
        Returns:
            是否成功
        """
        try:
            self._browser = PlaywrightDriver(headless=headless)
            
            if wait_selector:
                self._browser.get_with_wait(url, wait_selector)
            else:
                self._browser.get(url)
            
            # 捕获并保存 Cookie
            cookies = self._browser.get_captured_cookies(domain)
            if cookies:
                cookie_str = self._browser.get_cookie_string(domain)
                self._acquired_cookies[domain] = cookie_str
                
                # 注入到 ScraperBase 的 CookieManager
                self._scraper.cookie_manager.add_cookies(domain, cookies, expire_hours=24)
                
                logger.info(f"自动获取 {domain} Cookie 成功: {len(cookies)} 个")
                return True
            else:
                logger.warning(f"未能获取 {domain} 的 Cookie")
                return False
                
        except Exception as e:
            logger.error(f"自动获取 Cookie 失败: {e}")
            return False
        finally:
            if self._browser:
                self._browser.close()
                self._browser = None
    
    def acquire_cookies_interactive(
        self,
        url: str,
        domain: str,
        login_url: Optional[str] = None,
        success_indicator: str = "",
        timeout: int = 120
    ) -> bool:
        """
        交互式获取 Cookie（需要手动登录）
        
        会打开浏览器窗口，等待用户完成登录
        
        Args:
            url: 站点首页 URL
            domain: Cookie 域名
            login_url: 登录页面 URL（如果不同于首页）
            success_indicator: 登录成功的标志（CSS选择器或URL片段）
            timeout: 超时时间（秒）
        
        Returns:
            是否成功
        """
        try:
            # 必须使用有头模式
            self._browser = PlaywrightDriver(headless=False, slow_mo=50)
            
            target_url = login_url or url
            success = self._browser.wait_for_login(target_url, success_indicator, timeout)
            
            if success:
                cookies = self._browser.get_captured_cookies(domain)
                if cookies:
                    cookie_str = self._browser.get_cookie_string(domain)
                    self._acquired_cookies[domain] = cookie_str
                    
                    self._scraper.cookie_manager.add_cookies(domain, cookies, expire_hours=24)
                    
                    logger.info(f"交互式获取 {domain} Cookie 成功: {len(cookies)} 个")
                    
                    # 保存到环境变量文件（可选）
                    self._save_cookie_to_env(domain, cookie_str)
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"交互式获取 Cookie 失败: {e}")
            return False
        finally:
            if self._browser:
                self._browser.close()
                self._browser = None
    
    def _save_cookie_to_env(self, domain: str, cookie_str: str):
        """保存 Cookie 到 .env 文件（可选）"""
        env_var_name = domain.upper().replace('.', '_') + '_COOKIE'
        
        try:
            env_path = os.path.join(os.getcwd(), '.env')
            
            # 读取现有内容
            existing_content = ""
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
            
            # 检查是否已存在该变量
            import re
            pattern = rf'^{env_var_name}=.*$'
            if re.search(pattern, existing_content, re.MULTILINE):
                # 替换现有值
                existing_content = re.sub(
                    pattern,
                    f'{env_var_name}="{cookie_str}"',
                    existing_content,
                    flags=re.MULTILINE
                )
            else:
                # 追加新变量
                existing_content += f'\n{env_var_name}="{cookie_str}"\n'
            
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(existing_content)
            
            logger.info(f"Cookie 已保存到 .env: {env_var_name}")
            
        except Exception as e:
            logger.warning(f"保存 Cookie 到 .env 失败: {e}")
    
    # ============== 阶段2：高频采集 ==============
    
    def fetch_with_cookies(
        self,
        url: str,
        domain: str,
        params: Optional[Dict] = None,
        method: str = 'GET',
        **kwargs
    ) -> Optional[requests.Response]:
        """
        使用已获取的 Cookie 发送 HTTP 请求
        
        Args:
            url: 请求 URL
            domain: Cookie 域名
            params: 请求参数
            method: 请求方法
            **kwargs: 其他 requests 参数
        
        Returns:
            Response 对象
        """
        # 确保有 Cookie
        if domain not in self._acquired_cookies:
            # 尝试从 CookieManager 获取
            cookie_str = self._scraper.cookie_manager.get_cookie_string(domain)
            if cookie_str:
                self._acquired_cookies[domain] = cookie_str
            else:
                logger.warning(f"没有可用的 {domain} Cookie，请先调用 acquire_cookies_*")
                return None
        
        if method.upper() == 'GET':
            return self._scraper.get(url, params=params, use_cookies=True, **kwargs)
        else:
            return self._scraper.post(url, data=params, use_cookies=True, **kwargs)
    
    def fetch_json(
        self,
        url: str,
        domain: str,
        params: Optional[Dict] = None,
        **kwargs
    ) -> Optional[Dict]:
        """发送请求并解析 JSON"""
        response = self.fetch_with_cookies(url, domain, params, **kwargs)
        if response and response.status_code == 200:
            try:
                return response.json()
            except Exception as e:
                logger.error(f"JSON 解析失败: {e}")
        return None
    
    def has_valid_cookies(self, domain: str) -> bool:
        """检查是否有有效的 Cookie"""
        cookie_str = self._scraper.cookie_manager.get_cookie_string(domain)
        return bool(cookie_str)
    
    def close(self):
        """关闭所有资源"""
        if self._browser:
            self._browser.close()
            self._browser = None
        self._scraper.close()


# 雪球专用采集器
class XueqiuCollector(TwoPhaseCollector):
    """
    雪球两段式采集器
    
    针对雪球的特殊优化：
    - 自动检测登录状态
    - 支持多种 API 端点
    - 处理反爬响应
    """
    
    DOMAIN = "xueqiu.com"
    HOME_URL = "https://xueqiu.com"
    LOGIN_URL = "https://xueqiu.com"
    
    # API 端点
    API_STOCK_TIMELINE = "https://xueqiu.com/statuses/stock_timeline.json"
    API_USER_POSTS = "https://xueqiu.com/v4/statuses/user_timeline.json"
    API_STOCK_QUOTE = "https://stock.xueqiu.com/v5/stock/quote.json"
    
    def ensure_cookies(self, interactive: bool = False) -> bool:
        """
        确保有可用的雪球 Cookie
        
        Args:
            interactive: 是否允许交互式登录
        
        Returns:
            是否有可用 Cookie
        """
        # 检查现有 Cookie
        if self.has_valid_cookies(self.DOMAIN):
            logger.info("已有可用的雪球 Cookie")
            return True
        
        # 尝试自动获取（访客 Cookie）
        logger.info("尝试自动获取雪球访客 Cookie...")
        if self.acquire_cookies_auto(self.HOME_URL, self.DOMAIN, wait_selector=".nav"):
            return True
        
        # 如果允许，进行交互式登录
        if interactive:
            logger.info("需要交互式登录获取雪球 Cookie")
            return self.acquire_cookies_interactive(
                url=self.HOME_URL,
                domain=self.DOMAIN,
                success_indicator=".nav__user",  # 登录后出现的用户头像元素
                timeout=180
            )
        
        return False
    
    def get_stock_comments(
        self,
        symbol: str,
        count: int = 20,
        max_pages: int = 5
    ) -> List[Dict]:
        """
        获取股票评论
        
        Args:
            symbol: 股票代码（如 SH600519）
            count: 每页数量
            max_pages: 最大页数
        
        Returns:
            评论列表
        """
        # 优先使用Playwright直接获取（绕过WAF）
        comments = self._get_comments_via_playwright(symbol, count, max_pages)
        if comments:
            return comments
        
        # 备选：传统HTTP方式
        if not self.ensure_cookies():
            logger.warning("无法获取雪球 Cookie")
            return []
        
        all_comments = []
        max_id = None
        
        for page in range(max_pages):
            params = {
                "symbol": symbol,
                "count": count,
                "source": "all"
            }
            if max_id:
                params["max_id"] = max_id
            
            data = self.fetch_json(
                self.API_STOCK_TIMELINE,
                self.DOMAIN,
                params=params,
                referer=f"https://xueqiu.com/S/{symbol}"
            )
            
            if not data:
                break
            
            posts = data.get("list", [])
            if not posts:
                break
            
            all_comments.extend(posts)
            
            # 获取下一页的 max_id
            max_id = posts[-1].get("id") if posts else None
            
            time.sleep(0.3)  # 减少延迟
        
        return all_comments
    
    def _get_comments_via_playwright(
        self,
        symbol: str,
        count: int = 20,
        max_pages: int = 3
    ) -> List[Dict]:
        """
        使用Playwright直接获取评论（绕过WAF）
        
        策略：直接访问股票页面，从页面中提取评论数据
        """
        all_comments = []
        driver = None
        
        try:
            driver = PlaywrightDriver(headless=True)
            
            # 访问股票讨论页面
            stock_url = f"https://xueqiu.com/S/{symbol}"
            html = driver.get_with_scroll(stock_url, scroll_times=max_pages, wait_time=2.0)
            
            if html:
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(html, 'html.parser')
                
                # 从timeline__item元素提取评论
                for item in soup.select('.timeline__item'):
                    try:
                        # 内容 - 使用正确的class选择器
                        content_elem = item.select_one('.timeline__item__content, .content')
                        if not content_elem:
                            content_elem = item.select_one('.timeline__item__bd')
                        
                        text = content_elem.get_text(strip=True) if content_elem else ''
                        if not text:
                            continue
                        
                        # 作者
                        author_elem = item.select_one('.user-name')
                        author = author_elem.get_text(strip=True) if author_elem else ''
                        
                        # 时间 - date-and-source类
                        time_elem = item.select_one('.date-and-source')
                        pub_time = time_elem.get_text(strip=True) if time_elem else ''
                        
                        all_comments.append({
                            'text': text,
                            'created_at': pub_time,
                            'user': {'screen_name': author},
                            'source': 'xueqiu',
                            'symbol': symbol
                        })
                    except Exception as e:
                        continue
            
            if all_comments:
                logger.info(f"Playwright采集雪球 {symbol}: {len(all_comments)} 条")
                
        except Exception as e:
            logger.debug(f"Playwright采集雪球失败: {e}")
        finally:
            if driver:
                try:
                    driver.close()
                except:
                    pass
        
        return all_comments


# 东方财富专用采集器
class EastMoneyCollector(TwoPhaseCollector):
    """
    东方财富两段式采集器
    
    支持：
    - 股吧评论
    - 个股新闻
    - 研报数据
    """
    
    DOMAIN = "eastmoney.com"
    GUBA_DOMAIN = "guba.eastmoney.com"
    HOME_URL = "https://www.eastmoney.com"
    
    # API 端点
    API_GUBA_LIST = "https://guba.eastmoney.com/interface/GetData.aspx"
    API_NEWS = "https://newsapi.eastmoney.com/kuaixun/v1/getlist_102_ajaxResult_50_{page}_.html"
    
    def ensure_cookies(self) -> bool:
        """确保有可用的东方财富 Cookie"""
        if self.has_valid_cookies(self.DOMAIN):
            return True
        
        # 东方财富通常不需要登录，自动获取访客 Cookie 即可
        return self.acquire_cookies_auto(
            self.HOME_URL,
            self.DOMAIN,
            wait_selector="body"
        )
    
    def get_guba_comments(
        self,
        stock_code: str,
        max_pages: int = 10
    ) -> List[Dict]:
        """
        获取股吧评论（含内容）
        
        Args:
            stock_code: 股票代码（纯数字，如 600519）
            max_pages: 最大页数
        
        Returns:
            评论列表
        """
        self.ensure_cookies()
        
        all_comments = []
        
        for page in range(1, max_pages + 1):
            # 股吧帖子列表API
            list_url = f"https://guba.eastmoney.com/list,{stock_code},f_{page}.html"
            
            try:
                response = self._scraper.get(
                    list_url,
                    referer=f"https://guba.eastmoney.com/list,{stock_code}.html",
                    use_cookies=True
                )
                
                if not response or response.status_code != 200:
                    break
                
                # 解析帖子列表
                posts = self._parse_guba_list(response.text, stock_code)
                
                if not posts:
                    break
                
                # 获取帖子详情（内容）
                for post in posts:
                    if post.get('post_id'):
                        content = self._fetch_post_content(stock_code, post['post_id'])
                        if content:
                            post['content'] = content
                    all_comments.append(post)
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"获取股吧评论失败: {e}")
                break
        
        return all_comments
    
    def _parse_guba_list(self, html: str, stock_code: str) -> List[Dict]:
        """解析股吧帖子列表"""
        posts = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
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
                    import re
                    post_id_match = re.search(r',(\d+)\.html', href)
                    post_id = post_id_match.group(1) if post_id_match else ''
                    
                    # 作者
                    author_elem = item.select_one('.l4 a, .author a')
                    author = author_elem.get_text(strip=True) if author_elem else ''
                    
                    # 时间
                    time_elem = item.select_one('.l5, .update')
                    pub_time = time_elem.get_text(strip=True) if time_elem else ''
                    
                    # 阅读数
                    read_elem = item.select_one('.l1, .read')
                    read_count = read_elem.get_text(strip=True) if read_elem else '0'
                    
                    # 回复数
                    reply_elem = item.select_one('.l2, .reply')
                    reply_count = reply_elem.get_text(strip=True) if reply_elem else '0'
                    
                    posts.append({
                        'post_id': post_id,
                        'title': title,
                        'author': author,
                        'pub_time': pub_time,
                        'read_count': read_count,
                        'reply_count': reply_count,
                        'url': f"https://guba.eastmoney.com/news,{stock_code},{post_id}.html" if post_id else '',
                        'content': ''  # 稍后填充
                    })
                    
                except Exception as e:
                    continue
                    
        except ImportError:
            logger.warning("BeautifulSoup 未安装")
        except Exception as e:
            logger.debug(f"解析股吧列表失败: {e}")
        
        return posts
    
    def _fetch_post_content(self, stock_code: str, post_id: str) -> str:
        """获取帖子内容"""
        try:
            url = f"https://guba.eastmoney.com/news,{stock_code},{post_id}.html"
            
            response = self._scraper.get(
                url,
                referer=f"https://guba.eastmoney.com/list,{stock_code}.html",
                use_cookies=True
            )
            
            if response and response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找正文
                content_elem = soup.select_one('.stockcodec, .newstext, .post-content')
                if content_elem:
                    return content_elem.get_text(strip=True)
            
        except Exception as e:
            logger.debug(f"获取帖子内容失败: {e}")
        
        return ''
