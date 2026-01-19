"""
请求工具模块

提供请求伪装和HTTP工具函数：
- User-Agent轮换
- 请求头伪装
- Cookie管理
- 安全下载功能
"""

import os
import time
import random
import logging
import hashlib
from typing import Optional, Dict, Any, List
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .rate_limiter import get_rate_limiter, RateLimiter
from .proxy_pool import get_proxy_pool, ProxyPool

logger = logging.getLogger(__name__)


# 常用User-Agent列表
USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    # Firefox on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    # Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# 常用Referer列表
REFERERS = [
    "https://www.google.com/",
    "https://www.baidu.com/",
    "https://www.bing.com/",
    "https://cn.bing.com/",
    "https://www.so.com/",
]


class RequestDisguiser:
    """
    请求伪装器
    
    用于绕过简单的反爬机制
    """
    
    def __init__(
        self,
        user_agents: Optional[List[str]] = None,
        referers: Optional[List[str]] = None,
        randomize_order: bool = True
    ):
        """
        Args:
            user_agents: 自定义User-Agent列表
            referers: 自定义Referer列表
            randomize_order: 是否随机化请求头顺序
        """
        self._user_agents = user_agents or USER_AGENTS
        self._referers = referers or REFERERS
        self._randomize_order = randomize_order
        self._ua_index = 0
    
    def get_random_user_agent(self) -> str:
        """获取随机User-Agent"""
        return random.choice(self._user_agents)
    
    def get_next_user_agent(self) -> str:
        """轮换获取User-Agent"""
        ua = self._user_agents[self._ua_index % len(self._user_agents)]
        self._ua_index += 1
        return ua
    
    def get_random_referer(self) -> str:
        """获取随机Referer"""
        return random.choice(self._referers)
    
    def get_headers(
        self,
        custom_headers: Optional[Dict[str, str]] = None,
        include_referer: bool = True
    ) -> Dict[str, str]:
        """
        生成伪装的请求头
        
        Args:
            custom_headers: 自定义头（会覆盖生成的头）
            include_referer: 是否包含Referer
        
        Returns:
            请求头字典
        """
        headers = {
            'User-Agent': self.get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
        }
        
        if include_referer:
            headers['Referer'] = self.get_random_referer()
        
        # 随机化请求头顺序（部分反爬会检测固定顺序）
        if self._randomize_order:
            items = list(headers.items())
            random.shuffle(items)
            headers = dict(items)
        
        # 合并自定义头
        if custom_headers:
            headers.update(custom_headers)
        
        return headers
    
    def get_json_headers(
        self,
        custom_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """获取JSON请求头"""
        headers = self.get_headers(custom_headers, include_referer=True)
        headers['Accept'] = 'application/json, text/plain, */*'
        headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=UTF-8'
        return headers


def create_session(
    proxy: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    timeout: int = 30,
    disguiser: Optional[RequestDisguiser] = None
) -> requests.Session:
    """
    创建配置好的请求会话
    
    Args:
        proxy: 代理配置
        max_retries: 最大重试次数
        backoff_factor: 重试退避因子
        timeout: 超时时间
        disguiser: 请求伪装器
    
    Returns:
        配置好的Session对象
    """
    session = requests.Session()
    
    # 配置重试策略
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 设置代理
    if proxy:
        session.proxies.update(proxy)
    
    # 设置请求头
    if disguiser is None:
        disguiser = RequestDisguiser()
    session.headers.update(disguiser.get_headers())
    
    return session


def safe_request(
    url: str,
    method: str = 'GET',
    params: Optional[Dict] = None,
    data: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: int = 30,
    use_proxy: bool = False,
    rate_limit: bool = True,
    max_retries: int = 3
) -> Optional[requests.Response]:
    """
    发送带有保护机制的HTTP请求
    
    Args:
        url: 请求URL
        method: 请求方法
        params: URL参数
        data: 表单数据
        json_data: JSON数据
        headers: 自定义请求头
        timeout: 超时时间
        use_proxy: 是否使用代理
        rate_limit: 是否限速
        max_retries: 最大重试次数
    
    Returns:
        Response对象或None
    """
    disguiser = RequestDisguiser()
    rate_limiter = get_rate_limiter() if rate_limit else None
    proxy_pool = get_proxy_pool() if use_proxy else None
    
    # 准备请求头
    request_headers = disguiser.get_headers(headers)
    if json_data:
        request_headers = disguiser.get_json_headers(headers)
    
    # 获取代理
    proxy = None
    if proxy_pool:
        proxy = proxy_pool.get_proxy()
    
    last_error = None
    for attempt in range(max_retries):
        try:
            # 速率限制
            if rate_limiter:
                rate_limiter.wait(url)
            
            start_time = time.time()
            
            # 发送请求
            response = requests.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json_data,
                headers=request_headers,
                proxies=proxy,
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            
            # 报告结果
            if rate_limiter:
                if response.status_code in (429, 502, 503):
                    rate_limiter.report_error(url, response.status_code)
                else:
                    rate_limiter.report_success(url)
            
            if proxy_pool and proxy:
                if response.status_code < 400:
                    proxy_pool.report_success(proxy, response_time)
                else:
                    proxy_pool.report_failure(proxy)
            
            # 检查状态码
            if response.status_code == 200:
                return response
            elif response.status_code in (429, 502, 503):
                # 需要重试
                wait_time = (attempt + 1) * 2
                logger.warning(
                    f"请求返回 {response.status_code}，"
                    f"{wait_time}秒后重试 ({attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
                continue
            else:
                response.raise_for_status()
                return response
                
        except requests.exceptions.Timeout:
            last_error = "请求超时"
            logger.warning(f"请求超时: {url}")
            
        except requests.exceptions.ConnectionError as e:
            last_error = f"连接错误: {e}"
            logger.warning(f"连接错误: {url} - {e}")
            
            # 如果使用代理，标记代理失败并切换
            if proxy_pool and proxy:
                proxy_pool.report_failure(proxy)
                proxy = proxy_pool.get_proxy()
            
        except Exception as e:
            last_error = str(e)
            logger.error(f"请求异常: {url} - {e}")
        
        # 重试前等待
        if attempt < max_retries - 1:
            time.sleep((attempt + 1) * 0.3)
    
    logger.error(f"请求失败（已重试{max_retries}次）: {url} - {last_error}")
    return None


def safe_download_file(
    url: str,
    save_path: str,
    timeout: int = 60,
    use_proxy: bool = False,
    chunk_size: int = 8192,
    overwrite: bool = False
) -> bool:
    """
    安全下载文件
    
    Args:
        url: 文件URL
        save_path: 保存路径
        timeout: 超时时间
        use_proxy: 是否使用代理
        chunk_size: 下载分块大小
        overwrite: 是否覆盖已存在文件
    
    Returns:
        是否下载成功
    """
    save_path = Path(save_path)
    
    # 检查文件是否已存在
    if save_path.exists() and not overwrite:
        logger.info(f"文件已存在，跳过下载: {save_path}")
        return True
    
    # 创建目录
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 下载
    disguiser = RequestDisguiser()
    rate_limiter = get_rate_limiter()
    proxy_pool = get_proxy_pool() if use_proxy else None
    
    proxy = proxy_pool.get_proxy() if proxy_pool else None
    
    try:
        rate_limiter.wait(url)
        
        response = requests.get(
            url,
            headers=disguiser.get_headers(),
            proxies=proxy,
            timeout=timeout,
            stream=True
        )
        response.raise_for_status()
        
        # 写入文件
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        
        rate_limiter.report_success(url)
        logger.info(f"文件下载成功: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"文件下载失败: {url} - {e}")
        # 删除可能的不完整文件
        if save_path.exists():
            save_path.unlink()
        return False


def generate_file_hash(content: bytes) -> str:
    """生成内容的MD5哈希"""
    return hashlib.md5(content).hexdigest()


def get_file_extension(url: str, content_type: Optional[str] = None) -> str:
    """
    从URL或Content-Type获取文件扩展名
    
    Args:
        url: 文件URL
        content_type: Content-Type header值
    
    Returns:
        文件扩展名（如 .pdf）
    """
    # 从URL提取
    from urllib.parse import urlparse
    parsed = urlparse(url)
    path = parsed.path
    if '.' in path:
        ext = '.' + path.rsplit('.', 1)[-1].lower()
        if ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt', '.html']:
            return ext
    
    # 从Content-Type推断
    if content_type:
        content_type_map = {
            'application/pdf': '.pdf',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.ms-excel': '.xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'text/html': '.html',
            'text/plain': '.txt',
        }
        for ct, ext in content_type_map.items():
            if ct in content_type:
                return ext
    
    return '.pdf'  # 默认PDF


class RequestSession:
    """
    请求会话管理器
    
    封装了会话、代理、限速的管理
    """
    
    def __init__(
        self,
        use_proxy: bool = False,
        rate_limit: bool = True,
        timeout: int = 30
    ):
        self.use_proxy = use_proxy
        self.rate_limit = rate_limit
        self.timeout = timeout
        
        self._disguiser = RequestDisguiser()
        self._rate_limiter = get_rate_limiter() if rate_limit else None
        self._proxy_pool = get_proxy_pool() if use_proxy else None
        self._session: Optional[requests.Session] = None
    
    def __enter__(self):
        self._session = create_session(
            disguiser=self._disguiser,
            timeout=self.timeout
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            self._session.close()
    
    def get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """发送GET请求"""
        return self._request('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs) -> Optional[requests.Response]:
        """发送POST请求"""
        return self._request('POST', url, **kwargs)
    
    def _request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """发送请求"""
        # 速率限制
        if self._rate_limiter:
            self._rate_limiter.wait(url)
        
        # 获取代理
        proxy = None
        if self._proxy_pool:
            proxy = self._proxy_pool.get_proxy()
        
        # 更新headers
        headers = self._disguiser.get_headers(kwargs.pop('headers', None))
        
        try:
            if self._session:
                response = self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    proxies=proxy,
                    timeout=self.timeout,
                    **kwargs
                )
            else:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    proxies=proxy,
                    timeout=self.timeout,
                    **kwargs
                )
            
            # 报告结果
            if self._rate_limiter:
                if response.status_code in (429, 502, 503):
                    self._rate_limiter.report_error(url, response.status_code)
                else:
                    self._rate_limiter.report_success(url)
            
            return response
            
        except Exception as e:
            logger.error(f"请求失败: {url} - {e}")
            if self._proxy_pool and proxy:
                self._proxy_pool.report_failure(proxy)
            return None
