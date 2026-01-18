"""
速率限制器模块

提供请求速率控制，防止触发反爬机制：
- 令牌桶算法
- 自适应速率调整
- 指数退避策略
"""

import time
import threading
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    requests_per_second: float = 1.0    # 每秒请求数
    burst_size: int = 5                  # 突发请求数
    min_interval: float = 0.5            # 最小请求间隔（秒）
    max_interval: float = 30.0           # 最大请求间隔（秒）
    backoff_factor: float = 2.0          # 退避因子
    recovery_factor: float = 0.9         # 恢复因子


class TokenBucket:
    """
    令牌桶算法实现
    
    用于平滑控制请求速率
    """
    
    def __init__(
        self,
        rate: float = 1.0,
        capacity: int = 5
    ):
        """
        Args:
            rate: 令牌生成速率（令牌/秒）
            capacity: 桶容量（最大令牌数）
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def _add_tokens(self):
        """添加令牌"""
        now = time.time()
        elapsed = now - self.last_update
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update = now
    
    def acquire(self, tokens: int = 1, blocking: bool = True) -> bool:
        """
        获取令牌
        
        Args:
            tokens: 需要的令牌数
            blocking: 是否阻塞等待
        
        Returns:
            是否成功获取令牌
        """
        with self._lock:
            self._add_tokens()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            if not blocking:
                return False
            
            # 计算需要等待的时间
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.rate
            
        # 在锁外等待
        time.sleep(wait_time)
        
        with self._lock:
            self._add_tokens()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait(self):
        """等待获取一个令牌"""
        self.acquire(1, blocking=True)


class RateLimiter:
    """
    速率限制器
    
    支持多域名不同限制策略
    """
    
    # 预定义的域名速率配置
    DEFAULT_DOMAIN_LIMITS = {
        'cninfo.com.cn': RateLimitConfig(
            requests_per_second=0.67,  # 约1.5秒/请求（更合理的速度）
            burst_size=3,
            min_interval=1.0
        ),
        'tushare.pro': RateLimitConfig(
            requests_per_second=1.0,  # 1秒/请求
            burst_size=5,
            min_interval=0.5
        ),
        'default': RateLimitConfig(
            requests_per_second=1.0,
            burst_size=5,
            min_interval=0.5
        )
    }
    
    def __init__(
        self,
        default_config: Optional[RateLimitConfig] = None,
        domain_configs: Optional[Dict[str, RateLimitConfig]] = None
    ):
        """
        Args:
            default_config: 默认速率配置
            domain_configs: 域名特定配置
        """
        self._buckets: Dict[str, TokenBucket] = {}
        self._last_request_time: Dict[str, float] = defaultdict(float)
        self._current_interval: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # 合并配置
        self._configs = dict(self.DEFAULT_DOMAIN_LIMITS)
        if domain_configs:
            self._configs.update(domain_configs)
        if default_config:
            self._configs['default'] = default_config
    
    def _get_domain(self, url: str) -> str:
        """从URL提取域名"""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            return parsed.netloc or 'default'
        except Exception:
            return 'default'
    
    def _get_config(self, domain: str) -> RateLimitConfig:
        """获取域名的速率配置"""
        # 尝试直接匹配
        if domain in self._configs:
            return self._configs[domain]
        
        # 尝试后缀匹配
        for key, config in self._configs.items():
            if domain.endswith(key):
                return config
        
        return self._configs['default']
    
    def _get_bucket(self, domain: str) -> TokenBucket:
        """获取或创建域名的令牌桶"""
        if domain not in self._buckets:
            config = self._get_config(domain)
            self._buckets[domain] = TokenBucket(
                rate=config.requests_per_second,
                capacity=config.burst_size
            )
            self._current_interval[domain] = 1.0 / config.requests_per_second
        return self._buckets[domain]
    
    def wait(self, url: str):
        """
        等待直到可以发送请求
        
        Args:
            url: 请求URL
        """
        domain = self._get_domain(url)
        bucket = self._get_bucket(domain)
        config = self._get_config(domain)
        
        with self._lock:
            # 确保最小间隔
            now = time.time()
            elapsed = now - self._last_request_time[domain]
            if elapsed < config.min_interval:
                time.sleep(config.min_interval - elapsed)
        
        # 获取令牌
        bucket.wait()
        
        with self._lock:
            self._last_request_time[domain] = time.time()
    
    def report_error(self, url: str, status_code: int):
        """
        报告请求错误，触发退避
        
        Args:
            url: 请求URL
            status_code: HTTP状态码
        """
        if status_code in (429, 503, 502):
            domain = self._get_domain(url)
            config = self._get_config(domain)
            
            with self._lock:
                current = self._current_interval.get(domain, 1.0)
                new_interval = min(
                    current * config.backoff_factor,
                    config.max_interval
                )
                self._current_interval[domain] = new_interval
                
                # 重建令牌桶
                new_rate = 1.0 / new_interval
                self._buckets[domain] = TokenBucket(
                    rate=new_rate,
                    capacity=config.burst_size
                )
                
                logger.warning(
                    f"域名 {domain} 触发退避，"
                    f"请求间隔调整为 {new_interval:.1f} 秒"
                )
    
    def report_success(self, url: str):
        """
        报告请求成功，逐步恢复速率
        
        Args:
            url: 请求URL
        """
        domain = self._get_domain(url)
        config = self._get_config(domain)
        default_interval = 1.0 / config.requests_per_second
        
        with self._lock:
            current = self._current_interval.get(domain, default_interval)
            if current > default_interval:
                new_interval = max(
                    current * config.recovery_factor,
                    default_interval
                )
                self._current_interval[domain] = new_interval
                
                # 重建令牌桶
                new_rate = 1.0 / new_interval
                self._buckets[domain] = TokenBucket(
                    rate=new_rate,
                    capacity=config.burst_size
                )


class AdaptiveRateLimiter(RateLimiter):
    """
    自适应速率限制器
    
    根据响应状态动态调整速率
    """
    
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._consecutive_success: Dict[str, int] = defaultdict(int)
        self._consecutive_failure: Dict[str, int] = defaultdict(int)
        self._blocked_until: Dict[str, float] = {}
    
    def is_blocked(self, url: str) -> bool:
        """检查域名是否被临时封禁"""
        domain = self._get_domain(url)
        blocked_until = self._blocked_until.get(domain, 0)
        return time.time() < blocked_until
    
    def wait(self, url: str):
        """等待直到可以发送请求"""
        domain = self._get_domain(url)
        
        # 检查是否被封禁
        if self.is_blocked(url):
            blocked_until = self._blocked_until[domain]
            wait_time = blocked_until - time.time()
            if wait_time > 0:
                logger.warning(
                    f"域名 {domain} 被封禁，等待 {wait_time:.0f} 秒"
                )
                time.sleep(wait_time)
        
        super().wait(url)
    
    def report_error(self, url: str, status_code: int):
        """报告请求错误"""
        domain = self._get_domain(url)
        
        with self._lock:
            self._consecutive_success[domain] = 0
            self._consecutive_failure[domain] += 1
            failures = self._consecutive_failure[domain]
        
        # 连续失败多次，临时封禁
        if failures >= 5:
            block_duration = min(60 * failures, 3600)  # 最长1小时
            self._blocked_until[domain] = time.time() + block_duration
            logger.error(
                f"域名 {domain} 连续失败 {failures} 次，"
                f"封禁 {block_duration} 秒"
            )
        
        super().report_error(url, status_code)
    
    def report_success(self, url: str):
        """报告请求成功"""
        domain = self._get_domain(url)
        
        with self._lock:
            self._consecutive_failure[domain] = 0
            self._consecutive_success[domain] += 1
        
        super().report_success(url)


# 全局速率限制器实例
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """获取全局速率限制器"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = AdaptiveRateLimiter()
    return _global_rate_limiter


def set_rate_limiter(limiter: RateLimiter):
    """设置全局速率限制器"""
    global _global_rate_limiter
    _global_rate_limiter = limiter
