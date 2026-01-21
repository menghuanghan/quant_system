"""
速率限制器模块

提供请求速率控制，防止触发反爬机制：
- 令牌桶算法
- 自适应速率调整
- 指数退避策略
- 错误率监控（滑动窗口统计）
- 速率预热（启动时逐渐加速）
- 检查点持久化（重启后恢复状态）
"""

import time
import json
import random
import threading
import logging
from pathlib import Path
from typing import Dict, Optional, List, Deque, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    requests_per_second: float = 2.0    # 每秒请求数（提高到2）
    burst_size: int = 5                  # 突发请求数
    min_interval: float = 0.3            # 最小请求间隔（秒，降低到0.3）
    max_interval: float = 60.0           # 最大请求间隔（秒）
    backoff_factor: float = 2.0          # 退避因子
    recovery_factor: float = 0.9         # 恢复因子
    # 新增配置
    error_rate_threshold: float = 0.1    # 错误率阈值（超过则退避）
    success_rate_threshold: float = 0.95 # 成功率阈值（超过则恢复）
    window_size: int = 100               # 滑动窗口大小
    warmup_requests: int = 5             # 预热请求数（降低到5）
    jitter_factor: float = 0.2           # 抖动因子（防止惊群效应）


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


@dataclass
class RequestRecord:
    """请求记录"""
    timestamp: float
    success: bool
    status_code: int
    latency: float = 0.0


@dataclass
class DomainStats:
    """域名统计信息"""
    total_requests: int = 0
    total_errors: int = 0
    current_interval: float = 1.0
    warmup_completed: bool = False
    last_error_time: float = 0.0
    error_codes: Dict[int, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'current_interval': self.current_interval,
            'warmup_completed': self.warmup_completed,
            'last_error_time': self.last_error_time,
            'error_codes': self.error_codes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DomainStats':
        return cls(
            total_requests=data.get('total_requests', 0),
            total_errors=data.get('total_errors', 0),
            current_interval=data.get('current_interval', 1.0),
            warmup_completed=data.get('warmup_completed', False),
            last_error_time=data.get('last_error_time', 0.0),
            error_codes=data.get('error_codes', {})
        )


class SmartRateLimiter(RateLimiter):
    """
    智能速率限制器
    
    增强功能：
    1. 滑动窗口错误率监控 - 实时计算错误率
    2. 指数退避带抖动 - 防止惊群效应
    3. 速率预热 - 启动时逐渐提速
    4. 检查点持久化 - 重启后恢复状态
    5. 详细统计信息 - 便于监控和调试
    """
    
    # 需要触发退避的HTTP状态码
    BACKOFF_STATUS_CODES = {403, 429, 503, 502, 500}
    
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        auto_persist: bool = True,
        persist_interval: int = 100,
        **kwargs
    ):
        """
        Args:
            checkpoint_dir: 检查点保存目录
            auto_persist: 是否自动持久化
            persist_interval: 持久化间隔（请求数）
        """
        super().__init__(**kwargs)
        
        # 滑动窗口记录
        self._request_windows: Dict[str, Deque[RequestRecord]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # 域名统计
        self._domain_stats: Dict[str, DomainStats] = defaultdict(DomainStats)
        
        # 封禁状态
        self._blocked_until: Dict[str, float] = {}
        
        # 持久化设置
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._auto_persist = auto_persist
        self._persist_interval = persist_interval
        self._requests_since_persist = 0
        
        # 加载检查点
        if self._checkpoint_dir:
            self._load_checkpoint()
    
    def _calculate_error_rate(self, domain: str) -> Tuple[float, int]:
        """
        计算滑动窗口内的错误率
        
        Returns:
            (error_rate, window_size)
        """
        window = self._request_windows[domain]
        if not window:
            return 0.0, 0
        
        errors = sum(1 for r in window if not r.success)
        return errors / len(window), len(window)
    
    def _calculate_jitter_delay(self, base_delay: float, config: RateLimitConfig) -> float:
        """
        计算带抖动的延迟时间
        
        Args:
            base_delay: 基础延迟
            config: 速率配置
        
        Returns:
            实际延迟（带随机抖动）
        """
        jitter = base_delay * config.jitter_factor * random.random()
        return base_delay + jitter
    
    def _calculate_backoff_delay(
        self, 
        domain: str, 
        config: RateLimitConfig,
        consecutive_errors: int
    ) -> float:
        """
        计算指数退避延迟
        
        使用公式: delay = min(base * factor^errors, max_interval)
        """
        base = config.min_interval
        delay = base * (config.backoff_factor ** consecutive_errors)
        return min(delay, config.max_interval)
    
    def _get_warmup_interval(self, domain: str, config: RateLimitConfig) -> float:
        """
        获取预热阶段的请求间隔
        
        预热期间逐渐降低间隔，直到达到目标速率
        """
        stats = self._domain_stats[domain]
        if stats.warmup_completed:
            return 1.0 / config.requests_per_second
        
        # 预热进度 (0.0 ~ 1.0)
        progress = min(stats.total_requests / config.warmup_requests, 1.0)
        
        # 从 1.5x 目标间隔 逐渐降到 1x 目标间隔（优化：降低预热倍数）
        target_interval = 1.0 / config.requests_per_second
        warmup_interval = target_interval * (1.5 - 0.5 * progress)
        
        if progress >= 1.0:
            stats.warmup_completed = True
            logger.info(f"域名 {domain} 预热完成，切换到正常速率")
        
        return warmup_interval
    
    def wait(self, url: str):
        """等待直到可以发送请求"""
        domain = self._get_domain(url)
        config = self._get_config(domain)
        stats = self._domain_stats[domain]
        
        with self._lock:
            # 1. 检查是否被封禁
            if domain in self._blocked_until:
                blocked_until = self._blocked_until[domain]
                if time.time() < blocked_until:
                    wait_time = blocked_until - time.time()
                    logger.warning(
                        f"域名 {domain} 被封禁，等待 {wait_time:.1f} 秒"
                    )
                    time.sleep(wait_time)
                else:
                    del self._blocked_until[domain]
            
            # 2. 计算当前错误率
            error_rate, window_size = self._calculate_error_rate(domain)
            
            # 3. 根据状态决定间隔
            if error_rate > config.error_rate_threshold and window_size >= 10:
                # 高错误率：使用退避延迟
                consecutive_errors = sum(
                    1 for r in list(self._request_windows[domain])[-10:]
                    if not r.success
                )
                base_delay = self._calculate_backoff_delay(
                    domain, config, consecutive_errors
                )
                interval = self._calculate_jitter_delay(base_delay, config)
                logger.debug(
                    f"域名 {domain} 错误率 {error_rate:.1%}，"
                    f"使用退避延迟 {interval:.2f}s"
                )
            elif not stats.warmup_completed:
                # 预热阶段
                interval = self._get_warmup_interval(domain, config)
            else:
                # 正常速率
                interval = stats.current_interval
            
            # 4. 确保最小间隔
            interval = max(interval, config.min_interval)
            
            # 5. 计算实际等待时间
            now = time.time()
            elapsed = now - self._last_request_time.get(domain, 0)
            if elapsed < interval:
                time.sleep(interval - elapsed)
        
        # 获取令牌
        bucket = self._get_bucket(domain)
        bucket.wait()
        
        with self._lock:
            self._last_request_time[domain] = time.time()
    
    def report_error(self, url: str, status_code: int, latency: float = 0.0):
        """
        报告请求错误
        
        Args:
            url: 请求URL
            status_code: HTTP状态码
            latency: 请求延迟（秒）
        """
        domain = self._get_domain(url)
        config = self._get_config(domain)
        stats = self._domain_stats[domain]
        
        with self._lock:
            # 记录到滑动窗口
            self._request_windows[domain].append(RequestRecord(
                timestamp=time.time(),
                success=False,
                status_code=status_code,
                latency=latency
            ))
            
            # 更新统计
            stats.total_requests += 1
            stats.total_errors += 1
            stats.last_error_time = time.time()
            stats.error_codes[status_code] = stats.error_codes.get(status_code, 0) + 1
            
            # 计算错误率
            error_rate, window_size = self._calculate_error_rate(domain)
            
            # 如果是需要退避的状态码，增加间隔
            if status_code in self.BACKOFF_STATUS_CODES:
                new_interval = min(
                    stats.current_interval * config.backoff_factor,
                    config.max_interval
                )
                stats.current_interval = new_interval
                
                logger.warning(
                    f"域名 {domain} HTTP {status_code}，"
                    f"间隔调整为 {new_interval:.1f}s，"
                    f"错误率 {error_rate:.1%}"
                )
                
                # 重建令牌桶
                self._buckets[domain] = TokenBucket(
                    rate=1.0 / new_interval,
                    capacity=config.burst_size
                )
            
            # 错误率过高，临时封禁
            if error_rate > 0.5 and window_size >= 20:
                block_duration = 60 * (1 + error_rate)  # 60-120秒
                self._blocked_until[domain] = time.time() + block_duration
                logger.error(
                    f"域名 {domain} 错误率过高 ({error_rate:.1%})，"
                    f"封禁 {block_duration:.0f} 秒"
                )
        
        # 自动持久化
        self._maybe_persist()
    
    def report_success(self, url: str, latency: float = 0.0):
        """
        报告请求成功
        
        Args:
            url: 请求URL
            latency: 请求延迟（秒）
        """
        domain = self._get_domain(url)
        config = self._get_config(domain)
        stats = self._domain_stats[domain]
        
        with self._lock:
            # 记录到滑动窗口
            self._request_windows[domain].append(RequestRecord(
                timestamp=time.time(),
                success=True,
                status_code=200,
                latency=latency
            ))
            
            # 更新统计
            stats.total_requests += 1
            
            # 计算成功率
            error_rate, window_size = self._calculate_error_rate(domain)
            success_rate = 1 - error_rate
            
            # 高成功率时逐步恢复速率
            default_interval = 1.0 / config.requests_per_second
            if success_rate >= config.success_rate_threshold and window_size >= 20:
                if stats.current_interval > default_interval:
                    new_interval = max(
                        stats.current_interval * config.recovery_factor,
                        default_interval
                    )
                    stats.current_interval = new_interval
                    
                    # 重建令牌桶
                    self._buckets[domain] = TokenBucket(
                        rate=1.0 / new_interval,
                        capacity=config.burst_size
                    )
        
        # 自动持久化
        self._maybe_persist()
    
    def _maybe_persist(self):
        """检查是否需要持久化"""
        if not self._auto_persist or not self._checkpoint_dir:
            return
        
        self._requests_since_persist += 1
        if self._requests_since_persist >= self._persist_interval:
            self.save_checkpoint()
            self._requests_since_persist = 0
    
    def save_checkpoint(self):
        """保存检查点"""
        if not self._checkpoint_dir:
            return
        
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = self._checkpoint_dir / "rate_limiter_checkpoint.json"
        
        with self._lock:
            data = {
                'timestamp': datetime.now().isoformat(),
                'domain_stats': {
                    domain: stats.to_dict()
                    for domain, stats in self._domain_stats.items()
                },
                'blocked_until': {
                    domain: until
                    for domain, until in self._blocked_until.items()
                    if until > time.time()  # 只保存未过期的封禁
                }
            }
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"速率限制器检查点已保存: {checkpoint_file}")
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def _load_checkpoint(self):
        """加载检查点"""
        if not self._checkpoint_dir:
            return
        
        checkpoint_file = self._checkpoint_dir / "rate_limiter_checkpoint.json"
        if not checkpoint_file.exists():
            return
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 恢复域名统计
            for domain, stats_dict in data.get('domain_stats', {}).items():
                self._domain_stats[domain] = DomainStats.from_dict(stats_dict)
            
            # 恢复封禁状态（只恢复未过期的）
            now = time.time()
            for domain, until in data.get('blocked_until', {}).items():
                if until > now:
                    self._blocked_until[domain] = until
            
            logger.info(
                f"速率限制器检查点已加载: "
                f"{len(self._domain_stats)} 个域名统计"
            )
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}")
    
    def get_stats(self, domain: Optional[str] = None) -> Dict:
        """
        获取统计信息
        
        Args:
            domain: 指定域名，None表示所有域名
        
        Returns:
            统计信息字典
        """
        with self._lock:
            if domain:
                stats = self._domain_stats.get(domain)
                if not stats:
                    return {}
                error_rate, window_size = self._calculate_error_rate(domain)
                return {
                    'domain': domain,
                    **stats.to_dict(),
                    'current_error_rate': error_rate,
                    'window_size': window_size,
                    'is_blocked': domain in self._blocked_until and 
                                  self._blocked_until[domain] > time.time()
                }
            else:
                return {
                    domain: self.get_stats(domain)
                    for domain in self._domain_stats.keys()
                }
    
    def reset_domain(self, domain: str):
        """
        重置域名的限速状态
        
        Args:
            domain: 域名
        """
        with self._lock:
            if domain in self._domain_stats:
                del self._domain_stats[domain]
            if domain in self._request_windows:
                del self._request_windows[domain]
            if domain in self._blocked_until:
                del self._blocked_until[domain]
            if domain in self._buckets:
                del self._buckets[domain]
            if domain in self._current_interval:
                del self._current_interval[domain]
            
            logger.info(f"域名 {domain} 限速状态已重置")


# 全局速率限制器实例
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(
    smart: bool = True,
    checkpoint_dir: Optional[str] = None
) -> RateLimiter:
    """
    获取全局速率限制器
    
    Args:
        smart: 是否使用智能限速器
        checkpoint_dir: 检查点目录（仅SmartRateLimiter）
    
    Returns:
        速率限制器实例
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        if smart:
            _global_rate_limiter = SmartRateLimiter(
                checkpoint_dir=checkpoint_dir
            )
        else:
            _global_rate_limiter = AdaptiveRateLimiter()
    return _global_rate_limiter


def set_rate_limiter(limiter: RateLimiter):
    """设置全局速率限制器"""
    global _global_rate_limiter
    _global_rate_limiter = limiter


def reset_rate_limiter():
    """重置全局速率限制器"""
    global _global_rate_limiter
    _global_rate_limiter = None
