"""
代理池管理模块

提供动态代理池功能：
- 代理健康检查
- 失败自动切换
- 支持多种代理协议
"""

import os
import time
import random
import logging
import threading
from typing import Optional, List, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ProxyProtocol(Enum):
    """代理协议类型"""
    HTTP = "http"
    HTTPS = "https"
    SOCKS5 = "socks5"


class RotationStrategy(Enum):
    """代理轮换策略"""
    RANDOM = "random"           # 随机选择
    ROUND_ROBIN = "round_robin"  # 轮询
    LEAST_USED = "least_used"    # 最少使用


@dataclass
class ProxyInfo:
    """代理信息"""
    address: str                    # 代理地址（如 http://ip:port）
    protocol: ProxyProtocol = ProxyProtocol.HTTP
    username: Optional[str] = None
    password: Optional[str] = None
    
    # 健康状态
    is_healthy: bool = True
    last_check: float = 0.0
    failure_count: int = 0
    success_count: int = 0
    avg_response_time: float = 0.0
    
    # 使用统计
    use_count: int = 0
    last_used: float = 0.0
    
    def get_proxy_dict(self) -> Dict[str, str]:
        """获取requests库使用的代理字典"""
        if self.username and self.password:
            protocol = self.protocol.value
            # 格式：protocol://user:pass@host:port
            auth_part = f"{self.username}:{self.password}@"
            address = self.address.replace(f"{protocol}://", f"{protocol}://{auth_part}")
        else:
            address = self.address
        
        return {
            'http': address,
            'https': address
        }
    
    def mark_success(self, response_time: float):
        """标记请求成功"""
        self.success_count += 1
        self.failure_count = 0
        self.is_healthy = True
        
        # 更新平均响应时间
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * 0.7 + 
                                      response_time * 0.3)
    
    def mark_failure(self):
        """标记请求失败"""
        self.failure_count += 1
        if self.failure_count >= 3:
            self.is_healthy = False


@dataclass
class ProxyPoolConfig:
    """代理池配置"""
    check_url: str = "http://httpbin.org/ip"  # 健康检查URL
    check_interval: int = 300                   # 健康检查间隔（秒）
    check_timeout: int = 10                     # 检查超时时间（秒）
    max_failures: int = 3                       # 最大失败次数后标记不可用
    rotation_strategy: RotationStrategy = RotationStrategy.RANDOM
    min_healthy_proxies: int = 1                # 最少保持的健康代理数
    enable_direct: bool = True                  # 无可用代理时允许直连


class ProxyPool:
    """
    代理池管理器
    
    支持：
    - 从环境变量加载代理列表
    - 从配置文件加载代理列表
    - 代理健康检查
    - 自动轮换策略
    """
    
    def __init__(
        self,
        proxies: Optional[List[str]] = None,
        config: Optional[ProxyPoolConfig] = None
    ):
        """
        Args:
            proxies: 代理地址列表
            config: 代理池配置
        """
        self.config = config or ProxyPoolConfig()
        self._proxies: List[ProxyInfo] = []
        self._round_robin_index = 0
        self._lock = threading.Lock()
        self._check_thread: Optional[threading.Thread] = None
        self._running = False
        
        # 初始化代理列表
        if proxies:
            self._add_proxies(proxies)
        else:
            self._load_from_env()
    
    def _load_from_env(self):
        """从环境变量加载代理列表"""
        # 单个代理
        proxy = os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')
        if proxy:
            self._add_proxies([proxy])
        
        # 代理列表（逗号分隔）
        proxy_list = os.getenv('PROXY_LIST', '')
        if proxy_list:
            proxies = [p.strip() for p in proxy_list.split(',') if p.strip()]
            self._add_proxies(proxies)
    
    def _add_proxies(self, proxy_addresses: List[str]):
        """添加代理到池中"""
        for address in proxy_addresses:
            # 解析代理地址
            address = address.strip()
            if not address:
                continue
            
            # 确定协议
            if address.startswith('socks5://'):
                protocol = ProxyProtocol.SOCKS5
            elif address.startswith('https://'):
                protocol = ProxyProtocol.HTTPS
            else:
                protocol = ProxyProtocol.HTTP
                if not address.startswith('http://'):
                    address = f'http://{address}'
            
            proxy_info = ProxyInfo(
                address=address,
                protocol=protocol
            )
            
            # 避免重复
            existing = [p.address for p in self._proxies]
            if address not in existing:
                self._proxies.append(proxy_info)
                logger.info(f"添加代理: {address}")
    
    def add_proxy(self, address: str, username: Optional[str] = None,
                  password: Optional[str] = None):
        """添加单个代理"""
        if not address.startswith(('http://', 'https://', 'socks5://')):
            address = f'http://{address}'
        
        protocol = ProxyProtocol.HTTP
        if 'socks5://' in address:
            protocol = ProxyProtocol.SOCKS5
        elif 'https://' in address:
            protocol = ProxyProtocol.HTTPS
        
        proxy_info = ProxyInfo(
            address=address,
            protocol=protocol,
            username=username,
            password=password
        )
        
        with self._lock:
            self._proxies.append(proxy_info)
    
    def get_proxy(self) -> Optional[Dict[str, str]]:
        """
        获取一个可用代理
        
        Returns:
            代理字典（用于requests库）或 None（直连）
        """
        with self._lock:
            healthy_proxies = [p for p in self._proxies if p.is_healthy]
            
            if not healthy_proxies:
                if self.config.enable_direct:
                    logger.debug("无可用代理，使用直连")
                    return None
                else:
                    raise RuntimeError("无可用代理")
            
            # 根据策略选择
            strategy = self.config.rotation_strategy
            
            if strategy == RotationStrategy.RANDOM:
                proxy = random.choice(healthy_proxies)
            
            elif strategy == RotationStrategy.ROUND_ROBIN:
                self._round_robin_index %= len(healthy_proxies)
                proxy = healthy_proxies[self._round_robin_index]
                self._round_robin_index += 1
            
            elif strategy == RotationStrategy.LEAST_USED:
                proxy = min(healthy_proxies, key=lambda p: p.use_count)
            
            else:
                proxy = healthy_proxies[0]
            
            # 更新使用统计
            proxy.use_count += 1
            proxy.last_used = time.time()
            
            return proxy.get_proxy_dict()
    
    def report_success(self, proxy_dict: Optional[Dict[str, str]], 
                       response_time: float):
        """报告代理请求成功"""
        if proxy_dict is None:
            return
        
        address = proxy_dict.get('http', '')
        with self._lock:
            for proxy in self._proxies:
                if proxy.address in address or address in proxy.address:
                    proxy.mark_success(response_time)
                    break
    
    def report_failure(self, proxy_dict: Optional[Dict[str, str]]):
        """报告代理请求失败"""
        if proxy_dict is None:
            return
        
        address = proxy_dict.get('http', '')
        with self._lock:
            for proxy in self._proxies:
                if proxy.address in address or address in proxy.address:
                    proxy.mark_failure()
                    if not proxy.is_healthy:
                        logger.warning(f"代理已标记为不可用: {proxy.address}")
                    break
    
    def _check_proxy(self, proxy: ProxyInfo) -> bool:
        """检查单个代理可用性"""
        try:
            start_time = time.time()
            response = requests.get(
                self.config.check_url,
                proxies=proxy.get_proxy_dict(),
                timeout=self.config.check_timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                proxy.mark_success(response_time)
                proxy.last_check = time.time()
                return True
            else:
                proxy.mark_failure()
                return False
                
        except Exception as e:
            logger.debug(f"代理检查失败 {proxy.address}: {e}")
            proxy.mark_failure()
            return False
    
    def check_all_proxies(self):
        """检查所有代理可用性"""
        logger.info("开始检查所有代理...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self._check_proxy, proxy): proxy
                for proxy in self._proxies
            }
            
            healthy_count = 0
            for future in as_completed(futures):
                proxy = futures[future]
                try:
                    if future.result():
                        healthy_count += 1
                except Exception as e:
                    logger.debug(f"检查代理异常: {e}")
        
        logger.info(
            f"代理检查完成: {healthy_count}/{len(self._proxies)} 可用"
        )
    
    def start_health_check(self):
        """启动后台健康检查线程"""
        if self._running:
            return
        
        self._running = True
        self._check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._check_thread.start()
        logger.info("代理健康检查线程已启动")
    
    def stop_health_check(self):
        """停止健康检查"""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
    
    def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            self.check_all_proxies()
            time.sleep(self.config.check_interval)
    
    @property
    def healthy_count(self) -> int:
        """获取健康代理数量"""
        return sum(1 for p in self._proxies if p.is_healthy)
    
    @property
    def total_count(self) -> int:
        """获取总代理数量"""
        return len(self._proxies)
    
    def get_stats(self) -> Dict:
        """获取代理池统计信息"""
        with self._lock:
            return {
                'total': len(self._proxies),
                'healthy': self.healthy_count,
                'proxies': [
                    {
                        'address': p.address,
                        'is_healthy': p.is_healthy,
                        'success_count': p.success_count,
                        'failure_count': p.failure_count,
                        'avg_response_time': round(p.avg_response_time, 3),
                        'use_count': p.use_count
                    }
                    for p in self._proxies
                ]
            }


# 全局代理池实例
_global_proxy_pool: Optional[ProxyPool] = None


def get_proxy_pool() -> ProxyPool:
    """获取全局代理池"""
    global _global_proxy_pool
    if _global_proxy_pool is None:
        _global_proxy_pool = ProxyPool()
    return _global_proxy_pool


def set_proxy_pool(pool: ProxyPool):
    """设置全局代理池"""
    global _global_proxy_pool
    _global_proxy_pool = pool
