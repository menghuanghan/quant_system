"""
反爬虫配置模块

提供完整的反爬虫策略配置：
1. Playwright 浏览器指纹伪造
2. 动态 Cookie 获取和管理
3. 请求头伪装策略
4. 代理池集成
5. 自适应限速
"""

import os
import time
import random
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CrawlerStrategy(Enum):
    """爬虫策略"""
    API_ONLY = "api_only"           # 仅API采集
    BROWSER_FIRST = "browser_first" # 优先浏览器
    HYBRID = "hybrid"               # 混合策略（API失败用浏览器）


@dataclass
class SiteConfig:
    """站点反爬配置"""
    domain: str
    # 请求配置
    min_interval: float = 1.0        # 最小请求间隔（秒）
    max_interval: float = 3.0        # 最大请求间隔
    burst_requests: int = 3          # 允许的突发请求数
    timeout: int = 30                # 请求超时
    max_retries: int = 5             # 最大重试次数
    
    # 策略配置
    strategy: CrawlerStrategy = CrawlerStrategy.HYBRID
    use_proxy: bool = False          # 是否使用代理
    need_cookie: bool = True         # 是否需要Cookie
    need_login: bool = False         # 是否需要登录
    
    # 反检测配置
    random_ua: bool = True           # 随机UA
    random_referer: bool = True      # 随机Referer
    random_delay: bool = True        # 随机延迟
    
    # 浏览器配置
    headless: bool = True            # 无头模式
    block_media: bool = True         # 屏蔽媒体资源
    
    def get_delay(self) -> float:
        """获取随机延迟时间"""
        if self.random_delay:
            return random.uniform(self.min_interval, self.max_interval)
        return self.min_interval


# ============== 预置站点配置 ==============

# 股吧配置
GUBA_CONFIG = SiteConfig(
    domain="guba.eastmoney.com",
    min_interval=0.5,
    max_interval=2.0,
    burst_requests=5,
    strategy=CrawlerStrategy.HYBRID,
    need_cookie=True,
    need_login=False,
    random_ua=True,
)

# 雪球配置
XUEQIU_CONFIG = SiteConfig(
    domain="xueqiu.com",
    min_interval=1.0,
    max_interval=3.0,
    burst_requests=3,
    strategy=CrawlerStrategy.BROWSER_FIRST,
    need_cookie=True,
    need_login=True,  # 雪球需要登录获取完整数据
    random_ua=True,
    headless=False,   # 登录时需要有头模式
)

# 互动易配置
CNINFO_CONFIG = SiteConfig(
    domain="irm.cninfo.com.cn",
    min_interval=0.3,
    max_interval=1.0,
    burst_requests=10,
    strategy=CrawlerStrategy.API_ONLY,
    need_cookie=False,
    need_login=False,
)

# 东方财富配置
EASTMONEY_CONFIG = SiteConfig(
    domain="eastmoney.com",
    min_interval=0.5,
    max_interval=1.5,
    burst_requests=5,
    strategy=CrawlerStrategy.HYBRID,
    need_cookie=True,
    need_login=False,
)


# ============== 请求头模板 ==============

class HeaderTemplates:
    """请求头模板管理"""
    
    # 基础浏览器请求头
    BROWSER_HEADERS = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'Accept-Encoding': 'gzip, deflate, br',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Microsoft Edge";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
    }
    
    # API请求头
    API_HEADERS = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json;charset=UTF-8',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }
    
    # XHR请求头（股吧等AJAX接口）
    XHR_HEADERS = {
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'X-Requested-With': 'XMLHttpRequest',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }
    
    @classmethod
    def get_guba_headers(cls, symbol: str) -> Dict[str, str]:
        """获取股吧请求头"""
        headers = cls.XHR_HEADERS.copy()
        headers.update({
            'Referer': f'https://guba.eastmoney.com/list,{symbol}.html',
            'Origin': 'https://guba.eastmoney.com',
            'Host': 'guba.eastmoney.com',
        })
        return headers
    
    @classmethod
    def get_xueqiu_headers(cls, symbol: str = '') -> Dict[str, str]:
        """获取雪球请求头"""
        headers = cls.API_HEADERS.copy()
        headers.update({
            'Referer': f'https://xueqiu.com/S/{symbol}' if symbol else 'https://xueqiu.com/',
            'Origin': 'https://xueqiu.com',
            'Host': 'xueqiu.com',
        })
        return headers
    
    @classmethod
    def get_eastmoney_headers(cls, api_path: str = '') -> Dict[str, str]:
        """获取东方财富请求头"""
        headers = cls.API_HEADERS.copy()
        headers.update({
            'Referer': 'https://quote.eastmoney.com/',
            'Origin': 'https://quote.eastmoney.com',
        })
        return headers


# ============== 浏览器指纹配置 ==============

BROWSER_FINGERPRINTS = [
    # Windows + Chrome
    {
        "platform": "Win32",
        "viewport": {"width": 1920, "height": 1080},
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "timezone": "Asia/Shanghai",
        "locale": "zh-CN",
        "webgl_vendor": "Google Inc. (NVIDIA)",
        "webgl_renderer": "ANGLE (NVIDIA, NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0, D3D11)",
        "screen": {"width": 1920, "height": 1080, "depth": 24},
    },
    # Windows + Edge
    {
        "platform": "Win32",
        "viewport": {"width": 1536, "height": 864},
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
        "timezone": "Asia/Shanghai",
        "locale": "zh-CN",
        "webgl_vendor": "Google Inc. (Intel)",
        "webgl_renderer": "ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11 vs_5_0 ps_5_0, D3D11)",
        "screen": {"width": 1536, "height": 864, "depth": 24},
    },
    # Mac + Chrome
    {
        "platform": "MacIntel",
        "viewport": {"width": 1440, "height": 900},
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "timezone": "Asia/Shanghai",
        "locale": "zh-CN",
        "webgl_vendor": "Apple Inc.",
        "webgl_renderer": "Apple M1",
        "screen": {"width": 1440, "height": 900, "depth": 30},
    },
    # Windows + Firefox
    {
        "platform": "Win32",
        "viewport": {"width": 1366, "height": 768},
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "timezone": "Asia/Shanghai",
        "locale": "zh-CN",
        "webgl_vendor": "NVIDIA Corporation",
        "webgl_renderer": "GeForce GTX 1050/PCIe/SSE2",
        "screen": {"width": 1366, "height": 768, "depth": 24},
    },
]


# ============== 反检测脚本 ==============

STEALTH_JS = """
// 隐藏 webdriver 属性
Object.defineProperty(navigator, 'webdriver', {
    get: () => undefined
});

// 伪造 Chrome 对象
window.chrome = {
    runtime: {},
    loadTimes: function() {},
    csi: function() {},
    app: {}
};

// 伪造 plugins
Object.defineProperty(navigator, 'plugins', {
    get: () => {
        return [
            {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer'},
            {name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai'},
            {name: 'Native Client', filename: 'internal-nacl-plugin'}
        ];
    }
});

// 伪造 languages
Object.defineProperty(navigator, 'languages', {
    get: () => ['zh-CN', 'zh', 'en-US', 'en']
});

// 伪造 permissions
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) => (
    parameters.name === 'notifications' ?
        Promise.resolve({ state: Notification.permission }) :
        originalQuery(parameters)
);

// 伪造 WebGL 指纹
const getParameterOriginal = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(parameter) {
    if (parameter === 37445) {
        return 'Google Inc. (NVIDIA)';
    }
    if (parameter === 37446) {
        return 'ANGLE (NVIDIA, NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0, D3D11)';
    }
    return getParameterOriginal.call(this, parameter);
};

// 禁用 automation 标记
delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;

// 伪造 connection
Object.defineProperty(navigator, 'connection', {
    get: () => ({
        downlink: 10,
        effectiveType: '4g',
        rtt: 50,
        saveData: false
    })
});

// 伪造 hardwareConcurrency
Object.defineProperty(navigator, 'hardwareConcurrency', {
    get: () => 8
});

// 伪造 deviceMemory
Object.defineProperty(navigator, 'deviceMemory', {
    get: () => 8
});

// 防止 iframe 检测
try {
    if (window.frameElement) {
        Object.defineProperty(window, 'frameElement', {
            get: () => null
        });
    }
} catch(e) {}

console.log('[Stealth] Anti-detection script loaded');
"""


def get_anti_crawler_config(domain: str) -> SiteConfig:
    """
    获取站点反爬配置
    
    Args:
        domain: 站点域名
    
    Returns:
        站点配置对象
    """
    configs = {
        'guba.eastmoney.com': GUBA_CONFIG,
        'xueqiu.com': XUEQIU_CONFIG,
        'irm.cninfo.com.cn': CNINFO_CONFIG,
        'eastmoney.com': EASTMONEY_CONFIG,
    }
    
    # 精确匹配
    if domain in configs:
        return configs[domain]
    
    # 模糊匹配
    for key, config in configs.items():
        if key in domain or domain in key:
            return config
    
    # 默认配置
    return SiteConfig(
        domain=domain,
        min_interval=1.0,
        max_interval=2.0,
        strategy=CrawlerStrategy.HYBRID,
    )


def get_random_fingerprint() -> Dict[str, Any]:
    """获取随机浏览器指纹"""
    return random.choice(BROWSER_FINGERPRINTS)
