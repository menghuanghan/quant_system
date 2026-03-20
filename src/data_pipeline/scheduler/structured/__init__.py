"""
结构化数据调度器模块

包含：
- full: 全量采集调度器（历史数据一次性采集）
- increment: 增量采集调度器（定时增量更新）
"""

from . import full
from . import increment

__all__ = ['full', 'increment']

__version__ = '1.0.0'
__author__ = 'Quant Team'
