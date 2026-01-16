"""
数据采集器基类模块
提供数据源连接管理、自动切换和通用功能
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
from functools import wraps
from enum import Enum

import pandas as pd
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """数据源枚举"""
    TUSHARE = "tushare"
    AKSHARE = "akshare"
    BAOSTOCK = "baostock"


class DataSourcePriority:
    """数据源优先级管理"""
    DEFAULT_ORDER = [DataSource.TUSHARE, DataSource.AKSHARE, DataSource.BAOSTOCK]
    
    def __init__(self, order: Optional[List[DataSource]] = None):
        self.order = order or self.DEFAULT_ORDER
    
    def get_next(self, current: Optional[DataSource] = None) -> Optional[DataSource]:
        """获取下一个备用数据源"""
        if current is None:
            return self.order[0] if self.order else None
        
        try:
            idx = self.order.index(current)
            if idx + 1 < len(self.order):
                return self.order[idx + 1]
        except ValueError:
            pass
        return None


class DataSourceManager:
    """数据源连接管理器（单例模式）"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not DataSourceManager._initialized:
            self._tushare_api = None
            self._baostock_logged_in = False
            self._akshare_available = False
            DataSourceManager._initialized = True
    
    @property
    def tushare_api(self):
        """获取Tushare API实例（懒加载）"""
        if self._tushare_api is None:
            try:
                import tushare as ts
                token = os.getenv("TUSHARE_TOKEN")
                if not token:
                    raise ValueError("TUSHARE_TOKEN环境变量未设置")
                ts.set_token(token)
                self._tushare_api = ts.pro_api()
                logger.info("Tushare API初始化成功")
            except Exception as e:
                logger.error(f"Tushare API初始化失败: {e}")
                raise
        return self._tushare_api
    
    def ensure_baostock_login(self) -> bool:
        """确保BaoStock已登录"""
        if not self._baostock_logged_in:
            try:
                import baostock as bs
                lg = bs.login()
                if lg.error_code == '0':
                    self._baostock_logged_in = True
                    logger.info("BaoStock登录成功")
                else:
                    logger.error(f"BaoStock登录失败: {lg.error_msg}")
                    return False
            except Exception as e:
                logger.error(f"BaoStock登录异常: {e}")
                return False
        return True
    
    def logout_baostock(self):
        """登出BaoStock"""
        if self._baostock_logged_in:
            try:
                import baostock as bs
                bs.logout()
                self._baostock_logged_in = False
                logger.info("BaoStock已登出")
            except Exception as e:
                logger.warning(f"BaoStock登出异常: {e}")
    
    def check_akshare_available(self) -> bool:
        """检查AkShare是否可用"""
        try:
            import akshare
            self._akshare_available = True
            return True
        except ImportError:
            logger.warning("AkShare未安装")
            return False


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                     backoff_factor: float = 2.0):
    """重试装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"函数 {func.__name__} 第 {attempt + 1} 次调用失败: {e}, "
                            f"{current_delay:.1f}秒后重试..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"函数 {func.__name__} 在 {max_retries} 次重试后仍然失败"
                        )
            
            raise last_exception
        return wrapper
    return decorator


def fallback_on_error(fallback_sources: Optional[List[DataSource]] = None):
    """数据源降级装饰器 - 用于类方法"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            sources = fallback_sources or DataSourcePriority.DEFAULT_ORDER
            last_exception = None
            
            for source in sources:
                try:
                    # 将当前数据源传递给函数
                    result = func(self, *args, data_source=source, **kwargs)
                    if result is not None and not result.empty:
                        logger.info(f"使用 {source.value} 成功获取数据")
                        return result
                except Exception as e:
                    last_exception = e
                    logger.warning(f"数据源 {source.value} 获取失败: {e}")
                    continue
            
            if last_exception:
                raise last_exception
            return pd.DataFrame()
        return wrapper
    return decorator


class BaseCollector(ABC):
    """数据采集器基类"""
    
    def __init__(self):
        self.source_manager = DataSourceManager()
        self.priority = DataSourcePriority()
    
    @property
    def tushare_api(self):
        """获取Tushare API"""
        return self.source_manager.tushare_api
    
    def _standardize_columns(self, df: pd.DataFrame, 
                            column_mapping: Dict[str, str]) -> pd.DataFrame:
        """标准化列名
        
        Args:
            df: 原始DataFrame
            column_mapping: 列名映射字典 {原列名: 标准列名}
        
        Returns:
            标准化后的DataFrame
        """
        if df.empty:
            return df
        
        # 只重命名存在的列
        existing_mapping = {
            k: v for k, v in column_mapping.items() 
            if k in df.columns
        }
        return df.rename(columns=existing_mapping)
    
    def _ensure_columns(self, df: pd.DataFrame, 
                        required_columns: List[str],
                        fill_value: Any = None) -> pd.DataFrame:
        """确保DataFrame包含所有必需的列
        
        Args:
            df: DataFrame
            required_columns: 必需的列列表
            fill_value: 缺失列的填充值
        
        Returns:
            包含所有必需列的DataFrame
        """
        for col in required_columns:
            if col not in df.columns:
                df[col] = fill_value
        return df
    
    def _convert_date_format(self, df: pd.DataFrame, 
                             date_columns: List[str],
                             from_format: str = "%Y%m%d",
                             to_format: str = "%Y-%m-%d") -> pd.DataFrame:
        """转换日期格式
        
        Args:
            df: DataFrame
            date_columns: 日期列列表
            from_format: 原始日期格式
            to_format: 目标日期格式
        
        Returns:
            转换后的DataFrame
        """
        for col in date_columns:
            if col in df.columns and df[col].notna().any():
                try:
                    df[col] = pd.to_datetime(
                        df[col].astype(str), 
                        format=from_format, 
                        errors='coerce'
                    ).dt.strftime(to_format)
                except Exception as e:
                    logger.warning(f"日期列 {col} 格式转换失败: {e}")
        return df
    
    def _safe_merge(self, left: pd.DataFrame, right: pd.DataFrame,
                    on: str, how: str = 'left') -> pd.DataFrame:
        """安全合并两个DataFrame
        
        Args:
            left: 左侧DataFrame
            right: 右侧DataFrame
            on: 合并键
            how: 合并方式
        
        Returns:
            合并后的DataFrame
        """
        if left.empty:
            return right
        if right.empty:
            return left
        
        return pd.merge(left, right, on=on, how=how, suffixes=('', '_dup'))
    
    @abstractmethod
    def collect(self, **kwargs) -> pd.DataFrame:
        """采集数据的抽象方法，子类必须实现"""
        pass


class CollectorRegistry:
    """采集器注册表，用于管理所有采集器"""
    
    _collectors: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """注册采集器的装饰器"""
        def decorator(collector_class: type):
            cls._collectors[name] = collector_class
            return collector_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """获取采集器类"""
        return cls._collectors.get(name)
    
    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已注册的采集器"""
        return list(cls._collectors.keys())


# 标准化字段定义
class StandardFields:
    """标准化字段定义"""
    
    # 证券基础信息标准字段
    SECURITY_BASIC = [
        'ts_code',      # 证券代码（含交易所后缀）
        'symbol',       # 证券代码（纯数字）
        'name',         # 证券名称
        'area',         # 地区
        'industry',     # 行业
        'fullname',     # 公司全称
        'enname',       # 英文名称
        'cnspell',      # 拼音缩写
        'market',       # 市场类型
        'exchange',     # 交易所代码
        'curr_type',    # 交易货币
        'list_status',  # 上市状态
        'list_date',    # 上市日期
        'delist_date',  # 退市日期
        'is_hs',        # 是否沪深港通标的
    ]
    
    # 交易日历标准字段
    TRADE_CALENDAR = [
        'exchange',     # 交易所
        'cal_date',     # 日历日期
        'is_open',      # 是否交易日
        'pretrade_date',  # 上一个交易日
    ]
    
    # 股票曾用名标准字段
    NAME_CHANGE = [
        'ts_code',      # 证券代码
        'name',         # 证券名称
        'start_date',   # 开始日期
        'end_date',     # 结束日期
        'ann_date',     # 公告日期
        'change_reason',  # 变更原因
    ]
    
    # 停复牌信息标准字段
    SUSPEND_INFO = [
        'ts_code',      # 证券代码
        'trade_date',   # 交易日期
        'suspend_type', # 停复牌类型
        'suspend_timing',  # 停牌时间段
        'suspend_reason',  # 停牌原因
    ]
