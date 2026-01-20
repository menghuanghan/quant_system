"""
行情过滤器模块 (Market Data Filter)

用于舆情采集的事件驱动过滤：
- 仅采集异动股票（涨跌幅 > 阈值）的舆情
- 大幅减少无效数据采集，节省存储空间

设计思路：
- 舆情与股价往往相关，平淡行情的舆情信息价值较低
- 对于历史回溯，预先筛选异动日的股票，再针对性采集
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MarketEvent:
    """市场异动事件"""
    ts_code: str
    trade_date: str
    pct_change: float          # 涨跌幅 %
    turnover_rate: float       # 换手率 %
    volume_ratio: float        # 量比
    event_type: str            # 'surge', 'plunge', 'limit_up', 'limit_down', 'high_volume'


class MarketDataFilter:
    """
    行情数据过滤器
    
    用于从历史行情中筛选异动股票，为舆情采集提供目标池。
    
    异动判定规则：
    - surge: 涨幅 > 5%
    - plunge: 跌幅 > 5%
    - limit_up: 涨停
    - limit_down: 跌停
    - high_volume: 换手率 > 10% 或量比 > 3
    
    Example:
        >>> filter = MarketDataFilter()
        >>> filter.load_market_data("data/raw/structured/market_data/stock_daily")
        >>> 
        >>> # 获取某日的异动股票
        >>> targets = filter.get_abnormal_stocks("20240115")
        >>> print(targets)  # ['000001.SZ', '600519.SH', ...]
        >>> 
        >>> # 获取某月的异动日历
        >>> calendar = filter.get_abnormal_calendar("2024-01")
    """
    
    # 异动阈值配置
    DEFAULT_THRESHOLDS = {
        'surge_pct': 5.0,           # 大涨阈值
        'plunge_pct': -5.0,         # 大跌阈值
        'limit_up_pct': 9.5,        # 涨停阈值（考虑精度）
        'limit_down_pct': -9.5,     # 跌停阈值
        'high_turnover': 10.0,      # 高换手阈值
        'high_volume_ratio': 3.0,   # 高量比阈值
    }
    
    def __init__(
        self,
        market_data_dir: Optional[Path] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            market_data_dir: 行情数据目录
            thresholds: 自定义阈值
        """
        self.market_data_dir = market_data_dir or Path("data/raw/structured/market_data/stock_daily")
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        
        self._market_data: Optional[pd.DataFrame] = None
        self._cache: Dict[str, List[str]] = {}  # trade_date -> ts_codes
    
    def load_market_data(
        self,
        market_data_path: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        加载行情数据
        
        Args:
            market_data_path: 数据路径（Parquet 文件或目录）
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
        """
        path = Path(market_data_path) if market_data_path else self.market_data_dir
        
        if path.is_file():
            # 单文件
            df = pd.read_parquet(path)
        elif path.is_dir():
            # 目录：合并所有 Parquet 文件
            dfs = []
            for parquet_file in path.glob("*.parquet"):
                dfs.append(pd.read_parquet(parquet_file))
            if not dfs:
                logger.warning(f"未找到 Parquet 文件: {path}")
                return
            df = pd.concat(dfs, ignore_index=True)
        else:
            logger.error(f"路径不存在: {path}")
            return
        
        # 日期筛选
        if 'trade_date' in df.columns:
            df['trade_date'] = df['trade_date'].astype(str)
            if start_date:
                df = df[df['trade_date'] >= start_date]
            if end_date:
                df = df[df['trade_date'] <= end_date]
        
        self._market_data = df
        logger.info(f"加载行情数据: {len(df)} 条记录")
    
    def _ensure_loaded(self):
        """确保数据已加载"""
        if self._market_data is None:
            self.load_market_data()
            if self._market_data is None:
                raise RuntimeError("行情数据未加载")
    
    def get_abnormal_stocks(
        self,
        trade_date: str,
        event_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        获取某日的异动股票列表
        
        Args:
            trade_date: 交易日期 (YYYYMMDD)
            event_types: 筛选事件类型（默认全部）
        
        Returns:
            股票代码列表
        """
        # 检查缓存
        cache_key = f"{trade_date}_{event_types}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        self._ensure_loaded()
        
        df = self._market_data
        day_df = df[df['trade_date'] == trade_date].copy()
        
        if day_df.empty:
            return []
        
        # 筛选异动
        abnormal_codes = set()
        
        # 大涨
        if not event_types or 'surge' in event_types:
            surge = day_df[day_df['pct_chg'] >= self.thresholds['surge_pct']]
            abnormal_codes.update(surge['ts_code'].tolist())
        
        # 大跌
        if not event_types or 'plunge' in event_types:
            plunge = day_df[day_df['pct_chg'] <= self.thresholds['plunge_pct']]
            abnormal_codes.update(plunge['ts_code'].tolist())
        
        # 涨停
        if not event_types or 'limit_up' in event_types:
            limit_up = day_df[day_df['pct_chg'] >= self.thresholds['limit_up_pct']]
            abnormal_codes.update(limit_up['ts_code'].tolist())
        
        # 跌停
        if not event_types or 'limit_down' in event_types:
            limit_down = day_df[day_df['pct_chg'] <= self.thresholds['limit_down_pct']]
            abnormal_codes.update(limit_down['ts_code'].tolist())
        
        # 高换手
        if not event_types or 'high_volume' in event_types:
            if 'turnover_rate' in day_df.columns:
                high_turn = day_df[day_df['turnover_rate'] >= self.thresholds['high_turnover']]
                abnormal_codes.update(high_turn['ts_code'].tolist())
        
        result = list(abnormal_codes)
        self._cache[cache_key] = result
        
        return result
    
    def get_abnormal_calendar(
        self,
        month: str,
        min_stocks: int = 10
    ) -> Dict[str, List[str]]:
        """
        获取某月的异动日历
        
        Args:
            month: 月份 (YYYY-MM)
            min_stocks: 最小异动股票数（过滤非交易日）
        
        Returns:
            {trade_date: [ts_codes], ...}
        """
        self._ensure_loaded()
        
        # 转换月份格式
        year, mon = month.split('-')
        start_date = f"{year}{mon}01"
        end_date = f"{year}{mon}31"
        
        df = self._market_data
        month_df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
        
        calendar = {}
        for trade_date in month_df['trade_date'].unique():
            stocks = self.get_abnormal_stocks(trade_date)
            if len(stocks) >= min_stocks:
                calendar[trade_date] = stocks
        
        return calendar
    
    def get_target_pool(
        self,
        trade_date: str,
        pct_threshold: float = 3.0
    ) -> List[str]:
        """
        获取目标池（简化版）
        
        用于舆情采集的目标股票筛选。
        
        Args:
            trade_date: 交易日期
            pct_threshold: 涨跌幅阈值（绝对值）
        
        Returns:
            需要采集舆情的股票列表
        """
        self._ensure_loaded()
        
        df = self._market_data
        day_df = df[df['trade_date'] == trade_date]
        
        if day_df.empty:
            return []
        
        # abs(pct_change) > threshold
        if 'pct_chg' in day_df.columns:
            abnormal = day_df[abs(day_df['pct_chg']) >= pct_threshold]
            return abnormal['ts_code'].tolist()
        
        return []
    
    def analyze_abnormal_stats(self) -> Dict[str, any]:
        """
        分析异动统计
        
        用于了解数据分布，优化采集策略
        """
        self._ensure_loaded()
        
        df = self._market_data
        
        stats = {
            'total_records': len(df),
            'date_range': (df['trade_date'].min(), df['trade_date'].max()),
            'unique_stocks': df['ts_code'].nunique(),
            'unique_dates': df['trade_date'].nunique(),
        }
        
        # 统计各类异动数量
        if 'pct_chg' in df.columns:
            stats['surge_count'] = len(df[df['pct_chg'] >= 5.0])
            stats['plunge_count'] = len(df[df['pct_chg'] <= -5.0])
            stats['limit_up_count'] = len(df[df['pct_chg'] >= 9.5])
            stats['limit_down_count'] = len(df[df['pct_chg'] <= -9.5])
        
        # 计算日均异动数
        if stats['unique_dates'] > 0:
            total_abnormal = stats.get('surge_count', 0) + stats.get('plunge_count', 0)
            stats['avg_abnormal_per_day'] = total_abnormal / stats['unique_dates']
        
        return stats


class SentimentTargetSelector:
    """
    舆情采集目标选择器
    
    组合行情过滤和其他规则，生成最终的采集目标。
    
    选择逻辑：
    1. 异动股票必采集
    2. 重要事件（ST、停复牌）必采集
    3. 行业龙头持续关注
    """
    
    def __init__(self, market_filter: Optional[MarketDataFilter] = None):
        self.market_filter = market_filter or MarketDataFilter()
        
        # 重点关注列表（行业龙头等）
        self.watchlist: Set[str] = set()
    
    def add_to_watchlist(self, ts_codes: List[str]):
        """添加到关注列表"""
        self.watchlist.update(ts_codes)
    
    def load_watchlist(self, watchlist_path: str):
        """从文件加载关注列表"""
        path = Path(watchlist_path)
        if path.exists():
            with open(path, 'r') as f:
                codes = [line.strip() for line in f if line.strip()]
                self.watchlist.update(codes)
    
    def select_targets(
        self,
        trade_date: str,
        pct_threshold: float = 3.0,
        include_watchlist: bool = True
    ) -> List[str]:
        """
        选择采集目标
        
        Args:
            trade_date: 交易日期
            pct_threshold: 异动阈值
            include_watchlist: 是否包含关注列表
        
        Returns:
            目标股票列表
        """
        targets = set()
        
        # 1. 异动股票
        abnormal = self.market_filter.get_target_pool(trade_date, pct_threshold)
        targets.update(abnormal)
        
        # 2. 关注列表
        if include_watchlist:
            targets.update(self.watchlist)
        
        return list(targets)
    
    def estimate_workload(
        self,
        start_date: str,
        end_date: str,
        pct_threshold: float = 3.0
    ) -> Dict[str, any]:
        """
        估算采集工作量
        
        用于计划资源和时间
        """
        self.market_filter._ensure_loaded()
        
        df = self.market_filter._market_data
        date_df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
        
        total_targets = 0
        dates = date_df['trade_date'].unique()
        
        for trade_date in dates:
            targets = self.select_targets(trade_date, pct_threshold, include_watchlist=False)
            total_targets += len(targets)
        
        return {
            'date_range': (start_date, end_date),
            'trading_days': len(dates),
            'total_targets': total_targets,
            'avg_targets_per_day': total_targets / len(dates) if dates.size else 0,
            'watchlist_size': len(self.watchlist)
        }


# ============== 便捷函数 ==============

_global_filter: Optional[MarketDataFilter] = None


def get_market_filter() -> MarketDataFilter:
    """获取全局行情过滤器"""
    global _global_filter
    if _global_filter is None:
        _global_filter = MarketDataFilter()
        _global_filter.load_market_data()
    return _global_filter


def get_abnormal_stocks(trade_date: str) -> List[str]:
    """便捷函数：获取异动股票"""
    return get_market_filter().get_abnormal_stocks(trade_date)


def get_sentiment_targets(trade_date: str, pct_threshold: float = 3.0) -> List[str]:
    """便捷函数：获取舆情采集目标"""
    return get_market_filter().get_target_pool(trade_date, pct_threshold)
