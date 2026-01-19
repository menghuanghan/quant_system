"""
市场热度数据采集器

采集市场热度相关数据：
1. 实时热榜: 东方财富/雪球热榜 (AkShare)
2. 历史热度代理指标: 换手率 + 新闻条数合成

输出字段: [trade_date, ts_code, rank, search_index_proxy, source]
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from ..base import UnstructuredCollector, DataSourceType

load_dotenv()

logger = logging.getLogger(__name__)


class HotListSource(Enum):
    """热榜数据源"""
    EASTMONEY = "eastmoney"     # 东方财富
    XUEQIU = "xueqiu"           # 雪球
    BAIDU = "baidu"             # 百度指数
    SYNTHETIC = "synthetic"     # 合成指标


@dataclass
class HeatConfig:
    """热度计算配置"""
    # 换手率权重
    turnover_weight: float = 0.6
    # 新闻数量权重
    news_weight: float = 0.4
    # 换手率归一化参数
    turnover_cap: float = 30.0  # 换手率上限百分比
    # 新闻数量归一化参数
    news_cap: int = 50  # 单日新闻上限


class MarketHeatCollector(UnstructuredCollector):
    """
    市场热度采集器
    
    功能：
    1. 实时热榜采集（AkShare）
    2. 历史热度代理指标计算（换手率 + 新闻条数）
    
    输出字段:
        - trade_date: 交易日期
        - ts_code: 股票代码
        - rank: 热度排名（仅实时热榜有值）
        - search_index_proxy: 搜索热度代理指标 (0-100)
        - turnover_rate: 换手率
        - news_count: 新闻条数
        - source: 数据来源
    """
    
    STANDARD_FIELDS = [
        'trade_date',
        'ts_code',
        'name',
        'rank',
        'search_index_proxy',
        'turnover_rate',
        'news_count',
        'source',
    ]
    
    def __init__(self, config: Optional[HeatConfig] = None):
        super().__init__()
        self.config = config or HeatConfig()
        self._ak = None
        self._ts = None
        self._tushare_token = os.getenv('TUSHARE_TOKEN')
    
    @property
    def ak(self):
        """懒加载 AkShare"""
        if self._ak is None:
            import akshare as ak
            self._ak = ak
        return self._ak
    
    @property
    def ts(self):
        """懒加载 Tushare"""
        if self._ts is None and self._tushare_token:
            import tushare as ts
            ts.set_token(self._tushare_token)
            self._ts = ts.pro_api()
        return self._ts
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        ts_codes: Optional[List[str]] = None,
        source: HotListSource = HotListSource.SYNTHETIC,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集市场热度数据
        
        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            ts_codes: 股票代码列表（为空则采集热榜全部股票）
            source: 数据来源
        
        Returns:
            标准化的热度数据 DataFrame
        """
        if source == HotListSource.SYNTHETIC:
            return self._collect_synthetic_heat(start_date, end_date, ts_codes)
        elif source == HotListSource.EASTMONEY:
            return self._collect_eastmoney_hotlist()
        elif source == HotListSource.XUEQIU:
            return self._collect_xueqiu_hotlist()
        else:
            self.logger.warning(f"不支持的数据源: {source}")
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
    
    # ==================== 实时热榜采集 ====================
    
    def _collect_eastmoney_hotlist(self) -> pd.DataFrame:
        """
        采集东方财富实时热榜
        
        使用 AkShare 的 stock_hot_rank_em 接口
        """
        try:
            # 东方财富人气榜
            df = self.ak.stock_hot_rank_em()
            
            if df is None or df.empty:
                self.logger.warning("东方财富热榜数据为空")
                return pd.DataFrame(columns=self.STANDARD_FIELDS)
            
            # 打印列名用于调试
            self.logger.debug(f"东方财富热榜列名: {list(df.columns)}")
            
            # 字段映射（处理不同版本的AkShare）
            column_map = {
                '代码': 'ts_code',
                '股票代码': 'ts_code',
                '股票名称': 'name',
                '名称': 'name',
                '最新价': 'close',
                '涨跌幅': 'pct_chg',
                '排名': 'rank',
                '序号': 'rank',
            }
            
            for old_col, new_col in column_map.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
            
            # 如果没有排名字段，生成排名
            if 'rank' not in df.columns:
                df['rank'] = range(1, len(df) + 1)
            
            # 转换股票代码格式
            if 'ts_code' in df.columns:
                df['ts_code'] = df['ts_code'].apply(self._to_ts_code)
            
            df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
            df['source'] = HotListSource.EASTMONEY.value
            
            # 热度代理指标：使用排名反向映射 (排名越高，指标越大)
            max_rank = df['rank'].max()
            df['search_index_proxy'] = ((max_rank - df['rank'] + 1) / max_rank * 100).round(2)
            
            # 填充缺失字段
            df['turnover_rate'] = None
            df['news_count'] = None
            
            return self._ensure_fields(df)
            
        except Exception as e:
            self.logger.error(f"采集东方财富热榜失败: {e}")
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
    
    def _collect_xueqiu_hotlist(self) -> pd.DataFrame:
        """
        采集雪球实时热榜
        
        尝试使用 AkShare 的多个雪球相关接口
        """
        try:
            df = None
            
            # 尝试不同的接口
            interfaces = [
                ('stock_hot_follow_xq', {}),           # 雪球关注排行榜
                ('stock_hot_deal_xq', {}),             # 雪球交易排行榜
                ('stock_hot_tweet_xq', {}),            # 雪球讨论排行榜
            ]
            
            for interface_name, params in interfaces:
                try:
                    if hasattr(self.ak, interface_name):
                        func = getattr(self.ak, interface_name)
                        df = func(**params)
                        if df is not None and not df.empty:
                            self.logger.info(f"使用 {interface_name} 接口成功")
                            break
                except Exception as e:
                    self.logger.debug(f"{interface_name} 接口调用失败: {e}")
                    continue
            
            if df is None or df.empty:
                self.logger.warning("雪球热榜数据为空")
                return pd.DataFrame(columns=self.STANDARD_FIELDS)
            
            # 打印列名用于调试
            self.logger.debug(f"雪球热榜列名: {list(df.columns)}")
            
            # 字段映射（处理不同版本和接口）
            column_map = {
                '股票代码': 'ts_code',
                '代码': 'ts_code',
                '股票简称': 'name',
                '名称': 'name',
                '关注人数': 'follow_count',
                '讨论数': 'follow_count',
            }
            
            for old_col, new_col in column_map.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
            
            # 生成排名
            df['rank'] = range(1, len(df) + 1)
            
            # 转换股票代码格式
            if 'ts_code' in df.columns:
                df['ts_code'] = df['ts_code'].apply(self._to_ts_code)
            
            df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
            df['source'] = HotListSource.XUEQIU.value
            
            # 热度代理指标
            if 'follow_count' in df.columns:
                follow_count = pd.to_numeric(df['follow_count'], errors='coerce').fillna(0)
                max_follow = follow_count.max()
                if max_follow > 0:
                    df['search_index_proxy'] = (follow_count / max_follow * 100).round(2)
                else:
                    df['search_index_proxy'] = 50.0
            else:
                max_rank = len(df)
                df['search_index_proxy'] = ((max_rank - df['rank'] + 1) / max_rank * 100).round(2)
            
            # 填充缺失字段
            df['turnover_rate'] = None
            df['news_count'] = None
            
            return self._ensure_fields(df)
            
        except Exception as e:
            self.logger.error(f"采集雪球热榜失败: {e}")
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
    
    # ==================== 历史热度代理指标 ====================
    
    def _collect_synthetic_heat(
        self,
        start_date: str,
        end_date: str,
        ts_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        计算合成热度代理指标
        
        公式: search_index_proxy = turnover_weight * norm_turnover + news_weight * norm_news
        
        其中:
        - norm_turnover = min(turnover_rate / turnover_cap, 1.0) * 100
        - norm_news = min(news_count / news_cap, 1.0) * 100
        """
        all_data = []
        
        # 1. 获取换手率数据
        turnover_df = self._get_turnover_data(start_date, end_date, ts_codes)
        
        # 2. 获取新闻数量数据
        news_df = self._get_news_count(start_date, end_date, ts_codes)
        
        if turnover_df.empty:
            self.logger.warning("换手率数据为空，无法计算合成指标")
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        # 3. 合并数据
        if news_df.empty:
            merged = turnover_df.copy()
            merged['news_count'] = 0
        else:
            merged = pd.merge(
                turnover_df,
                news_df,
                on=['trade_date', 'ts_code'],
                how='left'
            )
            merged['news_count'] = merged['news_count'].fillna(0)
        
        # 4. 计算合成热度指标
        merged['norm_turnover'] = np.minimum(
            merged['turnover_rate'] / self.config.turnover_cap, 1.0
        ) * 100
        
        merged['norm_news'] = np.minimum(
            merged['news_count'] / self.config.news_cap, 1.0
        ) * 100
        
        merged['search_index_proxy'] = (
            self.config.turnover_weight * merged['norm_turnover'] +
            self.config.news_weight * merged['norm_news']
        ).round(2)
        
        # 5. 计算排名（按日期分组）
        merged['rank'] = merged.groupby('trade_date')['search_index_proxy'].rank(
            ascending=False, method='min'
        ).astype(int)
        
        merged['source'] = HotListSource.SYNTHETIC.value
        
        return self._ensure_fields(merged)
    
    def _get_turnover_data(
        self,
        start_date: str,
        end_date: str,
        ts_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取换手率数据
        
        优先使用 Tushare daily_basic，备选 AkShare
        """
        # 转换日期格式
        start_date_ts = start_date.replace('-', '')
        end_date_ts = end_date.replace('-', '')
        
        # 尝试 Tushare
        if self.ts:
            try:
                all_data = []
                
                if ts_codes:
                    # 按股票采集
                    for ts_code in ts_codes:
                        df = self.ts.daily_basic(
                            ts_code=ts_code,
                            start_date=start_date_ts,
                            end_date=end_date_ts,
                            fields='ts_code,trade_date,turnover_rate,volume_ratio'
                        )
                        if df is not None and not df.empty:
                            all_data.append(df)
                else:
                    # 按日期采集全市场
                    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
                    for date in date_range:
                        date_str = date.strftime('%Y%m%d')
                        try:
                            df = self.ts.daily_basic(
                                trade_date=date_str,
                                fields='ts_code,trade_date,turnover_rate,volume_ratio'
                            )
                            if df is not None and not df.empty:
                                all_data.append(df)
                        except Exception:
                            continue
                
                if all_data:
                    result = pd.concat(all_data, ignore_index=True)
                    result['trade_date'] = result['trade_date'].apply(self._standardize_date)
                    
                    # 获取股票名称
                    result = self._add_stock_names(result)
                    
                    return result
                    
            except Exception as e:
                self.logger.warning(f"Tushare 换手率采集失败: {e}")
        
        # 备选：AkShare
        try:
            return self._get_turnover_from_akshare(start_date, end_date, ts_codes)
        except Exception as e:
            self.logger.error(f"AkShare 换手率采集失败: {e}")
            return pd.DataFrame()
    
    def _get_turnover_from_akshare(
        self,
        start_date: str,
        end_date: str,
        ts_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """从 AkShare 获取换手率数据"""
        if not ts_codes:
            self.logger.warning("AkShare 换手率采集需要指定股票代码")
            return pd.DataFrame()
        
        all_data = []
        
        for ts_code in ts_codes:
            try:
                # 转换为 AkShare 格式
                symbol = ts_code.split('.')[0]
                
                df = self.ak.stock_zh_a_hist(
                    symbol=symbol,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''),
                    adjust=""
                )
                
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        '日期': 'trade_date',
                        '换手率': 'turnover_rate',
                    })
                    df['ts_code'] = ts_code
                    all_data.append(df[['ts_code', 'trade_date', 'turnover_rate']])
                    
            except Exception as e:
                self.logger.debug(f"AkShare 采集 {ts_code} 换手率失败: {e}")
                continue
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result['trade_date'] = pd.to_datetime(result['trade_date']).dt.strftime('%Y-%m-%d')
            return result
        
        return pd.DataFrame()
    
    def _get_news_count(
        self,
        start_date: str,
        end_date: str,
        ts_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取新闻数量数据
        
        使用 Tushare major_news 接口统计
        """
        if not self.ts:
            self.logger.warning("Tushare 未配置，无法获取新闻数量")
            return pd.DataFrame()
        
        try:
            all_data = []
            
            # 转换日期格式
            start_date_ts = start_date.replace('-', '')
            end_date_ts = end_date.replace('-', '')
            
            if ts_codes:
                # 按股票采集
                for ts_code in ts_codes:
                    try:
                        # 尝试获取个股新闻
                        df = self.ts.news(
                            src='sina',
                            start_date=start_date_ts,
                            end_date=end_date_ts
                        )
                        
                        if df is not None and not df.empty:
                            # 筛选包含该股票代码或简称的新闻
                            code_part = ts_code.split('.')[0]
                            # 这里简化处理：统计所有新闻
                            df['trade_date'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
                            counts = df.groupby('trade_date').size().reset_index(name='news_count')
                            counts['ts_code'] = ts_code
                            all_data.append(counts)
                            
                    except Exception as e:
                        self.logger.debug(f"获取 {ts_code} 新闻数量失败: {e}")
                        continue
            else:
                # 全市场新闻统计
                try:
                    df = self.ts.news(
                        src='sina',
                        start_date=start_date_ts,
                        end_date=end_date_ts
                    )
                    
                    if df is not None and not df.empty:
                        df['trade_date'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
                        # 返回每日新闻总数（无股票代码维度）
                        counts = df.groupby('trade_date').size().reset_index(name='news_count')
                        all_data.append(counts)
                        
                except Exception as e:
                    self.logger.warning(f"获取全市场新闻失败: {e}")
            
            if all_data:
                return pd.concat(all_data, ignore_index=True)
                
        except Exception as e:
            self.logger.warning(f"获取新闻数量失败: {e}")
        
        return pd.DataFrame()
    
    # ==================== 工具方法 ====================
    
    def _to_ts_code(self, code: str) -> str:
        """转换为 Tushare 股票代码格式"""
        code = str(code).strip()
        if '.' in code:
            return code
        
        # 根据代码前缀判断交易所
        if code.startswith(('6', '5', '9')):
            return f"{code}.SH"
        elif code.startswith(('0', '3', '1', '2')):
            return f"{code}.SZ"
        elif code.startswith(('8', '4')):
            return f"{code}.BJ"
        else:
            return f"{code}.SZ"
    
    def _add_stock_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加股票名称"""
        if 'name' in df.columns and df['name'].notna().all():
            return df
        
        if self.ts:
            try:
                stock_basic = self.ts.stock_basic(
                    fields='ts_code,name'
                )
                if stock_basic is not None and not stock_basic.empty:
                    stock_names = dict(zip(stock_basic['ts_code'], stock_basic['name']))
                    df['name'] = df['ts_code'].map(stock_names)
            except Exception:
                pass
        
        if 'name' not in df.columns:
            df['name'] = None
            
        return df
    
    def _ensure_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保输出包含所有标准字段"""
        for field in self.STANDARD_FIELDS:
            if field not in df.columns:
                df[field] = None
        
        return df[self.STANDARD_FIELDS]


# ==================== 便捷函数 ====================

def get_market_heat(
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    ts_codes: Optional[List[str]] = None,
    source: str = 'synthetic'
) -> pd.DataFrame:
    """
    获取市场热度数据（便捷函数）
    
    Args:
        trade_date: 交易日期（获取单日数据）
        start_date: 开始日期
        end_date: 结束日期
        ts_codes: 股票代码列表
        source: 数据来源 ('eastmoney', 'xueqiu', 'synthetic')
    
    Returns:
        热度数据 DataFrame
    """
    collector = MarketHeatCollector()
    
    if trade_date:
        start_date = trade_date
        end_date = trade_date
    elif not start_date:
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = start_date
    elif not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    source_enum = HotListSource(source) if isinstance(source, str) else source
    
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        ts_codes=ts_codes,
        source=source_enum
    )


def get_realtime_hotlist(source: str = 'eastmoney') -> pd.DataFrame:
    """
    获取实时热榜（便捷函数）
    
    Args:
        source: 'eastmoney' 或 'xueqiu'
    
    Returns:
        热榜 DataFrame
    """
    collector = MarketHeatCollector()
    
    if source == 'eastmoney':
        return collector._collect_eastmoney_hotlist()
    elif source == 'xueqiu':
        return collector._collect_xueqiu_hotlist()
    else:
        raise ValueError(f"不支持的热榜来源: {source}")


def get_historical_heat_proxy(
    ts_codes: List[str],
    start_date: str,
    end_date: str,
    config: Optional[HeatConfig] = None
) -> pd.DataFrame:
    """
    计算历史热度代理指标（便捷函数）
    
    Args:
        ts_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        config: 热度计算配置
    
    Returns:
        历史热度 DataFrame
    """
    collector = MarketHeatCollector(config=config)
    
    return collector._collect_synthetic_heat(
        start_date=start_date,
        end_date=end_date,
        ts_codes=ts_codes
    )
