"""
技术指标与衍生行情特征采集模块

数据类型包括：
- 每日基本指标（市盈率/市净率/换手率等）
- 技术指标（MA/RSI/MACD等）
- 官方技术因子（stk_factor）
"""

import logging
from typing import Optional, List, Literal
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from ..base import (
    BaseCollector,
    DataSource,
    DataSourceManager,
    retry_on_failure,
    StandardFields,
    CollectorRegistry
)

logger = logging.getLogger(__name__)


@CollectorRegistry.register("daily_basic")
class DailyBasicCollector(BaseCollector):
    """
    每日基本指标采集器
    
    采集股票每日基本面指标数据
    主数据源：Tushare (daily_basic)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'trade_date',       # 交易日期
        'close',            # 收盘价
        'turnover_rate',    # 换手率（%）
        'turnover_rate_f',  # 换手率（自由流通股）
        'volume_ratio',     # 量比
        'pe',               # 市盈率（总市值/净利润）
        'pe_ttm',           # 市盈率（TTM）
        'pb',               # 市净率（总市值/净资产）
        'ps',               # 市销率
        'ps_ttm',           # 市销率（TTM）
        'dv_ratio',         # 股息率（%）
        'dv_ttm',           # 股息率（TTM）（%）
        'total_share',      # 总股本（万股）
        'float_share',      # 流通股本（万股）
        'free_share',       # 自由流通股本（万股）
        'total_mv',         # 总市值（万元）
        'circ_mv',          # 流通市值（万元）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集每日基本指标数据
        
        Args:
            ts_code: 证券代码
            trade_date: 交易日期（YYYYMMDD格式）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的每日基本指标数据
            
        输出字段:
            - ts_code: 证券代码
            - trade_date: 交易日期
            - close: 收盘价
            - turnover_rate: 换手率
            - pe: 市盈率
            - pe_ttm: 市盈率TTM
            - pb: 市净率
            - total_mv: 总市值
            - circ_mv: 流通市值
            等...
        """
        if not start_date and not trade_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        if not end_date and not trade_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条每日基本指标数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取每日基本指标失败: {e}")
        
        # 降级到AkShare (仅当未指定历史范围采样时，AkShare的spot接口才有效)
        if not start_date and not end_date:
            try:
                df = self._collect_from_akshare(ts_code, trade_date)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条每日基本指标数据")
                    return df
            except Exception as e:
                logger.error(f"AkShare获取每日基本指标失败: {e}")
        else:
            logger.debug("历史数据模式下跳过AkShare实时接口降级")
        
        logger.error("所有数据源均无法获取每日基本指标数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取每日基本指标"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        # 请求所有字段
        fields = ','.join(self.OUTPUT_FIELDS)
        df = pro.daily_basic(**params, fields=fields)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        df = df.sort_values('trade_date', ascending=True)
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str]
    ) -> pd.DataFrame:
        """从AkShare获取每日基本指标"""
        import akshare as ak
        
        try:
            df = ak.stock_zh_a_spot_em()
        except Exception as e:
            logger.warning(f"AkShare获取实时行情失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '代码': 'symbol',
            '最新价': 'close',
            '换手率': 'turnover_rate',
            '量比': 'volume_ratio',
            '市盈率-动态': 'pe',
            '市净率': 'pb',
            '总市值': 'total_mv',
            '流通市值': 'circ_mv',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 生成ts_code
        if 'symbol' in df.columns:
            df['ts_code'] = df['symbol'].apply(self._symbol_to_tscode)
        
        df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # 筛选指定代码
        if ts_code:
            df = df[df['ts_code'] == ts_code]
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        logger.warning("AkShare数据仅包含部分基本指标，建议使用Tushare")
        
        return df[self.OUTPUT_FIELDS]
    
    @staticmethod
    def _symbol_to_tscode(symbol: str) -> str:
        """将纯数字代码转换为带交易所后缀的代码"""
        symbol = str(symbol).zfill(6)
        if symbol.startswith(('0', '2', '3')):
            return f"{symbol}.SZ"
        elif symbol.startswith(('6', '9')):
            return f"{symbol}.SH"
        elif symbol.startswith(('4', '8')):
            return f"{symbol}.BJ"
        return f"{symbol}.SZ"


@CollectorRegistry.register("technical_indicator")
class TechnicalIndicatorCollector(BaseCollector):
    """
    技术指标采集器
    
    计算常用技术指标：MA/RSI/MACD/BOLL等
    数据来源：基于K线数据计算
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'trade_date',       # 交易日期
        'close',            # 收盘价
        'ma5',              # 5日均线
        'ma10',             # 10日均线
        'ma20',             # 20日均线
        'ma60',             # 60日均线
        'ma120',            # 120日均线
        'ma250',            # 250日均线
        'ema12',            # 12日指数均线
        'ema26',            # 26日指数均线
        'macd',             # MACD值（DIF-DEA）
        'macd_dif',         # DIF
        'macd_dea',         # DEA
        'rsi6',             # 6日RSI
        'rsi12',            # 12日RSI
        'rsi24',            # 24日RSI
        'boll_upper',       # 布林带上轨
        'boll_mid',         # 布林带中轨
        'boll_lower',       # 布林带下轨
        'kdj_k',            # KDJ-K值
        'kdj_d',            # KDJ-D值
        'kdj_j',            # KDJ-J值
    ]
    
    def collect(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adj: str = 'qfq',
        **kwargs
    ) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            ts_code: 证券代码（必填）
            start_date: 开始日期
            end_date: 结束日期
            adj: 复权类型
        
        Returns:
            DataFrame: 包含技术指标的数据
            
        输出字段:
            - ts_code, trade_date, close
            - ma5, ma10, ma20, ma60, ma120, ma250（均线）
            - macd, macd_dif, macd_dea（MACD）
            - rsi6, rsi12, rsi24（RSI）
            - boll_upper, boll_mid, boll_lower（布林带）
            - kdj_k, kdj_d, kdj_j（KDJ）
        """
        if not ts_code:
            logger.error("计算技术指标需要指定ts_code")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 需要更多历史数据来计算指标
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y%m%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 首先获取K线数据
        try:
            from .price_kline import get_stock_daily
            kline_df = get_stock_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                adj=adj
            )
            
            if kline_df.empty:
                logger.error(f"无法获取 {ts_code} 的K线数据")
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            # 计算技术指标
            df = self._calculate_indicators(kline_df)
            logger.info(f"成功计算 {len(df)} 条技术指标数据")
            return df
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算各项技术指标"""
        # 确保按日期排序
        df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
        
        close = df['close'].astype(float)
        high = df['high'].astype(float) if 'high' in df.columns else close
        low = df['low'].astype(float) if 'low' in df.columns else close
        
        # 1. 均线（MA）
        for period in [5, 10, 20, 60, 120, 250]:
            df[f'ma{period}'] = close.rolling(window=period, min_periods=1).mean()
        
        # 2. 指数移动平均（EMA）
        df['ema12'] = close.ewm(span=12, adjust=False).mean()
        df['ema26'] = close.ewm(span=26, adjust=False).mean()
        
        # 3. MACD
        df['macd_dif'] = df['ema12'] - df['ema26']
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = 2 * (df['macd_dif'] - df['macd_dea'])
        
        # 4. RSI
        for period in [6, 12, 24]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df[f'rsi{period}'] = 100 - (100 / (1 + rs))
        
        # 5. 布林带（BOLL）
        boll_period = 20
        df['boll_mid'] = close.rolling(window=boll_period, min_periods=1).mean()
        std = close.rolling(window=boll_period, min_periods=1).std()
        df['boll_upper'] = df['boll_mid'] + 2 * std
        df['boll_lower'] = df['boll_mid'] - 2 * std
        
        # 6. KDJ
        kdj_period = 9
        low_min = low.rolling(window=kdj_period, min_periods=1).min()
        high_max = high.rolling(window=kdj_period, min_periods=1).max()
        rsv = (close - low_min) / (high_max - low_min).replace(0, np.nan) * 100
        
        df['kdj_k'] = rsv.ewm(alpha=1/3, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(alpha=1/3, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("stk_factor")
class StkFactorCollector(BaseCollector):
    """
    股票技术因子采集器
    
    采集Tushare官方计算的技术因子
    主数据源：Tushare (stk_factor)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'trade_date',       # 交易日期
        'close',            # 收盘价
        'open',             # 开盘价
        'high',             # 最高价
        'low',              # 最低价
        'pre_close',        # 昨收价
        'change',           # 涨跌额
        'vol',              # 成交量
        'amount',           # 成交额
        'adj_factor',       # 复权因子
        'open_hfq',         # 开盘价后复权
        'open_qfq',         # 开盘价前复权
        'close_hfq',        # 收盘价后复权
        'close_qfq',        # 收盘价前复权
        'high_hfq',         # 最高价后复权
        'high_qfq',         # 最高价前复权
        'low_hfq',          # 最低价后复权
        'low_qfq',          # 最低价前复权
        'pre_close_hfq',    # 昨收价后复权
        'pre_close_qfq',    # 昨收价前复权
        'macd_dif',         # MACD DIF
        'macd_dea',         # MACD DEA
        'macd',             # MACD
        'kdj_k',            # KDJ K值
        'kdj_d',            # KDJ D值
        'kdj_j',            # KDJ J值
        'rsi_6',            # RSI 6日
        'rsi_12',           # RSI 12日
        'rsi_24',           # RSI 24日
        'boll_upper',       # 布林带上轨
        'boll_mid',         # 布林带中轨
        'boll_lower',       # 布林带下轨
        'cci',              # CCI指标
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股票技术因子数据
        
        Args:
            ts_code: 证券代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 股票技术因子数据
            
        注意：
            该接口需要5000积分以上，当前账户积分不足可能无法调用
        """
        if not start_date and not trade_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        if not end_date and not trade_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条技术因子数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取技术因子失败（可能积分不足）: {e}")
        
        # 降级：使用自计算的技术指标
        if ts_code:
            try:
                logger.info("使用自计算技术指标替代官方因子")
                indicator_collector = TechnicalIndicatorCollector()
                return indicator_collector.collect(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
            except Exception as e:
                logger.error(f"自计算技术指标失败: {e}")
        
        logger.error("无法获取技术因子数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取技术因子"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = pro.stk_factor(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        df = df.sort_values('trade_date', ascending=True)
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_daily_basic(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取每日基本指标数据
    
    Args:
        ts_code: 证券代码
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 每日基本指标数据
    
    Example:
        >>> df = get_daily_basic(ts_code='000001.SZ', trade_date='20240115')
        >>> df = get_daily_basic(trade_date='20240115')  # 全市场
    """
    collector = DailyBasicCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date)


def get_technical_indicator(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adj: str = 'qfq'
) -> pd.DataFrame:
    """
    计算技术指标（MA/RSI/MACD/BOLL/KDJ）
    
    Args:
        ts_code: 证券代码（必填）
        start_date: 开始日期
        end_date: 结束日期
        adj: 复权类型
    
    Returns:
        DataFrame: 技术指标数据
    
    Example:
        >>> df = get_technical_indicator(ts_code='000001.SZ', start_date='20240101')
    """
    collector = TechnicalIndicatorCollector()
    return collector.collect(ts_code=ts_code, start_date=start_date,
                            end_date=end_date, adj=adj)


def get_stk_factor(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取股票技术因子（Tushare官方）
    
    Args:
        ts_code: 证券代码
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 技术因子数据
    
    Note:
        该接口需要5000积分以上，积分不足时会降级为自计算指标
    
    Example:
        >>> df = get_stk_factor(ts_code='000001.SZ', start_date='20240101')
    """
    collector = StkFactorCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date)
