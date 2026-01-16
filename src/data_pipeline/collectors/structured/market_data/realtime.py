"""
实时与准实时行情（Realtime Market）采集模块

数据类型包括：
- 实时行情报价
- 涨跌幅排名/龙虎榜
"""

import logging
from typing import Optional, List, Literal
from datetime import datetime

import pandas as pd

from ..base import (
    BaseCollector,
    DataSource,
    DataSourceManager,
    retry_on_failure,
    StandardFields,
    CollectorRegistry
)

logger = logging.getLogger(__name__)


@CollectorRegistry.register("realtime_quote")
class RealtimeQuoteCollector(BaseCollector):
    """
    实时行情采集器
    
    采集实时行情数据（盘中使用）
    主数据源：AkShare (stock_zh_a_spot_em)
    备用数据源：Tushare需要更高权限
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'name',             # 证券名称
        'open',             # 开盘价
        'high',             # 最高价
        'low',              # 最低价
        'close',            # 最新价
        'pre_close',        # 昨收价
        'change',           # 涨跌额
        'pct_chg',          # 涨跌幅（%）
        'vol',              # 成交量（手）
        'amount',           # 成交额（元）
        'turnover_rate',    # 换手率（%）
        'pe_ratio',         # 市盈率
        'pb_ratio',         # 市净率
        'total_mv',         # 总市值
        'circ_mv',          # 流通市值
        'update_time',      # 更新时间
    ]
    
    def collect(
        self,
        ts_codes: Optional[List[str]] = None,
        market: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集实时行情数据
        
        Args:
            ts_codes: 证券代码列表（可选，不传则获取全市场）
            market: 市场类型（可选，如 主板/创业板/科创板）
        
        Returns:
            DataFrame: 标准化的实时行情数据
        """
        # 优先使用AkShare（免费）
        try:
            df = self._collect_from_akshare(ts_codes)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条实时行情数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取实时行情失败: {e}")
        
        logger.error("无法获取实时行情数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(
        self,
        ts_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """从AkShare获取实时行情"""
        import akshare as ak
        
        try:
            df = ak.stock_zh_a_spot_em()
        except Exception as e:
            logger.warning(f"AkShare获取实时行情失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        # 直接重命名列
        rename_map = {
            '代码': 'symbol',
            '名称': 'name',
            '最新价': 'close',
            '涨跌幅': 'pct_chg',
            '涨跌额': 'change',
            '成交量': 'vol',
            '成交额': 'amount',
            '今开': 'open',
            '最高': 'high',
            '最低': 'low',
            '昨收': 'pre_close',
            '换手率': 'turnover_rate',
            '市盈率-动态': 'pe_ratio',
            '市净率': 'pb_ratio',
            '总市值': 'total_mv',
            '流通市值': 'circ_mv',
        }
        df = df.rename(columns=rename_map)
        
        # 生成ts_code
        if 'symbol' in df.columns:
            df['ts_code'] = df['symbol'].apply(self._symbol_to_tscode)
        
        # 添加更新时间
        df['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 筛选指定代码
        if ts_codes:
            df = df[df['ts_code'].isin(ts_codes)]
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
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


@CollectorRegistry.register("top_list")
class TopListCollector(BaseCollector):
    """
    龙虎榜数据采集器
    
    采集每日龙虎榜数据
    主数据源：Tushare (top_list)
    备用数据源：AkShare (stock_lhb_detail_em)
    """
    
    OUTPUT_FIELDS = [
        'trade_date',       # 交易日期
        'ts_code',          # 证券代码
        'name',             # 证券名称
        'close',            # 收盘价
        'turnover_rate',    # 换手率
        'amount',           # 成交额
        'l_sell',           # 龙虎榜卖出额
        'l_buy',            # 龙虎榜买入额
        'l_amount',         # 龙虎榜成交额
        'net_amount',       # 龙虎榜净买额
        'net_rate',         # 龙虎榜净买占比
        'reason',           # 上榜原因
    ]
    
    def collect(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集龙虎榜数据
        
        Args:
            trade_date: 交易日期（YYYYMMDD格式）
            ts_code: 证券代码
        
        Returns:
            DataFrame: 标准化的龙虎榜数据
        """
        if not trade_date:
            trade_date = datetime.now().strftime('%Y%m%d')
        
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(trade_date, ts_code)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条龙虎榜数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取龙虎榜失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(trade_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条龙虎榜数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取龙虎榜失败: {e}")
        
        logger.error("所有数据源均无法获取龙虎榜数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        trade_date: str,
        ts_code: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取龙虎榜数据"""
        pro = self.tushare_api
        
        params = {'trade_date': trade_date}
        if ts_code:
            params['ts_code'] = ts_code
        
        df = pro.top_list(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, trade_date: str) -> pd.DataFrame:
        """从AkShare获取龙虎榜数据"""
        import akshare as ak
        
        # 转换日期格式
        date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
        
        try:
            df = ak.stock_lhb_detail_em(start_date=date_str, end_date=date_str)
        except Exception as e:
            logger.warning(f"AkShare获取龙虎榜失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 直接重命名列
        rename_map = {
            '代码': 'symbol',
            '名称': 'name',
            '收盘价': 'close',
            '涨跌幅': 'pct_chg',
            '换手率': 'turnover_rate',
            '成交额': 'amount',
            '龙虎榜卖出额': 'l_sell',
            '龙虎榜买入额': 'l_buy',
            '龙虎榜成交额': 'l_amount',
            '龙虎榜净买额': 'net_amount',
            '上榜原因': 'reason',
        }
        df = df.rename(columns=rename_map)
        
        # 数据清洗：处理涨跌幅和换手率
        # AkShare可能返回带%的字符串或浮点数
        for col in ['pct_chg', 'turnover_rate']:
            if col in df.columns:
                # 转为字符串处理
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                # 转为数字，无法转换的变为NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 生成ts_code
        if 'symbol' in df.columns:
            df['ts_code'] = df['symbol'].apply(lambda x: self._symbol_to_tscode(str(x).zfill(6)))
        
        df['trade_date'] = date_str
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
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


# ============= 便捷函数接口 =============

def get_realtime_quote(
    ts_codes: Optional[List[str]] = None,
    market: Optional[str] = None
) -> pd.DataFrame:
    """
    获取实时行情数据
    
    Args:
        ts_codes: 证券代码列表
        market: 市场类型
    
    Returns:
        DataFrame: 实时行情数据
    
    Example:
        >>> df = get_realtime_quote()  # 全市场
        >>> df = get_realtime_quote(ts_codes=['000001.SZ', '600000.SH'])
    """
    collector = RealtimeQuoteCollector()
    return collector.collect(ts_codes=ts_codes, market=market)


def get_top_list(
    trade_date: Optional[str] = None,
    ts_code: Optional[str] = None
) -> pd.DataFrame:
    """
    获取龙虎榜数据
    
    Args:
        trade_date: 交易日期
        ts_code: 证券代码
    
    Returns:
        DataFrame: 龙虎榜数据
    
    Example:
        >>> df = get_top_list(trade_date='20240115')
    """
    collector = TopListCollector()
    return collector.collect(trade_date=trade_date, ts_code=ts_code)
