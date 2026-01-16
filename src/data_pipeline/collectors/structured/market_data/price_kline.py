"""
K线与价格序列（Price & OHLCV）采集模块

数据类型包括：
- 股票日/周/月K线（支持前复权/后复权/不复权）
- 指数日/周/月K线
- ETF日/周/月K线
"""

import logging
from typing import Optional, List, Literal
from datetime import datetime, timedelta

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


# 复权类型
AdjustType = Literal['qfq', 'hfq', None]  # 前复权/后复权/不复权


@CollectorRegistry.register("stock_daily")
class StockDailyCollector(BaseCollector):
    """
    股票日K线采集器
    
    采集股票日线行情数据（OHLCV）
    主数据源：Tushare (daily / pro_bar)
    备用数据源：AkShare (stock_zh_a_hist), BaoStock (query_history_k_data_plus)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'trade_date',       # 交易日期
        'open',             # 开盘价
        'high',             # 最高价
        'low',              # 最低价
        'close',            # 收盘价
        'pre_close',        # 昨收价
        'change',           # 涨跌额
        'pct_chg',          # 涨跌幅（%）
        'vol',              # 成交量（手）
        'amount',           # 成交额（千元）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adj: AdjustType = 'qfq',
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股票日K线数据
        
        Args:
            ts_code: 证券代码（如 000001.SZ）
            trade_date: 交易日期（获取某日全市场数据）
            start_date: 开始日期（YYYYMMDD格式）
            end_date: 结束日期（YYYYMMDD格式）
            adj: 复权类型，qfq=前复权，hfq=后复权，None=不复权
        
        Returns:
            DataFrame: 标准化的日K线数据
            
        输出字段:
            - ts_code: 证券代码
            - trade_date: 交易日期
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - pre_close: 昨收价
            - change: 涨跌额
            - pct_chg: 涨跌幅
            - vol: 成交量
            - amount: 成交额
        """
        # 设置默认日期范围（最近一年）
        if not start_date and not trade_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if not end_date and not trade_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date, adj)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条股票日K线数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取股票日K线失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code, start_date, end_date, adj)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条股票日K线数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取股票日K线失败: {e}")
        
        # 降级到BaoStock
        try:
            df = self._collect_from_baostock(ts_code, start_date, end_date, adj)
            if not df.empty:
                logger.info(f"从BaoStock成功获取 {len(df)} 条股票日K线数据")
                return df
        except Exception as e:
            logger.error(f"BaoStock获取股票日K线失败: {e}")
        
        logger.error("所有数据源均无法获取股票日K线数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        adj: AdjustType
    ) -> pd.DataFrame:
        """从Tushare获取股票日K线"""
        import tushare as ts
        
        pro = self.tushare_api
        
        if adj:
            # 使用pro_bar获取复权数据
            df = ts.pro_bar(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                adj=adj,
                asset='E',  # 股票
                freq='D'    # 日线
            )
        else:
            # 使用daily获取未复权数据
            params = {}
            if ts_code:
                params['ts_code'] = ts_code
            if trade_date:
                params['trade_date'] = trade_date
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            df = pro.daily(**params)
        
        if df is None or df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化日期格式
        df = self._convert_date_format(df, ['trade_date'])
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        # 按日期排序
        df = df.sort_values('trade_date', ascending=True)
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(
        self,
        ts_code: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        adj: AdjustType
    ) -> pd.DataFrame:
        """从AkShare获取股票日K线"""
        import akshare as ak
        
        if not ts_code:
            logger.warning("AkShare需要指定ts_code")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 转换代码格式：000001.SZ -> 000001
        symbol = ts_code.split('.')[0]
        
        # 复权类型映射
        adjust_map = {
            'qfq': 'qfq',
            'hfq': 'hfq',
            None: ''
        }
        adjust = adjust_map.get(adj, '')
        
        # 转换日期格式
        start_dt = datetime.strptime(start_date, '%Y%m%d').strftime('%Y%m%d') if start_date else '19900101'
        end_dt = datetime.strptime(end_date, '%Y%m%d').strftime('%Y%m%d') if end_date else datetime.now().strftime('%Y%m%d')
        
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period='daily',
                start_date=start_dt,
                end_date=end_dt,
                adjust=adjust
            )
        except Exception as e:
            logger.warning(f"AkShare获取日K线失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '日期': 'trade_date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'vol',
            '成交额': 'amount',
            '涨跌幅': 'pct_chg',
            '涨跌额': 'change',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置ts_code
        df['ts_code'] = ts_code
        
        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        
        # 成交量单位转换（AkShare返回的是股，转换为手）
        if 'vol' in df.columns:
            df['vol'] = df['vol'] / 100
        
        # 成交额单位转换（AkShare返回的是元，转换为千元）
        if 'amount' in df.columns:
            df['amount'] = df['amount'] / 1000
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_baostock(
        self,
        ts_code: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        adj: AdjustType
    ) -> pd.DataFrame:
        """从BaoStock获取股票日K线"""
        import baostock as bs
        
        if not ts_code:
            logger.warning("BaoStock需要指定ts_code")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保登录
        if not self.source_manager.ensure_baostock_login():
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 转换代码格式：000001.SZ -> sz.000001
        symbol = ts_code.split('.')[0]
        exchange = ts_code.split('.')[1].lower()
        bs_code = f"{exchange}.{symbol}"
        
        # 复权类型映射
        adj_map = {
            'qfq': '2',
            'hfq': '1',
            None: '3'
        }
        adjustflag = adj_map.get(adj, '3')
        
        # 转换日期格式
        start_dt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}" if start_date else '1990-01-01'
        end_dt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}" if end_date else datetime.now().strftime('%Y-%m-%d')
        
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,preclose,volume,amount,pctChg",
            start_date=start_dt,
            end_date=end_dt,
            frequency="d",
            adjustflag=adjustflag
        )
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 标准化字段
        column_mapping = {
            'date': 'trade_date',
            'code': 'bs_code',
            'preclose': 'pre_close',
            'volume': 'vol',
            'pctChg': 'pct_chg',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置ts_code
        df['ts_code'] = ts_code
        
        # 计算涨跌额
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['pre_close'] = pd.to_numeric(df['pre_close'], errors='coerce')
        df['change'] = df['close'] - df['pre_close']
        
        # 成交量单位转换（BaoStock返回的是股，转换为手）
        df['vol'] = pd.to_numeric(df['vol'], errors='coerce') / 100
        
        # 成交额单位转换（BaoStock返回的是元，转换为千元）
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce') / 1000
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("stock_weekly")
class StockWeeklyCollector(BaseCollector):
    """
    股票周K线采集器
    
    主数据源：Tushare (weekly)
    备用数据源：AkShare, BaoStock
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'trade_date',       # 交易日期（周五）
        'open',             # 开盘价
        'high',             # 最高价
        'low',              # 最低价
        'close',            # 收盘价
        'pre_close',        # 上周收盘价
        'change',           # 涨跌额
        'pct_chg',          # 涨跌幅（%）
        'vol',              # 成交量（手）
        'amount',           # 成交额（千元）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adj: AdjustType = 'qfq',
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股票周K线数据
        
        Args:
            ts_code: 证券代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            adj: 复权类型
        
        Returns:
            DataFrame: 标准化的周K线数据
        """
        if not start_date and not trade_date:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')
        if not end_date and not trade_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date, adj)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条股票周K线数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取股票周K线失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code, start_date, end_date, adj)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条股票周K线数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取股票周K线失败: {e}")
        
        # 降级到BaoStock
        try:
            df = self._collect_from_baostock(ts_code, start_date, end_date, adj)
            if not df.empty:
                logger.info(f"从BaoStock成功获取 {len(df)} 条股票周K线数据")
                return df
        except Exception as e:
            logger.error(f"BaoStock获取股票周K线失败: {e}")
        
        logger.error("所有数据源均无法获取股票周K线数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        adj: AdjustType
    ) -> pd.DataFrame:
        """从Tushare获取股票周K线"""
        import tushare as ts
        
        pro = self.tushare_api
        
        if adj:
            # 使用pro_bar获取复权数据
            df = ts.pro_bar(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                adj=adj,
                asset='E',
                freq='W'  # 周线
            )
        else:
            # 使用weekly获取未复权数据
            params = {}
            if ts_code:
                params['ts_code'] = ts_code
            if trade_date:
                params['trade_date'] = trade_date
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            df = pro.weekly(**params)
        
        if df is None or df.empty:
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
        start_date: Optional[str],
        end_date: Optional[str],
        adj: AdjustType
    ) -> pd.DataFrame:
        """从AkShare获取股票周K线"""
        import akshare as ak
        
        if not ts_code:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        symbol = ts_code.split('.')[0]
        adjust_map = {'qfq': 'qfq', 'hfq': 'hfq', None: ''}
        adjust = adjust_map.get(adj, '')
        
        start_dt = start_date if start_date else '19900101'
        end_dt = end_date if end_date else datetime.now().strftime('%Y%m%d')
        
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period='weekly',
                start_date=start_dt,
                end_date=end_dt,
                adjust=adjust
            )
        except Exception as e:
            logger.warning(f"AkShare获取周K线失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '日期': 'trade_date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'vol',
            '成交额': 'amount',
            '涨跌幅': 'pct_chg',
            '涨跌额': 'change',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        
        if 'vol' in df.columns:
            df['vol'] = df['vol'] / 100
        if 'amount' in df.columns:
            df['amount'] = df['amount'] / 1000
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_baostock(
        self,
        ts_code: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        adj: AdjustType
    ) -> pd.DataFrame:
        """从BaoStock获取股票周K线"""
        import baostock as bs
        
        if not ts_code:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if not self.source_manager.ensure_baostock_login():
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        symbol = ts_code.split('.')[0]
        exchange = ts_code.split('.')[1].lower()
        bs_code = f"{exchange}.{symbol}"
        
        adj_map = {'qfq': '2', 'hfq': '1', None: '3'}
        adjustflag = adj_map.get(adj, '3')
        
        start_dt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}" if start_date else '1990-01-01'
        end_dt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}" if end_date else datetime.now().strftime('%Y-%m-%d')
        
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,preclose,volume,amount,pctChg",
            start_date=start_dt,
            end_date=end_dt,
            frequency="w",  # 周线
            adjustflag=adjustflag
        )
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        column_mapping = {
            'date': 'trade_date',
            'preclose': 'pre_close',
            'volume': 'vol',
            'pctChg': 'pct_chg',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['pre_close'] = pd.to_numeric(df['pre_close'], errors='coerce')
        df['change'] = df['close'] - df['pre_close']
        df['vol'] = pd.to_numeric(df['vol'], errors='coerce') / 100
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce') / 1000
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("stock_monthly")
class StockMonthlyCollector(BaseCollector):
    """
    股票月K线采集器
    
    主数据源：Tushare (monthly)
    备用数据源：AkShare, BaoStock
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'trade_date',       # 交易日期（月末）
        'open',             # 开盘价
        'high',             # 最高价
        'low',              # 最低价
        'close',            # 收盘价
        'pre_close',        # 上月收盘价
        'change',           # 涨跌额
        'pct_chg',          # 涨跌幅（%）
        'vol',              # 成交量（手）
        'amount',           # 成交额（千元）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adj: AdjustType = 'qfq',
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股票月K线数据
        
        Args:
            ts_code: 证券代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            adj: 复权类型
        
        Returns:
            DataFrame: 标准化的月K线数据
        """
        if not start_date and not trade_date:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y%m%d')
        if not end_date and not trade_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date, adj)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条股票月K线数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取股票月K线失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code, start_date, end_date, adj)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条股票月K线数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取股票月K线失败: {e}")
        
        # 降级到BaoStock
        try:
            df = self._collect_from_baostock(ts_code, start_date, end_date, adj)
            if not df.empty:
                logger.info(f"从BaoStock成功获取 {len(df)} 条股票月K线数据")
                return df
        except Exception as e:
            logger.error(f"BaoStock获取股票月K线失败: {e}")
        
        logger.error("所有数据源均无法获取股票月K线数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        adj: AdjustType
    ) -> pd.DataFrame:
        """从Tushare获取股票月K线"""
        import tushare as ts
        
        pro = self.tushare_api
        
        if adj:
            df = ts.pro_bar(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                adj=adj,
                asset='E',
                freq='M'  # 月线
            )
        else:
            params = {}
            if ts_code:
                params['ts_code'] = ts_code
            if trade_date:
                params['trade_date'] = trade_date
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            df = pro.monthly(**params)
        
        if df is None or df.empty:
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
        start_date: Optional[str],
        end_date: Optional[str],
        adj: AdjustType
    ) -> pd.DataFrame:
        """从AkShare获取股票月K线"""
        import akshare as ak
        
        if not ts_code:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        symbol = ts_code.split('.')[0]
        adjust_map = {'qfq': 'qfq', 'hfq': 'hfq', None: ''}
        adjust = adjust_map.get(adj, '')
        
        start_dt = start_date if start_date else '19900101'
        end_dt = end_date if end_date else datetime.now().strftime('%Y%m%d')
        
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period='monthly',
                start_date=start_dt,
                end_date=end_dt,
                adjust=adjust
            )
        except Exception as e:
            logger.warning(f"AkShare获取月K线失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '日期': 'trade_date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'vol',
            '成交额': 'amount',
            '涨跌幅': 'pct_chg',
            '涨跌额': 'change',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        
        if 'vol' in df.columns:
            df['vol'] = df['vol'] / 100
        if 'amount' in df.columns:
            df['amount'] = df['amount'] / 1000
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_baostock(
        self,
        ts_code: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        adj: AdjustType
    ) -> pd.DataFrame:
        """从BaoStock获取股票月K线"""
        import baostock as bs
        
        if not ts_code:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if not self.source_manager.ensure_baostock_login():
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        symbol = ts_code.split('.')[0]
        exchange = ts_code.split('.')[1].lower()
        bs_code = f"{exchange}.{symbol}"
        
        adj_map = {'qfq': '2', 'hfq': '1', None: '3'}
        adjustflag = adj_map.get(adj, '3')
        
        start_dt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}" if start_date else '1990-01-01'
        end_dt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}" if end_date else datetime.now().strftime('%Y-%m-%d')
        
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,preclose,volume,amount,pctChg",
            start_date=start_dt,
            end_date=end_dt,
            frequency="m",  # 月线
            adjustflag=adjustflag
        )
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        column_mapping = {
            'date': 'trade_date',
            'preclose': 'pre_close',
            'volume': 'vol',
            'pctChg': 'pct_chg',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['pre_close'] = pd.to_numeric(df['pre_close'], errors='coerce')
        df['change'] = df['close'] - df['pre_close']
        df['vol'] = pd.to_numeric(df['vol'], errors='coerce') / 100
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce') / 1000
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("index_daily")
class IndexDailyCollector(BaseCollector):
    """
    指数日K线采集器
    
    主数据源：Tushare (index_daily)
    备用数据源：AkShare (stock_zh_index_daily_em)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 指数代码
        'trade_date',       # 交易日期
        'open',             # 开盘价
        'high',             # 最高价
        'low',              # 最低价
        'close',            # 收盘价
        'pre_close',        # 昨收价
        'change',           # 涨跌额
        'pct_chg',          # 涨跌幅（%）
        'vol',              # 成交量（手）
        'amount',           # 成交额（千元）
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
        采集指数日K线数据
        
        Args:
            ts_code: 指数代码（如 000001.SH 上证指数）
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的指数日K线数据
        """
        if not start_date and not trade_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if not end_date and not trade_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条指数日K线数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取指数日K线失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code, start_date, end_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条指数日K线数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取指数日K线失败: {e}")
        
        logger.error("所有数据源均无法获取指数日K线数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取指数日K线"""
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
        
        df = pro.index_daily(**params)
        
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
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从AkShare获取指数日K线"""
        import akshare as ak
        
        if not ts_code:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 指数代码映射
        index_map = {
            '000001.SH': 'sh000001',
            '399001.SZ': 'sz399001',
            '399006.SZ': 'sz399006',
            '000300.SH': 'sh000300',
            '000016.SH': 'sh000016',
            '000905.SH': 'sh000905',
        }
        
        ak_code = index_map.get(ts_code, ts_code)
        
        try:
            df = ak.stock_zh_index_daily_em(symbol=ak_code)
        except Exception as e:
            logger.warning(f"AkShare获取指数日K线失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            'date': 'trade_date',
            'open': 'open',
            'close': 'close',
            'high': 'high',
            'low': 'low',
            'volume': 'vol',
            'amount': 'amount',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        
        # 筛选日期范围
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[pd.to_datetime(df['trade_date']) >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[pd.to_datetime(df['trade_date']) <= end_dt]
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("etf_daily")
class ETFDailyCollector(BaseCollector):
    """
    ETF日K线采集器
    
    主数据源：Tushare (fund_daily)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # ETF代码
        'trade_date',       # 交易日期
        'open',             # 开盘价
        'high',             # 最高价
        'low',              # 最低价
        'close',            # 收盘价
        'pre_close',        # 昨收价
        'change',           # 涨跌额
        'pct_chg',          # 涨跌幅（%）
        'vol',              # 成交量（手）
        'amount',           # 成交额（千元）
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
        采集ETF日K线数据
        
        Args:
            ts_code: ETF代码（如 510050.SH）
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的ETF日K线数据
        """
        if not start_date and not trade_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if not end_date and not trade_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条ETF日K线数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取ETF日K线失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code, start_date, end_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条ETF日K线数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取ETF日K线失败: {e}")
        
        logger.error("所有数据源均无法获取ETF日K线数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取ETF日K线"""
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
        
        df = pro.fund_daily(**params)
        
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
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从AkShare获取ETF日K线"""
        import akshare as ak
        
        if not ts_code:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        symbol = ts_code.split('.')[0]
        
        try:
            df = ak.fund_etf_hist_em(
                symbol=symbol,
                period='daily',
                start_date=start_date if start_date else '19900101',
                end_date=end_date if end_date else datetime.now().strftime('%Y%m%d'),
                adjust='qfq'
            )
        except Exception as e:
            logger.warning(f"AkShare获取ETF日K线失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '日期': 'trade_date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'vol',
            '成交额': 'amount',
            '涨跌幅': 'pct_chg',
            '涨跌额': 'change',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_stock_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adj: AdjustType = 'qfq'
) -> pd.DataFrame:
    """
    获取股票日K线数据
    
    Args:
        ts_code: 证券代码（如 000001.SZ）
        trade_date: 交易日期（获取某日全市场数据）
        start_date: 开始日期（YYYYMMDD格式）
        end_date: 结束日期（YYYYMMDD格式）
        adj: 复权类型，qfq=前复权，hfq=后复权，None=不复权
    
    Returns:
        DataFrame: 股票日K线数据
    
    Example:
        >>> df = get_stock_daily(ts_code='000001.SZ', start_date='20240101', end_date='20241231')
        >>> df = get_stock_daily(ts_code='000001.SZ', adj='hfq')  # 后复权
    """
    collector = StockDailyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date, adj=adj)


def get_stock_weekly(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adj: AdjustType = 'qfq'
) -> pd.DataFrame:
    """
    获取股票周K线数据
    
    Args:
        ts_code: 证券代码
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
        adj: 复权类型
    
    Returns:
        DataFrame: 股票周K线数据
    """
    collector = StockWeeklyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date, adj=adj)


def get_stock_monthly(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adj: AdjustType = 'qfq'
) -> pd.DataFrame:
    """
    获取股票月K线数据
    
    Args:
        ts_code: 证券代码
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
        adj: 复权类型
    
    Returns:
        DataFrame: 股票月K线数据
    """
    collector = StockMonthlyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date, adj=adj)


def get_index_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取指数日K线数据
    
    Args:
        ts_code: 指数代码（如 000001.SH）
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 指数日K线数据
    
    Example:
        >>> df = get_index_daily(ts_code='000001.SH', start_date='20240101')
    """
    collector = IndexDailyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date)


def get_etf_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取ETF日K线数据
    
    Args:
        ts_code: ETF代码（如 510050.SH）
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: ETF日K线数据
    
    Example:
        >>> df = get_etf_daily(ts_code='510050.SH', start_date='20240101')
    """
    collector = ETFDailyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date)
