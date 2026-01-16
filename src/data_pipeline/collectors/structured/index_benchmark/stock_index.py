"""
股票指数数据（Stock Index Data）采集模块

数据类型包括：
- 指数基本信息
- 指数日线行情
- 指数成分股
- 指数成分权重
"""

import logging
from typing import Optional, List
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


@CollectorRegistry.register("index_basic")
class IndexBasicCollector(BaseCollector):
    """
    指数基本信息采集器
    
    采集A股指数的基本信息
    主数据源：Tushare (index_basic) - 需120积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 指数代码
        'name',                 # 指数简称
        'market',               # 市场（MSCI/CSI/SSE/SZSE/CICC/SW/OTH）
        'publisher',            # 发布方
        'category',             # 指数类别
        'base_date',            # 基期
        'base_point',           # 基点
        'list_date',            # 发布日期
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        market: Optional[str] = None,
        publisher: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集指数基本信息
        
        Args:
            ts_code: 指数代码
            market: 市场类型（MSCI/CSI/SSE/SZSE/CICC/SW/OTH）
            publisher: 发布商
            category: 指数类别
        
        Returns:
            DataFrame: 标准化的指数基本信息数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, market, publisher, category)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条指数基本信息")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取指数基本信息失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(market)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条指数基本信息")
                return df
        except Exception as e:
            logger.error(f"AkShare获取指数基本信息失败: {e}")
        
        logger.error("所有数据源均无法获取指数基本信息")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        market: Optional[str],
        publisher: Optional[str],
        category: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取指数基本信息"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if market:
            params['market'] = market
        if publisher:
            params['publisher'] = publisher
        if category:
            params['category'] = category
        
        df = pro.index_basic(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, market: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取指数基本信息"""
        import akshare as ak
        
        try:
            # AkShare获取实时指数行情作为基本信息
            df = ak.stock_zh_index_spot_em()
        except Exception as e:
            logger.warning(f"AkShare获取指数基本信息失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '代码': 'ts_code',
            '名称': 'name',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 格式化代码
        if 'ts_code' in df.columns:
            def format_code(code):
                code = str(code)
                if code.startswith('0'):
                    return code + '.SH'
                elif code.startswith('39'):
                    return code + '.SZ'
                return code
            df['ts_code'] = df['ts_code'].apply(format_code)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("index_daily")
class IndexDailyCollector(BaseCollector):
    """
    指数日线行情采集器
    
    采集A股指数的日线行情数据
    主数据源：Tushare (index_daily) - 需120积分
    备用数据源：AkShare, BaoStock
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 指数代码
        'trade_date',           # 交易日期
        'close',                # 收盘点位
        'open',                 # 开盘点位
        'high',                 # 最高点位
        'low',                  # 最低点位
        'pre_close',            # 昨收点位
        'change',               # 涨跌点
        'pct_chg',              # 涨跌幅（%）
        'vol',                  # 成交量（手）
        'amount',               # 成交额（千元）
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
        采集指数日线行情数据
        
        Args:
            ts_code: 指数代码
            trade_date: 交易日期（YYYYMMDD）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的指数日线行情数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条指数日线数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取指数日线失败: {e}")
        
        # 降级到AkShare
        try:
            if ts_code:
                df = self._collect_from_akshare(ts_code, start_date, end_date)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条指数日线数据")
                    return df
        except Exception as e:
            logger.warning(f"AkShare获取指数日线失败: {e}")
        
        # 降级到BaoStock
        try:
            if ts_code:
                df = self._collect_from_baostock(ts_code, start_date, end_date)
                if not df.empty:
                    logger.info(f"从BaoStock成功获取 {len(df)} 条指数日线数据")
                    return df
        except Exception as e:
            logger.error(f"BaoStock获取指数日线失败: {e}")
        
        logger.error("所有数据源均无法获取指数日线数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取指数日线"""
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
        
        # 日期格式转换
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """从AkShare获取指数日线"""
        import akshare as ak
        
        try:
            # 提取代码
            code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            
            # 判断市场
            if ts_code.endswith('.SH') or code.startswith('0'):
                df = ak.stock_zh_index_daily(symbol=f'sh{code}')
            else:
                df = ak.stock_zh_index_daily(symbol=f'sz{code}')
        except Exception as e:
            logger.warning(f"AkShare获取指数日线失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            'date': 'trade_date',
            'open': 'open',
            'close': 'close',
            'high': 'high',
            'low': 'low',
            'volume': 'vol',
        }
        df = self._standardize_columns(df, column_mapping)
        
        df['ts_code'] = ts_code
        
        # 日期筛选
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
            if start_date:
                start_dt = pd.to_datetime(start_date, format='%Y%m%d')
                df = df[pd.to_datetime(df['trade_date']) >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date, format='%Y%m%d')
                df = df[pd.to_datetime(df['trade_date']) <= end_dt]
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_baostock(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """从BaoStock获取指数日线"""
        import baostock as bs
        
        try:
            # 登录BaoStock
            lg = bs.login()
            
            # 转换代码格式
            code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            if ts_code.endswith('.SH') or code.startswith('0'):
                bs_code = f'sh.{code}'
            else:
                bs_code = f'sz.{code}'
            
            # 格式化日期
            start = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d') if start_date else '2020-01-01'
            end = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
            
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                start_date=start,
                end_date=end,
                frequency='d'
            )
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            bs.logout()
            
            if not data_list:
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
        except Exception as e:
            logger.warning(f"BaoStock获取指数日线失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            'date': 'trade_date',
            'preclose': 'pre_close',
            'volume': 'vol',
            'pctChg': 'pct_chg',
        }
        df = self._standardize_columns(df, column_mapping)
        
        df['ts_code'] = ts_code
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("index_weight")
class IndexWeightCollector(BaseCollector):
    """
    指数成分权重采集器
    
    采集指数成分股及权重
    主数据源：Tushare (index_weight) - 需400积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'index_code',           # 指数代码
        'con_code',             # 成分股代码
        'trade_date',           # 交易日期
        'weight',               # 权重（%）
    ]
    
    def collect(
        self,
        index_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集指数成分权重数据
        
        Args:
            index_code: 指数代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的指数成分权重数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(index_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条指数权重数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取指数权重失败: {e}")
        
        # 降级到AkShare
        try:
            if index_code:
                df = self._collect_from_akshare(index_code)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条指数权重数据")
                    return df
        except Exception as e:
            logger.error(f"AkShare获取指数权重失败: {e}")
        
        logger.error("所有数据源均无法获取指数权重数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        index_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取指数权重"""
        pro = self.tushare_api
        
        params = {}
        if index_code:
            params['index_code'] = index_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = pro.index_weight(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, index_code: str) -> pd.DataFrame:
        """从AkShare获取指数权重"""
        import akshare as ak
        
        try:
            code = index_code.split('.')[0] if '.' in index_code else index_code
            
            # 沪深300成分
            if code in ['000300', '399300']:
                df = ak.index_stock_cons_weight_csindex(symbol='000300')
            # 上证50
            elif code in ['000016']:
                df = ak.index_stock_cons_weight_csindex(symbol='000016')
            # 中证500
            elif code in ['000905', '399905']:
                df = ak.index_stock_cons_weight_csindex(symbol='000905')
            else:
                logger.warning(f"AkShare不支持该指数的权重查询: {index_code}")
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        except Exception as e:
            logger.warning(f"AkShare获取指数权重失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '成分券代码': 'con_code',
            '权重': 'weight',
            '日期': 'trade_date',
        }
        df = self._standardize_columns(df, column_mapping)
        
        df['index_code'] = index_code
        
        # 格式化成分股代码
        if 'con_code' in df.columns:
            def format_code(code):
                code = str(code)
                if code.startswith('6'):
                    return code + '.SH'
                elif code.startswith(('0', '3')):
                    return code + '.SZ'
                return code
            df['con_code'] = df['con_code'].apply(format_code)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("index_member")
class IndexMemberCollector(BaseCollector):
    """
    指数成分股采集器
    
    采集指数成分股列表（不含权重）
    主数据源：Tushare (index_member) - 需5000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'index_code',           # 指数代码
        'con_code',             # 成分股代码
        'con_name',             # 成分股名称
        'in_date',              # 纳入日期
        'is_new',               # 是否最新
    ]
    
    def collect(
        self,
        index_code: Optional[str] = None,
        ts_code: Optional[str] = None,
        is_new: str = 'Y',
        **kwargs
    ) -> pd.DataFrame:
        """
        采集指数成分股数据
        
        Args:
            index_code: 指数代码
            ts_code: 成分股代码（反查所属指数）
            is_new: 是否最新（Y/N）
        
        Returns:
            DataFrame: 标准化的指数成分股数据
        """
        # 优先使用Tushare（需5000积分，2120可能不够）
        try:
            df = self._collect_from_tushare(index_code, ts_code, is_new)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条指数成分股数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取指数成分股失败（可能积分不足）: {e}")
        
        # 降级到AkShare
        try:
            if index_code:
                df = self._collect_from_akshare(index_code)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条指数成分股数据")
                    return df
        except Exception as e:
            logger.error(f"AkShare获取指数成分股失败: {e}")
        
        logger.error("所有数据源均无法获取指数成分股数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        index_code: Optional[str],
        ts_code: Optional[str],
        is_new: str
    ) -> pd.DataFrame:
        """从Tushare获取指数成分股"""
        pro = self.tushare_api
        
        params = {'is_new': is_new}
        if index_code:
            params['index_code'] = index_code
        if ts_code:
            params['ts_code'] = ts_code
        
        df = pro.index_member(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, index_code: str) -> pd.DataFrame:
        """从AkShare获取指数成分股"""
        import akshare as ak
        
        try:
            code = index_code.split('.')[0] if '.' in index_code else index_code
            
            # 沪深300
            if code in ['000300', '399300']:
                df = ak.index_stock_cons(symbol='000300')
            # 上证50
            elif code in ['000016']:
                df = ak.index_stock_cons(symbol='000016')
            # 中证500
            elif code in ['000905', '399905']:
                df = ak.index_stock_cons(symbol='000905')
            # 中证1000
            elif code in ['000852']:
                df = ak.index_stock_cons(symbol='000852')
            else:
                logger.warning(f"AkShare不支持该指数: {index_code}")
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        except Exception as e:
            logger.warning(f"AkShare获取指数成分股失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '品种代码': 'con_code',
            '品种名称': 'con_name',
            '纳入日期': 'in_date',
        }
        df = self._standardize_columns(df, column_mapping)
        
        df['index_code'] = index_code
        df['is_new'] = 'Y'
        
        # 格式化成分股代码
        if 'con_code' in df.columns:
            def format_code(code):
                code = str(code)
                if code.startswith('6'):
                    return code + '.SH'
                elif code.startswith(('0', '3')):
                    return code + '.SZ'
                return code
            df['con_code'] = df['con_code'].apply(format_code)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_index_basic(
    ts_code: Optional[str] = None,
    market: Optional[str] = None,
    publisher: Optional[str] = None,
    category: Optional[str] = None
) -> pd.DataFrame:
    """
    获取指数基本信息
    
    Args:
        ts_code: 指数代码
        market: 市场类型
        publisher: 发布商
        category: 指数类别
    
    Returns:
        DataFrame: 指数基本信息数据
    
    Example:
        >>> df = get_index_basic(market='SSE')
        >>> df = get_index_basic(ts_code='000001.SH')
    """
    collector = IndexBasicCollector()
    return collector.collect(ts_code=ts_code, market=market, publisher=publisher, category=category)


def get_index_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取指数日线行情
    
    Args:
        ts_code: 指数代码
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 指数日线行情数据
    
    Example:
        >>> df = get_index_daily(ts_code='000001.SH', start_date='20250101', end_date='20251231')
    """
    collector = IndexDailyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date)


def get_index_weight(
    index_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取指数成分权重
    
    Args:
        index_code: 指数代码
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 指数成分权重数据
    
    Example:
        >>> df = get_index_weight(index_code='000300.SH', trade_date='20250115')
    """
    collector = IndexWeightCollector()
    return collector.collect(index_code=index_code, trade_date=trade_date, start_date=start_date, end_date=end_date)


def get_index_member(
    index_code: Optional[str] = None,
    ts_code: Optional[str] = None,
    is_new: str = 'Y'
) -> pd.DataFrame:
    """
    获取指数成分股
    
    Args:
        index_code: 指数代码
        ts_code: 成分股代码（反查所属指数）
        is_new: 是否最新
    
    Returns:
        DataFrame: 指数成分股数据
    
    Example:
        >>> df = get_index_member(index_code='000300.SH')
        >>> df = get_index_member(ts_code='000001.SZ')
    """
    collector = IndexMemberCollector()
    return collector.collect(index_code=index_code, ts_code=ts_code, is_new=is_new)
