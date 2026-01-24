"""
ETF与基金（ETF & Fund）采集模块

数据类型包括：
- ETF基本信息
- ETF/LOF行情
- 基金净值
- 基金持仓
- 基金规模
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


@CollectorRegistry.register("fund_basic")
class FundBasicCollector(BaseCollector):
    """
    公募基金列表采集器
    
    获取公募基金基本信息列表，包括场内和场外基金
    主数据源：Tushare (fund_basic, 2000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 基金代码
        'name',             # 简称
        'management',       # 管理人
        'custodian',        # 托管人
        'fund_type',        # 投资类型
        'found_date',       # 成立日期
        'due_date',         # 到期日期
        'list_date',        # 上市时间
        'issue_date',       # 发行日期
        'delist_date',      # 退市日期
        'issue_amount',     # 发行份额(亿)
        'm_fee',            # 管理费
        'c_fee',            # 托管费
        'duration_year',    # 存续期
        'p_value',          # 面值
        'min_amount',       # 起点金额(万元)
        'exp_return',       # 预期收益率
        'benchmark',        # 业绩比较基准
        'status',           # 存续状态
        'invest_type',      # 投资风格
        'type',             # 基金类型
        'market',           # E场内 O场外
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        market: str = 'E',
        status: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集公募基金列表
        
        Args:
            ts_code: 基金代码
            market: 交易市场（E场内，O场外，默认E）
            status: 存续状态（D摘牌，I发行，L上市中）
        
        Returns:
            DataFrame: 标准化的基金列表数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, market, status)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条基金列表数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取基金列表失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(market)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条基金列表数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取基金列表失败: {e}")
        
        logger.error("无法获取基金列表数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        market: str,
        status: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取基金列表"""
        params = {'market': market}
        if ts_code:
            params['ts_code'] = ts_code
        if status:
            params['status'] = status
        
        df = self.tushare_api.fund_basic(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保所有输出字段存在
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, market: str = 'E') -> pd.DataFrame:
        """从AkShare获取基金列表"""
        import akshare as ak
        
        try:
            if market == 'E':
                # 场内ETF基金列表
                df = ak.fund_etf_spot_em()
            else:
                # 场外基金
                df = ak.fund_open_fund_rank_em(symbol="全部")
        except Exception as e:
            logger.warning(f"AkShare获取基金列表失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '代码': 'ts_code',
            '基金代码': 'ts_code',
            '名称': 'name',
            '基金简称': 'name',
            '类型': 'fund_type',
        }
        df = self._standardize_columns(df, column_mapping)
        df['market'] = market
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("fund_daily")
class FundDailyCollector(BaseCollector):
    """
    ETF日线行情采集器
    
    获取ETF行情每日收盘数据
    主数据源：Tushare (fund_daily, 5000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # TS代码
        'trade_date',       # 交易日期
        'open',             # 开盘价
        'high',             # 最高价
        'low',              # 最低价
        'close',            # 收盘价
        'pre_close',        # 昨收盘价
        'change',           # 涨跌额
        'pct_chg',          # 涨跌幅(%)
        'vol',              # 成交量(手)
        'amount',           # 成交额(千元)
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
        采集ETF日线行情
        
        Args:
            ts_code: 基金代码
            trade_date: 交易日期（YYYYMMDD）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的ETF行情数据
        """
        # 优先使用Tushare（需要5000积分）
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条ETF行情数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取ETF行情失败（可能积分不足，需5000积分）: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code, start_date, end_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条ETF行情数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取ETF行情失败: {e}")
        
        logger.error("无法获取ETF行情数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取ETF行情"""
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = self.tushare_api.fund_daily(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(
        self,
        ts_code: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从AkShare获取ETF行情"""
        import akshare as ak
        
        if not ts_code:
            logger.warning("AkShare需要指定ts_code获取ETF行情")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 提取纯代码
        code = ts_code.split('.')[0]
        
        try:
            df = ak.fund_etf_hist_em(symbol=code, adjust="")
        except Exception as e:
            logger.warning(f"AkShare获取ETF历史行情失败: {e}")
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
        df['ts_code'] = ts_code
        
        # 日期格式转换
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
            if start_date:
                df = df[df['trade_date'] >= start_date]
            if end_date:
                df = df[df['trade_date'] <= end_date]
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("fund_nav")
class FundNavCollector(BaseCollector):
    """
    公募基金净值采集器
    
    获取公募基金净值数据
    主数据源：Tushare (fund_nav, 2000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # TS代码
        'ann_date',         # 公告日期
        'nav_date',         # 净值日期
        'unit_nav',         # 单位净值
        'accum_nav',        # 累计净值
        # 'accum_div',      # 累计分红（数据源不提供）
        # 'net_asset',      # 资产净值（ETF类型通常不提供）
        # 'total_netasset', # 合计资产净值（ETF类型通常不提供）
        'adj_nav',          # 复权单位净值
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        nav_date: Optional[str] = None,
        market: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集基金净值数据
        
        Args:
            ts_code: TS基金代码
            nav_date: 净值日期
            market: E场内 O场外
            start_date: 净值开始日期
            end_date: 净值结束日期
        
        Returns:
            DataFrame: 标准化的基金净值数据
        """
        # 仅使用Tushare获取，因为AkShare不提供公告日期(ann_date)
        try:
            df = self._collect_from_tushare(ts_code, nav_date, market, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条基金净值数据")
                return df
        except Exception as e:
            logger.error(f"获取基金净值最后失败: {e}")
        
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=10, delay=10.0, backoff_factor=1.5)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        nav_date: Optional[str],
        market: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取基金净值"""
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if nav_date:
            params['nav_date'] = nav_date
        if market:
            params['market'] = market
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = self.tushare_api.fund_nav(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
        # 严格过滤日期范围
        if 'nav_date' in df.columns and end_date:
            df = df[df['nav_date'] <= end_date]
        if 'ann_date' in df.columns and end_date:
            df = df[df['ann_date'] <= end_date]
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    



@CollectorRegistry.register("fund_portfolio")
class FundPortfolioCollector(BaseCollector):
    """
    公募基金持仓采集器
    
    获取公募基金持仓数据，季度更新
    主数据源：Tushare (fund_portfolio, 5000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # TS基金代码
        'ann_date',         # 公告日期
        'end_date',         # 截止日期
        'symbol',           # 股票代码
        'mkv',              # 持有股票市值(元)
        'amount',           # 持有股票数量(股)
        'stk_mkv_ratio',    # 占股票市值比
        'stk_float_ratio',  # 占流通股本比例
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        symbol: Optional[str] = None,
        ann_date: Optional[str] = None,
        period: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集基金持仓数据
        
        Args:
            ts_code: 基金代码
            symbol: 股票代码
            ann_date: 公告日期（YYYYMMDD）
            period: 季度（每个季度最后一天，如20231231）
            start_date: 报告期开始日期
            end_date: 报告期结束日期
        
        Returns:
            DataFrame: 标准化的基金持仓数据
        """
        # 优先使用Tushare（需要5000积分）
        try:
            df = self._collect_from_tushare(ts_code, symbol, ann_date, period, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条基金持仓数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取基金持仓失败（可能积分不足，需5000积分）: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条基金持仓数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取基金持仓失败: {e}")
        
        logger.error("无法获取基金持仓数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        symbol: Optional[str],
        ann_date: Optional[str],
        period: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取基金持仓"""
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if symbol:
            params['symbol'] = symbol
        if ann_date:
            params['ann_date'] = ann_date
        if period:
            params['period'] = period
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = self.tushare_api.fund_portfolio(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, ts_code: Optional[str]) -> pd.DataFrame:
        """从AkShare获取基金持仓"""
        import akshare as ak
        
        if not ts_code:
            logger.warning("AkShare需要指定ts_code获取基金持仓")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 提取纯代码
        code = ts_code.split('.')[0]
        
        try:
            df = ak.fund_portfolio_hold_em(symbol=code)
        except Exception as e:
            logger.warning(f"AkShare获取基金持仓失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '股票代码': 'symbol',
            '股票名称': 'stk_name',
            '持仓市值': 'mkv',
            '持仓数量': 'amount',
            '占净值比例': 'stk_mkv_ratio',
            '季度': 'period',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("fund_share")
class FundShareCollector(BaseCollector):
    """
    基金规模采集器
    
    获取基金规模（份额）数据
    主数据源：Tushare (fund_share, 2000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 基金代码
        'trade_date',       # 交易日期
        'fd_share',         # 基金份额（万）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        market: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集基金规模数据
        
        Args:
            ts_code: TS基金代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            market: 市场代码（SH/SZ）
        
        Returns:
            DataFrame: 标准化的基金规模数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date, market)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条基金规模数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取基金规模失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条基金规模数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取基金规模失败: {e}")
        
        logger.error("无法获取基金规模数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        market: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取基金规模"""
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if market:
            params['market'] = market
        
        df = self.tushare_api.fund_share(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, ts_code: Optional[str]) -> pd.DataFrame:
        """从AkShare获取基金规模"""
        import akshare as ak
        
        if not ts_code:
            logger.warning("AkShare需要指定ts_code获取基金规模")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        code = ts_code.split('.')[0]
        
        try:
            # 获取ETF实时行情中的规模数据
            df = ak.fund_etf_spot_em()
            df = df[df['代码'] == code]
        except Exception as e:
            logger.warning(f"AkShare获取基金规模失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 构造输出
        result = pd.DataFrame({
            'ts_code': [ts_code],
            'trade_date': [datetime.now().strftime('%Y%m%d')],
            'fd_share': [None]  # AkShare可能没有份额数据
        })
        
        return result


# ============= 便捷函数接口 =============

def get_fund_basic(
    ts_code: Optional[str] = None,
    market: str = 'E',
    status: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取公募基金列表
    
    Args:
        ts_code: 基金代码
        market: 交易市场（E场内，O场外）
        status: 存续状态（D摘牌，I发行，L上市中）
        **kwargs: 其他参数（由调度器传入）
    
    Returns:
        DataFrame: 基金列表数据
    
    Example:
        >>> df = get_fund_basic(market='E')  # 场内基金
        >>> df = get_fund_basic(market='O', status='L')  # 场外上市基金
    """
    collector = FundBasicCollector()
    return collector.collect(ts_code=ts_code, market=market, status=status, **kwargs)


def get_fund_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取ETF日线行情
    
    Args:
        ts_code: 基金代码
        trade_date: 交易日期（YYYYMMDD）
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: ETF行情数据
    
    Example:
        >>> df = get_fund_daily(ts_code='510300.SH', start_date='20250101')
    """
    collector = FundDailyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date, **kwargs)


def get_fund_nav(
    ts_code: Optional[str] = None,
    nav_date: Optional[str] = None,
    market: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取公募基金净值
    
    Args:
        ts_code: TS基金代码
        nav_date: 净值日期
        market: E场内 O场外
        start_date: 净值开始日期
        end_date: 净值结束日期
    
    Returns:
        DataFrame: 基金净值数据
    
    Example:
        >>> df = get_fund_nav(ts_code='165509.SZ')
        >>> df = get_fund_nav(nav_date='20250115')
    """
    collector = FundNavCollector()
    return collector.collect(ts_code=ts_code, nav_date=nav_date,
                            market=market, start_date=start_date, end_date=end_date, **kwargs)


def get_fund_portfolio(
    ts_code: Optional[str] = None,
    symbol: Optional[str] = None,
    ann_date: Optional[str] = None,
    period: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取公募基金持仓
    
    Args:
        ts_code: 基金代码
        symbol: 股票代码
        ann_date: 公告日期
        period: 季度（如20231231）
        **kwargs: 其他参数（由调度器传入）
    
    Returns:
        DataFrame: 基金持仓数据
    
    Example:
        >>> df = get_fund_portfolio(ts_code='001753.OF')
        >>> df = get_fund_portfolio(period='20231231')
    """
    collector = FundPortfolioCollector()
    return collector.collect(ts_code=ts_code, symbol=symbol,
                            ann_date=ann_date, period=period, **kwargs)


def get_fund_share(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取基金规模（份额）
    
    Args:
        ts_code: TS基金代码
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 基金规模数据
    
    Example:
        >>> df = get_fund_share(ts_code='150018.SZ')
    """
    collector = FundShareCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date, **kwargs)
