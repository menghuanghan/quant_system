"""
交易日历与制度信息（Trading Calendar）采集模块

数据类型包括：
- 交易日历（股票/期货/港股/美股）
- 停复牌信息
- 涨跌停规则
- 集合竞价时间
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta

import time
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


@CollectorRegistry.register("trade_calendar")
class TradeCalendarCollector(BaseCollector):
    """
    交易日历采集器
    
    采集各交易所交易日历数据
    主数据源：Tushare (trade_cal)
    备用数据源：AkShare (tool_trade_date_hist_sina), BaoStock (query_trade_dates)
    """
    
    # 交易所代码映射
    EXCHANGE_MAP = {
        # Tushare交易所代码
        'SSE': '上海证券交易所',
        'SZSE': '深圳证券交易所',
        'BSE': '北京证券交易所',
        'CFFEX': '中国金融期货交易所',
        'SHFE': '上海期货交易所',
        'CZCE': '郑州商品交易所',
        'DCE': '大连商品交易所',
        'INE': '上海国际能源交易中心',
        'HKEX': '香港交易所',
        'NYSE': '纽约证券交易所',
        'NASDAQ': '纳斯达克',
    }
    
    OUTPUT_FIELDS = [
        'exchange',         # 交易所代码
        'cal_date',         # 日历日期
        'is_open',          # 是否交易日（1=是，0=否）
        'pretrade_date',    # 上一个交易日
    ]
    
    def collect(
        self,
        exchange: str = 'SSE',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        is_open: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集交易日历数据
        
        Args:
            exchange: 交易所代码，默认SSE（上交所）
                     可选值：SSE/SZSE/BSE/CFFEX/SHFE/CZCE/DCE/INE
            start_date: 开始日期（YYYYMMDD格式）
            end_date: 结束日期（YYYYMMDD格式）
            is_open: 筛选交易日（1=交易日，0=非交易日，None=全部）
        
        Returns:
            DataFrame: 标准化的交易日历数据
            
        输出字段:
            - exchange: 交易所代码
            - cal_date: 日历日期
            - is_open: 是否交易日
            - pretrade_date: 上一个交易日
        """
        # 优先使用传入参数，如果没有则设置默认日期范围
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if not end_date:
            # 默认不应超过当前时间太多，除非是显式扩充。这里根据全量采集习惯设定。
            end_date = kwargs.get('end_date') or datetime.now().strftime('%Y%m%d')
        
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(exchange, start_date, end_date, is_open)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {exchange} 交易日历 {len(df)} 条")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取交易日历失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(exchange, start_date, end_date, is_open)
            if not df.empty:
                logger.info(f"从AkShare成功获取交易日历 {len(df)} 条")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取交易日历失败: {e}")
        
        # 降级到BaoStock
        try:
            df = self._collect_from_baostock(start_date, end_date)
            if not df.empty:
                logger.info(f"从BaoStock成功获取交易日历 {len(df)} 条")
                return df
        except Exception as e:
            logger.error(f"BaoStock获取交易日历失败: {e}")
        
        logger.error("所有数据源均无法获取交易日历数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        exchange: str,
        start_date: str,
        end_date: str,
        is_open: Optional[int]
    ) -> pd.DataFrame:
        """从Tushare获取交易日历"""
        pro = self.tushare_api
        
        params = {
            'exchange': exchange,
            'start_date': start_date,
            'end_date': end_date,
        }
        if is_open is not None:
            params['is_open'] = str(is_open)
        
        df = pro.trade_cal(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化日期格式
        df = self._convert_date_format(df, ['cal_date', 'pretrade_date'])
        
        # 确保包含所有字段
        df = self._ensure_columns(df, self.OUTPUT_FIELDS)
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(
        self,
        exchange: str,
        start_date: str,
        end_date: str,
        is_open: Optional[int]
    ) -> pd.DataFrame:
        """从AkShare获取交易日历"""
        import akshare as ak
        
        # AkShare提供的是新浪交易日历（A股通用）
        df = ak.tool_trade_date_hist_sina()
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        df = df.rename(columns={'trade_date': 'cal_date'})
        df['exchange'] = exchange  # 使用传入的交易所
        df['is_open'] = 1  # AkShare只返回交易日
        df['pretrade_date'] = df['cal_date'].shift(1)
        
        # 转换日期格式
        df['cal_date'] = pd.to_datetime(df['cal_date']).dt.strftime('%Y-%m-%d')
        df['pretrade_date'] = pd.to_datetime(df['pretrade_date']).dt.strftime('%Y-%m-%d')
        
        # 筛选日期范围
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df['_cal_dt'] = pd.to_datetime(df['cal_date'])
        df = df[(df['_cal_dt'] >= start_dt) & (df['_cal_dt'] <= end_dt)]
        df = df.drop(columns=['_cal_dt'])
        
        # 筛选is_open
        if is_open is not None:
            df = df[df['is_open'] == is_open]
        
        # 确保包含所有字段
        df = self._ensure_columns(df, self.OUTPUT_FIELDS)
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_baostock(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """从BaoStock获取交易日历"""
        import baostock as bs
        
        # 确保登录
        if not self.source_manager.ensure_baostock_login():
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 转换日期格式（BaoStock需要带连字符）
        start_dt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end_dt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        
        rs = bs.query_trade_dates(start_date=start_dt, end_date=end_dt)
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 标准化字段
        column_mapping = {
            'calendar_date': 'cal_date',
            'is_trading_day': 'is_open',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置交易所（BaoStock默认上交所）
        df['exchange'] = 'SSE'
        
        # 计算上一个交易日
        trade_days = df[df['is_open'].astype(int) == 1]['cal_date'].tolist()
        pretrade_map = {trade_days[i]: trade_days[i-1] for i in range(1, len(trade_days))}
        df['pretrade_date'] = df['cal_date'].map(pretrade_map)
        
        # 确保包含所有字段
        df = self._ensure_columns(df, self.OUTPUT_FIELDS)
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("suspend_info")
class SuspendInfoCollector(BaseCollector):
    """
    停复牌信息采集器
    
    采集股票每日停复牌数据
    主数据源：Tushare (suspend_d)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'trade_date',       # 交易日期
        'suspend_type',     # 停复牌类型（S=停牌，R=复牌）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        suspend_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集停复牌信息
        
        Args:
            ts_code: 证券代码
            trade_date: 交易日期（YYYYMMDD格式）
            suspend_type: 停复牌类型（S=停牌，R=复牌）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的停复牌数据
            
        输出字段:
            - ts_code: 证券代码
            - trade_date: 交易日期
            - suspend_type: 停复牌类型
            - suspend_timing: 停牌时间段
            - suspend_reason: 停牌原因
            - ann_date: 公告日期
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(
                ts_code, trade_date, suspend_type, start_date, end_date
            )
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条停复牌记录")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取停复牌信息失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(trade_date, end_date=end_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条停复牌记录")
                return df
        except Exception as e:
            logger.error(f"AkShare获取停复牌信息失败: {e}")
        
        logger.error("所有数据源均无法获取停复牌数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        suspend_type: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取停复牌信息"""
        pro = self.tushare_api
        
        params = {}
        if ts_code: params['ts_code'] = ts_code
        if trade_date: params['trade_date'] = trade_date
        if suspend_type: params['suspend_type'] = suspend_type
        
        # 范围查询优化：支持分块采集以绕过条数限制 (Tushare 限制 5000 条)
        if start_date and end_date and not ts_code and not trade_date:
            all_dfs = []
            try:
                from dateutil.relativedelta import relativedelta
                s_dt = datetime.strptime(start_date, '%Y%m%d')
                e_dt = datetime.strptime(end_date, '%Y%m%d')
                
                curr_dt = s_dt
                while curr_dt <= e_dt:
                    # 按月采集以确保条数不超过 5000
                    chunk_start = curr_dt.strftime('%Y%m%d')
                    chunk_next = curr_dt + relativedelta(months=1)
                    chunk_end = (chunk_next - timedelta(days=1)).strftime('%Y%m%d')
                    if chunk_end > end_date:
                        chunk_end = end_date
                    
                    logger.info(f"正在采集停复牌信息: {chunk_start} - {chunk_end}")
                    chunk = pro.suspend_d(start_date=chunk_start, end_date=chunk_end, **params)
                    if not chunk.empty:
                        all_dfs.append(chunk)
                        if len(chunk) >= 5000:
                            logger.warning(f"警告：{chunk_start}-{chunk_end} 采集达到 5000 条上限，可能存在数据截断！")
                    
                    curr_dt = chunk_next
                    time.sleep(0.4) # 避免频率限制
                
                if not all_dfs: return pd.DataFrame(columns=self.OUTPUT_FIELDS)
                df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
                return self._post_process_suspend(df)
            except Exception as e:
                logger.error(f"停复牌范围采集逻辑错误: {e}")

        # 单次或已指定代码的查询
        if start_date: params['start_date'] = start_date
        if end_date: params['end_date'] = end_date
        if not params: params['trade_date'] = trade_date or end_date or datetime.now().strftime('%Y%m%d')
        
        df = pro.suspend_d(**params)
        return self._post_process_suspend(df)

    def _post_process_suspend(self, df: pd.DataFrame) -> pd.DataFrame:
        """停复牌数据后处理"""
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化日期格式
        df = self._convert_date_format(df, ['trade_date', 'ann_date'])
        
        # 映射suspend_reason（如果存在）
        if 'suspend_reason' not in df.columns:
            df['suspend_reason'] = None
        
        # 确保包含所有字段
        df = self._ensure_columns(df, self.OUTPUT_FIELDS)
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, trade_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取今日停复牌信息"""
        import akshare as ak
        
        try:
            # 标记日期：优先使用显式 trade_date，其次是 end_date，最后是当前
            eff_date = trade_date or end_date or datetime.now().strftime('%Y%m%d')
            if len(eff_date.replace('-', '')) == 8:
                eff_date = f"{eff_date[:4]}-{eff_date[4:6]}-{eff_date[6:8]}"
            
            df = ak.stock_tfp_em()  # 东方财富停复牌数据
            
            if df.empty:
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            # 标准化字段
            column_mapping = {
                '代码': 'symbol',
                '名称': 'name',
                '停牌时间': 'suspend_start',
                '停牌截止时间': 'suspend_end',
                '停牌原因': 'suspend_reason',
                '停牌期限': 'suspend_period',
            }
            df = self._standardize_columns(df, column_mapping)
            
            # 生成ts_code
            if 'symbol' in df.columns:
                df['ts_code'] = df['symbol'].apply(self._symbol_to_tscode)
            
            # 设置trade_date
            df['trade_date'] = eff_date
            df['suspend_type'] = 'S'  # 停牌
            df['suspend_timing'] = None
            df['ann_date'] = None
            
            # 确保包含所有字段
            df = self._ensure_columns(df, self.OUTPUT_FIELDS)
            
            return df[self.OUTPUT_FIELDS]
        except Exception as e:
            logger.warning(f"AkShare停复牌接口调用失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
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


@CollectorRegistry.register("price_limit_rule")
class PriceLimitRuleCollector(BaseCollector):
    """
    涨跌停规则采集器
    
    采集各市场板块的涨跌停规则配置
    注：该数据为静态规则配置，非动态行情数据
    """
    
    OUTPUT_FIELDS = [
        'market',           # 市场/板块
        'exchange',         # 交易所
        'limit_up_pct',     # 涨停幅度（%）
        'limit_down_pct',   # 跌停幅度（%）
        'first_day_limit',  # 首日涨跌幅限制
        'st_limit_pct',     # ST股涨跌幅限制
        'effective_date',   # 生效日期
        'description',      # 规则说明
    ]
    
    # A股涨跌停规则配置（静态数据）
    A_SHARE_RULES = [
        {
            'market': '主板',
            'exchange': 'SSE/SZSE',
            'limit_up_pct': 10.0,
            'limit_down_pct': -10.0,
            'first_day_limit': '44%涨幅，无跌幅限制',
            'st_limit_pct': 5.0,
            'effective_date': '1996-12-16',
            'description': 'A股主板涨跌停制度',
        },
        {
            'market': '创业板',
            'exchange': 'SZSE',
            'limit_up_pct': 20.0,
            'limit_down_pct': -20.0,
            'first_day_limit': '无涨跌幅限制（前5日）',
            'st_limit_pct': 20.0,
            'effective_date': '2020-08-24',
            'description': '创业板注册制涨跌幅规则',
        },
        {
            'market': '科创板',
            'exchange': 'SSE',
            'limit_up_pct': 20.0,
            'limit_down_pct': -20.0,
            'first_day_limit': '无涨跌幅限制（前5日）',
            'st_limit_pct': 20.0,
            'effective_date': '2019-07-22',
            'description': '科创板涨跌幅规则',
        },
        {
            'market': '北交所',
            'exchange': 'BSE',
            'limit_up_pct': 30.0,
            'limit_down_pct': -30.0,
            'first_day_limit': '无涨跌幅限制',
            'st_limit_pct': 30.0,
            'effective_date': '2021-11-15',
            'description': '北交所涨跌幅规则',
        },
    ]
    
    def collect(
        self,
        market: Optional[str] = None,
        exchange: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        获取涨跌停规则配置
        
        Args:
            market: 市场/板块（主板/创业板/科创板/北交所）
            exchange: 交易所代码
        
        Returns:
            DataFrame: 涨跌停规则配置数据
            
        输出字段:
            - market: 市场/板块
            - exchange: 交易所
            - limit_up_pct: 涨停幅度
            - limit_down_pct: 跌停幅度
            - first_day_limit: 首日涨跌幅限制
            - st_limit_pct: ST股涨跌幅限制
            - effective_date: 生效日期
            - description: 规则说明
        """
        df = pd.DataFrame(self.A_SHARE_RULES)
        
        # 筛选市场
        if market:
            df = df[df['market'] == market]
        
        # 筛选交易所
        if exchange:
            df = df[df['exchange'].str.contains(exchange)]
        
        logger.info(f"获取到 {len(df)} 条涨跌停规则配置")
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("auction_time")
class AuctionTimeCollector(BaseCollector):
    """
    集合竞价时间采集器
    
    采集各市场的集合竞价时间配置
    """
    
    OUTPUT_FIELDS = [
        'market',               # 市场/板块
        'exchange',             # 交易所
        'open_call_start',      # 开盘集合竞价开始时间
        'open_call_end',        # 开盘集合竞价结束时间
        'open_match_time',      # 开盘撮合时间
        'close_call_start',     # 收盘集合竞价开始时间
        'close_call_end',       # 收盘集合竞价结束时间
        'close_match_time',     # 收盘撮合时间
        'continuous_start',     # 连续竞价开始时间
        'continuous_end',       # 连续竞价结束时间
        'lunch_break_start',    # 午休开始时间
        'lunch_break_end',      # 午休结束时间
        'description',          # 说明
    ]
    
    # 交易时间配置（静态数据）
    TRADING_TIME_RULES = [
        {
            'market': 'A股',
            'exchange': 'SSE/SZSE',
            'open_call_start': '09:15',
            'open_call_end': '09:25',
            'open_match_time': '09:25',
            'close_call_start': '14:57',
            'close_call_end': '15:00',
            'close_match_time': '15:00',
            'continuous_start': '09:30',
            'continuous_end': '15:00',
            'lunch_break_start': '11:30',
            'lunch_break_end': '13:00',
            'description': 'A股沪深交易时间',
        },
        {
            'market': '北交所',
            'exchange': 'BSE',
            'open_call_start': '09:15',
            'open_call_end': '09:25',
            'open_match_time': '09:25',
            'close_call_start': '14:57',
            'close_call_end': '15:00',
            'close_match_time': '15:00',
            'continuous_start': '09:30',
            'continuous_end': '15:00',
            'lunch_break_start': '11:30',
            'lunch_break_end': '13:00',
            'description': '北交所交易时间',
        },
        {
            'market': '港股',
            'exchange': 'HKEX',
            'open_call_start': '09:00',
            'open_call_end': '09:20',
            'open_match_time': '09:20',
            'close_call_start': '16:00',
            'close_call_end': '16:08',
            'close_match_time': '16:08',
            'continuous_start': '09:30',
            'continuous_end': '16:00',
            'lunch_break_start': '12:00',
            'lunch_break_end': '13:00',
            'description': '港股交易时间',
        },
    ]
    
    def collect(
        self,
        market: Optional[str] = None,
        exchange: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        获取集合竞价时间配置
        
        Args:
            market: 市场（A股/港股/美股/北交所）
            exchange: 交易所代码
        
        Returns:
            DataFrame: 集合竞价时间配置数据
            
        输出字段:
            - market: 市场/板块
            - exchange: 交易所
            - open_call_start: 开盘集合竞价开始时间
            - open_call_end: 开盘集合竞价结束时间
            - open_match_time: 开盘撮合时间
            - close_call_start: 收盘集合竞价开始时间
            - close_call_end: 收盘集合竞价结束时间
            - close_match_time: 收盘撮合时间
            - continuous_start: 连续竞价开始时间
            - continuous_end: 连续竞价结束时间
            - lunch_break_start: 午休开始时间
            - lunch_break_end: 午休结束时间
            - description: 说明
        """
        df = pd.DataFrame(self.TRADING_TIME_RULES)
        
        # 筛选市场
        if market:
            df = df[df['market'] == market]
        
        # 筛选交易所
        if exchange:
            df = df[df['exchange'].str.contains(exchange)]
        
        logger.info(f"获取到 {len(df)} 条集合竞价时间配置")
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("hk_trade_calendar")
class HKTradeCalendarCollector(BaseCollector):
    """
    港股交易日历采集器
    
    主数据源：Tushare (trade_cal exchange='HKEX')
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'exchange',         # 交易所代码
        'cal_date',         # 日历日期
        'is_open',          # 是否交易日
        'pretrade_date',    # 上一个交易日
    ]
    
    def collect(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集港股交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 港股交易日历数据
        """
        # 使用通用交易日历采集器，指定港交所
        collector = TradeCalendarCollector()
        return collector.collect(exchange='HKEX', start_date=start_date, end_date=end_date)


@CollectorRegistry.register("us_trade_calendar")
class USTradeCalendarCollector(BaseCollector):
    """
    美股交易日历采集器
    
    主数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'exchange',         # 交易所代码
        'cal_date',         # 日历日期
        'is_open',          # 是否交易日
        'pretrade_date',    # 上一个交易日
    ]
    
    def collect(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集美股交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 美股交易日历数据
        """
        try:
            df = self._collect_from_akshare(start_date, end_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取美股交易日历 {len(df)} 条")
                return df
        except Exception as e:
            logger.error(f"获取美股交易日历失败: {e}")
        
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从AkShare获取美股交易日历"""
        import akshare as ak
        
        # 尝试获取美股交易日历
        try:
            df = ak.tool_trade_date_hist_sina()
            
            if df.empty:
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            # 标准化字段
            df = df.rename(columns={'trade_date': 'cal_date'})
            df['exchange'] = 'NYSE'
            df['is_open'] = 1
            df['pretrade_date'] = df['cal_date'].shift(1)
            
            # 转换日期格式
            df['cal_date'] = pd.to_datetime(df['cal_date']).dt.strftime('%Y-%m-%d')
            df['pretrade_date'] = pd.to_datetime(df['pretrade_date']).dt.strftime('%Y-%m-%d')
            
            # 筛选日期范围
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[pd.to_datetime(df['cal_date']) >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[pd.to_datetime(df['cal_date']) <= end_dt]
            
            return df[self.OUTPUT_FIELDS]
        except Exception as e:
            logger.warning(f"AkShare美股交易日历接口调用失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)


@CollectorRegistry.register("futures_trade_calendar")
class FuturesTradeCalendarCollector(BaseCollector):
    """
    期货交易日历采集器
    
    采集各期货交易所交易日历
    主数据源：Tushare
    """
    
    OUTPUT_FIELDS = [
        'exchange',         # 交易所代码
        'cal_date',         # 日历日期
        'is_open',          # 是否交易日
        'pretrade_date',    # 上一个交易日
    ]
    
    # 期货交易所列表
    FUTURES_EXCHANGES = ['CFFEX', 'SHFE', 'CZCE', 'DCE', 'INE']
    
    def collect(
        self,
        exchange: str = 'CFFEX',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集期货交易日历
        
        Args:
            exchange: 期货交易所代码（CFFEX/SHFE/CZCE/DCE/INE）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 期货交易日历数据
        """
        if exchange not in self.FUTURES_EXCHANGES:
            logger.warning(f"不支持的期货交易所: {exchange}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 使用通用交易日历采集器
        collector = TradeCalendarCollector()
        return collector.collect(exchange=exchange, start_date=start_date, end_date=end_date)


# ============= 便捷函数接口 =============

def get_trade_calendar(
    exchange: str = 'SSE',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    is_open: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取交易日历数据
    
    Args:
        exchange: 交易所代码（SSE/SZSE/BSE/CFFEX/SHFE/CZCE/DCE/INE）
        start_date: 开始日期（YYYYMMDD格式）
        end_date: 结束日期（YYYYMMDD格式）
        is_open: 筛选交易日（1=交易日，0=非交易日）
    
    Returns:
        DataFrame: 交易日历数据
    
    Example:
        >>> df = get_trade_calendar(exchange='SSE', start_date='20240101', end_date='20241231')
        >>> df = get_trade_calendar(is_open=1)  # 只获取交易日
    """
    collector = TradeCalendarCollector()
    return collector.collect(exchange=exchange, start_date=start_date, 
                            end_date=end_date, is_open=is_open, **kwargs)


def get_suspend_info(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    suspend_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取停复牌信息
    
    Args:
        ts_code: 证券代码
        trade_date: 交易日期
        suspend_type: 停复牌类型（S=停牌，R=复牌）
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 停复牌数据
    
    Example:
        >>> df = get_suspend_info(trade_date='20240115')
        >>> df = get_suspend_info(suspend_type='S')  # 只查停牌
    """
    collector = SuspendInfoCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            suspend_type=suspend_type, start_date=start_date,
                            end_date=end_date, **kwargs)


def get_price_limit_rule(
    market: Optional[str] = None,
    exchange: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取涨跌停规则配置
    
    Args:
        market: 市场/板块（主板/创业板/科创板/北交所）
        exchange: 交易所代码
    
    Returns:
        DataFrame: 涨跌停规则配置
    
    Example:
        >>> df = get_price_limit_rule(market='创业板')
    """
    collector = PriceLimitRuleCollector()
    return collector.collect(market=market, exchange=exchange, **kwargs)


def get_auction_time(
    market: Optional[str] = None,
    exchange: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取集合竞价时间配置
    
    Args:
        market: 市场（A股/港股/美股）
        exchange: 交易所代码
    
    Returns:
        DataFrame: 集合竞价时间配置
    
    Example:
        >>> df = get_auction_time(market='A股')
    """
    collector = AuctionTimeCollector()
    return collector.collect(market=market, exchange=exchange, **kwargs)


def get_trade_dates(
    exchange: str = 'SSE',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[str]:
    """
    获取交易日列表（便捷函数）
    
    Args:
        exchange: 交易所代码
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        List[str]: 交易日日期列表
    
    Example:
        >>> dates = get_trade_dates(start_date='20240101', end_date='20240131')
    """
    df = get_trade_calendar(exchange=exchange, start_date=start_date, 
                           end_date=end_date, is_open=1)
    if df.empty:
        return []
    return df['cal_date'].tolist()


def is_trade_date(date: str, exchange: str = 'SSE') -> bool:
    """
    判断是否为交易日
    
    Args:
        date: 日期（YYYYMMDD或YYYY-MM-DD格式）
        exchange: 交易所代码
    
    Returns:
        bool: 是否为交易日
    """
    # 标准化日期格式
    date_clean = date.replace('-', '')
    
    df = get_trade_calendar(exchange=exchange, start_date=date_clean, end_date=date_clean)
    if df.empty:
        return False
    
    return df.iloc[0]['is_open'] == 1
