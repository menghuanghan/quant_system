"""
证券与标的基础信息（Security Master）采集模块

数据类型包括：
- 股票列表（A股/B股/科创板/北交所/港股/美股）
- 证券代码、交易所、上市状态
- 股票曾用名、证券代码变更
- ST/*ST/风险警示标识
- A+H/CDR/两网及退市股票
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


@CollectorRegistry.register("stock_list_a")
class StockListACollector(BaseCollector):
    """
    A股股票列表采集器
    
    采集范围：主板、中小板、创业板、科创板、北交所
    主数据源：Tushare (stock_basic)
    备用数据源：AkShare (stock_info_a_code_name), BaoStock (query_stock_basic)
    """
    
    # Tushare市场类型映射
    TUSHARE_MARKET_MAP = {
        '主板': 'main',
        '中小板': 'sme',
        '创业板': 'gem',
        '科创板': 'star',
        '北交所': 'bse',
    }
    
    # 标准输出字段
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码（含交易所后缀，如 000001.SZ）
        'symbol',           # 证券代码（纯数字）
        'name',             # 证券简称
        'area',             # 地区
        'industry',         # 所属行业
        'fullname',         # 公司全称
        'enname',           # 英文名称
        'cnspell',          # 拼音缩写
        'market',           # 市场类型（main/sme/gem/star/bse）
        'exchange',         # 交易所代码（SSE/SZSE/BSE）
        'curr_type',        # 交易货币（CNY）
        'list_status',      # 上市状态（L上市/D退市/P暂停上市）
        'list_date',        # 上市日期
        'delist_date',      # 退市日期
        'is_hs',            # 是否沪深港通标的（N/H/S）
        'act_name',         # 实际控制人
        'act_ent_type',     # 实控人企业性质
    ]
    
    def collect(
        self,
        market: Optional[str] = None,
        exchange: Optional[str] = None,
        list_status: str = 'L',
        is_hs: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集A股股票列表
        
        Args:
            market: 市场类型，可选值：main(主板)/sme(中小板)/gem(创业板)/star(科创板)/bse(北交所)
            exchange: 交易所代码，可选值：SSE(上交所)/SZSE(深交所)/BSE(北交所)
            list_status: 上市状态，可选值：L(上市)/D(退市)/P(暂停上市)，默认L
            is_hs: 是否沪深港通标的，可选值：N(否)/H(沪股通)/S(深股通)
        
        Returns:
            DataFrame: 标准化的股票列表数据
            
        输出字段:
            - ts_code: 证券代码（含交易所后缀）
            - symbol: 证券代码（纯数字）
            - name: 证券简称
            - area: 地区
            - industry: 所属行业
            - fullname: 公司全称
            - enname: 英文名称
            - cnspell: 拼音缩写
            - market: 市场类型
            - exchange: 交易所代码
            - curr_type: 交易货币
            - list_status: 上市状态
            - list_date: 上市日期
            - delist_date: 退市日期
            - is_hs: 是否沪深港通标的
            - act_name: 实际控制人
            - act_ent_type: 实控人企业性质
        """
        try:
            # 优先从 kwargs 提取日期，若无则使用显式参数
            s_date = start_date or kwargs.get('start_date')
            e_date = end_date or kwargs.get('end_date')
            
            df = self._collect_from_tushare(market, exchange, list_status, is_hs, s_date, e_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条A股股票数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取A股列表失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(list_status, **kwargs)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条A股股票数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取A股列表失败: {e}")
        
        # 降级到BaoStock
        try:
            df = self._collect_from_baostock(**kwargs)
            if not df.empty:
                logger.info(f"从BaoStock成功获取 {len(df)} 条A股股票数据")
                return df
        except Exception as e:
            logger.error(f"BaoStock获取A股列表失败: {e}")
        
        logger.error("所有数据源均无法获取A股列表数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        market: Optional[str],
        exchange: Optional[str],
        list_status: str,
        is_hs: Optional[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """从Tushare获取A股列表"""
        pro = self.tushare_api
        
        # 构建查询参数 - 不在这里筛选market，而是获取全部后再筛选
        params = {
            'list_status': list_status,
        }
        if exchange:
            params['exchange'] = exchange
        if is_hs:
            params['is_hs'] = is_hs
        
        # 请求所有字段
        fields = ','.join([
            'ts_code', 'symbol', 'name', 'area', 'industry',
            'fullname', 'enname', 'cnspell', 'market', 'exchange',
            'curr_type', 'list_status', 'list_date', 'delist_date',
            'is_hs', 'act_name', 'act_ent_type'
        ])
        
        df = pro.stock_basic(**params, fields=fields)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 筛选市场类型（使用映射）
        if market and not df.empty:
            # 反向映射：从英文代码到中文名称
            market_name_map = {v: k for k, v in self.TUSHARE_MARKET_MAP.items()}
            if market in market_name_map:
                # 如果传入的是英文代码，转换为中文
                df = df[df['market'] == market_name_map[market]]
            else:
                # 如果传入的是中文名称，直接筛选
                df = df[df['market'] == market]
        
        # 标准化日期格式
        df = self._convert_date_format(df, ['list_date', 'delist_date'])
        
        # 动态过滤：根据传入的 end_date 过滤上市日期
        if not df.empty and end_date and 'list_date' in df.columns:
            df = df[df['list_date'] <= str(end_date)]
        
        # 动态过滤：根据传入的 start_date 过滤
        if not df.empty and start_date and 'list_date' in df.columns:
             # 注意：对于基础列表，通常我们关心的是 "截至某日的存量"
             # 如果需要严格过滤 start_date 之后的上市，可以加
             pass
        
        # 确保包含所有字段（但不填充None，保持原始数据）
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, list_status: str = 'L', **kwargs) -> pd.DataFrame:
        """从AkShare获取A股列表"""
        import akshare as ak
        
        # 获取A股实时行情（包含基础信息）
        try:
            df = ak.stock_info_a_code_name()
        except Exception as e:
            logger.warning(f"AkShare获取A股列表失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # AkShare字段标准化映射
        column_mapping = {
            'code': 'symbol',
            'name': 'name',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 生成ts_code
        if 'symbol' in df.columns:
            df['ts_code'] = df['symbol'].apply(self._symbol_to_tscode)
        
        # 从ts_code提取交易所
        if 'ts_code' in df.columns:
            df['exchange'] = df['ts_code'].str.split('.').str[1]
        
        # 设置默认值
        df['list_status'] = 'L'  # AkShare只返回上市股票
        df['curr_type'] = 'CNY'
        
        # 注意：AkShare数据字段有限，以下字段将为空
        # area, industry, fullname, enname, cnspell, market, list_date, delist_date, is_hs, act_name, act_ent_type
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        logger.warning("使用AkShare数据源，部分字段（如area、industry等）将为空")
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_baostock(self, **kwargs) -> pd.DataFrame:
        """从BaoStock获取A股列表"""
        import baostock as bs
        
        # 确保登录
        if not self.source_manager.ensure_baostock_login():
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 获取基准日期
        target_date = kwargs.get('end_date') or datetime.now().strftime('%Y-%m-%d')
        if len(target_date.replace('-', '')) == 8:
            target_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}"
            
        # BaoStock需要按日期查询
        rs = bs.query_all_stock(day=target_date)
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # BaoStock字段标准化映射
        column_mapping = {
            'code': 'ts_code',
            'code_name': 'name',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 转换代码格式：bs格式(sh.600000) -> ts格式(600000.SH)
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].apply(self._bscode_to_tscode)
            df['symbol'] = df['ts_code'].str.split('.').str[0]
            df['exchange'] = df['ts_code'].str.split('.').str[1]
        
        # 筛选A股（排除指数等）
        df = df[df['symbol'].str.match(r'^[0368]\d{5}$|^4[38]\d{4}$')]
        
        # 设置默认值
        df['list_status'] = 'L'
        df['curr_type'] = 'CNY'
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        logger.warning("使用BaoStock数据源，部分字段（如area、industry等）将为空")
        
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
    
    @staticmethod
    def _bscode_to_tscode(bs_code: str) -> str:
        """将BaoStock代码格式转换为Tushare格式"""
        # sh.600000 -> 600000.SH
        if '.' in bs_code:
            parts = bs_code.split('.')
            if len(parts) == 2:
                exchange = parts[0].upper()
                symbol = parts[1]
                return f"{symbol}.{exchange}"
        return bs_code


@CollectorRegistry.register("stock_list_hk")
class StockListHKCollector(BaseCollector):
    """
    港股股票列表采集器
    
    主数据源：Tushare (hk_basic)
    备用数据源：AkShare (stock_hk_spot)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码（如 00001.HK）
        'symbol',           # 证券代码（纯数字）
        'name',             # 证券简称（中文）
        'enname',           # 英文名称
        'fullname',         # 公司全称
        'market',           # 板块类型
        'exchange',         # 交易所代码（HKEX）
        'curr_type',        # 交易货币（HKD）
        'list_status',      # 上市状态
        'list_date',        # 上市日期
        'isin',             # ISIN代码
    ]
    
    def collect(
        self,
        list_status: str = 'L',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集港股股票列表
        
        Args:
            list_status: 上市状态，可选值：L(上市)/D(退市)
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的港股列表数据
        """
        # 优先从 kwargs 提取日期，若无则使用显式参数
        s_date = start_date or kwargs.get('start_date')
        e_date = end_date or kwargs.get('end_date')
        
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(list_status, s_date, e_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条港股数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取港股列表失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(**kwargs)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条港股数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取港股列表失败: {e}")
        
        logger.error("所有数据源均无法获取港股列表数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        list_status: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """从Tushare获取港股列表"""
        pro = self.tushare_api
        
        df = pro.hk_basic(list_status=list_status)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化日期格式
        df = self._convert_date_format(df, ['list_date', 'delist_date'])
        
        # 动态过滤：上市日期不得晚于 end_date
        if not df.empty and end_date and 'list_date' in df.columns:
            df = df[df['list_date'] <= str(end_date)]
        
        # 添加symbol字段
        if 'ts_code' in df.columns:
            df['symbol'] = df['ts_code'].str.split('.').str[0]
        
        # 设置交易所
        df['exchange'] = 'HKEX'
        
        # 标准化日期格式
        df = self._convert_date_format(df, ['list_date', 'delist_date'])
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, list_status: str = 'L', **kwargs) -> pd.DataFrame:
        """从AkShare获取港股列表"""
        import akshare as ak
        
        # 获取港股实时行情
        try:
            df = ak.stock_hk_spot()
        except Exception as e:
            logger.warning(f"AkShare获取港股列表失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # AkShare字段标准化映射
        column_mapping = {
            '代码': 'symbol',
            '中文名称': 'name',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 生成ts_code
        if 'symbol' in df.columns:
            df['ts_code'] = df['symbol'].apply(lambda x: f"{str(x).zfill(5)}.HK")
        
        # 设置默认值
        df['exchange'] = 'HKEX'
        df['curr_type'] = 'HKD'
        df['list_status'] = 'L'
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        logger.warning("使用AkShare数据源，部分字段（如fullname、list_date等）将为空")
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("stock_list_us")
class StockListUSCollector(BaseCollector):
    """
    美股股票列表采集器
    
    主数据源：Tushare (us_basic) - 需要更高积分
    备用数据源：AkShare (stock_us_spot_em)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码（如 AAPL）
        'symbol',           # 证券代码
        'name',             # 证券简称（中文）
        'exchange',         # 交易所代码
        'curr_type',        # 交易货币（USD）
        'list_status',      # 上市状态
        'market_cap',       # 市值
        'country',          # 国家/地区
    ]
    
    def collect(
        self,
        exchange: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集美股股票列表
        
        Args:
            exchange: 交易所代码，可选值：NYSE/NASDAQ/AMEX
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的美股列表数据
        """
        # 优先从 kwargs 提取日期，若无则使用显式参数
        e_date = end_date or kwargs.get('end_date')
        
        # 优先使用AkShare（Tushare美股接口需要更高积分）
        try:
            df = self._collect_from_akshare(exchange, end_date=e_date, **kwargs)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条美股数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取美股列表失败: {e}")
        
        logger.error("无法获取美股列表数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self, exchange: Optional[str] = None, end_date: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """从AkShare获取美股列表"""
        import akshare as ak
        
        # 获取美股实时行情
        df = ak.stock_us_spot_em()
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
        # 注意：美股行情接口通常不含上市日期，若有 end_date 需求，需结合基础信息库
        # 此处仅保留参数占位以维持架构统一
        
        # AkShare字段标准化映射
        column_mapping = {
            '代码': 'symbol',
            '名称': 'name',
            '最新价': 'price',
            '总市值': 'market_cap',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 生成ts_code（与symbol相同）
        if 'symbol' in df.columns:
            df['ts_code'] = df['symbol']
        
        # 从代码格式判断交易所（简化处理）
        df['exchange'] = 'US'  # 通用标识
        df['curr_type'] = 'USD'
        df['list_status'] = 'L'
        df['country'] = 'US'
        
        # 筛选交易所
        if exchange and 'exchange' in df.columns:
            df = df[df['exchange'] == exchange]
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        logger.warning("使用AkShare数据源，部分字段（如exchange、list_date等）将为空")
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("name_change")
class NameChangeCollector(BaseCollector):
    """
    股票曾用名采集器
    
    采集股票历史名称变更记录
    主数据源：Tushare (namechange)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'name',             # 证券名称
        'start_date',       # 开始日期
        'end_date',         # 结束日期
        'ann_date',         # 公告日期
        'change_reason',    # 变更原因
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股票曾用名数据
        
        Args:
            ts_code: 证券代码（可选，不传则获取全部）
            start_date: 开始日期（YYYYMMDD格式）
            end_date: 结束日期（YYYYMMDD格式）
        
        Returns:
            DataFrame: 标准化的股票曾用名数据
            
        输出字段:
            - ts_code: 证券代码
            - name: 证券名称
            - start_date: 开始日期
            - end_date: 结束日期
            - ann_date: 公告日期
            - change_reason: 变更原因
        """
        try:
            df = self._collect_from_tushare(ts_code, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条股票曾用名记录")
                return df
        except Exception as e:
            logger.error(f"获取股票曾用名数据失败: {e}")
        
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取股票曾用名"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        fields = ','.join(self.OUTPUT_FIELDS)
        df = pro.namechange(**params, fields=fields)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化日期格式
        df = self._convert_date_format(df, ['start_date', 'end_date', 'ann_date'])
        
        # 动态过滤：仅包含在 end_date 之前的记录
        if not df.empty and end_date and 'start_date' in df.columns:
            df = df[df['start_date'] <= end_date]
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("st_status")
class STStatusCollector(BaseCollector):
    """
    ST标识状态采集器
    
    通过股票曾用名记录识别ST/*ST/风险警示状态
    主数据源：基于namechange接口的衍生分析
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'name',             # 当前名称
        'is_st',            # 是否ST股
        'is_star_st',       # 是否*ST股
        'st_start_date',    # ST开始日期
        'st_end_date',      # ST结束日期（如已摘帽）
        'st_reason',        # ST原因
    ]
    
    # ST标识关键词
    ST_KEYWORDS = ['ST', '*ST', 'S*ST', 'SST', 'S']
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        include_history: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股票ST标识状态
        
        Args:
            ts_code: 证券代码（可选，不传则获取全部）
            include_history: 是否包含历史ST记录
        
        Returns:
            DataFrame: 标准化的ST状态数据
        """
        # 首先获取股票曾用名数据
        name_change_collector = NameChangeCollector()
        name_df = name_change_collector.collect(ts_code=ts_code, **kwargs)
        
        if name_df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 分析ST状态
        st_records = []
        
        for _, row in name_df.iterrows():
            name = str(row.get('name', ''))
            is_st = any(name.startswith(kw) for kw in ['ST', 'SST'])
            is_star_st = '*ST' in name or 'S*ST' in name
            if is_st or is_star_st:
                # 动态过滤：排除晚于传入结束日期的 ST 记录
                target_end = kwargs.get('end_date')
                if row.get('start_date') and target_end and str(row.get('start_date')) > str(target_end):
                    continue
                    
                st_records.append({
                    'ts_code': row.get('ts_code'),
                    'name': name,
                    'is_st': is_st,
                    'is_star_st': is_star_st,
                    'st_start_date': row.get('start_date'),
                    'st_end_date': row.get('end_date'),
                    'st_reason': row.get('change_reason'),
                })
        
        if not st_records:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = pd.DataFrame(st_records)
        
        # 如果只需要当前状态，筛选end_date为空的记录
        if not include_history:
            df = df[df['st_end_date'].isna() | (df['st_end_date'] == '')]
        
        logger.info(f"识别到 {len(df)} 条ST状态记录")
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("ah_stock")
class AHStockCollector(BaseCollector):
    """
    A+H股票列表采集器
    
    采集同时在A股和H股上市的股票
    主数据源：通过A股列表和港股列表交叉匹配
    """
    
    OUTPUT_FIELDS = [
        'a_ts_code',        # A股证券代码
        'a_name',           # A股名称
        'a_list_date',      # A股上市日期
        'is_hs',            # 沪深港通标识
    ]
    
    def collect(self, **kwargs) -> pd.DataFrame:
        """
        采集A+H股票列表
        
        Returns:
            DataFrame: A+H股票对照数据
        """
        # 获取A股列表（标记沪港通/深港通标的）
        a_collector = StockListACollector()
        a_df = a_collector.collect(is_hs='H', **kwargs)  # 沪股通标的
        
        a_df_s = a_collector.collect(is_hs='S', **kwargs)  # 深股通标的
        a_df = pd.concat([a_df, a_df_s], ignore_index=True).drop_duplicates(subset=['ts_code'])
        
        if a_df.empty:
            logger.warning("未获取到A股沪深港通标的数据")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # TODO: 实际A+H匹配需要额外的映射表
        # 这里先返回沪深港通标的作为候选
        result_df = pd.DataFrame({
            'a_ts_code': a_df['ts_code'],
            'a_name': a_df['name'],
            'a_list_date': a_df['list_date'],
            'is_hs': a_df['is_hs'],
        })
        
        logger.info(f"获取到 {len(result_df)} 只沪深港通标的股票")
        return result_df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_stock_list_a(
    market: Optional[str] = None,
    exchange: Optional[str] = None,
    list_status: str = 'L',
    is_hs: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取A股股票列表
    
    Args:
        market: 市场类型（main/sme/gem/star/bse）
        exchange: 交易所（SSE/SZSE/BSE）
        list_status: 上市状态（L/D/P）
        is_hs: 是否沪深港通标的（N/H/S）
    
    Returns:
        DataFrame: A股股票列表
    
    Example:
        >>> df = get_stock_list_a(market='main', list_status='L')
        >>> df = get_stock_list_a(exchange='SSE')
    """
    collector = StockListACollector()
    return collector.collect(market=market, exchange=exchange, 
                            list_status=list_status, is_hs=is_hs, **kwargs)


def get_stock_list_hk(list_status: str = 'L', **kwargs) -> pd.DataFrame:
    """
    获取港股股票列表
    
    Args:
        list_status: 上市状态（L/D）
    
    Returns:
        DataFrame: 港股股票列表
    """
    collector = StockListHKCollector()
    return collector.collect(list_status=list_status, **kwargs)


def get_stock_list_us(exchange: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    获取美股股票列表
    
    Args:
        exchange: 交易所（NYSE/NASDAQ/AMEX）
    
    Returns:
        DataFrame: 美股股票列表
    """
    collector = StockListUSCollector()
    return collector.collect(exchange=exchange, **kwargs)


def get_name_change(
    ts_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取股票曾用名记录
    
    Args:
        ts_code: 证券代码
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 股票曾用名记录
    
    Example:
        >>> df = get_name_change(ts_code='600848.SH')
    """
    collector = NameChangeCollector()
    return collector.collect(ts_code=ts_code, start_date=start_date, end_date=end_date, **kwargs)


def get_st_status(
    ts_code: Optional[str] = None,
    include_history: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    获取股票ST标识状态
    
    Args:
        ts_code: 证券代码
        include_history: 是否包含历史ST记录
    
    Returns:
        DataFrame: ST状态记录
    """
    collector = STStatusCollector()
    return collector.collect(ts_code=ts_code, include_history=include_history, **kwargs)


def get_ah_stock(**kwargs) -> pd.DataFrame:
    """
    获取A+H股票对照表
    
    Returns:
        DataFrame: A+H股票对照表
    """
    collector = AHStockCollector()
    return collector.collect(**kwargs)
