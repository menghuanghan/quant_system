"""
期货与期权（Futures & Options）采集模块

数据类型包括：
- 期货合约信息
- 期货日线行情（主力/连续）
- 仓单日报
- 每日持仓排名
- 期权合约信息
- 期权日线行情
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


@CollectorRegistry.register("fut_basic")
class FuturesBasicCollector(BaseCollector):
    """
    期货合约信息采集器
    
    获取期货合约列表数据
    主数据源：Tushare (fut_basic, 2000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 合约代码
        'symbol',           # 交易标识
        'exchange',         # 交易市场
        'name',             # 中文简称
        'fut_code',         # 合约产品代码
        'multiplier',       # 合约乘数
        'trade_unit',       # 交易计量单位
        'per_unit',         # 交易单位(每手)
        'quote_unit',       # 报价单位
        'quote_unit_desc',  # 最小报价单位说明
        'd_mode_desc',      # 交割方式说明
        'list_date',        # 上市日期
        'delist_date',      # 最后交易日期
        'd_month',          # 交割月份
        'last_ddate',       # 最后交割日
        # 'trade_time_desc',  # 交易时间说明（Tushare部分字段缺失）
    ]
    
    def collect(
        self,
        exchange: Optional[str] = None,
        fut_type: Optional[str] = None,
        fut_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集期货合约信息
        
        Args:
            exchange: 交易所代码（CFFEX/DCE/CZCE/SHFE/INE/GFEX）
            fut_type: 合约类型（1普通合约，2主力与连续合约）
            fut_code: 标准合约代码（如AG白银）
        
        Returns:
            DataFrame: 标准化的期货合约信息
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(exchange, fut_type, fut_code)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条期货合约数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取期货合约信息失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(exchange)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条期货合约数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取期货合约信息失败: {e}")
        
        logger.error("无法获取期货合约信息")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        exchange: Optional[str],
        fut_type: Optional[str],
        fut_code: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取期货合约信息"""
        params = {}
        if exchange:
            params['exchange'] = exchange
        if fut_type:
            params['fut_type'] = fut_type
        if fut_code:
            params['fut_code'] = fut_code
        
        df = self.tushare_api.fut_basic(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, exchange: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取期货合约信息"""
        import akshare as ak
        
        try:
            # 获取期货品种列表
            df = ak.futures_display_main_sina()
        except Exception as e:
            logger.warning(f"AkShare获取期货合约失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '品种': 'name',
            '代码': 'symbol',
        }
        df = self._standardize_columns(df, column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("fut_daily")
class FuturesDailyCollector(BaseCollector):
    """
    期货日线行情采集器
    
    获取期货日线行情数据
    主数据源：Tushare (fut_daily, 2000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # TS合约代码
        'trade_date',       # 交易日期
        'pre_close',        # 昨收盘价
        'pre_settle',       # 昨结算价
        'open',             # 开盘价
        'high',             # 最高价
        'low',              # 最低价
        'close',            # 收盘价
        'settle',           # 结算价
        'change1',          # 涨跌1（收盘-昨结算）
        'change2',          # 涨跌2（结算-昨结算）
        'vol',              # 成交量(手)
        'amount',           # 成交金额(万元)
        'oi',               # 持仓量(手)
        'oi_chg',           # 持仓量变化
        'delv_settle',      # 交割结算价
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        exchange: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集期货日线行情
        
        Args:
            ts_code: 合约代码
            trade_date: 交易日期（YYYYMMDD）
            exchange: 交易所代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的期货日线行情
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, exchange, start_date, end_date)
            if not df.empty:
                # Check if we should split
                if start_date and end_date and not ts_code:
                     self._split_and_save_fut_daily(df)
                     return pd.DataFrame(columns=self.OUTPUT_FIELDS)
                else:
                     logger.info(f"从Tushare成功获取 {len(df)} 条期货行情数据")
                     return df
        except Exception as e:
            logger.warning(f"Tushare获取期货行情失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code, start_date, end_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条期货行情数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取期货行情失败: {e}")
        
        logger.error("无法获取期货行情数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)

    def _split_and_save_fut_daily(self, df: pd.DataFrame):
        """将聚合的期货行情按 ts_code 拆分并保存"""
        from pathlib import Path
        if df.empty or 'ts_code' not in df.columns:
            return
            
        # Use a specific directory for split files
        output_base = Path("data/raw/structured/derivatives/fut_daily")
        output_base.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"正在拆分期货行情，涉及 {len(df['ts_code'].unique())} 个合约...")
        
        # Groupby is efficient enough for ~1M rows
        for code, group in df.groupby('ts_code'):
            # Sanitize filename
            safe_code = code.replace('.', '_')
            file_path = output_base / f"{safe_code}.parquet"
            # Append if exists? No, collection is usually range-based update or overwrite.
            # Assuming overwrite behavior for simplicity in this repair context.
            group.to_parquet(file_path, index=False, compression='snappy')
        
        logger.info("期货行情拆分保存完成")
    
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        exchange: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取期货行情"""
        params = {}
        if ts_code: params['ts_code'] = ts_code
        if trade_date: params['trade_date'] = trade_date
        if exchange: params['exchange'] = exchange
        
        # 分块采集以避免API限制
        if start_date and end_date and not ts_code:
            try:
                from datetime import timedelta
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y%m%d')
                
                all_dfs = []
                current_dt = start_dt
                while current_dt <= end_dt:
                    t_date = current_dt.strftime('%Y%m%d')
                    p = params.copy()
                    p['trade_date'] = t_date
                    
                    try:
                        chunk_df = self.tushare_api.fut_daily(**p)
                        if not chunk_df.empty:
                            all_dfs.append(chunk_df)
                            if len(all_dfs) % 10 == 0:
                                logger.info(f"期货日线进度: {t_date}")
                    except Exception as e:
                        if '抱歉' not in str(e): # 忽略权限错误
                            logger.warning(f"期货日线采集失败 ({t_date}): {e}")
                    
                    current_dt = current_dt + timedelta(days=1)
                
                if all_dfs:
                    return pd.concat(all_dfs, ignore_index=True)
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            except Exception as e:
                logger.error(f"期货日线分块逻辑异常: {e}")
        
        if start_date: params['start_date'] = start_date
        if end_date: params['end_date'] = end_date
        df = self.tushare_api.fut_daily(**params)
        
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
        """从AkShare获取期货行情"""
        import akshare as ak
        
        if not ts_code:
            logger.warning("AkShare需要指定ts_code获取期货行情")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 提取合约代码
        symbol = ts_code.split('.')[0].lower()
        
        try:
            df = ak.futures_zh_daily_sina(symbol=symbol)
        except Exception as e:
            logger.warning(f"AkShare获取期货行情失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            'date': 'trade_date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'vol',
            'hold': 'oi',
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


@CollectorRegistry.register("fut_holding")
class FuturesHoldingCollector(BaseCollector):
    """
    期货每日持仓排名采集器
    
    获取每日成交持仓排名数据
    主数据源：Tushare (fut_holding, 2000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',       # 交易日期
        'symbol',           # 合约代码或类型
        'broker',           # 期货公司会员简称
        'vol',              # 成交量
        'vol_chg',          # 成交量变化
        'long_hld',         # 持买仓量
        'long_chg',         # 持买仓量变化
        'short_hld',        # 持卖仓量
        'short_chg',        # 持卖仓量变化
        # 'exchange',       # 交易所（数据源未提供）
    ]
    
    def collect(
        self,
        trade_date: Optional[str] = None,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集期货持仓排名
        
        Args:
            trade_date: 交易日期
            symbol: 合约或产品代码
            exchange: 交易所代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的持仓排名数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(trade_date, symbol, exchange, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条持仓排名数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取持仓排名失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(trade_date, symbol, exchange)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条持仓排名数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取持仓排名失败: {e}")
        
        logger.error("无法获取持仓排名数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        trade_date: Optional[str],
        symbol: Optional[str],
        exchange: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取持仓排名"""
        params = {}
        if trade_date:
            params['trade_date'] = trade_date
        if symbol:
            params['symbol'] = symbol
        if exchange:
            params['exchange'] = exchange
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = self.tushare_api.fut_holding(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(
        self,
        trade_date: Optional[str],
        symbol: Optional[str],
        exchange: Optional[str]
    ) -> pd.DataFrame:
        """从AkShare获取持仓排名"""
        import akshare as ak
        
        if not trade_date or not symbol:
            logger.warning("AkShare需要指定trade_date和symbol获取持仓排名")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 格式转换
        date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
        
        try:
            df = ak.get_dce_rank_table(date=date_str, vars_list=[symbol])
        except Exception as e:
            logger.warning(f"AkShare获取持仓排名失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df['trade_date'] = trade_date
        df['symbol'] = symbol
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("fut_wsr")
class FuturesWarehouseCollector(BaseCollector):
    """
    仓单日报采集器
    
    获取仓单日报数据
    主数据源：Tushare (fut_wsr, 2000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',       # 交易日期
        'symbol',           # 产品代码
        'fut_name',         # 产品名称
        'warehouse',        # 仓库名称
        'wh_id',            # 仓库编号
        'pre_vol',          # 昨日仓单量
        'vol',              # 今日仓单量
        'vol_chg',          # 增减量
        'area',             # 地区
        'year',             # 年度
        'grade',            # 等级
        'brand',            # 品牌
        'place',            # 产地
        'pd',               # 升贴水
        'is_ct',            # 是否折算仓单
        'unit',             # 单位
        'exchange',         # 交易所
    ]
    
    def collect(
        self,
        trade_date: Optional[str] = None,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集仓单日报
        
        Args:
            trade_date: 交易日期
            symbol: 产品代码
            exchange: 交易所代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的仓单日报数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(trade_date, symbol, exchange, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条仓单日报数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取仓单日报失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(trade_date, exchange)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条仓单日报数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取仓单日报失败: {e}")
        
        logger.error("无法获取仓单日报数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        trade_date: Optional[str],
        symbol: Optional[str],
        exchange: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取仓单日报"""
        params = {}
        if trade_date:
            params['trade_date'] = trade_date
        if symbol:
            params['symbol'] = symbol
        if exchange:
            params['exchange'] = exchange
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = self.tushare_api.fut_wsr(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(
        self,
        trade_date: Optional[str],
        exchange: Optional[str]
    ) -> pd.DataFrame:
        """从AkShare获取仓单日报"""
        import akshare as ak
        
        try:
            # 获取上期所仓单数据
            df = ak.futures_shfe_warehouse_receipt()
        except Exception as e:
            logger.warning(f"AkShare获取仓单日报失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '品种': 'fut_name',
            '仓库': 'warehouse',
            '今日仓单量': 'vol',
            '增减': 'vol_chg',
        }
        df = self._standardize_columns(df, column_mapping)
        
        if trade_date:
            df['trade_date'] = trade_date
        else:
            df['trade_date'] = datetime.now().strftime('%Y%m%d')
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("opt_basic")
class OptionsBasicCollector(BaseCollector):
    """
    期权合约信息采集器
    
    获取期权合约信息
    主数据源：Tushare (opt_basic, 5000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # TS代码
        'exchange',         # 交易市场
        'name',             # 合约名称
        'per_unit',         # 合约单位
        'opt_code',         # 标的合约代码
        'opt_type',         # 合约类型
        'call_put',         # 期权类型
        'exercise_type',    # 行权方式
        'exercise_price',   # 行权价格
        's_month',          # 结算月
        'maturity_date',    # 到期日
        'list_price',       # 挂牌基准价
        'list_date',        # 开始交易日期
        'delist_date',      # 最后交易日期
        'last_edate',       # 最后行权日期
        'last_ddate',       # 最后交割日期
        'quote_unit',       # 报价单位
        'min_price_chg',    # 最小价格波幅
    ]
    
    EXCHANGES = ['SSE', 'SZSE', 'CFFEX', 'DCE', 'SHFE', 'CZCE', 'GFEX']

    def collect(
        self,
        ts_code: Optional[str] = None,
        exchange: Optional[str] = None,
        opt_code: Optional[str] = None,
        call_put: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集期权合约信息
        
        Args:
            ts_code: TS期权代码
            exchange: 交易所代码
            opt_code: 标准合约代码
            call_put: 期权类型
        
        Returns:
            DataFrame: 标准化的期权合约信息
        """
        # 如果未指定交易所，则遍历所有交易所采集
        if not exchange and not ts_code:
            all_dfs = []
            for exc in self.EXCHANGES:
                try:
                    df = self._collect_from_tushare(ts_code, exc, opt_code, call_put)
                    if not df.empty:
                        all_dfs.append(df)
                except Exception as e:
                    logger.warning(f"采集 {exc} 期权合约失败: {e}")
            
            if all_dfs:
                return pd.concat(all_dfs, ignore_index=True)
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)

        # 优先使用Tushare（需要5000积分）
        try:
            df = self._collect_from_tushare(ts_code, exchange, opt_code, call_put)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条期权合约数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取期权合约失败（可能积分不足，需5000积分）: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(exchange)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条期权合约数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取期权合约失败: {e}")
        
        logger.error("无法获取期权合约数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        exchange: Optional[str],
        opt_code: Optional[str],
        call_put: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取期权合约信息"""
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if exchange:
            params['exchange'] = exchange
        if opt_code:
            params['opt_code'] = opt_code
        if call_put:
            params['call_put'] = call_put
        
        df = self.tushare_api.opt_basic(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, exchange: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取期权合约信息"""
        import akshare as ak
        
        try:
            if exchange == 'SSE':
                df = ak.option_sse_list_sina(symbol="510050")
            else:
                # 默认获取50ETF期权
                df = ak.option_sse_list_sina(symbol="510050")
        except Exception as e:
            logger.warning(f"AkShare获取期权合约失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '期权代码': 'ts_code',
            '期权名称': 'name',
            '行权价': 'exercise_price',
            '到期日': 'maturity_date',
        }
        df = self._standardize_columns(df, column_mapping)
        df['exchange'] = exchange or 'SSE'
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("opt_daily")
class OptionsDailyCollector(BaseCollector):
    """
    期权日线行情采集器
    
    获取期权日线行情数据
    主数据源：Tushare (opt_daily, 2000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # TS代码
        'trade_date',       # 交易日期
        'exchange',         # 交易市场
        'pre_settle',       # 昨结算价
        'pre_close',        # 前收盘价
        'open',             # 开盘价
        'high',             # 最高价
        'low',              # 最低价
        'close',            # 收盘价
        'settle',           # 结算价
        'vol',              # 成交量(手)
        'amount',           # 成交金额(万元)
        'oi',               # 持仓量(手)
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        exchange: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集期权日线行情
        
        Args:
            ts_code: TS合约代码
            trade_date: 交易日期
            exchange: 交易所
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的期权日线行情
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, exchange, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条期权行情数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取期权行情失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code, trade_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条期权行情数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取期权行情失败: {e}")
        
        logger.error("无法获取期权行情数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        exchange: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取期权行情"""
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
        if exchange:
            params['exchange'] = exchange
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = self.tushare_api.opt_daily(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str]
    ) -> pd.DataFrame:
        """从AkShare获取期权行情"""
        import akshare as ak
        
        try:
            # 获取50ETF期权当前行情
            df = ak.option_sse_spot_price_sina(symbol="510050")
        except Exception as e:
            logger.warning(f"AkShare获取期权行情失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '期权代码': 'ts_code',
            '最新价': 'close',
            '成交量': 'vol',
            '成交额': 'amount',
            '持仓量': 'oi',
        }
        df = self._standardize_columns(df, column_mapping)
        df['trade_date'] = trade_date or datetime.now().strftime('%Y%m%d')
        df['exchange'] = 'SSE'
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_fut_basic(
    exchange: Optional[str] = None,
    fut_type: Optional[str] = None,
    fut_code: Optional[str] = None
) -> pd.DataFrame:
    """
    获取期货合约信息
    
    Args:
        exchange: 交易所代码（CFFEX/DCE/CZCE/SHFE/INE/GFEX）
        fut_type: 合约类型（1普通合约，2主力与连续合约）
        fut_code: 标准合约代码
    
    Returns:
        DataFrame: 期货合约信息
    
    Example:
        >>> df = get_fut_basic(exchange='DCE')
        >>> df = get_fut_basic(fut_type='1')  # 普通合约
    """
    collector = FuturesBasicCollector()
    return collector.collect(exchange=exchange, fut_type=fut_type, fut_code=fut_code)


def get_fut_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    exchange: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取期货日线行情
    
    Args:
        ts_code: 合约代码
        trade_date: 交易日期
        exchange: 交易所代码
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 期货日线行情
    
    Example:
        >>> df = get_fut_daily(ts_code='CU2401.SHF', start_date='20240101')
        >>> df = get_fut_daily(trade_date='20250115', exchange='DCE')
    """
    collector = FuturesDailyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date, exchange=exchange,
                            start_date=start_date, end_date=end_date)


def get_fut_holding(
    trade_date: Optional[str] = None,
    symbol: Optional[str] = None,
    exchange: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取期货每日持仓排名
    
    Args:
        trade_date: 交易日期
        symbol: 合约或产品代码
        exchange: 交易所代码
        **kwargs: 其他参数（由调度器传入）
    
    Returns:
        DataFrame: 持仓排名数据
    
    Example:
        >>> df = get_fut_holding(trade_date='20250115', symbol='C', exchange='DCE')
    """
    collector = FuturesHoldingCollector()
    return collector.collect(trade_date=trade_date, symbol=symbol, exchange=exchange, **kwargs)


def get_fut_wsr(
    trade_date: Optional[str] = None,
    symbol: Optional[str] = None,
    exchange: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取仓单日报
    
    Args:
        trade_date: 交易日期
        symbol: 产品代码
        exchange: 交易所代码
        **kwargs: 其他参数（由调度器传入）
    
    Returns:
        DataFrame: 仓单日报数据
    
    Example:
        >>> df = get_fut_wsr(trade_date='20250115', symbol='ZN')
    """
    collector = FuturesWarehouseCollector()
    return collector.collect(trade_date=trade_date, symbol=symbol, exchange=exchange, **kwargs)


def get_opt_basic(
    ts_code: Optional[str] = None,
    exchange: Optional[str] = None,
    opt_code: Optional[str] = None,
    call_put: Optional[str] = None
) -> pd.DataFrame:
    """
    获取期权合约信息
    
    Args:
        ts_code: TS期权代码
        exchange: 交易所代码
        opt_code: 标准合约代码
        call_put: 期权类型（C/P）
    
    Returns:
        DataFrame: 期权合约信息
    
    Example:
        >>> df = get_opt_basic(exchange='DCE')
        >>> df = get_opt_basic(opt_code='OPM2407.DCE')
    """
    collector = OptionsBasicCollector()
    return collector.collect(ts_code=ts_code, exchange=exchange,
                            opt_code=opt_code, call_put=call_put)


def get_opt_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    exchange: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取期权日线行情
    
    Args:
        ts_code: TS合约代码
        trade_date: 交易日期
        exchange: 交易所
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 期权日线行情
    
    Example:
        >>> df = get_opt_daily(trade_date='20250115')
        >>> df = get_opt_daily(ts_code='10001313.SH', start_date='20250101')
    """
    collector = OptionsDailyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date, exchange=exchange,
                            start_date=start_date, end_date=end_date)
