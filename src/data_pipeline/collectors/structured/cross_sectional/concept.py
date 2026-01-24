"""
概念与主题板块（Concept & Theme）采集模块

数据类型包括：
- 同花顺概念板块列表
- 同花顺概念板块成分
- 同花顺概念指数行情
- 东方财富概念板块列表
- 东方财富概念板块成分
- 热点题材数据

注意：部分Tushare接口需要6000积分，2120积分用户将自动降级到AkShare
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


@CollectorRegistry.register("ths_index")
class THSIndexCollector(BaseCollector):
    """
    同花顺概念/行业板块列表采集器
    
    采集同花顺板块指数列表
    主数据源：Tushare (ths_index) - 需6000积分
    备用数据源：AkShare
    
    注意：Tushare需要6000积分，2120积分不足时自动降级
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 板块指数代码
        'name',                 # 板块名称
        'exchange',             # 交易所（A-A股）
        'type',                 # 类型（N-概念指数,I-行业指数等）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        exchange: str = 'A',
        type: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集同花顺概念/行业板块列表
        
        Args:
            ts_code: 指数代码
            exchange: 市场类型（A-A股, HK-港股, US-美股）
            type: 指数类型（N-概念指数, I-行业指数, R-地域指数, 
                          S-同花顺特色指数, ST-同花顺风格指数, 
                          TH-同花顺主题指数, BB-同花顺宽基指数）
        
        Returns:
            DataFrame: 标准化的同花顺板块列表数据
        """
        # 优先使用Tushare（需6000积分，可能失败）
        try:
            df = self._collect_from_tushare(ts_code, exchange, type)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条同花顺板块数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取同花顺板块失败（可能积分不足，需6000积分）: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(type)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条同花顺板块数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取同花顺板块失败: {e}")
        
        logger.error("所有数据源均无法获取同花顺板块数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        exchange: str,
        type: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取同花顺板块列表"""
        pro = self.tushare_api
        
        params = {'exchange': exchange}
        if ts_code:
            params['ts_code'] = ts_code
        if type:
            params['type'] = type
        
        df = pro.ths_index(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, type: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取同花顺板块列表"""
        import akshare as ak
        
        try:
            # AkShare获取同花顺概念板块
            if type == 'N' or type is None:
                df = ak.stock_board_concept_name_ths()
            else:
                # 行业板块
                df = ak.stock_board_industry_name_ths()
        except Exception as e:
            logger.warning(f"AkShare获取同花顺板块失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '代码': 'ts_code',
            'code': 'ts_code',
            '板块名称': 'name',
            '概念名称': 'name',
            'name': 'name',
            '成分股数量': 'count',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置默认值
        df['exchange'] = 'A'
        df['type'] = type if type else 'N'
        df['list_date'] = None
        
        # 转换代码格式为Tushare格式
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].astype(str) + '.TI'
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("ths_member")
class THSMemberCollector(BaseCollector):
    """
    同花顺概念板块成分采集器
    
    采集同花顺概念板块成分股列表
    主数据源：Tushare (ths_member) - 需6000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 板块指数代码
        'con_code',             # 成分股票代码
        'con_name',             # 成分股票名称
        'weight',               # 权重（暂无）
        'in_date',              # 纳入日期（暂无）
        'out_date',             # 剔除日期（暂无）
        'is_new',               # 是否最新（Y/N）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        con_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集同花顺概念板块成分
        
        Args:
            ts_code: 板块指数代码
            con_code: 成分股票代码
        
        Returns:
            DataFrame: 标准化的同花顺板块成分数据
        """
        # 优先使用Tushare（需6000积分，可能失败）
        try:
            df = self._collect_from_tushare(ts_code, con_code)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条同花顺板块成分数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取同花顺板块成分失败（可能积分不足）: {e}")
        
        # 降级到AkShare
        try:
            if ts_code:
                df = self._collect_from_akshare(ts_code)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条同花顺板块成分数据")
                    return df
        except Exception as e:
            logger.error(f"AkShare获取同花顺板块成分失败: {e}")
        
        logger.error("所有数据源均无法获取同花顺板块成分数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        con_code: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取同花顺板块成分"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if con_code:
            params['con_code'] = con_code
        
        df = pro.ths_member(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, ts_code: str) -> pd.DataFrame:
        """从AkShare获取同花顺板块成分"""
        import akshare as ak
        
        try:
            # 从ts_code提取板块名称
            # 首先获取板块列表来匹配
            concept_df = ak.stock_board_concept_name_ths()
            
            # 提取代码
            code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            
            # 查找对应的板块名称
            matched = concept_df[concept_df['代码'].astype(str) == code]
            if matched.empty:
                logger.warning(f"未找到对应的同花顺板块: {ts_code}")
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            board_name = matched.iloc[0]['概念名称']
            df = ak.stock_board_concept_cons_ths(symbol=board_name)
        except Exception as e:
            logger.warning(f"AkShare获取同花顺板块成分失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '代码': 'con_code',
            '名称': 'con_name',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置板块代码
        df['ts_code'] = ts_code
        df['is_new'] = 'Y'
        
        # 格式化股票代码
        if 'con_code' in df.columns:
            def format_code(code):
                code = str(code)
                if code.startswith('6'):
                    return code + '.SH'
                elif code.startswith(('0', '3')):
                    return code + '.SZ'
                elif code.startswith(('4', '8')):
                    return code + '.BJ'
                return code
            df['con_code'] = df['con_code'].apply(format_code)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("ths_daily")
class THSDailyCollector(BaseCollector):
    """
    同花顺板块指数行情采集器
    
    采集同花顺板块指数日线行情
    主数据源：Tushare (ths_daily) - 需6000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 指数代码
        'trade_date',           # 交易日期
        'close',                # 收盘点位
        'open',                 # 开盘点位
        'high',                 # 最高点位
        'low',                  # 最低点位
        'pre_close',            # 昨日收盘点
        'avg_price',            # 平均价
        'change',               # 涨跌点位
        'pct_change',           # 涨跌幅（%）
        'vol',                  # 成交量
        'turnover_rate',        # 换手率
        'total_mv',             # 总市值
        'float_mv',             # 流通市值
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
        采集同花顺板块指数行情
        
        Args:
            ts_code: 指数代码
            trade_date: 交易日期（YYYYMMDD）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的同花顺板块指数行情数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条同花顺板块行情数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取同花顺板块行情失败: {e}")
        
        # 降级到AkShare
        try:
            if ts_code:
                df = self._collect_from_akshare(ts_code, start_date, end_date)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条同花顺板块行情数据")
                    return df
        except Exception as e:
            logger.error(f"AkShare获取同花顺板块行情失败: {e}")
        
        logger.error("所有数据源均无法获取同花顺板块行情数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取同花顺板块行情"""
        pro = self.tushare_api
        import time
        from datetime import datetime, timedelta
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
            return pro.ths_daily(**params)
            
        # 范围查询优化
        if start_date and end_date and not ts_code:
            try:
                cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
                dates = cal['cal_date'].tolist()
            except Exception:
                dates = [d.strftime('%Y%m%d') for d in pd.date_range(start_date, end_date)]
                
            all_dfs = []
            for date in dates:
                try:
                    # 同花顺板块接口限流极严 (10次/分钟)
                    time.sleep(6.2) 
                    df = pro.ths_daily(trade_date=date)
                    if not df.empty:
                        all_dfs.append(df)
                except Exception as e:
                    if "访问该接口" in str(e):
                        time.sleep(30)
                    else:
                        logger.warning(f"同花顺行情 {date} 采集失败: {e}")
            
            if all_dfs:
                df = pd.concat(all_dfs, ignore_index=True)
            else:
                df = pd.DataFrame()
        else:
            if start_date: params['start_date'] = start_date
            if end_date: params['end_date'] = end_date
            df = pro.ths_daily(**params)
            
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
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
        """从AkShare获取同花顺板块行情"""
        import akshare as ak
        
        try:
            # 从ts_code提取板块名称
            concept_df = ak.stock_board_concept_name_ths()
            code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            
            matched = concept_df[concept_df['代码'].astype(str) == code]
            if matched.empty:
                logger.warning(f"未找到对应的同花顺板块: {ts_code}")
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            board_name = matched.iloc[0]['概念名称']
            df = ak.stock_board_concept_hist_ths(symbol=board_name)
        except Exception as e:
            logger.warning(f"AkShare获取同花顺板块行情失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '日期': 'trade_date',
            '收盘价': 'close',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '成交量': 'vol',
            '涨跌幅': 'pct_change',
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


@CollectorRegistry.register("dc_index")
class DCIndexCollector(BaseCollector):
    """
    东方财富概念板块采集器
    
    采集东方财富概念板块每日数据
    主数据源：Tushare (dc_index) - 需6000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 概念代码
        'trade_date',           # 交易日期
        'name',                 # 概念名称
        'leading',              # 领涨股票名称
        'leading_code',         # 领涨股票代码
        'pct_change',           # 涨跌幅（%）
        'leading_pct',          # 领涨股票涨跌幅
        'total_mv',             # 总市值（万元）
        'turnover_rate',        # 换手率
        'up_num',               # 上涨家数
        'down_num',             # 下降家数
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        name: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集东方财富概念板块数据
        
        Args:
            ts_code: 概念代码
            name: 板块名称
            trade_date: 交易日期（YYYYMMDD）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的东方财富概念板块数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, name, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条东方财富概念板块数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取东方财富概念板块失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(name)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条东方财富概念板块数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取东方财富概念板块失败: {e}")
        
        logger.error("所有数据源均无法获取东方财富概念板块数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        name: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取东方财富概念板块"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if name:
            params['name'] = name
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = pro.dc_index(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, name: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取东方财富概念板块"""
        import akshare as ak
        
        try:
            # 获取东方财富概念板块列表
            df = ak.stock_board_concept_name_em()
        except Exception as e:
            logger.warning(f"AkShare获取东方财富概念板块失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '板块代码': 'ts_code',
            '板块名称': 'name',
            '涨跌幅': 'pct_change',
            '总市值': 'total_mv',
            '换手率': 'turnover_rate',
            '上涨家数': 'up_num',
            '下跌家数': 'down_num',
            '领涨股票': 'leading',
            '领涨股票-涨跌幅': 'leading_pct',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置当前日期
        df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # 如果指定了名称，过滤
        if name and 'name' in df.columns:
            df = df[df['name'].str.contains(name, na=False)]
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("dc_member")
class DCMemberCollector(BaseCollector):
    """
    东方财富概念板块成分采集器
    
    采集东方财富概念板块成分股列表
    主数据源：Tushare (dc_member) - 需6000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',           # 交易日期
        'ts_code',              # 概念代码
        'con_code',             # 成分股票代码
        'name',                 # 成分股票名称
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        con_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集东方财富概念板块成分
        
        Args:
            ts_code: 板块指数代码
            con_code: 成分股票代码
            trade_date: 交易日期（YYYYMMDD）
        
        Returns:
            DataFrame: 标准化的东方财富板块成分数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, con_code, trade_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条东方财富板块成分数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取东方财富板块成分失败: {e}")
        
        # 降级到AkShare
        try:
            if ts_code:
                df = self._collect_from_akshare(ts_code)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条东方财富板块成分数据")
                    return df
        except Exception as e:
            logger.error(f"AkShare获取东方财富板块成分失败: {e}")
        
        logger.error("所有数据源均无法获取东方财富板块成分数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        con_code: Optional[str],
        trade_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取东方财富板块成分"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if con_code:
            params['con_code'] = con_code
        if trade_date:
            params['trade_date'] = trade_date
        
        df = pro.dc_member(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, ts_code: str) -> pd.DataFrame:
        """从AkShare获取东方财富板块成分"""
        import akshare as ak
        
        try:
            # 首先获取板块列表来匹配
            concept_df = ak.stock_board_concept_name_em()
            
            # 尝试匹配代码或名称
            matched = concept_df[concept_df['板块代码'].astype(str) == ts_code.split('.')[0]]
            
            if matched.empty:
                logger.warning(f"未找到对应的东方财富板块: {ts_code}")
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            board_name = matched.iloc[0]['板块名称']
            df = ak.stock_board_concept_cons_em(symbol=board_name)
        except Exception as e:
            logger.warning(f"AkShare获取东方财富板块成分失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '代码': 'con_code',
            '名称': 'name',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置板块代码和日期
        df['ts_code'] = ts_code
        df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # 格式化股票代码
        if 'con_code' in df.columns:
            def format_code(code):
                code = str(code)
                if code.startswith('6'):
                    return code + '.SH'
                elif code.startswith(('0', '3')):
                    return code + '.SZ'
                elif code.startswith(('4', '8')):
                    return code + '.BJ'
                return code
            df['con_code'] = df['con_code'].apply(format_code)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("kpl_concept")
class KPLConceptCollector(BaseCollector):
    """
    开盘啦题材库采集器
    
    采集开盘啦概念题材列表
    主数据源：Tushare (kpl_concept) - 需5000积分
    备用数据源：AkShare（功能有限）
    
    注意：此接口因源站改版暂无新增数据
    """
    
    OUTPUT_FIELDS = [
        'trade_date',           # 交易日期
        'ts_code',              # 题材代码
        'name',                 # 题材名称
        'z_t_num',              # 涨停数量
        'up_num',               # 排名上升位数
    ]
    
    def collect(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集开盘啦题材库数据
        
        Args:
            trade_date: 交易日期（YYYYMMDD）
            ts_code: 题材代码
            name: 题材名称
        
        Returns:
            DataFrame: 标准化的题材库数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(trade_date, ts_code, name)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条题材数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取题材数据失败（可能积分不足或接口暂停）: {e}")
        
        logger.error("无法获取题材数据（开盘啦接口暂无新增数据）")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        trade_date: Optional[str],
        ts_code: Optional[str],
        name: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取题材库"""
        pro = self.tushare_api
        
        params = {}
        if trade_date:
            params['trade_date'] = trade_date
        if ts_code:
            params['ts_code'] = ts_code
        if name:
            params['name'] = name
        
        df = pro.kpl_concept(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("kpl_concept_cons")
class KPLConceptConsCollector(BaseCollector):
    """
    开盘啦题材成分采集器
    
    采集开盘啦概念题材的成分股
    主数据源：Tushare (kpl_concept_cons) - 需5000积分
    
    注意：此接口因源站改版暂无新增数据
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 题材代码
        'name',                 # 题材名称
        'con_name',             # 股票名称
        'con_code',             # 股票代码
        'trade_date',           # 交易日期
        'desc',                 # 描述
        'hot_num',              # 人气值
    ]
    
    def collect(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        con_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集开盘啦题材成分数据
        
        Args:
            trade_date: 交易日期（YYYYMMDD）
            ts_code: 题材代码
            con_code: 成分股票代码
        
        Returns:
            DataFrame: 标准化的题材成分数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(trade_date, ts_code, con_code)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条题材成分数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取题材成分失败: {e}")
        
        logger.error("无法获取题材成分数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        trade_date: Optional[str],
        ts_code: Optional[str],
        con_code: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取题材成分"""
        pro = self.tushare_api
        
        params = {}
        if trade_date:
            params['trade_date'] = trade_date
        if ts_code:
            params['ts_code'] = ts_code
        if con_code:
            params['con_code'] = con_code
        
        df = pro.kpl_concept_cons(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_ths_index(
    ts_code: Optional[str] = None,
    exchange: str = 'A',
    type: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取同花顺概念/行业板块列表
    
    Args:
        ts_code: 指数代码
        exchange: 市场类型（A-A股）
        type: 指数类型（N-概念指数, I-行业指数等）
    
    Returns:
        DataFrame: 同花顺板块列表数据
    
    Example:
        >>> df = get_ths_index(type='N')  # 获取概念指数
        >>> df = get_ths_index(type='I')  # 获取行业指数
    """
    collector = THSIndexCollector()
    return collector.collect(ts_code=ts_code, exchange=exchange, type=type, **kwargs)


def get_ths_member(
    ts_code: Optional[str] = None,
    con_code: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取同花顺概念板块成分
    
    Args:
        ts_code: 板块指数代码
        con_code: 成分股票代码
    
    Returns:
        DataFrame: 同花顺板块成分数据
    
    Example:
        >>> df = get_ths_member(ts_code='885800.TI')
    """
    collector = THSMemberCollector()
    return collector.collect(ts_code=ts_code, con_code=con_code, **kwargs)


def get_ths_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取同花顺板块指数行情
    
    Args:
        ts_code: 指数代码
        trade_date: 交易日期（YYYYMMDD）
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 同花顺板块指数行情数据
    
    Example:
        >>> df = get_ths_daily(ts_code='885800.TI', start_date='20250101')
    """
    collector = THSDailyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date, **kwargs)


def get_dc_index(
    ts_code: Optional[str] = None,
    name: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取东方财富概念板块
    
    Args:
        ts_code: 概念代码
        name: 板块名称
        trade_date: 交易日期（YYYYMMDD）
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 东方财富概念板块数据
    
    Example:
        >>> df = get_dc_index(trade_date='20250115')
        >>> df = get_dc_index(name='人形机器人')
    """
    collector = DCIndexCollector()
    return collector.collect(ts_code=ts_code, name=name, trade_date=trade_date, start_date=start_date, end_date=end_date, **kwargs)


def get_dc_member(
    ts_code: Optional[str] = None,
    con_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取东方财富概念板块成分
    
    Args:
        ts_code: 板块指数代码
        con_code: 成分股票代码
        trade_date: 交易日期（YYYYMMDD）
    
    Returns:
        DataFrame: 东方财富板块成分数据
    
    Example:
        >>> df = get_dc_member(ts_code='BK1184.DC', trade_date='20250102')
    """
    collector = DCMemberCollector()
    return collector.collect(ts_code=ts_code, con_code=con_code, trade_date=trade_date, **kwargs)


def get_kpl_concept(
    trade_date: Optional[str] = None,
    ts_code: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取开盘啦题材库
    
    Args:
        trade_date: 交易日期（YYYYMMDD）
        ts_code: 题材代码
        name: 题材名称
    
    Returns:
        DataFrame: 题材库数据
    
    Example:
        >>> df = get_kpl_concept(trade_date='20241014')
    
    注意：此接口因源站改版暂无新增数据
    """
    collector = KPLConceptCollector()
    return collector.collect(trade_date=trade_date, ts_code=ts_code, name=name, **kwargs)


def get_kpl_concept_cons(
    trade_date: Optional[str] = None,
    ts_code: Optional[str] = None,
    con_code: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取开盘啦题材成分
    
    Args:
        trade_date: 交易日期（YYYYMMDD）
        ts_code: 题材代码
        con_code: 成分股票代码
    
    Returns:
        DataFrame: 题材成分数据
    
    Example:
        >>> df = get_kpl_concept_cons(trade_date='20241014')
    
    注意：此接口因源站改版暂无新增数据
    """
    collector = KPLConceptConsCollector()
    return collector.collect(trade_date=trade_date, ts_code=ts_code, con_code=con_code, **kwargs)
