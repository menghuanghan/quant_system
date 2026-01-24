"""
债券与可转债（Bond & Convertible Bond）采集模块

数据类型包括：
- 国债收益率曲线
- 可转债基本信息
- 可转债行情
- 可转债溢价率
- 债券回购日行情
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


@CollectorRegistry.register("yc_cb")
class YieldCurveCollector(BaseCollector):
    """
    国债收益率曲线采集器
    
    获取中债收益率曲线数据
    主数据源：Tushare (yc_cb, 特殊权限)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',       # 交易日期
        'ts_code',          # 曲线编码
        'curve_name',       # 曲线名称
        'curve_type',       # 曲线类型（0-到期，1-即期）
        'curve_term',       # 期限(年)
        'yield',            # 收益率(%)
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = '1001.CB',
        curve_type: Optional[str] = '0',
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        curve_term: Optional[float] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集国债收益率曲线
        
        Args:
            ts_code: 收益率曲线编码（1001.CB-国债收益率曲线）
            curve_type: 曲线类型（0-到期，1-即期）
            trade_date: 交易日期
            start_date: 查询起始日期
            end_date: 查询结束日期
            curve_term: 期限
        
        Returns:
            DataFrame: 标准化的国债收益率曲线数据
        """
        # 优先使用Tushare（需要特殊权限）
        try:
            df = self._collect_from_tushare(ts_code, curve_type, trade_date, start_date, end_date, curve_term)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条收益率曲线数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取收益率曲线失败（需要特殊权限）: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(trade_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条收益率曲线数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取收益率曲线失败: {e}")
        
        logger.error("无法获取收益率曲线数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        curve_type: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        curve_term: Optional[float]
    ) -> pd.DataFrame:
        """从Tushare获取收益率曲线"""
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if curve_type:
            params['curve_type'] = curve_type
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if curve_term:
            params['curve_term'] = curve_term
        
        df = self.tushare_api.yc_cb(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取收益率曲线"""
        import akshare as ak
        
        try:
            # 获取国债收益率曲线
            df = ak.bond_china_yield(start_date="2020-01-01")
        except Exception as e:
            logger.warning(f"AkShare获取收益率曲线失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '日期': 'trade_date',
            '中债国债到期收益率:1年': 'y_1y',
            '中债国债到期收益率:2年': 'y_2y',
            '中债国债到期收益率:5年': 'y_5y',
            '中债国债到期收益率:10年': 'y_10y',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 转换为长格式
        result = []
        for _, row in df.iterrows():
            date = row.get('trade_date', '')
            if pd.notna(date):
                date_str = pd.to_datetime(date).strftime('%Y%m%d')
                for term, col in [(1, 'y_1y'), (2, 'y_2y'), (5, 'y_5y'), (10, 'y_10y')]:
                    if col in row and pd.notna(row[col]):
                        result.append({
                            'trade_date': date_str,
                            'ts_code': '1001.CB',
                            'curve_name': '中债国债收益率曲线',
                            'curve_type': '0',
                            'curve_term': float(term),
                            'yield': float(row[col])
                        })
        
        if not result:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df_result = pd.DataFrame(result)
        
        # 日期筛选
        if trade_date:
            df_result = df_result[df_result['trade_date'] == trade_date]
        
        return df_result[self.OUTPUT_FIELDS]


@CollectorRegistry.register("cb_basic")
class CBBasicCollector(BaseCollector):
    """
    可转债基本信息采集器
    
    获取可转债基本信息
    主数据源：Tushare (cb_basic, 2000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 转债代码
        'bond_full_name',       # 转债名称
        'bond_short_name',      # 转债简称
        'cb_code',              # 转股申报代码
        'stk_code',             # 正股代码
        'stk_short_name',       # 正股简称
        'maturity',             # 发行期限(年)
        'par',                  # 面值
        'issue_price',          # 发行价格
        'issue_size',           # 发行总额(元)
        'remain_size',          # 债券余额(元)
        'value_date',           # 起息日期
        'maturity_date',        # 到期日期
        'rate_type',            # 利率类型
        'coupon_rate',          # 票面利率(%)
        'add_rate',             # 补偿利率(%)
        'pay_per_year',         # 年付息次数
        'list_date',            # 上市日期
        'delist_date',          # 摘牌日
        'exchange',             # 上市地点
        'conv_start_date',      # 转股起始日
        'conv_end_date',        # 转股截止日
        'first_conv_price',     # 初始转股价
        'conv_price',           # 最新转股价
        'issue_rating',         # 发行信用等级
        'newest_rating',        # 最新信用等级
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        list_date: Optional[str] = None,
        exchange: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集可转债基本信息
        
        Args:
            ts_code: 转债代码
            list_date: 上市日期
            exchange: 上市地点
        
        Returns:
            DataFrame: 标准化的可转债基本信息
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, list_date, exchange)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条可转债基本信息")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取可转债基本信息失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条可转债基本信息")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取可转债基本信息失败: {e}")
        
        logger.error("无法获取可转债基本信息")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        list_date: Optional[str],
        exchange: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取可转债基本信息"""
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if list_date:
            params['list_date'] = list_date
        if exchange:
            params['exchange'] = exchange
        
        df = self.tushare_api.cb_basic(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取可转债基本信息"""
        import akshare as ak
        
        try:
            df = ak.bond_cb_jsl()
        except Exception as e:
            logger.warning(f"AkShare获取可转债基本信息失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '转债代码': 'ts_code',
            '转债名称': 'bond_short_name',
            '正股代码': 'stk_code',
            '正股名称': 'stk_short_name',
            '转股价': 'conv_price',
            '到期时间': 'maturity_date',
            '评级': 'newest_rating',
        }
        df = self._standardize_columns(df, column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("cb_daily")
class CBDailyCollector(BaseCollector):
    """
    可转债行情采集器
    
    获取可转债日线行情
    主数据源：Tushare (cb_daily, 2000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 转债代码
        'trade_date',       # 交易日期
        'pre_close',        # 昨收盘价
        'open',             # 开盘价
        'high',             # 最高价
        'low',              # 最低价
        'close',            # 收盘价
        'change',           # 涨跌(元)
        'pct_chg',          # 涨跌幅(%)
        'vol',              # 成交量(手)
        'amount',           # 成交金额(万元)
        'bond_value',       # 纯债价值
        'bond_over_rate',   # 纯债溢价率(%)
        'cb_value',         # 转股价值
        'cb_over_rate',     # 转股溢价率(%)
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
        采集可转债行情
        
        Args:
            ts_code: TS代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的可转债行情数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条可转债行情数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取可转债行情失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code, trade_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条可转债行情数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取可转债行情失败: {e}")
        
        logger.error("无法获取可转债行情数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取可转债行情"""
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = self.tushare_api.cb_daily(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # Tushare的cb_daily不包含溢价率字段，尝试从AkShare补充
        if 'bond_value' not in df.columns or df['bond_value'].isna().all():
            try:
                import akshare as ak
                df_premium = ak.bond_cb_jsl()
                if not df_premium.empty:
                    # 正确的字段映射（使用实际字段名）
                    # 处理代码格式：AkShare返回6位代码，需要添加交易所后缀
                    def add_exchange_suffix(code):
                        if pd.isna(code):
                            return None
                        code_str = str(int(code)) if isinstance(code, float) else str(code)
                        code_str = code_str.zfill(6)
                        if code_str.startswith('11'):
                            return f"{code_str}.SH"
                        elif code_str.startswith('12'):
                            return f"{code_str}.SZ"
                        elif code_str.startswith('40'):
                            return f"{code_str}.SZ"
                        else:
                            return code_str
                    
                    df_premium['ts_code'] = df_premium['代码'].apply(add_exchange_suffix)
                    
                    # 提取需要的字段
                    premium_cols = {
                        '转股溢价率': 'cb_over_rate',
                        '转股价值': 'cb_value',
                    }
                    
                    for akshare_col, std_col in premium_cols.items():
                        if akshare_col in df_premium.columns:
                            df_premium[std_col] = df_premium[akshare_col]
                    
                    # 合并数据（只合并有ts_code的行）
                    merge_cols = ['ts_code', 'cb_over_rate', 'cb_value']
                    existing_cols = [c for c in merge_cols if c in df_premium.columns]
                    
                    if 'ts_code' in df_premium.columns:
                        df = df.merge(
                            df_premium[existing_cols].drop_duplicates('ts_code'),
                            on='ts_code',
                            how='left',
                            suffixes=('', '_new')
                        )
                        # 使用新数据填充空值
                        for col in ['cb_over_rate', 'cb_value']:
                            if col + '_new' in df.columns:
                                df[col] = df[col].fillna(df[col + '_new'])
                                df.drop(columns=[col + '_new'], inplace=True)
                        logger.info(f"从AkShare补充了 {df['cb_over_rate'].notna().sum()} 条可转债溢价率数据")
                    
                    # 纯债价值AkShare不提供，保持为空
                    if 'bond_value' not in df.columns:
                        df['bond_value'] = None
                    if 'bond_over_rate' not in df.columns:
                        df['bond_over_rate'] = None
                        
            except Exception as e:
                logger.warning(f"从AkShare补充溢价率数据失败: {e}")
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str]
    ) -> pd.DataFrame:
        """从AkShare获取可转债行情"""
        import akshare as ak
        
        try:
            # 获取实时行情
            df = ak.bond_cb_jsl()
        except Exception as e:
            logger.warning(f"AkShare获取可转债行情失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 正确的字段映射
        column_mapping = {
            '代码': 'raw_code',
            '转债名称': 'name',
            '现价': 'close',              # 注意：是"现价"
            '涨跌幅': 'pct_chg',
            '转股溢价率': 'cb_over_rate',
            '转股价值': 'cb_value',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 补充ts_code（添加交易所后缀）
        if 'raw_code' in df.columns:
            def add_exchange_suffix(code):
                if pd.isna(code):
                    return None
                code_str = str(int(code)) if isinstance(code, float) else str(code)
                code_str = code_str.zfill(6)
                if code_str.startswith('11'):
                    return f"{code_str}.SH"
                elif code_str.startswith('12'):
                    return f"{code_str}.SZ"
                elif code_str.startswith('40'):
                    return f"{code_str}.SZ"
                else:
                    return code_str
            
            df['ts_code'] = df['raw_code'].apply(add_exchange_suffix)
        
        df['trade_date'] = trade_date or datetime.now().strftime('%Y%m%d')
        
        # 筛选指定转债
        if ts_code:
            df = df[df['ts_code'] == ts_code]
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("repo_daily")
class RepoDailyCollector(BaseCollector):
    """
    债券回购日行情采集器
    
    获取债券回购日行情
    主数据源：Tushare (repo_daily, 2000积分)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # TS代码
        'trade_date',       # 交易日期
        'repo_maturity',    # 期限品种
        'pre_close',        # 前收盘(%)
        'open',             # 开盘价(%)
        'high',             # 最高价(%)
        'low',              # 最低价(%)
        'close',            # 收盘价(%)
        'weight',           # 加权价(%)
        'weight_r',         # 加权价(利率债)(%)
        'amount',           # 成交金额(万元)
        'num',              # 成交笔数(笔)
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
        采集债券回购日行情
        
        Args:
            ts_code: TS代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的债券回购日行情
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条回购行情数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取回购行情失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(trade_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条回购行情数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取回购行情失败: {e}")
        
        logger.error("无法获取回购行情数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取回购行情"""
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
            return self.tushare_api.repo_daily(**params)
            
        # 分块采集以避免API限制 (Tushare限制约2000-5000条)
        if start_date and end_date:
            try:
                from datetime import timedelta
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y%m%d')
                
                all_dfs = []
                current_dt = start_dt
                while current_dt <= end_dt:
                    # 每次采集一个月
                    next_month = current_dt.replace(day=28) + timedelta(days=4)
                    next_month = next_month.replace(day=1)
                    chunk_end_dt = min(next_month - timedelta(days=1), end_dt)
                    
                    p = params.copy()
                    p['start_date'] = current_dt.strftime('%Y%m%d')
                    p['end_date'] = chunk_end_dt.strftime('%Y%m%d')
                    
                    try:
                        chunk_df = self.tushare_api.repo_daily(**p)
                        if not chunk_df.empty:
                            all_dfs.append(chunk_df)
                    except Exception as e:
                        logger.warning(f"回购行情分块采集失败 ({p['start_date']}-{p['end_date']}): {e}")
                    
                    current_dt = chunk_end_dt + timedelta(days=1)
                
                if all_dfs:
                    df = pd.concat(all_dfs, ignore_index=True)
                    # 执行分文件存储逻辑
                    self._split_and_save_repo_daily(df)
                else:
                    df = pd.DataFrame(columns=self.OUTPUT_FIELDS)
            except Exception as e:
                logger.error(f"回购行情分块逻辑异常: {e}, 尝试直接采集")
                if start_date: params['start_date'] = start_date
                if end_date: params['end_date'] = end_date
                df = self.tushare_api.repo_daily(**params)
                self._split_and_save_repo_daily(df)
        else:
            if start_date: params['start_date'] = start_date
            if end_date: params['end_date'] = end_date
            df = self.tushare_api.repo_daily(**params)
            self._split_and_save_repo_daily(df)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]

    def _split_and_save_repo_daily(self, df: pd.DataFrame):
        """将聚合的回购行情按 ts_code 拆分并保存"""
        from pathlib import Path
        if df.empty or 'ts_code' not in df.columns:
            return
            
        output_base = Path("data/raw/structured/derivatives/repo_daily")
        output_base.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"正在拆分回购行情，涉及 {len(df['ts_code'].unique())} 个品种...")
        
        for code, group in df.groupby('ts_code'):
            file_path = output_base / f"{code.replace('.', '_')}.parquet"
            group.to_parquet(file_path, index=False, compression='snappy')
        
        logger.info("回购行情拆分保存完成")
    
    def _collect_from_akshare(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取回购行情"""
        import akshare as ak
        
        try:
            # 获取上交所国债逆回购实时行情
            df = ak.bond_repo_zh_sse()
        except Exception as e:
            logger.warning(f"AkShare获取回购行情失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '代码': 'ts_code',
            '名称': 'repo_maturity',
            '最新价': 'close',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '成交额': 'amount',
        }
        df = self._standardize_columns(df, column_mapping)
        df['trade_date'] = trade_date or datetime.now().strftime('%Y%m%d')
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("cb_premium")
class CBPremiumCollector(BaseCollector):
    """
    可转债溢价率采集器
    
    获取可转债溢价率数据
    主数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 转债代码
        'bond_short_name',  # 转债简称
        'stk_code',         # 正股代码
        'trade_date',       # 交易日期
        'close',            # 转债价格
        'stk_close',        # 正股价格
        'conv_price',       # 转股价
        'cb_value',         # 转股价值
        'cb_over_rate',     # 转股溢价率(%)
        'bond_value',       # 纯债价值
        'bond_over_rate',   # 纯债溢价率(%)
        'double_low',       # 双低值
        'ytm',              # 到期收益率(%)
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集可转债溢价率数据
        
        Args:
            ts_code: 转债代码
            trade_date: 交易日期
        
        Returns:
            DataFrame: 标准化的可转债溢价率数据
        """
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条可转债溢价率数据")
                
                # 筛选指定转债
                if ts_code:
                    df = df[df['ts_code'] == ts_code]
                
                return df
        except Exception as e:
            logger.error(f"获取可转债溢价率失败: {e}")
        
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取可转债溢价率"""
        import akshare as ak
        
        try:
            df = ak.bond_cb_jsl()
        except Exception as e:
            logger.warning(f"AkShare获取可转债溢价率失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 正确的字段映射（基于实际返回字段）
        column_mapping = {
            '代码': 'raw_code',          # 6位代码，需要补充交易所后缀
            '转债名称': 'bond_short_name',
            '正股代码': 'stk_code',
            '现价': 'close',              # 注意：是"现价"不是"价格"
            '正股价': 'stk_close',
            '转股价': 'conv_price',
            '转股价值': 'cb_value',
            '转股溢价率': 'cb_over_rate',
            '到期税前收益': 'ytm',        # 注意：是"到期税前收益"不是"到期收益率"
            '双低': 'double_low',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 补充交易所后缀生成完整ts_code
        if 'raw_code' in df.columns:
            def add_exchange_suffix(code):
                """根据代码规则添加交易所后缀"""
                if pd.isna(code):
                    return None
                code_str = str(int(code)) if isinstance(code, float) else str(code)
                code_str = code_str.zfill(6)  # 补齐6位
                # 11开头是上交所，12开头是深交所
                if code_str.startswith('11'):
                    return f"{code_str}.SH"
                elif code_str.startswith('12'):
                    return f"{code_str}.SZ"
                elif code_str.startswith('40'):  # 退市转债
                    return f"{code_str}.SZ"
                else:
                    return code_str  # 无法判断的保持原样
            
            df['ts_code'] = df['raw_code'].apply(add_exchange_suffix)
        
        # 手动计算纯债价值和纯债溢价率（如果数据源未提供）
        if 'bond_value' not in df.columns or df['bond_value'].isna().all():
            # 简化计算：纯债价值 ≈ 转债价格 / (1 + 转股溢价率/100)
            # 注意：这是估算值，准确值需要用债券定价模型
            df['bond_value'] = None  # AkShare不提供，设为None
            logger.info("AkShare不提供纯债价值，该字段为空")
        
        if 'bond_over_rate' not in df.columns or df['bond_over_rate'].isna().all():
            # 纯债溢价率 = (转债价格 - 纯债价值) / 纯债价值 * 100
            # 由于纯债价值未知，该字段也设为None
            df['bond_over_rate'] = None
            logger.info("无法计算纯债溢价率（缺少纯债价值）")
        
        df['trade_date'] = datetime.now().strftime('%Y%m%d')
        
        # 确保所有输出字段存在
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_yield_curve(
    ts_code: str = '1001.CB',
    curve_type: str = '0',
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取国债收益率曲线
    
    Args:
        ts_code: 收益率曲线编码（1001.CB-国债收益率曲线）
        curve_type: 曲线类型（0-到期，1-即期）
        trade_date: 交易日期
        start_date: 查询起始日期
        end_date: 查询结束日期
    
    Returns:
        DataFrame: 国债收益率曲线数据
    
    Example:
        >>> df = get_yield_curve(trade_date='20250115')
    """
    collector = YieldCurveCollector()
    return collector.collect(ts_code=ts_code, curve_type=curve_type, trade_date=trade_date,
                            start_date=start_date, end_date=end_date)


def get_cb_basic(
    ts_code: Optional[str] = None,
    list_date: Optional[str] = None,
    exchange: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取可转债基本信息
    
    Args:
        ts_code: 转债代码
        list_date: 上市日期
        exchange: 上市地点
        **kwargs: 其他参数（由调度器传入）
    
    Returns:
        DataFrame: 可转债基本信息
    
    Example:
        >>> df = get_cb_basic()  # 获取全部
        >>> df = get_cb_basic(exchange='SH')  # 上交所可转债
    """
    collector = CBBasicCollector()
    return collector.collect(ts_code=ts_code, list_date=list_date, exchange=exchange, **kwargs)


def get_cb_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取可转债行情
    
    Args:
        ts_code: TS代码
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 可转债行情数据
    
    Example:
        >>> df = get_cb_daily(trade_date='20250115')
        >>> df = get_cb_daily(ts_code='110030.SH', start_date='20250101')
    """
    collector = CBDailyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date)


def get_repo_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取债券回购日行情
    
    Args:
        ts_code: TS代码
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 债券回购日行情
    
    Example:
        >>> df = get_repo_daily(trade_date='20250115')
    """
    collector = RepoDailyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date)


def get_cb_premium(ts_code: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    获取可转债溢价率数据
    
    Args:
        ts_code: 转债代码
        **kwargs: 其他参数（由调度器传入）
    
    Returns:
        DataFrame: 可转债溢价率数据
    
    Example:
        >>> df = get_cb_premium()  # 获取全部
        >>> df = get_cb_premium(ts_code='110030.SH')  # 指定转债
    """
    collector = CBPremiumCollector()
    return collector.collect(ts_code=ts_code, **kwargs)
