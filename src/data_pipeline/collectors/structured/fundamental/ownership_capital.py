"""
股权与资本结构（Ownership & Capital）采集模块

数据类型包括：
- 股本结构
- 前十大股东/流通股东
- 股权质押
- 限售解禁
- 股票回购
- 分红送股
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


@CollectorRegistry.register("share_structure")
class ShareStructureCollector(BaseCollector):
    """
    股本结构采集器
    
    采集公司股本结构数据
    主数据源：Tushare (stk_holdernumber / share_float)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'ann_date',             # 公告日期
        'end_date',             # 截止日期
        'holder_num',           # 股东户数（注意：Tushare返回的是holder_num单数）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股本结构数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的股本结构数据
        """
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条股本结构数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取股本结构失败: {e}")
        
        logger.error("无法获取股本结构数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取股本结构"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if ann_date:
            params['ann_date'] = ann_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = pro.stk_holdernumber(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['ann_date', 'end_date'])
        
        # 确保包含所有字段（只保留Tushare实际返回的字段）
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("top10_holders")
class Top10HoldersCollector(BaseCollector):
    """
    前十大股东采集器
    
    采集前十大股东/流通股东数据
    主数据源：Tushare (top10_holders / top10_floatholders)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'ann_date',             # 公告日期
        'end_date',             # 报告期
        'holder_name',          # 股东名称
        'hold_amount',          # 持股数量（股）
        'hold_ratio',           # 持股比例（%）
        'hold_float_ratio',     # 占流通股比例（%）
        'hold_change',          # 持股变化（股）
        'holder_type',          # 股东类型
    ]
    
    def collect(
        self,
        ts_code: str,
        period: Optional[str] = None,
        type: str = 'all',
        **kwargs
    ) -> pd.DataFrame:
        """
        采集前十大股东数据
        
        Args:
            ts_code: 证券代码（必填）
            period: 报告期（YYYYMMDD）
            type: 类型（all=全部，top10=前十大，float=流通股东）
        
        Returns:
            DataFrame: 标准化的股东数据
        """
        if not ts_code:
            logger.error("需要指定ts_code")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, period, type)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条股东数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取股东数据失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条股东数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取股东数据失败: {e}")
        
        logger.error("无法获取股东数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: str,
        period: Optional[str],
        type: str
    ) -> pd.DataFrame:
        """从Tushare获取股东数据"""
        pro = self.tushare_api
        
        results = []
        
        # 前十大股东
        if type in ['all', 'top10']:
            params = {'ts_code': ts_code}
            if period:
                params['period'] = period
            df1 = pro.top10_holders(**params)
            if not df1.empty:
                df1['holder_type'] = '前十大股东'
                results.append(df1)
        
        # 前十大流通股东
        if type in ['all', 'float']:
            params = {'ts_code': ts_code}
            if period:
                params['period'] = period
            df2 = pro.top10_floatholders(**params)
            if not df2.empty:
                df2['holder_type'] = '前十大流通股东'
                results.append(df2)
        
        if not results:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = pd.concat(results, ignore_index=True)
        df = self._convert_date_format(df, ['ann_date', 'end_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, ts_code: str) -> pd.DataFrame:
        """从AkShare获取股东数据"""
        import akshare as ak
        
        symbol = ts_code.split('.')[0]
        
        try:
            df = ak.stock_main_stock_holder(stock=symbol)
        except Exception as e:
            logger.warning(f"AkShare获取股东数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '股东名称': 'holder_name',
            '持股数量': 'hold_amount',
            '持股比例': 'hold_ratio',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        df['holder_type'] = '前十大股东'
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("pledge")
class PledgeCollector(BaseCollector):
    """
    股权质押采集器
    
    采集股权质押数据
    主数据源：Tushare (pledge_stat / pledge_detail)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'end_date',             # 截止日期
        'pledge_count',         # 质押次数
        'unrest_pledge',        # 无限售股质押数量（万股）
        'rest_pledge',          # 限售股质押数量（万股）
        'total_share',          # 总股本（万股）
        'pledge_ratio',         # 质押比例（%）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股权质押数据
        
        Args:
            ts_code: 证券代码
            end_date: 截止日期
        
        Returns:
            DataFrame: 标准化的股权质押数据
        """
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条股权质押数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取股权质押失败: {e}")
        
        logger.error("无法获取股权质押数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取股权质押"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if end_date:
            params['end_date'] = end_date
        
        df = pro.pledge_stat(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['end_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("share_float")
class ShareFloatCollector(BaseCollector):
    """
    限售解禁采集器
    
    采集限售股解禁数据
    主数据源：Tushare (share_float)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'ann_date',             # 公告日期
        'float_date',           # 解禁日期
        'float_share',          # 解禁股份（万股）
        'float_ratio',          # 解禁股份占总股本比例（%）
        'holder_name',          # 股东名称
        'share_type',           # 股份类型
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        float_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集限售解禁数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            float_date: 解禁日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的限售解禁数据
        """
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, float_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条限售解禁数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取限售解禁失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                if ts_code:
                    df = df[df['ts_code'] == ts_code]
                logger.info(f"从AkShare成功获取 {len(df)} 条限售解禁数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取限售解禁失败: {e}")
        
        logger.error("无法获取限售解禁数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        float_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取限售解禁"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if ann_date:
            params['ann_date'] = ann_date
        if float_date:
            params['float_date'] = float_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = pro.share_float(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['ann_date', 'float_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取限售解禁"""
        import akshare as ak
        
        try:
            df = ak.stock_restricted_release_summary_em()
        except Exception as e:
            logger.warning(f"AkShare获取限售解禁失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '代码': 'symbol',
            '解禁日期': 'float_date',
            '解禁数量': 'float_share',
            '解禁市值': 'float_mv',
        }
        df = self._standardize_columns(df, column_mapping)
        
        if 'symbol' in df.columns:
            df['ts_code'] = df['symbol'].apply(
                lambda x: f"{str(x).zfill(6)}.SZ" if str(x).startswith(('0', '3')) 
                else f"{str(x).zfill(6)}.SH"
            )
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("repurchase")
class RepurchaseCollector(BaseCollector):
    """
    股票回购采集器
    
    采集股票回购数据
    主数据源：Tushare (repurchase)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'ann_date',             # 公告日期
        'end_date',             # 截止日期
        'proc',                 # 进度
        'vol',                  # 回购数量（股）
        'amount',               # 回购金额（元）
        'high_limit',           # 回购价格上限（元）
        'low_limit',            # 回购价格下限（元）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股票回购数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的回购数据
        """
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条回购数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取回购数据失败: {e}")
        
        logger.error("无法获取回购数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取回购数据"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if ann_date:
            params['ann_date'] = ann_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = pro.repurchase(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['ann_date', 'end_date', 'exp_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("dividend")
class DividendCollector(BaseCollector):
    """
    分红送股采集器
    
    采集分红送股数据
    主数据源：Tushare (dividend)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'end_date',             # 分红年度
        'ann_date',             # 预案公告日
        'div_proc',             # 实施进度
        'stk_div',              # 每股送转（股）
        'cash_div',             # 每股分红（税前）（元）
        'cash_div_tax',         # 每股分红（税后）（元）
        'record_date',          # 股权登记日
        'ex_date',              # 除权除息日
        'pay_date',             # 派息日
        'imp_ann_date',         # 实施公告日
        # 注意：base_date和base_share字段Tushare不返回，已移除
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        record_date: Optional[str] = None,
        ex_date: Optional[str] = None,
        imp_ann_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集分红送股数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            record_date: 股权登记日
            ex_date: 除权除息日
            imp_ann_date: 实施公告日
        
        Returns:
            DataFrame: 标准化的分红送股数据
        """
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, record_date, ex_date, imp_ann_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条分红数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取分红数据失败: {e}")
        
        # 降级到AkShare
        try:
            if ts_code:
                df = self._collect_from_akshare(ts_code)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条分红数据")
                    return df
        except Exception as e:
            logger.error(f"AkShare获取分红数据失败: {e}")
        
        logger.error("无法获取分红数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        record_date: Optional[str],
        ex_date: Optional[str],
        imp_ann_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取分红数据"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if ann_date:
            params['ann_date'] = ann_date
        if record_date:
            params['record_date'] = record_date
        if ex_date:
            params['ex_date'] = ex_date
        if imp_ann_date:
            params['imp_ann_date'] = imp_ann_date
        
        df = pro.dividend(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 只转换实际存在的日期字段
        df = self._convert_date_format(df, ['end_date', 'ann_date', 'record_date', 
                                           'ex_date', 'pay_date', 'div_listdate',
                                           'imp_ann_date'])
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, ts_code: str) -> pd.DataFrame:
        """从AkShare获取分红数据"""
        import akshare as ak
        
        symbol = ts_code.split('.')[0]
        
        try:
            df = ak.stock_fhps_detail_em(symbol=symbol)
        except Exception as e:
            logger.warning(f"AkShare获取分红数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '公告日期': 'ann_date',
            '分红年度': 'end_date',
            '送股': 'stk_bo_rate',
            '转增': 'stk_co_rate',
            '派息': 'cash_div',
            '股权登记日': 'record_date',
            '除权除息日': 'ex_date',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_share_structure(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取股本结构数据
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 股本结构数据
    """
    collector = ShareStructureCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            start_date=start_date, end_date=end_date)


def get_top10_holders(
    ts_code: str,
    period: Optional[str] = None,
    type: str = 'all'
) -> pd.DataFrame:
    """
    获取前十大股东数据
    
    Args:
        ts_code: 证券代码（必填）
        period: 报告期
        type: all=全部，top10=前十大，float=流通股东
    
    Returns:
        DataFrame: 股东数据
    
    Example:
        >>> df = get_top10_holders(ts_code='000001.SZ')
    """
    collector = Top10HoldersCollector()
    return collector.collect(ts_code=ts_code, period=period, type=type)


def get_pledge(
    ts_code: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取股权质押数据
    
    Args:
        ts_code: 证券代码
        end_date: 截止日期
    
    Returns:
        DataFrame: 股权质押数据
    """
    collector = PledgeCollector()
    return collector.collect(ts_code=ts_code, end_date=end_date)


def get_share_float(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    float_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取限售解禁数据
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        float_date: 解禁日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 限售解禁数据
    """
    collector = ShareFloatCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            float_date=float_date, start_date=start_date,
                            end_date=end_date)


def get_repurchase(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取股票回购数据
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 回购数据
    """
    collector = RepurchaseCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            start_date=start_date, end_date=end_date)


def get_dividend(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    record_date: Optional[str] = None,
    ex_date: Optional[str] = None,
    imp_ann_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取分红送股数据
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        record_date: 股权登记日
        ex_date: 除权除息日
        imp_ann_date: 实施公告日
    
    Returns:
        DataFrame: 分红送股数据
    
    Example:
        >>> df = get_dividend(ts_code='000001.SZ')
    """
    collector = DividendCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            record_date=record_date, ex_date=ex_date,
                            imp_ann_date=imp_ann_date)
