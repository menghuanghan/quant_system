"""
行业数据（Industry Data）采集模块

数据类型包括：
- 申万行业分类（一/二/三级）
- 申万行业成分股
- 申万行业指数行情
- 中信行业指数行情（待实现）
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


@CollectorRegistry.register("sw_index_classify")
class SWIndexClassifyCollector(BaseCollector):
    """
    申万行业分类采集器
    
    采集申万行业分类列表（支持2014和2021版本）
    主数据源：Tushare (index_classify) - 需2000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'index_code',           # 行业指数代码
        'industry_name',        # 行业名称
        'parent_code',          # 父级代码
        'level',                # 行业层级（L1/L2/L3）
        'industry_code',        # 行业代码
        'is_pub',               # 是否发布指数
        'src',                  # 行业分类版本（SW2014/SW2021）
    ]
    
    def collect(
        self,
        level: Optional[str] = None,
        src: str = 'SW2021',
        parent_code: Optional[str] = None,
        index_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集申万行业分类数据
        
        Args:
            level: 行业分级（L1/L2/L3），L1一级，L2二级，L3三级
            src: 指数来源（SW2014：申万2014年版本，SW2021：申万2021年版本）
            parent_code: 父级代码
            index_code: 指数代码
        
        Returns:
            DataFrame: 标准化的申万行业分类数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(level, src, parent_code, index_code)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条申万行业分类数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取申万行业分类失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(level)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条申万行业分类数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取申万行业分类失败: {e}")
        
        logger.error("所有数据源均无法获取申万行业分类数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        level: Optional[str],
        src: str,
        parent_code: Optional[str],
        index_code: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取申万行业分类"""
        pro = self.tushare_api
        
        params = {'src': src}
        if level:
            params['level'] = level
        if parent_code:
            params['parent_code'] = parent_code
        if index_code:
            params['index_code'] = index_code
        
        df = pro.index_classify(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, level: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取申万行业分类"""
        import akshare as ak
        
        try:
            # AkShare提供申万一级行业
            df = ak.sw_index_first_info()
        except Exception as e:
            logger.warning(f"AkShare获取申万行业分类失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '行业代码': 'index_code',
            '行业名称': 'industry_name',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置默认值
        df['level'] = 'L1'
        df['src'] = 'SW2021'
        df['parent_code'] = '0'
        df['industry_code'] = df['index_code']
        df['is_pub'] = '1'
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        logger.warning("AkShare只能获取申万一级行业，部分字段可能为空")
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("sw_index_member")
class SWIndexMemberCollector(BaseCollector):
    """
    申万行业成分采集器
    
    采集申万行业成分股列表（按三级分类）
    主数据源：Tushare (index_member_all) - 需2000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'l1_code',              # 一级行业代码
        'l1_name',              # 一级行业名称
        'l2_code',              # 二级行业代码
        'l2_name',              # 二级行业名称
        'l3_code',              # 三级行业代码
        'l3_name',              # 三级行业名称
        'ts_code',              # 成分股票代码
        'name',                 # 成分股票名称
        'in_date',              # 纳入日期
        'out_date',             # 剔除日期
        'is_new',               # 是否最新（Y/N）
    ]
    
    def collect(
        self,
        l1_code: Optional[str] = None,
        l2_code: Optional[str] = None,
        l3_code: Optional[str] = None,
        ts_code: Optional[str] = None,
        is_new: str = 'Y',
        **kwargs
    ) -> pd.DataFrame:
        """
        采集申万行业成分数据
        
        Args:
            l1_code: 一级行业代码
            l2_code: 二级行业代码
            l3_code: 三级行业代码
            ts_code: 股票代码（查询股票所属行业）
            is_new: 是否最新（默认Y）
        
        Returns:
            DataFrame: 标准化的申万行业成分数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(l1_code, l2_code, l3_code, ts_code, is_new)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条申万行业成分数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取申万行业成分失败: {e}")
        
        # 降级到AkShare
        try:
            if l1_code or l2_code:
                df = self._collect_from_akshare(l1_code)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条申万行业成分数据")
                    return df
        except Exception as e:
            logger.error(f"AkShare获取申万行业成分失败: {e}")
        
        logger.error("所有数据源均无法获取申万行业成分数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        l1_code: Optional[str],
        l2_code: Optional[str],
        l3_code: Optional[str],
        ts_code: Optional[str],
        is_new: str
    ) -> pd.DataFrame:
        """从Tushare获取申万行业成分"""
        pro = self.tushare_api
        
        params = {'is_new': is_new}
        if l1_code:
            params['l1_code'] = l1_code
        if l2_code:
            params['l2_code'] = l2_code
        if l3_code:
            params['l3_code'] = l3_code
        if ts_code:
            params['ts_code'] = ts_code
        
        df = pro.index_member_all(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, index_code: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取申万行业成分"""
        import akshare as ak
        
        try:
            # AkShare需要行业代码
            if not index_code:
                logger.warning("AkShare需要指定行业代码")
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            # 去除后缀
            code = index_code.split('.')[0] if '.' in index_code else index_code
            df = ak.sw_index_cons(index_code=code)
        except Exception as e:
            logger.warning(f"AkShare获取申万行业成分失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '股票代码': 'ts_code',
            '股票名称': 'name',
            '开始日期': 'in_date',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置默认值
        df['l1_code'] = index_code
        df['l1_name'] = None
        df['l2_code'] = None
        df['l2_name'] = None
        df['l3_code'] = None
        df['l3_name'] = None
        df['is_new'] = 'Y'
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        logger.warning("AkShare申万行业成分数据字段有限")
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("sw_daily")
class SWDailyCollector(BaseCollector):
    """
    申万行业指数日行情采集器
    
    采集申万行业指数日线行情（默认申万2021版）
    主数据源：Tushare (sw_daily) - 需5000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 指数代码
        'trade_date',           # 交易日期
        'name',                 # 指数名称
        'open',                 # 开盘点位
        'high',                 # 最高点位
        'low',                  # 最低点位
        'close',                # 收盘点位
        'change',               # 涨跌点位
        'pct_change',           # 涨跌幅（%）
        'vol',                  # 成交量（万股）
        'amount',               # 成交额（万元）
        'pe',                   # 市盈率
        'pb',                   # 市净率
        'float_mv',             # 流通市值（万元）
        'total_mv',             # 总市值（万元）
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
        采集申万行业指数日行情数据
        
        Args:
            ts_code: 行业指数代码
            trade_date: 交易日期（YYYYMMDD）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的申万行业指数日行情数据
        """
        # 优先使用Tushare（注意：需要5000积分，但尝试调用）
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条申万行业指数行情数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取申万行业指数行情失败（可能积分不足）: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code, start_date, end_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条申万行业指数行情数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取申万行业指数行情失败: {e}")
        
        logger.error("所有数据源均无法获取申万行业指数行情数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取申万行业指数日行情"""
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
        
        df = pro.sw_daily(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 日期格式转换
        df = self._convert_date_format(df, ['trade_date'])
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(
        self,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """从AkShare获取申万行业指数日行情"""
        import akshare as ak
        
        try:
            if not ts_code:
                logger.warning("AkShare需要指定行业指数代码")
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            # 提取代码
            code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            df = ak.sw_index_daily_indicator(symbol=code)
        except Exception as e:
            logger.warning(f"AkShare获取申万行业指数行情失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '日期': 'trade_date',
            '收盘': 'close',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'vol',
            '成交额': 'amount',
            '涨跌幅': 'pct_change',
            '涨跌点': 'change',
            '市盈率': 'pe',
            '市净率': 'pb',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置ts_code
        df['ts_code'] = ts_code
        df['name'] = None
        
        # 日期筛选
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


# ============= 便捷函数接口 =============

def get_sw_index_classify(
    level: Optional[str] = None,
    src: str = 'SW2021',
    parent_code: Optional[str] = None,
    index_code: Optional[str] = None
) -> pd.DataFrame:
    """
    获取申万行业分类
    
    Args:
        level: 行业分级（L1/L2/L3）
        src: 指数来源（SW2014/SW2021）
        parent_code: 父级代码
        index_code: 指数代码
    
    Returns:
        DataFrame: 申万行业分类数据
    
    Example:
        >>> df = get_sw_index_classify(level='L1', src='SW2021')
        >>> df = get_sw_index_classify(level='L2', parent_code='801010.SI')
    """
    collector = SWIndexClassifyCollector()
    return collector.collect(level=level, src=src, parent_code=parent_code, index_code=index_code)


def get_sw_index_member(
    l1_code: Optional[str] = None,
    l2_code: Optional[str] = None,
    l3_code: Optional[str] = None,
    ts_code: Optional[str] = None,
    is_new: str = 'Y'
) -> pd.DataFrame:
    """
    获取申万行业成分股
    
    Args:
        l1_code: 一级行业代码
        l2_code: 二级行业代码
        l3_code: 三级行业代码
        ts_code: 股票代码（查询所属行业）
        is_new: 是否最新（默认Y）
    
    Returns:
        DataFrame: 申万行业成分数据
    
    Example:
        >>> df = get_sw_index_member(l1_code='801010.SI')
        >>> df = get_sw_index_member(ts_code='000001.SZ')
    """
    collector = SWIndexMemberCollector()
    return collector.collect(l1_code=l1_code, l2_code=l2_code, l3_code=l3_code, ts_code=ts_code, is_new=is_new)


def get_sw_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取申万行业指数日行情
    
    Args:
        ts_code: 行业指数代码
        trade_date: 交易日期（YYYYMMDD）
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 申万行业指数日行情数据
    
    Example:
        >>> df = get_sw_daily(trade_date='20250115')
        >>> df = get_sw_daily(ts_code='801010.SI', start_date='20250101', end_date='20250115')
    """
    collector = SWDailyCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date)
