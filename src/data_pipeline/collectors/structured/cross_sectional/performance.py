"""
板块行情与强弱（Sector Performance）采集模块

数据类型包括：
- 板块涨跌幅排行
- 行业板块行情
- 概念板块行情
- 板块热度排行
- 板块轮动数据
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


@CollectorRegistry.register("sector_performance")
class SectorPerformanceCollector(BaseCollector):
    """
    板块涨跌幅排行采集器
    
    采集行业/概念板块当日涨跌幅排行
    主数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'sector_code',          # 板块代码
        'sector_name',          # 板块名称
        'sector_type',          # 板块类型（industry/concept）
        'trade_date',           # 交易日期
        'pct_change',           # 涨跌幅（%）
        'change',               # 涨跌额
        'close',                # 最新价
        'total_mv',             # 总市值
        'turnover_rate',        # 换手率
        'up_num',               # 上涨家数
        'down_num',             # 下跌家数
        'leading_stock',        # 领涨股
        'leading_pct',          # 领涨股涨幅
    ]
    
    def collect(
        self,
        sector_type: str = 'industry',
        trade_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集板块涨跌幅排行数据
        
        Args:
            sector_type: 板块类型（industry-行业, concept-概念）
            trade_date: 交易日期（暂不支持历史数据）
        
        Returns:
            DataFrame: 标准化的板块涨跌幅排行数据
        """
        try:
            df = self._collect_from_akshare(sector_type)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条板块涨跌幅数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取板块涨跌幅失败: {e}")
        
        logger.error("无法获取板块涨跌幅数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self, sector_type: str = 'industry') -> pd.DataFrame:
        """从AkShare获取板块涨跌幅"""
        import akshare as ak
        
        try:
            if sector_type == 'industry':
                # 东方财富行业板块排行
                df = ak.stock_board_industry_name_em()
            else:
                # 东方财富概念板块排行
                df = ak.stock_board_concept_name_em()
        except Exception as e:
            logger.warning(f"AkShare获取{sector_type}板块数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '板块名称': 'sector_name',
            '板块代码': 'sector_code',
            '涨跌幅': 'pct_change',
            '最新价': 'close',
            '涨跌额': 'change',
            '成交量': 'vol',
            '成交额': 'amount',
            '换手率': 'turnover_rate',
            '总市值': 'total_mv',
            '上涨家数': 'up_num',
            '下跌家数': 'down_num',
            '领涨股票': 'leading_stock',
            '领涨股票-涨跌幅': 'leading_pct',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置板块类型和日期
        df['sector_type'] = sector_type
        df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("industry_board_em")
class IndustryBoardEMCollector(BaseCollector):
    """
    东方财富行业板块行情采集器
    
    采集东方财富行业板块实时行情
    主数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'sector_code',          # 板块代码
        'sector_name',          # 板块名称
        'trade_date',           # 交易日期
        'pct_change',           # 涨跌幅（%）
        'close',                # 最新价
        'change',               # 涨跌额
        # 'vol',                # 成交量（AkShare接口不返回）
        # 'amount',             # 成交额（AkShare接口不返回）
        'turnover_rate',        # 换手率
        'total_mv',             # 总市值
        'up_num',               # 上涨家数
        'down_num',             # 下跌家数
        'leading_stock',        # 领涨股
        'leading_pct',          # 领涨股涨幅
    ]
    
    def collect(self, **kwargs) -> pd.DataFrame:
        """
        采集东方财富行业板块行情
        
        Returns:
            DataFrame: 标准化的行业板块行情数据
        """
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条东方财富行业板块数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取东方财富行业板块失败: {e}")
        
        logger.error("无法获取东方财富行业板块数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取东方财富行业板块"""
        import akshare as ak
        
        try:
            df = ak.stock_board_industry_name_em()
        except Exception as e:
            logger.warning(f"AkShare获取东方财富行业板块失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '板块代码': 'sector_code',
            '板块名称': 'sector_name',
            '涨跌幅': 'pct_change',
            '最新价': 'close',
            '涨跌额': 'change',
            '成交量': 'vol',
            '成交额': 'amount',
            '换手率': 'turnover_rate',
            '总市值': 'total_mv',
            '上涨家数': 'up_num',
            '下跌家数': 'down_num',
            '领涨股票': 'leading_stock',
            '领涨股票-涨跌幅': 'leading_pct',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置日期
        df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("concept_board_em")
class ConceptBoardEMCollector(BaseCollector):
    """
    东方财富概念板块行情采集器
    
    采集东方财富概念板块实时行情
    主数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'sector_code',          # 板块代码
        'sector_name',          # 板块名称
        'trade_date',           # 交易日期
        'pct_change',           # 涨跌幅（%）
        'close',                # 最新价
        'change',               # 涨跌额
        # 'vol',                # 成交量（AkShare接口不返回）
        # 'amount',             # 成交额（AkShare接口不返回）
        'turnover_rate',        # 换手率
        'total_mv',             # 总市值
        'up_num',               # 上涨家数
        'down_num',             # 下跌家数
        'leading_stock',        # 领涨股
        'leading_pct',          # 领涨股涨幅
    ]
    
    def collect(self, **kwargs) -> pd.DataFrame:
        """
        采集东方财富概念板块行情
        
        Returns:
            DataFrame: 标准化的概念板块行情数据
        """
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条东方财富概念板块数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取东方财富概念板块失败: {e}")
        
        logger.error("无法获取东方财富概念板块数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取东方财富概念板块"""
        import akshare as ak
        
        try:
            df = ak.stock_board_concept_name_em()
        except Exception as e:
            logger.warning(f"AkShare获取东方财富概念板块失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '板块代码': 'sector_code',
            '板块名称': 'sector_name',
            '涨跌幅': 'pct_change',
            '最新价': 'close',
            '涨跌额': 'change',
            '成交量': 'vol',
            '成交额': 'amount',
            '换手率': 'turnover_rate',
            '总市值': 'total_mv',
            '上涨家数': 'up_num',
            '下跌家数': 'down_num',
            '领涨股票': 'leading_stock',
            '领涨股票-涨跌幅': 'leading_pct',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置日期
        df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("sector_hist")
class SectorHistCollector(BaseCollector):
    """
    板块历史行情采集器
    
    采集行业/概念板块历史日线行情
    主数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'sector_name',          # 板块名称
        'trade_date',           # 交易日期
        'open',                 # 开盘
        'high',                 # 最高
        'low',                  # 最低
        'close',                # 收盘
        'vol',                  # 成交量
        'amount',               # 成交额
        'pct_change',           # 涨跌幅
        'change',               # 涨跌额
        'amplitude',            # 振幅
        'turnover_rate',        # 换手率
    ]
    
    def collect(
        self,
        sector_name: str,
        sector_type: str = 'concept',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集板块历史行情数据
        
        Args:
            sector_name: 板块名称
            sector_type: 板块类型（concept-概念, industry-行业）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的板块历史行情数据
        """
        try:
            df = self._collect_from_akshare(sector_name, sector_type, start_date, end_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条板块历史行情数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取板块历史行情失败: {e}")
        
        logger.error("无法获取板块历史行情数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(
        self,
        sector_name: str,
        sector_type: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从AkShare获取板块历史行情"""
        import akshare as ak
        
        try:
            if sector_type == 'concept':
                # 东方财富概念板块历史行情
                df = ak.stock_board_concept_hist_em(symbol=sector_name, adjust="")
            else:
                # 东方财富行业板块历史行情
                df = ak.stock_board_industry_hist_em(symbol=sector_name, adjust="")
        except Exception as e:
            logger.warning(f"AkShare获取板块历史行情失败: {e}")
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
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '振幅': 'amplitude',
            '换手率': 'turnover_rate',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置板块名称
        df['sector_name'] = sector_name
        
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


@CollectorRegistry.register("sector_rank")
class SectorRankCollector(BaseCollector):
    """
    板块热度排行采集器
    
    采集板块热度/关注度排行
    主数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'rank',                 # 排名
        'sector_name',          # 板块名称
        'sector_type',          # 板块类型
        'trade_date',           # 交易日期
        'pct_change',           # 涨跌幅
        'turnover_rate',        # 换手率
        'up_num',               # 上涨家数
        'down_num',             # 下跌家数
        # 'vol',                # 成交量（AkShare接口不返回）
        # 'amount',             # 成交额（AkShare接口不返回）
    ]
    
    def collect(
        self,
        sector_type: str = 'concept',
        rank_by: str = 'pct_change',
        top_n: int = 50,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集板块热度排行数据
        
        Args:
            sector_type: 板块类型（concept-概念, industry-行业）
            rank_by: 排序字段（pct_change-涨跌幅, turnover_rate-换手率）
            top_n: 返回前N条
        
        Returns:
            DataFrame: 标准化的板块热度排行数据
        """
        try:
            df = self._collect_from_akshare(sector_type, rank_by, top_n)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条板块热度排行数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取板块热度排行失败: {e}")
        
        logger.error("无法获取板块热度排行数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(
        self,
        sector_type: str,
        rank_by: str,
        top_n: int
    ) -> pd.DataFrame:
        """从AkShare获取板块热度排行"""
        import akshare as ak
        
        try:
            if sector_type == 'concept':
                df = ak.stock_board_concept_name_em()
            else:
                df = ak.stock_board_industry_name_em()
        except Exception as e:
            logger.warning(f"AkShare获取板块排行失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '板块名称': 'sector_name',
            '涨跌幅': 'pct_change',
            '换手率': 'turnover_rate',
            '成交量': 'vol',
            '成交额': 'amount',
            '上涨家数': 'up_num',
            '下跌家数': 'down_num',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 排序
        if rank_by in df.columns:
            df = df.sort_values(by=rank_by, ascending=False)
        
        # 取前N条
        df = df.head(top_n)
        
        # 添加排名
        df['rank'] = range(1, len(df) + 1)
        df['sector_type'] = sector_type
        df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("limit_up_pool")
class LimitUpPoolCollector(BaseCollector):
    """
    涨停板池采集器
    
    采集当日涨停股票列表
    主数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 股票代码
        'name',                 # 股票名称
        'trade_date',           # 交易日期
        'close',                # 收盘价
        'pct_change',           # 涨跌幅
        'limit_up_time',        # 首次涨停时间
        'open_times',           # 打开次数
        'industry',             # 所属行业
        # 'concept',            # 相关概念（AkShare接口不返回）
        # 'reason',             # 涨停原因（AkShare接口不返回）
        'amount',               # 成交额
        'float_mv',             # 流通市值
    ]
    
    def collect(
        self,
        trade_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集涨停板池数据
        
        Args:
            trade_date: 交易日期（暂不支持历史）
        
        Returns:
            DataFrame: 标准化的涨停板池数据
        """
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条涨停数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取涨停数据失败: {e}")
        
        logger.error("无法获取涨停数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取涨停板池"""
        import akshare as ak
        
        try:
            df = ak.stock_zt_pool_em(date=datetime.now().strftime('%Y%m%d'))
        except Exception as e:
            logger.warning(f"AkShare获取涨停板池失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '代码': 'ts_code',
            '名称': 'name',
            '最新价': 'close',
            '涨跌幅': 'pct_change',
            '首次封板时间': 'limit_up_time',
            '炸板次数': 'open_times',
            '所属行业': 'industry',
            '成交额': 'amount',
            '流通市值': 'float_mv',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 设置日期
        df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # 格式化股票代码
        if 'ts_code' in df.columns:
            def format_code(code):
                code = str(code)
                if code.startswith('6'):
                    return code + '.SH'
                elif code.startswith(('0', '3')):
                    return code + '.SZ'
                elif code.startswith(('4', '8')):
                    return code + '.BJ'
                return code
            df['ts_code'] = df['ts_code'].apply(format_code)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_sector_performance(
    sector_type: str = 'industry',
    trade_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取板块涨跌幅排行
    
    Args:
        sector_type: 板块类型（industry-行业, concept-概念）
        trade_date: 交易日期
    
    Returns:
        DataFrame: 板块涨跌幅排行数据
    
    Example:
        >>> df = get_sector_performance(sector_type='industry')
        >>> df = get_sector_performance(sector_type='concept')
    """
    collector = SectorPerformanceCollector()
    return collector.collect(sector_type=sector_type, trade_date=trade_date)


def get_industry_board_em() -> pd.DataFrame:
    """
    获取东方财富行业板块行情
    
    Returns:
        DataFrame: 行业板块行情数据
    
    Example:
        >>> df = get_industry_board_em()
    """
    collector = IndustryBoardEMCollector()
    return collector.collect()


def get_concept_board_em() -> pd.DataFrame:
    """
    获取东方财富概念板块行情
    
    Returns:
        DataFrame: 概念板块行情数据
    
    Example:
        >>> df = get_concept_board_em()
    """
    collector = ConceptBoardEMCollector()
    return collector.collect()


def get_sector_hist(
    sector_name: str,
    sector_type: str = 'concept',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取板块历史行情
    
    Args:
        sector_name: 板块名称
        sector_type: 板块类型（concept-概念, industry-行业）
        start_date: 开始日期（YYYYMMDD）
        end_date: 结束日期
    
    Returns:
        DataFrame: 板块历史行情数据
    
    Example:
        >>> df = get_sector_hist('人工智能', 'concept')
        >>> df = get_sector_hist('银行', 'industry', start_date='20250101')
    """
    collector = SectorHistCollector()
    return collector.collect(sector_name=sector_name, sector_type=sector_type, 
                            start_date=start_date, end_date=end_date)


def get_sector_rank(
    sector_type: str = 'concept',
    rank_by: str = 'pct_change',
    top_n: int = 50
) -> pd.DataFrame:
    """
    获取板块热度排行
    
    Args:
        sector_type: 板块类型（concept-概念, industry-行业）
        rank_by: 排序字段（pct_change-涨跌幅, turnover_rate-换手率）
        top_n: 返回前N条
    
    Returns:
        DataFrame: 板块热度排行数据
    
    Example:
        >>> df = get_sector_rank(sector_type='concept', rank_by='pct_change', top_n=20)
    """
    collector = SectorRankCollector()
    return collector.collect(sector_type=sector_type, rank_by=rank_by, top_n=top_n)


def get_limit_up_pool(trade_date: Optional[str] = None) -> pd.DataFrame:
    """
    获取涨停板池
    
    Args:
        trade_date: 交易日期
    
    Returns:
        DataFrame: 涨停板池数据
    
    Example:
        >>> df = get_limit_up_pool()
    """
    collector = LimitUpPoolCollector()
    return collector.collect(trade_date=trade_date)
