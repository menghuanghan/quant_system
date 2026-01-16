"""
ESG评价（ESG Ratings）采集模块

数据类型包括：
- MSCI ESG评级
- 华证指数 ESG评级
- 路孚特 ESG评级
- 秩鼎 ESG评级
"""

import logging
from typing import Optional
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


@CollectorRegistry.register("esg_msci")
class MSCIESGCollector(BaseCollector):
    """
    MSCI ESG评级采集器
    
    获取MSCI提供的ESG评级数据
    主数据源：AkShare (stock_esg_msci_em)
    备用数据源：无
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 股票代码
        'rating_date',          # 评级日期
        'esg_rating',           # ESG评分
        'env_score',            # 环境总评
        'social_score',         # 社会责任总评
        'governance_score',     # 治理总评
        'market',               # 交易市场
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集MSCI ESG评级数据
        
        Args:
            ts_code: 股票代码 (可选)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: MSCI ESG评级数据
        """
        manager = DataSourceManager()
        df = pd.DataFrame()
        
        try:
            df = self._collect_from_akshare(ts_code, **kwargs)
            
            if not df.empty:
                logger.info(f"成功从 AkShare 采集 MSCI ESG 评级数据: {len(df)} 条")
        except Exception as e:
            logger.error(f"从 AkShare 采集 MSCI ESG 评级数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            logger.warning("未能采集到 MSCI ESG 评级数据")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有必需字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        ts_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare接口：stock_esg_msci_sina (MSCI ESG评级)
        df = ak.stock_esg_msci_sina()
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名 - AkShare实际返回: ['股票代码', 'ESG评分', '环境总评', '社会责任总评', '治理总评', '评级日期', '交易市场']
        column_mapping = {
            '股票代码': 'ts_code',
            'ESG评分': 'esg_rating',
            '环境总评': 'environment_score',
            '社会责任总评': 'social_score',
            '治理总评': 'governance_score',
            '评级日期': 'rating_date',
            '交易市场': 'market',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 股票代码过滤
        if ts_code:
            code = ts_code.split('.')[0]
            df = df[df['ts_code'] == code]
        
        return df


@CollectorRegistry.register("esg_hz")
class HZESGCollector(BaseCollector):
    """
    华证指数 ESG评级采集器
    
    获取华证指数提供的ESG评级数据
    主数据源：AkShare (stock_esg_hz_em)
    备用数据源：无
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 股票代码
        'stock_name',           # 股票名称
        'date',                 # 日期
        'esg_score',            # ESG评分
        'esg_grade',            # ESG等级
        'e_score',              # 环境分数
        'e_grade',              # 环境等级
        's_score',              # 社会分数
        's_grade',              # 社会等级
        'g_score',              # 公司治理分数
        'g_grade',              # 公司治理等级
        'market',               # 交易市场
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集华证指数 ESG评级数据
        
        Args:
            ts_code: 股票代码 (可选)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 华证指数 ESG评级数据
        """
        manager = DataSourceManager()
        df = pd.DataFrame()
        
        try:
            df = self._collect_from_akshare(ts_code, **kwargs)
            
            if not df.empty:
                logger.info(f"成功从 AkShare 采集华证指数 ESG 评级数据: {len(df)} 条")
        except Exception as e:
            logger.error(f"从 AkShare 采集华证指数 ESG 评级数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            logger.warning("未能采集到华证指数 ESG 评级数据")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有必需字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        ts_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare接口：stock_esg_hz_sina (华证指数 ESG评级)
        df = ak.stock_esg_hz_sina()
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名 - AkShare实际返回: ['日期', '股票代码', '交易市场', '股票名称', 'ESG评分', 'ESG等级', '环境', '环境等级', '社会', '社会等级', '公司治理', '公司治理等级']
        column_mapping = {
            '股票代码': 'ts_code',
            '股票名称': 'stock_name',
            '日期': 'date',
            'ESG评分': 'esg_score',
            'ESG等级': 'esg_grade',
            '环境': 'e_score',
            '环境等级': 'e_grade',
            '社会': 's_score',
            '社会等级': 's_grade',
            '公司治理': 'g_score',
            '公司治理等级': 'g_grade',
            '交易市场': 'market',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 股票代码过滤
        if ts_code:
            code = ts_code.split('.')[0]
            df = df[df['ts_code'] == code]
        
        return df


@CollectorRegistry.register("esg_refinitiv")
class RefinitivESGCollector(BaseCollector):
    """
    路孚特(Refinitiv) ESG评级采集器
    
    获取路孚特提供的ESG评级数据
    主数据源：AkShare (stock_esg_lft_em)
    备用数据源：无
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 股票代码
        'stock_name',           # 股票名称
        'date',                 # 日期
        'esg_score',            # ESG评分
        'esg_grade',            # ESG等级
        'e_score',              # 环境分数
        'e_grade',              # 环境等级
        's_score',              # 社会分数
        's_grade',              # 社会等级
        'g_score',              # 公司治理分数
        'g_grade',              # 公司治理等级
        'market',               # 交易市场
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集路孚特 ESG评级数据
        
        Args:
            ts_code: 股票代码 (可选)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 路孚特 ESG评级数据
        """
        manager = DataSourceManager()
        df = pd.DataFrame()
        
        try:
            df = self._collect_from_akshare(ts_code, **kwargs)
            
            if not df.empty:
                logger.info(f"成功从 AkShare 采集路孚特 ESG 评级数据: {len(df)} 条")
        except Exception as e:
            logger.error(f"从 AkShare 采集路孚特 ESG 评级数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            logger.warning("未能采集到路孚特 ESG 评级数据")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有必需字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        ts_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare接口：stock_esg_rft_sina (路孚特 ESG评级)
        df = ak.stock_esg_rft_sina()
        
        if df.empty:
            return pd.DataFrame()
        
        # AkShare实际返回: ['日期', '股票代码', '交易市场', '股票名称', 'ESG评分', 'ESG等级', '环境', '环境等级', '社会', '社会等级', '公司治理', '公司治理等级']
        column_mapping = {
            '股票代码': 'ts_code',
            '股票名称': 'stock_name',
            '日期': 'date',
            'ESG评分': 'esg_score',
            'ESG等级': 'esg_grade',
            '环境': 'e_score',
            '环境等级': 'e_grade',
            '社会': 's_score',
            '社会等级': 's_grade',
            '公司治理': 'g_score',
            '公司治理等级': 'g_grade',
            '交易市场': 'market',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 股票代码过滤
        if ts_code:
            code = ts_code.split('.')[0]
            if 'ts_code' in df.columns:
                df = df[df['ts_code'].astype(str) == code]
        
        return df


@CollectorRegistry.register("esg_zhiding")
class ZhidingESGCollector(BaseCollector):
    """
    秩鼎 ESG评级采集器
    
    获取秩鼎提供的ESG评级数据
    主数据源：AkShare (stock_esg_zd_em)
    备用数据源：无
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 股票代码
        'esg_score',            # ESG评分
        'env_score',            # 环境总评
        'social_score',         # 社会责任总评
        'governance_score',     # 治理总评
        'report_date',          # 评分日期
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集秩鼎 ESG评级数据
        
        Args:
            ts_code: 股票代码 (可选)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 秩鼎 ESG评级数据
        """
        manager = DataSourceManager()
        df = pd.DataFrame()
        
        try:
            df = self._collect_from_akshare(ts_code, **kwargs)
            
            if not df.empty:
                logger.info(f"成功从 AkShare 采集秩鼎 ESG 评级数据: {len(df)} 条")
        except Exception as e:
            logger.error(f"从 AkShare 采集秩鼎 ESG 评级数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            logger.warning("未能采集到秩鼎 ESG 评级数据")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有必需字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        ts_code: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare接口：stock_esg_zd_sina
        df = ak.stock_esg_zd_sina()
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名 - AkShare已经返回中文列名
        # ['股票代码', 'ESG评分', '环境总评', '社会责任总评', '治理总评', '评分日期']
        column_mapping = {
            '股票代码': 'ts_code',
            'ESG评分': 'esg_score',
            '环境总评': 'env_score',
            '社会责任总评': 'social_score',
            '治理总评': 'governance_score',
            '评分日期': 'report_date',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 股票代码过滤
        if ts_code:
            code = ts_code.split('.')[0]
            df = df[df['ts_code'] == code]
        
        return df


# 便捷函数
def get_esg_msci(
    ts_code: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取MSCI ESG评级数据
    
    Args:
        ts_code: 股票代码 (可选)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: MSCI ESG评级数据
    """
    collector = MSCIESGCollector()
    return collector.collect(ts_code=ts_code, **kwargs)


def get_esg_hz(
    ts_code: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取华证指数 ESG评级数据
    
    Args:
        ts_code: 股票代码 (可选)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 华证指数 ESG评级数据
    """
    collector = HZESGCollector()
    return collector.collect(ts_code=ts_code, **kwargs)


def get_esg_refinitiv(
    ts_code: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取路孚特 ESG评级数据
    
    Args:
        ts_code: 股票代码 (可选)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 路孚特 ESG评级数据
    """
    collector = RefinitivESGCollector()
    return collector.collect(ts_code=ts_code, **kwargs)


def get_esg_zhiding(
    ts_code: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取秩鼎 ESG评级数据
    
    Args:
        ts_code: 股票代码 (可选)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 秩鼎 ESG评级数据
    """
    collector = ZhidingESGCollector()
    return collector.collect(ts_code=ts_code, **kwargs)
