"""
中国宏观数据（China Macro Data）采集模块

数据类型包括：
- GDP（国内生产总值）
- CPI（消费者价格指数）
- PPI（工业生产者出厂价格指数）
- PMI（采购经理指数）
- 货币供应量（M0/M1/M2）
- 社会融资规模
- 利率（Shibor/LPR）
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


@CollectorRegistry.register("cn_gdp")
class ChinaGDPCollector(BaseCollector):
    """
    中国GDP数据采集器
    
    采集中国GDP季度数据
    主数据源：Tushare (cn_gdp) - 需120积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'quarter',              # 季度
        'gdp',                  # 国内生产总值（亿元）
        'gdp_yoy',              # GDP同比增速（%）
        'pi',                   # 第一产业增加值（亿元）
        'pi_yoy',               # 第一产业同比增速
        'si',                   # 第二产业增加值（亿元）
        'si_yoy',               # 第二产业同比增速
        'ti',                   # 第三产业增加值（亿元）
        'ti_yoy',               # 第三产业同比增速
    ]
    
    def collect(
        self,
        start_q: Optional[str] = None,
        end_q: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集中国GDP数据
        
        Args:
            start_q: 开始季度（如2024Q1）
            end_q: 结束季度
        
        Returns:
            DataFrame: 标准化的GDP数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(start_q, end_q)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条GDP数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取GDP失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条GDP数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取GDP失败: {e}")
        
        logger.error("所有数据源均无法获取GDP数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        start_q: Optional[str],
        end_q: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取GDP"""
        pro = self.tushare_api
        
        params = {}
        if start_q:
            params['start_q'] = start_q
        if end_q:
            params['end_q'] = end_q
        
        df = pro.cn_gdp(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取GDP"""
        import akshare as ak
        
        try:
            df = ak.macro_china_gdp()
        except Exception as e:
            logger.warning(f"AkShare获取GDP失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '季度': 'quarter',
            '国内生产总值-绝对值': 'gdp',
            '国内生产总值-同比增长': 'gdp_yoy',
            '第一产业-绝对值': 'pi',
            '第一产业-同比增长': 'pi_yoy',
            '第二产业-绝对值': 'si',
            '第二产业-同比增长': 'si_yoy',
            '第三产业-绝对值': 'ti',
            '第三产业-同比增长': 'ti_yoy',
        }
        df = self._standardize_columns(df, column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("cn_cpi")
class ChinaCPICollector(BaseCollector):
    """
    中国CPI数据采集器
    
    采集中国CPI月度数据
    主数据源：Tushare (cn_cpi) - 需120积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'month',                # 月份
        'nt_yoy',               # 全国当月同比（%）
        'nt_mom',               # 全国环比
        'nt_accu',              # 全国累计同比
        'town_yoy',             # 城市当月同比
        'town_mom',             # 城市环比
        'town_accu',            # 城市累计同比
        'cnt_yoy',              # 农村当月同比
        'cnt_mom',              # 农村环比
        'cnt_accu',             # 农村累计同比
    ]
    
    def collect(
        self,
        start_m: Optional[str] = None,
        end_m: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集中国CPI数据
        
        Args:
            start_m: 开始月份（如202401）
            end_m: 结束月份
        
        Returns:
            DataFrame: 标准化的CPI数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(start_m, end_m)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条CPI数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取CPI失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条CPI数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取CPI失败: {e}")
        
        logger.error("所有数据源均无法获取CPI数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        start_m: Optional[str],
        end_m: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取CPI"""
        pro = self.tushare_api
        
        params = {}
        if start_m:
            params['start_m'] = start_m
        if end_m:
            params['end_m'] = end_m
        
        df = pro.cn_cpi(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取CPI"""
        import akshare as ak
        
        try:
            df = ak.macro_china_cpi_monthly()
        except Exception as e:
            logger.warning(f"AkShare获取CPI失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '月份': 'month',
            '全国-当月': 'nt_yoy',
            '全国-同比增长': 'nt_yoy',
            '全国-环比增长': 'nt_mom',
            '全国-累计': 'nt_accu',
        }
        df = self._standardize_columns(df, column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("cn_ppi")
class ChinaPPICollector(BaseCollector):
    """
    中国PPI数据采集器
    
    采集中国PPI月度数据
    主数据源：Tushare (cn_ppi) - 需120积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'month',                # 月份
        'ppi_yoy',              # PPI当月同比（%）
        'ppi_mom',              # PPI环比
        'ppi_accu',             # PPI累计同比
        'ppi_mp_yoy',           # 生产资料同比
        'ppi_mp_mom',           # 生产资料环比
        'ppi_mp_accu',          # 生产资料累计同比
        'ppi_cg_yoy',           # 生活资料同比
        'ppi_cg_mom',           # 生活资料环比
        'ppi_cg_accu',          # 生活资料累计同比
    ]
    
    def collect(
        self,
        start_m: Optional[str] = None,
        end_m: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集中国PPI数据
        
        Args:
            start_m: 开始月份
            end_m: 结束月份
        
        Returns:
            DataFrame: 标准化的PPI数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(start_m, end_m)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条PPI数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取PPI失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条PPI数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取PPI失败: {e}")
        
        logger.error("所有数据源均无法获取PPI数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        start_m: Optional[str],
        end_m: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取PPI"""
        pro = self.tushare_api
        
        params = {}
        if start_m:
            params['start_m'] = start_m
        if end_m:
            params['end_m'] = end_m
        
        df = pro.cn_ppi(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取PPI"""
        import akshare as ak
        
        try:
            df = ak.macro_china_ppi()
        except Exception as e:
            logger.warning(f"AkShare获取PPI失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '月份': 'month',
            '当月同比': 'ppi_yoy',
            '当月环比': 'ppi_mom',
            '累计同比': 'ppi_accu',
        }
        df = self._standardize_columns(df, column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("cn_pmi")
class ChinaPMICollector(BaseCollector):
    """
    中国PMI数据采集器
    
    采集中国PMI月度数据（官方制造业PMI）
    主数据源：Tushare (cn_pmi) - 需120积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'month',                # 月份
        'pmi',                  # 制造业PMI
        'pmi_prod',             # 生产指数
        'pmi_new_order',        # 新订单指数
        'pmi_employ',           # 从业人员指数
    ]
    
    def collect(
        self,
        start_m: Optional[str] = None,
        end_m: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集中国PMI数据
        
        Args:
            start_m: 开始月份
            end_m: 结束月份
        
        Returns:
            DataFrame: 标准化的PMI数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(start_m, end_m)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条PMI数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取PMI失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条PMI数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取PMI失败: {e}")
        
        logger.error("所有数据源均无法获取PMI数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        start_m: Optional[str],
        end_m: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取PMI"""
        pro = self.tushare_api
        
        params = {}
        if start_m:
            params['start_m'] = start_m
        if end_m:
            params['end_m'] = end_m
        
        df = pro.cn_pmi(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # Tushare字段映射（注意：Tushare返回的列名是大写）
        column_mapping = {
            'MONTH': 'month',
            'PMI010000': 'pmi',          # 制造业PMI
            'PMI010400': 'pmi_prod',     # 生产指数
            'PMI010500': 'pmi_new_order', # 新订单指数
            'PMI010800': 'pmi_employ',   # 从业人员指数
        }
        df = df.rename(columns=column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取PMI"""
        import akshare as ak
        
        try:
            df = ak.macro_china_pmi()
        except Exception as e:
            logger.warning(f"AkShare获取PMI失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 直接重命名列
        df = df.rename(columns={
            '月份': 'month',
            '制造业-指数': 'pmi',
            '制造业-同比增长': 'pmi_yoy',
            '非制造业-指数': 'pmi_non',
            '非制造业-同比增长': 'pmi_non_yoy',
        })
        
        # 确保所有输出列存在
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("cn_m2")
class ChinaMoneySupplyCollector(BaseCollector):
    """
    中国货币供应量采集器
    
    采集中国M0/M1/M2月度数据
    主数据源：Tushare (cn_m) - 需800积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'month',                # 月份
        'm0',                   # M0（亿元）
        'm0_yoy',               # M0同比（%）
        'm0_mom',               # M0环比
        'm1',                   # M1（亿元）
        'm1_yoy',               # M1同比
        'm1_mom',               # M1环比
        'm2',                   # M2（亿元）
        'm2_yoy',               # M2同比
        'm2_mom',               # M2环比
    ]
    
    def collect(
        self,
        start_m: Optional[str] = None,
        end_m: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集中国货币供应量数据
        
        Args:
            start_m: 开始月份
            end_m: 结束月份
        
        Returns:
            DataFrame: 标准化的货币供应量数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(start_m, end_m)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条货币供应数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取货币供应失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条货币供应数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取货币供应失败: {e}")
        
        logger.error("所有数据源均无法获取货币供应数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        start_m: Optional[str],
        end_m: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取货币供应量"""
        pro = self.tushare_api
        
        params = {}
        if start_m:
            params['start_m'] = start_m
        if end_m:
            params['end_m'] = end_m
        
        df = pro.cn_m(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取货币供应量"""
        import akshare as ak
        
        try:
            df = ak.macro_china_money_supply()
        except Exception as e:
            logger.warning(f"AkShare获取货币供应失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '月份': 'month',
            '货币和准货币(M2)-数量': 'm2',
            '货币和准货币(M2)-同比增长': 'm2_yoy',
            '货币(M1)-数量': 'm1',
            '货币(M1)-同比增长': 'm1_yoy',
            '流通中的现金(M0)-数量': 'm0',
            '流通中的现金(M0)-同比增长': 'm0_yoy',
        }
        df = self._standardize_columns(df, column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("shibor")
class ShiborCollector(BaseCollector):
    """
    Shibor利率采集器
    
    采集上海银行间同业拆放利率
    主数据源：Tushare (shibor) - 需500积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'date',                 # 日期
        'on',                   # 隔夜（%）
        '1w',                   # 1周
        '2w',                   # 2周
        '1m',                   # 1月
        '3m',                   # 3月
        '6m',                   # 6月
        '9m',                   # 9月
        '1y',                   # 1年
    ]
    
    def collect(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集Shibor利率数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的Shibor利率数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条Shibor数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取Shibor失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条Shibor数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取Shibor失败: {e}")
        
        logger.error("所有数据源均无法获取Shibor数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取Shibor"""
        pro = self.tushare_api
        
        params = {}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = pro.shibor(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取Shibor"""
        import akshare as ak
        
        try:
            df = ak.rate_interbank(market='上海银行间同业拆放利率')
        except Exception as e:
            logger.warning(f"AkShare获取Shibor失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '报告日': 'date',
            'O/N': 'on',
            '1W': '1w',
            '2W': '2w',
            '1M': '1m',
            '3M': '3m',
            '6M': '6m',
            '9M': '9m',
            '1Y': '1y',
        }
        df = self._standardize_columns(df, column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("lpr")
class LPRCollector(BaseCollector):
    """
    LPR利率采集器
    
    采集贷款市场报价利率
    主数据源：Tushare (shibor_lpr) - 需500积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'date',                 # 日期
        'lpr_1y',               # 1年期LPR（%）
        'lpr_5y',               # 5年期LPR（%）
    ]
    
    def collect(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集LPR利率数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的LPR利率数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条LPR数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取LPR失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条LPR数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取LPR失败: {e}")
        
        logger.error("所有数据源均无法获取LPR数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取LPR"""
        pro = self.tushare_api
        
        params = {}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = pro.shibor_lpr(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # Tushare字段映射
        column_mapping = {
            '1y': 'lpr_1y',
            '5y': 'lpr_5y',
        }
        df = df.rename(columns=column_mapping)
        
        df = self._convert_date_format(df, ['date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取LPR"""
        import akshare as ak
        
        try:
            df = ak.macro_china_lpr()
        except Exception as e:
            logger.warning(f"AkShare获取LPR失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 直接重命名列
        df = df.rename(columns={
            'TRADE_DATE': 'date',
            'LPR1Y': 'lpr_1y',
            'LPR5Y': 'lpr_5y',
        })
        
        # 过滤掉LPR为空的行
        df = df.dropna(subset=['lpr_1y'], how='all')
        
        # 只保留需要的列
        available_cols = [col for col in self.OUTPUT_FIELDS if col in df.columns]
        return df[available_cols]


@CollectorRegistry.register("sf")
class SocialFinanceCollector(BaseCollector):
    """
    社会融资规模采集器
    
    采集社会融资规模月度数据
    主数据源：Tushare (sf) - 需2000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'month',                # 月份
        'total',                # 社融规模合计（亿元）
    ]
    
    def collect(
        self,
        start_m: Optional[str] = None,
        end_m: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集社会融资规模数据
        
        Args:
            start_m: 开始月份
            end_m: 结束月份
        
        Returns:
            DataFrame: 标准化的社融规模数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(start_m, end_m)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条社融数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取社融失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条社融数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取社融失败: {e}")
        
        logger.error("所有数据源均无法获取社融数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        start_m: Optional[str],
        end_m: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取社融"""
        pro = self.tushare_api
        
        params = {}
        if start_m:
            params['start_m'] = start_m
        if end_m:
            params['end_m'] = end_m
        
        df = pro.sf(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取社融"""
        import akshare as ak
        
        try:
            df = ak.macro_china_shrzgm()
        except Exception as e:
            logger.warning(f"AkShare获取社融失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 直接重命名列
        df = df.rename(columns={
            '月份': 'month',
            '社会融资规模增量': 'total',
        })
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_cn_gdp(
    start_q: Optional[str] = None,
    end_q: Optional[str] = None
) -> pd.DataFrame:
    """获取中国GDP数据"""
    collector = ChinaGDPCollector()
    return collector.collect(start_q=start_q, end_q=end_q)


def get_cn_cpi(
    start_m: Optional[str] = None,
    end_m: Optional[str] = None
) -> pd.DataFrame:
    """获取中国CPI数据"""
    collector = ChinaCPICollector()
    return collector.collect(start_m=start_m, end_m=end_m)


def get_cn_ppi(
    start_m: Optional[str] = None,
    end_m: Optional[str] = None
) -> pd.DataFrame:
    """获取中国PPI数据"""
    collector = ChinaPPICollector()
    return collector.collect(start_m=start_m, end_m=end_m)


def get_cn_pmi(
    start_m: Optional[str] = None,
    end_m: Optional[str] = None
) -> pd.DataFrame:
    """获取中国PMI数据"""
    collector = ChinaPMICollector()
    return collector.collect(start_m=start_m, end_m=end_m)


def get_cn_m2(
    start_m: Optional[str] = None,
    end_m: Optional[str] = None
) -> pd.DataFrame:
    """获取中国货币供应量数据"""
    collector = ChinaMoneySupplyCollector()
    return collector.collect(start_m=start_m, end_m=end_m)


def get_shibor(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """获取Shibor利率数据"""
    collector = ShiborCollector()
    return collector.collect(start_date=start_date, end_date=end_date)


def get_lpr(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """获取LPR利率数据"""
    collector = LPRCollector()
    return collector.collect(start_date=start_date, end_date=end_date)


def get_sf(
    start_m: Optional[str] = None,
    end_m: Optional[str] = None
) -> pd.DataFrame:
    """获取社会融资规模数据"""
    collector = SocialFinanceCollector()
    return collector.collect(start_m=start_m, end_m=end_m)
