"""
行业与现实经济映射数据（Industry Economy Data）采集模块

数据类型包括：
- 电影票房
- 汽车销量
- 其他行业高频数据（能源/生猪/物流等）
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


@CollectorRegistry.register("box_office")
class BoxOfficeCollector(BaseCollector):
    """
    电影票房数据采集器
    
    采集中国电影票房数据
    主数据源：Tushare (bo_cinema/bo_daily/bo_weekly) - 需1000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'date',                 # 日期
        'movie',                # 电影名称
        'box_office',           # 票房（万元）
        'box_office_ratio',     # 票房占比
        'avg_price',            # 平均票价
        'rank',                 # 排名
    ]
    
    def collect(
        self,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集电影票房数据
        
        Args:
            date: 日期（YYYYMMDD）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的电影票房数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条票房数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取票房数据失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条票房数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取票房数据失败: {e}")
        
        logger.error("所有数据源均无法获取票房数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取票房数据"""
        pro = self.tushare_api
        
        # bo_daily需要必填参数date，如果没有提供，使用end_date或当前日期
        if not date:
            if end_date:
                date = end_date
            else:
                from datetime import datetime
                date = datetime.now().strftime('%Y%m%d')
        
        params = {'date': date}
        
        # 尝试获取每日票房数据
        df = pro.bo_daily(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # Tushare字段映射
        column_mapping = {
            'name': 'movie',
            'day_amount': 'box_office',    # 当日票房（万元）
            'up_ratio': 'box_office_ratio', # 占比
            'avg_price': 'avg_price',      # 平均票价
        }
        df = df.rename(columns=column_mapping)
        
        df = self._convert_date_format(df, ['date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取票房数据"""
        import akshare as ak
        
        try:
            df = ak.movie_boxoffice_daily()
        except Exception as e:
            logger.warning(f"AkShare获取票房数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '日期': 'date',
            '影片名': 'movie',
            '票房': 'box_office',
            '场次': 'show_num',
            '票价': 'avg_price',
        }
        df = self._standardize_columns(df, column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("car_sales")
class CarSalesCollector(BaseCollector):
    """
    汽车销量数据采集器
    
    采集中国汽车销量月度数据
    主数据源：Tushare (car_sales) - 需2000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'month',                #  月份
        'brand',                # 品牌
        'sales_vol',            # 销量（辆）
        # 'model',              # 车型（数据源不提供）
        # 'sales_yoy',          # 销量同比（数据源不提供）
        # 'sales_mom',          # 销量环比（数据源不提供）
        # 'price_avg',          # 均价（数据源不提供）
        # 'market_share',       # 市场份额（数据源不提供）
    ]
    
    def collect(
        self,
        month: Optional[str] = None,
        brand: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集汽车销量数据
        
        Args:
            month: 月份（YYYYMM）
            brand: 品牌
        
        Returns:
            DataFrame: 标准化的汽车销量数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(month, brand)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条汽车销量数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取汽车销量失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条汽车销量数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取汽车销量失败: {e}")
        
        logger.error("所有数据源均无法获取汽车销量数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        month: Optional[str],
        brand: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取汽车销量"""
        pro = self.tushare_api
        
        params = {}
        if month:
            params['month'] = month
        if brand:
            params['brand'] = brand
        
        # 尝试获取汽车销量
        try:
            df = pro.car_sales(**params)
        except:
            # Tushare可能没有此接口或需更高积分
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取汽车销量"""
        import akshare as ak
        
        try:
            # 尝试获取广义乘用车零售销量数据 (CPCA)
            df = ak.car_market_cate_cpca()
            if not df.empty:
                # 转换格式：将年份列转为行
                years = [col for col in df.columns if '年' in col]
                result = []
                for _, row in df.iterrows():
                    month_str = row['月份'].replace('月', '').zfill(2)
                    for year_col in years:
                        year = year_col.replace('年', '')
                        val = row[year_col]
                        if pd.notna(val):
                            result.append({
                                'month': f"{year}{month_str}",
                                'brand': '乘用车(合计)',
                                'sales_vol': float(val) * 10000 # 原始单位通常是万辆
                            })
                return pd.DataFrame(result)
        except Exception as e:
            logger.warning(f"AkShare获取汽车销量失败: {e}")
        
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)


# ============= 便捷函数接口 =============

def get_box_office(
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取电影票房数据
    
    Args:
        date: 日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 电影票房数据
    
    Example:
        >>> df = get_box_office(date='20250115')
    """
    collector = BoxOfficeCollector()
    return collector.collect(date=date, start_date=start_date, end_date=end_date)


def get_car_sales(
    month: Optional[str] = None,
    brand: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取汽车销量数据
    
    Args:
        month: 月份
        brand: 品牌
        **kwargs: 其他参数（由调度器传入）
    
    Returns:
        DataFrame: 汽车销量数据
    
    Example:
        >>> df = get_car_sales(month='202501')
        >>> df = get_car_sales(brand='比亚迪')
    """
    collector = CarSalesCollector()
    return collector.collect(month=month, brand=brand, **kwargs)
