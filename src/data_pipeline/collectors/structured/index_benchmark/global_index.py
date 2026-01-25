"""
全球指数数据（Global Index Data）采集模块

数据类型包括：
- 全球主要股票指数行情
- 主要经济体指数（美股、欧股、亚太）
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


@CollectorRegistry.register("index_global")
class GlobalIndexCollector(BaseCollector):
    """
    全球股票指数采集器
    
    采集全球主要股票指数行情
    主数据源：Tushare (index_global) - 需2000积分
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',           # 交易日期
        'ts_code',              # 代码
        'open',                 # 开盘价
        'close',                # 收盘价
        'high',                 # 最高价
        'low',                  # 最低价
        'pre_close',            # 昨收价
        'change',               # 涨跌额
        'pct_chg',              # 涨跌幅
        'swing',                # 振幅
        'vol',                  # 成交量
        'amount',               # 成交额
    ]
    
    # 默认采集的主要全球指数
    DEFAULT_INDICES = [
        'SPX.GI',   # 标普500
        'DJI.GI',   # 道琼斯
        'IXIC.GI',  # 纳斯达克
        'HSI.HI',   # 恒生指数
        'N225.GI',  # 日经225
        'FTSE.GI',  # 英国富时100
        'GDAXI.GI', # 德国DAX
        'FCHI.GI',  # 法国CAC40
        'KS11.GI',  # 韩国KOSPI
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
        采集全球指数行情数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条全球指数数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取全球指数失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code, start_date, end_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条全球指数数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取全球指数失败: {e}")
        
        logger.error("所有数据源均无法获取全球指数数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取全球指数"""
        pro = self.tushare_api
        
        codes = [ts_code] if ts_code else self.DEFAULT_INDICES
        
        all_dfs = []
        for code in codes:
            params = {'ts_code': code}
            if trade_date:
                params['trade_date'] = trade_date
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            try:
                # 显式加延时防限流
                import time
                time.sleep(0.1) 
                
                df = pro.index_global(**params)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logger.debug(f"获取指数 {code} 失败: {e}")
        
        if not all_dfs:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
        df = pd.concat(all_dfs, ignore_index=True)
        # 转换日期格式
        df = self._convert_date_format(df, ['trade_date'])
        
        # 自动补全缺失字段
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
        """从AkShare获取全球指数"""
        import akshare as ak
        
        try:
            # 获取主要国际指数实时行情
            df = ak.index_stock_info()
        except Exception as e:
            logger.warning(f"AkShare获取全球指数失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '代码': 'ts_code',
            '名称': 'name',
            '最新价': 'close',
            '涨跌幅': 'pct_chg',
        }
        df = self._standardize_columns(df, column_mapping)
        
        df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_index_global(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取全球指数行情
    
    Args:
        ts_code: 指数代码
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 全球指数行情数据
    
    Example:
        >>> df = get_index_global(ts_code='SPX.GI', start_date='20250101')  # 标普500
        >>> df = get_index_global(ts_code='IXIC.GI')  # 纳斯达克
    """
    collector = GlobalIndexCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date)
