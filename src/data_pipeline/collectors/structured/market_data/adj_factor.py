"""
复权因子（Adjustment Factor）采集模块

数据类型：
- 股票复权因子（用于计算股票复权价格）

数据源：Tushare (adj_factor)

复权因子说明：
- 复权因子用于将不复权的价格转换为前复权或后复权价格
- 前复权价格 = 不复权价格 * 当日复权因子 / 最新复权因子
- 后复权价格 = 不复权价格 * 当日复权因子

Tushare接口说明：
- 接口：adj_factor
- 积分要求：2000积分起，5000以上可高频调取
- 更新时间：盘前9点15~20分完成当日复权因子入库
"""

import logging
import time
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


@CollectorRegistry.register("adj_factor")
class AdjFactorCollector(BaseCollector):
    """
    股票复权因子采集器
    
    采集股票复权因子数据，用于计算前复权和后复权价格。
    主数据源：Tushare (adj_factor, 2000积分)
    
    支持两种采集模式：
    1. 按股票代码采集：获取单只股票全部历史复权因子
    2. 按交易日期采集：获取某日全市场股票的复权因子
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 股票代码
        'trade_date',       # 交易日期
        'adj_factor',       # 复权因子
    ]
    
    # API调用频率控制（单位：秒）
    # Tushare adj_factor接口限制：5000积分以上可高频调取
    # 为了安全起见，设置适当的延迟
    API_CALL_INTERVAL = 0.3  # 每次调用间隔0.3秒
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股票复权因子数据
        
        Args:
            ts_code: 股票代码（如 000001.SZ）。可选，不填则按日期采集全市场
            trade_date: 交易日期（YYYYMMDD格式）。用于采集某日全市场数据
            start_date: 开始日期（YYYYMMDD格式）。与ts_code一起使用
            end_date: 结束日期（YYYYMMDD格式）。与ts_code一起使用
        
        Returns:
            DataFrame: 标准化的复权因子数据
            
        输出字段:
            - ts_code: 股票代码
            - trade_date: 交易日期
            - adj_factor: 复权因子
        
        使用示例:
            # 获取单只股票全部历史复权因子
            df = collector.collect(ts_code='000001.SZ')
            
            # 获取单只股票指定日期范围的复权因子
            df = collector.collect(ts_code='000001.SZ', start_date='20210101', end_date='20251231')
            
            # 获取某日全市场复权因子
            df = collector.collect(trade_date='20240115')
        """
        # 优先使用Tushare（唯一数据源）
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条复权因子数据")
                return df
        except Exception as e:
            logger.error(f"Tushare获取复权因子失败: {e}")
        
        logger.error("无法获取复权因子数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取股票复权因子
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 复权因子数据
        """
        pro = self.tushare_api
        
        # 构建请求参数
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        # 调用API
        df = pro.adj_factor(**params)
        
        if df is None or df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有输出字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        # 按日期排序（升序）
        df = df.sort_values('trade_date', ascending=True)
        
        return df[self.OUTPUT_FIELDS]
    
    def collect_batch_by_date(
        self,
        dates: List[str],
        delay: float = None
    ) -> pd.DataFrame:
        """批量按日期采集全市场复权因子
        
        Args:
            dates: 交易日期列表（YYYYMMDD格式）
            delay: 每次调用间隔（秒），默认使用API_CALL_INTERVAL
        
        Returns:
            DataFrame: 合并后的复权因子数据
        """
        if delay is None:
            delay = self.API_CALL_INTERVAL
        
        all_data = []
        for i, date in enumerate(dates):
            try:
                df = self.collect(trade_date=date)
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"[{i+1}/{len(dates)}] 采集日期 {date} 复权因子: {len(df)} 条")
            except Exception as e:
                logger.warning(f"采集日期 {date} 复权因子失败: {e}")
            
            # 频率控制
            if i < len(dates) - 1:
                time.sleep(delay)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # 去重（同一股票同一日期只保留一条）
            result = result.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
            return result.sort_values(['ts_code', 'trade_date'])
        
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def collect_batch_by_stock(
        self,
        ts_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        delay: float = None
    ) -> pd.DataFrame:
        """批量按股票代码采集复权因子
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期（YYYYMMDD格式）
            end_date: 结束日期（YYYYMMDD格式）
            delay: 每次调用间隔（秒），默认使用API_CALL_INTERVAL
        
        Returns:
            DataFrame: 合并后的复权因子数据
        """
        if delay is None:
            delay = self.API_CALL_INTERVAL
        
        all_data = []
        for i, ts_code in enumerate(ts_codes):
            try:
                df = self.collect(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"[{i+1}/{len(ts_codes)}] 采集股票 {ts_code} 复权因子: {len(df)} 条")
            except Exception as e:
                logger.warning(f"采集股票 {ts_code} 复权因子失败: {e}")
            
            # 频率控制
            if i < len(ts_codes) - 1:
                time.sleep(delay)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            return result.sort_values(['ts_code', 'trade_date'])
        
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)


# 便捷函数
def get_adj_factor(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取股票复权因子
    
    Args:
        ts_code: 股票代码（如 000001.SZ）
        trade_date: 交易日期（YYYYMMDD格式）
        start_date: 开始日期（YYYYMMDD格式）
        end_date: 结束日期（YYYYMMDD格式）
    
    Returns:
        DataFrame: 复权因子数据
        
    示例:
        # 获取单只股票全部历史复权因子
        df = get_adj_factor(ts_code='000001.SZ')
        
        # 获取单只股票指定日期范围的复权因子
        df = get_adj_factor(ts_code='000001.SZ', start_date='20210101', end_date='20251231')
        
        # 获取某日全市场复权因子
        df = get_adj_factor(trade_date='20240115')
    """
    collector = AdjFactorCollector()
    return collector.collect(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
