"""
资金流向（Capital Flow）采集模块

数据类型包括：
- 个股资金流向
- 行业/板块资金流向
- 大盘资金流向
- 沪深港通资金
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta

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


@CollectorRegistry.register("money_flow")
class MoneyFlowCollector(BaseCollector):
    """
    个股资金流向采集器
    
    采集个股每日资金流向数据
    主数据源：Tushare (moneyflow)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'trade_date',           # 交易日期
        'buy_sm_vol',           # 小单买入量（手）
        'buy_sm_amount',        # 小单买入金额（万元）
        'sell_sm_vol',          # 小单卖出量（手）
        'sell_sm_amount',       # 小单卖出金额（万元）
        'buy_md_vol',           # 中单买入量（手）
        'buy_md_amount',        # 中单买入金额（万元）
        'sell_md_vol',          # 中单卖出量（手）
        'sell_md_amount',       # 中单卖出金额（万元）
        'buy_lg_vol',           # 大单买入量（手）
        'buy_lg_amount',        # 大单买入金额（万元）
        'sell_lg_vol',          # 大单卖出量（手）
        'sell_lg_amount',       # 大单卖出金额（万元）
        'buy_elg_vol',          # 特大单买入量（手）
        'buy_elg_amount',       # 特大单买入金额（万元）
        'sell_elg_vol',         # 特大单卖出量（手）
        'sell_elg_amount',      # 特大单卖出金额（万元）
        'net_mf_vol',           # 净流入量（手）
        'net_mf_amount',        # 净流入金额（万元）
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
        采集个股资金流向数据
        
        Args:
            ts_code: 证券代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的资金流向数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条资金流向数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取资金流向失败: {e}")
        
        # 降级到AkShare
        try:
            if ts_code:
                df = self._collect_from_akshare(ts_code)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条资金流向数据")
                    return df
        except Exception as e:
            logger.error(f"AkShare获取资金流向失败: {e}")
        
        logger.error("所有数据源均无法获取资金流向数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取资金流向"""
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
        
        df = pro.moneyflow(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, ts_code: str) -> pd.DataFrame:
        """从AkShare获取资金流向"""
        import akshare as ak
        
        symbol = ts_code.split('.')[0]
        
        try:
            df = ak.stock_individual_fund_flow(stock=symbol, market="sh" if ts_code.endswith('.SH') else "sz")
        except Exception as e:
            logger.warning(f"AkShare获取资金流向失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '日期': 'trade_date',
            '主力净流入-净额': 'net_mf_amount',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        logger.warning("AkShare资金流向数据字段有限")
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("money_flow_industry")
class MoneyFlowIndustryCollector(BaseCollector):
    """
    行业资金流向采集器
    
    采集行业/板块资金流向数据
    主数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',           # 交易日期
        'industry_name',        # 行业名称
        'change_pct',           # 涨跌幅（%）
        'main_net_inflow',      # 主力净流入（亿元）
        'main_net_ratio',       # 主力净流入占比（%）
        'super_net_inflow',     # 超大单净流入（亿元）
        'super_net_ratio',      # 超大单净流入占比（%）
        'big_net_inflow',       # 大单净流入（亿元）
        'big_net_ratio',        # 大单净流入占比（%）
        'mid_net_inflow',       # 中单净流入（亿元）
        'mid_net_ratio',        # 中单净流入占比（%）
        'small_net_inflow',     # 小单净流入（亿元）
        'small_net_ratio',      # 小单净流入占比（%）
    ]
    
    def collect(
        self,
        sector_type: str = 'industry',
        **kwargs
    ) -> pd.DataFrame:
        """
        采集行业资金流向数据
        
        Args:
            sector_type: 板块类型（industry=行业，concept=概念，area=地区）
        
        Returns:
            DataFrame: 标准化的行业资金流向数据
        """
        # 使用AkShare
        try:
            df = self._collect_from_akshare(sector_type)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条行业资金流向数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取行业资金流向失败: {e}")
        
        logger.error("无法获取行业资金流向数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self, sector_type: str) -> pd.DataFrame:
        """从AkShare获取行业资金流向"""
        import akshare as ak
        
        try:
            if sector_type == 'industry':
                df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
            elif sector_type == 'concept':
                df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="概念资金流")
            else:
                df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="地区资金流")
        except Exception as e:
            logger.warning(f"AkShare获取行业资金流向失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '名称': 'industry_name',
            '今日涨跌幅': 'change_pct',
            '今日主力净流入-净额': 'main_net_inflow',
            '今日主力净流入-净占比': 'main_net_ratio',
            '今日超大单净流入-净额': 'super_net_inflow',
            '今日超大单净流入-净占比': 'super_net_ratio',
            '今日大单净流入-净额': 'big_net_inflow',
            '今日大单净流入-净占比': 'big_net_ratio',
            '今日中单净流入-净额': 'mid_net_inflow',
            '今日中单净流入-净占比': 'mid_net_ratio',
            '今日小单净流入-净额': 'small_net_inflow',
            '今日小单净流入-净占比': 'small_net_ratio',
        }
        df = self._standardize_columns(df, column_mapping)
        df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("money_flow_market")
class MoneyFlowMarketCollector(BaseCollector):
    """
    大盘资金流向采集器
    
    采集大盘（市场整体）资金流向数据
    主数据源：Tushare (moneyflow_mkt)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',           # 交易日期
        'main_net',             # 主力净流入（亿元）
        'retail_net',           # 散户净流入（亿元）
    ]
    
    def collect(
        self,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集大盘资金流向数据
        
        Args:
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的大盘资金流向数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条大盘资金流向数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取大盘资金流向失败: {e}")

        # 降级到AkShare（免费）
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条大盘资金流向数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取大盘资金流向失败: {e}")
        
        logger.error("无法获取大盘资金流向数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)

    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取大盘资金流向"""
        pro = self.tushare_api
        
        params = {}
        if trade_date:
            params['trade_date'] = trade_date
            return pro.moneyflow_mkt(**params)
        
        # 分块采集以避免API限制
        if start_date and end_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y%m%d')
                
                all_dfs = []
                current_dt = start_dt
                while current_dt <= end_dt:
                    # Tushare限制约300条。为了稳妥，按月采集。
                    from datetime import timedelta
                    next_month = (current_dt.replace(day=28) + timedelta(days=4)).replace(day=1)
                    chunk_end_dt = min(next_month - timedelta(days=1), end_dt)
                    
                    p = params.copy()
                    p['start_date'] = current_dt.strftime('%Y%m%d')
                    p['end_date'] = chunk_end_dt.strftime('%Y%m%d')
                    
                    try:
                        chunk_df = pro.moneyflow_mkt(**p)
                        if not chunk_df.empty:
                            all_dfs.append(chunk_df)
                            logger.info(f"大盘资金流向分块采集完成: {p['start_date']} - {p['end_date']} ({len(chunk_df)} 条)")
                    except Exception as e:
                        logger.warning(f"大盘资金流向分块采集失败 ({p['start_date']}-{p['end_date']}): {e}")
                    
                    current_dt = chunk_end_dt + timedelta(days=1)
                
                if all_dfs:
                    df = pd.concat(all_dfs, ignore_index=True)
                else:
                    df = pd.DataFrame(columns=self.OUTPUT_FIELDS)
            except Exception as e:
                logger.error(f"大盘资金流向分块逻辑异常: {e}")
                if start_date: params['start_date'] = start_date
                if end_date: params['end_date'] = end_date
                df = pro.moneyflow_mkt(**params)
        else:
             if start_date: params['start_date'] = start_date
             if end_date: params['end_date'] = end_date
             df = pro.moneyflow_mkt(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取大盘资金流向"""
        import akshare as ak
        
        try:
            df = ak.stock_market_fund_flow()
            
            if df.empty:
                logger.warning("AkShare返回空数据")
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            logger.info(f"AkShare返回列名: {df.columns.tolist()}")
            
            # 使用显式的列名重命名
            rename_map = {}
            for col in df.columns:
                if '日期' in col:
                    rename_map[col] = 'trade_date'
                elif '主力净流入' in col and '净额' in col:
                    rename_map[col] = 'main_net'
                elif '小单净流入' in col and '净额' in col:
                    rename_map[col] = 'retail_net'
            
            df = df.rename(columns=rename_map)
            
            # 确保包含所有必需字段
            for col in self.OUTPUT_FIELDS:
                if col not in df.columns:
                    df[col] = None
            
            # 转换日期格式
            if 'trade_date' in df.columns and df['trade_date'].notna().any():
                try:
                    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
                except:
                    pass
            
            result = df[self.OUTPUT_FIELDS]
            logger.info(f"成功处理大盘资金流向数据，共{len(result)}条")
            return result
            
        except Exception as e:
            logger.error(f"AkShare获取大盘资金流向失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)


@CollectorRegistry.register("hsgt_flow")
class HSGTFlowCollector(BaseCollector):
    """
    沪深港通资金流向采集器
    
    采集沪深港通北向/南向资金流向数据
    主数据源：Tushare (moneyflow_hsgt)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',           # 交易日期
        'ggt_ss',               # 港股通（沪）
        'ggt_sz',               # 港股通（深）
        'hgt',                  # 沪股通
        'sgt',                  # 深股通
        'north_money',          # 北向资金
        'south_money',          # 南向资金
    ]
    
    def collect(
        self,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集沪深港通资金流向数据
        
        Args:
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的沪深港通资金流向数据
        """
        # 设置默认日期
        if not start_date and not trade_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        if not end_date and not trade_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条沪深港通数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取沪深港通数据失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条沪深港通数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取沪深港通数据失败: {e}")
        
        logger.error("无法获取沪深港通资金数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取沪深港通数据"""
        pro = self.tushare_api
        
        params = {}
        if trade_date:
            params['trade_date'] = trade_date
            return pro.moneyflow_hsgt(**params)
        
        # 分块采集
        if start_date and end_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y%m%d')
                
                all_dfs = []
                current_dt = start_dt
                while current_dt <= end_dt:
                    next_year = current_dt.year + 1
                    chunk_end_dt = min(datetime(next_year, 1, 1) - timedelta(days=1), end_dt)
                    
                    p = params.copy()
                    p['start_date'] = current_dt.strftime('%Y%m%d')
                    p['end_date'] = chunk_end_dt.strftime('%Y%m%d')
                    
                    try:
                        chunk_df = pro.moneyflow_hsgt(**p)
                        if not chunk_df.empty:
                            all_dfs.append(chunk_df)
                    except Exception as e:
                        logger.warning(f"沪深港通资金分块采集失败 ({p['start_date']}-{p['end_date']}): {e}")
                    
                    current_dt = chunk_end_dt + timedelta(days=1)
                
                if all_dfs:
                    df = pd.concat(all_dfs, ignore_index=True)
                else:
                    df = pd.DataFrame(columns=self.OUTPUT_FIELDS)
            except Exception as e:
                logger.error(f"沪深港通资金分块逻辑异常: {e}")
                if start_date: params['start_date'] = start_date
                if end_date: params['end_date'] = end_date
                df = pro.moneyflow_hsgt(**params)
        else:
            if start_date: params['start_date'] = start_date
            if end_date: params['end_date'] = end_date
            df = pro.moneyflow_hsgt(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保数值列为 float 并计算北向/南向资金
        num_cols = ['hgt', 'sgt', 'ggt_ss', 'ggt_sz']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if 'hgt' in df.columns and 'sgt' in df.columns:
            df['north_money'] = df['hgt'] + df['sgt']
        if 'ggt_ss' in df.columns and 'ggt_sz' in df.columns:
            df['south_money'] = df['ggt_ss'] + df['ggt_sz']
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        df = df.sort_values('trade_date', ascending=True)
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取沪深港通数据"""
        import akshare as ak
        
        try:
            df = ak.stock_hsgt_hist_em(symbol="北向")
        except Exception as e:
            logger.warning(f"AkShare获取沪深港通失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '日期': 'trade_date',
            '当日资金流入': 'north_money',
        }
        df = self._standardize_columns(df, column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_money_flow(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取个股资金流向数据
    
    Args:
        ts_code: 证券代码
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 资金流向数据
    
    Example:
        >>> df = get_money_flow(ts_code='000001.SZ', start_date='20240101')
    """
    collector = MoneyFlowCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date)


def get_money_flow_industry(sector_type: str = 'industry', **kwargs) -> pd.DataFrame:
    """
    获取行业资金流向数据
    
    Args:
        sector_type: 板块类型（industry=行业，concept=概念，area=地区）
        **kwargs: 其他参数，如 start_date, end_date (由调度器传入)
    
    Returns:
        DataFrame: 行业资金流向数据
    
    Example:
        >>> df = get_money_flow_industry(sector_type='industry')
    """
    collector = MoneyFlowIndustryCollector()
    return collector.collect(sector_type=sector_type, **kwargs)


def get_money_flow_market(
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取大盘资金流向数据
    
    Args:
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 大盘资金流向数据
    """
    collector = MoneyFlowMarketCollector()
    return collector.collect(trade_date=trade_date,
                            start_date=start_date, end_date=end_date)


def get_hsgt_flow(
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取沪深港通资金流向数据
    
    Args:
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 沪深港通资金数据
    
    Example:
        >>> df = get_hsgt_flow(start_date='20240101')
    """
    collector = HSGTFlowCollector()
    return collector.collect(trade_date=trade_date,
                            start_date=start_date, end_date=end_date)
