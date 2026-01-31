"""
特殊交易行为采集模块

数据类型包括：
- 龙虎榜
- 大宗交易
- 营业部行为
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta

import time
from pathlib import Path
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


@CollectorRegistry.register("top_list")
class TopListCollector(BaseCollector):
    """
    龙虎榜采集器
    
    采集龙虎榜上榜股票数据
    主数据源：Tushare (top_list)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',           # 交易日期
        'ts_code',              # 证券代码
        'name',                 # 证券名称
        'close',                # 收盘价
        'pct_change',           # 涨跌幅（%）
        'turnover_rate',        # 换手率（%）
        'amount',               # 成交额（万元）
        'l_sell',               # 龙虎榜卖出额（万元）
        'l_buy',                # 龙虎榜买入额（万元）
        'l_amount',             # 龙虎榜成交额（万元）
        'net_amount',           # 龙虎榜净买入额（万元）
        'net_rate',             # 龙虎榜净买额占比（%）
        'amount_rate',          # 龙虎榜成交额占比（%）
        'float_values',         # 当日流通市值（万元）
        'reason',               # 上榜理由
    ]
    
    def collect(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集龙虎榜数据
        
        Args:
            trade_date: 交易日期
            ts_code: 证券代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的龙虎榜数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(trade_date, ts_code, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条龙虎榜数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取龙虎榜失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(trade_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条龙虎榜数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取龙虎榜失败: {e}")
        
        logger.error("所有数据源均无法获取龙虎榜数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        trade_date: Optional[str],
        ts_code: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取龙虎榜"""
        pro = self.tushare_api
        
        # 1. 单日查询
        if trade_date:
            params = {'trade_date': trade_date}
            if ts_code:
                params['ts_code'] = ts_code
            return pro.top_list(**params)
            
        # 2. 范围查询（循环）
        if start_date and end_date:
            try:
                # 获取交易日历
                cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
                dates = cal['cal_date'].tolist()
            except Exception as e:
                logger.warning(f"获取交易日历失败，尝试直接按日遍历: {e}")
                dates = [d.strftime('%Y%m%d') for d in pd.date_range(start_date, end_date)]
            
            all_dfs = []
            total_dates = len(dates)
            for i, date in enumerate(dates):
                try:
                    # 增加频控 (Tushare top_list 限制 200次/分钟)
                    time.sleep(0.35)
                    params = {'trade_date': date}
                    if ts_code:
                        params['ts_code'] = ts_code
                    df = pro.top_list(**params)
                    if not df.empty:
                        all_dfs.append(df)
                    
                    if (i + 1) % 20 == 0:
                        logger.info(f"龙虎榜数据采集进度: {i+1}/{total_dates} {date}")
                except Exception as e:
                    if "最多访问" in str(e):
                        logger.warning(f"触发流量限制，等待 15s... {date}")
                        time.sleep(15.0)
                    else:
                        logger.debug(f"{date} 采集失败: {e}")
                    
            if all_dfs:
                return pd.concat(all_dfs, ignore_index=True)
                
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self, trade_date: Optional[str]) -> pd.DataFrame:
        """从AkShare获取龙虎榜"""
        import akshare as ak
        
        try:
            if trade_date:
                date_str = datetime.strptime(trade_date, '%Y%m%d').strftime('%Y-%m-%d')
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')
            df = ak.stock_lhb_detail_em(symbol="近一月", start_date="", end_date="")
        except Exception as e:
            logger.warning(f"AkShare获取龙虎榜失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '上榜日': 'trade_date',
            '代码': 'symbol',
            '名称': 'name',
            '收盘价': 'close',
            '涨跌幅': 'pct_change',
            '龙虎榜净买额': 'net_amount',
            '龙虎榜买入额': 'l_buy',
            '龙虎榜卖出额': 'l_sell',
            '上榜原因': 'reason',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 转换证券代码
        if 'symbol' in df.columns:
            df['ts_code'] = df['symbol'].apply(
                lambda x: f"{str(x).zfill(6)}.SZ" if str(x).startswith(('0', '3'))
                else f"{str(x).zfill(6)}.SH"
            )
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("top_inst")
class TopInstCollector(BaseCollector):
    """
    龙虎榜营业部明细采集器
    
    采集龙虎榜上榜营业部/机构明细数据
    主数据源：Tushare (top_inst)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',           # 交易日期
        'ts_code',              # 证券代码
        'exalter',              # 营业部名称
        'buy',                  # 买入额（万元）
        'buy_rate',             # 买入占比（%）
        'sell',                 # 卖出额（万元）
        'sell_rate',            # 卖出占比（%）
        'net_buy',              # 净买入额（万元）
        'side',                 # 买卖方向（0=买，1=卖）
    ]
    
    def collect(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集龙虎榜营业部明细数据
        
        Args:
            trade_date: 交易日期
            ts_code: 证券代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的营业部明细数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(trade_date, ts_code, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条营业部明细数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取营业部明细失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条营业部明细数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取营业部明细失败: {e}")
        
        logger.error("所有数据源均无法获取营业部明细数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        trade_date: Optional[str],
        ts_code: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取营业部明细"""
        pro = self.tushare_api
        
        # 1. 单日查询 (已有逻辑)
        if trade_date:
            params = {'trade_date': trade_date}
            if ts_code:
                params['ts_code'] = ts_code
            return pro.top_inst(**params)
            
        # 2. 范围查询（优化：市场大盘轮询后拆分）
        if start_date and end_date:
            try:
                cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
                dates = cal['cal_date'].tolist()
            except Exception as e:
                logger.warning(f"获取交易日历失败，尝试直接按日遍历: {e}")
                dates = [d.strftime('%Y%m%d') for d in pd.date_range(start_date, end_date)]
            
            # 如果没有指定ts_code，说明是全局大批量采集
            if not ts_code:
                all_dfs = []
                total_dates = len(dates)
                for i, date in enumerate(dates):
                    try:
                        # 增加频控 (Tushare top_inst 限制较宽，但出于礼貌加个小延时)
                        time.sleep(0.12)
                        df = pro.top_inst(trade_date=date)
                        if not df.empty:
                            all_dfs.append(df)
                            if (i + 1) % 50 == 0:
                                logger.info(f"龙虎榜详情采集进度: {i+1}/{total_dates}")
                    except Exception as e:
                        if "最多访问" in str(e):
                            time.sleep(15.0)
                        else:
                            logger.debug(f"{date} 龙虎榜详情采集失败: {e}")
                
                if all_dfs:
                    full_df = pd.concat(all_dfs, ignore_index=True)
                    return full_df
            else:
                # 给定了 ts_code，只为该 code 循环获取历史
                all_dfs = []
                for date in dates:
                    try:
                        time.sleep(0.05)
                        df = pro.top_inst(trade_date=date, ts_code=ts_code)
                        if not df.empty:
                            all_dfs.append(df)
                    except: pass
                if all_dfs:
                    return pd.concat(all_dfs, ignore_index=True)
                    
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)

    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取营业部明细"""
        import akshare as ak
        
        try:
            df = ak.stock_lhb_jgstatistic_em(symbol="近一月")
        except Exception as e:
            logger.warning(f"AkShare获取营业部明细失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '营业部名称': 'exalter',
            '买入总额': 'buy',
            '卖出总额': 'sell',
            '净买入额': 'net_buy',
        }
        df = self._standardize_columns(df, column_mapping)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("block_trade")
class BlockTradeCollector(BaseCollector):
    """
    大宗交易采集器
    
    采集大宗交易数据
    主数据源：Tushare (block_trade)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'trade_date',           # 交易日期
        'price',                # 成交价
        'vol',                  # 成交量（万股）
        'amount',               # 成交金额（万元）
        'buyer',                # 买方营业部
        'seller',               # 卖方营业部
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
        采集大宗交易数据
        
        Args:
            ts_code: 证券代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的大宗交易数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, trade_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条大宗交易数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取大宗交易失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条大宗交易数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取大宗交易失败: {e}")
        
        logger.error("所有数据源均无法获取大宗交易数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取大宗交易"""
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
        
        # 如果有日期范围，进行分块采集以避免达到API限制
        if start_date and end_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y%m%d')
                
                all_dfs = []
                current_dt = start_dt
                while current_dt <= end_dt:
                    # 大宗交易全市场一天可能数百条，为稳妥起见按天采集
                    # Tushare 限制 200次/分钟，增加小延时
                    trade_date_str = current_dt.strftime('%Y%m%d')
                    
                    p = params.copy()
                    p['trade_date'] = trade_date_str
                    p.pop('start_date', None)
                    p.pop('end_date', None)
                    
                    # 内部重试机制
                    success = False
                    for attempt in range(3):
                        try:
                            chunk_df = pro.block_trade(**p)
                            if not chunk_df.empty:
                                all_dfs.append(chunk_df)
                                logger.info(f"大宗交易采集完成: {trade_date_str} ({len(chunk_df)} 条)")
                            success = True
                            break
                        except Exception as e:
                            if "最多访问该接口200次" in str(e):
                                time.sleep(20.0) # 限流了，多等一会儿
                            else:
                                logger.warning(f"大宗交易 {trade_date_str} 第 {attempt+1} 次尝试失败: {e}")
                                time.sleep(2.0)
                    
                    if not success:
                        logger.error(f"大宗交易 {trade_date_str} 采集彻底失败")
                    
                    # 即使成功也稍微等一下，防限流 (60s/200 = 0.3s)
                    time.sleep(0.35)
                    current_dt += timedelta(days=1)
                
                if all_dfs:
                    df = pd.concat(all_dfs, ignore_index=True)
                else:
                    df = pd.DataFrame()
            except Exception as e:
                logger.error(f"大宗交易分块逻辑异常: {e}, 尝试直接采集")
                df = pro.block_trade(**params)
        else:
            df = pro.block_trade(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取大宗交易"""
        import akshare as ak
        
        try:
            df = ak.stock_dzjy_mrmx(symbol="", start_date="", end_date="")
        except Exception as e:
            logger.warning(f"AkShare获取大宗交易失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '交易日期': 'trade_date',
            '证券代码': 'symbol',
            '成交价': 'price',
            '成交量': 'vol',
            '成交额': 'amount',
            '买方营业部': 'buyer',
            '卖方营业部': 'seller',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 转换证券代码
        if 'symbol' in df.columns:
            df['ts_code'] = df['symbol'].apply(
                lambda x: f"{str(x).zfill(6)}.SZ" if str(x).startswith(('0', '3'))
                else f"{str(x).zfill(6)}.SH"
            )
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_top_list(
    trade_date: Optional[str] = None,
    ts_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取龙虎榜数据
    
    Args:
        trade_date: 交易日期
        ts_code: 证券代码
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 龙虎榜数据
    
    Example:
        >>> df = get_top_list(trade_date='20240115')
    """
    collector = TopListCollector()
    return collector.collect(trade_date=trade_date, ts_code=ts_code,
                            start_date=start_date, end_date=end_date)


def get_top_inst(
    trade_date: Optional[str] = None,
    ts_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取龙虎榜营业部明细数据
    
    Args:
        trade_date: 交易日期
        ts_code: 证券代码
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 营业部明细数据
    
    Example:
        >>> df = get_top_inst(trade_date='20240115')
    """
    collector = TopInstCollector()
    return collector.collect(trade_date=trade_date, ts_code=ts_code,
                            start_date=start_date, end_date=end_date)


def get_block_trade(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取大宗交易数据
    
    Args:
        ts_code: 证券代码
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 大宗交易数据
    
    Example:
        >>> df = get_block_trade(trade_date='20240115')
    """
    collector = BlockTradeCollector()
    return collector.collect(ts_code=ts_code, trade_date=trade_date,
                            start_date=start_date, end_date=end_date)
