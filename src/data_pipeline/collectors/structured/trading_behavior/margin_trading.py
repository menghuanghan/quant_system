"""
融资融券与杠杆行为采集模块

数据类型包括：
- 融资融券汇总/明细
- 两融标的
- 转融通
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta

import time
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


@CollectorRegistry.register("margin_summary")
class MarginSummaryCollector(BaseCollector):
    """
    融资融券汇总采集器
    
    采集市场融资融券汇总数据
    主数据源：Tushare (margin)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',           # 交易日期
        'exchange_id',          # 交易所
        'rzye',                 # 融资余额（元）
        'rzmre',                # 融资买入额（元）
        'rzche',                # 融资偿还额（元）
        'rqye',                 # 融券余额（元）
        'rqmcl',                # 融券卖出量（股）
        'rzrqye',               # 融资融券余额（元）
        'rqyl',                 # 融券余量（股）
    ]
    
    def collect(
        self,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        exchange_id: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集融资融券汇总数据
        
        Args:
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            exchange_id: 交易所（SSE/SZSE）
        
        Returns:
            DataFrame: 标准化的融资融券汇总数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(trade_date, start_date, end_date, exchange_id)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条融资融券汇总数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取融资融券汇总失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条融资融券汇总数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取融资融券汇总失败: {e}")
        
        logger.error("所有数据源均无法获取融资融券汇总数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        trade_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        exchange_id: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取融资融券汇总"""
        pro = self.tushare_api
        
        # 如果是范围查询且未指定交易所，按交易所分别采集以避免4000条限制
        if (start_date or end_date) and not exchange_id and not trade_date:
            exchanges = ['SSE', 'SZSE', 'BSE']
            all_dfs = []
            
            for ex in exchanges:
                try:
                    params = {'exchange_id': ex}
                    if start_date:
                        params['start_date'] = start_date
                    if end_date:
                        params['end_date'] = end_date
                    
                    df = pro.margin(**params)
                    if not df.empty:
                        all_dfs.append(df)
                        logger.info(f"融资融券汇总 {ex}: {len(df)} 条记录")
                    
                    time.sleep(0.3)  # 频控
                except Exception as e:
                    logger.warning(f"获取 {ex} 融资融券汇总失败: {e}")
            
            if not all_dfs:
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            df = pd.concat(all_dfs, ignore_index=True)
        else:
            # 单日查询或指定了交易所
            params = {}
            if trade_date:
                params['trade_date'] = trade_date
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            if exchange_id:
                params['exchange_id'] = exchange_id
            
            df = pro.margin(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取融资融券汇总"""
        import akshare as ak
        
        try:
            df_sh = ak.stock_margin_sse(start_date="", end_date="")
            df_sh['exchange_id'] = 'SSE'
        except Exception as e:
            logger.warning(f"AkShare获取上交所融资融券失败: {e}")
            df_sh = pd.DataFrame()
        
        try:
            df_sz = ak.stock_margin_szse(start_date="", end_date="")
            df_sz['exchange_id'] = 'SZSE'
        except Exception as e:
            logger.warning(f"AkShare获取深交所融资融券失败: {e}")
            df_sz = pd.DataFrame()
        
        if df_sh.empty and df_sz.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        results = []
        for df in [df_sh, df_sz]:
            if not df.empty:
                column_mapping = {
                    '信用交易日期': 'trade_date',
                    '融资余额': 'rzye',
                    '融资买入额': 'rzmre',
                    '融券余额': 'rqye',
                    '融资融券余额': 'rzrqye',
                }
                df = self._standardize_columns(df, column_mapping)
                results.append(df)
        
        if not results:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df_all = pd.concat(results, ignore_index=True)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df_all.columns:
                df_all[col] = None
        
        return df_all[self.OUTPUT_FIELDS]


@CollectorRegistry.register("margin_detail")
class MarginDetailCollector(BaseCollector):
    """
    融资融券明细采集器
    
    采集个股融资融券明细数据
    主数据源：Tushare (margin_detail)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',           # 交易日期
        'ts_code',              # 证券代码
        'name',                 # 证券名称
        'rzye',                 # 融资余额（元）
        'rqye',                 # 融券余额（元）
        'rzmre',                # 融资买入额（元）
        'rqyl',                 # 融券余量（股）
        'rzche',                # 融资偿还额（元）
        'rqchl',                # 融券偿还量（股）
        'rqmcl',                # 融券卖出量（股）
        'rzrqye',               # 融资融券余额（元）
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
        采集融资融券明细数据
        
        Args:
            trade_date: 交易日期
            ts_code: 证券代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的融资融券明细数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(trade_date, ts_code, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条融资融券明细数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取融资融券明细失败: {e}")
        
        # 降级到AkShare
        try:
            if ts_code:
                df = self._collect_from_akshare(ts_code)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条融资融券明细数据")
                    return df
        except Exception as e:
            logger.error(f"AkShare获取融资融券明细失败: {e}")
        
        logger.error("所有数据源均无法获取融资融券明细数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        trade_date: Optional[str],
        ts_code: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取融资融券明细"""
        pro = self.tushare_api
        
        # 1. 指定股票或单日查询
        if ts_code or trade_date:
            params = {}
            if trade_date: params['trade_date'] = trade_date
            if ts_code: params['ts_code'] = ts_code
            return pro.margin_detail(**params)
            
        # 2. 全市场历史范围查询 (优化：循环日期获取全市场)
        if start_date and end_date:
            try:
                # 获取交易日历
                cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
                dates = cal['cal_date'].tolist()
                # 默认降序，如果用户习惯从旧到新，可以加 [::-1]，但之前用户说“不用反转”
            except Exception as e:
                logger.warning(f"获取交易日历失败，尝试直接按日遍历: {e}")
                dates = [d.strftime('%Y%m%d') for d in pd.date_range(start_date, end_date)]
            
            all_dfs = []
            total_dates = len(dates)
            logger.info(f"开始批量获取融资融券明细 (共 {total_dates} 个交易日)...")
            
            for i, date in enumerate(dates):
                try:
                    # 频控 (Tushare margin_detail 限制较高，但出于安全设置 0.4s)
                    time.sleep(0.4)
                    df = pro.margin_detail(trade_date=date)
                    if not df.empty:
                        all_dfs.append(df)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"融资融券明细批量采集进度: {i+1}/{total_dates} ({date})")
                except Exception as e:
                    if "最多访问" in str(e):
                        logger.warning(f"触发流量限制，等待 30s... {date}")
                        time.sleep(30.0)
                        # 重试一次本日期
                        try:
                            df = pro.margin_detail(trade_date=date)
                            if not df.empty: all_dfs.append(df)
                        except: pass
                    else:
                        logger.debug(f"{date} 采集失败: {e}")
            
            if all_dfs:
                return pd.concat(all_dfs, ignore_index=True)
                    
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self, ts_code: Optional[str] = None, trade_date: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取融资融券明细"""
        import akshare as ak
        
        # AkShare 接口需要日期，如果没有则取最新
        date_str = ""
        if trade_date:
            try:
                date_str = datetime.strptime(trade_date, '%Y%m%d').strftime('%Y%m%d')
            except:
                date_str = trade_date
        
        # 上海市场明细
        try:
            df_sh = ak.stock_margin_detail_sse(date=date_str)
        except:
            df_sh = pd.DataFrame()
            
        # 深圳市场明细 (AkShare 目前针对深交所的日报明细接口可能不同，此处做兼容)
        try:
            df_sz = ak.stock_margin_detail_szse(date=date_str)
        except:
            df_sz = pd.DataFrame()
            
        df = pd.concat([df_sh, df_sz], ignore_index=True)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化和筛选
        column_mapping = {
            '信用交易日期': 'trade_date',
            '证券代码': 'symbol',
            '证券简称': 'name',
            '融资余额': 'rzye',
            '融券余额': 'rqye',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 转换 ts_code
        if 'symbol' in df.columns:
            df['ts_code'] = df['symbol'].apply(
                lambda x: f"{str(x).zfill(6)}.SH" if str(x).startswith(('6', '9'))
                else f"{str(x).zfill(6)}.SZ"
            )
            
        if ts_code:
            df = df[df['ts_code'] == ts_code]
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("margin_target")
class MarginTargetCollector(BaseCollector):
    """
    两融标的采集器
    
    采集融资融券标的证券列表
    主数据源：Tushare (margin_target)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'name',                 # 证券名称
        'mg_type',              # 标的类型（融资/融券）
        'exchange',             # 交易所
        'start_date',           # 开始日期
        'end_date',             # 结束日期
        'is_new',               # 是否最新
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        is_new: str = 'Y',
        mg_type: Optional[str] = None,
        start_date: Optional[str] = None,  # 忽略，仅为兼容调度器
        end_date: Optional[str] = None,    # 忽略，仅为兼容调度器
        **kwargs
    ) -> pd.DataFrame:
        """
        采集两融标的数据
        
        Args:
            ts_code: 证券代码
            is_new: 是否最新（Y/N）
            mg_type: 标的类型（rz=融资，rq=融券）
        
        Returns:
            DataFrame: 标准化的两融标的数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, is_new, mg_type)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条两融标的数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取两融标的失败: {e}")

        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条两融标的数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取两融标的失败: {e}")
        
        logger.error("无法获取两融标的数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)

    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        is_new: str,
        mg_type: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取两融标的"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if is_new:
            params['is_new'] = is_new
        if mg_type:
            params['mg_type'] = mg_type
            
        df = pro.margin_target(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
        # 字段映射
        # Tushare 返回: ts_code, mg_type, is_new, name...
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
                
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取两融标的"""
        import akshare as ak
        
        results = []
        
        # 1. 深交所
        try:
            df_sz = ak.stock_margin_underlying_info_szse()
            if not df_sz.empty:
                df_sz['exchange'] = 'SZSE'
                results.append(df_sz)
        except Exception as e:
            logger.warning(f"AkShare获取深交所两融标的失败: {e}")
            
        # 2. 上交所
        try:
            df_sh = ak.stock_margin_underlying_info_sse()
            if not df_sh.empty:
                df_sh['exchange'] = 'SSE'
                results.append(df_sh)
        except Exception as e:
            logger.warning(f"AkShare获取上交所两融标的失败: {e}")
            
        if not results:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
        # 合并并处理
        df_all = pd.concat(results, ignore_index=True)
        
        final_rows = []
        for _, row in df_all.iterrows():
            ts_code = row.get('证券代码')
            name = row.get('证券简称')
            exchange = row.get('exchange')
            
            # 标准化代码
            if exchange == 'SZSE':
                full_code = f"{str(ts_code).zfill(6)}.SZ"
            else:
                full_code = f"{str(ts_code).zfill(6)}.SH"
            
            # 融资标的
            if row.get('融资标的') == 'Y':
                final_rows.append({
                    'ts_code': full_code,
                    'name': name,
                    'mg_type': 'rz',
                    'exchange': exchange,
                    'is_new': 'Y'
                })
                
            # 融券标的
            if row.get('融券标的') == 'Y':
                final_rows.append({
                    'ts_code': full_code,
                    'name': name,
                    'mg_type': 'rq',
                    'exchange': exchange,
                    'is_new': 'Y'
                })
                
        df_final = pd.DataFrame(final_rows)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df_final.columns:
                df_final[col] = None
                
        return df_final[self.OUTPUT_FIELDS]


@CollectorRegistry.register("slb")
class SLBCollector(BaseCollector):
    """
    转融通采集器
    
    采集转融通数据
    主数据源：Tushare (slb_len_mm / slb_sec)
    备用数据源：AkShare
    """
    
    OUTPUT_FIELDS = [
        'trade_date',           # 交易日期
        'ts_code',              # 证券代码
        'name',                 # 证券名称
        'ope_inv',              # 期初余量（股/份）
        'lent_qnt',             # 当日出借数量（股/份）
        'cls_inv',              # 期末余量（股/份）
        'end_bal',              # 期末余额（元）
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
        采集转融通数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(trade_date, ts_code, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条转融通数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取转融通失败: {e}")
            
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条转融通数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取转融通数据失败: {e}")
        
        logger.error("无法获取转融通数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)

    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        trade_date: Optional[str],
        ts_code: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取转融通数据 (详情)"""
        pro = self.tushare_api
        
        params = {}
        if ts_code: params['ts_code'] = ts_code
        
        # 如果是单日
        if trade_date:
            params['trade_date'] = trade_date
            return pro.slb_len_mm(**params)
            
        # 范围查询，需要分块请求
        if start_date and end_date:
            all_dfs = []
            try:
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y%m%d')
                
                current_dt = start_dt
                while current_dt <= end_dt:
                    d_str = current_dt.strftime('%Y%m%d')
                    
                    success = False
                    for attempt in range(3):
                        try:
                            chunk = pro.slb_len_mm(trade_date=d_str, **params)
                            if not chunk.empty:
                                all_dfs.append(chunk)
                                logger.info(f"成功获取转融通数据: {d_str} ({len(chunk)} 条)")
                            success = True
                            break
                        except Exception as e:
                            err_msg = str(e)
                            if "最多访问" in err_msg:
                                logger.warning(f"触发Tushare限制，等待60秒后重试 ({d_str})...")
                                time.sleep(60.0)
                            else:
                                logger.warning(f"转融通数据采集失败 ({d_str}), 第 {attempt+1} 次尝试: {e}")
                                time.sleep(2.0)
                    
                    if not success:
                        logger.error(f"转融通数据彻底采集失败: {d_str}")
                        
                    current_dt += timedelta(days=1)
                    # 适量延时，平滑请求
                    time.sleep(0.35)
                
                if all_dfs:
                    df = pd.concat(all_dfs, ignore_index=True)
                else:
                    df = pd.DataFrame(columns=self.OUTPUT_FIELDS)
            except Exception as e:
                logger.error(f"转融通分块逻辑错误: {e}")
                df = pd.DataFrame(columns=self.OUTPUT_FIELDS)
        else:
            if start_date: params['start_date'] = start_date
            if end_date: params['end_date'] = end_date
            df = pro.slb_len_mm(**params)
            
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
        df = self._convert_date_format(df, ['trade_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
                
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取转融通数据"""
        import akshare as ak
        
        try:
            # 这是一个汇总接口，不含明细，但姑且作为垫底
            df = ak.stock_margin_szse(date="")
            if df.empty: return pd.DataFrame()
            
            df = df.rename(columns={'信用交易日期': 'trade_date'})
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
            
            for col in self.OUTPUT_FIELDS:
                if col not in df.columns:
                    df[col] = None
            return df[self.OUTPUT_FIELDS]
        except:
            return pd.DataFrame()


# ============= 便捷函数接口 =============

def get_margin_summary(
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange_id: Optional[str] = None
) -> pd.DataFrame:
    """
    获取融资融券汇总数据
    
    Args:
        trade_date: 交易日期
        start_date: 开始日期
        end_date: 结束日期
        exchange_id: 交易所
    
    Returns:
        DataFrame: 融资融券汇总数据
    
    Example:
        >>> df = get_margin_summary(trade_date='20240115')
    """
    collector = MarginSummaryCollector()
    return collector.collect(trade_date=trade_date, start_date=start_date,
                            end_date=end_date, exchange_id=exchange_id)


def get_margin_detail(
    trade_date: Optional[str] = None,
    ts_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取融资融券明细数据
    
    Args:
        trade_date: 交易日期
        ts_code: 证券代码
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 融资融券明细数据
    
    Example:
        >>> df = get_margin_detail(ts_code='000001.SZ')
    """
    collector = MarginDetailCollector()
    return collector.collect(trade_date=trade_date, ts_code=ts_code,
                            start_date=start_date, end_date=end_date)


def get_margin_target(
    ts_code: Optional[str] = None,
    is_new: str = 'Y',
    mg_type: Optional[str] = None,
    start_date: Optional[str] = None,  # 忽略，仅为兼容调度器
    end_date: Optional[str] = None,    # 忽略，仅为兼容调度器
    **kwargs
) -> pd.DataFrame:
    """
    获取两融标的数据
    
    Args:
        ts_code: 证券代码
        is_new: 是否最新
        mg_type: 标的类型
        start_date: 忽略（时间无关数据）
        end_date: 忽略（时间无关数据）
    
    Returns:
        DataFrame: 两融标的数据
    """
    collector = MarginTargetCollector()
    return collector.collect(ts_code=ts_code, is_new=is_new, mg_type=mg_type, 
                           start_date=start_date, end_date=end_date)


def get_slb(
    trade_date: Optional[str] = None,
    ts_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取转融通数据
    
    Args:
        trade_date: 交易日期
        ts_code: 证券代码
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 转融通数据
    """
    collector = SLBCollector()
    return collector.collect(trade_date=trade_date, ts_code=ts_code,
                            start_date=start_date, end_date=end_date)
