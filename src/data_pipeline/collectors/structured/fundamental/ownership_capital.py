"""
股权与资本结构（Ownership & Capital）采集模块

数据类型包括：
- 股本结构
- 前十大股东/流通股东
- 股权质押
- 限售解禁
- 股票回购
- 分红送股
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


@CollectorRegistry.register("share_structure")
class ShareStructureCollector(BaseCollector):
    """
    股本结构采集器
    
    采集公司股本结构数据
    主数据源：Tushare (stk_holdernumber / share_float)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'ann_date',             # 公告日期
        'end_date',             # 截止日期
        'holder_num',           # 股东户数（注意：Tushare返回的是holder_num单数）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股本结构数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的股本结构数据
        """
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条股本结构数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取股本结构失败: {e}")
        
        logger.error("无法获取股本结构数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取股本结构"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if ann_date:
            params['ann_date'] = ann_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = pro.stk_holdernumber(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['ann_date', 'end_date'])
        
        # 确保包含所有字段（只保留Tushare实际返回的字段）
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("top10_holders")
class Top10HoldersCollector(BaseCollector):
    """
    前十大股东采集器
    
    采集前十大股东/流通股东数据
    主数据源：Tushare (top10_holders / top10_floatholders)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'ann_date',             # 公告日期
        'end_date',             # 报告期
        'holder_name',          # 股东名称
        'hold_amount',          # 持股数量（股）
        'hold_ratio',           # 持股比例（%）
        'hold_float_ratio',     # 占流通股比例（%）
        'hold_change',          # 持股变化（股）
        'holder_type',          # 股东类型
    ]
    
    def collect(
        self,
        ts_code: str,
        period: Optional[str] = None,
        type: str = 'all',
        **kwargs
    ) -> pd.DataFrame:
        """
        采集前十大股东数据
        
        Args:
            ts_code: 证券代码（必填）
            period: 报告期（YYYYMMDD）
            type: 类型（all=全部，top10=前十大，float=流通股东）
        
        Returns:
            DataFrame: 标准化的股东数据
        """
        if not ts_code:
            logger.error("需要指定ts_code")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, period, type)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条股东数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取股东数据失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条股东数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取股东数据失败: {e}")
        
        logger.error("无法获取股东数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: str,
        period: Optional[str],
        type: str
    ) -> pd.DataFrame:
        """从Tushare获取股东数据"""
        pro = self.tushare_api
        
        results = []
        
        # 前十大股东
        if type in ['all', 'top10']:
            params = {'ts_code': ts_code}
            if period:
                params['period'] = period
            df1 = pro.top10_holders(**params)
            if not df1.empty:
                df1['holder_type'] = '前十大股东'
                results.append(df1)
        
        # 前十大流通股东
        if type in ['all', 'float']:
            params = {'ts_code': ts_code}
            if period:
                params['period'] = period
            df2 = pro.top10_floatholders(**params)
            if not df2.empty:
                df2['holder_type'] = '前十大流通股东'
                results.append(df2)
        
        if not results:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = pd.concat(results, ignore_index=True)
        df = self._convert_date_format(df, ['ann_date', 'end_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, ts_code: str) -> pd.DataFrame:
        """从AkShare获取股东数据"""
        import akshare as ak
        
        symbol = ts_code.split('.')[0]
        
        try:
            df = ak.stock_main_stock_holder(stock=symbol)
        except Exception as e:
            logger.warning(f"AkShare获取股东数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '股东名称': 'holder_name',
            '持股数量': 'hold_amount',
            '持股比例': 'hold_ratio',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        df['holder_type'] = '前十大股东'
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("pledge")
class PledgeCollector(BaseCollector):
    """
    股权质押采集器
    
    采集股权质押数据
    主数据源：Tushare (pledge_stat / pledge_detail)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'end_date',             # 截止日期
        'pledge_count',         # 质押次数
        'unrest_pledge',        # 无限售股质押数量（万股）
        'rest_pledge',          # 限售股质押数量（万股）
        'total_share',          # 总股本（万股）
        'pledge_ratio',         # 质押比例（%）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股权质押数据
        
        Args:
            ts_code: 证券代码
            end_date: 截止日期
        
        Returns:
            DataFrame: 标准化的股权质押数据
        """
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条股权质押数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取股权质押失败: {e}")
        
        logger.error("无法获取股权质押数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取股权质押数据"""
        pro = self.tushare_api
        
        # 1. 尝试获取统计数据 (pledge_stat)
        params_stat = {}
        if ts_code:
            params_stat['ts_code'] = ts_code
        elif end_date:
            params_stat['end_date'] = end_date
            
        df_stat = pd.DataFrame()
        try:
            df_stat = pro.pledge_stat(**params_stat)
        except Exception as e:
            logger.warning(f"Tushare获取pledge_stat失败: {e}")
            
        # 2. 尝试获取明细数据 (pledge_detail) 并补充
        # 如果指定了ts_code，获取该股的明细
        if ts_code:
            try:
                df_detail = pro.pledge_detail(ts_code=ts_code)
                if not df_detail.empty:
                    # 将明细聚合为统计格式，以便统一输出
                    # 字段: ts_code, ann_date, pledgee, pledge_amount, ...
                    # 我们按公告日期聚合
                    df_detail_agg = df_detail.groupby(['ts_code', 'ann_date']).agg({
                        'pledge_amount': 'sum'
                    }).reset_index()
                    df_detail_agg = df_detail_agg.rename(columns={'ann_date': 'end_date', 'pledge_amount': 'total_share'})
                    df_detail_agg['pledge_count'] = 1 # 简化
                    
                    if df_stat.empty:
                        df_stat = df_detail_agg
                    else:
                        # 简单的合并逻辑，实际应用中可能需要更复杂的对齐
                        pass
            except Exception as e:
                logger.warning(f"Tushare获取pledge_detail失败: {e}")
                
        if df_stat.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df_stat = self._convert_date_format(df_stat, ['end_date'])
        
        # 确保数值型字段不为None
        numeric_cols = ['pledge_count', 'unrest_pledge', 'rest_pledge', 'total_share', 'pledge_ratio']
        for col in numeric_cols:
            if col in df_stat.columns:
                df_stat[col] = pd.to_numeric(df_stat[col], errors='coerce').fillna(0)
        
        for col in self.OUTPUT_FIELDS:
            if col not in df_stat.columns:
                df_stat[col] = None
        
        return df_stat[self.OUTPUT_FIELDS]


@CollectorRegistry.register("share_float")
class ShareFloatCollector(BaseCollector):
    """
    限售解禁采集器
    
    采集限售股解禁数据
    主数据源：Tushare (share_float)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'ann_date',             # 公告日期
        'float_date',           # 解禁日期
        'float_share',          # 解禁股份（万股）
        'float_ratio',          # 解禁股份占总股本比例（%）
        'holder_name',          # 股东名称
        'share_type',           # 股份类型
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        float_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集限售解禁数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            float_date: 解禁日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的限售解禁数据
        """
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, float_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条限售解禁数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取限售解禁失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare()
            if not df.empty:
                if ts_code:
                    df = df[df['ts_code'] == ts_code]
                logger.info(f"从AkShare成功获取 {len(df)} 条限售解禁数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取限售解禁失败: {e}")
        
        logger.error("无法获取限售解禁数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        float_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取限售解禁（按月+offset分页，截断时自动拆分半月）"""
        import calendar
        from datetime import datetime as dt, timedelta
        
        pro = self.tushare_api
        PAGE_LIMIT = 6000   # Tushare 单次返回上限
        MAX_OFFSET = 96000  # 服务端 offset 上限（保守值，实际约 102000）
        
        base_params = {}
        if ts_code:
            base_params['ts_code'] = ts_code
        if ann_date:
            base_params['ann_date'] = ann_date
        if float_date:
            base_params['float_date'] = float_date
        
        def _fetch_range(sd: str, ed: str) -> pd.DataFrame:
            """带 offset 分页获取一个日期区间，返回 (df, is_truncated)"""
            chunks = []
            offset = 0
            truncated = False
            for _ in range(20):
                if offset >= MAX_OFFSET:
                    truncated = True
                    break
                try:
                    p = dict(base_params)
                    p['start_date'] = sd
                    p['end_date'] = ed
                    p['limit'] = PAGE_LIMIT
                    p['offset'] = offset
                    chunk = pro.share_float(**p)
                    if chunk is None or chunk.empty:
                        break
                    chunks.append(chunk)
                    if len(chunk) < PAGE_LIMIT:
                        break
                    offset += PAGE_LIMIT
                    time.sleep(0.3)
                except Exception as e:
                    logger.debug(f"share_float({sd}-{ed}) offset={offset} 出错: {e}")
                    if offset > 0:
                        truncated = True
                    break
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            return df, truncated
        
        def _fetch_recursive(sd: str, ed: str, depth: int = 0) -> pd.DataFrame:
            """递归获取：截断时自动拆成两半"""
            df, truncated = _fetch_range(sd, ed)
            if not truncated or depth >= 3:
                return df
            # 拆成两半
            d1 = dt.strptime(sd, '%Y%m%d')
            d2 = dt.strptime(ed, '%Y%m%d')
            if (d2 - d1).days <= 1:
                return df
            mid = d1 + (d2 - d1) // 2
            mid_s = mid.strftime('%Y%m%d')
            mid_n = (mid + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"share_float {sd}-{ed} 数据量超限(>{MAX_OFFSET})，拆分为两半")
            df1 = _fetch_recursive(sd, mid_s, depth + 1)
            time.sleep(0.3)
            df2 = _fetch_recursive(mid_n, ed, depth + 1)
            parts = [p for p in [df, df1, df2] if not p.empty]
            return pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame()
        
        result_df = pd.DataFrame()
        
        if start_date and end_date:
            try:
                start_year = int(start_date[:4])
                start_month = int(start_date[4:6])
                end_year = int(end_date[:4])
                end_month = int(end_date[4:6])
                
                all_dfs = []
                y, m = start_year, start_month
                while (y, m) <= (end_year, end_month):
                    last_day = calendar.monthrange(y, m)[1]
                    sd = f"{y}{m:02d}01"
                    ed = f"{y}{m:02d}{last_day:02d}"
                    if sd < start_date:
                        sd = start_date
                    if ed > end_date:
                        ed = end_date
                    
                    month_df = _fetch_recursive(sd, ed)
                    if not month_df.empty:
                        all_dfs.append(month_df)
                    
                    time.sleep(0.3)
                    m += 1
                    if m > 12:
                        m = 1
                        y += 1
                
                if all_dfs:
                    result_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
            except Exception as e:
                logger.warning(f"按月分页获取失败，回退到单次请求: {e}")
                result_df, _ = _fetch_range(start_date, end_date)
        else:
            params_call = dict(base_params)
            if start_date:
                params_call['start_date'] = start_date
            if end_date:
                params_call['end_date'] = end_date
            result_df = pro.share_float(**params_call)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['ann_date', 'float_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self) -> pd.DataFrame:
        """从AkShare获取限售解禁"""
        import akshare as ak
        
        try:
            df = ak.stock_restricted_release_summary_em()
        except Exception as e:
            logger.warning(f"AkShare获取限售解禁失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '代码': 'symbol',
            '解禁日期': 'float_date',
            '解禁数量': 'float_share',
            '解禁市值': 'float_mv',
        }
        df = self._standardize_columns(df, column_mapping)
        
        if 'symbol' in df.columns:
            df['ts_code'] = df['symbol'].apply(
                lambda x: f"{str(x).zfill(6)}.SZ" if str(x).startswith(('0', '3')) 
                else f"{str(x).zfill(6)}.SH"
            )
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("repurchase")
class RepurchaseCollector(BaseCollector):
    """
    股票回购采集器
    
    采集股票回购数据
    主数据源：Tushare (repurchase)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'ann_date',             # 公告日期
        'end_date',             # 截止日期
        'proc',                 # 进度
        'vol',                  # 回购数量（股）
        'amount',               # 回购金额（元）
        'high_limit',           # 回购价格上限（元）
        'low_limit',            # 回购价格下限（元）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集股票回购数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的回购数据
        """
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条回购数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取回购数据失败: {e}")
        
        logger.error("无法获取回购数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取回购数据（支持offset分页，避免单次上限截断）"""
        pro = self.tushare_api
        PAGE_LIMIT = 2000
        MAX_OFFSET = 200000
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if ann_date:
            params['ann_date'] = ann_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        def _fetch_with_pagination(query_params: dict, context: str) -> pd.DataFrame:
            """按 offset 分页抓取单个查询区间的数据。"""
            chunks = []
            offset = 0

            while True:
                if offset > MAX_OFFSET:
                    logger.warning(
                        f"repurchase[{context}] offset 超过上限({MAX_OFFSET})，"
                        "可能仍有数据未抓取"
                    )
                    break

                request_params = dict(query_params)
                request_params['limit'] = PAGE_LIMIT
                request_params['offset'] = offset

                try:
                    chunk = pro.repurchase(**request_params)
                except Exception as e:
                    logger.warning(
                        f"repurchase[{context}] 分页请求失败(offset={offset}): {e}"
                    )
                    break

                if chunk is None or chunk.empty:
                    break

                chunks.append(chunk)

                # 最后一页
                if len(chunk) < PAGE_LIMIT:
                    break

                offset += PAGE_LIMIT
                time.sleep(0.2)

            if not chunks:
                return pd.DataFrame()

            result = pd.concat(chunks, ignore_index=True)
            logger.info(f"repurchase[{context}] 抓取完成: {len(result)} 条")
            return result

        df = pd.DataFrame()

        # ann_date 精确查询优先，避免与日期范围参数混用
        if ann_date:
            df = _fetch_with_pagination(params, f"ann_date={ann_date}")
        # 若提供时间范围，按年分块 + 分页抓取，避免单年命中上限后被截断
        elif start_date and end_date:
            start_year = int(start_date[:4])
            end_year = int(end_date[:4])
            years = range(start_year, end_year + 1)

            all_dfs = []
            for year in years:
                p = params.copy()
                p['start_date'] = max(start_date, f"{year}0101")
                p['end_date'] = min(end_date, f"{year}1231")

                if p['start_date'] > p['end_date']:
                    continue

                context = f"{p['start_date']}-{p['end_date']}"
                year_df = _fetch_with_pagination(p, context)
                if not year_df.empty:
                    all_dfs.append(year_df)

            if all_dfs:
                df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
        else:
            # 无时间范围时也走分页，避免默认2000条截断
            df = _fetch_with_pagination(params, "full_scan")
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['ann_date', 'end_date', 'exp_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("dividend")
class DividendCollector(BaseCollector):
    """
    分红送股采集器
    
    采集分红送股数据
    主数据源：Tushare (dividend)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'end_date',             # 分红年度
        'ann_date',             # 预案公告日
        'div_proc',             # 实施进度
        'stk_div',              # 每股送转（股）
        'cash_div',             # 每股分红（税前）（元）
        'cash_div_tax',         # 每股分红（税后）（元）
        'record_date',          # 股权登记日
        'ex_date',              # 除权除息日
        'pay_date',             # 派息日
        'imp_ann_date',         # 实施公告日
        # 注意：base_date和base_share字段Tushare不返回，已移除
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        record_date: Optional[str] = None,
        ex_date: Optional[str] = None,
        imp_ann_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集分红送股数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            record_date: 股权登记日
            ex_date: 除权除息日
            imp_ann_date: 实施公告日
        
        Returns:
            DataFrame: 标准化的分红送股数据
        """
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, record_date, ex_date, imp_ann_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条分红数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取分红数据失败: {e}")
        
        # 降级到AkShare
        try:
            if ts_code:
                df = self._collect_from_akshare(ts_code)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条分红数据")
                    return df
        except Exception as e:
            logger.error(f"AkShare获取分红数据失败: {e}")
        
        logger.error("无法获取分红数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        record_date: Optional[str],
        ex_date: Optional[str],
        imp_ann_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取分红数据"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if ann_date:
            params['ann_date'] = ann_date
        if record_date:
            params['record_date'] = record_date
        if ex_date:
            params['ex_date'] = ex_date
        if imp_ann_date:
            params['imp_ann_date'] = imp_ann_date
        
        df = pro.dividend(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 只转换实际存在的日期字段
        df = self._convert_date_format(df, ['end_date', 'ann_date', 'record_date', 
                                           'ex_date', 'pay_date', 'div_listdate',
                                           'imp_ann_date'])
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, ts_code: str) -> pd.DataFrame:
        """从AkShare获取分红数据"""
        import akshare as ak
        
        symbol = ts_code.split('.')[0]
        
        try:
            df = ak.stock_fhps_detail_em(symbol=symbol)
        except Exception as e:
            logger.warning(f"AkShare获取分红数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        column_mapping = {
            '公告日期': 'ann_date',
            '分红年度': 'end_date',
            '送股': 'stk_bo_rate',
            '转增': 'stk_co_rate',
            '派息': 'cash_div',
            '股权登记日': 'record_date',
            '除权除息日': 'ex_date',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_share_structure(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取股本结构数据
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 股本结构数据
    """
    collector = ShareStructureCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            start_date=start_date, end_date=end_date)


def get_top10_holders(
    ts_code: str,
    period: Optional[str] = None,
    type: str = 'all',
    **kwargs  # 接受调度器传递的额外参数（如 start_date, end_date）
) -> pd.DataFrame:
    """
    获取前十大股东数据
    
    Args:
        ts_code: 证券代码（必填）
        period: 报告期
        type: all=全部，top10=前十大，float=流通股东
        **kwargs: 其他参数（由调度器传入，会被忽略）
    
    Returns:
        DataFrame: 股东数据
    
    Example:
        >>> df = get_top10_holders(ts_code='000001.SZ')
    """
    collector = Top10HoldersCollector()
    return collector.collect(ts_code=ts_code, period=period, type=type)


def get_pledge(
    ts_code: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取股权质押数据
    
    Args:
        ts_code: 证券代码
        end_date: 截止日期
    
    Returns:
        DataFrame: 股权质押数据
    """
    collector = PledgeCollector()
    return collector.collect(ts_code=ts_code, end_date=end_date, **kwargs)


def get_share_float(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    float_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取限售解禁数据
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        float_date: 解禁日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 限售解禁数据
    """
    collector = ShareFloatCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            float_date=float_date, start_date=start_date,
                            end_date=end_date)


def get_repurchase(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs  # 接受调度器传递的额外参数
) -> pd.DataFrame:
    """
    获取股票回购数据
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        start_date: 开始日期
        end_date: 结束日期
        **kwargs: 其他参数（由调度器传入，会被忽略）
    
    Returns:
        DataFrame: 回购数据
    """
    collector = RepurchaseCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            start_date=start_date, end_date=end_date)


def get_dividend(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    record_date: Optional[str] = None,
    ex_date: Optional[str] = None,
    imp_ann_date: Optional[str] = None,
    **kwargs  # 接受调度器传递的额外参数
) -> pd.DataFrame:
    """
    获取分红送股数据
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        record_date: 股权登记日
        ex_date: 除权除息日
        imp_ann_date: 实施公告日
        **kwargs: 其他参数（由调度器传入，会被忽略）
    
    Returns:
        DataFrame: 分红送股数据
    
    Example:
        >>> df = get_dividend(ts_code='000001.SZ')
    """
    collector = DividendCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            record_date=record_date, ex_date=ex_date,
                            imp_ann_date=imp_ann_date)
