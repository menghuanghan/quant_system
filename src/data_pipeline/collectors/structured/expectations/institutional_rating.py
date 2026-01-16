"""
机构评级（Institutional Rating）采集模块

数据类型包括：
- 机构买入/增持评级统计
- 目标价预测
- 机构推荐关注度
- 评级变化时间序列
- 机构调研数据
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


@CollectorRegistry.register("inst_rating")
class InstitutionalRatingCollector(BaseCollector):
    """
    机构评级采集器
    
    采集机构对个股的投资评级数据
    主数据源：AkShare (stock_rank_forecast_cninfo)
    备用数据源：通过Tushare report_rc提取
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'name',                 # 股票名称
        'rating_date',          # 评级日期
        'org_name',             # 评级机构
        'analyst_name',         # 分析师
        'rating',               # 投资评级（买入/增持/中性/减持/卖出）
        'rating_change',        # 评级变化（上调/下调/维持/首次）
        'pre_rating',           # 前次评级
        'is_first',             # 是否首次评级
        'target_price_low',     # 目标价下限
        'target_price_high',    # 目标价上限
        'target_price',         # 目标价（中值）
        'report_title',         # 研报标题
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        rating_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        org_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集机构评级数据
        
        Args:
            ts_code: 证券代码
            rating_date: 评级日期
            start_date: 开始日期
            end_date: 结束日期
            org_name: 评级机构
        
        Returns:
            DataFrame: 标准化的机构评级数据
        """
        # 优先使用AkShare
        try:
            df = self._collect_from_akshare(rating_date or end_date)
            if not df.empty:
                # 按条件筛选
                if ts_code:
                    df = df[df['ts_code'] == ts_code]
                if org_name:
                    df = df[df['org_name'].str.contains(org_name, na=False)]
                logger.info(f"从AkShare成功获取 {len(df)} 条机构评级数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取机构评级失败: {e}")
        
        # 降级到Tushare report_rc
        try:
            df = self._collect_from_tushare(ts_code, start_date, end_date, org_name)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条机构评级数据")
                return df
        except Exception as e:
            logger.error(f"Tushare获取机构评级失败: {e}")
        
        logger.error("所有数据源均无法获取机构评级数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self, date: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取机构评级"""
        import akshare as ak
        
        try:
            # 使用机构推荐池接口
            df = ak.stock_institute_recommend(symbol="全部")
        except Exception as e:
            logger.warning(f"AkShare获取机构评级失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 字段映射
        column_mapping = {
            '股票代码': 'ts_code',
            '股票简称': 'name',
            '最新评级日期': 'rating_date',
            '机构名称': 'org_name',
            '最新评级': 'rating',
        }
        df = df.rename(columns=column_mapping)
        
        # 补充交易所后缀
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].apply(self._add_exchange_suffix)
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        org_name: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare report_rc提取评级数据"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if org_name:
            params['org_name'] = org_name
        
        try:
            df = pro.report_rc(**params)
        except Exception as e:
            logger.warning(f"Tushare report_rc接口调用失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 字段映射
        # Tushare report_rc 官方字段包含: ts_code, name, report_date, report_title, org_name, author_name, rating, max_price, min_price, avg_price
        column_mapping = {
            'report_date': 'rating_date',
            'author_name': 'analyst_name',
            'max_price': 'target_price_high',
            'min_price': 'target_price_low',
            'avg_price': 'target_price',
        }
        df = df.rename(columns=column_mapping)
        
        # 计算目标价中值 (如果 avg_price 为空)
        if df['target_price'].isnull().all():
            if 'target_price_low' in df.columns and 'target_price_high' in df.columns:
                df['target_price'] = (pd.to_numeric(df['target_price_low'], errors='coerce') + 
                                    pd.to_numeric(df['target_price_high'], errors='coerce')) / 2
        
        # 日期格式标准化
        df = self._convert_date_format(df, ['rating_date'])
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _add_exchange_suffix(self, code: str) -> str:
        """为股票代码添加交易所后缀"""
        if not code or '.' in str(code):
            return code
        code = str(code).zfill(6)
        if code.startswith(('6', '5')):
            return f"{code}.SH"
        elif code.startswith(('0', '3', '2')):
            return f"{code}.SZ"
        elif code.startswith(('4', '8')):
            return f"{code}.BJ"
        return code


@CollectorRegistry.register("rating_summary")
class RatingSummaryCollector(BaseCollector):
    """
    评级汇总统计采集器
    
    采集个股的评级汇总统计数据（买入/增持/中性/减持数量）
    主数据源：通过InstitutionalRatingCollector聚合计算
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'name',                 # 股票名称
        'stat_date',            # 统计日期
        'rating_count',         # 评级总数
        'buy_count',            # 买入评级数
        'outperform_count',     # 增持评级数
        'hold_count',           # 中性评级数
        'underperform_count',   # 减持评级数
        'sell_count',           # 卖出评级数
        'avg_target_price',     # 平均目标价
        'latest_close',         # 最新收盘价
        'upside',               # 上涨空间（%）
        'coverage_count',       # 关注机构数
    ]
    
    def collect(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集评级汇总统计数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的评级汇总数据
        """
        # 通过聚合InstitutionalRatingCollector数据计算
        try:
            df = self._calculate_from_ratings(start_date, end_date)
            if not df.empty:
                logger.info(f"从评级数据聚合得到 {len(df)} 条评级汇总数据")
                return df
        except Exception as e:
            logger.error(f"评级汇总聚合计算失败: {e}")
        
        logger.error("无法获取评级汇总数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _calculate_from_ratings(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从评级数据聚合计算汇总"""
        # 获取评级数据
        rating_collector = InstitutionalRatingCollector()
        df = rating_collector.collect(start_date=start_date, end_date=end_date)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 按股票聚合
        def count_rating(group, rating_type):
            if 'rating' not in group.columns:
                return 0
            return group['rating'].str.contains(rating_type, case=False, na=False).sum()
        
        result = df.groupby('ts_code').agg({
            'name': 'first',
            'rating': 'count',
            'target_price': 'mean',
            'org_name': 'nunique',
        }).reset_index()
        
        result.columns = ['ts_code', 'name', 'rating_count', 'avg_target_price', 'coverage_count']
        
        # 计算各评级数量
        for ts_code in result['ts_code'].unique():
            mask = df['ts_code'] == ts_code
            stock_ratings = df.loc[mask, 'rating'].fillna('')
            
            result.loc[result['ts_code'] == ts_code, 'buy_count'] = stock_ratings.str.contains('买入', na=False).sum()
            result.loc[result['ts_code'] == ts_code, 'outperform_count'] = stock_ratings.str.contains('增持', na=False).sum()
            result.loc[result['ts_code'] == ts_code, 'hold_count'] = stock_ratings.str.contains('中性|持有', na=False).sum()
            result.loc[result['ts_code'] == ts_code, 'underperform_count'] = stock_ratings.str.contains('减持', na=False).sum()
            result.loc[result['ts_code'] == ts_code, 'sell_count'] = stock_ratings.str.contains('卖出', na=False).sum()
        
        result['stat_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in result.columns:
                result[col] = None
        
        return result[self.OUTPUT_FIELDS]


@CollectorRegistry.register("inst_survey")
class InstitutionalSurveyCollector(BaseCollector):
    """
    机构调研采集器
    
    采集机构对上市公司的调研记录
    主数据源：Tushare (stk_surv) - 需2000积分
    备用数据源：AkShare (stock_jgdy_tj_em)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'name',                 # 股票名称
        'surv_date',            # 调研日期
        'fund_visitors',        # 基金公司调研人数
        'rece_place',           # 接待地点
        'rece_mode',            # 接待方式
        'rece_org',             # 接待对象
        'org_type',             # 机构类型
        'comp_rece',            # 公司接待人员
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        surv_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集机构调研数据
        
        Args:
            ts_code: 证券代码
            surv_date: 调研日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的机构调研数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, surv_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条机构调研数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取机构调研失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条机构调研数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取机构调研失败: {e}")
        
        logger.error("所有数据源均无法获取机构调研数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        surv_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取机构调研"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if surv_date:
            params['surv_date'] = surv_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = pro.stk_surv(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 日期格式标准化
        df = self._convert_date_format(df, ['surv_date'])
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, ts_code: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取机构调研"""
        import akshare as ak
        
        try:
            # 使用机构调研统计接口
            df = ak.stock_jgdy_tj_em(date=datetime.now().strftime('%Y%m%d'))
        except Exception as e:
            logger.warning(f"AkShare获取机构调研失败(stock_jgdy_tj_em): {e}")
            # 尝试备用接口
            try:
                df = ak.stock_jgdy_detail_em(date=datetime.now().strftime('%Y%m%d'))
            except Exception as e2:
                logger.warning(f"AkShare获取机构调研失败(stock_jgdy_detail_em): {e2}")
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 列名映射
        column_mapping = {
            '代码': 'ts_code',
            '名称': 'name',
            '证券代码': 'ts_code',
            '证券简称': 'name',
            '股票代码': 'ts_code',
            '股票名称': 'name',
            '最新调研日期': 'surv_date',
            '调研日期': 'surv_date',
            '接待地点': 'rece_place',
            '接待方式': 'rece_mode',
            '接待机构数': 'fund_visitors',
            '机构调研次数': 'fund_visitors',
            '接待机构': 'rece_org',
            '调研机构': 'rece_org',
            '机构类型': 'org_type',
            '接待人员': 'comp_rece',
        }
        df = df.rename(columns=column_mapping)
        
        # 补充交易所后缀
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].apply(self._add_exchange_suffix)
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _add_exchange_suffix(self, code: str) -> str:
        """为股票代码添加交易所后缀"""
        if not code or '.' in str(code):
            return code
        code = str(code).zfill(6)
        if code.startswith(('6', '5')):
            return f"{code}.SH"
        elif code.startswith(('0', '3', '2')):
            return f"{code}.SZ"
        elif code.startswith(('4', '8')):
            return f"{code}.BJ"
        return code
