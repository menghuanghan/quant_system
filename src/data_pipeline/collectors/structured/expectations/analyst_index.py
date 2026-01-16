"""
研究员指数（Analyst Index）采集模块

数据类型包括：
- 分析师指数排行
- 分析师详情
- 券商月度金股组合
- 分析师荐股收益率
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


@CollectorRegistry.register("analyst_rank")
class AnalystRankCollector(BaseCollector):
    """
    分析师排行采集器
    
    采集分析师综合排行数据
    主数据源：AkShare (stock_analyst_rank_em)
    """
    
    OUTPUT_FIELDS = [
        'analyst_name',         # 分析师姓名
        'analyst_id',           # 分析师ID
        'org_name',             # 所属机构
        'year',                 # 年度
        'industry',             # 研究行业
        'stock_count',          # 关注股票数
        'avg_return',           # 平均收益率（%）
        'success_rate',         # 成功率（%）
        'rank',                 # 综合排名
        'score',                # 综合得分
    ]
    
    def collect(
        self,
        year: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集分析师排行数据
        
        Args:
            year: 年度
        
        Returns:
            DataFrame: 标准化的分析师排行数据
        """
        # 使用AkShare
        try:
            df = self._collect_from_akshare(year)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条分析师排行数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取分析师排行失败: {e}")
        
        logger.error("所有数据源均无法获取分析师排行数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self, year: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取分析师排行"""
        import akshare as ak
        
        try:
            df = ak.stock_analyst_rank_em()
        except Exception as e:
            logger.warning(f"AkShare获取分析师排行失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 列名映射
        # 注意：AkShare 的收益率字段包含年份，如 '2024年收益率'
        current_year = datetime.now().year
        return_col = f"{current_year}年收益率"
        
        # 寻找包含“收益率”的列
        actual_return_col = next((c for c in df.columns if '收益率' in c and '年' in c), return_col)
        
        column_mapping = {
            '分析师名称': 'analyst_name',
            '分析师ID': 'analyst_id',
            '分析师单位': 'org_name',
            '行业': 'industry',
            '成分股个数': 'stock_count',
            actual_return_col: 'avg_return',
            '年度指数': 'score',
            '序号': 'rank',
        }
        df = df.rename(columns=column_mapping)
        
        df['year'] = year or str(current_year)
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("analyst_detail")
class AnalystDetailCollector(BaseCollector):
    """
    分析师详情采集器
    
    采集单个分析师的详细信息和荐股记录
    主数据源：AkShare (stock_analyst_detail_em)
    """
    
    OUTPUT_FIELDS = [
        'analyst_name',         # 分析师姓名
        'analyst_id',           # 分析师ID
        'org_name',             # 所属机构
        'ts_code',              # 推荐股票代码
        'stock_name',           # 股票名称
        'recommend_date',       # 推荐日期
        'rating',               # 评级
        'target_price',         # 目标价
        'recommend_return',     # 推荐收益率（%）
        'days_held',            # 持有天数
    ]
    
    def collect(
        self,
        analyst_id: Optional[str] = None,
        analyst_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集分析师详情数据
        
        Args:
            analyst_id: 分析师ID
            analyst_name: 分析师姓名
        
        Returns:
            DataFrame: 标准化的分析师详情数据
        """
        if not analyst_id and not analyst_name:
            logger.warning("需要提供analyst_id或analyst_name")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 使用AkShare
        try:
            df = self._collect_from_akshare(analyst_id)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条分析师详情数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取分析师详情失败: {e}")
        
        logger.error("所有数据源均无法获取分析师详情数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self, analyst_id: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取分析师详情"""
        import akshare as ak
        
        try:
            df = ak.stock_analyst_detail_em(analyst_id=analyst_id)
        except Exception as e:
            logger.warning(f"AkShare获取分析师详情失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 列名映射
        column_mapping = {
            '分析师': 'analyst_name',
            '分析师ID': 'analyst_id',
            '券商': 'org_name',
            '股票代码': 'ts_code',
            '股票名称': 'stock_name',
            '推荐日期': 'recommend_date',
            '评级': 'rating',
            '目标价': 'target_price',
            '收益率': 'recommend_return',
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


@CollectorRegistry.register("broker_gold_stock")
class BrokerGoldStockCollector(BaseCollector):
    """
    券商金股采集器
    
    采集券商月度金股组合数据
    主数据源：AkShare (stock_rank_xstp_ths)
    """
    
    OUTPUT_FIELDS = [
        'month',                # 月份（YYYYMM格式）
        'ts_code',              # 证券代码
        'name',                 # 股票名称
        'org_name',             # 推荐券商
        'recommend_reason',     # 推荐理由
        'industry',             # 所属行业
        'recommend_date',       # 推荐日期
        'target_price',         # 目标价
        'current_price',        # 当前价格
        'recommend_return',     # 推荐以来收益率（%）
    ]
    
    def collect(
        self,
        month: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集券商金股数据
        
        Args:
            month: 月份（YYYYMM格式）
        
        Returns:
            DataFrame: 标准化的券商金股数据
        """
        # 优先使用Tushare broker_recommend (需积分)
        try:
            df = self._collect_from_tushare(month)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条券商金股数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取券商金股失败: {e}")

        # 降级到AkShare
        try:
            df = self._collect_from_akshare(month)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条券商金股数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取券商金股失败: {e}")
        
        logger.error("所有数据源均无法获取券商金股数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_tushare(self, month: Optional[str] = None) -> pd.DataFrame:
        """从Tushare获取券商金股"""
        pro = self.tushare_api
        params = {'month': month or datetime.now().strftime('%Y%m')}
        
        try:
            df = pro.broker_recommend(**params)
        except Exception as e:
            logger.warning(f"Tushare broker_recommend 接口调用失败: {e}")
            return pd.DataFrame()
            
        if df.empty:
            return pd.DataFrame()
            
        # 映射
        column_mapping = {
            'broker': 'org_name',
        }
        df = df.rename(columns=column_mapping)
        return df

    def _collect_from_akshare(self, month: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取券商金股（尝试获取相似的精选研报排行）"""
        import akshare as ak
        
        try:
            # 尝试获取研报预测排行
            df = ak.stock_rank_forecast_cninfo()
        except Exception as e:
            logger.warning(f"AkShare获取券商金股失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 列名映射
        column_mapping = {
            '证券代码': 'ts_code',
            '证券简称': 'name',
            '研究机构简称': 'org_name',
            '投资评级': 'recommend_reason',
        }
        df = df.rename(columns=column_mapping)
        
        # 补充交易所后缀
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].apply(self._add_exchange_suffix)
        
        df['month'] = month or datetime.now().strftime('%Y%m')
        
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


@CollectorRegistry.register("forecast_revision")
class ForecastRevisionCollector(BaseCollector):
    """
    预测修正采集器
    
    采集分析师对个股盈利预测的修正数据
    主数据源：通过BrokerForecastCollector计算
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'name',                 # 股票名称
        'stat_date',            # 统计日期
        'period',               # 预测报告期
        'eps_current',          # 当前EPS预测
        'eps_previous',         # 前次EPS预测
        'eps_change',           # EPS预测变化
        'eps_change_pct',       # EPS预测变化率（%）
        'np_current',           # 当前净利润预测
        'np_previous',          # 前次净利润预测
        'np_change_pct',        # 净利润预测变化率（%）
        'upgrade_count',        # 上调预测机构数
        'downgrade_count',      # 下调预测机构数
        'revision_trend',       # 修正趋势（up/down/stable）
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集预测修正数据
        
        Args:
            ts_code: 证券代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的预测修正数据
        """
        # 通过AkShare获取
        try:
            df = self._collect_from_akshare(ts_code)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条预测修正数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取预测修正失败: {e}")
        
        logger.error("所有数据源均无法获取预测修正数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self, ts_code: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取预测修正 (使用东财盈利预测接口获取更丰富的数据)"""
        import akshare as ak
        
        try:
            # stock_profit_forecast_em 需要股票简称，不是代码
            # 这里我们使用一个通用的方法获取所有研报预测数据
            df = ak.stock_rank_forecast_cninfo()
        except Exception as e:
            logger.warning(f"AkShare获取预测修正失败 (stock_rank_forecast_cninfo): {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 映射 - stock_rank_forecast_cninfo 返回字段
        column_mapping = {
            '证券代码': 'ts_code',
            '证券简称': 'name',
            '发布日期': 'stat_date',
            '投资评级': 'revision_trend',
        }
        df = df.rename(columns=column_mapping)
        
        # 如果指定了股票代码，筛选
        if ts_code:
            symbol = ts_code.split('.')[0]
            df = df[df['ts_code'] == symbol] if 'ts_code' in df.columns else df
        
        # 补充交易所后缀
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].apply(self._add_exchange_suffix)
        
        if 'stat_date' not in df.columns:
            df['stat_date'] = datetime.now().strftime('%Y-%m-%d')
        
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
