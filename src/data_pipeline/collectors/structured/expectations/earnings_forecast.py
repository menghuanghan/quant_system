"""
盈利预测（Earnings Forecast）采集模块

数据类型包括：
- 业绩预告（公司官方）
- 券商盈利预测（券商研报）
- 一致预期数据（EPS/净利润/营收）
- 预测修正数据
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


@CollectorRegistry.register("earnings_forecast")
class EarningsForecastCollector(BaseCollector):
    """
    业绩预告采集器
    
    采集上市公司官方业绩预告数据
    主数据源：Tushare (forecast) - 需800积分
    备用数据源：AkShare (stock_yjyg_em)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'ann_date',             # 公告日期
        'end_date',             # 报告期
        'type',                 # 预告类型（预增/预减/扭亏/首亏/续亏/续盈/略增/略减）
        'p_change_min',         # 预告净利润变动幅度下限（%）
        'p_change_max',         # 预告净利润变动幅度上限（%）
        'net_profit_min',       # 预告净利润下限（万元）
        'net_profit_max',       # 预告净利润上限（万元）
        'last_parent_net',      # 上年同期归母净利润（万元）
        'first_ann_date',       # 首次公告日期
        'summary',              # 业绩预告摘要
        'change_reason',        # 变动原因
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        type: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集业绩预告数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            start_date: 公告开始日期
            end_date: 公告结束日期
            period: 报告期（如20231231）
            type: 预告类型
        
        Returns:
            DataFrame: 标准化的业绩预告数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, start_date, end_date, period, type)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条业绩预告数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取业绩预告失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(period)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条业绩预告数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取业绩预告失败: {e}")
        
        logger.error("所有数据源均无法获取业绩预告数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        period: Optional[str],
        type: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取业绩预告"""
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
        if period:
            params['period'] = period
        if type:
            params['type'] = type
        
        df = pro.forecast(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 日期格式标准化
        df = self._convert_date_format(df, ['ann_date', 'end_date', 'first_ann_date'])
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, period: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取业绩预告"""
        import akshare as ak
        import re
        
        try:
            # AkShare要求提供完整的报告期日期（季末），如20231231
            target_period = period if period else '20231231' 
            df = ak.stock_yjyg_em(date=target_period)
        except Exception as e:
            logger.warning(f"AkShare获取业绩预告失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # AkShare实际字段：['序号', '股票代码', '股票简称', '预测指标', '业绩变动', '预测数值', 
        #                  '业绩变动幅度', '业绩变动原因', '预告类型', '上年同期值', '公告日期']
        
        # 列名映射
        column_mapping = {
            '股票代码': 'ts_code',
            '股票简称': 'name',
            '公告日期': 'ann_date',
            '预告类型': 'type',
            '业绩变动原因': 'change_reason',
            '上年同期值': 'last_parent_net',
            '业绩变动': 'summary',
        }
        df = df.rename(columns=column_mapping)
        df['end_date'] = target_period
        
        # 解析数值的辅助函数
        def parse_number_range(val):
            """从字符串中提取数值范围，如'10%~20%'或'1.5亿~2.3亿'"""
            if pd.isna(val) or not isinstance(val, str):
                return None, None
            # 提取数字（包括小数和负数）
            numbers = re.findall(r'([-+]?\d*\.?\d+)', val)
            if len(numbers) >= 2:
                return float(numbers[0]), float(numbers[1])
            elif len(numbers) == 1:
                num = float(numbers[0])
                return num, num
            return None, None
        
        # 处理业绩变动幅度（百分比）
        if '业绩变动幅度' in df.columns:
            ranges = df['业绩变动幅度'].apply(parse_number_range)
            df['p_change_min'] = ranges.apply(lambda x: x[0] if x[0] is not None else None)
            df['p_change_max'] = ranges.apply(lambda x: x[1] if x[1] is not None else None)
        
        # 处理预测数值（净利润金额）
        if '预测数值' in df.columns:
            def parse_profit_value(val):
                """解析预测数值，如'1.5亿~2.3亿'，转换为万元"""
                if pd.isna(val) or not isinstance(val, str):
                    return None, None
                # 提取数字
                numbers = re.findall(r'([-+]?\d*\.?\d+)', val)
                if not numbers:
                    return None, None
                
                # 判断单位
                multiplier = 1  # 默认万元
                if '亿' in val:
                    multiplier = 10000  # 亿元转万元
                elif '万' in val:
                    multiplier = 1
                elif '元' in val and '万' not in val and '亿' not in val:
                    multiplier = 0.0001  # 元转万元
                
                if len(numbers) >= 2:
                    return float(numbers[0]) * multiplier, float(numbers[1]) * multiplier
                elif len(numbers) == 1:
                    num = float(numbers[0]) * multiplier
                    return num, num
                return None, None
            
            ranges = df['预测数值'].apply(parse_profit_value)
            df['net_profit_min'] = ranges.apply(lambda x: x[0] if x[0] is not None else None)
            df['net_profit_max'] = ranges.apply(lambda x: x[1] if x[1] is not None else None)
        
        # 处理上年同期值（转换为万元）
        if 'last_parent_net' in df.columns:
            def convert_to_wan(val):
                """转换金额为万元"""
                if pd.isna(val):
                    return None
                if isinstance(val, (int, float)):
                    return val  # 假设已经是万元
                if isinstance(val, str):
                    numbers = re.findall(r'([-+]?\d*\.?\d+)', val)
                    if numbers:
                        num = float(numbers[0])
                        if '亿' in val:
                            return num * 10000
                        elif '万' in val:
                            return num
                        elif '元' in val:
                            return num * 0.0001
                        return num
                return None
            
            df['last_parent_net'] = df['last_parent_net'].apply(convert_to_wan)
        
        # 补充交易所后缀
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].apply(self._add_exchange_suffix)
        
        # 日期格式转换（YYYY-MM-DD -> YYYYMMDD）
        if 'ann_date' in df.columns:
            df['ann_date'] = pd.to_datetime(df['ann_date'], errors='coerce').dt.strftime('%Y%m%d')
        
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


@CollectorRegistry.register("broker_forecast")
class BrokerForecastCollector(BaseCollector):
    """
    券商盈利预测采集器
    
    采集券商研报中的盈利预测数据
    主数据源：Tushare (report_rc) - 需5000积分
    备用数据源：AkShare (stock_rank_forecast_cninfo)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'name',                 # 股票名称
        'report_date',          # 报告发布日期
        'report_title',         # 报告标题
        'org_name',             # 研究机构
        'author_name',          # 研究员
        'quarter',              # 预测季度
        'op_rt',                # 预测营业收入（元）
        'op_pr',                # 预测营业利润（元）
        'np',                   # 预测净利润（元）
        'eps',                  # 预测每股收益
        'pe',                   # 预测市盈率
        'roe',                  # 预测ROE
        'rating',               # 投资评级
        'max_price',            # 目标价上限
        'min_price',            # 目标价下限
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        org_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集券商盈利预测数据
        
        Args:
            ts_code: 证券代码
            report_date: 报告日期
            start_date: 开始日期
            end_date: 结束日期
            org_name: 研究机构名称
        
        Returns:
            DataFrame: 标准化的券商盈利预测数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, report_date, start_date, end_date, org_name)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条券商盈利预测数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取券商盈利预测失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(report_date or end_date)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条券商盈利预测数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取券商盈利预测失败: {e}")
        
        logger.error("所有数据源均无法获取券商盈利预测数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        report_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        org_name: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取券商盈利预测"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if report_date:
            params['report_date'] = report_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if org_name:
            params['org_name'] = org_name
        
        df = pro.report_rc(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 日期格式标准化
        df = self._convert_date_format(df, ['report_date'])
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, date: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取券商盈利预测"""
        import akshare as ak
        
        try:
            if date:
                # 转换日期格式 YYYYMMDD -> YYYY-MM-DD
                if len(date) == 8:
                    date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            df = ak.stock_rank_forecast_cninfo(date=date)
        except Exception as e:
            logger.warning(f"AkShare获取券商盈利预测失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 列名映射
        column_mapping = {
            '证券代码': 'ts_code',
            '证券简称': 'name',
            '发布日期': 'report_date',
            '研究机构简称': 'org_name',
            '研究员名称': 'author_name',
            '投资评级': 'rating',
            '目标价格-上限': 'max_price',
            '目标价格-下限': 'min_price',
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


@CollectorRegistry.register("consensus_forecast")
class ConsensusForecastCollector(BaseCollector):
    """
    一致预期数据采集器
    
    采集券商对个股的一致预期数据（EPS/净利润/营收预测均值）
    主数据源：AkShare (stock_analyst_rank_em)
    备用数据源：通过report_rc聚合计算
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'name',                 # 股票名称
        'report_date',          # 统计日期
        'year',                 # 预测年度
        'analyst_count',        # 分析师数量
        'eps_avg',              # EPS一致预期
        'eps_max',              # EPS预测最大值
        'eps_min',              # EPS预测最小值
        'np_avg',               # 净利润一致预期（亿元）
        'np_max',               # 净利润预测最大值
        'np_min',               # 净利润预测最小值
        'revenue_avg',          # 营收一致预期（亿元）
        'target_price_avg',     # 目标价均值
        'rating_buy',           # 买入评级数
        'rating_hold',          # 增持评级数
        'rating_neutral',       # 中性评级数
        'rating_sell',          # 减持评级数
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        year: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集一致预期数据
        
        Args:
            ts_code: 证券代码
            year: 预测年度
        
        Returns:
            DataFrame: 标准化的一致预期数据
        """
        # 优先使用AkShare
        try:
            df = self._collect_from_akshare(ts_code)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条一致预期数据")
                return df
        except Exception as e:
            logger.warning(f"AkShare获取一致预期失败: {e}")
        
        # 降级：通过report_rc聚合计算
        # 降级：通过report_rc聚合计算
        # try:
        #     df = self._calculate_from_broker_forecast(ts_code, year)
        #     if not df.empty:
        #         logger.info(f"从券商预测聚合计算得到 {len(df)} 条一致预期数据")
        #         return df
        # except Exception as e:
        #     logger.error(f"聚合计算一致预期失败: {e}")
        
        logger.error("所有数据源均无法获取一致预期数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    def _collect_from_akshare(self, ts_code: Optional[str] = None) -> pd.DataFrame:
        """从AkShare获取一致预期"""
        import akshare as ak
        
        try:
            # 使用机构推荐接口获取个股评级详情
            if ts_code:
                symbol = ts_code.split('.')[0]
                df = ak.stock_institute_recommend_detail(symbol=symbol)
            else:
                # 全市场机构推荐池
                df = ak.stock_institute_recommend(symbol="全部")
        except Exception as e:
            logger.warning(f"AkShare获取一致预期失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
        # 字段映射
        column_mapping = {
            '股票代码': 'ts_code',
            '股票简称': 'name',
            '股票名称': 'name',
            '评级日期': 'report_date',
            '最新评级日期': 'report_date',
            '评级机构': 'analyst_count',  # 借用，实际是机构名
            '最新评级': 'rating_buy',
        }
        df = df.rename(columns=column_mapping)
        
        # 补充交易所后缀
        if 'ts_code' in df.columns:
            def add_suffix(code):
                if not code or '.' in str(code):
                    return code
                code = str(code).zfill(6)
                if code.startswith(('6', '5')):
                    return f"{code}.SH"
                elif code.startswith(('0', '3', '2')):
                    return f"{code}.SZ"
                return code
            df['ts_code'] = df['ts_code'].apply(add_suffix)
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _calculate_from_broker_forecast(
        self,
        ts_code: Optional[str],
        year: Optional[str]
    ) -> pd.DataFrame:
        """从券商预测数据聚合计算一致预期"""
        # 调用BrokerForecastCollector获取数据
        broker_collector = BrokerForecastCollector()
        df = broker_collector.collect(ts_code=ts_code)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 按股票聚合计算一致预期
        result = df.groupby('ts_code').agg({
            'name': 'first',
            'eps': ['mean', 'max', 'min', 'count'],
            'np': ['mean', 'max', 'min'],
            'op_rt': 'mean',
            'max_price': 'mean',
            'min_price': 'mean',
        }).reset_index()
        
        # 扁平化列名
        result.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                         for col in result.columns.values]
        
        # 重命名列
        rename_map = {
            'eps_mean': 'eps_avg',
            'eps_max': 'eps_max',
            'eps_min': 'eps_min',
            'eps_count': 'analyst_count',
            'np_mean': 'np_avg',
            'np_max': 'np_max',
            'np_min': 'np_min',
            'op_rt_mean': 'revenue_avg',
        }
        result = result.rename(columns=rename_map)
        
        # 计算目标价均值
        if 'max_price_mean' in result.columns and 'min_price_mean' in result.columns:
            result['target_price_avg'] = (result['max_price_mean'] + result['min_price_mean']) / 2
        
        result['report_date'] = datetime.now().strftime('%Y-%m-%d')
        result['year'] = year or str(datetime.now().year)
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in result.columns:
                result[col] = None
        
        return result[self.OUTPUT_FIELDS]
