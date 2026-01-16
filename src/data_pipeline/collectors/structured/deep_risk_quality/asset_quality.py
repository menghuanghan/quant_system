"""
资产质量异常（Asset Quality Anomaly）采集模块

数据类型包括：
- 个股商誉明细
- 商誉减值预期明细
- 破净股统计
"""

import logging
from typing import Optional
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


@CollectorRegistry.register("stock_goodwill")
class StockGoodwillCollector(BaseCollector):
    """
    个股商誉明细采集器
    
    获取上市公司商誉明细数据
    主数据源：AkShare (stock_sy_profile_em)
    备用数据源：Tushare (fina_indicator中的goodwill字段)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 股票代码
        'stock_name',           # 股票名称
        'report_date',          # 报告期
        'goodwill',             # 商誉(亿元)
        'net_assets',           # 净资产(亿元)
        'goodwill_ratio',       # 商誉占净资产比例(%)
        'total_assets',         # 总资产(亿元)
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集个股商誉明细数据
        
        Args:
            ts_code: 股票代码 (可选，不填则获取全市场)
            report_date: 报告期 (YYYYMMDD格式，可选)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 商誉明细数据
        """
        # 优先使用AkShare
        try:
            df = self._collect_from_akshare(ts_code, report_date, **kwargs)
            if not df.empty:
                logger.info(f"从 AkShare 成功采集商誉明细数据: {len(df)} 条")
                return df
        except Exception as e:
            logger.warning(f"从 AkShare 采集商誉明细失败: {e}")
        
        # 降级到Tushare
        try:
            df = self._collect_from_tushare(ts_code, report_date, **kwargs)
            if not df.empty:
                logger.info(f"从 Tushare 成功采集商誉明细数据: {len(df)} 条")
                return df
        except Exception as e:
            logger.error(f"从 Tushare 采集商誉明细失败: {e}")
        
        logger.error("所有数据源均采集失败")
        
        # 确保返回符合OUTPUT_FIELDS格式的空DataFrame
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare正确的接口：stock_sy_em (商誉数据)
        # 需要传入date参数，格式为YYYYMMDD
        date_param = report_date if report_date else datetime.now().strftime('%Y%m%d')
        df = ak.stock_sy_em(date=date_param)
        
        if df.empty:
            return pd.DataFrame()
        
        # AkShare实际返回字段：['序号', '股票代码', '股票简称', '商誉', '商誉占净资产比例', 
        #                        '净利润', '净利润同比', '上年商誉', '公告日期', '交易市场']
        
        # 标准化列名
        column_mapping = {
            '股票代码': 'ts_code',
            '股票简称': 'stock_name',
            '公告日期': 'report_date',
            '商誉': 'goodwill',
            '商誉占净资产比例': 'goodwill_ratio',
            '上年商誉': 'previous_goodwill',
            '净利润': 'net_profit',
            '净利润同比': 'net_profit_yoy',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 股票代码过滤
        if ts_code:
            # 移除.SH/.SZ后缀以匹配AkShare格式
            code = ts_code.split('.')[0]
            if 'ts_code' in df.columns:
                df = df[df['ts_code'].astype(str) == code]
        
        # 日期格式转换
        if 'report_date' in df.columns:
            df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce').dt.strftime('%Y%m%d')
        
        return df
    
    @retry_on_failure(max_retries=3)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从Tushare采集数据"""
        import tushare as ts
        
        pro = ts.pro_api()
        
        # Tushare需要通过fina_indicator获取商誉数据
        # 需要2000+积分
        if ts_code:
            df = pro.fina_indicator(
                ts_code=ts_code,
                period=report_date,
                fields='ts_code,end_date,goodwill,tangible_asset,total_assets'
            )
        else:
            # 批量获取需要按报告期查询
            if not report_date:
                # 默认最近一个季度
                report_date = datetime.now().strftime('%Y%m%d')
            
            df = pro.fina_indicator(
                period=report_date,
                fields='ts_code,end_date,goodwill,tangible_asset,total_assets'
            )
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名
        column_mapping = {
            'end_date': 'report_date',
            'tangible_asset': 'net_assets',  # 使用有形资产作为净资产的近似
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 计算商誉占比
        df['goodwill_ratio'] = (df['goodwill'] / df['net_assets'] * 100).round(2)
        
        # 缺失字段
        df['stock_name'] = None
        
        return df


@CollectorRegistry.register("goodwill_impairment")
class GoodwillImpairmentCollector(BaseCollector):
    """
    商誉减值预期明细采集器
    
    获取商誉减值预期数据
    主数据源：AkShare (stock_sy_jz_em)
    备用数据源：无
    """
    
    OUTPUT_FIELDS = [
        'ts_code',                  # 股票代码
        'stock_name',               # 股票名称
        'report_date',              # 报告期
        'goodwill',                 # 商誉(亿元)
        'expected_impairment',      # 预期减值(亿元)
        'impairment_ratio',         # 减值占商誉比例(%)
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集商誉减值预期明细数据
        
        Args:
            ts_code: 股票代码 (可选)
            report_date: 报告期 (YYYYMMDD格式，可选)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 商誉减值预期数据
        """
        manager = DataSourceManager()
        df = pd.DataFrame()
        
        try:
            df = self._collect_from_akshare(ts_code, report_date, **kwargs)
            
            if not df.empty:
                logger.info(f"成功从 AkShare 采集商誉减值数据: {len(df)} 条")
        except Exception as e:
            logger.error(f"从 AkShare 采集商誉减值数据失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            logger.warning("未能采集到商誉减值数据")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 确保包含所有必需字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare接口：stock_sy_jz_em (商誉减值预期)
        try:
            df = ak.stock_sy_jz_em()
        except Exception as e:
            logger.warning(f"AkShare API调用失败: {e}")
            return pd.DataFrame()
        
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            logger.warning("AkShare返回空数据")
            return pd.DataFrame()
        
        # 标准化列名
        column_mapping = {
            '股票代码': 'ts_code',
            '股票简称': 'stock_name',
            '报告期': 'report_date',
            '商誉': 'goodwill',
            '预期减值': 'expected_impairment',
            '减值占比': 'impairment_ratio',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 股票代码过滤
        if ts_code:
            code = ts_code.split('.')[0]
            df = df[df['ts_code'] == code]
        
        # 报告期过滤
        if report_date:
            df['report_date'] = pd.to_datetime(df['report_date'])
            report = pd.to_datetime(report_date)
            df = df[df['report_date'] == report]
            df['report_date'] = df['report_date'].dt.strftime('%Y%m%d')
        
        return df


@CollectorRegistry.register("break_net_stock")
class BreakNetStockCollector(BaseCollector):
    """
    破净股统计采集器
    
    获取破净股统计数据（市净率<1的股票）
    主数据源：AkShare (stock_a_below_net_asset_statistics)
    备用数据源：Tushare (daily_basic筛选pb<1)
    """
    
    OUTPUT_FIELDS = [
        'date',                 # 统计日期
        'total_count',          # 破净股总数
        'break_net_count_sh',   # 上海破净股数量
        'break_net_count_sz',   # 深圳破净股数量
        'break_net_ratio',      # 破净股占比(%)
    ]
    
    def collect(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集破净股统计数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 破净股统计数据
        """
        # 优先使用AkShare
        try:
            df = self._collect_from_akshare(start_date, end_date, **kwargs)
            if not df.empty:
                logger.info(f"从 AkShare 成功采集破净股统计数据: {len(df)} 条")
                return df
        except Exception as e:
            logger.warning(f"从 AkShare 采集破净股统计失败: {e}")
        
        # 降级到Tushare
        try:
            df = self._collect_from_tushare(start_date, end_date, **kwargs)
            if not df.empty:
                logger.info(f"从 Tushare 成功采集破净股统计数据: {len(df)} 条")
                return df
        except Exception as e:
            logger.error(f"从 Tushare 采集破净股统计失败: {e}")
        
        logger.error("所有数据源均采集失败")
        
        # 确保返回符合OUTPUT_FIELDS格式的空DataFrame
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3)
    def _collect_from_akshare(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从AkShare采集数据"""
        import akshare as ak
        
        # AkShare没有直接的破净股统计接口，我们使用市场全景图数据筛选PB\u003c1的股票
        # 使用 stock_a_lg_indicator 获取市场指标
        try:
            df = ak.stock_a_lg_indicator(symbol="沪深京A股")
        except Exception as e:
            logger.warning(f"AkShare获取市场指标失败: {e}")
            # 尝试另一个接口
            try:
                df = ak.stock_zh_a_spot_em()
                if df.empty:
                    return pd.DataFrame()
                # 筛选PB小于1的股票
                if '市净率' in df.columns:
                    df = df[pd.to_numeric(df['市净率'], errors='coerce') < 1]
            except:
                return pd.DataFrame()
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名（根据实际返回字段调整）
        column_mapping = {
            '日期': 'date',
            '代码': 'ts_code',
            '名称': 'stock_name',
            '市净率': 'pb',
            '最新价': 'close',
            '涨跌幅': 'pct_change',
        }
        df = self._standardize_columns(df, column_mapping)
        
        # 筛选市净率 < 1 的股票
        if 'pb' in df.columns:
            df['pb'] = pd.to_numeric(df['pb'], errors='coerce')
            df = df[df['pb'] < 1]
        
        # 统计信息
        current_date = datetime.now().strftime('%Y%m%d')
        summary_df = pd.DataFrame([{
            'date': current_date,
            'total_count': len(df),
            'break_net_ratio': len(df) / 5000 * 100 if len(df) > 0 else 0,  # 假设A股总数约5000
            'avg_pb': df['pb'].mean() if 'pb' in df.columns else None,
            'min_pb': df['pb'].min() if 'pb' in df.columns else None,
        }])
        
        return summary_df
    
    @retry_on_failure(max_retries=3)
    def _collect_from_tushare(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """从Tushare采集数据（需要手动统计）"""
        import tushare as ts
        
        pro = ts.pro_api()
        
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = end_date
        
        # 获取指定日期范围的破净股数据
        result_list = []
        
        # 按日期循环统计
        date_range = pd.date_range(
            start=pd.to_datetime(start_date),
            end=pd.to_datetime(end_date),
            freq='D'
        )
        
        for date in date_range:
            trade_date = date.strftime('%Y%m%d')
            
            # 获取当日所有股票的市净率
            df_daily = pro.daily_basic(
                trade_date=trade_date,
                fields='ts_code,trade_date,pb'
            )
            
            if df_daily.empty:
                continue
            
            # 筛选破净股（pb<1）
            break_net = df_daily[df_daily['pb'] < 1]
            
            # 统计
            total = len(break_net)
            sh_count = len(break_net[break_net['ts_code'].str.endswith('.SH')])
            sz_count = len(break_net[break_net['ts_code'].str.endswith('.SZ')])
            ratio = (total / len(df_daily) * 100) if len(df_daily) > 0 else 0
            
            result_list.append({
                'date': trade_date,
                'total_count': total,
                'break_net_count_sh': sh_count,
                'break_net_count_sz': sz_count,
                'break_net_ratio': round(ratio, 2)
            })
        
        if not result_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(result_list)
        return df


# 便捷函数
def get_stock_goodwill(
    ts_code: Optional[str] = None,
    report_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取个股商誉明细数据
    
    Args:
        ts_code: 股票代码 (可选)
        report_date: 报告期 (YYYYMMDD格式，可选)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 商誉明细数据
    """
    collector = StockGoodwillCollector()
    return collector.collect(
        ts_code=ts_code,
        report_date=report_date,
        **kwargs
    )


def get_goodwill_impairment(
    ts_code: Optional[str] = None,
    report_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取商誉减值预期明细数据
    
    Args:
        ts_code: 股票代码 (可选)
        report_date: 报告期 (YYYYMMDD格式，可选)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 商誉减值预期数据
    """
    collector = GoodwillImpairmentCollector()
    return collector.collect(
        ts_code=ts_code,
        report_date=report_date,
        **kwargs
    )


def get_break_net_stock(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取破净股统计数据
    
    Args:
        start_date: 开始日期 (YYYYMMDD格式)
        end_date: 结束日期 (YYYYMMDD格式)
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 破净股统计数据
    """
    collector = BreakNetStockCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
