"""
东方财富研报采集器

基于 AKShare 接口采集券商研报数据（仅采集元数据）
"""

import logging
import hashlib
import socket
import os
from typing import Optional, List
from datetime import datetime

import pandas as pd

from ..base import UnstructuredCollector

logger = logging.getLogger(__name__)

# 设置默认超时
socket.setdefaulttimeout(30)

# 禁用代理（如果有问题）
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'


class ReportRating:
    """研报评级"""
    BUY = "买入"
    OVERWEIGHT = "增持"
    NEUTRAL = "中性"
    UNDERWEIGHT = "减持"
    SELL = "卖出"
    HOLD = "持有"
    OUTPERFORM = "推荐"
    UNKNOWN = "未知"


class RatingChange:
    """评级变化"""
    UPGRADE = "上调"
    DOWNGRADE = "下调"
    MAINTAIN = "维持"
    INITIATE = "首次"
    UNKNOWN = "未知"


class EastMoneyReportCollector(UnstructuredCollector):
    """
    东方财富研报采集器
    
    使用AKShare stock_research_report_em接口
    
    数据源限制（2025-01版）：
    - analyst字段：AKShare接口不返回分析师姓名，需从PDF或其他渠道补充
    - target_price字段：接口不返回目标价，建议使用同花顺等数据源
    - rating_change字段：接口不返回评级变化，需对比历史数据计算
    """
    
    REPORT_FIELDS = [
        'report_id',
        'title',
        'stock_code',
        'stock_name',
        'broker',
        'analyst',  # 注意：接口不返回，留空
        'rating',
        'rating_change',  # 注意：接口不返回，默认为"未知"
        'target_price',  # 注意：接口不返回，留空
        'target_price_low',
        'target_price_high',
        'date',
        'pdf_url',
        'source',
    ]
    
    def __init__(self):
        super().__init__()
        self._ak = None
    
    @property
    def ak(self):
        """懒加载AKShare"""
        if self._ak is None:
            try:
                # 禁用requests的代理
                import requests
                session = requests.Session()
                session.trust_env = False  # 不使用系统代理
                
                import akshare as ak
                self._ak = ak
                logger.info("AKShare 初始化成功")
            except Exception as e:
                logger.error(f"AKShare 初始化失败: {e}")
                raise
        return self._ak
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        stock_codes: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集研报数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表（可选，为空则需要指定）
        
        Returns:
            研报数据DataFrame
        """
        if not stock_codes:
            logger.warning("未指定股票代码，请提供stock_codes参数")
            return pd.DataFrame(columns=self.REPORT_FIELDS)
        
        all_data = []
        
        for i, code in enumerate(stock_codes):
            logger.info(f"采集研报 [{i+1}/{len(stock_codes)}]: {code}")
            try:
                df = self._collect_by_stock(code)
                if not df.empty:
                    # 按日期过滤
                    df = self._filter_by_date(df, start_date, end_date)
                    if not df.empty:
                        all_data.append(df)
                        logger.info(f"  {code}: {len(df)} 条研报")
            except Exception as e:
                logger.warning(f"  {code}: 采集失败 - {e}")
                continue
        
        if not all_data:
            return pd.DataFrame(columns=self.REPORT_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['report_id'], keep='first')
        
        return self._standardize_output(result)
    
    def _collect_by_stock(self, stock_code: str) -> pd.DataFrame:
        """
        采集单只股票的研报
        
        Args:
            stock_code: 股票代码
        
        Returns:
            研报DataFrame
        """
        # 处理代码格式
        if '.' in stock_code:
            code = stock_code.split('.')[0]
        else:
            code = stock_code.zfill(6)
        
        try:
            # 设置请求超时
            import requests
            original_get = requests.get
            
            def timeout_get(*args, **kwargs):
                kwargs.setdefault('timeout', 15)
                kwargs.setdefault('proxies', {'http': None, 'https': None})
                return original_get(*args, **kwargs)
            
            requests.get = timeout_get
            
            try:
                df = self.ak.stock_research_report_em(symbol=code)
            finally:
                requests.get = original_get
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 映射字段
            result = self._map_fields(df, code)
            
            return result
            
        except (socket.timeout, TimeoutError) as e:
            logger.warning(f"采集 {code} 超时: {e}")
            return pd.DataFrame()
        except Exception as e:
            # 检查是否是代理/SSL问题
            err_str = str(e).lower()
            if 'ssl' in err_str or 'proxy' in err_str or 'connect' in err_str:
                logger.warning(f"采集 {code} 网络错误 (可能是代理问题): {e}")
            else:
                logger.warning(f"采集 {code} 研报失败: {e}")
            return pd.DataFrame()
    
    def _map_fields(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """映射字段到标准格式"""
        result = pd.DataFrame()
        n = len(df)
        
        # 辅助函数：安全获取列
        def safe_get(col_names, default=''):
            """安全获取列，支持多个候选列名"""
            if isinstance(col_names, str):
                col_names = [col_names]
            for col in col_names:
                if col in df.columns:
                    return df[col]
            return pd.Series([default] * n)
        
        # 基本字段
        result['title'] = safe_get('报告名称')
        result['stock_code'] = safe_get(['股票代码'], stock_code)
        result['stock_name'] = safe_get('股票简称')
        result['broker'] = safe_get(['机构', '研报机构'])
        
        # 分析师字段 - AKShare接口不直接提供，从机构名提取或留空
        result['analyst'] = safe_get(['研报作者', '作者', '分析师'], '')
        
        # 评级相关
        rating_col = safe_get(['东财评级', '最新评级', '评级'])
        result['rating'] = rating_col.apply(self._normalize_rating)
        
        rating_change_col = safe_get(['评级变化', '评级调整'])
        result['rating_change'] = rating_change_col.apply(self._normalize_rating_change)
        
        # 目标价 - 尝试从多个可能的字段提取
        result['target_price'] = safe_get(['目标价', '目标价格', '最新目标价'], None)
        result['target_price_low'] = safe_get(['目标价格-下限', '目标价-下限', '目标价区间-下限'], None)
        result['target_price_high'] = safe_get(['目标价格-上限', '目标价-上限', '目标价区间-上限'], None)
        
        # 日期 - 特殊处理datetime类型
        date_col = None
        for col_name in ['日期', '发布日期']:
            if col_name in df.columns:
                date_col = df[col_name]
                break
        
        if date_col is not None:
            result['date'] = date_col.apply(self._parse_date)
        else:
            result['date'] = pd.Series([''] * n)
        
        # PDF链接
        result['pdf_url'] = safe_get(['报告PDF链接', '研报链接', 'pdf_url'])
        
        # 生成唯一ID
        result['report_id'] = result.apply(
            lambda row: self._generate_id(str(row['title']), str(row['broker']), str(row['date'])),
            axis=1
        )
        
        result['source'] = 'eastmoney'
        
        return result
    
    def _normalize_rating(self, rating: str) -> str:
        """标准化评级"""
        if not rating or pd.isna(rating):
            return ReportRating.UNKNOWN
        
        rating = str(rating).strip()
        
        # 映射常见评级
        if rating in ['买入', '强烈推荐', '强推']:
            return ReportRating.BUY
        elif rating in ['增持', '推荐', '优于大市']:
            return ReportRating.OVERWEIGHT
        elif rating in ['中性', '持有', '同步大市']:
            return ReportRating.NEUTRAL
        elif rating in ['减持', '弱于大市']:
            return ReportRating.UNDERWEIGHT
        elif rating in ['卖出']:
            return ReportRating.SELL
        else:
            return rating
    
    def _normalize_rating_change(self, change: str) -> str:
        """标准化评级变化"""
        if not change or pd.isna(change):
            return RatingChange.UNKNOWN
        
        change = str(change).strip()
        
        if '上调' in change or '调高' in change:
            return RatingChange.UPGRADE
        elif '下调' in change or '调低' in change:
            return RatingChange.DOWNGRADE
        elif '维持' in change or '持' in change:
            return RatingChange.MAINTAIN
        elif '首次' in change or '新增' in change:
            return RatingChange.INITIATE
        else:
            return change
    
    def _parse_date(self, date_str) -> str:
        """解析日期"""
        if not date_str or pd.isna(date_str):
            return ''
        
        date_str = str(date_str)
        
        # 尝试多种格式
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d']:
            try:
                dt = datetime.strptime(date_str[:10], fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return date_str[:10] if len(date_str) >= 10 else date_str
    
    def _generate_id(self, title: str, broker: str, date: str) -> str:
        """生成唯一ID"""
        content = f"report_{title}_{broker}_{date}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """按日期过滤"""
        if df.empty or 'date' not in df.columns:
            return df
        
        df = df[df['date'] != '']
        
        if df.empty:
            return df
        
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        return df[mask].copy()
    
    def _standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化输出"""
        for col in self.REPORT_FIELDS:
            if col not in df.columns:
                df[col] = ''
        
        return df[self.REPORT_FIELDS]
    
    def collect_market_reports(
        self,
        start_date: str,
        end_date: str,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        采集市场热门股票研报
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            top_n: 采集前N只热门股票
        
        Returns:
            研报DataFrame
        """
        # 获取热门股票（使用固定列表作为示例）
        hot_stocks = [
            '000001', '600000', '000002', '600036', '600519',
            '000858', '601318', '600276', '000333', '002415',
            '600900', '601166', '600030', '000651', '601888'
        ][:top_n]
        
        return self.collect(start_date, end_date, stock_codes=hot_stocks)


# 便捷函数

def get_stock_reports(
    stock_codes: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    获取个股研报
    
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        研报DataFrame
    """
    collector = EastMoneyReportCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes
    )


def get_market_reports(
    start_date: str,
    end_date: str,
    top_n: int = 50
) -> pd.DataFrame:
    """
    获取市场热门研报
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        top_n: 热门股票数量
    
    Returns:
        研报DataFrame
    """
    collector = EastMoneyReportCollector()
    return collector.collect_market_reports(
        start_date=start_date,
        end_date=end_date,
        top_n=top_n
    )


def get_eps_forecast(stock_codes: List[str]) -> pd.DataFrame:
    """
    获取盈利预测（EPS）
    
    使用同花顺 stock_profit_forecast_ths 接口
    
    Args:
        stock_codes: 股票代码列表
    
    Returns:
        盈利预测DataFrame
    """
    import akshare as ak
    
    all_data = []
    
    for code in stock_codes:
        # 处理代码格式
        if '.' in code:
            code = code.split('.')[0]
        else:
            code = code.zfill(6)
        
        try:
            df = ak.stock_profit_forecast_ths(symbol=code)
            
            if df is not None and not df.empty:
                df['stock_code'] = code
                all_data.append(df)
                
        except Exception as e:
            logger.debug(f"获取 {code} 盈利预测失败: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    
    # 重命名字段
    column_mapping = {
        '年度': 'year',
        '预测机构数': 'forecast_count',
        '最小值': 'eps_min',
        '均值': 'eps_avg',
        '最大值': 'eps_max',
        '行业平均数': 'industry_avg',
    }
    result = result.rename(columns=column_mapping)
    
    return result
