"""
同花顺投资评级采集器

用于补充AKShare不提供的字段：
- 分析师姓名
- 目标价
- 盈利预测

使用AKShare的 stock_investment_rating_ths 接口
"""

import pandas as pd
import logging
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ThsRatingCollector:
    """同花顺投资评级采集器"""
    
    def __init__(self):
        self._ak = None
    
    @property
    def ak(self):
        """懒加载AKShare"""
        if self._ak is None:
            import akshare as ak
            self._ak = ak
        return self._ak
    
    def collect(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        采集同花顺投资评级数据
        
        Args:
            stock_code: 股票代码 (e.g., '600519')
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        
        Returns:
            投资评级DataFrame，包含：
            - analyst: 分析师名称
            - broker: 券商机构
            - rating: 投资评级
            - target_price: 目标价
            - pub_date: 发布日期
        """
        try:
            # 处理代码格式
            if '.' in stock_code:
                code = stock_code.split('.')[1]  # BaoStock格式转换
            else:
                code = stock_code.zfill(6)
            
            # 调用同花顺接口
            df = self.ak.stock_investment_rating_ths(symbol=code)
            
            if df.empty:
                logger.debug(f"同花顺无 {code} 的评级数据")
                return pd.DataFrame()
            
            # 标准化字段
            df = self._standardize_columns(df)
            
            # 按日期过滤
            if start_date and end_date:
                df = self._filter_by_date(df, start_date, end_date)
            
            logger.debug(f"同花顺采集 {code} 评级: {len(df)} 条")
            return df
            
        except Exception as e:
            logger.debug(f"同花顺采集失败 {stock_code}: {e}")
            return pd.DataFrame()
    
    def collect_batch(
        self,
        stock_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        批量采集多只股票的评级数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            合并后的评级DataFrame
        """
        all_data = []
        
        for code in stock_codes:
            df = self.collect(code, start_date, end_date)
            if not df.empty:
                df['stock_code'] = code
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化列名
        
        同花顺API返回列名可能包括：
        - 序号, 股票代码, 股票简称, 分析师, 券商, 评级, 目标价, 发布日期等
        """
        result = pd.DataFrame()
        
        # 分析师 - 可能的列名
        for col in ['分析师', '研究员', 'analyst']:
            if col in df.columns:
                result['analyst'] = df[col]
                break
        if 'analyst' not in result.columns:
            result['analyst'] = ''
        
        # 券商 - 可能的列名
        for col in ['券商', '机构', '证券公司', 'broker']:
            if col in df.columns:
                result['broker'] = df[col]
                break
        if 'broker' not in result.columns:
            result['broker'] = ''
        
        # 评级 - 可能的列名
        for col in ['评级', '投资评级', '研究评级', 'rating']:
            if col in df.columns:
                result['rating'] = df[col]
                break
        if 'rating' not in result.columns:
            result['rating'] = ''
        
        # 目标价 - 可能的列名
        for col in ['目标价', '目标价格', '目标价(元)', 'target_price']:
            if col in df.columns:
                result['target_price'] = pd.to_numeric(df[col], errors='coerce')
                break
        if 'target_price' not in result.columns:
            result['target_price'] = None
        
        # 发布日期 - 可能的列名
        for col in ['发布日期', '日期', '评级日期', 'pub_date']:
            if col in df.columns:
                result['pub_date'] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                break
        if 'pub_date' not in result.columns:
            result['pub_date'] = ''
        
        return result
    
    def _filter_by_date(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """按日期范围过滤"""
        if 'pub_date' not in df.columns or df.empty:
            return df
        
        df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
        df = df[(df['pub_date'] >= start_date) & (df['pub_date'] <= end_date)]
        
        return df.reset_index(drop=True)


# 便捷函数

def get_stock_ths_ratings(
    stock_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取同花顺股票投资评级
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        投资评级DataFrame
    """
    collector = ThsRatingCollector()
    return collector.collect(stock_code, start_date, end_date)


def enrich_reports_with_ths_data(
    reports_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    使用同花顺数据补充研报中缺失的字段
    
    Args:
        reports_df: 原始研报DataFrame (需包含 'stock_code' 列)
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        补充后的研报DataFrame
    """
    if reports_df.empty or 'stock_code' not in reports_df.columns:
        return reports_df
    
    result = reports_df.copy()
    
    # 获取所有涉及的股票代码
    stock_codes = result['stock_code'].unique().tolist()
    
    # 批量采集同花顺数据
    collector = ThsRatingCollector()
    ths_data = collector.collect_batch(stock_codes, start_date, end_date)
    
    if ths_data.empty:
        logger.warning("同花顺无相关评级数据，返回原始研报")
        return result
    
    # 按股票代码合并
    # 对每只股票，将最新的同花顺评级应用到对应的研报
    for stock_code in stock_codes:
        stock_reports = result[result['stock_code'] == stock_code].index
        stock_ths = ths_data[ths_data['stock_code'] == stock_code]
        
        if not stock_ths.empty:
            # 使用最新的评级数据
            latest_ths = stock_ths.sort_values('pub_date', ascending=False).iloc[0]
            
            for idx in stock_reports:
                # 补充缺失的分析师
                if not result.loc[idx, 'analyst'] and latest_ths['analyst']:
                    result.loc[idx, 'analyst'] = latest_ths['analyst']
                
                # 补充缺失的目标价
                if not result.loc[idx, 'target_price'] and latest_ths['target_price']:
                    result.loc[idx, 'target_price'] = latest_ths['target_price']
    
    logger.info(f"已补充 {len(stock_codes)} 只股票的同花顺数据")
    return result
