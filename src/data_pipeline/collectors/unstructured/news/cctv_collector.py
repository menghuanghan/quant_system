"""
央视新闻联播文字稿采集器

基于 AKShare news_cctv 接口采集新闻联播内容
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta
import hashlib

import pandas as pd

from ..base import UnstructuredCollector, DataSourceType, DateRangeIterator

logger = logging.getLogger(__name__)


class NewsCategory:
    """新闻分类"""
    MACRO = "宏观政策"
    STOCK = "个股新闻"
    INDUSTRY = "行业动态"
    MARKET = "市场行情"
    COMPANY = "公司公告解读"
    CCTV = "新闻联播"
    OTHER = "其他"


class CCTVNewsCollector(UnstructuredCollector):
    """
    央视新闻联播文字稿采集器
    
    数据来源：AKShare news_cctv 接口
    数据范围：2016年2月3日至今
    """
    
    STANDARD_FIELDS = [
        'news_id',
        'title',
        'content',
        'date',
        'source',
        'category',
        'url',
        'related_stocks',
        'keywords',
    ]
    
    def __init__(self):
        super().__init__()
        self._ak = None
    
    @property
    def ak(self):
        """懒加载AKShare"""
        if self._ak is None:
            try:
                import akshare as ak
                self._ak = ak
                logger.info("AKShare 初始化成功")
            except ImportError:
                raise ImportError("AKShare 未安装，请运行: pip install akshare")
        return self._ak
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集新闻联播文字稿
        
        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
        
        Returns:
            标准化的新闻数据DataFrame
        """
        all_data = []
        
        # 按日期逐天采集
        # 支持YYYYMMDD和YYYY-MM-DD格式
        if len(start_date) == 8 and '-' not in start_date:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
        else:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        
        if len(end_date) == 8 and '-' not in end_date:
            end_dt = datetime.strptime(end_date, '%Y%m%d')
        else:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start_dt
        while current <= end_dt:
            date_str = current.strftime('%Y%m%d')
            
            df = self._collect_by_date(date_str)
            if not df.empty:
                all_data.append(df)
                logger.debug(f"采集 {date_str} 新闻联播: {len(df)} 条")
            
            current += timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        return self._standardize_output(result)
    
    def _collect_by_date(self, date_str: str) -> pd.DataFrame:
        """
        采集单日新闻联播
        
        Args:
            date_str: 日期（YYYYMMDD格式）
        
        Returns:
            新闻数据DataFrame
        """
        try:
            df = self.ak.news_cctv(date=date_str)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 映射字段
            df = self._map_fields(df, date_str)
            
            return df
            
        except Exception as e:
            # 周末或节假日可能没有新闻联播
            if "无数据" not in str(e) and "empty" not in str(e).lower():
                logger.debug(f"采集 {date_str} 新闻联播: {e}")
            return pd.DataFrame()
    
    def _map_fields(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """映射字段到标准格式"""
        result = pd.DataFrame()
        
        # 基本字段映射
        result['title'] = df.get('title', df.get('标题', ''))
        result['content'] = df.get('content', df.get('内容', ''))
        result['date'] = df.get('date', date_str)
        
        # 生成唯一ID
        result['news_id'] = result.apply(
            lambda row: self._generate_id(row['title'], date_str),
            axis=1
        )
        
        # 设置固定值
        result['source'] = 'cctv'
        result['category'] = NewsCategory.CCTV
        result['url'] = ''
        result['related_stocks'] = ''
        result['keywords'] = ''
        
        return result
    
    def _generate_id(self, title: str, date_str: str) -> str:
        """生成新闻唯一ID"""
        content = f"cctv_{date_str}_{title}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化输出"""
        # 确保所有标准字段存在
        for col in self.STANDARD_FIELDS:
            if col not in df.columns:
                df[col] = ''
        
        return df[self.STANDARD_FIELDS]
    
    def collect_recent(self, days: int = 7) -> pd.DataFrame:
        """
        采集最近N天的新闻联播
        
        Args:
            days: 天数
        
        Returns:
            新闻数据DataFrame
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        return self.collect(start_date, end_date)


# 便捷函数

def get_cctv_news(
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    获取央视新闻联播文字稿
    
    Args:
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
    
    Returns:
        新闻数据DataFrame
    
    Example:
        >>> df = get_cctv_news('2025-01-01', '2025-01-31')
    """
    collector = CCTVNewsCollector()
    return collector.collect(start_date, end_date)


def get_cctv_news_recent(days: int = 7) -> pd.DataFrame:
    """
    获取最近N天的新闻联播
    
    Args:
        days: 天数
    
    Returns:
        新闻数据DataFrame
    """
    collector = CCTVNewsCollector()
    return collector.collect_recent(days)
