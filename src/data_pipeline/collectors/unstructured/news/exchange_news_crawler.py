"""
交易所公告新闻采集器

使用AKShare获取上交所/深交所公告数据
"""

import logging
import hashlib
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd

from ..base import UnstructuredCollector
from ..cleaning_adapter import CleaningMixin
from .cctv_collector import NewsCategory

logger = logging.getLogger(__name__)


class ExchangeNewsCrawler(UnstructuredCollector, CleaningMixin):
    """
    交易所公告采集器
    
    使用AKShare stock_notice_report接口获取公告数据
    """
    
    STANDARD_FIELDS = [
        'news_id',
        'title',
        'content',
        'summary',
        'pub_time',
        'pub_date',
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
            import akshare as ak
            self._ak = ak
        return self._ak
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        exchanges: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集交易所公告
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            exchanges: 交易所列表（暂不使用，接口返回全部）
        
        Returns:
            公告数据DataFrame
        """
        all_data = []
        
        # 按日期逐天采集
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start_dt
        while current <= end_dt:
            date_str = current.strftime('%Y%m%d')
            
            df = self._collect_by_date(date_str)
            if not df.empty:
                all_data.append(df)
                logger.debug(f"采集 {date_str} 公告: {len(df)} 条")
            
            current += timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['news_id'], keep='first')
        
        return self._standardize_output(result)
    
    def _collect_by_date(self, date_str: str) -> pd.DataFrame:
        """
        采集单日公告（带重试机制）
        
        Args:
            date_str: 日期（YYYYMMDD格式）
        
        Returns:
            公告DataFrame
        """
        max_retries = 3
        retry_delay = 2  # 秒
        
        for attempt in range(max_retries):
            try:
                df = self.ak.stock_notice_report(symbol='全部', date=date_str)
                
                if df is None or df.empty:
                    return pd.DataFrame()
                
                # 映射字段
                result = self._map_fields(df, date_str)
                
                return result
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # 周末或节假日可能没有公告
                if "无数据" not in str(e) and "empty" not in str(e).lower():
                    logger.debug(f"采集 {date_str} 公告失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(retry_delay)
                        continue
                return pd.DataFrame()
        
        return pd.DataFrame()
    
    def _map_fields(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """映射字段到标准格式"""
        result = pd.DataFrame()
        
        # 基本字段映射
        result['title'] = df.get('公告标题', df.get('title', ''))
        result['related_stocks'] = df.get('代码', df.get('code', '')).astype(str) + '.' + \
                                   df.get('名称', df.get('name', ''))
        
        # 生成唯一ID
        result['news_id'] = result['title'].apply(
            lambda x: self._generate_id(str(x), date_str)
        )
        
        # 从公告标题生成摘要
        result['summary'] = result['title'].apply(
            lambda x: str(x)[:100] if len(str(x)) > 100 else str(x)
        )
        
        result['content'] = ''  # 公告API不返回内容
        
        # 公告类型
        result['category'] = df.get('公告类型', '').apply(self._map_category)
        
        # 日期
        pub_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        result['pub_date'] = pub_date
        result['pub_time'] = pub_date
        
        # 来源和URL
        result['source'] = 'exchange'
        result['url'] = df.get('公告链接', df.get('url', ''))
        result['keywords'] = df.get('公告类型', '')
        
        return result
    
    def _map_category(self, ann_type: str) -> str:
        """映射公告类型"""
        if not ann_type:
            return NewsCategory.COMPANY
        
        ann_type = str(ann_type)
        
        if '年报' in ann_type or '季报' in ann_type or '中报' in ann_type:
            return '定期报告'
        elif '预告' in ann_type or '快报' in ann_type:
            return '业绩预告'
        elif '增持' in ann_type or '减持' in ann_type:
            return '股东变动'
        elif '并购' in ann_type or '重组' in ann_type:
            return '并购重组'
        elif '回购' in ann_type:
            return '股份回购'
        else:
            return NewsCategory.COMPANY
    
    def _generate_id(self, title: str, date_str: str) -> str:
        """生成唯一ID"""
        content = f"exchange_{date_str}_{title}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化输出"""
        for col in self.STANDARD_FIELDS:
            if col not in df.columns:
                df[col] = ''
        
        return df[self.STANDARD_FIELDS]
    
    def collect_recent(self, days: int = 7) -> pd.DataFrame:
        """
        采集最近N天的公告
        
        Args:
            days: 天数
        
        Returns:
            公告DataFrame
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        return self.collect(start_date, end_date)


# 便捷函数

def get_exchange_news(
    start_date: str,
    end_date: str,
    exchanges: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    获取交易所公告
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        exchanges: 交易所列表
    
    Returns:
        公告DataFrame
    """
    crawler = ExchangeNewsCrawler()
    return crawler.collect(
        start_date=start_date,
        end_date=end_date,
        exchanges=exchanges
    )
