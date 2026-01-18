"""
Tushare 公告数据采集器

基于 Tushare anns_d 接口采集上市公司公告：
- 支持按日期采集
- 支持按股票采集
- 支持退市公司公告采集
"""

import os
import logging
from typing import Optional, List
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv

from ..base import (
    UnstructuredCollector,
    AnnouncementCategory,
    DataSourceType,
    DateRangeIterator,
    parse_date_range
)

load_dotenv()

logger = logging.getLogger(__name__)


class TushareAnnouncementCollector(UnstructuredCollector):
    """
    Tushare 公告采集器
    
    基于 anns_d 接口获取上市公司公告
    注意：此接口需要单独权限申请
    """
    
    def __init__(self):
        super().__init__()
        self._api = None
        self._stock_list_cache = None
        self._cache_time = None
    
    @property
    def api(self):
        """懒加载Tushare API"""
        if self._api is None:
            try:
                import tushare as ts
                token = os.getenv('TUSHARE_TOKEN')
                if not token:
                    raise ValueError("TUSHARE_TOKEN 环境变量未设置")
                ts.set_token(token)
                self._api = ts.pro_api()
                logger.info("Tushare API 初始化成功")
            except Exception as e:
                logger.error(f"Tushare API 初始化失败: {e}")
                raise
        return self._api
    
    def _get_stock_list(
        self,
        list_status: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            list_status: 上市状态 L=上市 D=退市 P=暂停上市
            use_cache: 是否使用缓存
        
        Returns:
            股票列表DataFrame
        """
        cache_key = f"stock_list_{list_status}"
        
        # 检查缓存
        if use_cache and self._stock_list_cache is not None:
            if self._cache_time and (datetime.now() - self._cache_time).seconds < 3600:
                return self._stock_list_cache
        
        try:
            params = {}
            if list_status:
                params['list_status'] = list_status
            
            df = self.api.stock_basic(
                **params,
                fields='ts_code,symbol,name,list_status,list_date,delist_date'
            )
            
            if use_cache:
                self._stock_list_cache = df
                self._cache_time = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        ts_codes: Optional[List[str]] = None,
        include_delisted: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集公告数据
        
        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            ts_codes: 股票代码列表（为空则采集全市场）
            include_delisted: 是否包含退市公司
        
        Returns:
            标准化的公告数据DataFrame
        """
        all_data = []
        
        if ts_codes:
            # 按股票采集
            for ts_code in ts_codes:
                df = self._collect_by_stock(ts_code, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
        else:
            # 按日期范围采集全市场
            df = self._collect_by_date_range(start_date, end_date)
            if not df.empty:
                all_data.append(df)
            
            # 额外采集退市公司
            if include_delisted:
                delisted_df = self._collect_delisted(start_date, end_date)
                if not delisted_df.empty:
                    all_data.append(delisted_df)
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        result = self._deduplicate(result)
        
        return self._standardize_dataframe(result, DataSourceType.TUSHARE.value)
    
    def _collect_by_date(self, ann_date: str) -> pd.DataFrame:
        """
        按日期采集单日公告
        
        Args:
            ann_date: 公告日期（YYYYMMDD 格式）
        
        Returns:
            公告数据DataFrame
        """
        try:
            df = self.api.anns_d(ann_date=ann_date)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 字段映射
            df = self._map_fields(df)
            
            return df
            
        except Exception as e:
            logger.warning(f"采集 {ann_date} 公告失败: {e}")
            return pd.DataFrame()
    
    def _collect_by_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        按日期范围采集公告
        
        由于 anns_d 接口单次最多2000条，需分批采集
        """
        all_data = []
        
        # 转换日期格式
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start_dt
        while current <= end_dt:
            ann_date = current.strftime('%Y%m%d')
            df = self._collect_by_date(ann_date)
            
            if not df.empty:
                all_data.append(df)
                logger.debug(f"采集 {ann_date} 公告 {len(df)} 条")
            
            current += timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def _collect_by_stock(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        按股票代码采集公告
        
        Args:
            ts_code: 股票代码（如 000001.SZ）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            公告数据DataFrame
        """
        try:
            # 转换日期格式
            start = start_date.replace('-', '')
            end = end_date.replace('-', '')
            
            df = self.api.anns_d(
                ts_code=ts_code,
                start_date=start,
                end_date=end
            )
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            df = self._map_fields(df)
            
            return df
            
        except Exception as e:
            logger.warning(f"采集 {ts_code} 公告失败: {e}")
            return pd.DataFrame()
    
    def _collect_delisted(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        采集退市公司的历史公告
        
        消除幸存者偏差
        """
        # 获取退市股票列表
        delisted_stocks = self._get_stock_list(list_status='D')
        
        if delisted_stocks.empty:
            logger.info("无退市股票数据")
            return pd.DataFrame()
        
        logger.info(f"发现 {len(delisted_stocks)} 只退市股票")
        
        all_data = []
        for _, stock in delisted_stocks.iterrows():
            ts_code = stock['ts_code']
            
            # 检查股票退市日期，仅采集有效期内的公告
            delist_date = stock.get('delist_date')
            if delist_date:
                delist_dt = datetime.strptime(str(delist_date)[:8], '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                # 如果退市日期早于查询开始日期，跳过
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                if delist_dt < start_dt:
                    continue
                
                # 调整结束日期为退市日期
                if delist_dt < end_dt:
                    actual_end = delist_dt.strftime('%Y-%m-%d')
                else:
                    actual_end = end_date
            else:
                actual_end = end_date
            
            df = self._collect_by_stock(ts_code, start_date, actual_end)
            
            if not df.empty:
                df['list_status'] = 'D'
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"采集退市公司公告完成，共 {len(result)} 条")
        
        return result
    
    def _map_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        映射Tushare字段到标准字段
        """
        if df.empty:
            return df
        
        # 字段映射
        column_mapping = {
            'ann_date': 'ann_date',
            'ts_code': 'ts_code',
            'name': 'name',
            'title': 'title',
            'url': 'url',
        }
        
        # 重命名存在的列
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        # 添加类别
        if 'title' in df.columns:
            df['category'] = df['title'].apply(self._infer_category)
            df['is_correction'] = df['title'].apply(self._detect_correction)
        
        # 设置默认值
        df['list_status'] = 'L'
        
        return df
    
    def _infer_category(self, title: str) -> str:
        """从标题推断公告类型"""
        if not title:
            return AnnouncementCategory.OTHER.value
        
        title = str(title)
        
        # 定期报告
        if '年度报告' in title or '年报' in title:
            return AnnouncementCategory.PERIODIC_ANNUAL.value
        if '半年度报告' in title or '中报' in title:
            return AnnouncementCategory.PERIODIC_SEMI.value
        if '第一季度' in title or '一季报' in title:
            return AnnouncementCategory.PERIODIC_Q1.value
        if '第三季度' in title or '三季报' in title:
            return AnnouncementCategory.PERIODIC_Q3.value
        
        # 临时公告
        if any(kw in title for kw in ['并购', '重组', '收购', '合并']):
            return AnnouncementCategory.MERGER_ACQUISITION.value
        if any(kw in title for kw in ['增持', '买入']):
            return AnnouncementCategory.EQUITY_INCREASE.value
        if any(kw in title for kw in ['减持', '卖出']):
            return AnnouncementCategory.EQUITY_DECREASE.value
        if '重大合同' in title:
            return AnnouncementCategory.MAJOR_CONTRACT.value
        if any(kw in title for kw in ['诉讼', '仲裁', '起诉']):
            return AnnouncementCategory.LITIGATION.value
        if any(kw in title for kw in ['处罚', '警示', '监管']):
            return AnnouncementCategory.REGULATORY_PENALTY.value
        if '业绩预告' in title:
            return AnnouncementCategory.EARNINGS_FORECAST.value
        if '业绩快报' in title:
            return AnnouncementCategory.EARNINGS_EXPRESS.value
        if any(kw in title for kw in ['分红', '派息', '送股']):
            return AnnouncementCategory.DIVIDEND.value
        
        # 更正类
        if any(kw in title for kw in ['更正', '补充', '修订']):
            return AnnouncementCategory.CORRECTION.value
        
        return AnnouncementCategory.OTHER.value
    
    def collect_by_date(
        self,
        ann_date: str,
        ts_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        按日期采集公告（调度器友好接口）
        
        Args:
            ann_date: 公告日期（YYYY-MM-DD 或 YYYYMMDD）
            ts_codes: 股票代码列表（可选）
        
        Returns:
            公告数据DataFrame
        """
        # 统一日期格式
        ann_date = self._standardize_date(ann_date).replace('-', '')
        
        if ts_codes:
            all_data = []
            for ts_code in ts_codes:
                df = self._collect_by_stock(
                    ts_code,
                    ann_date[:4] + '-' + ann_date[4:6] + '-' + ann_date[6:8],
                    ann_date[:4] + '-' + ann_date[4:6] + '-' + ann_date[6:8]
                )
                if not df.empty:
                    all_data.append(df)
            
            if not all_data:
                return pd.DataFrame(columns=self.STANDARD_FIELDS)
            result = pd.concat(all_data, ignore_index=True)
        else:
            result = self._collect_by_date(ann_date)
        
        return self._standardize_dataframe(result, DataSourceType.TUSHARE.value)


# 便捷函数

def get_tushare_announcements(
    start_date: str,
    end_date: str,
    ts_codes: Optional[List[str]] = None,
    include_delisted: bool = True
) -> pd.DataFrame:
    """
    获取Tushare公告数据
    
    Args:
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
        ts_codes: 股票代码列表
        include_delisted: 是否包含退市公司
    
    Returns:
        公告数据DataFrame
    
    Example:
        >>> df = get_tushare_announcements('2024-01-01', '2024-01-31')
        >>> df = get_tushare_announcements('2024-01-01', '2024-12-31', 
        ...                                 ts_codes=['000001.SZ'])
    """
    collector = TushareAnnouncementCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        ts_codes=ts_codes,
        include_delisted=include_delisted
    )
