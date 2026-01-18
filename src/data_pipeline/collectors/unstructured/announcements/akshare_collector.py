"""
AKShare 公告数据采集器

基于 AKShare 接口采集上市公司公告
"""

import logging
from typing import Optional, List
from datetime import datetime

import pandas as pd

from ..base import (
    UnstructuredCollector,
    AnnouncementCategory,
    DataSourceType,
)

logger = logging.getLogger(__name__)


class AKShareAnnouncementCollector(UnstructuredCollector):
    """
    AKShare 公告采集器
    
    使用 AKShare 的公告相关接口
    """
    
    def __init__(self):
        super().__init__()
        self._ak = None
    
    @property
    def ak(self):
        """懒加载AKShare模块"""
        if self._ak is None:
            try:
                import akshare as ak
                self._ak = ak
                logger.info("AKShare 初始化成功")
            except ImportError:
                logger.error("AKShare 未安装，请运行: pip install akshare")
                raise
        return self._ak
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集公告数据
        
        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            symbols: 股票代码列表（纯数字，如 ['000001', '600000']）
        
        Returns:
            标准化的公告数据DataFrame
        """
        all_data = []
        
        if symbols:
            for symbol in symbols:
                df = self._collect_by_stock(symbol)
                if not df.empty:
                    # 按日期过滤
                    df = self._filter_by_date(df, start_date, end_date)
                    if not df.empty:
                        all_data.append(df)
        else:
            # AKShare 需要逐股票采集，这里获取部分股票列表
            logger.warning("AKShare 全市场公告采集较慢，建议指定股票代码")
            df = self._collect_market_announcements()
            if not df.empty:
                df = self._filter_by_date(df, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        result = self._deduplicate(result)
        
        return self._standardize_dataframe(result, DataSourceType.AKSHARE.value)
    
    def _collect_by_stock(self, symbol: str) -> pd.DataFrame:
        """
        按股票代码采集公告
        
        Args:
            symbol: 股票代码（纯数字）
        
        Returns:
            公告数据DataFrame
        """
        try:
            # 尝试获取个股公告
            # AKShare 的 stock_notice_report 接口
            df = self.ak.stock_notice_report(symbol=symbol)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 映射字段
            df = self._map_fields(df, symbol)
            
            return df
            
        except AttributeError:
            # 接口可能不存在或已更名
            logger.debug(f"AKShare stock_notice_report 接口不可用")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"AKShare 采集 {symbol} 公告失败: {e}")
            return pd.DataFrame()
    
    def _collect_market_announcements(self) -> pd.DataFrame:
        """
        采集市场公告
        
        使用东方财富公告接口
        """
        try:
            # 尝试多个可能的接口
            interfaces = [
                ('stock_gsrl_gsdt_em', {}),  # 东财公司动态
            ]
            
            for func_name, params in interfaces:
                if hasattr(self.ak, func_name):
                    try:
                        func = getattr(self.ak, func_name)
                        df = func(**params)
                        if df is not None and not df.empty:
                            return self._map_market_fields(df)
                    except Exception as e:
                        logger.debug(f"{func_name} 调用失败: {e}")
                        continue
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"AKShare 市场公告采集失败: {e}")
            return pd.DataFrame()
    
    def _map_fields(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """映射个股公告字段"""
        if df.empty:
            return df
        
        result = pd.DataFrame()
        
        # 常见字段映射
        field_candidates = {
            'title': ['公告标题', 'title', '标题', 'notice_title'],
            'ann_date': ['公告日期', 'notice_date', 'date', '日期'],
            'url': ['公告链接', 'url', 'link', '链接'],
        }
        
        for target, sources in field_candidates.items():
            for source in sources:
                if source in df.columns:
                    result[target] = df[source]
                    break
        
        # 添加股票代码
        result['symbol'] = symbol
        result['ts_code'] = self._symbol_to_tscode(symbol)
        result['name'] = ''  # AKShare 可能不提供名称
        
        # 推断类别
        if 'title' in result.columns:
            result['category'] = result['title'].apply(self._infer_category)
            result['is_correction'] = result['title'].apply(self._detect_correction)
        
        result['list_status'] = 'L'
        
        return result
    
    def _map_market_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """映射市场公告字段"""
        if df.empty:
            return df
        
        result = pd.DataFrame()
        
        # 根据实际返回的列进行映射
        column_mapping = {
            '股票代码': 'symbol',
            '股票名称': 'name',
            '公告标题': 'title',
            '公告日期': 'ann_date',
            '公告链接': 'url',
            'code': 'symbol',
            'name': 'name',
        }
        
        for src, dst in column_mapping.items():
            if src in df.columns:
                result[dst] = df[src]
        
        # 生成ts_code
        if 'symbol' in result.columns:
            result['ts_code'] = result['symbol'].apply(
                lambda x: self._symbol_to_tscode(str(x))
            )
        
        # 推断类别
        if 'title' in result.columns:
            result['category'] = result['title'].apply(self._infer_category)
            result['is_correction'] = result['title'].apply(self._detect_correction)
        
        result['list_status'] = 'L'
        
        return result
    
    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """按日期过滤数据"""
        if df.empty or 'ann_date' not in df.columns:
            return df
        
        # 标准化日期
        df['ann_date'] = df['ann_date'].apply(self._standardize_date)
        
        # 过滤
        mask = (df['ann_date'] >= start_date) & (df['ann_date'] <= end_date)
        return df[mask].copy()
    
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
        if any(kw in title for kw in ['并购', '重组', '收购']):
            return AnnouncementCategory.MERGER_ACQUISITION.value
        if '减持' in title:
            return AnnouncementCategory.EQUITY_DECREASE.value
        if '增持' in title:
            return AnnouncementCategory.EQUITY_INCREASE.value
        if '业绩预告' in title:
            return AnnouncementCategory.EARNINGS_FORECAST.value
        
        # 更正类
        if any(kw in title for kw in ['更正', '补充', '修订']):
            return AnnouncementCategory.CORRECTION.value
        
        return AnnouncementCategory.OTHER.value
    
    @staticmethod
    def _symbol_to_tscode(symbol: str) -> str:
        """将纯数字代码转换为带交易所后缀的代码"""
        symbol = str(symbol).zfill(6)
        if symbol.startswith(('0', '2', '3')):
            return f"{symbol}.SZ"
        elif symbol.startswith(('6', '9')):
            return f"{symbol}.SH"
        elif symbol.startswith(('4', '8')):
            return f"{symbol}.BJ"
        return f"{symbol}.SZ"


# 便捷函数

def get_akshare_announcements(
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    获取AKShare公告数据
    
    Args:
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
        symbols: 股票代码列表（纯数字）
    
    Returns:
        公告数据DataFrame
    
    Example:
        >>> df = get_akshare_announcements('2024-01-01', '2024-01-31', 
        ...                                 symbols=['000001'])
    """
    collector = AKShareAnnouncementCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols
    )
