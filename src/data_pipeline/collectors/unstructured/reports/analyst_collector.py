"""
分析师数据采集器

采集分析师排名、业绩和研报记录
"""

import logging
import hashlib
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd

from ..base import UnstructuredCollector

logger = logging.getLogger(__name__)


class AnalystCollector(UnstructuredCollector):
    """
    分析师数据采集器
    
    使用AKShare分析师相关接口
    """
    
    ANALYST_FIELDS = [
        'analyst_id',
        'analyst_name',
        'broker',
        'industry',
        'rank',
        'year',
        'return_1m',
        'return_3m',
        'return_6m',
        'return_12m',
        'report_count',
        'source',
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
    
    def collect_analyst_rank(
        self,
        year: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集分析师排名
        
        Args:
            year: 年份（可选）
        
        Returns:
            分析师排名DataFrame
        """
        try:
            df = self.ak.stock_analyst_rank_em()
            
            if df is None or df.empty:
                return pd.DataFrame(columns=self.ANALYST_FIELDS)
            
            # 映射字段
            result = self._map_analyst_fields(df)
            
            logger.info(f"采集分析师排名: {len(result)} 条")
            
            return result
            
        except Exception as e:
            logger.warning(f"采集分析师排名失败: {e}")
            return pd.DataFrame(columns=self.ANALYST_FIELDS)
    
    def _map_analyst_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """映射分析师字段"""
        result = pd.DataFrame()
        
        # 分析师ID（如果有的话）
        result['analyst_id'] = df.get('分析师代码', df.get('analyst_id', ''))
        if result['analyst_id'].isna().all() or (result['analyst_id'] == '').all():
            # 生成ID
            result['analyst_id'] = df.apply(
                lambda row: self._generate_analyst_id(
                    str(row.get('分析师', '')),
                    str(row.get('研究机构', ''))
                ),
                axis=1
            )
        
        result['analyst_name'] = df.get('分析师', df.get('analyst_name', ''))
        result['broker'] = df.get('研究机构', df.get('broker', ''))
        result['industry'] = df.get('行业', df.get('industry', ''))
        result['rank'] = df.get('排名', df.get('rank', ''))
        result['year'] = df.get('年份', datetime.now().year)
        
        # 收益率
        result['return_1m'] = df.get('1个月收益率', df.get('return_1m', None))
        result['return_3m'] = df.get('3个月收益率', df.get('return_3m', None))
        result['return_6m'] = df.get('6个月收益率', df.get('return_6m', None))
        result['return_12m'] = df.get('12个月收益率', df.get('return_12m', None))
        
        result['report_count'] = df.get('研报数', df.get('report_count', 0))
        result['source'] = 'eastmoney'
        
        return result
    
    def _generate_analyst_id(self, name: str, broker: str) -> str:
        """生成分析师ID"""
        content = f"analyst_{name}_{broker}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def collect_analyst_detail(
        self,
        analyst_id: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集分析师详情（研报记录）
        
        Args:
            analyst_id: 分析师ID
        
        Returns:
            研报记录DataFrame
        """
        try:
            df = self.ak.stock_analyst_detail_em(analyst_id=analyst_id)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            logger.info(f"采集分析师 {analyst_id} 详情: {len(df)} 条")
            
            return df
            
        except Exception as e:
            logger.warning(f"采集分析师详情失败: {e}")
            return pd.DataFrame()
    
    def collect(
        self,
        start_date: str = None,
        end_date: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集分析师数据（主接口）
        
        Returns:
            分析师排名DataFrame
        """
        return self.collect_analyst_rank(**kwargs)


# 便捷函数

def get_analyst_rank() -> pd.DataFrame:
    """
    获取分析师排名
    
    Returns:
        分析师排名DataFrame
    """
    collector = AnalystCollector()
    return collector.collect_analyst_rank()


def get_analyst_detail(analyst_id: str) -> pd.DataFrame:
    """
    获取分析师详情
    
    Args:
        analyst_id: 分析师ID
    
    Returns:
        研报记录DataFrame
    """
    collector = AnalystCollector()
    return collector.collect_analyst_detail(analyst_id)
