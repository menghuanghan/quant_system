"""
评级变化对比模块

通过对比历史评级数据，识别评级是否发生了变化。
支持以下变化类型：
- 上调 (Upgrade): 从低评级升到高评级
- 下调 (Downgrade): 从高评级降到低评级
- 维持 (Maintain): 评级保持不变
- 首次 (Initiate): 首次给出评级
"""

import pandas as pd
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# 评级等级定义
RATING_HIERARCHY = {
    '买入': 5,
    '强烈推荐': 5,
    '强推': 5,
    '增持': 4,
    '推荐': 4,
    '优于大市': 4,
    '中性': 3,
    '持有': 3,
    '同步大市': 3,
    '减持': 2,
    '弱于大市': 2,
    '卖出': 1,
    '未知': 0,
}


class RatingChangeTracker:
    """评级变化追踪器"""
    
    def __init__(self, history_path: Optional[str] = None):
        """
        初始化追踪器
        
        Args:
            history_path: 评级历史数据存储路径 (可选)
        """
        self.history_path = history_path
        self.history = self._load_history()
    
    def _load_history(self) -> pd.DataFrame:
        """从磁盘加载评级历史"""
        if not self.history_path:
            return pd.DataFrame()
        
        history_file = Path(self.history_path)
        
        if not history_file.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(history_file, parse_dates=['pub_date'])
            logger.info(f"已加载 {len(df)} 条评级历史记录")
            return df
        except Exception as e:
            logger.warning(f"加载评级历史失败: {e}")
            return pd.DataFrame()
    
    def save_history(self) -> bool:
        """保存评级历史到磁盘"""
        if not self.history_path or self.history.empty:
            return False
        
        try:
            history_file = Path(self.history_path)
            history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history.to_csv(history_file, index=False)
            logger.info(f"已保存 {len(self.history)} 条评级历史")
            return True
        except Exception as e:
            logger.error(f"保存评级历史失败: {e}")
            return False
    
    def get_previous_rating(
        self,
        stock_code: str,
        broker: str,
        pub_date: str
    ) -> Optional[Dict]:
        """
        获取特定券商对某只股票之前的评级
        
        Args:
            stock_code: 股票代码
            broker: 券商名称
            pub_date: 当前发布日期 (YYYY-MM-DD)
        
        Returns:
            上一条评级记录 {rating, date} 或 None
        """
        if self.history.empty:
            return None
        
        # 过滤相同股票和券商
        same_stock_broker = self.history[
            (self.history['stock_code'] == stock_code) & 
            (self.history['broker'] == broker)
        ]
        
        if same_stock_broker.empty:
            return None
        
        # 转换日期格式
        try:
            current_date = pd.to_datetime(pub_date)
        except:
            return None
        
        # 找到当前日期之前的最近一条记录
        same_stock_broker = same_stock_broker[
            same_stock_broker['pub_date'] < current_date
        ]
        
        if same_stock_broker.empty:
            return None
        
        latest = same_stock_broker.sort_values('pub_date').iloc[-1]
        
        return {
            'rating': latest['rating'],
            'date': latest['pub_date'].strftime('%Y-%m-%d')
        }
    
    def detect_rating_change(
        self,
        stock_code: str,
        broker: str,
        current_rating: str,
        pub_date: str
    ) -> str:
        """
        检测评级变化
        
        Args:
            stock_code: 股票代码
            broker: 券商名称
            current_rating: 当前评级
            pub_date: 发布日期
        
        Returns:
            变化类型: '上调', '下调', '维持', '首次', '未知'
        """
        # 规范化当前评级
        current_rating = self._normalize_rating(current_rating)
        
        # 获取历史评级
        prev = self.get_previous_rating(stock_code, broker, pub_date)
        
        if not prev:
            # 没有历史记录，说明是首次评级或历史数据不足
            return '首次'
        
        prev_rating = prev['rating']
        
        # 获取评级等级
        current_level = RATING_HIERARCHY.get(current_rating, 0)
        prev_level = RATING_HIERARCHY.get(prev_rating, 0)
        
        if current_level > prev_level:
            return '上调'
        elif current_level < prev_level:
            return '下调'
        else:
            return '维持'
    
    def add_records(self, reports_df: pd.DataFrame) -> None:
        """
        将新的研报记录添加到历史中
        
        Args:
            reports_df: 研报DataFrame (需包含 stock_code, broker, rating, pub_date)
        """
        required_cols = ['stock_code', 'broker', 'rating', 'pub_date']
        
        for col in required_cols:
            if col not in reports_df.columns:
                logger.warning(f"缺少必需列: {col}")
                return
        
        # 仅保留相关列
        new_records = reports_df[required_cols].copy()
        
        # 去重（基于stock_code, broker, pub_date）
        new_records = new_records.drop_duplicates(
            subset=['stock_code', 'broker', 'pub_date']
        )
        
        # 转换日期
        new_records['pub_date'] = pd.to_datetime(new_records['pub_date'])
        
        # 合并历史
        if self.history.empty:
            self.history = new_records
        else:
            self.history = pd.concat([self.history, new_records], ignore_index=True)
            self.history = self.history.drop_duplicates(
                subset=['stock_code', 'broker', 'pub_date'],
                keep='last'
            )
        
        logger.debug(f"已添加 {len(new_records)} 条新记录，历史总计 {len(self.history)} 条")
    
    def detect_changes_batch(
        self,
        reports_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        批量检测研报中的评级变化
        
        Args:
            reports_df: 研报DataFrame
        
        Returns:
            添加了 'rating_change' 列的DataFrame
        """
        if reports_df.empty:
            return reports_df
        
        result = reports_df.copy()
        
        # 确保必需列存在
        required_cols = ['stock_code', 'broker', 'rating', 'pub_date']
        for col in required_cols:
            if col not in result.columns:
                logger.warning(f"缺少必需列: {col}")
                result['rating_change'] = '未知'
                return result
        
        # 检测每条记录的评级变化
        changes = []
        for idx, row in result.iterrows():
            change = self.detect_rating_change(
                row['stock_code'],
                row['broker'],
                row['rating'],
                row['pub_date']
            )
            changes.append(change)
        
        result['rating_change'] = changes
        
        # 添加记录到历史
        self.add_records(result)
        
        return result
    
    def get_rating_stats(self, stock_code: str) -> Dict:
        """
        获取某只股票的评级统计
        
        Args:
            stock_code: 股票代码
        
        Returns:
            统计信息字典
        """
        if self.history.empty:
            return {}
        
        stock_history = self.history[self.history['stock_code'] == stock_code]
        
        if stock_history.empty:
            return {}
        
        return {
            'total_ratings': len(stock_history),
            'unique_brokers': stock_history['broker'].nunique(),
            'latest_date': stock_history['pub_date'].max().strftime('%Y-%m-%d'),
            'latest_rating': stock_history.sort_values('pub_date').iloc[-1]['rating'],
            'rating_distribution': stock_history['rating'].value_counts().to_dict()
        }
    
    def _normalize_rating(self, rating: str) -> str:
        """规范化评级"""
        if not rating or pd.isna(rating):
            return '未知'
        
        rating = str(rating).strip()
        
        # 检查是否在已知评级中
        if rating in RATING_HIERARCHY:
            return rating
        
        # 尝试模糊匹配
        rating_lower = rating.lower()
        for known_rating in RATING_HIERARCHY.keys():
            if known_rating in rating:
                return known_rating
        
        return rating


# 便捷函数

def create_rating_tracker(history_path: Optional[str] = None) -> RatingChangeTracker:
    """创建评级变化追踪器"""
    return RatingChangeTracker(history_path)


def detect_rating_changes(
    reports_df: pd.DataFrame,
    history_path: Optional[str] = None
) -> pd.DataFrame:
    """
    检测研报中的评级变化
    
    Args:
        reports_df: 研报DataFrame
        history_path: 历史数据路径
    
    Returns:
        添加了 'rating_change' 列的DataFrame
    """
    tracker = RatingChangeTracker(history_path)
    return tracker.detect_changes_batch(reports_df)
