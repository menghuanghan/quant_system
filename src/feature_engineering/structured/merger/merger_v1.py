"""
数据合并与股票池过滤

将三张 DWD 宽表合并为一张主表，并根据条件过滤不适合训练的样本。
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DataMerger:
    """数据合并与股票池过滤器"""
    
    def __init__(self, config, use_gpu: bool = True):
        """
        初始化合并器
        
        Args:
            config: UniverseFilterConfig 配置
            use_gpu: 是否使用 GPU
        """
        self.config = config
        self.use_gpu = use_gpu
        self.stats: Dict[str, Any] = {}
        
        if use_gpu:
            try:
                import cudf
                self.pd = cudf
                logger.info("🚀 DataMerger: GPU 加速已启用 (cuDF)")
            except ImportError:
                import pandas as pd
                self.pd = pd
                self.use_gpu = False
                logger.warning("⚠️ cuDF 不可用，回退到 pandas")
        else:
            import pandas as pd
            self.pd = pd
            logger.info("DataMerger: 使用 CPU 模式 (pandas)")
    
    def _unify_date_type(self, df: Any, date_col: str = "trade_date") -> Any:
        """
        统一日期列的类型为字符串格式 (YYYY-MM-DD)
        
        Args:
            df: DataFrame
            date_col: 日期列名
            
        Returns:
            处理后的 DataFrame
        """
        if date_col not in df.columns:
            return df
        
        if df[date_col].dtype == 'object':
            # 已经是字符串，不需要处理
            pass
        elif hasattr(df[date_col].dtype, 'name') and 'datetime' in df[date_col].dtype.name:
            # datetime64 转字符串
            df = df.copy()
            df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')
        
        return df
    
    def merge_tables(
        self,
        price_df: Any,
        fundamental_df: Any,
        status_df: Any
    ) -> Any:
        """
        合并三张宽表
        
        以 price 表为主表，left join fundamental 和 status 表。
        主键：ts_code + trade_date
        
        Args:
            price_df: 价格表
            fundamental_df: 基本面表 
            status_df: 状态表
            
        Returns:
            合并后的宽表
        """
        logger.info("=" * 60)
        logger.info("📋 Step 1: 数据合并")
        logger.info("=" * 60)
        
        # 统一日期类型（合并前必须）
        price_df = self._unify_date_type(price_df)
        fundamental_df = self._unify_date_type(fundamental_df)
        status_df = self._unify_date_type(status_df)
        
        original_rows = len(price_df)
        
        # 选择加入的基本面字段（避免重复列）
        fundamental_cols = ['ts_code', 'trade_date']
        for col in fundamental_df.columns:
            if col not in ['ts_code', 'trade_date'] and col not in price_df.columns:
                fundamental_cols.append(col)
        
        # 选择加入的状态表字段
        status_cols = ['ts_code', 'trade_date']
        for col in status_df.columns:
            if col not in ['ts_code', 'trade_date'] and col not in price_df.columns:
                status_cols.append(col)
        
        logger.info(f"  📊 价格表: {len(price_df):,} 行, {len(price_df.columns)} 列")
        logger.info(f"  📊 基本面表: {len(fundamental_df):,} 行, {len(fundamental_cols)-2} 列加入")
        logger.info(f"  📊 状态表: {len(status_df):,} 行, {len(status_cols)-2} 列加入")
        
        # Left Join 基本面表
        fundamental_subset = fundamental_df[fundamental_cols]
        df = price_df.merge(
            fundamental_subset,
            on=['ts_code', 'trade_date'],
            how='left'
        )
        
        logger.info(f"  ✓ 合并基本面表后: {len(df):,} 行")
        
        # Left Join 状态表
        status_subset = status_df[status_cols]
        df = df.merge(
            status_subset,
            on=['ts_code', 'trade_date'],
            how='left'
        )
        
        logger.info(f"  ✓ 合并状态表后: {len(df):,} 行, {len(df.columns)} 列")
        
        self.stats["merge_original_rows"] = original_rows
        self.stats["merge_final_rows"] = len(df)
        self.stats["merge_columns"] = len(df.columns)
        
        return df
    
    def filter_universe(self, df: Any) -> Any:
        """
        股票池过滤
        
        根据配置剔除不适合训练的样本。
        
        Args:
            df: 合并后的宽表
            
        Returns:
            过滤后的宽表
        """
        logger.info("=" * 60)
        logger.info("📋 Step 2: 股票池过滤")
        logger.info("=" * 60)
        
        original_rows = len(df)
        filter_stats = {}
        
        # 1. 剔除停牌股 (vol = 0 或 NaN)
        if self.config.exclude_suspended:
            if 'vol' in df.columns:
                mask = (df['vol'] > 0) & df['vol'].notna()
                removed = (~mask).sum()
                if self.use_gpu:
                    removed = int(removed)
                df = df[mask]
                filter_stats["suspended"] = removed
                logger.info(f"  🚫 剔除停牌股: {removed:,} 行 (vol = 0 或 NaN)")
        
        # 2. 剔除 ST 股
        if self.config.exclude_st:
            if 'is_st' in df.columns:
                mask = df['is_st'] != 1
                removed = (~mask).sum()
                if self.use_gpu:
                    removed = int(removed)
                df = df[mask]
                filter_stats["st"] = removed
                logger.info(f"  🚫 剔除 ST 股: {removed:,} 行")
        
        # 3. 剔除次新股 (上市不满 N 天)
        if self.config.exclude_new:
            if 'is_new' in df.columns:
                mask = df['is_new'] != 1
                removed = (~mask).sum()
                if self.use_gpu:
                    removed = int(removed)
                df = df[mask]
                filter_stats["new"] = removed
                logger.info(f"  🚫 剔除次新股: {removed:,} 行 (上市 < {self.config.new_days_threshold} 天)")
        
        # 4. 剔除涨跌停股 (可选)
        if self.config.exclude_limit:
            if 'is_limit_up' in df.columns and 'is_limit_down' in df.columns:
                mask = (df['is_limit_up'] != 1) & (df['is_limit_down'] != 1)
                removed = (~mask).sum()
                if self.use_gpu:
                    removed = int(removed)
                df = df[mask]
                filter_stats["limit"] = removed
                logger.info(f"  🚫 剔除涨跌停: {removed:,} 行")
        
        final_rows = len(df)
        total_removed = original_rows - final_rows
        retention_rate = final_rows / original_rows * 100
        
        logger.info(f"  📊 过滤统计:")
        logger.info(f"     原始样本: {original_rows:,}")
        logger.info(f"     剔除总数: {total_removed:,}")
        logger.info(f"     保留样本: {final_rows:,} ({retention_rate:.1f}%)")
        
        self.stats["filter_original"] = original_rows
        self.stats["filter_final"] = final_rows
        self.stats["filter_details"] = filter_stats
        
        return df
    
    def drop_columns(self, df: Any, columns: list) -> Any:
        """
        删除指定的列
        
        Args:
            df: 输入数据框
            columns: 要删除的列名列表
            
        Returns:
            删除列后的数据框
        """
        cols_to_drop = [c for c in columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"  🗑️ 删除字段: {cols_to_drop}")
        return df
    
    def process(
        self,
        price_df: Any,
        fundamental_df: Any,
        status_df: Any,
        drop_columns: list = None
    ) -> Any:
        """
        执行完整的合并与过滤流程
        
        Args:
            price_df: 价格表
            fundamental_df: 基本面表
            status_df: 状态表
            drop_columns: 要删除的列名列表
            
        Returns:
            处理后的主表
        """
        # 1. 合并
        df = self.merge_tables(price_df, fundamental_df, status_df)
        
        # 2. 删除指定列（如原始估值字段 pe_ttm/pb/ps_ttm）
        if drop_columns:
            df = self.drop_columns(df, drop_columns)
        
        # 3. 过滤
        df = self.filter_universe(df)
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.stats
