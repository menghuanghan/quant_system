"""
事件信号预处理器

处理 dwd_event_signal 表，主要包括：
1. 0/1 信号列 NaN 填充为 0
2. 质押率 Clipping（>100% 截断到 100）
3. 数据验证
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .base import BasePreprocessor
from .config import PreprocessConfig

logger = logging.getLogger(__name__)


class EventPreprocessor(BasePreprocessor):
    """事件信号预处理器"""
    
    def __init__(self, config: PreprocessConfig):
        super().__init__(config)
        self.stats: Dict[str, Any] = {}
    
    def process(self, df: Any) -> Any:
        """
        执行事件信号预处理
        
        处理步骤：
        1. 信号列 NaN 填充为 0
        2. 质押率 Clipping
        3. 数据验证
        
        Args:
            df: 输入的事件信号表 DataFrame
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📊 开始处理事件信号表 (dwd_event_signal)")
        logger.info("=" * 60)
        
        original_shape = df.shape
        df = df.copy()
        
        # 1. 信号列 NaN 填充
        df = self._fill_signal_nan(df)
        
        # 2. 质押率 Clipping
        df = self._clip_pledge_ratio(df)
        
        # 3. 数据验证
        df = self._validate_signals(df)
        
        # 记录统计信息
        self.stats["original_shape"] = original_shape
        self.stats["final_shape"] = df.shape
        
        logger.info(f"✅ 事件信号表处理完成: {original_shape} -> {df.shape}")
        
        return df
    
    def _fill_signal_nan(self, df: Any) -> Any:
        """
        信号列 NaN 填充为 0
        
        0/1 信号列不应有 NaN，NaN 表示"无事件"应为 0
        """
        logger.info("📌 Step 1: 信号列 NaN 填充")
        
        signal_columns = self.config.event.signal_columns
        
        fill_stats = {}
        for col in signal_columns:
            if col in df.columns:
                if self.use_gpu:
                    null_count = int(df[col].isna().sum())
                else:
                    null_count = df[col].isna().sum()
                
                if null_count > 0:
                    df[col] = df[col].fillna(0)
                    fill_stats[col] = null_count
                    logger.info(f"  ✓ {col}: 填充 {null_count:,} 个 NaN 为 0")
                else:
                    logger.info(f"  ✓ {col}: 无缺失")
        
        self.stats["signal_fill_stats"] = fill_stats
        
        return df
    
    def _clip_pledge_ratio(self, df: Any) -> Any:
        """
        质押率 Clipping
        
        pledge_ratio 不应超过 100%
        """
        logger.info("📌 Step 2: 质押率 Clipping")
        
        max_ratio = self.config.event.pledge_ratio_max
        
        pledge_fields = ["pledge_ratio", "pledge_ratio_high"]
        
        for field in pledge_fields:
            if field in df.columns:
                # 统计超过阈值的数量
                if self.use_gpu:
                    over_max = int((df[field] > max_ratio).sum())
                else:
                    over_max = (df[field] > max_ratio).sum()
                
                self.stats[f"{field}_over_100"] = over_max
                
                if over_max > 0:
                    df = self.clip_column(df, field, lower=0, upper=max_ratio, inplace=True)
                    logger.info(f"  ⚠️ {field}: 截断 {over_max:,} 行 (>{max_ratio}%)")
                else:
                    logger.info(f"  ✓ {field}: 无需截断")
        
        return df
    
    def _validate_signals(self, df: Any) -> Any:
        """
        数据验证
        """
        logger.info("📌 Step 3: 数据验证")
        
        # 检查信号列是否全为 0/1
        signal_columns = self.config.event.signal_columns
        
        for col in signal_columns:
            if col in df.columns:
                if self.use_gpu:
                    unique_vals = df[col].dropna().unique().to_pandas().tolist()
                else:
                    unique_vals = df[col].dropna().unique().tolist()
                
                # 检查是否只有 0 和 1
                valid_vals = {0, 1, 0.0, 1.0}
                invalid_vals = set(unique_vals) - valid_vals
                
                if invalid_vals:
                    logger.warning(f"  ⚠️ {col}: 存在非 0/1 值: {invalid_vals}")
                else:
                    # 统计事件数量
                    if self.use_gpu:
                        event_count = int((df[col] == 1).sum())
                    else:
                        event_count = (df[col] == 1).sum()
                    
                    self.stats[f"{col}_event_count"] = event_count
                    logger.info(f"  ✓ {col}: {event_count:,} 个事件")
        
        # 检查质押率分布
        if "pledge_ratio" in df.columns:
            if self.use_gpu:
                nonzero_count = int((df["pledge_ratio"] > 0).sum())
                total_count = len(df)
            else:
                nonzero_count = (df["pledge_ratio"] > 0).sum()
                total_count = len(df)
            
            nonzero_ratio = nonzero_count / total_count if total_count > 0 else 0
            self.stats["pledge_ratio_coverage"] = nonzero_ratio
            
            logger.info(f"  ✓ pledge_ratio: 非零覆盖率 {nonzero_ratio:.2%}")
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取预处理统计信息"""
        return self.stats
