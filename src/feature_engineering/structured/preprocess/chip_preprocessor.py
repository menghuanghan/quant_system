"""
筹码结构预处理器

处理 dwd_chip_structure 表，主要包括：
1. 数据截断 (Clipping)：top10_hold_ratio 等超过 100% 的值截断到 100
2. 缺失值填充：holder_num 等字段前向填充 (ffill)
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .base import BasePreprocessor
from .config import PreprocessConfig

logger = logging.getLogger(__name__)


class ChipPreprocessor(BasePreprocessor):
    """筹码结构预处理器"""
    
    def __init__(self, config: PreprocessConfig):
        super().__init__(config)
        self.stats: Dict[str, Any] = {}
    
    def process(self, df: Any) -> Any:
        """
        执行筹码结构预处理
        
        处理步骤：
        1. 比例 Clipping（>100% 截断到 100）
        2. 缺失值前向填充 (ffill)
        3. 股东户数有效性检查
        
        Args:
            df: 输入的筹码结构表 DataFrame
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📊 开始处理筹码结构表 (dwd_chip_structure)")
        logger.info("=" * 60)
        
        original_shape = df.shape
        df = df.copy()
        
        # 1. 比例 Clipping
        df = self._clip_ratios(df)
        
        # 2. 缺失值前向填充
        df = self._ffill_missing_values(df)
        
        # 3. 股东户数有效性检查
        df = self._validate_holder_num(df)
        
        # 记录统计信息
        self.stats["original_shape"] = original_shape
        self.stats["final_shape"] = df.shape
        
        logger.info(f"✅ 筹码结构表处理完成: {original_shape} -> {df.shape}")
        
        return df
    
    def _clip_ratios(self, df: Any) -> Any:
        """
        比例 Clipping：将超过 100% 的值截断到 100
        """
        logger.info("📌 Step 1: 比例 Clipping")
        
        max_ratio = self.config.chip.hold_ratio_max
        ratio_fields = self.config.chip.ratio_clip_fields
        
        clip_stats = {}
        for field in ratio_fields:
            if field in df.columns:
                # 统计超过阈值的数量
                if self.use_gpu:
                    over_max = int((df[field] > max_ratio).sum())
                else:
                    over_max = (df[field] > max_ratio).sum()
                
                clip_stats[field] = over_max
                
                if over_max > 0:
                    # 执行 Clip
                    df = self.clip_column(df, field, lower=0, upper=max_ratio, inplace=True)
                    logger.info(f"  ✓ {field}: 截断 {over_max:,} 行 (>{max_ratio}%)")
                else:
                    logger.info(f"  ✓ {field}: 无需截断")
        
        self.stats["clip_stats"] = clip_stats
        
        return df
    
    def _ffill_missing_values(self, df: Any) -> Any:
        """
        缺失值前向填充 (ffill)
        
        对 holder_num 等字段按股票分组进行前向填充
        """
        logger.info("📌 Step 2: 缺失值前向填充")
        
        ffill_fields = self.config.chip.ffill_fields
        
        # 确保按股票和日期排序
        df = df.sort_values(["ts_code", "trade_date"])
        
        ffill_stats = {}
        for field in ffill_fields:
            if field in df.columns:
                # 统计填充前的缺失数量
                if self.use_gpu:
                    before_null = int(df[field].isna().sum())
                else:
                    before_null = df[field].isna().sum()
                
                # 按股票分组前向填充
                if self.use_gpu:
                    # cuDF 的 groupby ffill
                    df[field] = df.groupby("ts_code")[field].ffill()
                else:
                    df[field] = df.groupby("ts_code")[field].ffill()
                
                # 统计填充后的缺失数量
                if self.use_gpu:
                    after_null = int(df[field].isna().sum())
                else:
                    after_null = df[field].isna().sum()
                
                filled = before_null - after_null
                ffill_stats[field] = {"before": before_null, "after": after_null, "filled": filled}
                logger.info(f"  ✓ {field}: 填充 {filled:,} 行 (剩余缺失: {after_null:,})")
        
        self.stats["ffill_stats"] = ffill_stats
        
        return df
    
    def _validate_holder_num(self, df: Any) -> Any:
        """
        股东户数有效性检查
        
        holder_num 应该 > 0，将 0 和负值标记为异常
        """
        logger.info("📌 Step 3: 股东户数有效性检查")
        
        if "holder_num" not in df.columns:
            logger.warning("  ⚠️ holder_num 列不存在，跳过")
            return df
        
        # 检查 0 和负值
        if self.use_gpu:
            zero_count = int((df["holder_num"] == 0).sum())
            negative_count = int((df["holder_num"] < 0).sum())
        else:
            zero_count = (df["holder_num"] == 0).sum()
            negative_count = (df["holder_num"] < 0).sum()
        
        self.stats["holder_num_zero"] = zero_count
        self.stats["holder_num_negative"] = negative_count
        
        if zero_count > 0 or negative_count > 0:
            logger.warning(f"  ⚠️ holder_num 异常值: 零值 {zero_count:,}, 负值 {negative_count:,}")
            # 将异常值置为 NaN，后续可由模型处理
            if self.use_gpu:
                df["holder_num"] = df["holder_num"].where(df["holder_num"] > 0, other=np.nan)
            else:
                df.loc[df["holder_num"] <= 0, "holder_num"] = np.nan
        else:
            logger.info("  ✓ holder_num 全部有效 (>0)")
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取预处理统计信息"""
        return self.stats
