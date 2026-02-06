"""
价格表预处理器

处理 dwd_stock_price 表，主要包括：
1. 单位换算（千元 -> 元）- 统一金额单位
2. 收益率去极值（Winsorization）- 裁剪到 [-30%, +100%]
3. 交易状态最终判定（结合 vol 和 is_trading）

注意：
- 不在 Preprocess 阶段做 Log 变换（保持原始物理意义）
- Log 变换应在 Feature Transformation 阶段进行
"""

import logging
from typing import Any, Dict

import numpy as np

from .base import BasePreprocessor
from .config import PreprocessConfig

logger = logging.getLogger(__name__)


class PricePreprocessor(BasePreprocessor):
    """价格表预处理器"""
    
    def __init__(self, config: PreprocessConfig):
        super().__init__(config)
        self.stats: Dict[str, Any] = {}
    
    def process(self, df: Any) -> Any:
        """
        执行价格表预处理
        
        处理步骤：
        1. 单位换算（千元 -> 元）
        2. 收益率去极值
        3. 交易状态最终判定
        
        Args:
            df: 输入的价格表 DataFrame
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📊 开始处理价格表 (dwd_stock_price)")
        logger.info("=" * 60)
        
        original_shape = df.shape
        df = df.copy()
        
        # 1. 单位换算（千元 -> 元）
        df = self._convert_amount_units(df)
        
        # 2. 收益率去极值
        df = self._winsorize_returns(df)
        
        # 3. 交易状态最终判定
        df = self._determine_trading_status(df)
        
        # 记录统计信息
        self.stats["original_shape"] = original_shape
        self.stats["final_shape"] = df.shape
        
        logger.info(f"✅ 价格表处理完成: {original_shape} -> {df.shape}")
        
        return df
    
    def _winsorize_returns(self, df: Any) -> Any:
        """
        收益率去极值
        
        将 return_1d 裁剪到 [return_1d_lower, return_1d_upper] 范围
        """
        logger.info("📌 Step 1: 收益率去极值 (Winsorization)")
        
        if "return_1d" not in df.columns:
            logger.warning("  ⚠️ return_1d 列不存在，跳过")
            return df
        
        lower = self.config.winsorize.return_1d_lower
        upper = self.config.winsorize.return_1d_upper
        
        # 统计原始极端值数量
        if self.use_gpu:
            below_lower = int((df["return_1d"] < lower).sum())
            above_upper = int((df["return_1d"] > upper).sum())
        else:
            below_lower = (df["return_1d"] < lower).sum()
            above_upper = (df["return_1d"] > upper).sum()
        
        # 保存原始列用于比较
        df["return_1d_raw"] = df["return_1d"]
        
        # 执行裁剪
        df = self.clip_column(df, "return_1d", lower=lower, upper=upper, inplace=True)
        
        self.stats["return_1d_below_lower"] = below_lower
        self.stats["return_1d_above_upper"] = above_upper
        
        logger.info(f"  📉 裁剪范围: [{lower:.0%}, {upper:.0%}]")
        logger.info(f"  📊 低于 {lower:.0%}: {below_lower:,} 行")
        logger.info(f"  📊 高于 {upper:.0%}: {above_upper:,} 行")
        
        return df
    
    def _convert_amount_units(self, df: Any) -> Any:
        """
        单位换算：千元 -> 元
        
        将 amount 字段乘以 1000，统一为元
        """
        logger.info("📌 Step 1: 单位换算 (千元 -> 元)")
        
        # price 表的 amount 单位是千元
        amount_multiplier = 1000.0
        amount_fields = ["amount"]
        
        converted_count = 0
        for field in amount_fields:
            if field in df.columns:
                df[field] = df[field] * amount_multiplier
                converted_count += 1
                logger.info(f"  ✓ {field} ×{amount_multiplier:.0f}")
        
        self.stats["converted_amount_fields"] = converted_count
        
        return df
    
    def _determine_trading_status(self, df: Any) -> Any:
        """
        交易状态最终判定
        
        策略："信资金，不信标签"
        is_trading_final = (vol > 0) | (is_trading == 1)
        
        如果成交量 > 0，无论状态表怎么说，它今天就是交易了
        """
        logger.info("📌 Step 3: 交易状态最终判定")
        
        if "vol" not in df.columns:
            logger.warning("  ⚠️ vol 列不存在，无法判定交易状态")
            return df
        
        if self.config.trading_status.trust_volume_over_label:
            # "信资金，不信标签" 策略
            if "is_trading" in df.columns:
                if self.use_gpu:
                    vol_positive = df["vol"] > 0
                    is_trading_flag = df["is_trading"] == 1
                    df["is_trading_final"] = (vol_positive | is_trading_flag).astype("int8")
                else:
                    df["is_trading_final"] = (
                        (df["vol"] > 0) | (df["is_trading"] == 1)
                    ).astype("int8")
                
                # 统计差异
                if self.use_gpu:
                    diff_count = int((df["is_trading_final"] != df["is_trading"]).sum())
                else:
                    diff_count = (df["is_trading_final"] != df["is_trading"]).sum()
                
                self.stats["trading_status_diff"] = diff_count
                logger.info(f"  🏷️ is_trading_final = (vol > 0) | (is_trading == 1)")
                logger.info(f"  📊 与原始 is_trading 不一致: {diff_count:,} 行")
            else:
                # 没有 is_trading 列，仅根据成交量判断
                if self.use_gpu:
                    df["is_trading_final"] = (df["vol"] > 0).astype("int8")
                else:
                    df["is_trading_final"] = (df["vol"] > 0).astype("int8")
                logger.info(f"  🏷️ is_trading_final = (vol > 0)")
        else:
            # 直接使用原始 is_trading
            if "is_trading" in df.columns:
                df["is_trading_final"] = df["is_trading"]
            else:
                df["is_trading_final"] = 1
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取预处理统计信息"""
        return self.stats
