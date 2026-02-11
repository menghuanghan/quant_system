"""
状态表预处理器

处理 dwd_stock_status 表，主要包括：
1. 交易状态最终判定（需要与价格表的 vol 结合，直接覆盖 is_trading）
2. 风险掩码生成
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from .base import BasePreprocessor
from .config import PreprocessConfig

logger = logging.getLogger(__name__)


class StatusPreprocessor(BasePreprocessor):
    """状态表预处理器"""
    
    def __init__(self, config: PreprocessConfig):
        super().__init__(config)
        self.stats: Dict[str, Any] = {}
    
    def process(
        self, 
        df: Any, 
        price_df: Optional[Any] = None
    ) -> Any:
        """
        执行状态表预处理
        
        处理步骤：
        1. 结合价格表的成交量，更新交易状态（直接覆盖 is_trading）
        2. 生成风险掩码（ST、新股、涨跌停等）
        
        Args:
            df: 输入的状态表 DataFrame
            price_df: 可选的价格表 DataFrame（用于获取 vol 信息）
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📊 开始处理状态表 (dwd_stock_status)")
        logger.info("=" * 60)
        
        original_shape = df.shape
        df = df.copy()
        
        # 1. 结合价格表更新交易状态（直接覆盖 is_trading）
        if price_df is not None:
            df = self._update_trading_status(df, price_df)
        else:
            logger.info("  ⚠️ 未提供价格表，保留原始 is_trading")
        
        # 2. 生成风险掩码
        df = self._generate_risk_masks(df)
        
        # 记录统计信息
        self.stats["original_shape"] = original_shape
        self.stats["final_shape"] = df.shape
        
        logger.info(f"✅ 状态表处理完成: {original_shape} -> {df.shape}")
        
        return df
    
    def _update_trading_status(self, df: Any, price_df: Any) -> Any:
        """
        结合价格表的成交量更新交易状态（直接覆盖 is_trading）
        
        策略："信资金，不信标签"
        从价格表中获取 is_trading（已经过 vol > 0 判定）覆盖原始值
        """
        logger.info("📌 Step 1: 更新交易状态 (is_trading)")
        
        # 从价格表获取 is_trading
        if "is_trading" not in price_df.columns:
            logger.warning("  ⚠️ 价格表中没有 is_trading 列")
            if "vol" in price_df.columns:
                logger.info("  📌 使用 vol > 0 计算 is_trading")
                if self.use_gpu:
                    price_trading = price_df[["trade_date", "ts_code"]].copy()
                    price_trading["is_trading_new"] = (price_df["vol"] > 0).astype("int8")
                else:
                    price_trading = price_df[["trade_date", "ts_code"]].copy()
                    price_trading["is_trading_new"] = (price_df["vol"] > 0).astype("int8")
            else:
                logger.warning("  ⚠️ 价格表中没有 vol，无法更新交易状态")
                return df
        else:
            price_trading = price_df[["trade_date", "ts_code", "is_trading"]].copy()
            price_trading = price_trading.rename(columns={"is_trading": "is_trading_new"})
        
        # 保存原始值用于统计
        original_is_trading = df["is_trading"].copy() if "is_trading" in df.columns else None
        
        # 合并价格表的交易状态
        if self.use_gpu:
            df = df.merge(
                price_trading,
                on=["trade_date", "ts_code"],
                how="left"
            )
        else:
            df = df.merge(
                price_trading,
                on=["trade_date", "ts_code"],
                how="left"
            )
        
        # 用新值覆盖原值，缺失值用原值填充
        if "is_trading_new" in df.columns:
            if "is_trading" in df.columns:
                if self.use_gpu:
                    df["is_trading"] = df["is_trading_new"].fillna(df["is_trading"]).astype("int8")
                else:
                    df["is_trading"] = df["is_trading_new"].fillna(df["is_trading"]).astype("int8")
            else:
                df["is_trading"] = df["is_trading_new"].fillna(0).astype("int8")
            
            # 删除临时列
            df = df.drop(columns=["is_trading_new"])
        
        # 统计变化
        if original_is_trading is not None and "is_trading" in df.columns:
            if self.use_gpu:
                diff_count = int((df["is_trading"] != original_is_trading).sum())
            else:
                diff_count = (df["is_trading"] != original_is_trading).sum()
            
            self.stats["trading_status_diff"] = diff_count
            logger.info(f"  📊 is_trading 更新: {diff_count:,} 行发生变化")
        
        return df
    
    def _generate_risk_masks(self, df: Any) -> Any:
        """
        生成风险掩码
        
        创建组合风险标记，便于下游快速过滤
        """
        logger.info("📌 Step 2: 生成风险掩码")
        
        # 1. 可交易掩码：非ST、非新股、非涨跌停、有交易
        tradable_conditions = []
        
        if "is_st" in df.columns:
            tradable_conditions.append(df["is_st"] == 0)
        if "is_new" in df.columns:
            tradable_conditions.append(df["is_new"] == 0)
        if "is_limit_up" in df.columns:
            tradable_conditions.append(df["is_limit_up"] == 0)
        if "is_limit_down" in df.columns:
            tradable_conditions.append(df["is_limit_down"] == 0)
        if "is_trading" in df.columns:
            tradable_conditions.append(df["is_trading"] == 1)
        
        if tradable_conditions:
            if self.use_gpu:
                # cuDF 中需要逐一合并条件
                result = tradable_conditions[0]
                for cond in tradable_conditions[1:]:
                    result = result & cond
                df["is_tradable"] = result.fillna(False).astype("int8")
            else:
                result = tradable_conditions[0]
                for cond in tradable_conditions[1:]:
                    result = result & cond
                df["is_tradable"] = result.fillna(False).astype("int8")
            
            if self.use_gpu:
                tradable_count = int(df["is_tradable"].sum())
            else:
                tradable_count = df["is_tradable"].sum()
            
            tradable_pct = tradable_count / len(df) * 100
            self.stats["tradable_count"] = tradable_count
            self.stats["tradable_pct"] = tradable_pct
            
            logger.info(f"  ✓ is_tradable: {tradable_count:,} 行 ({tradable_pct:.1f}%)")
        
        # 2. 风险标记：ST 或 新股
        if "is_st" in df.columns and "is_new" in df.columns:
            if self.use_gpu:
                df["is_risky"] = ((df["is_st"] == 1) | (df["is_new"] == 1)).fillna(False).astype("int8")
                risky_count = int(df["is_risky"].sum())
            else:
                df["is_risky"] = ((df["is_st"] == 1) | (df["is_new"] == 1)).fillna(False).astype("int8")
                risky_count = df["is_risky"].sum()
            
            logger.info(f"  ✓ is_risky (ST或新股): {risky_count:,} 行")
        
        # 3. 涨跌停标记
        if "is_limit_up" in df.columns and "is_limit_down" in df.columns:
            if self.use_gpu:
                df["is_limit"] = ((df["is_limit_up"] == 1) | (df["is_limit_down"] == 1)).fillna(False).astype("int8")
                limit_count = int(df["is_limit"].sum())
            else:
                df["is_limit"] = ((df["is_limit_up"] == 1) | (df["is_limit_down"] == 1)).fillna(False).astype("int8")
                limit_count = df["is_limit"].sum()
            
            logger.info(f"  ✓ is_limit (涨跌停): {limit_count:,} 行")
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取预处理统计信息"""
        return self.stats
