"""
资金流预处理器

处理 dwd_money_flow 表，主要包括：
1. 停牌清洗：停牌日资金流字段置 0
2. 极值检查：无 inf
3. 缺失值填充

注意：金额单位转换已在 DWD 层完成（所有金额字段已统一为元）
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .base import BasePreprocessor
from .config import PreprocessConfig

logger = logging.getLogger(__name__)


class MoneyFlowPreprocessor(BasePreprocessor):
    """资金流预处理器"""
    
    def __init__(self, config: PreprocessConfig):
        super().__init__(config)
        self.stats: Dict[str, Any] = {}
    
    def process(self, df: Any, status_df: Any = None) -> Any:
        """
        执行资金流预处理
        
        处理步骤：
        1. 停牌清洗（利用状态表）
        2. 极值检查
        3. 缺失值填充
        
        注意：金额单位转换已在 DWD 层完成
        
        Args:
            df: 输入的资金流表 DataFrame（金额字段已统一为元）
            status_df: 可选的状态表 DataFrame（用于停牌清洗）
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📊 开始处理资金流表 (dwd_money_flow)")
        logger.info("=" * 60)
        
        original_shape = df.shape
        df = df.copy()
        
        # 1. 停牌清洗（金额单位转换已在DWD层完成）
        if status_df is not None:
            df = self._clean_suspended_data(df, status_df)
        else:
            logger.warning("  ⚠️ 未提供状态表，跳过停牌清洗")
        
        # 2. 极值检查
        df = self._check_extreme_values(df)
        
        # 3. 缺失值填充
        df = self._fill_missing_values(df)
        
        # 记录统计信息
        self.stats["original_shape"] = original_shape
        self.stats["final_shape"] = df.shape
        
        logger.info(f"✅ 资金流表处理完成: {original_shape} -> {df.shape}")
        
        return df
    
    def _convert_amount_units(self, df: Any) -> Any:
        """
        [已弃用] 单位换算：万元 -> 元
        
        注意：此逻辑已移至 DWD 层 (money_flow_processor.py)
        保留此方法仅为向后兼容，不再调用
        """
        logger.warning("⚠️ _convert_amount_units 已弃用，金额转换已在DWD层完成")
        return df  # 直接返回，不做任何处理
    
    def _clean_suspended_data(self, df: Any, status_df: Any) -> Any:
        """
        停牌清洗：利用状态表的 is_suspended 字段
        
        将停牌日的所有资金流字段置为 0
        """
        logger.info("📌 Step 1: 停牌清洗")
        
        # 获取停牌信息
        if "is_suspended" not in status_df.columns:
            # 尝试从 is_trading 推断
            if "is_trading" in status_df.columns:
                status_df = status_df.copy()
                status_df["is_suspended"] = 1 - status_df["is_trading"]
            else:
                logger.warning("  ⚠️ 状态表缺少 is_suspended 和 is_trading 字段，跳过")
                return df
        
        # 合并停牌状态
        merge_cols = ["trade_date", "ts_code"]
        suspended_status = status_df[merge_cols + ["is_suspended"]]
        
        original_len = len(df)
        df = df.merge(suspended_status, on=merge_cols, how="left")
        
        # 将停牌日的资金流字段置 0
        suspend_zero_fields = self.config.money_flow.suspend_zero_fields
        
        if self.use_gpu:
            suspended_mask = df["is_suspended"] == 1
        else:
            suspended_mask = df["is_suspended"] == 1
        
        suspended_count = int(suspended_mask.sum())
        
        for field in suspend_zero_fields:
            if field in df.columns:
                if self.use_gpu:
                    df[field] = df[field].where(~suspended_mask, other=0)
                else:
                    df.loc[suspended_mask, field] = 0
        
        # 删除临时列
        df = df.drop(columns=["is_suspended"])
        
        self.stats["suspended_rows_cleaned"] = suspended_count
        logger.info(f"  ✓ 清洗 {suspended_count:,} 行停牌日数据")
        
        return df
    
    def _check_extreme_values(self, df: Any) -> Any:
        """
        极值检查：确保无 inf
        """
        logger.info("📌 Step 3: 极值检查")
        
        amount_fields = self.config.money_flow.amount_fields
        inf_found = 0
        
        for field in amount_fields:
            if field in df.columns:
                # 使用 cuDF 原生方法替代 cupy.isinf() 避免 GPU 兼容性问题
                col = df[field].fillna(0)
                # 检测 inf：值 == 正无穷 或 值 == 负无穷
                inf_mask = (col == np.inf) | (col == -np.inf)
                inf_count = int(inf_mask.sum())
                
                if inf_count > 0:
                    inf_found += inf_count
                    # 将 inf 替换为 NaN
                    df[field] = df[field].replace([np.inf, -np.inf], np.nan)
        
        self.stats["inf_values_found"] = inf_found
        
        if inf_found > 0:
            logger.warning(f"  ⚠️ 发现 {inf_found} 个 inf 值，已替换为 NaN")
        else:
            logger.info("  ✓ 无极值异常")
        
        return df
    
    def _fill_missing_values(self, df: Any) -> Any:
        """
        缺失值填充：将 NaN 填充为 0
        """
        logger.info("📌 Step 4: 缺失值填充")
        
        amount_fields = self.config.money_flow.amount_fields
        
        filled_count = 0
        for field in amount_fields:
            if field in df.columns:
                if self.use_gpu:
                    null_count = int(df[field].isna().sum())
                else:
                    null_count = df[field].isna().sum()
                
                if null_count > 0:
                    df[field] = df[field].fillna(0)
                    filled_count += null_count
        
        self.stats["null_values_filled"] = filled_count
        logger.info(f"  ✓ 填充 {filled_count:,} 个缺失值为 0")
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取预处理统计信息"""
        return self.stats
