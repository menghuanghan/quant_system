"""
宏观环境预处理器

处理 dwd_macro_env 表，主要包括：
1. 时滞处理：shift(1) 模拟"开盘前已知"
2. 缺失值填充
3. 数据验证
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .base import BasePreprocessor
from .config import PreprocessConfig

logger = logging.getLogger(__name__)


class MacroPreprocessor(BasePreprocessor):
    """宏观环境预处理器"""
    
    def __init__(self, config: PreprocessConfig):
        super().__init__(config)
        self.stats: Dict[str, Any] = {}
    
    def process(self, df: Any) -> Any:
        """
        执行宏观环境预处理
        
        处理步骤：
        1. 时滞处理 (shift)
        2. 缺失值填充 (ffill)
        3. 数据验证
        
        Args:
            df: 输入的宏观环境表 DataFrame
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📊 开始处理宏观环境表 (dwd_macro_env)")
        logger.info("=" * 60)
        
        original_shape = df.shape
        df = df.copy()
        
        # 确保按日期排序
        if "trade_date" in df.columns:
            df = df.sort_values("trade_date")
        
        # 1. 时滞处理
        df = self._apply_shift(df)
        
        # 2. 缺失值填充
        df = self._fill_missing_values(df)
        
        # 3. 数据验证
        df = self._validate_data(df)
        
        # 记录统计信息
        self.stats["original_shape"] = original_shape
        self.stats["final_shape"] = df.shape
        
        logger.info(f"✅ 宏观环境表处理完成: {original_shape} -> {df.shape}")
        
        return df
    
    def _apply_shift(self, df: Any) -> Any:
        """
        时滞处理：shift(1) 模拟"开盘前已知"
        
        DWD 层已做 PIT 处理（数据滞后到公布日），
        这里再滞后一天确保开盘时数据可用。
        """
        logger.info("📌 Step 1: 时滞处理")
        
        if not self.config.macro.apply_shift:
            logger.info("  ⚪ 跳过 shift（配置已禁用）")
            return df
        
        shift_days = self.config.macro.shift_days
        shift_prefixes = self.config.macro.shift_field_prefixes
        
        # 识别需要 shift 的列
        shift_columns = []
        for col in df.columns:
            if col == "trade_date":
                continue
            for prefix in shift_prefixes:
                if col.startswith(prefix):
                    shift_columns.append(col)
                    break
        
        # 对于未匹配前缀的列，除了 trade_date 都进行 shift
        for col in df.columns:
            if col == "trade_date":
                continue
            if col not in shift_columns:
                shift_columns.append(col)
        
        # 去重
        shift_columns = list(set(shift_columns))
        
        # 执行 shift
        for col in shift_columns:
            df[col] = df[col].shift(shift_days)
        
        self.stats["shifted_columns"] = len(shift_columns)
        logger.info(f"  ✓ shift({shift_days}) 应用于 {len(shift_columns)} 列")
        
        return df
    
    def _fill_missing_values(self, df: Any) -> Any:
        """
        缺失值填充
        
        使用前向填充 (ffill)，宏观数据通常持续有效直到下次更新
        """
        logger.info("📌 Step 2: 缺失值填充")
        
        # 统计填充前的缺失情况
        if self.use_gpu:
            before_null_total = int(df.isna().sum().sum())
        else:
            before_null_total = df.isna().sum().sum()
        
        # 对所有非 trade_date 列进行 ffill
        for col in df.columns:
            if col == "trade_date":
                continue
            df[col] = df[col].ffill()
        
        # 统计填充后的缺失情况
        if self.use_gpu:
            after_null_total = int(df.isna().sum().sum())
        else:
            after_null_total = df.isna().sum().sum()
        
        filled_count = before_null_total - after_null_total
        
        self.stats["null_before"] = before_null_total
        self.stats["null_after"] = after_null_total
        self.stats["null_filled"] = filled_count
        
        logger.info(f"  ✓ 填充 {filled_count:,} 个缺失值 (剩余: {after_null_total:,})")
        
        return df
    
    def _validate_data(self, df: Any) -> Any:
        """
        数据验证
        """
        logger.info("📌 Step 3: 数据验证")
        
        # 检查日期连续性
        if "trade_date" in df.columns:
            if self.use_gpu:
                date_count = df["trade_date"].nunique()
                min_date = df["trade_date"].min()
                max_date = df["trade_date"].max()
            else:
                date_count = df["trade_date"].nunique()
                min_date = df["trade_date"].min()
                max_date = df["trade_date"].max()
            
            self.stats["date_count"] = int(date_count)
            self.stats["date_range"] = (str(min_date), str(max_date))
            
            logger.info(f"  ✓ 日期范围: {min_date} ~ {max_date} ({date_count} 天)")
        
        # 检查关键字段
        key_fields = ["gdp_yoy", "cpi_yoy", "pmi"]
        for field in key_fields:
            if field in df.columns:
                if self.use_gpu:
                    null_count = int(df[field].isna().sum())
                else:
                    null_count = df[field].isna().sum()
                
                if null_count > 0:
                    logger.warning(f"  ⚠️ {field}: {null_count} 个缺失值")
                else:
                    logger.info(f"  ✓ {field}: 无缺失")
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取预处理统计信息"""
        return self.stats
