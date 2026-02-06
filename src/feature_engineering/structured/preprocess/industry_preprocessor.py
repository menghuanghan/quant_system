"""
行业分类预处理器

处理 dwd_stock_industry 表，主要包括：
1. 索引类型检查：确保 sw_l1_idx 等为 int 类型
2. 缺失值处理：将 -1 或 Unknown 归类为特殊索引
3. 编码验证
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .base import BasePreprocessor
from .config import PreprocessConfig

logger = logging.getLogger(__name__)


class IndustryPreprocessor(BasePreprocessor):
    """行业分类预处理器"""
    
    def __init__(self, config: PreprocessConfig):
        super().__init__(config)
        self.stats: Dict[str, Any] = {}
    
    def process(self, df: Any) -> Any:
        """
        执行行业分类预处理
        
        处理步骤：
        1. 索引类型转换（确保 int）
        2. 缺失值检查与处理
        3. 索引范围验证
        
        Args:
            df: 输入的行业分类表 DataFrame
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📊 开始处理行业分类表 (dwd_stock_industry)")
        logger.info("=" * 60)
        
        original_shape = df.shape
        df = df.copy()
        
        # 1. 索引类型转换
        df = self._convert_index_types(df)
        
        # 2. 缺失值检查与处理
        df = self._handle_missing_industries(df)
        
        # 3. 索引范围验证
        df = self._validate_index_range(df)
        
        # 记录统计信息
        self.stats["original_shape"] = original_shape
        self.stats["final_shape"] = df.shape
        
        logger.info(f"✅ 行业分类表处理完成: {original_shape} -> {df.shape}")
        
        return df
    
    def _convert_index_types(self, df: Any) -> Any:
        """
        索引类型转换：确保为 int 类型
        """
        logger.info("📌 Step 1: 索引类型转换")
        
        index_fields = self.config.industry.index_fields
        
        for field in index_fields:
            if field in df.columns:
                original_dtype = str(df[field].dtype)
                
                # 填充 NaN 为 unknown_idx
                unknown_idx = self.config.industry.unknown_idx
                df[field] = df[field].fillna(unknown_idx)
                
                # 转换为 int
                if self.use_gpu:
                    df[field] = df[field].astype("int32")
                else:
                    df[field] = df[field].astype("int32")
                
                logger.info(f"  ✓ {field}: {original_dtype} -> int32")
        
        return df
    
    def _handle_missing_industries(self, df: Any) -> Any:
        """
        缺失值检查与处理
        
        检查是否存在 -1 或 Unknown 的行业标签
        """
        logger.info("📌 Step 2: 缺失值检查与处理")
        
        unknown_idx = self.config.industry.unknown_idx
        
        # 检查各索引字段
        for field in self.config.industry.index_fields:
            if field in df.columns:
                if self.use_gpu:
                    unknown_count = int((df[field] == unknown_idx).sum())
                else:
                    unknown_count = (df[field] == unknown_idx).sum()
                
                self.stats[f"{field}_unknown_count"] = unknown_count
                
                if unknown_count > 0:
                    logger.warning(f"  ⚠️ {field}: {unknown_count:,} 行为未分类 (值={unknown_idx})")
                else:
                    logger.info(f"  ✓ {field}: 无未分类记录")
        
        # 检查 industry 字段（字符串）
        if "industry" in df.columns:
            if self.use_gpu:
                empty_count = int((df["industry"].isna() | (df["industry"] == "")).sum())
            else:
                empty_count = (df["industry"].isna() | (df["industry"] == "")).sum()
            
            self.stats["industry_empty_count"] = empty_count
            
            if empty_count > 0:
                logger.warning(f"  ⚠️ industry: {empty_count:,} 行为空")
            else:
                logger.info("  ✓ industry: 全部有值")
        
        return df
    
    def _validate_index_range(self, df: Any) -> Any:
        """
        索引范围验证
        """
        logger.info("📌 Step 3: 索引范围验证")
        
        for field in self.config.industry.index_fields:
            if field in df.columns:
                if self.use_gpu:
                    min_idx = int(df[field].min())
                    max_idx = int(df[field].max())
                    unique_count = df[field].nunique()
                else:
                    min_idx = df[field].min()
                    max_idx = df[field].max()
                    unique_count = df[field].nunique()
                
                self.stats[f"{field}_range"] = (min_idx, max_idx)
                self.stats[f"{field}_unique"] = unique_count
                
                logger.info(f"  ✓ {field}: 范围 [{min_idx}, {max_idx}], 唯一值 {unique_count}")
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取预处理统计信息"""
        return self.stats
