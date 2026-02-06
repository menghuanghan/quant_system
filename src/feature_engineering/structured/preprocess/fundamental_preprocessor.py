"""
基本面表预处理器

处理 dwd_stock_fundamental 表，主要包括：
1. 倒数变换（PE→EP, PB→BP, PS→SP）
2. 分位数去极值
3. 数据时滞过滤（超过180天的财报数据置为NaN）

注意：
- 不在 Preprocess 阶段做 Log 变换（保持原始物理意义）
- Log 变换应在 Feature Transformation 阶段进行
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .base import BasePreprocessor
from .config import PreprocessConfig

logger = logging.getLogger(__name__)


class FundamentalPreprocessor(BasePreprocessor):
    """基本面表预处理器"""
    
    def __init__(self, config: PreprocessConfig):
        super().__init__(config)
        self.stats: Dict[str, Any] = {}
    
    def process(self, df: Any) -> Any:
        """
        执行基本面表预处理
        
        处理步骤：
        0. 单位换算（万元→元）
        1. 计算数据时滞
        2. 倒数变换（估值指标）
        3. 分位数去极值
        4. 时滞数据过滤
        
        Args:
            df: 输入的基本面表 DataFrame
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📊 开始处理基本面表 (dwd_stock_fundamental)")
        logger.info("=" * 60)
        
        original_shape = df.shape
        df = df.copy()
        
        # 0. 单位换算（万元→元）
        df = self._convert_amount_units(df)
        
        # 1. 计算数据时滞
        df = self._calculate_data_lag(df)
        
        # 2. 倒数变换（估值指标）
        df = self._inverse_transform_valuations(df)
        
        # 3. 分位数去极值
        df = self._winsorize_by_quantile(df)
        
        # 4. 时滞数据过滤（最后执行，包括新增的倒数字段）
        df = self._filter_stale_data(df)
        
        # 5. 估值指标缺失值填充（亏损股 PE 无法计算，EP 填充为 0）
        df = self._fill_valuation_missing(df)
        
        # 记录统计信息
        self.stats["original_shape"] = original_shape
        self.stats["final_shape"] = df.shape
        
        logger.info(f"✅ 基本面表处理完成: {original_shape} -> {df.shape}")
        
        return df
    
    def _convert_amount_units(self, df: Any) -> Any:
        """
        单位换算：万元 -> 元
        
        处理字段：total_mv, circ_mv
        """
        logger.info("📌 Step 0: 单位换算 (万元→元)")
        
        amount_config = self.config.fundamental_amount
        converted_count = 0
        
        for col in amount_config.wan_yuan_fields:
            if col in df.columns:
                df[col] = df[col] * amount_config.wan_yuan_multiplier
                converted_count += 1
                logger.info(f"  ✓ {col}: ×{amount_config.wan_yuan_multiplier:.0f}")
        
        if converted_count == 0:
            logger.warning("  ⚠️ 无可换算字段")
        else:
            self.stats["unit_converted_fields"] = converted_count
        
        return df
    
    def _calculate_data_lag(self, df: Any) -> Any:
        """
        计算数据时滞
        
        lag_days = trade_date - report_date
        """
        logger.info("📌 Step 1: 计算数据时滞")
        
        if "report_date" not in df.columns:
            logger.warning("  ⚠️ report_date 列不存在，无法计算时滞")
            return df
        
        df = self.calculate_lag_days(
            df,
            trade_date_col="trade_date",
            report_date_col="report_date",
            output_col="lag_days"
        )
        
        # 统计时滞分布（处理 NaN 值）
        if "lag_days" in df.columns:
            valid_lag = df["lag_days"].dropna()
            
            if self.use_gpu:
                null_count = int(df["lag_days"].isna().sum())
                if len(valid_lag) > 0:
                    mean_lag = float(valid_lag.mean())
                    max_lag = int(valid_lag.max())
                    lag_gt_180 = int((valid_lag > 180).sum())
                    lag_gt_365 = int((valid_lag > 365).sum())
                else:
                    mean_lag, max_lag, lag_gt_180, lag_gt_365 = 0, 0, 0, 0
            else:
                null_count = df["lag_days"].isna().sum()
                if len(valid_lag) > 0:
                    mean_lag = valid_lag.mean()
                    max_lag = valid_lag.max()
                    lag_gt_180 = (valid_lag > 180).sum()
                    lag_gt_365 = (valid_lag > 365).sum()
                else:
                    mean_lag, max_lag, lag_gt_180, lag_gt_365 = 0, 0, 0, 0
            
            self.stats["mean_lag_days"] = mean_lag
            self.stats["max_lag_days"] = max_lag
            self.stats["lag_gt_180"] = lag_gt_180
            self.stats["lag_gt_365"] = lag_gt_365
            self.stats["lag_null_count"] = null_count
            
            logger.info(f"  📊 report_date 缺失: {null_count:,} 行")
            logger.info(f"  📊 平均时滞: {mean_lag:.1f} 天")
            logger.info(f"  📊 最大时滞: {max_lag:.0f} 天")
            logger.info(f"  📊 时滞 > 180 天: {lag_gt_180:,} 行")
            logger.info(f"  📊 时滞 > 365 天: {lag_gt_365:,} 行")
        
        return df
    
    def _inverse_transform_valuations(self, df: Any) -> Any:
        """
        倒数变换：将估值比率转换为收益率形式
        
        PE -> EP (盈利收益率 = 1/PE)
        PB -> BP (Book-to-Price = 1/PB)  
        PS -> SP (Sales-to-Price = 1/PS)
        
        这样做的好处：
        - 亏损股的 PE 为负或缺失，但 EP 可以为负值或0，不会有缺失
        - 数值更稳定，不会出现极大值
        """
        logger.info("📌 Step 2: 倒数变换 (估值指标)")
        
        inverse_config = self.config.inverse_transform
        epsilon = inverse_config.inverse_epsilon
        
        for input_col, output_col in inverse_config.inverse_fields.items():
            if input_col in df.columns:
                df = self.inverse_transform(
                    df, input_col, output_col,
                    epsilon=epsilon,
                    inplace=True
                )
                
                # 统计新字段的分布
                if self.use_gpu:
                    non_null = int(df[output_col].notna().sum())
                    mean_val = float(df[output_col].mean()) if non_null > 0 else 0
                else:
                    non_null = df[output_col].notna().sum()
                    mean_val = df[output_col].mean() if non_null > 0 else 0
                
                logger.info(f"  ✓ {input_col} -> {output_col} (非空: {non_null:,}, 均值: {mean_val:.4f})")
            else:
                logger.warning(f"  ⚠️ {input_col} 列不存在，跳过")
        
        return df
    
    def _log_transform_large_values(self, df: Any) -> Any:
        """
        大数值字段对数变换
        
        对 total_mv, revenue_ttm 等绝对数值取对数，
        将跨度巨大的数值压缩到线性区间
        """
        logger.info("📌 Step 3: 大数值字段对数变换")
        
        epsilon = self.config.log_transform.log_epsilon
        
        for field in self.config.log_transform.fundamental_log_fields:
            if field in df.columns:
                df = self.log_transform(
                    df, field,
                    epsilon=epsilon,
                    output_column=f"{field}_log",
                    inplace=True
                )
                logger.info(f"  ✓ {field} -> {field}_log")
            else:
                logger.debug(f"  ⚠️ {field} 列不存在，跳过")
        
        return df
    
    def _winsorize_by_quantile(self, df: Any) -> Any:
        """
        分位数去极值
        
        对指定字段按分位数进行裁剪，去除极端值
        """
        logger.info("📌 Step 4: 分位数去极值")
        
        quantile_config = self.config.winsorize.quantile_clip_fields
        
        for field, (lower_q, upper_q) in quantile_config.items():
            if field in df.columns:
                df = self.winsorize_by_quantile(
                    df, field,
                    lower_quantile=lower_q,
                    upper_quantile=upper_q,
                    inplace=True
                )
                logger.info(f"  ✓ {field}: [{lower_q:.0%}, {upper_q:.0%}]")
            else:
                logger.debug(f"  ⚠️ {field} 列不存在，跳过")
        
        return df
    
    def _filter_stale_data(self, df: Any) -> Any:
        """
        时滞数据过滤
        
        对于 lag_days > max_lag_days 的行，将指定字段置为 NaN
        这些数据太陈旧，不应用于预测
        """
        logger.info("📌 Step 5: 时滞数据过滤")
        
        if "lag_days" not in df.columns:
            logger.warning("  ⚠️ lag_days 列不存在，无法过滤时滞数据")
            return df
        
        max_lag = self.config.data_lag.max_lag_days
        fields_to_mask = self.config.data_lag.lag_sensitive_fields
        
        # 过滤掉不存在的字段
        existing_fields = [f for f in fields_to_mask if f in df.columns]
        
        df = self.mask_stale_data(
            df,
            lag_col="lag_days",
            fields_to_mask=existing_fields,
            max_lag_days=max_lag
        )
        
        logger.info(f"  🕐 最大允许时滞: {max_lag} 天")
        logger.info(f"  📋 受影响字段: {len(existing_fields)} 个")
        
        return df
    
    def _fill_valuation_missing(self, df: Any) -> Any:
        """
        估值指标缺失值填充
        
        对于亏损股，PE/PB/PS 无法计算或为负，导致 EP/BP/SP 为 NaN。
        将这些缺失值填充为 0，表示"无法计算" / "无有意义的收益率"。
        
        这样做的理由：
        - EP/BP/SP 是用于建模的特征，缺失会导致样本损失
        - 亏损股的 EP 应为 0 或负值（意味着负的盈利收益率）
        - 填充后可保留全部样本用于模型训练
        """
        logger.info("📌 Step 6: 估值指标缺失值填充")
        
        valuation_fields = ["ep", "bp", "sp"]
        fill_stats = {}
        
        for field in valuation_fields:
            if field in df.columns:
                # 统计填充前的缺失数量
                if self.use_gpu:
                    before_null = int(df[field].isna().sum())
                else:
                    before_null = df[field].isna().sum()
                
                # 执行 fillna(0)
                df[field] = df[field].fillna(0)
                
                fill_stats[field] = before_null
                logger.info(f"  ✓ {field}: 填充 {before_null:,} 个缺失值为 0")
        
        self.stats["valuation_fill_stats"] = fill_stats
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取预处理统计信息"""
        return self.stats
