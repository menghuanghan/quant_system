"""
后处理模块

包括缺失值清洗、数据切片、标准化等。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PostProcessor:
    """后处理器"""
    
    def __init__(self, config, data_config, use_gpu: bool = True):
        """
        初始化后处理器
        
        Args:
            config: NormalizationConfig 配置
            data_config: DataConfig 配置
            use_gpu: 是否使用 GPU
        """
        self.config = config
        self.data_config = data_config
        self.use_gpu = use_gpu
        self.stats: Dict[str, Any] = {}
        
        if use_gpu:
            try:
                import cudf
                self.pd = cudf
                self.cudf = cudf
                logger.info("🚀 PostProcessor: GPU 加速已启用 (cuDF)")
            except ImportError:
                import pandas as pd
                self.pd = pd
                self.cudf = None
                self.use_gpu = False
                logger.warning("⚠️ cuDF 不可用，回退到 pandas")
        else:
            import pandas as pd
            self.pd = pd
            self.cudf = None
    
    def process(self, df: Any) -> Any:
        """
        执行完整的后处理流程
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📋 Step 5: 后处理与标准化")
        logger.info("=" * 60)
        
        # 1. 数据切片：保留正式期数据
        df = self._slice_data(df)
        
        # 2. 截面标准化 (Z-Score)
        if self.config.cross_sectional_zscore:
            df = self._cross_sectional_zscore(df)
        
        # 3. 缺失值终极清洗（基于关键字段）
        df = self._final_clean(df)
        
        # === 清洗闭环（新增） ===
        # 4. 特征 NaN 填充 (fillna(0))
        df = self._fill_feature_nan(df)
        
        # 5. 标签完整性检查（删除标签为 NaN 的行）
        df = self._ensure_label_integrity(df)
        
        # 6. 剔除常量列 (std=0)
        df = self._drop_constant_columns(df)
        
        # 7. 浮点数精度控制 (float64 → float32)
        df = self._reduce_float_precision(df)
        
        return df
        
        return df
    
    def _slice_data(self, df: Any) -> Any:
        """
        数据切片
        
        丢弃预热期数据，只保留正式期（训练/验证/测试用）。
        """
        train_start = self.data_config.train_start
        
        logger.info(f"  📊 数据切片: 保留 {train_start} 之后的数据")
        
        original_rows = len(df)
        
        # 确保 trade_date 是字符串格式进行比较
        if self.use_gpu:
            # cuDF 的日期列可能是 datetime64，需要转换比较
            if df['trade_date'].dtype != 'object':
                import cudf
                train_start_dt = cudf.to_datetime(train_start)
                mask = df['trade_date'] >= train_start_dt
            else:
                mask = df['trade_date'] >= train_start
        else:
            if df['trade_date'].dtype != 'object':
                import pandas as pd
                train_start_dt = pd.to_datetime(train_start)
                mask = df['trade_date'] >= train_start_dt
            else:
                mask = df['trade_date'] >= train_start
        
        df = df[mask]
        
        final_rows = len(df)
        removed_rows = original_rows - final_rows
        
        logger.info(f"    ✓ 丢弃预热期: {removed_rows:,} 行")
        logger.info(f"    ✓ 保留正式期: {final_rows:,} 行")
        
        self.stats["slice_original"] = original_rows
        self.stats["slice_removed"] = removed_rows
        self.stats["slice_final"] = final_rows
        
        return df
    
    def _cross_sectional_zscore(self, df: Any) -> Any:
        """
        截面标准化 (Cross-Sectional Z-Score)
        
        对每一天的所有股票做 Z-Score: (x - mean) / std
        这消除了大盘波动的影响，让模型专注于选股（Alpha）。
        """
        logger.info("  📊 截面标准化 (每日 Z-Score)")
        
        zscore_fields = [f for f in self.config.zscore_fields if f in df.columns]
        clip_value = self.config.zscore_clip
        
        logger.info(f"    ✓ 标准化字段: {len(zscore_fields)} 个")
        logger.info(f"    ✓ Z-Score 裁剪范围: [-{clip_value}, {clip_value}]")
        
        # 确保是副本以避免 SettingWithCopyWarning
        df = df.copy()
        
        # 按日期分组计算 Z-Score
        for field in zscore_fields:
            # 计算每日的均值和标准差
            daily_mean = df.groupby('trade_date', sort=False)[field].transform('mean')
            daily_std = df.groupby('trade_date', sort=False)[field].transform('std')
            
            # 避免除以0
            daily_std = daily_std.replace(0, 1e-10).fillna(1e-10)
            
            # 计算 Z-Score 并裁剪
            zscore_col = f'{field}_zscore'
            zscore_values = ((df[field] - daily_mean) / daily_std).clip(lower=-clip_value, upper=clip_value)
            df[zscore_col] = zscore_values
        
        logger.info(f"    ✅ 完成 {len(zscore_fields)} 个字段的截面标准化")
        
        return df
    
    def _final_clean(self, df: Any) -> Any:
        """
        缺失值终极清洗
        
        丢弃所有仍含有 NaN 的关键字段行。
        """
        logger.info("  📊 缺失值终极清洗")
        
        original_rows = len(df)
        
        # 定义必须有值的关键字段
        # 主要标签
        primary_label = f"ret_{self.data_config.train_start.split('-')[0] if hasattr(self.data_config, 'primary_label_days') else 5}d"
        # 使用配置中的 primary_label_days（从 LabelConfig）
        # 这里简化处理，检查所有 ret_*d 标签
        label_cols = [col for col in df.columns if col.startswith('ret_') and col.endswith('d')]
        
        # 检查关键特征字段（MA250 等）
        key_features = ['ma_250'] if 'ma_250' in df.columns else []
        
        # 合并需要检查的列
        check_cols = label_cols + key_features
        check_cols = [c for c in check_cols if c in df.columns]
        
        if check_cols:
            # 计算任意关键列为 NaN 的行
            null_mask = df[check_cols].isna().any(axis=1)
            if self.use_gpu:
                null_count = int(null_mask.sum())
            else:
                null_count = null_mask.sum()
            
            # 丢弃这些行
            df = df[~null_mask]
            
            logger.info(f"    ✓ 检查字段: {check_cols}")
            logger.info(f"    ✓ 丢弃含 NaN 的行: {null_count:,}")
        
        final_rows = len(df)
        
        logger.info(f"    ✓ 最终样本数: {final_rows:,} 行")
        
        self.stats["clean_original"] = original_rows
        self.stats["clean_final"] = final_rows
        
        return df
    
    def _fill_feature_nan(self, df: Any) -> Any:
        """
        特征列 NaN 填充
        
        对所有特征列（非标签列）执行零值填充：
        - 基本面比率字段（ROE, 毛利率等）：0 代表"无显著特征"
        - Z-Score 字段：0 代表"全市场平均水平"
        - 其他残留 NaN：兜底填 0
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            填充后的 DataFrame
        """
        logger.info("  📊 特征列 NaN 填充 (fillna(0))")
        
        # 定义标签列（不填充）
        label_cols = [col for col in df.columns if col.startswith('ret_') or col.startswith('label_')]
        base_cols = ['ts_code', 'trade_date']
        
        # 特征列 = 全部列 - 标签列 - 基础列
        feature_cols = [col for col in df.columns if col not in label_cols + base_cols]
        
        # 统计填充前 NaN
        nan_before = df[feature_cols].isna().sum().sum()
        nan_cols_count = (df[feature_cols].isna().sum() > 0).sum()
        
        # 执行填充
        df = df.copy()
        for col in feature_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
        
        logger.info(f"    ✓ 填充特征列数: {len(feature_cols)} 个")
        logger.info(f"    ✓ 有 NaN 的列: {nan_cols_count} 个")
        logger.info(f"    ✓ 填充 NaN 数量: {nan_before:,} 个")
        
        self.stats["fill_feature_cols"] = len(feature_cols)
        self.stats["fill_nan_count"] = int(nan_before)
        
        return df
    
    def _ensure_label_integrity(self, df: Any) -> Any:
        """
        确保标签完整性
        
        删除所有标签为 NaN 的行（不填 0，避免标签泄露）。
        特别是最后 N 天的样本（无法计算未来收益率）必须删除。
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            清洗后的 DataFrame
        """
        logger.info("  📊 标签完整性检查 (删除标签为 NaN 的行)")
        
        # 定义需要检查的标签列
        label_to_check = ['ret_1d', 'ret_5d', 'ret_10d', 'ret_20d']
        label_cols = [col for col in label_to_check if col in df.columns]
        
        if not label_cols:
            logger.warning("    ⚠️ 未找到标签列，跳过")
            return df
        
        original_rows = len(df)
        
        # 检查每个标签列的 NaN
        for col in label_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.info(f"    ⚠️ {col} 有 {nan_count:,} 个 NaN")
        
        # 删除任意标签为 NaN 的行
        null_mask = df[label_cols].isna().any(axis=1)
        if self.use_gpu:
            removed_count = int(null_mask.sum())
        else:
            removed_count = null_mask.sum()
        
        df = df[~null_mask]
        
        final_rows = len(df)
        
        logger.info(f"    ✓ 删除标签 NaN 行: {removed_count:,}")
        logger.info(f"    ✓ 剩余样本: {final_rows:,} 行")
        
        self.stats["label_removed_rows"] = removed_count
        self.stats["label_final_rows"] = final_rows
        
        return df
    
    def _drop_constant_columns(self, df: Any) -> Any:
        """
        剔除常量列
        
        删除标准差为 0 的列（无差异即无信息）。
        这些通常是过滤后剩下的"尸体"（如 is_trading 全为 1）。
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            剔除后的 DataFrame
        """
        logger.info("  📊 常量列剔除 (std=0)")
        
        # 仅检查数值列
        numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int64', 'int32']).columns.tolist()
        
        # 排除标签列和基础列
        label_cols = [col for col in df.columns if col.startswith('ret_') or col.startswith('label_')]
        base_cols = ['ts_code', 'trade_date']
        check_cols = [col for col in numeric_cols if col not in label_cols + base_cols]
        
        # 计算标准差
        constant_cols = []
        for col in check_cols:
            std_val = df[col].std()
            if std_val == 0 or (hasattr(std_val, 'item') and std_val.item() == 0):
                constant_cols.append(col)
        
        if constant_cols:
            logger.info(f"    ✓ 发现常量列: {constant_cols}")
            df = df.drop(columns=constant_cols)
            logger.info(f"    ✓ 剔除 {len(constant_cols)} 个常量列")
        else:
            logger.info(f"    ✓ 无常量列需剔除")
        
        self.stats["constant_cols_dropped"] = constant_cols
        self.stats["constant_cols_count"] = len(constant_cols)
        
        return df
    
    def _reduce_float_precision(self, df: Any) -> Any:
        """
        浮点数精度控制
        
        将 float64 列转换为 float32 以节省显存和存储空间。
        float32 可提供约 7 位有效数字精度，足以满足量化模型需求。
        
        优势：
        - 显存节省 50%
        - 存储空间节省 50%
        - 对模型精度影响可忽略（通常训练时也会转为 float32）
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            转换后的 DataFrame
        """
        logger.info("  📊 浮点数精度控制 (float64 → float32)")
        
        df = df.copy()
        
        # 统计
        converted_cols = []
        original_size = df.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
        
        # 获取 float64 列
        float64_cols = [
            col for col in df.columns 
            if df[col].dtype == 'float64' or str(df[col].dtype) == 'Float64'
        ]
        
        for col in float64_cols:
            # 转换为 float32
            # cuDF 和 pandas 都支持 astype('float32')
            try:
                df[col] = df[col].astype('float32')
                converted_cols.append(col)
            except Exception as e:
                logger.warning(f"    ⚠️ 列 {col} 转换失败: {e}")
        
        final_size = df.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
        saved_size = original_size - final_size
        saved_pct = (saved_size / original_size * 100) if original_size > 0 else 0
        
        logger.info(f"    ✓ 转换列数: {len(converted_cols)} 个")
        logger.info(f"    ✓ 内存占用: {original_size:.1f} MB → {final_size:.1f} MB")
        logger.info(f"    ✓ 节省空间: {saved_size:.1f} MB ({saved_pct:.1f}%)")
        
        self.stats["precision_converted_cols"] = len(converted_cols)
        self.stats["precision_original_mb"] = round(original_size, 2)
        self.stats["precision_final_mb"] = round(final_size, 2)
        self.stats["precision_saved_pct"] = round(saved_pct, 1)
        
        return df
    
    def select_output_columns(self, df: Any, feature_cols: List[str], label_cols: List[str]) -> Any:
        """
        选择最终输出的列
        
        Args:
            df: DataFrame
            feature_cols: 特征列名列表
            label_cols: 标签列名列表
            
        Returns:
            只包含所需列的 DataFrame
        """
        # 基础列
        base_cols = ['ts_code', 'trade_date']
        
        # 过滤存在的列
        feature_cols = [c for c in feature_cols if c in df.columns]
        label_cols = [c for c in label_cols if c in df.columns]
        
        output_cols = base_cols + feature_cols + label_cols
        
        logger.info(f"  📊 输出列选择: {len(output_cols)} 列")
        logger.info(f"    ✓ 基础列: {len(base_cols)}")
        logger.info(f"    ✓ 特征列: {len(feature_cols)}")
        logger.info(f"    ✓ 标签列: {len(label_cols)}")
        
        return df[output_cols]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.stats
