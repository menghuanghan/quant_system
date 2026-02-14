"""
公共清洗模块

执行 LightGBM 和 GRU 处理之前的通用清洗步骤：0. 内存优化 (float64 -> float32)  # [内存优化] 减少 50% 内存占用1. 类别编码 (删除字符串列，对无索引列手动编码)
2. 静态列剔除 (std == 0)
3. 标签清洗 (删除 ret_5d 为 NaN 的行)
4. 无限值处理 (inf -> NaN)
"""

import gc
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .config import CommonCleanConfig

logger = logging.getLogger(__name__)


class CommonCleaner:
    """公共清洗器"""
    
    def __init__(self, config: Optional[CommonCleanConfig] = None, use_gpu: bool = False):
        """
        初始化
        
        Args:
            config: 公共清洗配置
            use_gpu: 是否使用 GPU
        """
        self.config = config or CommonCleanConfig()
        self.use_gpu = use_gpu
        self.stats: Dict[str, Any] = {}
        
        # 初始化 pandas
        if use_gpu:
            try:
                import cudf
                self.pd = cudf
                logger.info("🚀 CommonCleaner: GPU 加速已启用 (cuDF)")
            except ImportError:
                import pandas as pd
                self.pd = pd
                self.use_gpu = False
                logger.warning("⚠️ cuDF 不可用，回退到 pandas")
        else:
            import pandas as pd
            self.pd = pd
    
    def process(self, df: Any) -> Any:
        """
        执行公共清洗流程
        
        顺序：
        1. 类别编码 (先处理字符串列，避免后续步骤报错)
        2. 无限值处理 (避免影响后续统计)
        3. 标签清洗 (删除无标签的行)
        4. 静态列剔除 (删除常量列)
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            清洗后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📋 公共清洗 (Common Cleaning)")
        logger.info("=" * 60)
        
        original_rows = len(df)
        original_cols = len(df.columns)
        
        # Step 0: 内存优化（float64 -> float32）
        df = self._downcast_dtypes(df)
        gc.collect()
        
        # Step 1: 类别编码（删除字符串列，手动编码无索引列）
        df = self._encode_categorical_columns(df)
        gc.collect()
        
        # Step 2: 无限值处理
        df = self._replace_infinite_values(df)
        gc.collect()
        
        # Step 3: 标签清洗
        df = self._clean_labels(df)
        gc.collect()
        
        # Step 4: 静态列剔除
        df = self._drop_constant_columns(df)
        gc.collect()
        
        final_rows = len(df)
        final_cols = len(df.columns)
        
        logger.info("-" * 60)
        logger.info(f"  📊 公共清洗完成:")
        logger.info(f"     行数: {original_rows:,} -> {final_rows:,} (删除 {original_rows - final_rows:,})")
        logger.info(f"     列数: {original_cols} -> {final_cols} (删除 {original_cols - final_cols})")
        
        self.stats["original_rows"] = original_rows
        self.stats["original_cols"] = original_cols
        self.stats["final_rows"] = final_rows
        self.stats["final_cols"] = final_cols
        
        return df
    
    def _downcast_dtypes(self, df: Any) -> Any:
        """
        内存优化：降低数据精度
        
        float64 -> float32，减少约 50% 内存占用
        """
        logger.info("  📊 Step 0: 内存优化 (float64 -> float32)")
        
        import pandas as pd
        
        # 检查当前内存使用（只在 pandas 下有效）
        memory_before = 0
        if not self.use_gpu:
            try:
                memory_before = df.memory_usage(deep=True).sum() / (1024 ** 3)
            except:
                pass
        
        downcast_count = 0
        
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if dtype_str == 'float64':
                try:
                    df[col] = df[col].astype('float32')
                    downcast_count += 1
                except Exception as e:
                    logger.debug(f"     跳过 {col}: {e}")
        
        # 检查优化后内存
        memory_after = 0
        if not self.use_gpu:
            try:
                memory_after = df.memory_usage(deep=True).sum() / (1024 ** 3)
                saved = memory_before - memory_after
                logger.info(f"     ✓ 降精度列数: {downcast_count}")
                logger.info(f"     ✓ 内存: {memory_before:.2f}GB -> {memory_after:.2f}GB (节省 {saved:.2f}GB)")
            except:
                logger.info(f"     ✓ 降精度列数: {downcast_count}")
        else:
            logger.info(f"     ✓ 降精度列数: {downcast_count}")
        
        self.stats["downcast_count"] = downcast_count
        
        return df
    
    def _encode_categorical_columns(self, df: Any) -> Any:
        """
        类别编码
        
        1. 删除有对应索引的字符串列（如 industry, sw_l1_name 等）
        2. 对无索引的类别列进行手动编码（如 market）
        """
        logger.info("  📊 Step 1: 类别编码 (删除字符串列 + 手动编码)")
        
        dropped_cols = []
        encoded_cols = []
        
        # 1. 删除配置中指定的字符串列
        for col in self.config.drop_string_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
                dropped_cols.append(col)
        
        logger.info(f"     ✓ 删除字符串列: {len(dropped_cols)}")
        if dropped_cols:
            for col in dropped_cols:
                logger.info(f"        - {col}")
        
        # 2. 手动编码无索引的类别列
        for col, mapping in self.config.manual_encode_mapping.items():
            if col in df.columns:
                # 使用映射进行编码，未知值填充为 -1
                if self.use_gpu:
                    # cuDF 需要转换为 pandas 处理后再转回
                    col_data = df[col].to_pandas() if hasattr(df[col], 'to_pandas') else df[col]
                    encoded = col_data.map(mapping).fillna(-1).astype('int32')
                    df[col] = encoded.values
                else:
                    df[col] = df[col].map(mapping).fillna(-1).astype('int32')
                encoded_cols.append(col)
        
        logger.info(f"     ✓ 手动编码列: {len(encoded_cols)}")
        if encoded_cols:
            for col in encoded_cols:
                unique_vals = df[col].unique()
                logger.info(f"        - {col}: {len(unique_vals)} unique values")
        
        # 检查是否还有剩余的字符串列（排除 ts_code 和 trade_date）
        remaining_str_cols = []
        for col in df.columns:
            if col in ['ts_code', 'trade_date']:
                continue
            dtype_str = str(df[col].dtype).lower()
            if 'str' in dtype_str or 'object' in dtype_str or dtype_str == 'string':
                remaining_str_cols.append(col)
        
        if remaining_str_cols:
            logger.warning(f"     ⚠️ 仍有未处理的字符串列: {remaining_str_cols}")
        else:
            logger.info("     ✓ 所有字符串列已处理")
        
        self.stats["dropped_string_cols"] = dropped_cols
        self.stats["encoded_cols"] = encoded_cols
        self.stats["remaining_string_cols"] = remaining_str_cols
        
        return df
    
    def _replace_infinite_values(self, df: Any) -> Any:
        """
        无限值处理
        
        将所有 np.inf / -np.inf 替换为 NaN，防止计算 Loss 时梯度爆炸。
        """
        logger.info("  📊 Step 2: 无限值处理 (inf -> NaN)")
        
        # 获取数值列
        numeric_cols = self._get_numeric_columns(df)
        
        inf_count = 0
        neg_inf_count = 0
        affected_cols = []
        
        for col in numeric_cols:
            if self.use_gpu:
                # cuDF 使用不同的方法检测 inf
                try:
                    col_data = df[col]
                    pos_inf_mask = col_data == float('inf')
                    neg_inf_mask = col_data == float('-inf')
                    pos_count = int(pos_inf_mask.sum())
                    neg_count = int(neg_inf_mask.sum())
                except:
                    continue
            else:
                col_data = df[col]
                pos_inf_mask = np.isinf(col_data) & (col_data > 0)
                neg_inf_mask = np.isinf(col_data) & (col_data < 0)
                pos_count = pos_inf_mask.sum()
                neg_count = neg_inf_mask.sum()
            
            if pos_count > 0 or neg_count > 0:
                inf_count += pos_count
                neg_inf_count += neg_count
                affected_cols.append(col)
                # 替换 inf 为 NaN
                df[col] = df[col].replace([float('inf'), float('-inf')], np.nan)
        
        logger.info(f"     ✓ +inf 数量: {inf_count:,}")
        logger.info(f"     ✓ -inf 数量: {neg_inf_count:,}")
        logger.info(f"     ✓ 涉及列数: {len(affected_cols)}")
        
        self.stats["inf_count"] = inf_count
        self.stats["neg_inf_count"] = neg_inf_count
        self.stats["inf_affected_cols"] = len(affected_cols)
        
        return df
    
    def _clean_labels(self, df: Any) -> Any:
        """
        标签清洗
        
        删除主要标签 (ret_5d) 为 NaN 的行，因为没有标签的数据对监督学习无意义。
        """
        logger.info(f"  📊 Step 3: 标签清洗 (删除 {self.config.primary_label} 为 NaN 的行)")
        
        primary_label = self.config.primary_label
        
        if primary_label not in df.columns:
            logger.warning(f"     ⚠️ 主标签列 '{primary_label}' 不存在，跳过")
            return df
        
        original_rows = len(df)
        
        # 删除标签为 NaN 的行
        df = df.dropna(subset=[primary_label])
        
        final_rows = len(df)
        removed_rows = original_rows - final_rows
        
        logger.info(f"     ✓ 删除 NaN 标签行: {removed_rows:,}")
        logger.info(f"     ✓ 保留有效行: {final_rows:,}")
        
        self.stats["label_nan_removed"] = removed_rows
        
        return df
    
    def _drop_constant_columns(self, df: Any) -> Any:
        """
        静态列剔除
        
        删除 std == 0 的列（常量列），这些列对模型无意义。
        """
        logger.info("  📊 Step 4: 静态列剔除 (std == 0)")
        
        # 获取数值列
        numeric_cols = self._get_numeric_columns(df)
        
        constant_cols = []
        
        for col in numeric_cols:
            try:
                col_data = df[col]
                # 跳过全 NaN 列
                non_nan_count = col_data.notna().sum()
                if self.use_gpu:
                    non_nan_count = int(non_nan_count)
                
                if non_nan_count == 0:
                    constant_cols.append(col)
                    continue
                
                # 计算标准差
                std_val = col_data.std()
                if self.use_gpu:
                    std_val = float(std_val)
                
                if std_val == 0 or np.isnan(std_val):
                    constant_cols.append(col)
            except Exception as e:
                logger.debug(f"     跳过列 {col}: {e}")
                continue
        
        if constant_cols:
            logger.info(f"     ✓ 发现常量列: {len(constant_cols)}")
            for col in constant_cols[:10]:  # 只显示前10个
                logger.info(f"        - {col}")
            if len(constant_cols) > 10:
                logger.info(f"        ... (还有 {len(constant_cols) - 10} 个)")
            
            df = df.drop(columns=constant_cols)
        else:
            logger.info(f"     ✓ 无常量列")
        
        self.stats["constant_cols_dropped"] = constant_cols
        self.stats["constant_cols_count"] = len(constant_cols)
        
        return df
    
    def _get_numeric_columns(self, df: Any) -> List[str]:
        """获取数值类型的列"""
        import pandas as pd
        
        numeric_cols = []
        for col in df.columns:
            dtype = df[col].dtype
            # 使用 pandas 的 is_numeric_dtype 判断
            if pd.api.types.is_numeric_dtype(dtype):
                numeric_cols.append(col)
        return numeric_cols
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats
