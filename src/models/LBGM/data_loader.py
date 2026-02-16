"""
数据加载与时间序列切分

量化特点：样本之间不独立（大盘影响 + 时间序列影响）
- 严禁使用 train_test_split (随机打乱)
- 必须按时间轴切割
- 实现 Purging 机制防止标签泄露

改造说明（2026.02）:
- 适配全域数据 train_lgb.parquet
- 使用动态特征识别
- 添加 inf 值处理
- 正确处理类别特征
"""

import gc
import logging
from typing import Tuple, Optional, List, Dict, Any

import numpy as np

from .config import get_feature_columns, CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)


def process_label(df, label_col: str = 'excess_ret_5d', use_gpu: bool = False):
    """
    Label 清洗与变换
    
    解决分布漂移和极端值问题：
    1. Winsorization: 3σ 截断极端值
    2. 截面标准化: 每天 Z-Score 归一化
    
    Args:
        df: 数据 DataFrame (cuDF 或 pandas)
        label_col: 标签列名
        use_gpu: 是否使用 GPU (cuDF)
        
    Returns:
        处理后的 DataFrame
    """
    if label_col not in df.columns:
        return df
    
    logger.info(f"  🏷️ Label 清洗: {label_col}")
    
    # 原始统计
    orig_mean = float(df[label_col].mean())
    orig_std = float(df[label_col].std())
    logger.info(f"    原始: mean={orig_mean:.6f}, std={orig_std:.6f}")
    
    # 1. Winsorization: 3σ 截断
    lower_bound = orig_mean - 3 * orig_std
    upper_bound = orig_mean + 3 * orig_std
    
    if use_gpu:
        # cuDF clip
        df[label_col] = df[label_col].clip(lower=lower_bound, upper=upper_bound)
    else:
        df[label_col] = df[label_col].clip(lower=lower_bound, upper=upper_bound)
    
    clipped_count = ((df[label_col] == lower_bound) | (df[label_col] == upper_bound)).sum()
    logger.info(f"    截断: 边界 [{lower_bound:.4f}, {upper_bound:.4f}], 影响 {int(clipped_count):,} 条")
    
    # 2. 截面标准化 (Cross-Sectional Z-Score)
    # 每天把所有股票的收益变成均值0、方差1的分布
    if use_gpu:
        # cuDF groupby transform
        grouped = df.groupby('trade_date')[label_col]
        means = grouped.transform('mean')
        stds = grouped.transform('std')
        # 避免除以0
        stds = stds.replace(0, 1)
        df[label_col] = (df[label_col] - means) / stds
    else:
        # pandas groupby transform
        df[label_col] = df.groupby('trade_date')[label_col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    
    # 处理后统计
    new_mean = float(df[label_col].mean())
    new_std = float(df[label_col].std())
    logger.info(f"    标准化后: mean={new_mean:.6f}, std={new_std:.6f}")
    
    return df


class DataLoader:
    """
    量化数据加载器
    
    支持:
    - cuDF GPU 加速读取
    - 自动 float64 -> float32 转换
    - 时间序列切分 (Train/Valid/Test)
    - Purging 机制 (防止标签泄露)
    """
    
    def __init__(self, config, use_gpu: bool = True):
        """
        初始化数据加载器
        
        Args:
            config: DataConfig 配置对象
            use_gpu: 是否使用 GPU (cuDF)
        """
        self.config = config
        self.use_gpu = use_gpu
        
        # 选择 DataFrame 库
        if use_gpu:
            try:
                import cudf
                self.pd = cudf
                logger.info("🚀 DataLoader: GPU 加速已启用 (cuDF)")
            except ImportError:
                import pandas as pd
                self.pd = pd
                self.use_gpu = False
                logger.warning("⚠️ cuDF 不可用，回退到 pandas")
        else:
            import pandas as pd
            self.pd = pd
            logger.info("DataLoader: 使用 CPU 模式 (pandas)")
        
        # 数据缓存
        self._data = None
        self._feature_cols = None
        self._stats = {}
    
    def load(self) -> None:
        """
        加载数据并进行预处理
        
        1. 读取 parquet 文件
        2. 处理日期格式 (datetime64 -> str)
        3. float64 -> float32 转换
        4. 处理 inf 值
        5. 类别特征类型转换
        6. 动态特征识别
        """
        logger.info("=" * 60)
        logger.info("📊 开始加载数据")
        logger.info("=" * 60)
        
        data_path = self.config.data_path
        logger.info(f"  📖 数据路径: {data_path}")
        
        # 读取数据
        df = self.pd.read_parquet(str(data_path))
        logger.info(f"  ✓ 原始数据: {len(df):,} 行, {len(df.columns)} 列")
        
        # 处理日期格式 (datetime64 -> str 'YYYY-MM-DD')
        if hasattr(df['trade_date'].dtype, 'name') and 'datetime' in str(df['trade_date'].dtype):
            if self.use_gpu:
                # cuDF 日期处理
                df['trade_date'] = df['trade_date'].astype(str).str.slice(0, 10)
            else:
                df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')
            logger.info("  ✓ 日期格式转换: datetime64 -> str")
        elif df['trade_date'].dtype == 'object' or str(df['trade_date'].dtype) == 'str':
            # 已经是字符串，确保格式一致
            pass
        
        # 内存优化: float64 -> float32
        # LightGBM 默认使用 float32，这能节省一半内存
        float64_cols = df.select_dtypes(include=['float64', 'Float64']).columns.tolist()
        if float64_cols:
            for col in float64_cols:
                df[col] = df[col].astype('float32')
            logger.info(f"  ✓ 类型转换: {len(float64_cols)} 列 float64 -> float32")
        
        # 处理 inf 值 (LightGBM 虽然能处理 NaN，但 inf 会导致问题)
        numeric_cols = df.select_dtypes(include=['float32', 'float64', 'Float32', 'Float64']).columns.tolist()
        inf_count = 0
        for col in numeric_cols:
            if self.use_gpu:
                # cuDF 处理
                inf_mask = (df[col] == float('inf')) | (df[col] == float('-inf'))
                inf_count += inf_mask.sum()
                df[col] = df[col].replace([float('inf'), float('-inf')], np.nan)
            else:
                inf_mask = np.isinf(df[col].values)
                if inf_mask.any():
                    inf_count += inf_mask.sum()
                    df.loc[inf_mask, col] = np.nan
        if inf_count > 0:
            logger.info(f"  ✓ inf 值处理: 替换 {inf_count:,} 个 inf -> NaN")
        
        # 类别特征类型转换 (LightGBM 原生支持)
        cat_converted = 0
        # 修复分类特征负值（LightGBM CUDA 不支持负值分类特征）
        for cat_col in CATEGORICAL_FEATURES:
            if cat_col in df.columns:
                # 先处理负值: -1 表示未知/缺失，转为 max+1 作为独立类别
                if self.use_gpu:
                    min_val = df[cat_col].min()
                    if min_val is not None and min_val < 0:
                        max_val = df[cat_col].max()
                        unknown_code = int(max_val) + 1 if max_val is not None else 0
                        # 将负值替换为 unknown_code
                        df[cat_col] = df[cat_col].fillna(-999).astype('int32')
                        mask = df[cat_col] < 0
                        df.loc[mask, cat_col] = unknown_code
                        logger.info(f"  ✓ 分类特征 {cat_col}: 负值 -> {unknown_code} (未知类别)")
                else:
                    min_val = df[cat_col].min()
                    if min_val is not None and min_val < 0:
                        max_val = df[cat_col].max()
                        unknown_code = int(max_val) + 1 if max_val is not None else 0
                        df.loc[df[cat_col] < 0, cat_col] = unknown_code
                        logger.info(f"  ✓ 分类特征 {cat_col}: 负值 -> {unknown_code} (未知类别)")
                
                # 类别特征类型转换
                if self.use_gpu:
                    # cuDF category（如果不支持就保持 int）
                    try:
                        df[cat_col] = df[cat_col].astype('category')
                        cat_converted += 1
                    except Exception:
                        pass  # 保持原类型
                else:
                    df[cat_col] = df[cat_col].astype('category')
                    cat_converted += 1
        if cat_converted > 0:
            logger.info(f"  ✓ 类别特征转换: {cat_converted} 列 -> category")
        
        # 动态特征识别（排除法）
        all_cols = df.columns.tolist()
        extra_exclude = self.config.extra_exclude_cols if hasattr(self.config, 'extra_exclude_cols') else []
        self._feature_cols = get_feature_columns(all_cols, exclude_cols=extra_exclude)
        
        # 过滤掉非数值列（保留类别特征）
        numeric_types = ['float32', 'float64', 'int32', 'int64', 'int8', 'Int32', 'Float32', 'Float64', 'category']
        valid_cols = df.select_dtypes(include=numeric_types + ['category']).columns.tolist()
        self._feature_cols = [c for c in self._feature_cols if c in valid_cols]
        
        logger.info(f"  📋 特征列: {len(self._feature_cols)} 个")
        logger.info(f"  📋 目标列: {self.config.target_col}")
        
        # 打印部分特征名（调试用）
        sample_features = self._feature_cols[:10]
        logger.debug(f"  📋 特征示例: {sample_features}...")
        
        # 统计信息
        self._stats["total_rows"] = len(df)
        self._stats["total_cols"] = len(df.columns)
        self._stats["feature_cols"] = len(self._feature_cols)
        
        # Label 清洗与变换 (Winsorization + 截面标准化)
        df = process_label(df, label_col=self.config.target_col, use_gpu=self.use_gpu)
        
        self._data = df
        logger.info("  ✅ 数据加载完成")
    
    def split(self) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """
        时间序列切分
        
        Returns:
            X_train, y_train, X_valid, y_valid, X_test, y_test
        """
        if self._data is None:
            raise RuntimeError("请先调用 load() 加载数据")
        
        logger.info("=" * 60)
        logger.info("📊 时间序列切分 (Time-Series Split)")
        logger.info("=" * 60)
        
        df = self._data
        target_col = self.config.target_col
        purge_days = self.config.purge_days
        
        # 获取日期边界
        train_start = self.config.train_start
        train_end = self.config.train_end
        valid_start = self.config.valid_start
        valid_end = self.config.valid_end
        test_start = self.config.test_start
        test_end = self.config.test_end
        
        logger.info(f"  📅 训练集: {train_start} ~ {train_end}")
        logger.info(f"  📅 验证集: {valid_start} ~ {valid_end}")
        logger.info(f"  📅 测试集: {test_start} ~ {test_end}")
        logger.info(f"  🔒 Purging: {purge_days} 天")
        
        # 切分数据
        # 注意：trade_date 是字符串格式 'YYYY-MM-DD'
        train_mask = (df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)
        valid_mask = (df['trade_date'] >= valid_start) & (df['trade_date'] <= valid_end)
        test_mask = (df['trade_date'] >= test_start) & (df['trade_date'] <= test_end)
        
        train_df = df[train_mask]
        valid_df = df[valid_mask]
        test_df = df[test_mask]
        
        # Purging: 移除边界数据以防止标签泄露
        # 训练集末尾的 purge_days 天的样本，其标签可能涉及验证集的数据
        if purge_days > 0:
            # 获取训练集的最后 purge_days 个交易日
            train_dates = train_df['trade_date'].unique()
            if self.use_gpu:
                train_dates = train_dates.to_numpy()
            train_dates = sorted(train_dates)
            
            if len(train_dates) > purge_days:
                purge_cutoff = train_dates[-purge_days]
                train_df = train_df[train_df['trade_date'] < purge_cutoff]
                logger.info(f"  🔒 Purging 训练集: 移除 {purge_cutoff} 之后的数据")
        
        # 提取特征和标签
        X_train = train_df[self._feature_cols]
        y_train = train_df[target_col]
        
        X_valid = valid_df[self._feature_cols]
        y_valid = valid_df[target_col]
        
        X_test = test_df[self._feature_cols]
        y_test = test_df[target_col]
        
        # 保存日期信息（用于后续分析）
        self._train_dates = train_df['trade_date']
        self._valid_dates = valid_df['trade_date']
        self._test_dates = test_df['trade_date']
        self._test_codes = test_df['ts_code']
        
        # 统计信息
        logger.info(f"  📊 切分结果:")
        logger.info(f"     训练集: {len(X_train):,} 样本")
        logger.info(f"     验证集: {len(X_valid):,} 样本")
        logger.info(f"     测试集: {len(X_test):,} 样本")
        
        self._stats["train_samples"] = len(X_train)
        self._stats["valid_samples"] = len(X_valid)
        self._stats["test_samples"] = len(X_test)
        
        # 转换为 numpy (LightGBM 需要)
        # 注意: cuDF to_numpy 可能有 GPU 上下文问题，使用 to_arrow + to_numpy
        if self.use_gpu:
            import numpy as np
            import pandas as pd
            # 使用 PyArrow 作为中间格式，避免 numba cuda 上下文问题
            # 先转为 pandas，统一转为 float64 (支持 NA/NaN)，再转为 numpy float32
            def to_numpy_safe(df):
                pdf = df.to_arrow().to_pandas()
                # 直接转为 float64 再转 float32，避免 nullable int 的 NA 问题
                return pdf.astype(np.float64).fillna(np.nan).values.astype(np.float32)
            
            def to_numpy_safe_series(s):
                ps = s.to_arrow().to_pandas()
                return ps.astype(np.float64).fillna(np.nan).values.astype(np.float32)
            
            X_train = to_numpy_safe(X_train)
            y_train = to_numpy_safe_series(y_train)
            X_valid = to_numpy_safe(X_valid)
            y_valid = to_numpy_safe_series(y_valid)
            X_test = to_numpy_safe(X_test)
            y_test = to_numpy_safe_series(y_test)
        else:
            import numpy as np
            # pandas 模式同样需要处理 NA
            def to_numpy_safe_pd(df):
                # 直接转为 float64 再转 float32
                return df.astype(np.float64).fillna(np.nan).values.astype(np.float32)
            
            def to_numpy_safe_series_pd(s):
                return s.astype(np.float64).fillna(np.nan).values.astype(np.float32)
            
            X_train = to_numpy_safe_pd(X_train.copy())
            y_train = to_numpy_safe_series_pd(y_train.copy())
            X_valid = to_numpy_safe_pd(X_valid.copy())
            y_valid = to_numpy_safe_series_pd(y_valid.copy())
            X_test = to_numpy_safe_pd(X_test.copy())
            y_test = to_numpy_safe_series_pd(y_test.copy())
        
        logger.info("  ✅ 时间序列切分完成")
        
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    
    def get_feature_names(self) -> List[str]:
        """获取特征列名"""
        return self._feature_cols
    
    def get_test_info(self) -> Dict[str, Any]:
        """获取测试集的日期和股票代码信息"""
        if self.use_gpu:
            return {
                "dates": self._test_dates.reset_index(drop=True).to_numpy(),
                "codes": self._test_codes.reset_index(drop=True).to_numpy(),
            }
        return {
            "dates": self._test_dates,
            "codes": self._test_codes,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats
    
    def cleanup(self) -> None:
        """释放内存"""
        del self._data
        self._data = None
        gc.collect()
        logger.info("  🧹 内存已释放")
