"""
数据加载与时间序列切分

量化特点：样本之间不独立（大盘影响 + 时间序列影响）
- 严禁使用 train_test_split (随机打乱)
- 必须按时间轴切割
- 实现 Purging 机制防止标签泄露
"""

import gc
import logging
from typing import Tuple, Optional, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


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
        2. float64 -> float32 转换
        3. 解析日期列
        """
        logger.info("=" * 60)
        logger.info("📊 开始加载数据")
        logger.info("=" * 60)
        
        data_path = self.config.data_path
        logger.info(f"  📖 数据路径: {data_path}")
        
        # 读取数据
        df = self.pd.read_parquet(str(data_path))
        logger.info(f"  ✓ 原始数据: {len(df):,} 行, {len(df.columns)} 列")
        
        # 内存优化: float64 -> float32
        # LightGBM 默认使用 float32，这能节省一半内存
        float64_cols = df.select_dtypes(include=['float64']).columns.tolist()
        if float64_cols:
            df[float64_cols] = df[float64_cols].astype('float32')
            logger.info(f"  ✓ 类型转换: {len(float64_cols)} 列 float64 -> float32")
        
        # 确保 trade_date 是字符串格式 (便于比较)
        if df['trade_date'].dtype != 'object':
            if self.use_gpu:
                df['trade_date'] = df['trade_date'].astype(str)
            else:
                df['trade_date'] = df['trade_date'].astype(str)
        
        # 提取特征列 (只保留数值类型)
        all_cols = set(df.columns.tolist())
        exclude_cols = set(self.config.exclude_cols)
        
        # 过滤字符串/对象类型的列
        numeric_cols = set(df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.tolist())
        self._feature_cols = sorted(list((all_cols - exclude_cols) & numeric_cols))
        
        logger.info(f"  📋 特征列: {len(self._feature_cols)} 个")
        logger.info(f"  📋 目标列: {self.config.target_col}")
        
        # 统计信息
        self._stats["total_rows"] = len(df)
        self._stats["total_cols"] = len(df.columns)
        self._stats["feature_cols"] = len(self._feature_cols)
        
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
            # 使用 PyArrow 作为中间格式，避免 numba cuda 上下文问题
            X_train = X_train.to_arrow().to_pandas().values.astype(np.float32)
            y_train = y_train.to_arrow().to_pandas().values.astype(np.float32)
            X_valid = X_valid.to_arrow().to_pandas().values.astype(np.float32)
            y_valid = y_valid.to_arrow().to_pandas().values.astype(np.float32)
            X_test = X_test.to_arrow().to_pandas().values.astype(np.float32)
            y_test = y_test.to_arrow().to_pandas().values.astype(np.float32)
        else:
            X_train = X_train.values
            y_train = y_train.values
            X_valid = X_valid.values
            y_valid = y_valid.values
            X_test = X_test.values
            y_test = y_test.values
        
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
