"""
预处理器基类

提供 GPU/CPU 自适应的 DataFrame 操作，以及通用的预处理工具方法。
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import PreprocessConfig

# 尝试导入 cuDF，如果失败则回退到 pandas
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    import pandas as cudf  # 类型别名，方便后续代码统一
    cp = np  # 类型别名

import pandas as pd

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """预处理器基类"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.use_gpu = config.use_gpu and GPU_AVAILABLE
        
        if self.use_gpu:
            logger.info("🚀 GPU 加速已启用 (RAPIDS cuDF)")
        else:
            logger.info("⚠️ GPU 不可用，使用 CPU 模式 (pandas)")
    
    @abstractmethod
    def process(self, df: Any) -> Any:
        """
        执行预处理
        
        Args:
            df: 输入 DataFrame (cudf 或 pandas)
            
        Returns:
            处理后的 DataFrame
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取预处理统计信息"""
        pass
    
    # ========================
    # 通用工具方法
    # ========================
    
    def read_parquet(self, path: Path) -> Any:
        """读取 parquet 文件"""
        if self.use_gpu:
            import cudf
            return cudf.read_parquet(str(path))
        else:
            return pd.read_parquet(str(path))
    
    def to_parquet(self, df: Any, path: Path) -> None:
        """保存 parquet 文件"""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(path), index=False)
        logger.info(f"✅ 已保存: {path} ({len(df):,} 行)")
    
    def to_pandas(self, df: Any) -> pd.DataFrame:
        """将 DataFrame 转换为 pandas（如果是 cudf）"""
        if self.use_gpu and hasattr(df, 'to_pandas'):
            return df.to_pandas()
        return df
    
    def clip_column(
        self,
        df: Any,
        column: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        inplace: bool = False
    ) -> Any:
        """
        对列进行裁剪（Clipping）
        
        Args:
            df: DataFrame
            column: 列名
            lower: 下界
            upper: 上界
            inplace: 是否原地修改
            
        Returns:
            处理后的 DataFrame
        """
        if not inplace:
            df = df.copy()
        
        if column not in df.columns:
            logger.warning(f"列 {column} 不存在，跳过 clip")
            return df
        
        original_nulls = df[column].isna().sum()
        
        if self.use_gpu:
            # cuDF 使用 clip 方法
            df[column] = df[column].clip(lower=lower, upper=upper)
        else:
            # pandas 使用 clip 方法
            df[column] = df[column].clip(lower=lower, upper=upper)
        
        return df
    
    def winsorize_by_quantile(
        self,
        df: Any,
        column: str,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        inplace: bool = False
    ) -> Any:
        """
        基于分位数的去极值（Winsorization）
        
        Args:
            df: DataFrame
            column: 列名
            lower_quantile: 下分位数
            upper_quantile: 上分位数
            inplace: 是否原地修改
            
        Returns:
            处理后的 DataFrame
        """
        if not inplace:
            df = df.copy()
        
        if column not in df.columns:
            logger.warning(f"列 {column} 不存在，跳过 winsorize")
            return df
        
        # 计算分位数边界
        if self.use_gpu:
            lower_bound = float(df[column].quantile(lower_quantile))
            upper_bound = float(df[column].quantile(upper_quantile))
        else:
            lower_bound = df[column].quantile(lower_quantile)
            upper_bound = df[column].quantile(upper_quantile)
        
        # 应用裁剪
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        if self.config.verbose:
            logger.debug(
                f"Winsorize {column}: [{lower_bound:.4f}, {upper_bound:.4f}]"
            )
        
        return df
    
    def log_transform(
        self,
        df: Any,
        column: str,
        epsilon: float = 1.0,
        output_column: Optional[str] = None,
        inplace: bool = False
    ) -> Any:
        """
        对数变换：log(max(x, epsilon))
        
        对于负值，使用 sign(x) * log(|x| + epsilon) 保留符号
        
        Args:
            df: DataFrame
            column: 输入列名
            epsilon: 最小值（避免 log(0)）
            output_column: 输出列名（默认为 column + '_log'）
            inplace: 是否原地修改
            
        Returns:
            处理后的 DataFrame
        """
        if not inplace:
            df = df.copy()
        
        if column not in df.columns:
            logger.warning(f"列 {column} 不存在，跳过 log_transform")
            return df
        
        if output_column is None:
            output_column = f"{column}_log"
        
        if self.use_gpu:
            import cudf
            
            values = df[column].fillna(0)
            # 使用 cuDF 原生方法进行符号保留的对数变换
            # sign(x) * log(|x| + epsilon)
            abs_values = values.abs()
            # 使用 numpy 对 cuDF Series 进行 log 运算（cuDF 支持 numpy 互操作）
            log_abs = np.log(abs_values + epsilon)
            # 计算符号：正数为1，负数为-1，零为0
            sign = (values > 0).astype('float64') - (values < 0).astype('float64')
            df[output_column] = sign * log_abs
        else:
            values = df[column].fillna(0)
            sign = np.sign(values)
            abs_values = np.abs(values)
            df[output_column] = sign * np.log(abs_values + epsilon)
        
        return df
    
    def inverse_transform(
        self,
        df: Any,
        column: str,
        output_column: str,
        epsilon: float = 0.01,
        inplace: bool = False
    ) -> Any:
        """
        倒数变换：output = 1 / max(|x|, epsilon) * sign(x)
        
        用于将估值比率转换为收益率形式，例如 PE -> EP
        
        Args:
            df: DataFrame
            column: 输入列名
            output_column: 输出列名
            epsilon: 最小分母（避免除以0）
            inplace: 是否原地修改
            
        Returns:
            处理后的 DataFrame
        """
        if not inplace:
            df = df.copy()
        
        if column not in df.columns:
            logger.warning(f"列 {column} 不存在，跳过 inverse_transform")
            return df
        
        if self.use_gpu:
            import cudf
            
            values = df[column]
            # 使用 cuDF 原生方法
            # 计算符号：正数为1，负数为-1
            sign = (values > 0).astype('int8') - (values < 0).astype('int8')
            abs_values = values.abs()
            # 避免除以0：使用 epsilon 作为下界
            safe_values = abs_values.clip(lower=epsilon)
            # 倒数变换
            df[output_column] = sign / safe_values
            # 保持原始 NaN（cuDF where 语法不同）
            df[output_column] = df[output_column].where(values.notna(), other=np.nan)
        else:
            values = df[column]
            sign = np.sign(values)
            abs_values = np.abs(values)
            safe_values = np.maximum(abs_values, epsilon)
            df[output_column] = sign / safe_values
            # 保持原始 NaN
            df.loc[values.isna(), output_column] = np.nan
        
        return df
    
    def calculate_lag_days(
        self,
        df: Any,
        trade_date_col: str = "trade_date",
        report_date_col: str = "report_date",
        output_col: str = "lag_days"
    ) -> Any:
        """
        计算数据时滞（交易日与财报日期的差值）
        
        Args:
            df: DataFrame
            trade_date_col: 交易日期列名
            report_date_col: 财报日期列名
            output_col: 输出列名
            
        Returns:
            添加了 lag_days 列的 DataFrame
        """
        df = df.copy()
        
        if report_date_col not in df.columns:
            logger.warning(f"列 {report_date_col} 不存在，无法计算时滞")
            return df
        
        if self.use_gpu:
            import cudf
            
            # 记录原始空值位置
            report_is_null = df[report_date_col].isna()
            
            # 确保日期列是 datetime 类型
            if df[trade_date_col].dtype == 'object':
                df[trade_date_col] = cudf.to_datetime(df[trade_date_col])
            
            if df[report_date_col].dtype == 'object':
                # 使用 pandas 作为中间转换（更鲁棒地处理空值）
                report_pd = df[report_date_col].to_pandas()
                report_pd = pd.to_datetime(report_pd, errors='coerce')
                df[report_date_col] = cudf.Series.from_pandas(report_pd)
            
            # 计算天数差
            diff = df[trade_date_col] - df[report_date_col]
            # 转换为天数（cuDF timedelta64[ns] -> 天数）
            df[output_col] = diff.astype('float64') / (10**9 * 86400)
            
            # 对于原始 report_date 为空的行，lag_days 置为 NaN
            df[output_col] = df[output_col].where(~report_is_null, other=np.nan)
        else:
            df[trade_date_col] = pd.to_datetime(df[trade_date_col], errors='coerce')
            df[report_date_col] = pd.to_datetime(df[report_date_col], errors='coerce')
            df[output_col] = (df[trade_date_col] - df[report_date_col]).dt.days
        
        return df
    
    def mask_stale_data(
        self,
        df: Any,
        lag_col: str,
        fields_to_mask: List[str],
        max_lag_days: int = 180
    ) -> Any:
        """
        将超过时滞阈值的数据标记为 NaN
        
        Args:
            df: DataFrame
            lag_col: 时滞列名
            fields_to_mask: 需要置为 NaN 的字段列表
            max_lag_days: 最大允许时滞（天）
            
        Returns:
            处理后的 DataFrame
        """
        df = df.copy()
        
        if lag_col not in df.columns:
            logger.warning(f"列 {lag_col} 不存在，无法过滤时滞数据")
            return df
        
        # 标记超过时滞的行
        stale_mask = df[lag_col] > max_lag_days
        stale_count = int(stale_mask.sum())
        
        if stale_count > 0:
            for field in fields_to_mask:
                if field in df.columns:
                    if self.use_gpu:
                        # cuDF 需要使用 where 或直接索引
                        df[field] = df[field].where(~stale_mask, other=np.nan)
                    else:
                        df.loc[stale_mask, field] = np.nan
            
            logger.info(
                f"🕐 数据时滞过滤: {stale_count:,} 行超过 {max_lag_days} 天阈值"
            )
        
        return df
