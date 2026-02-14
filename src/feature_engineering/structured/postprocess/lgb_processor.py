"""
LightGBM 专用后处理模块

策略核心：保留原始物理意义，利用树模型自动分箱能力
1. 保留 NaN（LightGBM 原生支持）
2. 不做归一化/标准化（树模型基于阈值分裂，归一化无效）
3. 保留异常值（龙虎榜爆发等对捕捉"妖股"有帮助）
4. 类别特征处理（转 int/category）
5. 数据排序（trade_date -> ts_code，便于 TimeSeriesSplit）
6. 数据切分（剔除 2019-2020）
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import LGBConfig

logger = logging.getLogger(__name__)


class LGBProcessor:
    """LightGBM 专用处理器"""
    
    def __init__(self, config: Optional[LGBConfig] = None, use_gpu: bool = False):
        """
        初始化
        
        Args:
            config: LGB 处理配置
            use_gpu: 是否使用 GPU
        """
        self.config = config or LGBConfig()
        self.use_gpu = use_gpu
        self.stats: Dict[str, Any] = {}
        
        # 初始化 pandas
        if use_gpu:
            try:
                import cudf
                self.pd = cudf
                self.cudf = cudf
                logger.info("🚀 LGBProcessor: GPU 加速已启用 (cuDF)")
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
        执行 LightGBM 专用处理流程
        
        顺序：
        1. 类别特征处理
        2. NaN 填充策略（保留 NaN 或简单填充）
        3. 数据排序
        4. 数据切分
        
        Args:
            df: 输入 DataFrame（已经过公共清洗）
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📋 LightGBM 专用处理")
        logger.info("=" * 60)
        
        original_rows = len(df)
        
        # Step 1: 类别特征处理
        df = self._process_category_features(df)
        
        # Step 2: NaN 填充策略
        df = self._handle_nan(df)
        
        # Step 3: 数据排序
        df = self._sort_data(df)
        
        # Step 4: 数据切分（剔除 2019-2020）
        df = self._slice_data(df)
        
        final_rows = len(df)
        
        logger.info("-" * 60)
        logger.info(f"  📊 LightGBM 处理完成:")
        logger.info(f"     行数: {original_rows:,} -> {final_rows:,}")
        
        self.stats["original_rows"] = original_rows
        self.stats["final_rows"] = final_rows
        
        return df
    
    def _process_category_features(self, df: Any) -> Any:
        """
        类别特征处理
        
        确保 industry_idx 等列的数据类型为 int 或 category。
        """
        logger.info("  📊 Step 1: 类别特征处理")
        
        category_cols = [c for c in self.config.category_columns if c in df.columns]
        
        if not category_cols:
            logger.info("     ✓ 无需处理的类别列")
            return df
        
        for col in category_cols:
            try:
                # 先填充 NaN 为 -1（表示缺失类别）
                df[col] = df[col].fillna(-1)
                
                # 转换为整数
                if self.use_gpu:
                    df[col] = df[col].astype('int32')
                else:
                    df[col] = df[col].astype('int32')
                
                logger.info(f"     ✓ {col}: -> int32")
            except Exception as e:
                logger.warning(f"     ⚠️ {col} 转换失败: {e}")
        
        self.stats["category_cols_processed"] = category_cols
        
        return df
    
    def _handle_nan(self, df: Any) -> Any:
        """
        NaN 填充策略
        
        LightGBM 原生支持缺失值，将 NaN 单独分到一个方向。
        - 如果配置了 fill_nan_value，则用该值填充
        - 否则保留 NaN
        """
        logger.info("  📊 Step 2: NaN 填充策略")
        
        fill_value = self.config.fill_nan_value
        
        if fill_value is None:
            logger.info("     ✓ 保留 NaN (LightGBM 原生支持)")
            self.stats["nan_strategy"] = "keep"
        else:
            # 获取数值列
            import pandas as pd
            numeric_cols = [c for c in df.columns 
                          if pd.api.types.is_numeric_dtype(df[c].dtype)]
            
            # 排除标签列和主键列
            exclude_cols = ['ts_code', 'trade_date']
            label_cols = [c for c in df.columns if c.startswith(('ret_', 'label_'))]
            exclude_cols.extend(label_cols)
            
            fill_cols = [c for c in numeric_cols if c not in exclude_cols]
            
            # 填充
            nan_before = df[fill_cols].isna().sum().sum()
            for col in fill_cols:
                df[col] = df[col].fillna(fill_value)
            nan_after = df[fill_cols].isna().sum().sum()
            
            logger.info(f"     ✓ 填充 NaN 为 {fill_value}")
            logger.info(f"     ✓ 填充数量: {nan_before - nan_after:,}")
            
            self.stats["nan_strategy"] = f"fill_{fill_value}"
            self.stats["nan_filled"] = int(nan_before - nan_after)
        
        return df
    
    def _sort_data(self, df: Any) -> Any:
        """
        数据排序
        
        先按 trade_date，再按 ts_code 排序，方便 TimeSeriesSplit。
        """
        logger.info("  📊 Step 3: 数据排序")
        
        sort_by = self.config.sort_by
        logger.info(f"     排序字段: {sort_by}")
        
        df = df.sort_values(sort_by).reset_index(drop=True)
        
        logger.info(f"     ✓ 排序完成")
        
        self.stats["sort_by"] = sort_by
        
        return df
    
    def _slice_data(self, df: Any) -> Any:
        """
        数据切分
        
        剔除 2019-2020 的数据（预热期）。
        """
        logger.info("  📊 Step 4: 数据切分")
        
        cut_start = self.config.cut_start
        cut_end = self.config.cut_end
        
        logger.info(f"     剔除范围: {cut_start} ~ {cut_end}")
        
        original_rows = len(df)
        
        # 转换日期格式
        if self.use_gpu:
            import cudf
            cut_start_dt = cudf.to_datetime(cut_start)
            cut_end_dt = cudf.to_datetime(cut_end)
        else:
            import pandas as pd
            cut_start_dt = pd.to_datetime(cut_start)
            cut_end_dt = pd.to_datetime(cut_end)
        
        # 确保 trade_date 是 datetime
        if df['trade_date'].dtype == 'object':
            if self.use_gpu:
                df['trade_date'] = self.cudf.to_datetime(df['trade_date'])
            else:
                import pandas as pd
                df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 剔除指定范围
        mask = ~((df['trade_date'] >= cut_start_dt) & (df['trade_date'] <= cut_end_dt))
        df = df[mask].reset_index(drop=True)
        
        final_rows = len(df)
        removed_rows = original_rows - final_rows
        
        logger.info(f"     ✓ 剔除行数: {removed_rows:,}")
        logger.info(f"     ✓ 保留行数: {final_rows:,}")
        
        self.stats["slice_removed"] = removed_rows
        self.stats["slice_final"] = final_rows
        
        return df
    
    def save(self, df: Any, output_dir: Path) -> Path:
        """
        保存处理结果
        
        Args:
            df: 处理后的 DataFrame
            output_dir: 输出目录
            
        Returns:
            输出文件路径
        """
        import gc
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / self.config.output_file
        
        logger.info(f"  💾 保存 LGB 数据: {output_path}")
        
        # GPU DataFrame 转 CPU 后保存
        if self.use_gpu:
            df_pd = df.to_pandas()
            df_pd.to_parquet(str(output_path), index=False, engine='pyarrow')
            file_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"     ✓ {len(df_pd):,} 行, {len(df_pd.columns)} 列, {file_size:.1f} MB")
            del df_pd
            gc.collect()
        else:
            df.to_parquet(str(output_path), index=False, engine='pyarrow')
            file_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"     ✓ {len(df):,} 行, {len(df.columns)} 列, {file_size:.1f} MB")
        
        self.stats["output_path"] = str(output_path)
        self.stats["output_size_mb"] = file_size
        
        return output_path
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats
