"""
GRU 深度学习专用后处理模块

策略核心：平稳化和正态化，让神经网络更容易学习
1. Log1p 变换 - 高偏度特征（成交量、金额等长尾分布）
2. 时序填充 - ffill() 模拟信息延续
3. 截面填充 - ffill 后仍空则用截面中位数或 0
4. Clip 去极值 - 防止极端值影响梯度
5. 截面标准化 - Daily Z-Score 消除市场 Beta
6. 数据切分 - 剔除 2019.01.01-2020.06.30
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import GRUConfig

logger = logging.getLogger(__name__)


class GRUProcessor:
    """GRU 专用处理器"""
    
    def __init__(self, config: Optional[GRUConfig] = None, use_gpu: bool = False):
        """
        初始化
        
        Args:
            config: GRU 处理配置
            use_gpu: 是否使用 GPU
        """
        self.config = config or GRUConfig()
        self.use_gpu = use_gpu
        self.stats: Dict[str, Any] = {}
        
        # 初始化 pandas
        if use_gpu:
            try:
                import cudf
                self.pd = cudf
                self.cudf = cudf
                logger.info("🚀 GRUProcessor: GPU 加速已启用 (cuDF)")
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
        执行 GRU 专用处理流程
        
        顺序：
        1. 删除非平稳列（价格、均线等）
        2. Log1p 变换（高偏度特征）
        3. 时序填充 + 截面填充
        4. Clip 去极值
        5. 滚动 Z-Score（市场级数据）
        6. 截面标准化 (Daily Z-Score)（个股级数据）
        7. 确保无 NaN（严禁保留 NaN）
        8. 数据排序
        9. 数据切分
        
        Args:
            df: 输入 DataFrame（已经过公共清洗）
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📋 GRU 专用处理")
        logger.info("=" * 60)
        
        original_rows = len(df)
        original_cols = len(df.columns)
        
        # Step 1: 删除非平稳列
        df = self._drop_nonstationary_cols(df)
        gc.collect()  # [内存优化]
        
        # Step 2: Log1p 变换
        df = self._apply_log1p(df)
        gc.collect()  # [内存优化]
        
        # Step 3: 时序填充 + 截面填充
        df = self._fill_missing(df)
        gc.collect()  # [内存优化]
        
        # Step 4: Clip 去极值
        df = self._clip_extreme_values(df)
        gc.collect()  # [内存优化]
        
        # Step 5: 滚动 Z-Score（市场级数据）
        df = self._rolling_zscore(df)
        gc.collect()  # [内存优化]
        
        # Step 6: 截面标准化（个股级数据）
        df = self._cross_sectional_zscore(df)
        gc.collect()  # [内存优化]
        
        # Step 7: 确保无 NaN（兜底填充）
        df = self._final_fill(df)
        gc.collect()  # [内存优化]
        
        # Step 8: 数据排序（ts_code -> trade_date，便于序列化）
        df = self._sort_data(df)
        gc.collect()  # [内存优化]
        
        # Step 9: 数据切分
        df = self._slice_data(df)
        gc.collect()  # [内存优化]
        
        final_rows = len(df)
        final_cols = len(df.columns)
        
        logger.info("-" * 60)
        logger.info(f"  📊 GRU 处理完成:")
        logger.info(f"     行数: {original_rows:,} -> {final_rows:,}")
        logger.info(f"     列数: {original_cols:,} -> {final_cols:,}")
        
        self.stats["original_rows"] = original_rows
        self.stats["original_cols"] = original_cols
        self.stats["final_rows"] = final_rows
        self.stats["final_cols"] = final_cols
        
        return df
    
    def _drop_nonstationary_cols(self, df: Any) -> Any:
        """
        删除非平稳列
        
        GRU 应该学习"涨跌幅"和"波动率"，而不是"股价是 10 元还是 100 元"。
        原始价格、均线、指数绝对点位等非平稳列会干扰模型学习。
        """
        logger.info("  📊 Step 1: 删除非平稳列 (价格、均线等)")
        
        drop_cols = [c for c in self.config.drop_cols if c in df.columns]
        
        if not drop_cols:
            logger.info("     ✓ 无需删除的列")
            return df
        
        df = df.drop(columns=drop_cols)
        
        logger.info(f"     ✓ 删除列数: {len(drop_cols)}")
        for col in drop_cols[:10]:  # 最多显示 10 个
            logger.info(f"        - {col}")
        if len(drop_cols) > 10:
            logger.info(f"        ... 还有 {len(drop_cols) - 10} 列")
        
        self.stats["dropped_cols"] = drop_cols
        self.stats["dropped_cols_count"] = len(drop_cols)
        
        return df
    
    def _apply_log1p(self, df: Any) -> Any:
        """
        Log1p 变换
        
        对高偏度特征（成交量、金额等长尾分布）应用 np.log1p(x)，
        将长尾分布拉回正态分布，让神经网络更容易学习。
        """
        logger.info("  📊 Step 2: Log1p 变换 (高偏度特征)")
        
        log1p_cols = [c for c in self.config.log1p_features if c in df.columns]
        
        if not log1p_cols:
            logger.info("     ✓ 无需 Log1p 变换的列")
            return df
        
        transformed_count = 0
        
        for col in log1p_cols:
            try:
                # 获取列数据
                col_data = df[col]
                
                # 确保非负（对负值取绝对值后变换，保留符号）
                if self.use_gpu:
                    is_negative = col_data < 0
                    abs_data = col_data.abs()
                    # log1p(|x|) * sign(x)
                    import cupy as cp
                    transformed = cp.log1p(abs_data.values)
                    transformed = self.cudf.Series(transformed, index=col_data.index)
                    # 恢复符号
                    transformed = transformed.where(~is_negative, -transformed)
                else:
                    is_negative = col_data < 0
                    abs_data = col_data.abs()
                    transformed = np.log1p(abs_data)
                    # 恢复符号
                    transformed = transformed.where(~is_negative, -transformed)
                
                df[col] = transformed
                transformed_count += 1
                
            except Exception as e:
                logger.warning(f"     ⚠️ {col} Log1p 变换失败: {e}")
        
        logger.info(f"     ✓ 变换列数: {transformed_count}/{len(log1p_cols)}")
        
        self.stats["log1p_cols"] = log1p_cols
        self.stats["log1p_transformed"] = transformed_count
        
        return df
    
    def _fill_missing(self, df: Any) -> Any:
        """
        缺失值填充
        
        1. 时序填充：对宏观数据、技术指标先做 ffill()（前向填充，模拟信息延续）
        2. 截面填充：ffill 后仍有空（如上市首日），用截面中位数或 0 填充
        
        [内存优化] 批量处理以减少 groupby 开销
        """
        logger.info("  📊 Step 3: 缺失值填充")
        
        # 按 ts_code 分组做时序填充
        ffill_cols = [c for c in self.config.ffill_features if c in df.columns]
        
        if ffill_cols:
            logger.info(f"     📝 时序填充 (ffill): {len(ffill_cols)} 列")
            
            # 确保数据按时间排序
            df = df.sort_values(['ts_code', 'trade_date'])
            
            # [内存优化] 批量 ffill，减少 groupby 调用次数
            # 分批处理，每批 20 列
            batch_size = 20
            for i in range(0, len(ffill_cols), batch_size):
                batch_cols = ffill_cols[i:i+batch_size]
                for col in batch_cols:
                    try:
                        if self.use_gpu:
                            df[col] = df.groupby('ts_code')[col].transform(
                                lambda x: x.fillna(method='ffill')
                            )
                        else:
                            df[col] = df.groupby('ts_code')[col].ffill()
                    except Exception as e:
                        logger.debug(f"        {col} ffill 失败: {e}")
                # [内存优化] 每批后清理
                gc.collect()
            
            logger.info(f"     ✓ 时序填充完成")
        
        # 截面填充（用 0 填充，避免引入未来信息）
        # 获取所有数值列
        import pandas as pd
        numeric_cols = [c for c in df.columns 
                       if pd.api.types.is_numeric_dtype(df[c].dtype)]
        
        # 排除主键和标签
        exclude_cols = ['ts_code', 'trade_date']
        label_cols = [c for c in df.columns if c.startswith(('ret_', 'label_'))]
        exclude_cols.extend(label_cols)
        
        fill_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        nan_before = df[fill_cols].isna().sum().sum()
        if self.use_gpu:
            nan_before = int(nan_before)
        
        for col in fill_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
        
        nan_after = df[fill_cols].isna().sum().sum()
        if self.use_gpu:
            nan_after = int(nan_after)
        
        logger.info(f"     ✓ 截面填充 (fillna(0)): {nan_before - nan_after:,} 个值")
        
        self.stats["ffill_cols"] = len(ffill_cols)
        self.stats["fill_nan_count"] = nan_before - nan_after
        
        return df
    
    def _clip_extreme_values(self, df: Any) -> Any:
        """
        Clip 去极值
        
        使用分位数 Clip，防止极端值影响梯度。
        
        [防泄露] 若配置了 clip_train_end，则仅用训练集数据计算分位数边界
        [内存优化] 分批处理，及时释放
        """
        logger.info("  📊 Step 4: Clip 去极值")
        
        lower_pct = self.config.clip_lower_percentile
        upper_pct = self.config.clip_upper_percentile
        clip_train_end = getattr(self.config, 'clip_train_end', None)
        
        logger.info(f"     分位数范围: [{lower_pct:.1%}, {upper_pct:.1%}]")
        if clip_train_end:
            logger.info(f"     [防泄露] 仅用训练集计算分位数: trade_date <= {clip_train_end}")
        
        # 获取数值列（排除主键、标签、类别）
        import pandas as pd
        numeric_cols = [c for c in df.columns 
                       if pd.api.types.is_numeric_dtype(df[c].dtype)]
        
        exclude_cols = ['ts_code', 'trade_date']
        label_cols = [c for c in df.columns if c.startswith(('ret_', 'label_'))]
        category_cols = [c for c in df.columns if 'idx' in c or 'code' in c.lower()]
        exclude_cols.extend(label_cols)
        exclude_cols.extend(category_cols)
        
        clip_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        # [防泄露] 如果配置了 clip_train_end，使用训练集子集计算分位数
        if clip_train_end:
            train_mask = df['trade_date'] <= clip_train_end
            train_df = df[train_mask]
            logger.info(f"     训练集用于计算分位数: {int(train_mask.sum()):,} 行 / {len(df):,} 总行")
        else:
            train_df = df
        
        clipped_count = 0
        
        # [内存优化] 分批处理，每批 30 列
        batch_size = 30
        for i in range(0, len(clip_cols), batch_size):
            batch_cols = clip_cols[i:i+batch_size]
            
            for col in batch_cols:
                try:
                    # [防泄露] 使用 train_df 计算分位数（不使用未来数据）
                    if self.use_gpu:
                        lower_val = float(train_df[col].quantile(lower_pct))
                        upper_val = float(train_df[col].quantile(upper_pct))
                    else:
                        lower_val = train_df[col].quantile(lower_pct)
                        upper_val = train_df[col].quantile(upper_pct)
                    
                    # 跳过全 NaN 或常量列
                    if np.isnan(lower_val) or np.isnan(upper_val) or lower_val == upper_val:
                        continue
                    
                    # Clip 应用到全量数据（使用训练集边界）
                    df[col] = df[col].clip(lower=lower_val, upper=upper_val)
                    clipped_count += 1
                    
                except Exception as e:
                    logger.debug(f"     {col} clip 失败: {e}")
            
            # [内存优化] 每批后清理
            gc.collect()
        
        # [内存优化] 释放训练集引用
        if clip_train_end:
            del train_df
            gc.collect()
        
        logger.info(f"     ✓ Clip 列数: {clipped_count}/{len(clip_cols)}")
        
        self.stats["clip_cols"] = clipped_count
        
        return df
    
    def _rolling_zscore(self, df: Any) -> Any:
        """
        滚动 Z-Score（市场级数据）
        
        对于市场级数据（北向资金、宏观指标、指数等），同一天在截面上是常数，
        无法做截面标准化。因此使用历史滚动窗口标准化：
        z_t = (x_t - RollMean(x, 250)) / RollStd(x, 250)
        
        衡量当天相对于过去一年的偏离程度。
        """
        logger.info("  📊 Step 5: 滚动 Z-Score (市场级数据)")
        
        rolling_cols = [c for c in self.config.rolling_zscore_features if c in df.columns]
        window = self.config.rolling_window
        clip_val = self.config.zscore_clip
        
        if not rolling_cols:
            logger.info("     ✓ 无需滚动标准化的列")
            return df
        
        logger.info(f"     滚动窗口: {window} 天")
        logger.info(f"     待处理列数: {len(rolling_cols)}")
        logger.info(f"     Clip 范围: [-{clip_val}, {clip_val}]")
        
        # 确保数据按日期排序
        df = df.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        
        # 获取唯一的 (trade_date, 市场级数据) 对，避免重复计算
        # 对于市场级数据，同一天所有股票的值相同
        import pandas as pd
        
        normalized_count = 0
        
        for col in rolling_cols:
            try:
                # 提取每日唯一值（市场级数据每天只有一个值）
                daily_values = df.groupby('trade_date')[col].first()
                
                # 计算滚动均值和标准差
                roll_mean = daily_values.rolling(window=window, min_periods=30).mean()
                roll_std = daily_values.rolling(window=window, min_periods=30).std()
                
                # 避免除以 0
                roll_std = roll_std.replace(0, 1e-10).fillna(1e-10)
                
                # 计算滚动 Z-Score
                daily_zscore = (daily_values - roll_mean) / roll_std
                
                # Clip
                daily_zscore = daily_zscore.clip(lower=-clip_val, upper=clip_val)
                
                # 映射回原 DataFrame
                date_to_zscore = daily_zscore.to_dict()
                df[col] = df['trade_date'].map(date_to_zscore)
                
                normalized_count += 1
                
            except Exception as e:
                logger.debug(f"     {col} 滚动 zscore 失败: {e}")
        
        logger.info(f"     ✓ 滚动标准化完成: {normalized_count}/{len(rolling_cols)}")
        
        self.stats["rolling_zscore_cols"] = rolling_cols
        self.stats["rolling_zscore_count"] = normalized_count
        self.stats["rolling_window"] = window
        
        return df
    
    def _cross_sectional_zscore(self, df: Any) -> Any:
        """
        截面标准化 (Daily Z-Score)
        
        对每一天（trade_date）计算当天所有股票某特征的 Mean 和 Std，
        做 (x - mean) / std，消除市场 Beta，保持平稳性。
        
        [内存优化] 分批处理，及时释放中间变量
        """
        logger.info("  📊 Step 6: 截面标准化 (Daily Z-Score)")
        
        zscore_cols = [c for c in self.config.zscore_features if c in df.columns]
        clip_val = self.config.zscore_clip
        
        if not zscore_cols:
            logger.info("     ✓ 无需标准化的列")
            return df
        
        logger.info(f"     标准化列数: {len(zscore_cols)}")
        logger.info(f"     Clip 范围: [-{clip_val}, {clip_val}]")
        
        normalized_count = 0
        
        # [内存优化] 分批处理，每批 15 列
        batch_size = 15
        for i in range(0, len(zscore_cols), batch_size):
            batch_cols = zscore_cols[i:i+batch_size]
            
            for col in batch_cols:
                try:
                    # 计算每日均值和标准差
                    daily_mean = df.groupby('trade_date', sort=False)[col].transform('mean')
                    daily_std = df.groupby('trade_date', sort=False)[col].transform('std')
                    
                    # 避免除以 0
                    daily_std = daily_std.replace(0, 1e-10).fillna(1e-10)
                    
                    # 计算 Z-Score 并直接赋值（避免创建临时变量）
                    df[col] = ((df[col] - daily_mean) / daily_std).clip(lower=-clip_val, upper=clip_val)
                    
                    # [内存优化] 立即释放中间变量
                    del daily_mean, daily_std
                    
                    normalized_count += 1
                    
                except Exception as e:
                    logger.debug(f"     {col} zscore 失败: {e}")
            
            # [内存优化] 每批后清理
            gc.collect()
        
        logger.info(f"     ✓ 标准化完成: {normalized_count}/{len(zscore_cols)}")
        
        self.stats["zscore_cols"] = normalized_count
        
        return df
    
    def _final_fill(self, df: Any) -> Any:
        """
        最终填充
        
        严禁保留 NaN（PyTorch/TensorFlow 遇到 NaN 会报错），
        对所有剩余 NaN 填 0。
        """
        logger.info("  📊 Step 7: 最终填充 (确保无 NaN)")
        
        # 获取数值列
        import pandas as pd
        numeric_cols = [c for c in df.columns 
                       if pd.api.types.is_numeric_dtype(df[c].dtype)]
        
        # 排除主键
        exclude_cols = ['ts_code', 'trade_date']
        fill_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        nan_count = df[fill_cols].isna().sum().sum()
        if self.use_gpu:
            nan_count = int(nan_count)
        
        if nan_count > 0:
            for col in fill_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(0)
            logger.info(f"     ✓ 填充剩余 NaN: {nan_count:,}")
        else:
            logger.info(f"     ✓ 无剩余 NaN")
        
        # 验证
        final_nan = df[fill_cols].isna().sum().sum()
        if self.use_gpu:
            final_nan = int(final_nan)
        
        if final_nan > 0:
            logger.warning(f"     ⚠️ 仍有 {final_nan} 个 NaN!")
        
        self.stats["final_nan_filled"] = nan_count
        
        return df
    
    def _sort_data(self, df: Any) -> Any:
        """
        数据排序
        
        按 ts_code -> trade_date 排序，便于序列化（构建时序窗口）。
        """
        logger.info("  📊 Step 8: 数据排序")
        
        sort_by = self.config.sort_by
        logger.info(f"     排序字段: {sort_by}")
        
        df = df.sort_values(sort_by).reset_index(drop=True)
        
        logger.info(f"     ✓ 排序完成")
        
        self.stats["sort_by"] = sort_by
        
        return df
    
    def _slice_data(self, df: Any) -> Any:
        """
        数据切分
        
        剔除 2019.01.01-2020.06.30 的数据。
        """
        logger.info("  📊 Step 9: 数据切分")
        
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
        
        logger.info(f"  💾 保存 GRU 数据: {output_path}")
        
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
