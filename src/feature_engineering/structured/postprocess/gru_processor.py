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
        self._deferred_slice_bounds: Optional[Tuple[np.datetime64, np.datetime64]] = None
        
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
        8. 数据切分
        9. 数据排序
        
        Args:
            df: 输入 DataFrame（已经过公共清洗）
            
        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📋 GRU 专用处理")
        logger.info("=" * 60)

        # 重置延迟切分状态（防止复用实例时污染）
        self._deferred_slice_bounds = None
        
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

        # Step 5.5: 数据切分（前移以降低后续步骤内存基线）
        slice_after_rolling = bool(getattr(self.config, 'slice_after_rolling', True))
        self.stats["slice_after_rolling"] = slice_after_rolling
        if slice_after_rolling:
            df = self._slice_data(df, log_step="Step 5.5")
            gc.collect()  # [内存优化]
        
        # Step 6: 截面标准化（个股级数据）
        df = self._cross_sectional_zscore(df)
        gc.collect()  # [内存优化]
        
        # Step 7: 确保无 NaN（兜底填充）
        df = self._final_fill(df)
        gc.collect()  # [内存优化]
        
        # Step 8: 数据切分
        if slice_after_rolling:
            logger.info("  📊 Step 8: 数据切分")
            logger.info("     ✓ 已在 Step 5.5 完成切分，跳过重复执行")
        else:
            df = self._slice_data(df)
            gc.collect()  # [内存优化]
        
        # Step 9: 数据排序（ts_code -> trade_date，便于序列化）
        df = self._sort_data(df)
        gc.collect()  # [内存优化]
        
        final_rows = len(df)
        if bool(self.stats.get("slice_deferred", False)):
            final_rows = max(0, final_rows - int(self.stats.get("slice_removed", 0)))
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

        # [内存优化] 避免对超大宽表做全量 isna().sum().sum() 统计，
        # 直接逐列填充并记录受影响列数。
        filled_cols = 0
        for col in fill_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
                filled_cols += 1

        logger.info(f"     ✓ 截面填充 (fillna(0)) 完成: {filled_cols}/{len(fill_cols)} 列")
        
        self.stats["ffill_cols"] = len(ffill_cols)
        self.stats["fill_nan_cols"] = filled_cols
        
        return df
    
    def _clip_extreme_values(self, df: Any) -> Any:
        """
        Clip 去极值
        
        使用分位数 Clip，防止极端值影响梯度。
        
        策略由配置决定（clip_mode）：
        - global: 使用全量数据计算分位数
        - fixed_train_end: 仅用 clip_train_end 之前的数据计算分位数
        - disable: 跳过 Clip

        [内存优化] 分批处理，及时释放
        """
        logger.info("  📊 Step 4: Clip 去极值")
        
        lower_pct = self.config.clip_lower_percentile
        upper_pct = self.config.clip_upper_percentile
        clip_mode = getattr(self.config, 'clip_mode', 'global')
        clip_train_end = getattr(self.config, 'clip_train_end', None)

        valid_modes = {'global', 'fixed_train_end', 'disable'}
        if clip_mode not in valid_modes:
            logger.warning(f"     ⚠️ 未知 clip_mode={clip_mode}，回退到 global")
            clip_mode = 'global'
        
        logger.info(f"     分位数范围: [{lower_pct:.1%}, {upper_pct:.1%}]")
        logger.info(f"     Clip 策略: {clip_mode}")

        if clip_mode == 'disable':
            logger.info("     ✓ Clip 已禁用，跳过")
            self.stats["clip_cols"] = 0
            self.stats["clip_mode"] = clip_mode
            return df
        
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
        
        # 根据策略确定分位数样本区间
        # [内存优化] fixed_train_end 仅保留布尔 mask，避免创建 train_df 全表副本
        if clip_mode == 'fixed_train_end':
            if clip_train_end:
                train_mask = df['trade_date'] <= clip_train_end
                logger.info(
                    f"     [防泄露] 训练集用于计算分位数: "
                    f"trade_date <= {clip_train_end}, "
                    f"{int(train_mask.sum()):,} 行 / {len(df):,} 总行"
                )
            else:
                logger.warning("     ⚠️ clip_mode=fixed_train_end 但 clip_train_end 未设置，回退到 global")
                train_mask = None
                clip_mode = 'global'
                logger.info("     [工程模式] 使用全量数据计算分位数")
        else:
            train_mask = None
            logger.info("     [工程模式] 使用全量数据计算分位数")
        
        clipped_count = 0
        
        # [内存优化] 分批处理，每批 30 列
        batch_size = 30
        for i in range(0, len(clip_cols), batch_size):
            batch_cols = clip_cols[i:i+batch_size]
            
            for col in batch_cols:
                quantile_source = None
                try:
                    # [防泄露] 使用训练区间计算分位数（不使用未来数据）
                    # [内存优化] 按列切片，避免 train_df 全表副本
                    if train_mask is None:
                        quantile_source = df[col]
                    else:
                        quantile_source = df.loc[train_mask, col]

                    if self.use_gpu:
                        lower_val = float(quantile_source.quantile(lower_pct))
                        upper_val = float(quantile_source.quantile(upper_pct))
                    else:
                        lower_val = quantile_source.quantile(lower_pct)
                        upper_val = quantile_source.quantile(upper_pct)
                    
                    # 跳过全 NaN 或常量列
                    if np.isnan(lower_val) or np.isnan(upper_val) or lower_val == upper_val:
                        continue
                    
                    # Clip 应用到全量数据（使用训练集边界）
                    df[col] = df[col].clip(lower=lower_val, upper=upper_val)
                    clipped_count += 1
                    
                except Exception as e:
                    logger.debug(f"     {col} clip 失败: {e}")
                finally:
                    if quantile_source is not None:
                        del quantile_source
            
            # [内存优化] 每批后清理
            gc.collect()
        
        # [内存优化] 释放 mask 引用
        if train_mask is not None:
            del train_mask
            gc.collect()
        
        logger.info(f"     ✓ Clip 列数: {clipped_count}/{len(clip_cols)}")
        
        self.stats["clip_cols"] = clipped_count
        self.stats["clip_mode"] = clip_mode
        
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
        
        normalized_count = 0
        batch_size = max(1, int(getattr(self.config, 'rolling_zscore_batch_size', 16)))
        logger.info(f"     分批大小: {batch_size} 列/批")

        # [内存优化] 预编码 trade_date，避免每列 map(dict) 的高峰值开销
        # factorize(sort=True) 保证编码顺序与 groupby(sort=True) 一致
        date_codes, date_uniques = self.pd.factorize(df['trade_date'], sort=True)
        has_null_date = (date_codes < 0).any()
        if has_null_date:
            logger.warning("     ⚠️ trade_date 存在缺失值，相关行在滚动标准化后将被填充为 0")

        for idx, col in enumerate(rolling_cols, start=1):
            try:
                # [内存优化] 单列聚合，避免多列 groupby 的临时宽表
                daily_series = df.groupby('trade_date', sort=True)[col].first().reindex(date_uniques)

                # 计算滚动均值和标准差（按交易日序列）
                roll_mean = daily_series.rolling(window=window, min_periods=30).mean()
                roll_std = daily_series.rolling(window=window, min_periods=30).std()

                # 避免除以 0
                roll_std = roll_std.replace(0, 1e-10).fillna(1e-10)

                # 计算每日滚动 Z-Score 并 Clip（交易日级别）
                daily_zscore = ((daily_series - roll_mean) / roll_std).clip(
                    lower=-clip_val,
                    upper=clip_val,
                )

                # [内存优化] 通过日期编码回填，避免构建 dict+hash map
                z_values = daily_zscore.to_numpy(dtype=np.float32, copy=False)
                if has_null_date:
                    mapped = np.empty(len(df), dtype=np.float32)
                    valid_mask = date_codes >= 0
                    mapped[valid_mask] = z_values[date_codes[valid_mask]]
                    mapped[~valid_mask] = 0.0
                    df[col] = mapped
                    del mapped, valid_mask
                else:
                    df[col] = z_values[date_codes]

                normalized_count += 1

                # [内存优化] 立即释放中间变量
                del daily_series, roll_mean, roll_std, daily_zscore, z_values

                if idx % batch_size == 0 or idx == len(rolling_cols):
                    logger.info(f"     ↳ 已完成: {idx}/{len(rolling_cols)} 列")
                    gc.collect()

            except Exception as e:
                logger.debug(f"     {col} 滚动 zscore 失败: {e}")
                gc.collect()

        del date_codes, date_uniques
        gc.collect()
        
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

        [内存优化] 避免对超大宽表执行两次全量 isna().sum().sum() 统计。
        """
        logger.info("  📊 Step 7: 最终填充 (确保无 NaN)")
        
        # 获取数值列
        import pandas as pd
        numeric_cols = [c for c in df.columns 
                       if pd.api.types.is_numeric_dtype(df[c].dtype)]
        
        # 排除主键
        exclude_cols = ['ts_code', 'trade_date']
        fill_cols = [c for c in numeric_cols if c not in exclude_cols]

        filled_cols = 0
        batch_size = 16
        for i in range(0, len(fill_cols), batch_size):
            batch_cols = fill_cols[i:i + batch_size]
            for col in batch_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(0)
                    filled_cols += 1
            gc.collect()

        logger.info(f"     ✓ 最终填充完成: {filled_cols}/{len(fill_cols)} 列")

        self.stats["final_fill_cols"] = filled_cols
        self.stats["final_fill_validation"] = "skipped_full_scan"
        
        return df
    
    def _sort_data(self, df: Any) -> Any:
        """
        数据排序
        
        按 ts_code -> trade_date 排序，便于序列化（构建时序窗口）。
        """
        logger.info("  📊 Step 9: 数据排序")
        
        sort_by = self.config.sort_by
        logger.info(f"     排序字段: {sort_by}")

        skip_if_sorted = getattr(self.config, "skip_sort_if_already_sorted", True)
        if skip_if_sorted and self._is_already_sorted(df, sort_by):
            logger.info("     ✓ 输入数据已按目标顺序排列，跳过排序")
            self.stats["sort_by"] = sort_by
            self.stats["sort_skipped"] = True
            return df
        
        df = df.sort_values(sort_by).reset_index(drop=True)
        
        logger.info(f"     ✓ 排序完成")
        
        self.stats["sort_by"] = sort_by
        self.stats["sort_skipped"] = False
        
        return df

    def _is_already_sorted(self, df: Any, sort_by: List[str]) -> bool:
        """检查输入数据是否已按指定字段字典序排序。"""
        if self.use_gpu:
            # cuDF 下保守处理：直接执行排序，避免比较行为差异
            return False

        if len(df) <= 1:
            return True

        if any(col not in df.columns for col in sort_by):
            return False

        prev_equal = None
        valid_order = None

        for col in sort_by:
            series = df[col]
            prev_series = series.shift(1)

            ge_mask = series >= prev_series
            eq_mask = series == prev_series

            if prev_equal is None:
                valid_order = ge_mask.copy()
                prev_equal = eq_mask.copy()
            else:
                valid_order = valid_order & (~prev_equal | ge_mask)
                prev_equal = prev_equal & eq_mask

        valid_order = valid_order.fillna(False)
        valid_order.iloc[0] = True
        return bool(valid_order.all())
    
    def _slice_data(self, df: Any, log_step: str = "Step 8") -> Any:
        """
        数据切分
        
        剔除 2019.01.01-2020.06.30 的数据。
        """
        logger.info(f"  📊 {log_step}: 数据切分")
        
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
            # pandas 下使用 np.datetime64，避免额外对象开销
            cut_start_dt = np.datetime64(cut_start)
            cut_end_dt = np.datetime64(cut_end)

        # 确保 trade_date 是 datetime
        if df['trade_date'].dtype == 'object':
            if self.use_gpu:
                df['trade_date'] = self.cudf.to_datetime(df['trade_date'])
            else:
                import pandas as pd
                df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 先仅计算待删除索引（或延迟到保存阶段执行），避免整表复制峰值
        drop_mask = (df['trade_date'] >= cut_start_dt) & (df['trade_date'] <= cut_end_dt)
        removed_rows = int(drop_mask.sum())

        defer_slice = (not self.use_gpu) and bool(getattr(self.config, 'slice_defer_to_save', True))
        if defer_slice:
            self._deferred_slice_bounds = (cut_start_dt, cut_end_dt)
            final_rows = original_rows - removed_rows

            logger.info("     ✓ 启用延迟切分：保存阶段按分块过滤写出，避免内存峰值")
            logger.info(f"     ✓ 预计剔除行数: {removed_rows:,}")
            logger.info(f"     ✓ 预计保留行数: {final_rows:,}")

            self.stats["slice_deferred"] = True
            self.stats["slice_removed"] = removed_rows
            self.stats["slice_final"] = final_rows

            del drop_mask
            gc.collect()
            return df

        if self.use_gpu:
            if removed_rows > 0:
                drop_idx = df.index[drop_mask]
                df = df.drop(index=drop_idx)
                df = df.reset_index(drop=True)
                del drop_idx
        else:
            if removed_rows > 0:
                drop_idx = df.index[drop_mask]
                df.drop(index=drop_idx, inplace=True)
                # 原地重建 RangeIndex，避免后续 sort/check 的旧索引开销
                df.reset_index(drop=True, inplace=True)
                del drop_idx

        del drop_mask
        gc.collect()
        
        final_rows = len(df)
        if final_rows + removed_rows != original_rows:
            removed_rows = original_rows - final_rows
        
        logger.info(f"     ✓ 剔除行数: {removed_rows:,}")
        logger.info(f"     ✓ 保留行数: {final_rows:,}")
        
        self.stats["slice_removed"] = removed_rows
        self.stats["slice_final"] = final_rows
        self.stats["slice_deferred"] = False
        
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
            if self._deferred_slice_bounds is not None:
                import pyarrow as pa
                import pyarrow.parquet as pq

                cut_start_dt, cut_end_dt = self._deferred_slice_bounds
                chunk_rows = max(50_000, int(getattr(self.config, 'slice_save_chunk_rows', 300_000)))
                total_rows = len(df)
                written_rows = 0
                writer = None

                logger.info(f"     ↳ 延迟切分写出: 分块 {chunk_rows:,} 行")

                try:
                    for start in range(0, total_rows, chunk_rows):
                        end = min(start + chunk_rows, total_rows)
                        chunk = df.iloc[start:end]
                        keep_mask = (chunk['trade_date'] < cut_start_dt) | (chunk['trade_date'] > cut_end_dt)

                        keep_count = int(keep_mask.sum())
                        if keep_count == 0:
                            del chunk, keep_mask
                            continue

                        if keep_count == len(chunk):
                            out_chunk = chunk
                        else:
                            out_chunk = chunk.loc[keep_mask]

                        table = pa.Table.from_pandas(out_chunk, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(str(output_path), table.schema, compression='snappy')
                        writer.write_table(table)
                        written_rows += keep_count

                        del chunk, keep_mask, out_chunk, table
                        gc.collect()

                    if writer is None:
                        df.head(0).to_parquet(str(output_path), index=False, engine='pyarrow')
                    else:
                        writer.close()
                        writer = None

                finally:
                    if writer is not None:
                        writer.close()

                file_size = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"     ✓ {written_rows:,} 行, {len(df.columns)} 列, {file_size:.1f} MB")
                self.stats["slice_written_rows"] = written_rows
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
