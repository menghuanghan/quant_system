"""
标签生成模块

生成预测目标（未来 N 日收益率）并进行处理。

支持的标签类型：
1. 基础收益率标签: ret_5d, ret_10d, ret_20d
2. 分类标签: label_5d (三分类)
3. 超额收益标签: excess_ret_5d, excess_ret_10d
4. 截面排名标签: rank_ret_5d, rank_ret_10d
5. 夏普标签: sharpe_5d, sharpe_10d
6. 分位数分类标签: label_bin_5d (三分类)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .advanced_label_generator import AdvancedLabelGenerator

logger = logging.getLogger(__name__)


class LabelGenerator:
    """标签生成器"""
    
    def __init__(
        self,
        config,
        use_gpu: bool = True,
        ref_data: Optional[Dict[str, Any]] = None
    ):
        """
        初始化标签生成器
        
        Args:
            config: LabelConfig 配置
            use_gpu: 是否使用 GPU
            ref_data: 参考数据（benchmark 等，用于高级标签）
        """
        self.config = config
        self.use_gpu = use_gpu
        self.ref_data = ref_data or {}
        self.stats: Dict[str, Any] = {}
        
        if use_gpu:
            try:
                import cudf
                self.pd = cudf
                logger.info("🚀 LabelGenerator: GPU 加速已启用 (cuDF)")
            except ImportError:
                import pandas as pd
                self.pd = pd
                self.use_gpu = False
                logger.warning("⚠️ cuDF 不可用，回退到 pandas")
        else:
            import pandas as pd
            self.pd = pd
    
    def generate_labels(self, df: Any) -> Any:
        """
        生成所有标签
        
        Args:
            df: 输入 DataFrame (需包含 ts_code, trade_date, close_hfq)
            
        Returns:
            添加了标签的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📋 Step 4: 标签生成")
        logger.info("=" * 60)
        
        # 确保按股票和日期排序
        df = df.sort_values(['ts_code', 'trade_date'])
        
        price_col = 'close_hfq' if 'close_hfq' in df.columns else 'close'
        
        # 1. 生成未来 N 日收益率
        for days in self.config.forward_days:
            label_col = f'ret_{days}d'
            df = self._generate_forward_return(df, price_col, days, label_col)
        
        logger.info(f"  ✓ 未来收益率标签: {self.config.forward_days} 日")
        
        # 2. 标签去极值
        df = self._clip_labels(df)
        
        # 3. 标记无效样本（停牌导致）
        df = self._mark_invalid_samples(df)
        
        # 4. 生成分类标签（可选）
        if self.config.generate_class_labels:
            df = self._generate_class_labels(df)
        
        # 5. 生成高级标签（超额收益、排名、夏普、分位数分类）
        df = self._generate_advanced_labels(df)
        
        return df
    
    def _generate_advanced_labels(self, df: Any) -> Any:
        """
        生成高级标签
        
        包括：超额收益、截面排名、夏普标签、分位数分类
        """
        # 检查是否启用任何高级标签
        should_generate = any([
            getattr(self.config, 'generate_excess_return', True),
            getattr(self.config, 'generate_rank_labels', True),
            getattr(self.config, 'generate_sharpe_labels', True),
            getattr(self.config, 'generate_bin_labels', True),
        ])
        
        if not should_generate:
            logger.info("  ⏭️ 高级标签生成已禁用")
            return df
        
        # 初始化高级标签生成器
        advanced_gen = AdvancedLabelGenerator(
            config=self.config,
            ref_data=self.ref_data,
            use_gpu=self.use_gpu
        )
        
        # 生成高级标签
        df = advanced_gen.generate_advanced_labels(df)
        
        # 合并统计信息
        self.stats['advanced_labels'] = advanced_gen.get_stats()
        
        return df
    
    def _generate_forward_return(
        self,
        df: Any,
        price_col: str,
        forward_days: int,
        output_col: str
    ) -> Any:
        """
        计算未来 N 日收益率
        
        ret_Nd = (Close_{t+N} - Close_t) / Close_t
        
        使用 shift(-N) 实现"偷看未来"
        """
        # 确保按 ts_code 和 trade_date 排序
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        # cuDF 兼容的 shift 操作：
        # 1. 先计算 shift 结果，保存为临时列
        # 2. 使用 groupby + cumcount 检测边界，防止跨股票 shift
        
        if self.use_gpu:
            # cuDF 原生 groupby shift
            # 使用 apply + shift 替代 transform
            grouped = df.groupby('ts_code', sort=False)
            
            # 直接使用 shift（cuDF 原生支持）
            shift_col = f'_future_{forward_days}'
            df[shift_col] = grouped[price_col].shift(-forward_days)
            
            # 计算收益率
            df[output_col] = (df[shift_col] - df[price_col]) / df[price_col]
            
            # 删除临时列 (使用 del 避免 cuDF 深拷贝)
            del df[shift_col]
        else:
            # pandas 使用 transform
            grouped = df.groupby('ts_code', sort=False)
            future_price = grouped[price_col].transform(lambda x: x.shift(-forward_days))
            df[output_col] = (future_price - df[price_col]) / df[price_col]
        
        # 统计
        if self.use_gpu:
            null_count = int(df[output_col].isna().sum())
        else:
            null_count = df[output_col].isna().sum()
        
        logger.info(f"    ✓ {output_col}: 缺失 {null_count:,} 行 (末尾 {forward_days} 天无未来数据)")
        
        return df
    
    def _clip_labels(self, df: Any) -> Any:
        """
        标签去极值
        
        将极端收益率裁剪到合理范围，防止模型被异常值主导。
        """
        lower = self.config.label_clip_lower
        upper = self.config.label_clip_upper
        
        logger.info(f"  📊 标签去极值: [{lower:.0%}, {upper:.0%}]")
        
        for days in self.config.forward_days:
            label_col = f'ret_{days}d'
            if label_col in df.columns:
                # 统计极值数量
                if self.use_gpu:
                    below = int((df[label_col] < lower).sum())
                    above = int((df[label_col] > upper).sum())
                else:
                    below = (df[label_col] < lower).sum()
                    above = (df[label_col] > upper).sum()
                
                # 裁剪
                df[label_col] = df[label_col].clip(lower=lower, upper=upper)
                
                if below > 0 or above > 0:
                    logger.info(f"    {label_col}: 裁剪 {below + above:,} 个极值 (< {lower:.0%}: {below:,}, > {upper:.0%}: {above:,})")
        
        return df
    
    def _mark_invalid_samples(self, df: Any) -> Any:
        """
        标记无效样本
        
        如果未来 N 日内停牌超过一定天数，标记为无效。
        """
        logger.info("  📊 停牌无效样本处理")
        
        primary_days = self.config.primary_label_days
        max_suspended = self.config.max_suspended_days
        
        # 检查是否有成交量列
        if 'vol' not in df.columns:
            logger.warning("    ⚠️ 无 vol 列，跳过停牌检测")
            return df
        
        # 创建停牌标记 (vol = 0)
        df['is_suspended_day'] = (df['vol'] == 0).astype('int32')
        
        # 计算未来 N 日的停牌天数
        # 使用 shift 累加方式实现"未来N日停牌天数"
        grouped = df.groupby('ts_code', sort=False)
        
        if self.use_gpu:
            # cuDF 兼容方式：累加未来 primary_days 日的停牌标记
            # 未来停牌 = sum(shift(-1), shift(-2), ..., shift(-N))
            future_sum = grouped['is_suspended_day'].shift(-1).fillna(0)
            for i in range(2, primary_days + 1):
                future_sum = future_sum + grouped['is_suspended_day'].shift(-i).fillna(0)
            df['future_suspended_days'] = future_sum.astype('int32')
        else:
            # pandas 使用 rolling 反向窗口
            future_suspended = grouped['is_suspended_day'].transform(
                lambda x: x.iloc[::-1].rolling(window=primary_days, min_periods=1).sum().iloc[::-1]
            )
            df['future_suspended_days'] = future_suspended
        
        # 标记无效样本
        label_col = f'ret_{primary_days}d'
        if label_col in df.columns:
            invalid_mask = df['future_suspended_days'] > max_suspended
            if self.use_gpu:
                invalid_count = int(invalid_mask.sum())
            else:
                invalid_count = invalid_mask.sum()
            
            # 将无效样本的标签设为 NaN
            if self.use_gpu:
                df[label_col] = df[label_col].where(~invalid_mask, other=np.nan)
            else:
                df.loc[invalid_mask, label_col] = np.nan
            
            logger.info(f"    ✓ 标记无效样本: {invalid_count:,} 行 (未来 {primary_days} 日停牌 > {max_suspended} 天)")
        
        # 清理临时列 (使用 del 避免 cuDF 深拷贝)
        del df['is_suspended_day']
        del df['future_suspended_days']
        
        return df
    
    def _generate_class_labels(self, df: Any) -> Any:
        """
        生成分类标签
        
        将连续的收益率标签转换为分类标签：
        - 0: 下跌 (ret < -threshold)
        - 1: 平盘 (-threshold <= ret <= threshold)
        - 2: 上涨 (ret > threshold)
        """
        threshold = self.config.class_threshold
        
        logger.info(f"  📊 生成分类标签 (阈值: ±{threshold:.1%})")
        
        for days in self.config.forward_days:
            ret_col = f'ret_{days}d'
            class_col = f'label_{days}d'
            
            if ret_col in df.columns:
                # 默认为平盘 (1)
                if self.use_gpu:
                    import cudf
                    df[class_col] = cudf.Series([1] * len(df), index=df.index)
                else:
                    df[class_col] = 1
                
                # 下跌 (0)
                df.loc[df[ret_col] < -threshold, class_col] = 0
                # 上涨 (2)
                df.loc[df[ret_col] > threshold, class_col] = 2
                
                # 保持 NaN
                if self.use_gpu:
                    df[class_col] = df[class_col].where(df[ret_col].notna(), other=np.nan)
                else:
                    df.loc[df[ret_col].isna(), class_col] = np.nan
                
                # 统计分布
                if self.use_gpu:
                    down_count = int((df[class_col] == 0).sum())
                    flat_count = int((df[class_col] == 1).sum())
                    up_count = int((df[class_col] == 2).sum())
                else:
                    down_count = (df[class_col] == 0).sum()
                    flat_count = (df[class_col] == 1).sum()
                    up_count = (df[class_col] == 2).sum()
                
                total = down_count + flat_count + up_count
                if total > 0:
                    logger.info(
                        f"    {class_col}: 下跌 {down_count:,} ({down_count/total:.1%}), "
                        f"平盘 {flat_count:,} ({flat_count/total:.1%}), "
                        f"上涨 {up_count:,} ({up_count/total:.1%})"
                    )
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.stats
    
    def generate_labels_column_by_column(
        self, 
        parquet_path: str,
        use_gpu: bool = True
    ) -> Dict[str, Any]:
        """
        逐列计算标签（内存高效模式）
        
        策略：
        1. 每次只读取计算当前标签所需的列
        2. 计算完成后立即释放源列
        3. 标签列驻留显存
        
        Args:
            parquet_path: 中间表 parquet 文件路径
            use_gpu: 是否使用 GPU
            
        Returns:
            标签列字典 {column_name: column_data}
        """
        import gc
        from pathlib import Path
        
        if use_gpu:
            import cudf
            import cupy as cp
            pd_lib = cudf
        else:
            import pandas as pd
            pd_lib = pd
        
        label_columns = {}
        parquet_path = Path(parquet_path)
        
        # 读取需要的基础列（包含 return_1d 用于夏普标签计算）
        base_cols = ['ts_code', 'trade_date', 'close_hfq', 'is_trading', 'return_1d', 'vol']
        available_cols = []
        for col in base_cols:
            try:
                _ = pd_lib.read_parquet(str(parquet_path), columns=[col])
                available_cols.append(col)
            except:
                pass
        
        df = pd_lib.read_parquet(str(parquet_path), columns=available_cols)
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        price_col = 'close_hfq' if 'close_hfq' in df.columns else 'close'
        
        # ============================
        # 1. 生成基础收益率标签
        # ============================
        logger.info("  📊 计算未来收益率标签...")
        
        for days in self.config.forward_days:
            label_col = f'ret_{days}d'
            df = self._generate_forward_return(df, price_col, days, label_col)
            label_columns[label_col] = df[label_col].copy()
        
        logger.info(f"    ✓ 收益率标签: {list(self.config.forward_days)}")
        
        # ============================
        # 2. 标签去极值
        # ============================
        df = self._clip_labels(df)
        
        # 更新去极值后的收益率标签
        for days in self.config.forward_days:
            label_col = f'ret_{days}d'
            if label_col in df.columns:
                label_columns[label_col] = df[label_col].copy()
        
        # ============================
        # 3. 标记无效样本
        # ============================
        df = self._mark_invalid_samples(df)
        
        # 再次更新
        for days in self.config.forward_days:
            label_col = f'ret_{days}d'
            if label_col in df.columns:
                label_columns[label_col] = df[label_col].copy()
        
        # ============================
        # 4. 生成分类标签
        # ============================
        if self.config.generate_class_labels:
            logger.info("  📊 计算分类标签...")
            df = self._generate_class_labels(df)
            
            for days in self.config.forward_days:
                class_col = f'label_{days}d'
                if class_col in df.columns:
                    label_columns[class_col] = df[class_col].copy()
        
        # ============================
        # 5. 生成高级标签
        # ============================
        logger.info("  📊 计算高级标签...")
        df = self._generate_advanced_labels(df)
        
        # 收集所有新生成的标签列
        advanced_label_prefixes = ['excess_ret_', 'rank_ret_', 'sharpe_', 'label_bin_']
        for col in df.columns:
            for prefix in advanced_label_prefixes:
                if col.startswith(prefix):
                    label_columns[col] = df[col].copy()
                    break
        
        # 释放内存
        del df
        gc.collect()
        if use_gpu:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
        
        logger.info(f"  ✅ 标签列: {list(label_columns.keys())}")
        
        return label_columns
