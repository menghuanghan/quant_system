"""
高级标签生成模块

在基础收益率标签基础上，增加以下高级标签：
1. 超额收益 (Excess Return / Alpha Return) - R_stock - R_benchmark
2. 截面排名 (Cross-Sectional Rank) - 每日排名归一化
3. 夏普标签 (Risk-Adjusted Return) - Return / Volatility
4. 分位数分类 (Quantile Bins) - Top/Middle/Bottom 三分类

注意事项：
- 使用后复权价 (close_hfq) 计算收益率
- 绝对禁止将标签列作为特征输入
- 停牌股票的超额收益为负值（机会成本）
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdvancedLabelGenerator:
    """高级标签生成器"""
    
    def __init__(
        self,
        config: Any,
        ref_data: Optional[Dict[str, Any]] = None,
        use_gpu: bool = True
    ):
        """
        初始化
        
        Args:
            config: LabelConfig 配置
            ref_data: 参考数据（需包含 'benchmark' 用于计算超额收益）
            use_gpu: 是否使用 GPU
        """
        self.config = config
        self.ref_data = ref_data or {}
        self.use_gpu = use_gpu
        self.stats: Dict[str, Any] = {'generated_labels': []}
        
        self._cudf = None
        if use_gpu:
            try:
                import cudf
                self._cudf = cudf
                self.df_lib = cudf
                logger.info("🚀 AdvancedLabelGenerator: GPU 模式")
            except ImportError:
                self.df_lib = pd
                self.use_gpu = False
        else:
            self.df_lib = pd
    
    def generate_advanced_labels(self, df: Any) -> Any:
        """
        生成所有高级标签
        
        Args:
            df: 输入 DataFrame (需包含 ts_code, trade_date, close_hfq, ret_Nd)
            
        Returns:
            添加高级标签后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📋 高级标签生成")
        logger.info("=" * 60)
        
        # 1. 超额收益标签
        if getattr(self.config, 'generate_excess_return', True):
            df = self._generate_excess_return_labels(df)
        
        # 2. 截面排名标签
        if getattr(self.config, 'generate_rank_labels', True):
            df = self._generate_rank_labels(df)
        
        # 3. 夏普标签
        if getattr(self.config, 'generate_sharpe_labels', True):
            df = self._generate_sharpe_labels(df)
        
        # 4. 分位数分类标签
        if getattr(self.config, 'generate_bin_labels', True):
            df = self._generate_bin_labels(df)
        
        logger.info(f"  ✓ 共生成 {len(self.stats['generated_labels'])} 个高级标签")
        
        return df
    
    def _generate_excess_return_labels(self, df: Any) -> Any:
        """
        生成超额收益标签
        
        excess_ret_Nd = ret_Nd - benchmark_ret_Nd
        
        个股停牌时 ret_Nd = 0，但基准可能上涨，导致超额收益为负（机会成本）
        """
        logger.info("  📊 超额收益标签...")
        
        if 'benchmark' not in self.ref_data or self.ref_data['benchmark'] is None:
            logger.warning("    ⚠️ 缺少基准数据，跳过超额收益标签")
            return df
        
        benchmark = self.ref_data['benchmark']
        benchmark_code = getattr(self.config, 'benchmark_code', '000300.SH')
        excess_days = getattr(self.config, 'excess_return_days', [5, 10])
        
        # 筛选基准指数
        if self.use_gpu and self._cudf:
            bench_df = benchmark[benchmark['ts_code'] == benchmark_code].to_pandas()
        else:
            bench_df = benchmark[benchmark['ts_code'] == benchmark_code].copy()
        
        if len(bench_df) == 0:
            logger.warning(f"    ⚠️ 基准 {benchmark_code} 数据为空")
            return df
        
        # 计算基准的未来 N 日收益率
        bench_df['trade_date'] = pd.to_datetime(bench_df['trade_date'])
        bench_df = bench_df.sort_values('trade_date')
        
        for days in excess_days:
            bench_col = f'bench_ret_{days}d'
            
            # 使用收盘价计算基准收益率
            if 'close' in bench_df.columns:
                bench_df[bench_col] = (
                    bench_df['close'].shift(-days) / bench_df['close'] - 1
                )
            elif 'pct_chg' in bench_df.columns:
                # 如果只有日收益率，需要累积
                bench_df[bench_col] = (
                    (1 + bench_df['pct_chg'] / 100).rolling(window=days).apply(
                        lambda x: x.prod() - 1, raw=True
                    ).shift(-days)
                )
            
            bench_merge = bench_df[['trade_date', bench_col]].copy()
        
        # 转换主表到 pandas 进行 merge
        was_gpu = False
        if self.use_gpu and self._cudf and hasattr(df, 'to_pandas'):
            was_gpu = True
            df_pd = df.to_pandas()
        else:
            df_pd = df.copy() if isinstance(df, pd.DataFrame) else df
        
        df_pd['trade_date'] = pd.to_datetime(df_pd['trade_date'])
        n_before = len(df_pd)
        
        # Merge 基准收益率
        for days in excess_days:
            bench_col = f'bench_ret_{days}d'
            ret_col = f'ret_{days}d'
            excess_col = f'excess_ret_{days}d'
            
            if ret_col not in df_pd.columns:
                logger.warning(f"    ⚠️ 缺少 {ret_col}，跳过 {excess_col}")
                continue
            
            # Merge
            bench_merge = bench_df[['trade_date', bench_col]].copy()
            df_pd = df_pd.merge(bench_merge, on='trade_date', how='left')
            
            # 计算超额收益
            df_pd[excess_col] = df_pd[ret_col] - df_pd[bench_col].fillna(0)
            
            # 删除临时列
            df_pd = df_pd.drop(columns=[bench_col])
            
            self.stats['generated_labels'].append(excess_col)
            
            # 统计
            valid_count = df_pd[excess_col].notna().sum()
            logger.info(f"    ✓ {excess_col}: 有效 {valid_count:,} 行")
        
        # 验证行数
        if len(df_pd) != n_before:
            logger.warning(f"    ⚠️ Merge 后行数变化: {n_before} -> {len(df_pd)}")
        
        # 转回 GPU
        if was_gpu and self._cudf:
            df = self._cudf.from_pandas(df_pd)
        else:
            df = df_pd
        
        return df
    
    def _generate_rank_labels(self, df: Any) -> Any:
        """
        生成截面排名标签
        
        rank_ret_Nd = Rank(ret_Nd) / Count(stocks_on_day)
        
        归一化到 [0, 1] 区间，对全市场涨跌极其鲁棒
        """
        logger.info("  📊 截面排名标签...")
        
        rank_days = getattr(self.config, 'rank_label_days', [5, 10])
        
        # 转换主表到 pandas（groupby().transform().rank() GPU 支持有限）
        was_gpu = False
        if self.use_gpu and self._cudf and hasattr(df, 'to_pandas'):
            was_gpu = True
            df_pd = df.to_pandas()
        else:
            df_pd = df.copy() if isinstance(df, pd.DataFrame) else df
        
        for days in rank_days:
            ret_col = f'ret_{days}d'
            rank_col = f'rank_ret_{days}d'
            
            if ret_col not in df_pd.columns:
                logger.warning(f"    ⚠️ 缺少 {ret_col}，跳过 {rank_col}")
                continue
            
            # 每日截面排名，归一化到 [0, 1]
            # pct=True 表示返回百分位排名
            df_pd[rank_col] = df_pd.groupby('trade_date')[ret_col].transform(
                lambda x: x.rank(pct=True, na_option='keep')
            ).astype('float32')
            
            self.stats['generated_labels'].append(rank_col)
            
            # 统计
            valid_count = df_pd[rank_col].notna().sum()
            mean_val = df_pd[rank_col].mean()
            logger.info(f"    ✓ {rank_col}: 有效 {valid_count:,} 行, 均值 {mean_val:.3f}")
        
        # 转回 GPU
        if was_gpu and self._cudf:
            df = self._cudf.from_pandas(df_pd)
        else:
            df = df_pd
        
        return df
    
    def _generate_sharpe_labels(self, df: Any) -> Any:
        """
        生成夏普标签（风险调整收益）
        
        sharpe_Nd = Return_(t→t+N) / Std(daily_returns_(t→t+N))
        
        让模型倾向于选择稳健上涨的股票
        """
        logger.info("  📊 夏普标签...")
        
        sharpe_days = getattr(self.config, 'sharpe_label_days', [5, 10, 20])
        
        # 确保数据排序
        if self.use_gpu and self._cudf and hasattr(df, 'to_pandas'):
            was_gpu = True
            df_pd = df.to_pandas()
        else:
            was_gpu = False
            df_pd = df.copy() if isinstance(df, pd.DataFrame) else df
        
        df_pd = df_pd.sort_values(['ts_code', 'trade_date'])
        
        # 需要日收益率来计算波动率
        # 如果没有 return_1d，用 pct_chg
        daily_ret_col = 'return_1d' if 'return_1d' in df_pd.columns else 'pct_chg'
        if daily_ret_col not in df_pd.columns:
            logger.warning("    ⚠️ 缺少日收益率列，跳过夏普标签")
            return df
        
        for days in sharpe_days:
            ret_col = f'ret_{days}d'
            sharpe_col = f'sharpe_{days}d'
            
            if ret_col not in df_pd.columns:
                logger.warning(f"    ⚠️ 缺少 {ret_col}，跳过 {sharpe_col}")
                continue
            
            # 计算未来 N 日的波动率（滚动标准差，shift 看向未来）
            # 使用反向滚动
            def calc_future_std(group):
                """计算未来 N 日的波动率"""
                vol = group[daily_ret_col].iloc[::-1].rolling(
                    window=days, min_periods=max(2, days // 2)
                ).std().iloc[::-1]
                return vol
            
            df_pd['_future_vol'] = df_pd.groupby('ts_code', sort=False).apply(
                calc_future_std
            ).reset_index(level=0, drop=True)
            
            # 夏普比率 = 收益 / 波动率
            # 避免除以零
            df_pd[sharpe_col] = np.where(
                df_pd['_future_vol'] > 0.0001,
                df_pd[ret_col] / df_pd['_future_vol'],
                0.0
            )
            
            # 去极值（夏普比率可能很大）
            df_pd[sharpe_col] = df_pd[sharpe_col].clip(-10, 10).astype('float32')
            
            self.stats['generated_labels'].append(sharpe_col)
            
            # 统计
            valid_count = df_pd[sharpe_col].notna().sum()
            mean_val = df_pd[sharpe_col].mean()
            logger.info(f"    ✓ {sharpe_col}: 有效 {valid_count:,} 行, 均值 {mean_val:.3f}")
        
        # 清理临时列
        df_pd = df_pd.drop(columns=['_future_vol'], errors='ignore')
        
        # 转回 GPU
        if was_gpu and self._cudf:
            df = self._cudf.from_pandas(df_pd)
        else:
            df = df_pd
        
        return df
    
    def _generate_bin_labels(self, df: Any) -> Any:
        """
        生成分位数分类标签（三分类）
        
        label_bin_Nd:
        - 2: Top X% (做多)
        - 1: Middle (观望)
        - 0: Bottom X% (做空/避险)
        
        每日截面分位数划分
        """
        logger.info("  📊 分位数分类标签...")
        
        bin_days = getattr(self.config, 'bin_label_days', [5])
        top_pct = getattr(self.config, 'bin_top_pct', 0.30)
        bottom_pct = getattr(self.config, 'bin_bottom_pct', 0.30)
        
        # 转换到 pandas
        if self.use_gpu and self._cudf and hasattr(df, 'to_pandas'):
            was_gpu = True
            df_pd = df.to_pandas()
        else:
            was_gpu = False
            df_pd = df.copy() if isinstance(df, pd.DataFrame) else df
        
        for days in bin_days:
            ret_col = f'ret_{days}d'
            bin_col = f'label_bin_{days}d'
            
            if ret_col not in df_pd.columns:
                logger.warning(f"    ⚠️ 缺少 {ret_col}，跳过 {bin_col}")
                continue
            
            # 计算每日的分位数阈值
            def assign_bin(group):
                """分配三分类标签"""
                ret = group[ret_col]
                q_bottom = ret.quantile(bottom_pct)
                q_top = ret.quantile(1 - top_pct)
                
                labels = pd.Series(1, index=group.index)  # 默认中间
                labels[ret <= q_bottom] = 0  # 底部
                labels[ret >= q_top] = 2     # 顶部
                labels[ret.isna()] = np.nan  # 保持 NaN
                
                return labels
            
            df_pd[bin_col] = df_pd.groupby('trade_date', sort=False).apply(
                assign_bin
            ).reset_index(level=0, drop=True).astype('float32')
            
            self.stats['generated_labels'].append(bin_col)
            
            # 统计分布
            valid = df_pd[bin_col].notna()
            if valid.sum() > 0:
                bottom_cnt = (df_pd.loc[valid, bin_col] == 0).sum()
                middle_cnt = (df_pd.loc[valid, bin_col] == 1).sum()
                top_cnt = (df_pd.loc[valid, bin_col] == 2).sum()
                total = bottom_cnt + middle_cnt + top_cnt
                
                logger.info(
                    f"    ✓ {bin_col}: 做空 {bottom_cnt:,} ({bottom_cnt/total:.1%}), "
                    f"观望 {middle_cnt:,} ({middle_cnt/total:.1%}), "
                    f"做多 {top_cnt:,} ({top_cnt/total:.1%})"
                )
        
        # 转回 GPU
        if was_gpu and self._cudf:
            df = self._cudf.from_pandas(df_pd)
        else:
            df = df_pd
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats
