"""
量化专属评估指标（Evaluator）

核心指标：
- IC (Information Coefficient): 预测值与真实收益的 Pearson 相关系数
- RankIC: 预测值排名与真实收益排名的 Spearman 相关系数
- ICIR: IC 的均值除以其标准差，衡量 IC 的稳定性
- 多空收益: 做多 Top 组、做空 Bottom 组的收益
- 分组单调性: 各分组收益是否呈单调递增/递减
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """单期评估结果"""
    date: Any
    ic: float
    rank_ic: float
    n_samples: int


@dataclass  
class FactorPerformance:
    """因子表现汇总"""
    ic_mean: float
    ic_std: float
    icir: float
    rank_ic_mean: float
    rank_ic_std: float
    rank_icir: float
    ic_positive_ratio: float  # IC > 0 的比例
    t_stat: float             # IC 均值的 t 统计量
    p_value: float            # t 检验 p 值
    n_periods: int            # 评估期数


class QuantEvaluator:
    """
    量化模型评估器
    
    计算 IC, RankIC, ICIR, 多空收益等量化专属指标
    
    Example:
        >>> evaluator = QuantEvaluator()
        >>> metrics = evaluator.evaluate(oof_df, y_pred_col="y_pred", y_true_col="y_true")
        >>> print(metrics)
    """
    
    def __init__(
        self,
        date_col: str = "trade_date",
        code_col: str = "ts_code",
        n_groups: int = 10,
    ):
        """
        初始化评估器
        
        Args:
            date_col: 日期列名
            code_col: 股票代码列名
            n_groups: 分组数量（用于多空收益计算）
        """
        self.date_col = date_col
        self.code_col = code_col
        self.n_groups = n_groups
    
    def _calc_ic(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
    ) -> float:
        """
        计算 IC (Pearson 相关系数)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            ic: Pearson 相关系数
        """
        if len(y_true) < 3:
            return np.nan
        
        # 处理 NaN
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if mask.sum() < 3:
            return np.nan
        
        corr, _ = stats.pearsonr(y_true[mask], y_pred[mask])
        return corr
    
    def _calc_rank_ic(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
    ) -> float:
        """
        计算 RankIC (Spearman 相关系数)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            rank_ic: Spearman 相关系数
        """
        if len(y_true) < 3:
            return np.nan
        
        # 处理 NaN
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if mask.sum() < 3:
            return np.nan
        
        corr, _ = stats.spearmanr(y_true[mask], y_pred[mask])
        return corr
    
    def calc_daily_metrics(
        self,
        df: pd.DataFrame,
        y_pred_col: str = "y_pred",
        y_true_col: str = "y_true",
    ) -> pd.DataFrame:
        """
        计算每日的 IC 和 RankIC
        
        Args:
            df: 包含预测值和真实值的 DataFrame
            y_pred_col: 预测值列名
            y_true_col: 真实值列名
            
        Returns:
            daily_metrics: 每日指标 DataFrame
        """
        results = []
        
        for date, group in df.groupby(self.date_col):
            y_true = group[y_true_col].values
            y_pred = group[y_pred_col].values
            
            ic = self._calc_ic(y_true, y_pred)
            rank_ic = self._calc_rank_ic(y_true, y_pred)
            
            results.append(EvaluationResult(
                date=date,
                ic=ic,
                rank_ic=rank_ic,
                n_samples=len(group),
            ))
        
        daily_df = pd.DataFrame([
            {
                "date": r.date,
                "ic": r.ic,
                "rank_ic": r.rank_ic,
                "n_samples": r.n_samples,
            }
            for r in results
        ])
        
        return daily_df
    
    def calc_factor_performance(
        self,
        df: pd.DataFrame,
        y_pred_col: str = "y_pred",
        y_true_col: str = "y_true",
    ) -> FactorPerformance:
        """
        计算因子总体表现
        
        Args:
            df: 包含预测值和真实值的 DataFrame
            y_pred_col: 预测值列名
            y_true_col: 真实值列名
            
        Returns:
            performance: 因子表现汇总
        """
        daily_metrics = self.calc_daily_metrics(df, y_pred_col, y_true_col)
        
        # 过滤 NaN
        ic_values = daily_metrics["ic"].dropna().values
        rank_ic_values = daily_metrics["rank_ic"].dropna().values
        
        if len(ic_values) < 2:
            logger.warning("Not enough valid IC values for evaluation")
            return FactorPerformance(
                ic_mean=np.nan, ic_std=np.nan, icir=np.nan,
                rank_ic_mean=np.nan, rank_ic_std=np.nan, rank_icir=np.nan,
                ic_positive_ratio=np.nan, t_stat=np.nan, p_value=np.nan,
                n_periods=len(ic_values),
            )
        
        # IC 统计
        ic_mean = np.mean(ic_values)
        ic_std = np.std(ic_values, ddof=1)
        icir = ic_mean / ic_std if ic_std > 0 else np.nan
        
        # RankIC 统计
        rank_ic_mean = np.mean(rank_ic_values)
        rank_ic_std = np.std(rank_ic_values, ddof=1)
        rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else np.nan
        
        # IC 正向比例
        ic_positive_ratio = (ic_values > 0).mean()
        
        # t 检验（检验 IC 均值是否显著不为 0）
        t_stat, p_value = stats.ttest_1samp(ic_values, 0)
        
        return FactorPerformance(
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            rank_ic_mean=rank_ic_mean,
            rank_ic_std=rank_ic_std,
            rank_icir=rank_icir,
            ic_positive_ratio=ic_positive_ratio,
            t_stat=t_stat,
            p_value=p_value,
            n_periods=len(ic_values),
        )
    
    def calc_group_returns(
        self,
        df: pd.DataFrame,
        y_pred_col: str = "y_pred",
        y_true_col: str = "y_true",
        n_groups: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        计算分组收益
        
        按预测值分组，计算每组的平均真实收益
        
        Args:
            df: 包含预测值和真实值的 DataFrame
            y_pred_col: 预测值列名
            y_true_col: 真实值列名
            n_groups: 分组数量
            
        Returns:
            group_returns: 分组收益 DataFrame
        """
        n_groups = n_groups or self.n_groups
        
        results = []
        
        for date, group in df.groupby(self.date_col):
            if len(group) < n_groups:
                continue
            
            # 按预测值分组
            group = group.copy()
            group["group"] = pd.qcut(
                group[y_pred_col].rank(method="first"),
                q=n_groups,
                labels=range(1, n_groups + 1),
            )
            
            # 计算每组平均收益
            for g in range(1, n_groups + 1):
                g_data = group[group["group"] == g]
                if len(g_data) > 0:
                    results.append({
                        "date": date,
                        "group": g,
                        "mean_return": g_data[y_true_col].mean(),
                        "n_stocks": len(g_data),
                    })
        
        return pd.DataFrame(results)
    
    def calc_long_short_returns(
        self,
        df: pd.DataFrame,
        y_pred_col: str = "y_pred",
        y_true_col: str = "y_true",
        n_groups: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        计算多空收益
        
        做多 Top 组（预测最高）、做空 Bottom 组（预测最低）
        
        Args:
            df: 包含预测值和真实值的 DataFrame
            y_pred_col: 预测值列名
            y_true_col: 真实值列名
            n_groups: 分组数量
            
        Returns:
            long_short_df: 多空收益 DataFrame
        """
        n_groups = n_groups or self.n_groups
        group_returns = self.calc_group_returns(df, y_pred_col, y_true_col, n_groups)
        
        if group_returns.empty:
            return pd.DataFrame()
        
        # 按日期聚合
        pivot = group_returns.pivot(index="date", columns="group", values="mean_return")
        
        # 多空收益 = Top 组 - Bottom 组
        long_short = pd.DataFrame({
            "date": pivot.index,
            "long_return": pivot[n_groups].values,      # Top 组（做多）
            "short_return": pivot[1].values,            # Bottom 组（做空）
            "long_short_return": pivot[n_groups].values - pivot[1].values,
        })
        
        return long_short
    
    def calc_monotonicity(
        self,
        df: pd.DataFrame,
        y_pred_col: str = "y_pred",
        y_true_col: str = "y_true",
        n_groups: Optional[int] = None,
    ) -> float:
        """
        计算分组单调性
        
        检查各分组收益是否呈单调递增
        返回 Spearman 相关系数（组号 vs 平均收益）
        
        Args:
            df: 包含预测值和真实值的 DataFrame
            y_pred_col: 预测值列名
            y_true_col: 真实值列名
            n_groups: 分组数量
            
        Returns:
            monotonicity: 单调性系数 (-1 到 1)
        """
        n_groups = n_groups or self.n_groups
        group_returns = self.calc_group_returns(df, y_pred_col, y_true_col, n_groups)
        
        if group_returns.empty:
            return np.nan
        
        # 计算各组平均收益
        avg_by_group = group_returns.groupby("group")["mean_return"].mean()
        
        if len(avg_by_group) < 3:
            return np.nan
        
        # 计算组号与收益的 Spearman 相关系数
        corr, _ = stats.spearmanr(avg_by_group.index, avg_by_group.values)
        return corr
    
    def evaluate(
        self,
        df: pd.DataFrame,
        y_pred_col: str = "y_pred",
        y_true_col: str = "y_true",
    ) -> Dict[str, Any]:
        """
        综合评估
        
        Args:
            df: 包含预测值和真实值的 DataFrame
            y_pred_col: 预测值列名
            y_true_col: 真实值列名
            
        Returns:
            metrics: 评估指标字典
        """
        # 因子表现
        factor_perf = self.calc_factor_performance(df, y_pred_col, y_true_col)
        
        # 多空收益
        long_short = self.calc_long_short_returns(df, y_pred_col, y_true_col)
        
        # 分组单调性
        monotonicity = self.calc_monotonicity(df, y_pred_col, y_true_col)
        
        # 多空累计收益
        if not long_short.empty:
            cum_long_short = (1 + long_short["long_short_return"]).cumprod().iloc[-1] - 1
            avg_long_short = long_short["long_short_return"].mean()
            sharpe_long_short = (
                long_short["long_short_return"].mean() / 
                long_short["long_short_return"].std() * np.sqrt(252)
                if long_short["long_short_return"].std() > 0 else np.nan
            )
        else:
            cum_long_short = np.nan
            avg_long_short = np.nan
            sharpe_long_short = np.nan
        
        return {
            # IC 指标
            "ic_mean": factor_perf.ic_mean,
            "ic_std": factor_perf.ic_std,
            "icir": factor_perf.icir,
            "rank_ic_mean": factor_perf.rank_ic_mean,
            "rank_ic_std": factor_perf.rank_ic_std,
            "rank_icir": factor_perf.rank_icir,
            
            # 统计检验
            "ic_positive_ratio": factor_perf.ic_positive_ratio,
            "t_stat": factor_perf.t_stat,
            "p_value": factor_perf.p_value,
            "n_periods": factor_perf.n_periods,
            
            # 多空收益
            "avg_long_short_return": avg_long_short,
            "cum_long_short_return": cum_long_short,
            "sharpe_long_short": sharpe_long_short,
            
            # 单调性
            "monotonicity": monotonicity,
        }
    
    def print_report(
        self,
        df: pd.DataFrame,
        y_pred_col: str = "y_pred",
        y_true_col: str = "y_true",
        target_name: str = "Factor",
    ) -> None:
        """
        打印评估报告
        
        Args:
            df: 包含预测值和真实值的 DataFrame
            y_pred_col: 预测值列名
            y_true_col: 真实值列名
            target_name: 因子/标签名称
        """
        metrics = self.evaluate(df, y_pred_col, y_true_col)
        
        print("=" * 60)
        print(f"Factor Evaluation Report: {target_name}")
        print("=" * 60)
        
        print("\n[IC Analysis]")
        print(f"  IC Mean:            {metrics['ic_mean']:.4f}")
        print(f"  IC Std:             {metrics['ic_std']:.4f}")
        print(f"  ICIR:               {metrics['icir']:.4f}")
        print(f"  Rank IC Mean:       {metrics['rank_ic_mean']:.4f}")
        print(f"  Rank IC Std:        {metrics['rank_ic_std']:.4f}")
        print(f"  Rank ICIR:          {metrics['rank_icir']:.4f}")
        
        print("\n[Statistical Tests]")
        print(f"  IC Positive Ratio:  {metrics['ic_positive_ratio']:.2%}")
        print(f"  T-Statistic:        {metrics['t_stat']:.4f}")
        print(f"  P-Value:            {metrics['p_value']:.4f}")
        sig = "***" if metrics['p_value'] < 0.01 else "**" if metrics['p_value'] < 0.05 else "*" if metrics['p_value'] < 0.1 else ""
        print(f"  Significance:       {sig}")
        
        print("\n[Long-Short Returns]")
        print(f"  Avg Daily Return:   {metrics['avg_long_short_return']:.4%}")
        print(f"  Cumulative Return:  {metrics['cum_long_short_return']:.2%}")
        print(f"  Sharpe Ratio:       {metrics['sharpe_long_short']:.4f}")
        
        print("\n[Monotonicity]")
        print(f"  Group Monotonicity: {metrics['monotonicity']:.4f}")
        
        print("\n" + "=" * 60)