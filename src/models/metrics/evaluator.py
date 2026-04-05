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
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


_HOLDING_PERIOD_PATTERN = re.compile(r"_(\d+)d?$", re.IGNORECASE)


def parse_holding_period_days(col_name: Optional[str], default: int = 1) -> int:
    """
    从列名解析持有期天数。

    示例:
    - rank_ret_5d -> 5
    - sharpe_20d -> 20
    - ret_1d -> 1

    Args:
        col_name: 列名
        default: 无法解析时的默认值

    Returns:
        持有期天数（最小为 1）
    """
    safe_default = max(int(default), 1)
    if not col_name:
        return safe_default

    match = _HOLDING_PERIOD_PATTERN.search(str(col_name))
    if not match:
        return safe_default

    try:
        return max(int(match.group(1)), 1)
    except (TypeError, ValueError):
        return safe_default


def infer_pnl_return_col(
    target_col: str,
    available_cols: Optional[Union[List[str], pd.Index]] = None,
) -> Optional[str]:
    """
    根据目标列推断用于真实 PnL 计算的收益列。

    规则（优先级）:
    1) ret_{N}d
    2) return_{N}d
    3) target_col 本身

    Args:
        target_col: 目标列名
        available_cols: 可用列集合；若为 None，仅返回首选候选列

    Returns:
        收益列名；若 available_cols 提供但都不存在则返回 None
    """
    holding_days = parse_holding_period_days(target_col, default=1)
    candidates = [f"ret_{holding_days}d", f"return_{holding_days}d", target_col]

    if available_cols is None:
        return candidates[0]

    available = set(map(str, list(available_cols)))
    for col in candidates:
        if col in available:
            return col

    return None


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

    def _resolve_return_col(
        self,
        df: pd.DataFrame,
        y_true_col: str,
        return_col: Optional[str],
    ) -> str:
        """解析真实收益列；缺失时回退到 y_true_col。"""
        if return_col and return_col in df.columns:
            return return_col

        if return_col and return_col not in df.columns:
            logger.warning(
                "Return column '%s' not found, fallback to y_true column '%s'",
                return_col,
                y_true_col,
            )

        if y_true_col in df.columns:
            return y_true_col

        raise KeyError(f"Neither return_col='{return_col}' nor y_true_col='{y_true_col}' exists")

    @staticmethod
    def _deoverlap_returns(returns: np.ndarray, holding_period_days: int) -> np.ndarray:
        """
        对重叠持有期收益序列做简单去重叠抽样。

        当持有期为 N 天、按日滚动产生 N 日收益时，
        直接逐日连乘会高估累计收益并抬高 Sharpe。
        这里使用每 N 条抽 1 条的非重叠近似序列。
        """
        if returns.size == 0:
            return returns

        step = max(int(holding_period_days), 1)
        if step == 1:
            return returns

        return returns[::step]
    
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
        return_col: Optional[str] = None,
        n_groups: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        计算分组收益
        
        按预测值分组，计算每组的平均真实收益
        
        Args:
            df: 包含预测值和真实值的 DataFrame
            y_pred_col: 预测值列名
            y_true_col: 用于兼容旧接口的列名（当 return_col 未提供时生效）
            return_col: 用于计算分组收益的真实收益列
            n_groups: 分组数量
            
        Returns:
            group_returns: 分组收益 DataFrame
        """
        n_groups = n_groups or self.n_groups
        effective_return_col = self._resolve_return_col(df, y_true_col=y_true_col, return_col=return_col)
        
        results = []
        
        for date, group in df.groupby(self.date_col):
            group = group[[y_pred_col, effective_return_col]].dropna(subset=[y_pred_col, effective_return_col]).copy()
            if len(group) < n_groups:
                continue
            
            # 按预测值分组
            try:
                group["group"] = (
                    pd.qcut(
                        group[y_pred_col].rank(method="first"),
                        q=n_groups,
                        labels=False,
                        duplicates="drop",
                    )
                    + 1
                )
            except ValueError:
                continue
            
            # 计算每组平均收益
            for g in sorted(group["group"].dropna().unique().tolist()):
                g_data = group[group["group"] == g]
                if len(g_data) > 0:
                    results.append({
                        "date": date,
                        "group": g,
                        "mean_return": g_data[effective_return_col].mean(),
                        "n_stocks": len(g_data),
                    })
        
        return pd.DataFrame(results)
    
    def calc_long_short_returns(
        self,
        df: pd.DataFrame,
        y_pred_col: str = "y_pred",
        y_true_col: str = "y_true",
        return_col: Optional[str] = None,
        n_groups: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        计算多空收益
        
        做多 Top 组（预测最高）、做空 Bottom 组（预测最低）
        
        Args:
            df: 包含预测值和真实值的 DataFrame
            y_pred_col: 预测值列名
            y_true_col: 用于兼容旧接口的列名（当 return_col 未提供时生效）
            return_col: 用于计算多空收益的真实收益列
            n_groups: 分组数量
            
        Returns:
            long_short_df: 多空收益 DataFrame
        """
        n_groups = n_groups or self.n_groups
        group_returns = self.calc_group_returns(
            df,
            y_pred_col=y_pred_col,
            y_true_col=y_true_col,
            return_col=return_col,
            n_groups=n_groups,
        )
        
        if group_returns.empty:
            return pd.DataFrame()
        
        # 按日期聚合
        pivot = group_returns.pivot(index="date", columns="group", values="mean_return")
        if pivot.shape[1] < 2:
            return pd.DataFrame()

        bottom_group = int(pivot.columns.min())
        top_group = int(pivot.columns.max())
        
        # 多空收益 = Top 组 - Bottom 组（按实际可用分组）
        long_short = pd.DataFrame({
            "date": pivot.index,
            "long_return": pivot[top_group].values,
            "short_return": pivot[bottom_group].values,
            "long_short_return": pivot[top_group].values - pivot[bottom_group].values,
        })
        
        return long_short
    
    def calc_monotonicity(
        self,
        df: pd.DataFrame,
        y_pred_col: str = "y_pred",
        y_true_col: str = "y_true",
        return_col: Optional[str] = None,
        n_groups: Optional[int] = None,
    ) -> float:
        """
        计算分组单调性
        
        检查各分组收益是否呈单调递增
        返回 Spearman 相关系数（组号 vs 平均收益）
        
        Args:
            df: 包含预测值和真实值的 DataFrame
            y_pred_col: 预测值列名
            y_true_col: 用于兼容旧接口的列名（当 return_col 未提供时生效）
            return_col: 用于计算分组收益单调性的真实收益列
            n_groups: 分组数量
            
        Returns:
            monotonicity: 单调性系数 (-1 到 1)
        """
        n_groups = n_groups or self.n_groups
        group_returns = self.calc_group_returns(
            df,
            y_pred_col=y_pred_col,
            y_true_col=y_true_col,
            return_col=return_col,
            n_groups=n_groups,
        )
        
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
        return_col: Optional[str] = None,
        holding_period_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        综合评估
        
        Args:
            df: 包含预测值和真实值的 DataFrame
            y_pred_col: 预测值列名
            y_true_col: 用于 IC/RankIC 评估的目标列
            return_col: 用于真实 PnL 评估的收益列（默认与 y_true_col 相同）
            holding_period_days: 持有期天数；None 时从 return_col/y_true_col 自动解析
            
        Returns:
            metrics: 评估指标字典
        """
        effective_return_col = self._resolve_return_col(df, y_true_col=y_true_col, return_col=return_col)
        inferred_holding_days = parse_holding_period_days(effective_return_col, default=1)
        if inferred_holding_days == 1:
            inferred_holding_days = parse_holding_period_days(y_true_col, default=1)
        effective_holding_days = max(int(holding_period_days or inferred_holding_days), 1)

        # 因子表现
        factor_perf = self.calc_factor_performance(df, y_pred_col, y_true_col)
        
        # 多空收益
        long_short = self.calc_long_short_returns(
            df,
            y_pred_col=y_pred_col,
            y_true_col=y_true_col,
            return_col=effective_return_col,
        )
        
        # 分组单调性
        monotonicity = self.calc_monotonicity(
            df,
            y_pred_col=y_pred_col,
            y_true_col=y_true_col,
            return_col=effective_return_col,
        )
        
        # 多空累计收益（对重叠持有期做去重叠修正）
        raw_periods = 0
        effective_periods = 0
        if not long_short.empty:
            raw_returns = long_short["long_short_return"].dropna().to_numpy(dtype=np.float64)
            raw_periods = int(raw_returns.size)
            effective_returns = self._deoverlap_returns(raw_returns, effective_holding_days)
            effective_periods = int(effective_returns.size)

            avg_long_short = float(np.mean(raw_returns)) if raw_periods > 0 else np.nan
            cum_long_short = (
                float(np.prod(1.0 + effective_returns) - 1.0)
                if effective_periods > 0
                else np.nan
            )

            if effective_periods >= 2:
                eff_std = float(np.std(effective_returns, ddof=1))
                if eff_std > 0:
                    annualization = np.sqrt(252 / effective_holding_days)
                    sharpe_long_short = float(
                        np.mean(effective_returns) / eff_std * annualization
                    )
                else:
                    sharpe_long_short = np.nan
            else:
                sharpe_long_short = np.nan
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
            "return_col": effective_return_col,
            "holding_period_days": effective_holding_days,
            "long_short_periods_raw": raw_periods,
            "long_short_periods_effective": effective_periods,
            
            # 单调性
            "monotonicity": monotonicity,
        }
    
    def print_report(
        self,
        df: pd.DataFrame,
        y_pred_col: str = "y_pred",
        y_true_col: str = "y_true",
        return_col: Optional[str] = None,
        holding_period_days: Optional[int] = None,
        target_name: str = "Factor",
    ) -> None:
        """
        打印评估报告
        
        Args:
            df: 包含预测值和真实值的 DataFrame
            y_pred_col: 预测值列名
            y_true_col: 用于 IC/RankIC 的目标列
            return_col: 用于真实 PnL 的收益列
            holding_period_days: 持有期天数
            target_name: 因子/标签名称
        """
        metrics = self.evaluate(
            df,
            y_pred_col=y_pred_col,
            y_true_col=y_true_col,
            return_col=return_col,
            holding_period_days=holding_period_days,
        )
        
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
        print(f"  Avg Rebalance Return: {metrics['avg_long_short_return']:.4%}")
        print(f"  Cumulative Return:  {metrics['cum_long_short_return']:.2%}")
        print(f"  Sharpe Ratio:       {metrics['sharpe_long_short']:.4f}")
        print(f"  Return Column:      {metrics['return_col']}")
        print(f"  Holding Days:       {metrics['holding_period_days']}")
        print(
            "  Raw/Effective N:    "
            f"{metrics['long_short_periods_raw']}/{metrics['long_short_periods_effective']}"
        )
        
        print("\n[Monotonicity]")
        print(f"  Group Monotonicity: {metrics['monotonicity']:.4f}")
        
        print("\n" + "=" * 60)