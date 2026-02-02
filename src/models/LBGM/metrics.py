"""
自定义评价指标

量化投资中，预测值和真实值偏差多少不重要，
重要的是方向对不对以及排名对不对。

核心指标:
- IC (Information Coefficient): 预测值与真实值的皮尔逊相关系数
- RankIC: 预测值排名与真实值排名的斯皮尔曼相关系数
- ICIR (IC Information Ratio): IC均值 / IC标准差，衡量稳定性
"""

import logging
from typing import Tuple, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算皮尔逊相关系数
    
    Args:
        x: 预测值
        y: 真实值
        
    Returns:
        相关系数 [-1, 1]
    """
    # 处理 NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return 0.0
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    # 计算相关系数
    x_mean = x_clean.mean()
    y_mean = y_clean.mean()
    
    x_std = x_clean.std()
    y_std = y_clean.std()
    
    if x_std == 0 or y_std == 0:
        return 0.0
    
    corr = ((x_clean - x_mean) * (y_clean - y_mean)).mean() / (x_std * y_std)
    return float(corr)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算斯皮尔曼秩相关系数 (RankIC)
    
    使用纯 numpy 实现，避免 scipy 依赖问题
    
    Args:
        x: 预测值
        y: 真实值
        
    Returns:
        秩相关系数 [-1, 1]
    """
    # 处理 NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return 0.0
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    # 计算排名 (使用 argsort 两次得到排名)
    def rank_array(arr):
        """计算排名 (平均排名处理相同值)"""
        temp = arr.argsort()
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(arr))
        return ranks
    
    x_rank = rank_array(x_clean)
    y_rank = rank_array(y_clean)
    
    # 计算皮尔逊相关系数 (对排名)
    return pearson_corr(x_rank, y_rank)


def ic_eval(preds: np.ndarray, train_data) -> Tuple[str, float, bool]:
    """
    LightGBM 自定义评价函数: IC (Information Coefficient)
    
    在每一轮 Boosting 结束后计算 IC，用于 Early Stopping。
    
    Args:
        preds: 模型预测值
        train_data: LightGBM Dataset 对象
        
    Returns:
        (eval_name, eval_result, is_higher_better)
        - eval_name: 指标名称
        - eval_result: 指标值
        - is_higher_better: True 表示越大越好
    """
    labels = train_data.get_label()
    
    # 计算皮尔逊 IC
    ic = pearson_corr(preds, labels)
    
    return ('IC', ic, True)


def rank_ic_eval(preds: np.ndarray, train_data) -> Tuple[str, float, bool]:
    """
    LightGBM 自定义评价函数: RankIC (Spearman Correlation)
    
    RankIC 比 IC 更稳定，对极值不敏感。
    
    Args:
        preds: 模型预测值
        train_data: LightGBM Dataset 对象
        
    Returns:
        (eval_name, eval_result, is_higher_better)
    """
    labels = train_data.get_label()
    
    # 计算斯皮尔曼 RankIC
    rank_ic = spearman_corr(preds, labels)
    
    return ('RankIC', rank_ic, True)


def rank_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    计算 RankIC (独立函数，用于测试集评估)
    
    Args:
        preds: 预测值
        labels: 真实值
        
    Returns:
        RankIC 值
    """
    return spearman_corr(preds, labels)


def calculate_daily_ic(
    preds: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    use_rank: bool = True
) -> Tuple[List[float], float, float, float]:
    """
    计算每日 IC 并汇总
    
    Args:
        preds: 预测值
        labels: 真实值
        dates: 日期数组
        use_rank: 是否使用 RankIC (默认 True)
        
    Returns:
        (daily_ics, ic_mean, ic_std, icir)
        - daily_ics: 每日 IC 列表
        - ic_mean: IC 均值
        - ic_std: IC 标准差
        - icir: IC 信息比率 (IC均值 / IC标准差)
    """
    # 获取唯一日期
    unique_dates = np.unique(dates)
    
    daily_ics = []
    
    for date in unique_dates:
        mask = dates == date
        
        if mask.sum() < 10:  # 样本太少，跳过
            continue
        
        day_preds = preds[mask]
        day_labels = labels[mask]
        
        if use_rank:
            ic = spearman_corr(day_preds, day_labels)
        else:
            ic = pearson_corr(day_preds, day_labels)
        
        if not np.isnan(ic):
            daily_ics.append(ic)
    
    if len(daily_ics) == 0:
        return [], 0.0, 0.0, 0.0
    
    daily_ics = np.array(daily_ics)
    ic_mean = float(daily_ics.mean())
    ic_std = float(daily_ics.std())
    
    # ICIR: IC 信息比率
    # > 0.5 表示模型发挥稳定
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    
    return daily_ics.tolist(), ic_mean, ic_std, icir


def evaluate_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    dates: Optional[np.ndarray] = None
) -> dict:
    """
    综合评估预测结果
    
    Args:
        preds: 预测值
        labels: 真实值
        dates: 日期数组 (可选，用于计算每日指标)
        
    Returns:
        评估指标字典
    """
    results = {}
    
    # 整体 IC
    results["IC"] = pearson_corr(preds, labels)
    results["RankIC"] = spearman_corr(preds, labels)
    
    # RMSE (参考)
    mask = ~(np.isnan(preds) | np.isnan(labels))
    if mask.sum() > 0:
        rmse = np.sqrt(((preds[mask] - labels[mask]) ** 2).mean())
        results["RMSE"] = float(rmse)
    
    # 每日指标
    if dates is not None:
        daily_ics, ic_mean, ic_std, icir = calculate_daily_ic(
            preds, labels, dates, use_rank=True
        )
        results["DailyRankIC_Mean"] = ic_mean
        results["DailyRankIC_Std"] = ic_std
        results["ICIR"] = icir
        results["DailyIC_Count"] = len(daily_ics)
    
    return results
