"""
时序切分引擎（TimeSeriesSplitter）

核心职责：处理时间轴，生成无未来函数的 Train/Valid 索引
支持三种训练模式：
- Rolling（滚动窗口）：训练窗口固定长度，窗口整体向前滑动
- Expanding（扩展窗口）：训练窗口起点固定，终点不断延伸
- Single_Full（单次划分）：全量数据训练，仅保留最后一段做早停
"""

import re
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ..config import SplitConfig, SplitMode

logger = logging.getLogger(__name__)


@dataclass
class FoldInfo:
    """单个 Fold 的信息"""
    fold_idx: int                    # Fold 序号（从0开始）
    train_start: datetime            # 训练起始日期
    train_end: datetime              # 训练结束日期
    valid_start: datetime            # 验证起始日期
    valid_end: datetime              # 验证结束日期
    train_indices: np.ndarray        # 训练样本索引
    valid_indices: np.ndarray        # 验证样本索引
    gap_days: int                    # Train/Valid 之间的 gap（交易日）


class TimeSeriesSplitter:
    """
    时序切分引擎
    
    通过生成器模式兼容三种训练模式：rolling, expanding, single_full
    内置标签泄露防护：自动从 target_col 解析 gap_days
    
    Example:
        >>> splitter = TimeSeriesSplitter(df, target_col="rank_ret_5d")
        >>> for fold in splitter.split(mode=SplitMode.ROLLING):
        ...     X_train = df.iloc[fold.train_indices][feature_cols]
        ...     y_train = df.iloc[fold.train_indices][target_col]
        ...     ...
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str = "trade_date",
        config: Optional[SplitConfig] = None,
    ):
        """
        初始化时序切分器
        
        Args:
            df: 输入 DataFrame（需包含 date_col 列）
            target_col: 目标标签列名（如 "rank_ret_5d", "excess_ret_10d"）
            date_col: 日期列名
            config: 切分配置
        """
        self.df = df
        self.target_col = target_col
        self.date_col = date_col
        self.config = config or SplitConfig()
        
        # 解析 gap_days
        self.gap_days = self._parse_gap_days(target_col)
        
        # 确保日期列为 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            self.df = df.copy()
            self.df[date_col] = pd.to_datetime(df[date_col])
        
        # 获取交易日历表（从数据中提取唯一日期）
        self.trade_dates = sorted(self.df[date_col].dropna().unique())
        self.trade_dates_set = set(self.trade_dates)
        
        # 日期范围
        self.min_date = pd.Timestamp(self.config.data_start_date)
        self.max_date = pd.Timestamp(self.config.data_end_date)
        
        logger.info(
            f"TimeSeriesSplitter initialized: target={target_col}, gap_days={self.gap_days}, "
            f"date_range=[{self.min_date.date()}, {self.max_date.date()}], "
            f"total_trade_days={len(self.trade_dates)}"
        )
    
    def _parse_gap_days(self, target_col: str) -> int:
        """
        从标签列名中解析 gap_days
        
        规则：提取标签名末尾的数字（如 rank_ret_5d -> 5, excess_ret_10d -> 10）
        
        Args:
            target_col: 标签列名
            
        Returns:
            gap_days: 需要跳过的交易日天数
        """
        # 正则匹配列名末尾的数字（如 _5d, _10d, _20d）
        match = re.search(r'_(\d+)d?$', target_col, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # 尝试匹配其他模式（如 5d, 10d）
        match = re.search(r'(\d+)d?$', target_col, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # 默认值
        logger.warning(f"Cannot parse gap_days from target_col='{target_col}', using default=5")
        return 5
    
    def _get_date_after_gap(self, date: pd.Timestamp, gap_days: int) -> pd.Timestamp:
        """
        获取指定日期之后 gap_days 个交易日的日期
        
        Args:
            date: 起始日期
            gap_days: 要跳过的交易日数
            
        Returns:
            target_date: gap_days 个交易日后的日期
        """
        # 找到 date 在交易日历中的位置
        try:
            idx = self.trade_dates.index(date)
        except ValueError:
            # date 不在交易日历中，找最近的
            for i, d in enumerate(self.trade_dates):
                if d >= date:
                    idx = i
                    break
            else:
                idx = len(self.trade_dates) - 1
        
        # 向前移动 gap_days
        target_idx = idx + gap_days
        if target_idx >= len(self.trade_dates):
            return self.trade_dates[-1]
        
        return self.trade_dates[target_idx]
    
    def _get_mask_by_date_range(
        self, 
        start: pd.Timestamp, 
        end: pd.Timestamp
    ) -> np.ndarray:
        """
        获取指定日期范围内的样本索引
        
        Args:
            start: 起始日期（含）
            end: 结束日期（含）
            
        Returns:
            indices: 样本索引数组
        """
        dates = self.df[self.date_col]
        mask = (dates >= start) & (dates <= end)
        return np.where(mask)[0]
    
    def split(
        self,
        mode: Optional[SplitMode] = None,
        train_window_months: Optional[int] = None,
        valid_window_months: Optional[int] = None,
        step_months: Optional[int] = None,
    ) -> Generator[FoldInfo, None, None]:
        """
        生成 Train/Valid 切分的 Fold 迭代器
        
        Args:
            mode: 切分模式（覆盖 config）
            train_window_months: 训练窗口月数（覆盖 config）
            valid_window_months: 验证窗口月数（覆盖 config）
            step_months: 滑动步长月数（覆盖 config）
            
        Yields:
            FoldInfo: 每个 Fold 的详细信息
        """
        mode = mode or self.config.mode
        train_months = train_window_months or self.config.train_window_months
        valid_months = valid_window_months or self.config.valid_window_months
        step = step_months or self.config.step_months
        
        if mode == SplitMode.ROLLING:
            yield from self._split_rolling(train_months, valid_months, step)
        elif mode == SplitMode.EXPANDING:
            yield from self._split_expanding(train_months, valid_months, step)
        elif mode == SplitMode.SINGLE_FULL:
            yield from self._split_single_full()
        else:
            raise ValueError(f"Unknown split mode: {mode}")
    
    def _split_rolling(
        self,
        train_months: int,
        valid_months: int,
        step_months: int,
    ) -> Generator[FoldInfo, None, None]:
        """
        滚动窗口切分
        
        训练窗口长度固定，整体向前滑动
        例：train=24月, valid=3月, step=3月
            Fold0: train=[2021-01, 2022-12], valid=[2023-01+gap, 2023-03]
            Fold1: train=[2021-04, 2023-03], valid=[2023-04+gap, 2023-06]
        """
        fold_idx = 0
        train_start = self.min_date
        
        # 使用实际数据的最大日期作为硬边界
        actual_max_date = pd.Timestamp(self.trade_dates[-1]) if self.trade_dates else self.max_date
        
        # 计算最小验证窗口天数（验证窗口至少要有一半长度才有效）
        min_valid_days = int(valid_months * 30 * 0.5)  # 至少一半窗口长度
        
        while True:
            # 计算训练结束日期（不截断，保持完整窗口长度）
            train_end_expected = train_start + relativedelta(months=train_months) - timedelta(days=1)
            
            # 【关键修复】训练窗口必须保持完整长度
            # 如果训练结束日期超过实际数据范围，说明无法形成完整训练窗口，停止
            if pd.Timestamp(train_end_expected) > actual_max_date:
                logger.info(
                    f"[Rolling] Stopping: train_end ({train_end_expected.date()}) exceeds "
                    f"data max_date ({actual_max_date.date()}), cannot form complete {train_months}-month window"
                )
                break
            
            train_end = pd.Timestamp(train_end_expected)
            
            # 如果训练起始日期已经超过数据范围，停止
            if train_start > actual_max_date:
                break
            
            # 计算验证起始日期（跳过 gap_days）
            valid_start_base = train_end + timedelta(days=1)
            valid_start = self._get_date_after_gap(pd.Timestamp(valid_start_base), self.gap_days)
            
            # 计算验证结束日期
            valid_end_expected = pd.Timestamp(valid_start) + relativedelta(months=valid_months) - timedelta(days=1)
            
            # 验证起始日期超过数据范围，停止
            if valid_start > actual_max_date:
                logger.info(
                    f"[Rolling] Stopping: valid_start ({valid_start.date()}) exceeds "
                    f"data max_date ({actual_max_date.date()})"
                )
                break
            
            # 截断验证结束日期到数据范围内
            valid_end = min(pd.Timestamp(valid_end_expected), actual_max_date)
            
            # 【关键修复】检查验证窗口的实际时间跨度
            actual_valid_days = (valid_end - valid_start).days
            if actual_valid_days < min_valid_days:
                logger.info(
                    f"[Rolling] Stopping: valid window too short ({actual_valid_days} days < {min_valid_days} days minimum)"
                )
                break
            
            # 获取索引
            train_indices = self._get_mask_by_date_range(train_start, train_end)
            valid_indices = self._get_mask_by_date_range(valid_start, valid_end)
            
            # 跳过空 Fold
            if len(train_indices) == 0 or len(valid_indices) == 0:
                train_start = train_start + relativedelta(months=step_months)
                if train_start > actual_max_date:
                    break
                continue
            
            yield FoldInfo(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                train_indices=train_indices,
                valid_indices=valid_indices,
                gap_days=self.gap_days,
            )
            
            logger.info(
                f"[Rolling] Fold {fold_idx}: "
                f"train=[{train_start.date()}, {train_end.date()}]({len(train_indices)} samples), "
                f"valid=[{valid_start.date()}, {valid_end.date()}]({len(valid_indices)} samples)"
            )
            
            fold_idx += 1
            train_start = train_start + relativedelta(months=step_months)
    
    def _split_expanding(
        self,
        initial_train_months: int,
        valid_months: int,
        step_months: int,
    ) -> Generator[FoldInfo, None, None]:
        """
        扩展窗口切分
        
        训练窗口起点固定（锚定全局最小日期），终点不断向后延伸
        例：train起点=2021-01, valid=3月, step=3月
            Fold0: train=[2021-01, 2022-12], valid=[2023-01+gap, 2023-03]
            Fold1: train=[2021-01, 2023-03], valid=[2023-04+gap, 2023-06]
        """
        fold_idx = 0
        train_start = self.min_date  # 起点固定
        train_end = train_start + relativedelta(months=initial_train_months) - timedelta(days=1)
        
        while True:
            # 计算验证起始日期（跳过 gap_days）
            valid_start_base = train_end + timedelta(days=1)
            valid_start = self._get_date_after_gap(pd.Timestamp(valid_start_base), self.gap_days)
            
            # 计算验证结束日期
            valid_end = pd.Timestamp(valid_start) + relativedelta(months=valid_months) - timedelta(days=1)
            
            # 确保不超过数据范围
            if valid_start > self.max_date:
                break
            
            valid_end = min(pd.Timestamp(valid_end), self.max_date)
            
            # 获取索引
            train_indices = self._get_mask_by_date_range(train_start, train_end)
            valid_indices = self._get_mask_by_date_range(valid_start, valid_end)
            
            # 跳过空 Fold
            if len(train_indices) == 0 or len(valid_indices) == 0:
                train_end = train_end + relativedelta(months=step_months)
                continue
            
            yield FoldInfo(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                train_indices=train_indices,
                valid_indices=valid_indices,
                gap_days=self.gap_days,
            )
            
            logger.info(
                f"[Expanding] Fold {fold_idx}: "
                f"train=[{train_start.date()}, {train_end.date()}]({len(train_indices)} samples), "
                f"valid=[{valid_start.date()}, {valid_end.date()}]({len(valid_indices)} samples)"
            )
            
            fold_idx += 1
            train_end = train_end + relativedelta(months=step_months)
    
    def _split_single_full(self) -> Generator[FoldInfo, None, None]:
        """
        单次划分（全量重训）
        
        只产出一个 Fold，Train 涵盖所有历史数据，仅截取最后 1 个月做早停监控
        适用于生产环境全量重训
        """
        # 验证窗口：最后 1 个月
        valid_end = self.max_date
        valid_start = valid_end - relativedelta(months=1)
        
        # 训练窗口：从起点到验证前（减去 gap_days）
        train_start = self.min_date
        # 往回推 gap_days 天，确保没有标签泄露
        train_end_candidates = [d for d in self.trade_dates if d < valid_start]
        if len(train_end_candidates) <= self.gap_days:
            raise ValueError("Not enough data for single_full split with gap protection")
        
        train_end_idx = len(train_end_candidates) - self.gap_days - 1
        train_end = train_end_candidates[train_end_idx]
        
        # 获取索引
        train_indices = self._get_mask_by_date_range(train_start, train_end)
        valid_indices = self._get_mask_by_date_range(valid_start, valid_end)
        
        yield FoldInfo(
            fold_idx=0,
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            train_indices=train_indices,
            valid_indices=valid_indices,
            gap_days=self.gap_days,
        )
        
        logger.info(
            f"[Single_Full] Fold 0: "
            f"train=[{train_start.date()}, {train_end.date()}]({len(train_indices)} samples), "
            f"valid=[{valid_start.date()}, {valid_end.date()}]({len(valid_indices)} samples)"
        )
    
    def get_n_splits(self, mode: Optional[SplitMode] = None) -> int:
        """
        获取指定模式下的 Fold 数量
        
        注意：会实际执行一次迭代来计算，如果数据量大建议缓存结果
        """
        mode = mode or self.config.mode
        return sum(1 for _ in self.split(mode=mode))
    
    def __repr__(self) -> str:
        return (
            f"TimeSeriesSplitter(target={self.target_col}, gap_days={self.gap_days}, "
            f"mode={self.config.mode.value})"
        )