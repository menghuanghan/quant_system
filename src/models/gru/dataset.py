"""
GRU 专属数据引擎（dataset.py）

核心职责:
1. GRUTimeSeriesSplitter — 多目标时序切分器
   - 三模式合一生成器 (rolling / expanding / single_full)
   - 自动解析 target_cols 中的最大天数 → gap_days，防止标签泄露
2. GRUTensorDataset — PyTorch 3D 张量滑窗构造器
   - 校验每个 index 前方是否有连续 seq_len-1 天的同股票历史
   - 返回 (X_tensor: (seq_len, num_features), Y_tensor: (num_targets,))

改造说明（2026.02）:
- 适配多目标标签向量
- 三模式时序切分
- GPU 预加载 + 零拷贝切片
"""

import re
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dateutil.relativedelta import relativedelta

from ..config import GRUDataConfig, GRUSplitConfig

logger = logging.getLogger(__name__)


# ============================================================================
# FoldInfo 数据类
# ============================================================================

@dataclass
class GRUFoldInfo:
    """单个 Fold 的信息"""
    fold_idx: int
    train_start: datetime
    train_end: datetime
    valid_start: datetime
    valid_end: datetime
    train_indices: np.ndarray   # 全局行索引
    valid_indices: np.ndarray
    gap_days: int


# ============================================================================
# GRUTimeSeriesSplitter — 多目标时序切分器
# ============================================================================

class GRUTimeSeriesSplitter:
    """
    多目标时序切分器

    三模式合一的生成器:
    - Rolling: Train 窗口固定长度，起终点同时向未来滑动 step 个月
    - Expanding: Train 起点锚定 data_start_date，仅终点向未来延伸
    - Single_Full: 只产出一个 Fold，Train 涵盖 [data_start_date, data_end_date - 1月]，
                   仅截取最后 1 个月做早停

    标签泄露防护:
    - 自动从 target_cols 解析出最大天数 → gap_days
    - Valid 起始 = Train 结束 + gap_days 个交易日

    日期边界说明:
    - min_date / max_date: 由 config.data_start_date / data_end_date 决定的 **逻辑** 边界
    - 数据物理范围可能更宽（含预热窗口），但切分只在逻辑范围内进行
    - 与 LGB TimeSeriesSplitter 保持一致的边界语义

    Args:
        df: 完整 DataFrame（需含 trade_date 列，已按 ts_code+trade_date 排序）
        target_cols: 多目标标签列名列表（如 ['rank_ret_5d', 'excess_ret_10d', 'sharpe_20d']）
        date_col: 日期列名
        config: 切分配置（含 data_start_date / data_end_date）
        seq_len: 序列窗口长度，切分时额外往前延长作为预热窗口
        data_start_date: 覆盖 config 的逻辑起始日期
        data_end_date: 覆盖 config 的逻辑结束日期
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        date_col: str = "trade_date",
        config: Optional[GRUSplitConfig] = None,
        seq_len: int = 20,
        data_start_date: Optional[str] = None,
        data_end_date: Optional[str] = None,
    ):
        self.df = df
        self.target_cols = target_cols
        self.date_col = date_col
        self.config = config or GRUSplitConfig()
        self.seq_len = seq_len

        # 解析 gap_days
        self.gap_days = self._parse_gap_days(target_cols)

        # 确保日期为 datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            self.df = df.copy()
            self.df[date_col] = pd.to_datetime(df[date_col])

        # 交易日历（物理范围内全部交易日）
        self.trade_dates = sorted(self.df[date_col].dropna().unique())

        # 逻辑日期边界（参考 LGB TimeSeriesSplitter）
        self.min_date = pd.Timestamp(
            data_start_date or self.config.data_start_date
        )
        self.max_date = pd.Timestamp(
            data_end_date or self.config.data_end_date
        )

        # 计算逻辑范围内的总月数（用于日志）
        total_months = (
            (self.max_date.year - self.min_date.year) * 12
            + (self.max_date.month - self.min_date.month) + 1
        )

        logger.info(
            f"GRUTimeSeriesSplitter: targets={target_cols}, gap_days={self.gap_days}, "
            f"trade_days={len(self.trade_dates)}, seq_len={seq_len}, "
            f"logical_range=[{self.min_date.date()}, {self.max_date.date()}]"
        )

    @staticmethod
    def _parse_gap_days(target_cols: List[str]) -> int:
        """
        遍历 target_cols，用正则提取所有天数，取最大值
        例如 ['rank_ret_5d', 'excess_ret_10d', 'sharpe_20d'] → max_gap=20
        """
        days_found = []
        for col in target_cols:
            match = re.search(r'(\d+)d?$', col, re.IGNORECASE)
            if match:
                days_found.append(int(match.group(1)))
        if not days_found:
            logger.warning(f"无法从 {target_cols} 解析天数，默认 gap_days=5")
            return 5
        max_gap = max(days_found)
        logger.info(f"解析 gap_days: 检测到天数 {days_found}, max_gap={max_gap}")
        return max_gap

    def _get_date_after_gap(self, date: pd.Timestamp, gap: int) -> pd.Timestamp:
        """获取 date 之后第 gap 个交易日"""
        try:
            idx = self.trade_dates.index(date)
        except ValueError:
            for i, d in enumerate(self.trade_dates):
                if d >= date:
                    idx = i
                    break
            else:
                idx = len(self.trade_dates) - 1
        target_idx = min(idx + gap, len(self.trade_dates) - 1)
        return self.trade_dates[target_idx]

    def _get_indices_by_date(self, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        """获取 [start, end] 日期范围内的行索引（同时要把预热窗口往前扩展 seq_len 个交易日）"""
        dates = self.df[self.date_col]
        mask = (dates >= start) & (dates <= end)
        return np.where(mask)[0]

    def _get_warmup_indices(self, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        """
        获取带预热窗口的行索引:
        实际 start 往前推 seq_len 个交易日，以便构建完整的滑窗序列
        """
        try:
            idx = self.trade_dates.index(start)
        except ValueError:
            for i, d in enumerate(self.trade_dates):
                if d >= start:
                    idx = i
                    break
            else:
                idx = 0
        warmup_idx = max(0, idx - self.seq_len)
        warmup_start = self.trade_dates[warmup_idx]

        dates = self.df[self.date_col]
        mask = (dates >= warmup_start) & (dates <= end)
        return np.where(mask)[0]

    def split(
        self,
        mode: Optional[str] = None,
        train_window_months: Optional[int] = None,
        valid_window_months: Optional[int] = None,
        step_months: Optional[int] = None,
    ) -> Generator[GRUFoldInfo, None, None]:
        """生成 Fold 迭代器"""
        mode = mode or self.config.mode
        tw = train_window_months or self.config.train_window_months
        vw = valid_window_months or self.config.valid_window_months
        step = step_months or self.config.step_months

        if mode == "rolling":
            yield from self._split_rolling(tw, vw, step)
        elif mode == "expanding":
            yield from self._split_expanding(tw, vw, step)
        elif mode == "single_full":
            yield from self._split_single_full()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _split_rolling(self, tw: int, vw: int, step: int) -> Generator[GRUFoldInfo, None, None]:
        """
        滚动窗口切分

        Train 起终点同时向未来滑动 step 个月，Valid 紧随其后
        使用 self.min_date / self.max_date 逻辑边界（而非数据物理边界）
        """
        fold_idx = 0
        train_start = self.min_date
        min_valid_days = int(vw * 30 * 0.5)
        # 使用逻辑边界与实际数据中较小的那个
        actual_max = pd.Timestamp(self.trade_dates[-1]) if self.trade_dates else self.max_date
        effective_max = min(self.max_date, actual_max)

        while True:
            train_end = train_start + relativedelta(months=tw) - timedelta(days=1)
            if pd.Timestamp(train_end) > effective_max:
                break

            valid_start_base = pd.Timestamp(train_end) + timedelta(days=1)
            valid_start = self._get_date_after_gap(valid_start_base, self.gap_days)
            valid_end = pd.Timestamp(valid_start) + relativedelta(months=vw) - timedelta(days=1)

            if valid_start > effective_max:
                break
            valid_end = min(pd.Timestamp(valid_end), effective_max)

            actual_valid_days = (valid_end - valid_start).days
            if actual_valid_days < min_valid_days:
                break

            # 带预热窗口的 train_indices
            train_indices = self._get_warmup_indices(train_start, pd.Timestamp(train_end))
            valid_indices = self._get_warmup_indices(valid_start, valid_end)

            if len(train_indices) == 0 or len(valid_indices) == 0:
                train_start = train_start + relativedelta(months=step)
                continue

            yield GRUFoldInfo(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=pd.Timestamp(train_end),
                valid_start=valid_start,
                valid_end=valid_end,
                train_indices=train_indices,
                valid_indices=valid_indices,
                gap_days=self.gap_days,
            )

            logger.info(
                f"[Rolling] Fold {fold_idx}: "
                f"train=[{train_start.date()}, {pd.Timestamp(train_end).date()}] "
                f"({len(train_indices)} rows), "
                f"valid=[{valid_start.date()}, {valid_end.date()}] "
                f"({len(valid_indices)} rows)"
            )

            fold_idx += 1
            train_start = train_start + relativedelta(months=step)

    def _split_expanding(self, tw: int, vw: int, step: int) -> Generator[GRUFoldInfo, None, None]:
        """
        扩展窗口切分

        Train 起点锚定 self.min_date，仅终点向未来延伸
        使用逻辑日期边界
        """
        fold_idx = 0
        train_start = self.min_date
        actual_max = pd.Timestamp(self.trade_dates[-1]) if self.trade_dates else self.max_date
        effective_max = min(self.max_date, actual_max)
        train_end = train_start + relativedelta(months=tw) - timedelta(days=1)

        while True:
            # 安全检查：train_end 超过逻辑边界则退出
            if pd.Timestamp(train_end) > effective_max:
                break

            valid_start_base = pd.Timestamp(train_end) + timedelta(days=1)
            valid_start = self._get_date_after_gap(valid_start_base, self.gap_days)
            valid_end = pd.Timestamp(valid_start) + relativedelta(months=vw) - timedelta(days=1)

            if valid_start > effective_max:
                break
            valid_end = min(pd.Timestamp(valid_end), effective_max)

            train_indices = self._get_warmup_indices(train_start, pd.Timestamp(train_end))
            valid_indices = self._get_warmup_indices(valid_start, valid_end)

            if len(train_indices) == 0 or len(valid_indices) == 0:
                train_end = pd.Timestamp(train_end) + relativedelta(months=step)
                continue

            yield GRUFoldInfo(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=pd.Timestamp(train_end),
                valid_start=valid_start,
                valid_end=valid_end,
                train_indices=train_indices,
                valid_indices=valid_indices,
                gap_days=self.gap_days,
            )

            logger.info(
                f"[Expanding] Fold {fold_idx}: "
                f"train=[{train_start.date()}, {pd.Timestamp(train_end).date()}] "
                f"({len(train_indices)} rows), "
                f"valid=[{valid_start.date()}, {valid_end.date()}] "
                f"({len(valid_indices)} rows)"
            )

            fold_idx += 1
            train_end = pd.Timestamp(train_end) + relativedelta(months=step)

    def _split_single_full(self) -> Generator[GRUFoldInfo, None, None]:
        """
        单次划分（全量重训）

        只产出一个 Fold:
        - 逻辑范围: [min_date, max_date]（由 data_start_date / data_end_date 决定）
        - 总月数 N = (max_date - min_date) 的月数 + 1
        - Train: 前 N-1 个月
        - Valid: 最后 1 个月（用于早停监控）
        - Train 结束日期往回推 gap_days 个交易日，防止标签泄露

        例如 2021-01-01 ~ 2025-12-31 → 共 60 月 → Train 59 月 + Valid 1 月
        """
        # 使用逻辑边界（而非数据物理边界）
        actual_max = pd.Timestamp(self.trade_dates[-1]) if self.trade_dates else self.max_date
        effective_max = min(self.max_date, actual_max)

        valid_end = effective_max
        valid_start = effective_max - relativedelta(months=1)

        train_start = self.min_date

        # Train 结束日期: valid_start 往回推 gap_days 个交易日
        train_end_candidates = [d for d in self.trade_dates if d < valid_start]
        if len(train_end_candidates) <= self.gap_days:
            raise ValueError("Not enough data for single_full split with gap protection")
        train_end = train_end_candidates[-(self.gap_days + 1)]

        train_indices = self._get_warmup_indices(train_start, pd.Timestamp(train_end))
        valid_indices = self._get_warmup_indices(valid_start, valid_end)

        total_months = (
            (effective_max.year - train_start.year) * 12
            + (effective_max.month - train_start.month) + 1
        )
        train_months = (
            (pd.Timestamp(train_end).year - train_start.year) * 12
            + (pd.Timestamp(train_end).month - train_start.month) + 1
        )

        yield GRUFoldInfo(
            fold_idx=0,
            train_start=train_start,
            train_end=pd.Timestamp(train_end),
            valid_start=valid_start,
            valid_end=valid_end,
            train_indices=train_indices,
            valid_indices=valid_indices,
            gap_days=self.gap_days,
        )

        logger.info(
            f"[Single_Full] Fold 0: total={total_months}月, "
            f"train={train_months}月, valid=1月, gap={self.gap_days}天"
        )
        logger.info(
            f"  Train: [{train_start.date()}, {pd.Timestamp(train_end).date()}] "
            f"({len(train_indices)} rows)"
        )
        logger.info(
            f"  Valid: [{valid_start.date()}, {valid_end.date()}] "
            f"({len(valid_indices)} rows)"
        )


# ============================================================================
# GRUTensorDataset — PyTorch 3D 张量滑窗构造器
# ============================================================================

class GRUTensorDataset(Dataset):
    """
    PyTorch 3D 张量滑窗构造器（零拷贝版本）

    职责:
    - 接收 Trainer 预计算的共享 Feature/Label 张量（不复制数据）
    - 校验每个 index 对应的股票在它前面是否有连续 seq_len-1 天历史
    - __getitem__ 返回 (X: (seq_len, num_features), Y: (num_targets,))
    - 多 Fold / 多种子间共享同一份张量，避免重复分配内存

    Args:
        features: (N_total, num_features) 共享特征张量（由 Trainer 预计算）
        labels: (N_total, num_targets) 共享标签张量
        dates: (N_total,) 日期元数据 (numpy datetime64)
        codes: (N_total,) 股票代码元数据 (numpy str)
        indices: 上游 Splitter 提供的候选行索引（带预热窗口）
        seq_len: 回看窗口长度
        target_cols: 多目标标签列名列表
        date_range: (start, end) 只有 T 在此范围内的才出现在可用列表
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        dates: np.ndarray,
        codes: np.ndarray,
        indices: np.ndarray,
        seq_len: int,
        target_cols: List[str],
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ):
        self.seq_len = seq_len
        self.target_cols = target_cols
        self.device = str(features.device)

        # ---- 共享引用（零拷贝，不复制数据） ----
        self.features = features
        self.labels = labels
        self.dates = dates
        self.codes = codes

        # ---- 构建合法索引 ----
        self.valid_indices = self._build_valid_indices(indices, date_range)

        logger.info(
            f"GRUTensorDataset: "
            f"样本={len(self.valid_indices):,}, "
            f"特征={features.shape[1]}, "
            f"目标={labels.shape[1]}, "
            f"seq_len={seq_len}, "
            f"device={self.device}"
        )

    def _build_valid_indices(
        self,
        indices: np.ndarray,
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
    ) -> np.ndarray:
        """
        构建合法样本索引

        对于每个 index T:
        1. T 必须在 indices 中
        2. [T - seq_len + 1, T] 范围内的所有行必须属于同一只股票
        3. 如果指定了 date_range，T 的日期必须在 [start, end] 内
        """
        indices_set = set(indices.tolist())
        valid = []

        for T in indices:
            T = int(T)
            # 1) 检查窗口是否越界
            start_idx = T - self.seq_len + 1
            if start_idx < 0:
                continue

            # 2) 检查窗口内是否为同一只股票
            code_T = self.codes[T]
            same_stock = True
            for k in range(start_idx, T):
                if self.codes[k] != code_T:
                    same_stock = False
                    break
            if not same_stock:
                continue

            # 3) 检查日期范围
            if date_range is not None:
                dt = pd.Timestamp(self.dates[T])
                if dt < date_range[0] or dt > date_range[1]:
                    continue

            valid.append(T)

        return np.array(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        T = int(self.valid_indices[idx])
        start = T - self.seq_len + 1

        # GPU 上直接切片，零拷贝
        X = self.features[start: T + 1]          # (seq_len, num_features)
        Y = self.labels[T]                        # (num_targets,)

        return X, Y

    # ---- 辅助方法 ----

    def get_meta(self, idx: int) -> Tuple[str, str]:
        """返回第 idx 个样本的 (trade_date_str, ts_code)"""
        T = int(self.valid_indices[idx])
        dt = self.dates[T]
        if hasattr(dt, 'strftime'):
            dt_str = dt.strftime('%Y-%m-%d')
        else:
            dt_str = str(dt)[:10]
        code = str(self.codes[T])
        return dt_str, code

    def get_all_dates(self) -> np.ndarray:
        """返回所有合法样本的日期字符串"""
        result = []
        for T in self.valid_indices:
            T = int(T)
            dt = self.dates[T]
            if hasattr(dt, 'strftime'):
                result.append(dt.strftime('%Y-%m-%d'))
            else:
                result.append(str(dt)[:10])
        return np.array(result)

    def get_all_codes(self) -> np.ndarray:
        """返回所有合法样本的股票代码"""
        return np.array([str(self.codes[int(T)]) for T in self.valid_indices])


# ============================================================================
# 工具函数
# ============================================================================

def create_dataloader(
    dataset: GRUTensorDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    创建 DataLoader

    注意: GPU 数据集 num_workers 必须为 0
    """
    is_gpu = hasattr(dataset, 'device') and 'cuda' in str(dataset.device)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 if is_gpu else num_workers,
        pin_memory=not is_gpu,
        drop_last=shuffle,
    )
