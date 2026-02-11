"""
事件信号宽表 (dwd_event_signal) 检查器

核心目标：确保信号稀疏但有效，逻辑正确

检查项：
- days_to_unlock: 有效性检查（-1比例）
- pledge_ratio: 范围检查 [0, 100]
- is_repurchase_ann: 分布检查（是否体现2024-2025回购潮）
"""

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from .base import BaseChecker, CheckResult, CheckSeverity
from .common_checks import CommonChecker

logger = logging.getLogger(__name__)


class EventSignalChecker(BaseChecker):
    """事件信号宽表检查器"""
    
    @property
    def table_name(self) -> str:
        return "dwd_event_signal"
    
    def run_specific_checks(self):
        """运行事件信号表特定检查"""
        
        # ======================================================================
        # 第一部分：通用核验
        # ======================================================================
        common = CommonChecker(self)
        common.run_all_common_checks(
            pk_columns=["trade_date", "ts_code"],
            critical_columns=["trade_date", "ts_code"],
            expected_types={},
        )
        
        # ======================================================================
        # 第二部分：表特定检查
        # ======================================================================
        
        # 1. 解禁天数有效性检查（重点）
        self._check_unlock_days()
        
        # 2. 质押率范围检查
        self._check_pledge_ratio()
        
        # 3. 回购公告分布检查
        self._check_repurchase_distribution()
        
        # 4. 信号稀疏性检查
        self._check_signal_sparsity()
    
    def _check_unlock_days(self):
        """解禁天数有效性检查"""
        
        if "days_to_unlock" not in self.df.columns:
            self.add_result(
                check_name="解禁天数有效性检查",
                passed=True,
                message="days_to_unlock 列不存在，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        series = self.df["days_to_unlock"]
        total = series.notna().sum()
        
        # 统计 -1 的比例（-1 表示无解禁）
        minus_one_count = (series == -1).sum()
        minus_one_rate = minus_one_count / total if total > 0 else 0
        
        # 正数比例（有解禁信号）
        positive_count = (series > 0).sum()
        positive_rate = positive_count / total if total > 0 else 0
        
        # 如果全为 -1，说明数据可能有问题
        all_minus_one = minus_one_rate > 0.99
        
        self.add_result(
            check_name="解禁天数有效性检查",
            passed=not all_minus_one,
            message=f"-1比例: {minus_one_rate:.4%}, 正数比例: {positive_rate:.4%}",
            severity=CheckSeverity.PASS if not all_minus_one else CheckSeverity.ERROR,
            details={
                "minus_one_rate": float(minus_one_rate),
                "positive_rate": float(positive_rate),
            },
            metric_name="days_to_unlock_valid_rate",
            metric_value=positive_rate,
        )
        
        # 检查正数的分布
        positive_values = series[series > 0]
        if len(positive_values) > 0:
            self.add_result(
                check_name="解禁天数分布",
                passed=True,
                message=f"正数均值: {positive_values.mean():.1f}天, "
                        f"中位数: {positive_values.median():.1f}天, "
                        f"最大: {positive_values.max():.0f}天",
                severity=CheckSeverity.INFO,
            )
    
    def _check_pledge_ratio(self):
        """质押率范围检查"""
        
        if "pledge_ratio" not in self.df.columns:
            self.add_result(
                check_name="质押率范围检查",
                passed=True,
                message="pledge_ratio 列不存在，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        series = self.df["pledge_ratio"].dropna()
        
        # 范围检查 [0, 100]
        below_zero = (series < 0).sum()
        above_100 = (series > 100).sum()
        
        passed = below_zero == 0 and above_100 == 0
        
        self.add_result(
            check_name="质押率范围检查 (0-100%)",
            passed=passed,
            message=f"<0: {below_zero}行, >100: {above_100}行",
            severity=CheckSeverity.PASS if passed else CheckSeverity.ERROR,
            details={
                "below_zero": int(below_zero),
                "above_100": int(above_100),
            },
        )
        
        # 分布统计
        if len(series) > 0:
            mean_val = series.mean()
            median_val = series.median()
            
            self.add_result(
                check_name="质押率分布",
                passed=True,
                message=f"均值={mean_val:.2f}%, 中位数={median_val:.2f}%",
                severity=CheckSeverity.INFO,
                metric_name="pledge_ratio_mean",
                metric_value=mean_val,
            )
    
    def _check_repurchase_distribution(self):
        """回购公告分布检查"""
        
        if "is_repurchase_ann" not in self.df.columns:
            self.add_result(
                check_name="回购公告分布检查",
                passed=True,
                message="is_repurchase_ann 列不存在，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        if "trade_date" not in self.df.columns:
            return
        
        # 按年统计回购公告数量
        self.df["_year"] = pd.to_datetime(self.df["trade_date"]).dt.year
        yearly_repurchase = self.df.groupby("_year")["is_repurchase_ann"].sum()
        
        self.add_result(
            check_name="回购公告年度分布",
            passed=True,
            message=f"年度回购数量: {yearly_repurchase.to_dict()}",
            severity=CheckSeverity.INFO,
            details={"yearly_repurchase": yearly_repurchase.to_dict()},
        )
        
        # 检查是否体现2024-2025回购潮趋势
        if 2024 in yearly_repurchase.index and 2021 in yearly_repurchase.index:
            count_2024 = yearly_repurchase.get(2024, 0)
            count_2021 = yearly_repurchase.get(2021, 1)
            
            # 2024年回购应显著多于2021年
            ratio = count_2024 / count_2021 if count_2021 > 0 else 0
            
            self.add_result(
                check_name="回购潮趋势检查 (2024 vs 2021)",
                passed=True,  # 仅为信息
                message=f"2024年: {count_2024}, 2021年: {count_2021}, 比值: {ratio:.2f}",
                severity=CheckSeverity.INFO,
            )
        
        self.df.drop(columns=["_year"], inplace=True)
    
    def _check_signal_sparsity(self):
        """信号稀疏性检查"""
        
        # 检查各类事件信号的非零比例
        signal_cols = [
            "is_repurchase_ann", "is_dividend_ann", "is_earnings_ann",
            "is_merger_ann", "is_restrict_unlock",
        ]
        
        for col in signal_cols:
            if col not in self.df.columns:
                continue
            
            series = self.df[col]
            total = series.notna().sum()
            nonzero = (series == 1).sum()
            rate = nonzero / total if total > 0 else 0
            
            # 事件信号应该是稀疏的（通常<5%）
            is_sparse = rate < 0.1
            
            self.add_result(
                check_name=f"信号稀疏性检查 ({col})",
                passed=is_sparse,
                message=f"信号比例: {rate:.4%} ({nonzero}次)",
                severity=CheckSeverity.PASS if is_sparse else CheckSeverity.WARNING,
                metric_name=f"{col}_signal_rate",
                metric_value=rate,
            )
