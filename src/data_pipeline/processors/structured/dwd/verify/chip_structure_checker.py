"""
筹码结构宽表 (dwd_chip_structure) 检查器

核心目标：确保比例不溢出，数值符合物理常识

检查项：
- top10_hold_ratio: 范围检查 [0, 100]
- holder_num: 整数检查，非零检查
- chip_report_date: PIT检查(必须早于或等于trade_date)
"""

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from .base import BaseChecker, CheckResult, CheckSeverity
from .common_checks import CommonChecker

logger = logging.getLogger(__name__)


class ChipStructureChecker(BaseChecker):
    """筹码结构宽表检查器"""
    
    @property
    def table_name(self) -> str:
        return "dwd_chip_structure"
    
    def run_specific_checks(self):
        """运行筹码结构表特定检查"""
        
        # ======================================================================
        # 第一部分：通用核验
        # ======================================================================
        common = CommonChecker(self)
        common.run_all_common_checks(
            pk_columns=["trade_date", "ts_code"],
            critical_columns=["trade_date", "ts_code"],
            expected_types={
                "amount_cols": ["top10_hold_amount"] if "top10_hold_amount" in self.df.columns else [],
            },
        )
        
        # ======================================================================
        # 第二部分：表特定检查
        # ======================================================================
        
        # 1. 十大股东占比范围检查（重点）
        self._check_hold_ratio_range()
        
        # 2. 股东户数检查
        self._check_holder_num()
        
        # 3. 筹码报告日期PIT检查
        self._check_chip_report_pit()
    
    def _check_hold_ratio_range(self):
        """十大股东占比范围检查"""
        
        if "top10_hold_ratio" not in self.df.columns:
            self.add_result(
                check_name="十大股东占比范围检查",
                passed=True,
                message="top10_hold_ratio 列不存在，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        series = self.df["top10_hold_ratio"].dropna()
        
        # 范围检查 [0, 100]
        below_zero = (series < 0).sum()
        above_100 = (series > 100).sum()
        
        passed = below_zero == 0 and above_100 == 0
        
        self.add_result(
            check_name="十大股东占比范围检查 (0-100%)",
            passed=passed,
            message=f"<0: {below_zero}行, >100: {above_100}行",
            severity=CheckSeverity.PASS if passed else CheckSeverity.ERROR,
            details={
                "below_zero": int(below_zero),
                "above_100": int(above_100),
            },
        )
        
        # 分布统计
        mean_val = series.mean()
        median_val = series.median()
        
        self.add_result(
            check_name="十大股东占比分布",
            passed=True,
            message=f"均值={mean_val:.2f}%, 中位数={median_val:.2f}%",
            severity=CheckSeverity.INFO,
            metric_name="top10_hold_ratio_mean",
            metric_value=mean_val,
        )
    
    def _check_holder_num(self):
        """股东户数检查"""
        
        if "holder_num" not in self.df.columns:
            self.add_result(
                check_name="股东户数检查",
                passed=True,
                message="holder_num 列不存在，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        series = self.df["holder_num"].dropna()
        
        # 整数检查 (允许float但值应为整数)
        non_integer = (series != series.astype(int)).sum()
        
        self.add_result(
            check_name="股东户数整数检查",
            passed=non_integer == 0,
            message=f"非整数值: {non_integer}行",
            severity=CheckSeverity.PASS if non_integer == 0 else CheckSeverity.WARNING,
            metric_name="holder_num_non_integer",
            metric_value=non_integer,
        )
        
        # 非零检查（除冷启动期外）
        # 只检查2021年以后的数据
        if "trade_date" in self.df.columns:
            df_train = self.df[self.df["trade_date"] >= "2021-01-01"]
            if "holder_num" in df_train.columns:
                zero_count = (df_train["holder_num"] == 0).sum()
                zero_rate = zero_count / len(df_train) if len(df_train) > 0 else 0
                
                # 允许少量为0（新股等）
                passed = zero_rate < 0.01
                
                self.add_result(
                    check_name="股东户数非零检查(2021年后)",
                    passed=passed,
                    message=f"零值比例: {zero_rate:.4%}",
                    severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
                    metric_name="holder_num_zero_rate",
                    metric_value=zero_rate,
                )
    
    def _check_chip_report_pit(self):
        """筹码报告日期PIT检查"""
        
        if "chip_report_date" not in self.df.columns:
            self.add_result(
                check_name="筹码报告日期PIT检查",
                passed=True,
                message="chip_report_date 列不存在，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        if "trade_date" not in self.df.columns:
            return
        
        # chip_report_date 必须早于或等于 trade_date
        df_valid = self.df[
            self.df["chip_report_date"].notna() & 
            self.df["trade_date"].notna()
        ].copy()
        
        if len(df_valid) == 0:
            self.add_result(
                check_name="筹码报告日期PIT检查",
                passed=True,
                message="无有效数据",
                severity=CheckSeverity.INFO,
            )
            return
        
        df_valid["chip_report_date_dt"] = pd.to_datetime(df_valid["chip_report_date"])
        df_valid["trade_date_dt"] = pd.to_datetime(df_valid["trade_date"])
        
        # 未来数据（穿越）
        future_leak = (df_valid["chip_report_date_dt"] > df_valid["trade_date_dt"]).sum()
        
        self.add_result(
            check_name="筹码报告日期PIT检查 (无未来函数)",
            passed=future_leak == 0,
            message=f"未来数据行数: {future_leak}",
            severity=CheckSeverity.PASS if future_leak == 0 else CheckSeverity.CRITICAL,
            metric_name="chip_future_leak_count",
            metric_value=future_leak,
        )
