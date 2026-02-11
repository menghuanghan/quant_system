"""
通用核验维度检查器

适用于所有DWD表的基础检查：
1. 主键唯一性 (PK Uniqueness)
2. 时间连续性 (Time Continuity)
3. 缺失值概览 (Null Rate)
4. 数据类型 (Data Types)
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Set, Tuple

import pandas as pd
import numpy as np

from .base import BaseChecker, CheckResult, CheckSeverity, TableCheckReport

logger = logging.getLogger(__name__)


class CommonChecker:
    """通用检查器 - 提供适用于所有表的基础检查"""
    
    # 中国股票交易日历（简化版本 - 使用pandas工作日）
    # 实际应用中应使用真实交易日历
    
    def __init__(self, checker: BaseChecker):
        """
        Args:
            checker: 具体表的检查器实例
        """
        self.checker = checker
        self.df = checker.df
    
    def check_pk_uniqueness(
        self,
        pk_columns: List[str],
    ) -> CheckResult:
        """
        检查主键唯一性
        
        Args:
            pk_columns: 主键列名列表，如 ['trade_date', 'ts_code']
        """
        # 检查列是否存在
        missing_cols = [c for c in pk_columns if c not in self.df.columns]
        if missing_cols:
            self.checker.add_result(
                check_name="主键唯一性检查",
                passed=False,
                message=f"主键列不存在: {missing_cols}",
                severity=CheckSeverity.CRITICAL,
            )
            return self.checker._results[-1]
        
        # 检查重复
        duplicates = self.df.duplicated(subset=pk_columns, keep=False)
        dup_count = duplicates.sum()
        passed = dup_count == 0
        
        details = {"duplicate_count": int(dup_count)}
        if dup_count > 0 and dup_count <= 10:
            # 显示少量重复样例
            dup_samples = self.df.loc[duplicates, pk_columns].head(10)
            details["samples"] = dup_samples.to_dict(orient="records")
        
        self.checker.add_result(
            check_name="主键唯一性检查",
            passed=passed,
            message=f"主键 {pk_columns} 重复行数: {dup_count}",
            severity=CheckSeverity.PASS if passed else CheckSeverity.CRITICAL,
            details=details,
            metric_name="pk_duplicate_count",
            metric_value=dup_count,
        )
        return self.checker._results[-1]
    
    def check_time_continuity(
        self,
        date_column: str = "trade_date",
        start_date: str = "2019-01-01",
        end_date: str = "2025-12-31",
        check_weekends: bool = True,
    ) -> List[CheckResult]:
        """
        检查时间连续性
        
        Args:
            date_column: 日期列名
            start_date: 期望开始日期
            end_date: 期望结束日期
            check_weekends: 是否检查周末数据
        """
        results = []
        
        if date_column not in self.df.columns:
            self.checker.add_result(
                check_name="时间连续性检查",
                passed=False,
                message=f"日期列 {date_column} 不存在",
                severity=CheckSeverity.CRITICAL,
            )
            return self.checker._results[-1:]
        
        # 获取唯一日期
        dates = pd.to_datetime(self.df[date_column]).dt.date.unique()
        dates = pd.Series(sorted(dates))
        
        # 检查日期范围覆盖
        actual_start = dates.min()
        actual_end = dates.max()
        expected_start = pd.to_datetime(start_date).date()
        expected_end = pd.to_datetime(end_date).date()
        
        cover_start = actual_start <= expected_start
        cover_end = actual_end >= expected_end
        
        self.checker.add_result(
            check_name="日期范围覆盖检查",
            passed=cover_start and cover_end,
            message=f"实际范围: {actual_start} ~ {actual_end}, "
                    f"预期范围: {expected_start} ~ {expected_end}",
            severity=CheckSeverity.PASS if (cover_start and cover_end) 
                     else CheckSeverity.WARNING,
            details={
                "actual_start": str(actual_start),
                "actual_end": str(actual_end),
                "expected_start": start_date,
                "expected_end": end_date,
            },
        )
        results.append(self.checker._results[-1])
        
        # 检查是否有周末数据
        if check_weekends:
            dates_dt = pd.to_datetime(dates)
            weekend_mask = dates_dt.dt.dayofweek >= 5
            weekend_count = weekend_mask.sum()
            
            self.checker.add_result(
                check_name="周末数据检查",
                passed=weekend_count == 0,
                message=f"周末日期数量: {weekend_count}",
                severity=CheckSeverity.PASS if weekend_count == 0 
                         else CheckSeverity.WARNING,
                details={
                    "weekend_dates": dates[weekend_mask].astype(str).tolist()[:10]
                } if weekend_count > 0 else {},
            )
            results.append(self.checker._results[-1])
        
        # 统计交易日数量
        total_dates = len(dates)
        self.checker.add_result(
            check_name="交易日统计",
            passed=True,
            message=f"共 {total_dates} 个交易日",
            severity=CheckSeverity.INFO,
            metric_name="total_trade_dates",
            metric_value=total_dates,
        )
        results.append(self.checker._results[-1])
        
        return results
    
    def check_column_null_rates(
        self,
        critical_columns: Optional[List[str]] = None,
        max_null_rate: float = 0.05,
    ) -> List[CheckResult]:
        """
        检查各列缺失率
        
        Args:
            critical_columns: 关键列（缺失率必须为0）
            max_null_rate: 一般列的最大允许缺失率
        """
        results = []
        critical_columns = critical_columns or []
        
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            null_rate = null_count / len(self.df) if len(self.df) > 0 else 0
            
            if col in critical_columns:
                passed = null_rate == 0
                severity = CheckSeverity.PASS if passed else CheckSeverity.CRITICAL
                threshold = "0%"
            else:
                passed = null_rate <= max_null_rate
                severity = CheckSeverity.PASS if passed else CheckSeverity.WARNING
                threshold = f"{max_null_rate:.2%}"
            
            self.checker.add_result(
                check_name=f"列缺失率检查: {col}",
                passed=passed,
                message=f"缺失率 {null_rate:.4%} (阈值 {threshold})",
                severity=severity,
                metric_name=f"{col}_null_rate",
                metric_value=null_rate,
            )
            results.append(self.checker._results[-1])
        
        return results
    
    def check_data_types(
        self,
        expected_types: dict,
    ) -> List[CheckResult]:
        """
        检查数据类型
        
        Args:
            expected_types: 期望的数据类型映射，如:
                {
                    "amount_cols": ["amount", "total_mv"],  # 应为 float
                    "status_cols": ["is_trading"],           # 应为 int/bool
                    "date_cols": ["trade_date"],             # 应为 datetime/str
                }
        """
        results = []
        
        # 检查金额列（应为 float）
        for col in expected_types.get("amount_cols", []):
            if col not in self.df.columns:
                continue
            dtype = str(self.df[col].dtype)
            is_float = "float" in dtype.lower()
            
            self.checker.add_result(
                check_name=f"数据类型检查: {col}",
                passed=is_float,
                message=f"类型 {dtype}, 预期 float",
                severity=CheckSeverity.PASS if is_float else CheckSeverity.WARNING,
            )
            results.append(self.checker._results[-1])
        
        # 检查状态列（应为 int/bool）
        for col in expected_types.get("status_cols", []):
            if col not in self.df.columns:
                continue
            dtype = str(self.df[col].dtype)
            is_int_or_bool = "int" in dtype.lower() or "bool" in dtype.lower()
            
            self.checker.add_result(
                check_name=f"数据类型检查: {col}",
                passed=is_int_or_bool,
                message=f"类型 {dtype}, 预期 int/bool",
                severity=CheckSeverity.PASS if is_int_or_bool else CheckSeverity.WARNING,
            )
            results.append(self.checker._results[-1])
        
        # 检查所有列是否为 float32（内存优化）
        float64_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == np.float64:
                float64_cols.append(col)
        
        if float64_cols:
            self.checker.add_result(
                check_name="Float32类型检查",
                passed=False,
                message=f"以下列仍为float64: {float64_cols[:10]}{'...' if len(float64_cols) > 10 else ''}",
                severity=CheckSeverity.WARNING,
                details={"float64_columns": float64_cols},
            )
        else:
            self.checker.add_result(
                check_name="Float32类型检查",
                passed=True,
                message="所有浮点列均为float32",
                severity=CheckSeverity.PASS,
            )
        results.append(self.checker._results[-1])
        
        return results
    
    def run_all_common_checks(
        self,
        pk_columns: List[str],
        critical_columns: Optional[List[str]] = None,
        expected_types: Optional[dict] = None,
    ) -> List[CheckResult]:
        """
        运行所有通用检查
        
        Args:
            pk_columns: 主键列
            critical_columns: 关键列（缺失率要求0%）
            expected_types: 期望的数据类型映射
        """
        results = []
        
        # 1. 主键唯一性
        results.append(self.check_pk_uniqueness(pk_columns))
        
        # 2. 时间连续性
        if "trade_date" in self.df.columns:
            results.extend(self.check_time_continuity())
        
        # 3. 缺失率检查
        critical_cols = critical_columns or pk_columns
        results.extend(self.check_column_null_rates(critical_columns=critical_cols))
        
        # 4. 数据类型检查
        if expected_types:
            results.extend(self.check_data_types(expected_types))
        
        return results
