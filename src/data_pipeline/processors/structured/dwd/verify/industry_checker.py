"""
行业分类宽表 (dwd_stock_industry) 检查器

核心目标：确保分类覆盖率高，无非法索引

检查项：
- sw_l1_idx, sw_l2_idx: 有效性检查（统计-1或NaN比例）
- 索引范围检查 [0, 31]
- industry_changed: 分布检查（稀疏事件）
"""

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from .base import BaseChecker, CheckResult, CheckSeverity
from .common_checks import CommonChecker

logger = logging.getLogger(__name__)


class IndustryChecker(BaseChecker):
    """行业分类宽表检查器"""
    
    @property
    def table_name(self) -> str:
        return "dwd_stock_industry"
    
    def run_specific_checks(self):
        """运行行业分类表特定检查"""
        
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
        
        # 1. 行业索引有效性检查（重点）
        self._check_industry_index_validity()
        
        # 2. 行业索引范围检查
        self._check_industry_index_range()
        
        # 3. 行业变更分布检查
        self._check_industry_changed()
        
        # 4. 行业覆盖率检查
        self._check_industry_coverage()
    
    def _check_industry_index_validity(self):
        """行业索引有效性检查"""
        
        index_cols = ["sw_l1_idx", "sw_l2_idx", "sw_l3_idx"]
        
        for col in index_cols:
            if col not in self.df.columns:
                continue
            
            series = self.df[col]
            total = len(series)
            
            # 统计 -1 或 NaN 的比例
            invalid_mask = series.isna() | (series == -1)
            invalid_count = invalid_mask.sum()
            invalid_rate = invalid_count / total if total > 0 else 0
            
            # 一级行业索引应力争 0 缺失
            if col == "sw_l1_idx":
                passed = invalid_rate < 0.01  # 允许1%
                severity = CheckSeverity.PASS if passed else CheckSeverity.ERROR
            else:
                passed = invalid_rate < 0.05  # 二三级允许5%
                severity = CheckSeverity.PASS if passed else CheckSeverity.WARNING
            
            self.add_result(
                check_name=f"行业索引有效性检查 ({col})",
                passed=passed,
                message=f"无效值(-1或NaN)比例: {invalid_rate:.4%} ({invalid_count}行)",
                severity=severity,
                metric_name=f"{col}_invalid_rate",
                metric_value=invalid_rate,
            )
    
    def _check_industry_index_range(self):
        """行业索引范围检查"""
        
        # 申万一级行业通常 31 个 (0-30)
        if "sw_l1_idx" in self.df.columns:
            series = self.df["sw_l1_idx"].dropna()
            series = series[series >= 0]  # 排除 -1
            
            if len(series) > 0:
                min_idx = int(series.min())
                max_idx = int(series.max())
                unique_count = series.nunique()
                
                # 范围应在 [0, 31]
                in_range = min_idx >= 0 and max_idx <= 31
                
                self.add_result(
                    check_name="一级行业索引范围检查 (0-31)",
                    passed=in_range,
                    message=f"范围: [{min_idx}, {max_idx}], 唯一值数量: {unique_count}",
                    severity=CheckSeverity.PASS if in_range else CheckSeverity.WARNING,
                    details={
                        "min_idx": min_idx,
                        "max_idx": max_idx,
                        "unique_count": unique_count,
                    },
                )
        
        # 二级行业检查
        if "sw_l2_idx" in self.df.columns:
            series = self.df["sw_l2_idx"].dropna()
            series = series[series >= 0]
            
            if len(series) > 0:
                unique_count = series.nunique()
                
                self.add_result(
                    check_name="二级行业索引分布",
                    passed=True,
                    message=f"唯一值数量: {unique_count}",
                    severity=CheckSeverity.INFO,
                    metric_name="sw_l2_unique_count",
                    metric_value=unique_count,
                )
    
    def _check_industry_changed(self):
        """行业变更分布检查"""
        
        if "industry_changed" not in self.df.columns:
            self.add_result(
                check_name="行业变更分布检查",
                passed=True,
                message="industry_changed 列不存在，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        series = self.df["industry_changed"]
        
        # 变更通常是稀疏事件
        change_count = (series == 1).sum()
        total_count = series.notna().sum()
        change_rate = change_count / total_count if total_count > 0 else 0
        
        # 行业变更率应很低（<1%）
        is_sparse = change_rate < 0.01
        
        self.add_result(
            check_name="行业变更稀疏性检查",
            passed=is_sparse,
            message=f"变更比例: {change_rate:.4%} ({change_count}次)",
            severity=CheckSeverity.PASS if is_sparse else CheckSeverity.WARNING,
            metric_name="industry_change_rate",
            metric_value=change_rate,
        )
        
        # 按日期检查是否有大量变更（数据源重置）
        if "trade_date" in self.df.columns:
            daily_changes = self.df.groupby("trade_date")["industry_changed"].sum()
            max_daily_changes = daily_changes.max()
            max_change_date = daily_changes.idxmax() if max_daily_changes > 0 else None
            
            # 单日变更超过100可能是数据源问题
            is_normal = max_daily_changes < 100
            
            self.add_result(
                check_name="单日行业变更检查",
                passed=is_normal,
                message=f"单日最大变更数: {max_daily_changes} (日期: {max_change_date})",
                severity=CheckSeverity.PASS if is_normal else CheckSeverity.WARNING,
                details={
                    "max_daily_changes": int(max_daily_changes),
                    "max_change_date": str(max_change_date),
                },
            )
    
    def _check_industry_coverage(self):
        """行业覆盖率检查"""
        
        if "sw_l1_name" not in self.df.columns and "sw_l1_idx" not in self.df.columns:
            return
        
        # 使用索引计算覆盖
        col = "sw_l1_idx" if "sw_l1_idx" in self.df.columns else "sw_l1_name"
        
        # 检查每日股票是否都有行业分类
        if "trade_date" in self.df.columns:
            coverage_by_date = self.df.groupby("trade_date").apply(
                lambda x: x[col].notna().mean() if col in x.columns else 1.0
            )
            
            min_coverage = coverage_by_date.min()
            mean_coverage = coverage_by_date.mean()
            
            passed = min_coverage > 0.95
            
            self.add_result(
                check_name="行业分类覆盖率检查",
                passed=passed,
                message=f"平均覆盖率: {mean_coverage:.4%}, 最低覆盖率: {min_coverage:.4%}",
                severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
                details={
                    "mean_coverage": float(mean_coverage),
                    "min_coverage": float(min_coverage),
                },
            )
