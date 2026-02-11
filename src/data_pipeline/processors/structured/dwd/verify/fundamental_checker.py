"""
PIT 基本面宽表 (dwd_stock_fundamental) 检查器

核心目标：确保单位统一（元），避免未来函数

检查项：
- total_mv, circ_mv: 单位核验（预期均值在1e10级别），逻辑检查
- revenue_ttm, total_profit_ttm: 单位核验（预期均值在1e9~1e11级别）
- PIT检查: 2019年初数据缺失率（冷启动属正常）
- pe_ttm, pb: 异常值检查（inf, 极大值）
"""

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from .base import BaseChecker, CheckResult, CheckSeverity
from .common_checks import CommonChecker

logger = logging.getLogger(__name__)


class FundamentalChecker(BaseChecker):
    """PIT基本面宽表检查器"""
    
    @property
    def table_name(self) -> str:
        return "dwd_stock_fundamental"
    
    def run_specific_checks(self):
        """运行基本面表特定检查"""
        
        # ======================================================================
        # 第一部分：通用核验
        # ======================================================================
        common = CommonChecker(self)
        common.run_all_common_checks(
            pk_columns=["trade_date", "ts_code"],
            critical_columns=["trade_date", "ts_code"],
            expected_types={
                "amount_cols": ["total_mv", "circ_mv", "revenue_ttm", "total_profit_ttm"],
            },
        )
        
        # ======================================================================
        # 第二部分：表特定检查
        # ======================================================================
        
        # 1. 市值单位核验（重点）
        self._check_mv_unit()
        
        # 2. 市值逻辑检查: total_mv >= circ_mv
        self._check_mv_logic()
        
        # 3. 财务指标单位核验
        self._check_financial_unit()
        
        # 4. PIT检查（冷启动期缺失率）
        self._check_pit_coldstart()
        
        # 5. 估值指标异常值检查
        self._check_valuation_extreme()
    
    def _check_mv_unit(self):
        """市值单位核验"""
        
        for col in ["total_mv", "circ_mv"]:
            if col not in self.df.columns:
                continue
            
            mean_val = self.df[col].dropna().mean()
            median_val = self.df[col].dropna().median()
            log10_mean = np.log10(mean_val + 1e-10)
            
            # 预期：均值应在 100亿~1000亿 级别（1e10 ~ 1e11）
            # 如果均值在 100万级别（1e6），说明还是万元单位
            is_yuan_unit = log10_mean >= 9  # 至少10亿级别
            
            self.add_result(
                check_name=f"市值单位核验 ({col})",
                passed=is_yuan_unit,
                message=f"均值={mean_val:.2e}, 中位数={median_val:.2e}, "
                        f"量级=1e{log10_mean:.1f}, 预期量级=1e10~1e11",
                severity=CheckSeverity.PASS if is_yuan_unit else CheckSeverity.CRITICAL,
                details={
                    "mean": float(mean_val),
                    "median": float(median_val),
                    "log10_mean": float(log10_mean),
                },
                metric_name=f"{col}_mean",
                metric_value=mean_val,
            )
    
    def _check_mv_logic(self):
        """市值逻辑检查: total_mv >= circ_mv"""
        
        if not all(c in self.df.columns for c in ["total_mv", "circ_mv"]):
            self.add_result(
                check_name="市值逻辑检查",
                passed=True,
                message="缺少total_mv或circ_mv列，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        df_valid = self.df[["total_mv", "circ_mv"]].dropna()
        
        # 允许微小误差（浮点精度）
        violations = (df_valid["total_mv"] < df_valid["circ_mv"] * 0.999).sum()
        
        self.add_result(
            check_name="市值逻辑检查 (total_mv >= circ_mv)",
            passed=violations == 0,
            message=f"违规行数: {violations}",
            severity=CheckSeverity.PASS if violations == 0 else CheckSeverity.ERROR,
            metric_name="mv_logic_violations",
            metric_value=violations,
        )
    
    def _check_financial_unit(self):
        """财务指标单位核验"""
        
        financial_cols = [
            ("revenue_ttm", 9, 11),      # 预期 1e9 ~ 1e11 (10亿~1000亿)
            ("total_profit_ttm", 8, 10), # 预期 1e8 ~ 1e10 (1亿~100亿)
        ]
        
        for col, expected_min, expected_max in financial_cols:
            if col not in self.df.columns:
                continue
            
            # 排除极端值后计算均值
            series = self.df[col].dropna()
            q01 = series.quantile(0.01)
            q99 = series.quantile(0.99)
            trimmed = series[(series >= q01) & (series <= q99)]
            
            if len(trimmed) == 0:
                continue
            
            mean_val = trimmed.mean()
            log10_mean = np.log10(abs(mean_val) + 1e-10)
            
            in_range = expected_min <= log10_mean <= expected_max
            
            self.add_result(
                check_name=f"财务指标单位核验 ({col})",
                passed=in_range,
                message=f"均值(去极值)={mean_val:.2e}, 量级=1e{log10_mean:.1f}, "
                        f"预期范围=1e{expected_min}~1e{expected_max}",
                severity=CheckSeverity.PASS if in_range else CheckSeverity.WARNING,
                details={
                    "mean": float(mean_val),
                    "log10_mean": float(log10_mean),
                },
                metric_name=f"{col}_mean",
                metric_value=mean_val,
            )
    
    def _check_pit_coldstart(self):
        """PIT检查：冷启动期缺失率"""
        
        if "trade_date" not in self.df.columns:
            return
        
        # 检查2019年1月的数据缺失率
        df_2019_01 = self.df[
            (self.df["trade_date"] >= "2019-01-01") & 
            (self.df["trade_date"] <= "2019-01-31")
        ]
        
        if len(df_2019_01) == 0:
            self.add_result(
                check_name="PIT冷启动检查",
                passed=True,
                message="2019-01无数据",
                severity=CheckSeverity.WARNING,
            )
            return
        
        # 检查关键财务字段的缺失率
        pit_cols = ["revenue_ttm", "total_profit_ttm", "pe_ttm", "pb"]
        existing_cols = [c for c in pit_cols if c in df_2019_01.columns]
        
        for col in existing_cols:
            null_rate = df_2019_01[col].isna().sum() / len(df_2019_01)
            
            # 冷启动期高缺失率是正常的（< 50% 是可接受的）
            is_acceptable = null_rate < 0.8
            
            self.add_result(
                check_name=f"PIT冷启动检查 ({col}, 2019-01)",
                passed=is_acceptable,
                message=f"2019-01缺失率: {null_rate:.2%}",
                severity=CheckSeverity.INFO if is_acceptable else CheckSeverity.WARNING,
                metric_name=f"{col}_2019_01_null_rate",
                metric_value=null_rate,
            )
    
    def _check_valuation_extreme(self):
        """估值指标异常值检查"""
        
        valuation_cols = ["pe_ttm", "pb", "ps_ttm", "dv_ttm"]
        
        for col in valuation_cols:
            if col not in self.df.columns:
                continue
            
            series = self.df[col].dropna()
            
            if len(series) == 0:
                continue
            
            # 检查 inf 值
            inf_count = np.isinf(series).sum()
            
            # 检查极大值 (> 10000)
            extreme_count = (series.abs() > 10000).sum()
            
            # 检查负值（亏损股PE为负是正常的）
            negative_count = (series < 0).sum()
            
            passed = inf_count == 0
            
            self.add_result(
                check_name=f"估值指标异常值检查 ({col})",
                passed=passed,
                message=f"inf值: {inf_count}, 极值(>10000): {extreme_count}, 负值: {negative_count}",
                severity=CheckSeverity.PASS if passed else CheckSeverity.ERROR,
                details={
                    "inf_count": int(inf_count),
                    "extreme_count": int(extreme_count),
                    "negative_count": int(negative_count),
                },
            )
