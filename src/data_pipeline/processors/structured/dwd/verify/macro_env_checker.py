"""
宏观环境表 (dwd_macro_env) 检查器

核心目标：确保无断点，数值无异常跳变

检查项：
- 全表字段缺失值检查：必须为0 NaN
- gdp_yoy, cpi_yoy: 范围检查 (-10% ~ +20%)
- shibor_*: 范围检查 (0 ~ 10%)
"""

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from .base import BaseChecker, CheckResult, CheckSeverity
from .common_checks import CommonChecker

logger = logging.getLogger(__name__)


class MacroEnvChecker(BaseChecker):
    """宏观环境表检查器"""
    
    @property
    def table_name(self) -> str:
        return "dwd_macro_env"
    
    def run_specific_checks(self):
        """运行宏观环境表特定检查"""
        
        # ======================================================================
        # 第一部分：通用核验（宏观表只有日期主键）
        # ======================================================================
        common = CommonChecker(self)
        common.check_pk_uniqueness(pk_columns=["trade_date"])
        common.check_time_continuity()
        
        # ======================================================================
        # 第二部分：表特定检查
        # ======================================================================
        
        # 1. 全表字段缺失值检查（必须为0）
        self._check_zero_nulls()
        
        # 2. GDP/CPI 范围检查
        self._check_macro_indicator_range()
        
        # 3. Shibor 利率范围检查
        self._check_shibor_range()
        
        # 4. 指数成交额单位检查
        self._check_index_amount_unit()
        
        # 5. 连续性检查（FFill正确性）
        self._check_continuity()
    
    def _check_zero_nulls(self):
        """全表字段缺失值检查（除了特定列外应为0）"""
        
        # 宏观数据通常需要前向填充保证每日都有值
        null_summary = {}
        all_zero = True
        
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            null_summary[col] = int(null_count)
            if null_count > 0:
                all_zero = False
        
        # 找出有缺失的列
        cols_with_null = {k: v for k, v in null_summary.items() if v > 0}
        
        self.add_result(
            check_name="全表缺失值检查",
            passed=all_zero,
            message=f"有缺失值的列: {cols_with_null}" if cols_with_null else "所有列无缺失值",
            severity=CheckSeverity.PASS if all_zero else CheckSeverity.ERROR,
            details={"null_summary": null_summary},
        )
    
    def _check_macro_indicator_range(self):
        """GDP/CPI 范围检查"""
        
        # GDP YoY 通常在 -10% ~ +20%
        if "gdp_yoy" in self.df.columns:
            series = self.df["gdp_yoy"].dropna()
            if len(series) > 0:
                min_val = series.min()
                max_val = series.max()
                
                in_range = min_val >= -20 and max_val <= 30
                
                self.add_result(
                    check_name="GDP同比范围检查 (-20%~30%)",
                    passed=in_range,
                    message=f"范围: [{min_val:.2f}%, {max_val:.2f}%]",
                    severity=CheckSeverity.PASS if in_range else CheckSeverity.WARNING,
                    details={
                        "min": float(min_val),
                        "max": float(max_val),
                    },
                )
        
        # CPI YoY 通常在 -5% ~ +10%
        if "cpi_yoy" in self.df.columns:
            series = self.df["cpi_yoy"].dropna()
            if len(series) > 0:
                min_val = series.min()
                max_val = series.max()
                
                in_range = min_val >= -10 and max_val <= 15
                
                self.add_result(
                    check_name="CPI同比范围检查 (-10%~15%)",
                    passed=in_range,
                    message=f"范围: [{min_val:.2f}%, {max_val:.2f}%]",
                    severity=CheckSeverity.PASS if in_range else CheckSeverity.WARNING,
                    details={
                        "min": float(min_val),
                        "max": float(max_val),
                    },
                )
    
    def _check_shibor_range(self):
        """Shibor 利率范围检查"""
        
        shibor_cols = [c for c in self.df.columns if "shibor" in c.lower()]
        
        for col in shibor_cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue
            
            min_val = series.min()
            max_val = series.max()
            
            # 利率通常在 0 ~ 10% 之间，不应出现负数
            in_range = min_val >= 0 and max_val <= 15
            
            self.add_result(
                check_name=f"Shibor利率范围检查 ({col})",
                passed=in_range,
                message=f"范围: [{min_val:.4f}%, {max_val:.4f}%]",
                severity=CheckSeverity.PASS if in_range else CheckSeverity.WARNING,
                details={
                    "min": float(min_val),
                    "max": float(max_val),
                },
            )
    
    def _check_index_amount_unit(self):
        """指数成交额单位检查"""
        
        amount_cols = [c for c in self.df.columns if "amount" in c.lower()]
        
        for col in amount_cols:
            if col not in self.df.columns:
                continue
            
            mean_val = self.df[col].dropna().mean()
            log10_mean = np.log10(mean_val + 1e-10)
            
            # 指数成交额预期在 百亿~千亿 级别（1e10 ~ 1e12）
            is_yuan_unit = log10_mean >= 9  # 至少10亿级别
            
            self.add_result(
                check_name=f"指数成交额单位检查 ({col})",
                passed=is_yuan_unit,
                message=f"均值={mean_val:.2e}, 量级=1e{log10_mean:.1f}, 预期量级=1e10~1e12",
                severity=CheckSeverity.PASS if is_yuan_unit else CheckSeverity.CRITICAL,
                details={
                    "mean": float(mean_val),
                    "log10_mean": float(log10_mean),
                },
                metric_name=f"{col}_mean",
                metric_value=mean_val,
            )
    
    def _check_continuity(self):
        """连续性检查（FFill正确性）"""
        
        if "trade_date" not in self.df.columns:
            return
        
        # 宏观数据应该是连续的（每个交易日都有数据）
        df_sorted = self.df.sort_values("trade_date")
        
        # 检查是否有日期间断
        dates = pd.to_datetime(df_sorted["trade_date"])
        date_diffs = dates.diff().dropna()
        
        # 找出间隔超过7天的（排除长假）
        large_gaps = date_diffs[date_diffs > pd.Timedelta(days=7)]
        
        self.add_result(
            check_name="日期连续性检查",
            passed=len(large_gaps) == 0,
            message=f"超过7天间隔的数量: {len(large_gaps)}",
            severity=CheckSeverity.PASS if len(large_gaps) == 0 else CheckSeverity.WARNING,
            details={
                "large_gaps": large_gaps.index.tolist()[:10] if len(large_gaps) > 0 else [],
            },
        )
        
        # 检查数值是否有异常跳变（某列日度变化超过50%）
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "trade_date" in numeric_cols:
            numeric_cols.remove("trade_date")
        
        jump_issues = []
        for col in numeric_cols[:10]:  # 只检查前10列
            series = df_sorted[col].dropna()
            if len(series) < 2:
                continue
            
            pct_change = series.pct_change().abs()
            large_jumps = (pct_change > 0.5).sum()
            
            if large_jumps > len(series) * 0.01:  # 超过1%的行有大跳变
                jump_issues.append(f"{col}: {large_jumps}次")
        
        self.add_result(
            check_name="数值跳变检查",
            passed=len(jump_issues) == 0,
            message=f"存在异常跳变的列: {jump_issues}" if jump_issues else "无异常跳变",
            severity=CheckSeverity.PASS if len(jump_issues) == 0 else CheckSeverity.INFO,
        )
