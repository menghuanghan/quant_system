"""
状态与风险表 (dwd_stock_status) 检查器

核心目标：确保交易规则标记准确

检查项：
- limit_ratio: 值域检查 {0.05, 0.1, 0.2, 0.3}
- market: 覆盖检查（主板、创业板、科创板、北交所）
- is_st: 逻辑检查（ST股票limit_ratio通常为0.05）
"""

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from .base import BaseChecker, CheckResult, CheckSeverity
from .common_checks import CommonChecker

logger = logging.getLogger(__name__)


class StatusChecker(BaseChecker):
    """状态与风险表检查器"""
    
    @property
    def table_name(self) -> str:
        return "dwd_stock_status"
    
    def run_specific_checks(self):
        """运行状态表特定检查"""
        
        # ======================================================================
        # 第一部分：通用核验
        # ======================================================================
        common = CommonChecker(self)
        common.run_all_common_checks(
            pk_columns=["trade_date", "ts_code"],
            critical_columns=["trade_date", "ts_code"],
            expected_types={
                "status_cols": ["is_st", "is_hs300", "is_sz50", "is_zz500"],
            },
        )
        
        # ======================================================================
        # 第二部分：表特定检查
        # ======================================================================
        
        # 1. 涨跌停幅度值域检查
        self._check_limit_ratio()
        
        # 2. 市场板块覆盖检查
        self._check_market_coverage()
        
        # 3. ST标记逻辑检查
        self._check_st_logic()
        
        # 4. 指数成分股标记检查
        self._check_index_membership()
    
    def _check_limit_ratio(self):
        """涨跌停幅度值域检查"""
        
        if "limit_ratio" not in self.df.columns:
            self.add_result(
                check_name="涨跌停幅度值域检查",
                passed=True,
                message="limit_ratio 列不存在，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        series = self.df["limit_ratio"].dropna()
        
        # 预期值: {0.05, 0.1, 0.2, 0.3}
        expected_values = {0.05, 0.1, 0.2, 0.3}
        actual_values = set(series.unique())
        
        # 检查是否有非预期值（考虑浮点精度）
        def is_expected(v):
            return any(abs(v - ev) < 0.001 for ev in expected_values)
        
        unexpected_values = {v for v in actual_values if not is_expected(v)}
        
        passed = len(unexpected_values) == 0
        
        self.add_result(
            check_name="涨跌停幅度值域检查",
            passed=passed,
            message=f"预期值: {expected_values}, 实际唯一值: {actual_values}, "
                    f"非预期值: {unexpected_values}",
            severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
            details={
                "expected": list(expected_values),
                "actual": list(actual_values),
                "unexpected": list(unexpected_values),
            },
        )
        
        # 分布统计
        value_counts = series.value_counts().to_dict()
        self.add_result(
            check_name="涨跌停幅度分布",
            passed=True,
            message=f"分布: {value_counts}",
            severity=CheckSeverity.INFO,
            details={"value_counts": value_counts},
        )
    
    def _check_market_coverage(self):
        """市场板块覆盖检查"""
        
        if "market" not in self.df.columns:
            self.add_result(
                check_name="市场板块覆盖检查",
                passed=True,
                message="market 列不存在，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        actual_markets = set(self.df["market"].dropna().unique())
        
        # 预期包含的市场
        expected_markets = {"主板", "创业板", "科创板", "北交所"}
        
        # 检查覆盖（允许不同的名称表示）
        # 也可能是数字编码
        if all(isinstance(m, (int, float)) for m in actual_markets):
            # 数值编码
            self.add_result(
                check_name="市场板块覆盖检查",
                passed=True,
                message=f"市场为数值编码, 唯一值: {actual_markets}",
                severity=CheckSeverity.INFO,
                details={"market_values": list(actual_markets)},
            )
        else:
            # 字符串
            missing = expected_markets - actual_markets
            
            self.add_result(
                check_name="市场板块覆盖检查",
                passed=len(missing) == 0,
                message=f"实际市场: {actual_markets}, 缺失: {missing}",
                severity=CheckSeverity.PASS if len(missing) == 0 else CheckSeverity.WARNING,
                details={
                    "actual": list(actual_markets),
                    "missing": list(missing),
                },
            )
    
    def _check_st_logic(self):
        """ST标记逻辑检查"""
        
        if "is_st" not in self.df.columns:
            self.add_result(
                check_name="ST标记逻辑检查",
                passed=True,
                message="is_st 列不存在，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        # ST股票数量统计
        st_count = (self.df["is_st"] == 1).sum()
        total_count = self.df["is_st"].notna().sum()
        st_rate = st_count / total_count if total_count > 0 else 0
        
        self.add_result(
            check_name="ST股票比例统计",
            passed=True,
            message=f"ST比例: {st_rate:.4%} ({st_count}行)",
            severity=CheckSeverity.INFO,
            metric_name="st_rate",
            metric_value=st_rate,
        )
        
        # ST股票的limit_ratio应为0.05（创业板/科创板除外）
        if "limit_ratio" in self.df.columns:
            st_stocks = self.df[self.df["is_st"] == 1]
            
            if len(st_stocks) > 0:
                # ST股票中limit_ratio不是0.05的比例
                non_005_count = (abs(st_stocks["limit_ratio"] - 0.05) > 0.001).sum()
                non_005_rate = non_005_count / len(st_stocks) if len(st_stocks) > 0 else 0
                
                # 允许一定比例（创业板/科创板ST可能有不同规则）
                passed = non_005_rate < 0.3
                
                self.add_result(
                    check_name="ST股票涨跌停幅度检查",
                    passed=passed,
                    message=f"ST股票中limit_ratio非0.05的比例: {non_005_rate:.4%}",
                    severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
                )
    
    def _check_index_membership(self):
        """指数成分股标记检查"""
        
        index_cols = ["is_hs300", "is_sz50", "is_zz500", "is_zz1000"]
        
        for col in index_cols:
            if col not in self.df.columns:
                continue
            
            series = self.df[col]
            member_count = (series == 1).sum()
            total_count = series.notna().sum()
            member_rate = member_count / total_count if total_count > 0 else 0
            
            # 成分股应该是少数
            is_reasonable = member_rate < 0.2
            
            self.add_result(
                check_name=f"指数成分股检查 ({col})",
                passed=is_reasonable,
                message=f"成分股比例: {member_rate:.4%} ({member_count}行)",
                severity=CheckSeverity.PASS if is_reasonable else CheckSeverity.WARNING,
                metric_name=f"{col}_rate",
                metric_value=member_rate,
            )
