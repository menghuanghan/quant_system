"""
资金流宽表 (dwd_money_flow) 检查器

核心目标：确保所有资金字段均为"元"，且北向资金无漂移

检查项：
- buy_sm_amount, sell_lg_amount...: 单位核验（均值应在1e7级别）
- 非负检查：买入/卖出金额必须 >= 0
- net_mf_amount: 逻辑检查 (net = buy - sell)
- hsgt_north: 漂移检测（2020年 vs 2025年均值比较）
"""

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from .base import BaseChecker, CheckResult, CheckSeverity
from .common_checks import CommonChecker

logger = logging.getLogger(__name__)


class MoneyFlowChecker(BaseChecker):
    """资金流宽表检查器"""
    
    @property
    def table_name(self) -> str:
        return "dwd_money_flow"
    
    def run_specific_checks(self):
        """运行资金流表特定检查"""
        
        # ======================================================================
        # 第一部分：通用核验
        # ======================================================================
        common = CommonChecker(self)
        
        # 资金流字段
        amount_cols = [
            "buy_sm_amount", "sell_sm_amount", 
            "buy_md_amount", "sell_md_amount",
            "buy_lg_amount", "sell_lg_amount", 
            "buy_elg_amount", "sell_elg_amount",
            "net_mf_amount", "block_trade_amount",
        ]
        existing_amount_cols = [c for c in amount_cols if c in self.df.columns]
        
        common.run_all_common_checks(
            pk_columns=["trade_date", "ts_code"],
            critical_columns=["trade_date", "ts_code"],
            expected_types={
                "amount_cols": existing_amount_cols,
            },
        )
        
        # ======================================================================
        # 第二部分：表特定检查
        # ======================================================================
        
        # 1. 分单资金单位核验（重点）
        self._check_flow_amount_unit()
        
        # 2. 买入卖出金额非负检查
        self._check_non_negative()
        
        # 3. 净流入逻辑检查
        self._check_net_flow_logic()
        
        # 4. 北向资金漂移检测（重点）
        self._check_hsgt_drift()
        
        # 5. 大宗交易检查
        self._check_block_trade()
    
    def _check_flow_amount_unit(self):
        """分单资金单位核验"""
        
        # 按规模分的资金流字段
        flow_cols = [
            "buy_sm_amount", "sell_sm_amount",  # 小单
            "buy_md_amount", "sell_md_amount",  # 中单
            "buy_lg_amount", "sell_lg_amount",  # 大单
            "buy_elg_amount", "sell_elg_amount", # 特大单
        ]
        
        for col in flow_cols:
            if col not in self.df.columns:
                continue
            
            mean_val = self.df[col].dropna().mean()
            median_val = self.df[col].dropna().median()
            log10_mean = np.log10(mean_val + 1e-10)
            
            # 预期：均值应在 千万~亿 级别（1e7 ~ 1e8）
            # 如果均值在 千元级别（1e3），说明还是万元单位
            is_yuan_unit = log10_mean >= 6  # 至少百万级别
            
            self.add_result(
                check_name=f"资金流单位核验 ({col})",
                passed=is_yuan_unit,
                message=f"均值={mean_val:.2e}, 中位数={median_val:.2e}, "
                        f"量级=1e{log10_mean:.1f}, 预期量级=1e7~1e8",
                severity=CheckSeverity.PASS if is_yuan_unit else CheckSeverity.CRITICAL,
                details={
                    "mean": float(mean_val),
                    "median": float(median_val),
                    "log10_mean": float(log10_mean),
                },
                metric_name=f"{col}_mean",
                metric_value=mean_val,
            )
    
    def _check_non_negative(self):
        """买入卖出金额非负检查"""
        
        buy_sell_cols = [
            "buy_sm_amount", "sell_sm_amount",
            "buy_md_amount", "sell_md_amount",
            "buy_lg_amount", "sell_lg_amount",
            "buy_elg_amount", "sell_elg_amount",
        ]
        
        for col in buy_sell_cols:
            if col not in self.df.columns:
                continue
            
            negative_count = (self.df[col].dropna() < 0).sum()
            
            self.add_result(
                check_name=f"非负检查 ({col})",
                passed=negative_count == 0,
                message=f"负值行数: {negative_count}",
                severity=CheckSeverity.PASS if negative_count == 0 else CheckSeverity.ERROR,
                metric_name=f"{col}_negative_count",
                metric_value=negative_count,
            )
    
    def _check_net_flow_logic(self):
        """净流入逻辑检查: net = buy - sell"""
        
        # 检查 net_mf_amount 是否等于各档买入减卖出之和
        buy_cols = ["buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount"]
        sell_cols = ["sell_sm_amount", "sell_md_amount", "sell_lg_amount", "sell_elg_amount"]
        
        existing_buy = [c for c in buy_cols if c in self.df.columns]
        existing_sell = [c for c in sell_cols if c in self.df.columns]
        
        if "net_mf_amount" not in self.df.columns or len(existing_buy) == 0:
            self.add_result(
                check_name="净流入逻辑检查",
                passed=True,
                message="缺少必要列，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        # 计算预期净流入
        total_buy = self.df[existing_buy].sum(axis=1)
        total_sell = self.df[existing_sell].sum(axis=1)
        expected_net = total_buy - total_sell
        
        # 计算相关性
        actual_net = self.df["net_mf_amount"]
        
        # 排除NaN
        mask = actual_net.notna() & expected_net.notna()
        if mask.sum() < 100:
            self.add_result(
                check_name="净流入逻辑检查",
                passed=True,
                message="有效数据不足，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        correlation = actual_net[mask].corr(expected_net[mask])
        
        # 相关性应接近1.0
        passed = correlation > 0.99
        
        self.add_result(
            check_name="净流入逻辑检查 (net_mf_amount vs sum(buy-sell))",
            passed=passed,
            message=f"相关系数: {correlation:.6f}",
            severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
            metric_name="net_flow_correlation",
            metric_value=correlation,
        )
    
    def _check_hsgt_drift(self):
        """北向资金漂移检测"""
        
        if "hsgt_north" not in self.df.columns:
            self.add_result(
                check_name="北向资金漂移检测",
                passed=True,
                message="hsgt_north 列不存在，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        if "trade_date" not in self.df.columns:
            return
        
        # 分别计算 2020年 和 2025年 的日均值绝对值
        self.df["_year"] = pd.to_datetime(self.df["trade_date"]).dt.year
        
        yearly_stats = self.df.groupby("_year")["hsgt_north"].agg(["mean", "std", "count"])
        
        # 检查 2020 和 2025 的均值是否在同一数量级
        if 2020 in yearly_stats.index and 2025 in yearly_stats.index:
            mean_2020 = abs(yearly_stats.loc[2020, "mean"])
            mean_2025 = abs(yearly_stats.loc[2025, "mean"])
            
            # 两者应在同一数量级（比值在0.1~10之间）
            if mean_2020 > 0 and mean_2025 > 0:
                ratio = mean_2025 / mean_2020
                passed = 0.1 <= ratio <= 10
                
                self.add_result(
                    check_name="北向资金漂移检测 (2020 vs 2025)",
                    passed=passed,
                    message=f"2020均值={mean_2020:.2e}, 2025均值={mean_2025:.2e}, 比值={ratio:.2f}",
                    severity=CheckSeverity.PASS if passed else CheckSeverity.CRITICAL,
                    details={
                        "mean_2020": float(mean_2020),
                        "mean_2025": float(mean_2025),
                        "ratio": float(ratio),
                    },
                )
            else:
                self.add_result(
                    check_name="北向资金漂移检测",
                    passed=True,
                    message="2020或2025年均值为0，跳过漂移检测",
                    severity=CheckSeverity.INFO,
                )
        else:
            self.add_result(
                check_name="北向资金漂移检测",
                passed=True,
                message=f"缺少2020或2025年数据，可用年份: {list(yearly_stats.index)}",
                severity=CheckSeverity.INFO,
            )
        
        # 年度趋势
        self.add_result(
            check_name="北向资金年度趋势",
            passed=True,
            message=f"年度均值: {yearly_stats['mean'].to_dict()}",
            severity=CheckSeverity.INFO,
            details={"yearly_stats": yearly_stats.to_dict()},
        )
        
        self.df.drop(columns=["_year"], inplace=True)
    
    def _check_block_trade(self):
        """大宗交易检查"""
        
        if "block_trade_amount" not in self.df.columns:
            return
        
        mean_val = self.df["block_trade_amount"].dropna().mean()
        nonzero_count = (self.df["block_trade_amount"] > 0).sum()
        total_count = self.df["block_trade_amount"].notna().sum()
        
        # 大宗交易是稀疏的，大部分日期应该为0
        nonzero_rate = nonzero_count / total_count if total_count > 0 else 0
        
        self.add_result(
            check_name="大宗交易检查",
            passed=True,
            message=f"均值={mean_val:.2e}, 非零比例={nonzero_rate:.4%}",
            severity=CheckSeverity.INFO,
            details={
                "mean": float(mean_val),
                "nonzero_rate": float(nonzero_rate),
            },
        )
