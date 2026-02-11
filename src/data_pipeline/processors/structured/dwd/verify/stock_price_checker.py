"""
量价宽表 (dwd_stock_price) 检查器

核心目标：确保价格未失真，成交量级统一（元）

检查项：
- open, high, low, close: 非负检查、逻辑检查、极值检查
- *_hfq: 趋势一致性（复权因子变化率）
- amount: 单位核验（预期均值在1e8级别）
- vol: 逻辑自洽（implied_price 与 close 差异）
- is_trading: 一致性检查
"""

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from .base import BaseChecker, CheckResult, CheckSeverity
from .common_checks import CommonChecker

logger = logging.getLogger(__name__)


class StockPriceChecker(BaseChecker):
    """量价宽表检查器"""
    
    @property
    def table_name(self) -> str:
        return "dwd_stock_price"
    
    def run_specific_checks(self):
        """运行量价表特定检查"""
        
        # ======================================================================
        # 第一部分：通用核验
        # ======================================================================
        common = CommonChecker(self)
        common.run_all_common_checks(
            pk_columns=["trade_date", "ts_code"],
            critical_columns=["trade_date", "ts_code", "close"],
            expected_types={
                "amount_cols": ["open", "high", "low", "close", "amount", "vol"],
                "status_cols": ["is_trading"],
            },
        )
        
        # ======================================================================
        # 第二部分：表特定检查
        # ======================================================================
        
        # 1. 价格非负检查
        for col in ["open", "high", "low", "close"]:
            if col in self.df.columns:
                self.check_positive(col)
        
        # 2. 价格逻辑检查: high >= low, high >= open/close, low <= open/close
        self._check_price_logic()
        
        # 3. 价格极值检查
        self._check_price_extremes()
        
        # 4. 后复权价格趋势一致性
        self._check_hfq_trend()
        
        # 5. 成交额单位核验（重点）
        self._check_amount_unit()
        
        # 6. 成交量逻辑自洽检查
        self._check_vol_consistency()
        
        # 7. is_trading 一致性检查
        self._check_trading_status()
    
    def _check_price_logic(self):
        """价格逻辑检查: high >= low, high >= open/close, low <= open/close"""
        
        df = self.df[["high", "low", "open", "close"]].dropna()
        
        # high >= low
        violations_hl = (df["high"] < df["low"]).sum()
        self.add_result(
            check_name="价格逻辑检查: high >= low",
            passed=violations_hl == 0,
            message=f"违规行数: {violations_hl}",
            severity=CheckSeverity.PASS if violations_hl == 0 else CheckSeverity.ERROR,
            metric_name="price_logic_hl_violations",
            metric_value=violations_hl,
        )
        
        # high >= open
        violations_ho = (df["high"] < df["open"]).sum()
        self.add_result(
            check_name="价格逻辑检查: high >= open",
            passed=violations_ho == 0,
            message=f"违规行数: {violations_ho}",
            severity=CheckSeverity.PASS if violations_ho == 0 else CheckSeverity.ERROR,
            metric_name="price_logic_ho_violations",
            metric_value=violations_ho,
        )
        
        # high >= close
        violations_hc = (df["high"] < df["close"]).sum()
        self.add_result(
            check_name="价格逻辑检查: high >= close",
            passed=violations_hc == 0,
            message=f"违规行数: {violations_hc}",
            severity=CheckSeverity.PASS if violations_hc == 0 else CheckSeverity.ERROR,
            metric_name="price_logic_hc_violations",
            metric_value=violations_hc,
        )
        
        # low <= open
        violations_lo = (df["low"] > df["open"]).sum()
        self.add_result(
            check_name="价格逻辑检查: low <= open",
            passed=violations_lo == 0,
            message=f"违规行数: {violations_lo}",
            severity=CheckSeverity.PASS if violations_lo == 0 else CheckSeverity.ERROR,
            metric_name="price_logic_lo_violations",
            metric_value=violations_lo,
        )
        
        # low <= close
        violations_lc = (df["low"] > df["close"]).sum()
        self.add_result(
            check_name="价格逻辑检查: low <= close",
            passed=violations_lc == 0,
            message=f"违规行数: {violations_lc}",
            severity=CheckSeverity.PASS if violations_lc == 0 else CheckSeverity.ERROR,
            metric_name="price_logic_lc_violations",
            metric_value=violations_lc,
        )
    
    def _check_price_extremes(self):
        """价格极值检查"""
        
        close_col = self.df["close"].dropna()
        
        # 高价股（如茅台 > 1000）
        high_price_count = (close_col > 1000).sum()
        max_price = close_col.max()
        
        self.add_result(
            check_name="高价股检查 (close > 1000)",
            passed=True,  # 仅为信息
            message=f"高价股行数: {high_price_count}, 最高价: {max_price:.2f}",
            severity=CheckSeverity.INFO,
            metric_name="high_price_count",
            metric_value=high_price_count,
        )
        
        # 低价股（< 1元，可能是退市风险股）
        low_price_count = (close_col < 1).sum()
        min_price = close_col.min()
        
        # 负价格检查
        negative_price = (close_col < 0).sum()
        
        self.add_result(
            check_name="低价股/异常价格检查 (close < 1 或 < 0)",
            passed=negative_price == 0,
            message=f"低价股行数: {low_price_count}, 最低价: {min_price:.4f}, 负价格行数: {negative_price}",
            severity=CheckSeverity.PASS if negative_price == 0 else CheckSeverity.ERROR,
            details={
                "low_price_count": int(low_price_count),
                "min_price": float(min_price),
                "negative_count": int(negative_price),
            },
        )
    
    def _check_hfq_trend(self):
        """后复权价格趋势一致性检查"""
        
        hfq_cols = [c for c in self.df.columns if c.endswith("_hfq")]
        
        if not hfq_cols:
            self.add_result(
                check_name="后复权价格趋势检查",
                passed=True,
                message="未发现后复权价格列",
                severity=CheckSeverity.INFO,
            )
            return
        
        # 检查 close_hfq 的日度变化率
        if "close_hfq" in self.df.columns:
            df_sorted = self.df.sort_values(["ts_code", "trade_date"])
            
            # 计算日度变化率
            df_sorted["close_hfq_pct"] = df_sorted.groupby("ts_code")["close_hfq"].pct_change()
            
            # 检查是否存在超过100%的单日跳变（可能是除权数据错误）
            extreme_jumps = (df_sorted["close_hfq_pct"].abs() > 1.0).sum()
            
            # 允许一定数量的极端跳变（涨跌停、除权等）
            passed = extreme_jumps < len(df_sorted) * 0.001  # 小于0.1%
            
            self.add_result(
                check_name="后复权价格趋势检查",
                passed=passed,
                message=f"单日涨跌幅超过100%的行数: {extreme_jumps}",
                severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
                metric_name="hfq_extreme_jump_count",
                metric_value=extreme_jumps,
            )
    
    def _check_amount_unit(self):
        """成交额单位核验（重点）"""
        
        if "amount" not in self.df.columns:
            self.add_result(
                check_name="成交额单位核验",
                passed=False,
                message="amount 列不存在",
                severity=CheckSeverity.ERROR,
            )
            return
        
        amount_mean = self.df["amount"].dropna().mean()
        amount_median = self.df["amount"].dropna().median()
        
        # 预期：均值应在 1亿~100亿 级别（1e8 ~ 1e10）
        log10_mean = np.log10(amount_mean + 1e-10)
        
        # 如果均值在 10万级别（1e5），说明单位还是千元
        is_yuan_unit = log10_mean >= 7  # 至少千万级别
        
        self.add_result(
            check_name="成交额单位核验 (amount)",
            passed=is_yuan_unit,
            message=f"均值={amount_mean:.2e}, 中位数={amount_median:.2e}, "
                    f"量级=1e{log10_mean:.1f}, 预期量级=1e8~1e10",
            severity=CheckSeverity.PASS if is_yuan_unit else CheckSeverity.CRITICAL,
            details={
                "mean": float(amount_mean),
                "median": float(amount_median),
                "log10_mean": float(log10_mean),
            },
            metric_name="amount_mean",
            metric_value=amount_mean,
        )
        
        # 按年份检查漂移
        if "trade_date" in self.df.columns:
            self.df["_year"] = pd.to_datetime(self.df["trade_date"]).dt.year
            yearly_mean = self.df.groupby("_year")["amount"].mean()
            
            self.add_result(
                check_name="成交额年度趋势检查",
                passed=True,
                message=f"年度均值: {yearly_mean.to_dict()}",
                severity=CheckSeverity.INFO,
                details={"yearly_mean": yearly_mean.to_dict()},
            )
            self.df.drop(columns=["_year"], inplace=True)
    
    def _check_vol_consistency(self):
        """成交量逻辑自洽检查"""
        
        if not all(c in self.df.columns for c in ["amount", "vol", "close"]):
            self.add_result(
                check_name="成交量逻辑自洽检查",
                passed=True,
                message="缺少必要列，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        # 计算隐含价格: amount / (vol * 100)
        # 注意：vol 单位是手（100股）
        df_valid = self.df[(self.df["vol"] > 0) & (self.df["close"] > 0)].copy()
        
        if len(df_valid) == 0:
            self.add_result(
                check_name="成交量逻辑自洽检查",
                passed=True,
                message="无有效数据",
                severity=CheckSeverity.INFO,
            )
            return
        
        df_valid["implied_price"] = df_valid["amount"] / (df_valid["vol"] * 100 + 1e-5)
        df_valid["price_diff_pct"] = (
            (df_valid["implied_price"] - df_valid["close"]).abs() / df_valid["close"]
        )
        
        # 检查差异是否在合理范围（< 20%）
        large_diff_count = (df_valid["price_diff_pct"] > 0.2).sum()
        large_diff_rate = large_diff_count / len(df_valid) if len(df_valid) > 0 else 0
        
        passed = large_diff_rate < 0.01  # 允许1%的异常
        
        self.add_result(
            check_name="成交量逻辑自洽检查 (implied_price vs close)",
            passed=passed,
            message=f"差异>20%的比例: {large_diff_rate:.4%} ({large_diff_count}行)",
            severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
            details={
                "large_diff_count": int(large_diff_count),
                "large_diff_rate": float(large_diff_rate),
            },
        )
    
    def _check_trading_status(self):
        """is_trading 一致性检查"""
        
        if "is_trading" not in self.df.columns:
            self.add_result(
                check_name="交易状态一致性检查",
                passed=True,
                message="is_trading 列不存在，跳过检查",
                severity=CheckSeverity.INFO,
            )
            return
        
        # 如果 is_trading == 0，则 vol 和 amount 应为 0
        not_trading = self.df[self.df["is_trading"] == 0]
        
        if len(not_trading) == 0:
            self.add_result(
                check_name="交易状态一致性检查",
                passed=True,
                message="无停牌数据",
                severity=CheckSeverity.INFO,
            )
            return
        
        # 检查停牌日的成交量和成交额
        vol_nonzero = (not_trading["vol"] > 0).sum() if "vol" in not_trading.columns else 0
        amount_nonzero = (not_trading["amount"] > 0).sum() if "amount" in not_trading.columns else 0
        
        passed = vol_nonzero == 0 and amount_nonzero == 0
        
        self.add_result(
            check_name="交易状态一致性检查 (停牌日vol/amount应为0)",
            passed=passed,
            message=f"停牌日数: {len(not_trading)}, vol非零: {vol_nonzero}, amount非零: {amount_nonzero}",
            severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
            details={
                "not_trading_count": len(not_trading),
                "vol_nonzero": int(vol_nonzero),
                "amount_nonzero": int(amount_nonzero),
            },
        )
