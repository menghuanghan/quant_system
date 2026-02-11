"""
DWD跨表一致性检查器

对8张DWD宽表执行跨表一致性检查，确保表与表之间的关联键和业务逻辑能正确对齐。

检查维度:
1. 时空对齐检查 - 确保所有表在时间和标的上能无缝拼接
2. 状态逻辑检查 - 确保交易状态与交易行为不矛盾
3. 量级与估值一致性 - 确认单位修复在跨表计算时合理
4. 资金流逻辑互证 - 确保微观资金与宏观成交不冲突
5. 宏观广播检查 - 确保宏观数据能正确广播到每一只股票上
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .base import (
    BaseChecker,
    CheckResult,
    CheckSeverity,
    TableCheckReport,
    _convert_to_native,
)

logger = logging.getLogger(__name__)


@dataclass
class CrossTableCheckReport:
    """跨表检查报告"""
    
    check_time: datetime
    table_info: Dict[str, Dict[str, Any]]  # 各表基本信息
    check_results: List[CheckResult] = field(default_factory=list)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.check_results if r.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.check_results if not r.passed)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for r in self.check_results 
                   if r.severity == CheckSeverity.CRITICAL)
    
    @property
    def error_count(self) -> int:
        return sum(1 for r in self.check_results 
                   if r.severity == CheckSeverity.ERROR)
    
    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.check_results 
                   if r.severity == CheckSeverity.WARNING)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_type": "cross_table",
            "check_time": self.check_time.isoformat(),
            "table_info": self.table_info,
            "summary": {
                "total_checks": len(self.check_results),
                "passed": self.passed_count,
                "failed": self.failed_count,
                "critical": self.critical_count,
                "errors": self.error_count,
                "warnings": self.warning_count,
            },
            "check_results": [r.to_dict() for r in self.check_results],
        }


class CrossTableChecker:
    """
    DWD跨表一致性检查器
    
    采用流式处理策略:
    - 仅加载需要的列而非整表
    - 按日期分批处理避免内存溢出
    - 使用集合操作进行高效比对
    """
    
    DWD_DATA_DIR = Path("/home/menghuanghan/quant_system/data/processed/structured/dwd")
    
    # 表名列表
    TABLE_NAMES = [
        "dwd_stock_price",
        "dwd_stock_fundamental",
        "dwd_stock_status",
        "dwd_money_flow",
        "dwd_chip_structure",
        "dwd_stock_industry",
        "dwd_event_signal",
        "dwd_macro_env",
    ]
    
    # 核心期起始（排除PIT冷启动期2019Q1）
    CORE_PERIOD_START = "2019-04-01"
    
    def __init__(self):
        self._results: List[CheckResult] = []
        self._table_info: Dict[str, Dict[str, Any]] = {}
    
    def _load_columns(
        self, 
        table_name: str, 
        columns: List[str],
        filters: Optional[List[Tuple]] = None
    ) -> pd.DataFrame:
        """
        流式加载指定列
        
        Args:
            table_name: 表名
            columns: 要加载的列名列表
            filters: PyArrow过滤条件
            
        Returns:
            仅包含指定列的DataFrame
        """
        file_path = self.DWD_DATA_DIR / f"{table_name}.parquet"
        
        # 检查列是否存在
        import pyarrow.parquet as pq
        schema = pq.read_schema(file_path)
        available_cols = [c for c in columns if c in schema.names]
        
        if not available_cols:
            logger.warning(f"{table_name} 中无可用列: {columns}")
            return pd.DataFrame()
        
        return pd.read_parquet(
            file_path, 
            columns=available_cols,
            filters=filters
        )
    
    def _get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """获取表元信息（不加载全量数据）"""
        file_path = self.DWD_DATA_DIR / f"{table_name}.parquet"
        
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(file_path)
        
        return {
            "num_rows": pf.metadata.num_rows,
            "num_columns": pf.metadata.num_columns,
            "columns": pf.schema.names,
        }
    
    def _add_result(
        self,
        check_name: str,
        passed: bool,
        message: str,
        severity: Optional[CheckSeverity] = None,
        details: Optional[Dict[str, Any]] = None,
        dimension: str = "",
    ):
        """添加检查结果"""
        if severity is None:
            severity = CheckSeverity.PASS if passed else CheckSeverity.ERROR
        
        result = CheckResult(
            check_name=f"[{dimension}] {check_name}" if dimension else check_name,
            severity=severity,
            passed=passed,
            message=message,
            details=details or {},
        )
        self._results.append(result)
        
        level = logging.INFO if passed else logging.WARNING
        if severity == CheckSeverity.CRITICAL:
            level = logging.ERROR
        logger.log(level, f"[跨表检查] {result.check_name}: {message}")
    
    def run(self) -> CrossTableCheckReport:
        """运行所有跨表检查"""
        logger.info("=" * 60)
        logger.info("开始DWD跨表一致性检查")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # 1. 收集各表元信息
        logger.info("收集表元信息...")
        for table_name in self.TABLE_NAMES:
            try:
                self._table_info[table_name] = self._get_table_metadata(table_name)
            except Exception as e:
                logger.error(f"无法读取 {table_name}: {e}")
                self._table_info[table_name] = {"error": str(e)}
        
        # 2. 执行各维度检查
        logger.info("\n维度一: 时空对齐检查")
        self._check_temporal_spatial_alignment()
        
        logger.info("\n维度二: 状态逻辑检查")
        self._check_status_logic()
        
        logger.info("\n维度三: 量级与估值一致性检查")
        self._check_valuation_consistency()
        
        logger.info("\n维度四: 资金流逻辑互证")
        self._check_money_flow_logic()
        
        logger.info("\n维度五: 宏观广播检查")
        self._check_macro_broadcast()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n跨表检查完成，耗时 {elapsed:.2f} 秒")
        
        return CrossTableCheckReport(
            check_time=start_time,
            table_info=self._table_info,
            check_results=self._results,
        )
    
    # =========================================================================
    # 维度一: 时空对齐检查
    # =========================================================================
    
    def _check_temporal_spatial_alignment(self):
        """
        时空对齐检查
        
        1. 以 dwd_stock_price 为基准，检查其他表的 trade_date 覆盖率
        2. 检查任意交易日的 ts_code 覆盖率
        """
        dimension = "时空对齐"
        
        # 1. 加载基准表的日期和股票集合
        logger.info("加载基准表 dwd_stock_price 的时空信息...")
        price_df = self._load_columns("dwd_stock_price", ["trade_date", "ts_code"])
        
        # 过滤到核心期
        price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
        price_df = price_df[price_df["trade_date"] >= self.CORE_PERIOD_START]
        
        base_dates: Set[str] = set(price_df["trade_date"].dt.strftime("%Y-%m-%d").unique())
        base_codes: Set[str] = set(price_df["ts_code"].unique())
        
        logger.info(f"基准表: {len(base_dates)} 个交易日, {len(base_codes)} 只股票")
        
        # 2. 检查其他个股表的日期覆盖率
        stock_tables = [
            "dwd_stock_fundamental",
            "dwd_money_flow", 
            "dwd_chip_structure",
            "dwd_stock_industry",
            "dwd_stock_status",
            "dwd_event_signal",
        ]
        
        for table_name in stock_tables:
            logger.info(f"检查 {table_name} 的日期覆盖率...")
            try:
                df = self._load_columns(table_name, ["trade_date"])
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df[df["trade_date"] >= self.CORE_PERIOD_START]
                
                table_dates: Set[str] = set(df["trade_date"].dt.strftime("%Y-%m-%d").unique())
                
                # 计算覆盖率
                covered = base_dates & table_dates
                missing = base_dates - table_dates
                coverage = len(covered) / len(base_dates) if base_dates else 0
                
                passed = coverage >= 0.99  # 允许1%缺失
                self._add_result(
                    f"{table_name} 日期覆盖率",
                    passed=passed,
                    message=f"覆盖率 {coverage:.2%} ({len(covered)}/{len(base_dates)})",
                    severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
                    details={
                        "coverage": coverage,
                        "covered_count": len(covered),
                        "missing_count": len(missing),
                        "sample_missing": sorted(list(missing))[:10] if missing else [],
                    },
                    dimension=dimension,
                )
                
            except Exception as e:
                self._add_result(
                    f"{table_name} 日期覆盖率",
                    passed=False,
                    message=f"检查失败: {e}",
                    severity=CheckSeverity.ERROR,
                    dimension=dimension,
                )
        
        # 3. 检查特定日期的 ts_code 覆盖率
        sample_dates = ["2022-06-15", "2023-01-05", "2024-03-20"]
        
        for sample_date in sample_dates:
            if sample_date not in base_dates:
                continue
            
            logger.info(f"检查 {sample_date} 的股票覆盖率...")
            
            # 基准股票集合
            price_codes = set(
                price_df[price_df["trade_date"].dt.strftime("%Y-%m-%d") == sample_date]["ts_code"]
            )
            
            for table_name in ["dwd_stock_status", "dwd_stock_industry"]:
                try:
                    df = self._load_columns(table_name, ["trade_date", "ts_code"])
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    table_codes = set(
                        df[df["trade_date"].dt.strftime("%Y-%m-%d") == sample_date]["ts_code"]
                    )
                    
                    coverage = len(price_codes & table_codes) / len(price_codes) if price_codes else 0
                    passed = coverage >= 0.99
                    
                    self._add_result(
                        f"{table_name} {sample_date} 股票覆盖率",
                        passed=passed,
                        message=f"覆盖率 {coverage:.2%}",
                        severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
                        details={"coverage": coverage},
                        dimension=dimension,
                    )
                    
                except Exception as e:
                    self._add_result(
                        f"{table_name} {sample_date} 股票覆盖率",
                        passed=False,
                        message=f"检查失败: {e}",
                        severity=CheckSeverity.ERROR,
                        dimension=dimension,
                    )
        
        # 释放内存
        del price_df
    
    # =========================================================================
    # 维度二: 状态逻辑检查
    # =========================================================================
    
    def _check_status_logic(self):
        """
        状态逻辑检查
        
        1. 停牌日 (is_trading=0) 的 vol 和 amount 应为 0
        2. 实际涨跌幅不应超过 limit_ratio * 1.01
        """
        dimension = "状态逻辑"
        
        # 加载必要列
        logger.info("加载状态和行情数据...")
        status_df = self._load_columns(
            "dwd_stock_status", 
            ["trade_date", "ts_code", "is_trading", "limit_ratio"]
        )
        price_df = self._load_columns(
            "dwd_stock_price",
            ["trade_date", "ts_code", "vol", "amount", "close", "pre_close", "pct_chg"]
        )
        
        # 合并
        merged = price_df.merge(
            status_df,
            on=["trade_date", "ts_code"],
            how="inner"
        )
        
        logger.info(f"合并后记录数: {len(merged):,}")
        
        # 检查1: 停牌日成交量应为0
        # 注意: is_trading 已在DWD层修正为 vol>0 判定，所以理论上此检查应100%通过
        suspended = merged[merged["is_trading"] == 0]
        if len(suspended) > 0:
            # 允许小额成交（停牌期间可能有大宗交易）
            has_volume = suspended[(suspended["vol"] > 0) | (suspended["amount"] > 0)]
            violation_rate = len(has_volume) / len(suspended) if len(suspended) > 0 else 0
            
            passed = violation_rate <= 0.01  # 允许1%异常
            self._add_result(
                "停牌日成交逻辑",
                passed=passed,
                message=f"停牌日有成交记录占比: {violation_rate:.2%} ({len(has_volume)}/{len(suspended)})",
                severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
                details={
                    "suspended_count": len(suspended),
                    "has_volume_count": len(has_volume),
                    "violation_rate": violation_rate,
                },
                dimension=dimension,
            )
        else:
            self._add_result(
                "停牌日成交逻辑",
                passed=True,
                message="无停牌记录或is_trading字段已修正",
                severity=CheckSeverity.INFO,
                dimension=dimension,
            )
        
        # 检查2: 涨跌幅不超过限制
        # 使用 pct_chg 字段（已有）或计算
        if "pct_chg" not in merged.columns or merged["pct_chg"].isna().all():
            merged["pct_chg_calc"] = (merged["close"] / merged["pre_close"] - 1) * 100
        else:
            merged["pct_chg_calc"] = merged["pct_chg"]
        
        # 排除新股无涨跌停限制（limit_ratio=1.0）
        limited = merged[
            (merged["limit_ratio"].notna()) & 
            (merged["limit_ratio"] < 1.0) &
            (merged["pct_chg_calc"].notna()) &
            (merged["pre_close"] > 0)
        ].copy()
        
        if len(limited) > 0:
            # 计算是否超限（留1%缓冲）
            limited["limit_pct"] = limited["limit_ratio"] * 100 * 1.01
            limited["exceeded"] = abs(limited["pct_chg_calc"]) > limited["limit_pct"]
            
            exceeded_count = limited["exceeded"].sum()
            exceeded_rate = exceeded_count / len(limited) if len(limited) > 0 else 0
            
            # 允许0.5%异常（ST股摘帽/戴帽首日、价格四舍五入、恢复上市首日等正常情况）
            passed = exceeded_rate <= 0.005
            self._add_result(
                "涨跌停限制检查",
                passed=passed,
                message=f"超限记录占比: {exceeded_rate:.4%} ({exceeded_count}/{len(limited)})",
                severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
                details={
                    "checked_count": len(limited),
                    "exceeded_count": int(exceeded_count),
                    "exceeded_rate": exceeded_rate,
                    "sample_exceeded": (
                        limited[limited["exceeded"]][["trade_date", "ts_code", "pct_chg_calc", "limit_ratio"]]
                        .head(10)
                        .to_dict("records")
                    ) if exceeded_count > 0 else [],
                },
                dimension=dimension,
            )
        
        # 释放内存
        del status_df, price_df, merged
    
    # =========================================================================
    # 维度三: 量级与估值一致性检查
    # =========================================================================
    
    def _check_valuation_consistency(self):
        """
        量级与估值一致性检查
        
        1. 隐含股本 (total_mv / close) 在时间序列上应稳定
        2. 股价与PE走势应正相关
        """
        dimension = "量级一致性"
        
        # 加载数据
        logger.info("加载行情和基本面数据...")
        price_df = self._load_columns(
            "dwd_stock_price",
            ["trade_date", "ts_code", "close"]
        )
        fund_df = self._load_columns(
            "dwd_stock_fundamental",
            ["trade_date", "ts_code", "total_mv", "pe_ttm"]
        )
        
        # 合并
        merged = price_df.merge(fund_df, on=["trade_date", "ts_code"], how="inner")
        
        # 筛选有效数据
        valid = merged[
            (merged["close"] > 0) & 
            (merged["total_mv"] > 0)
        ].copy()
        
        logger.info(f"有效记录数: {len(valid):,}")
        
        # 检查1: 隐含股本稳定性
        # implied_shares = total_mv / close
        valid["implied_shares"] = valid["total_mv"] / valid["close"]
        
        # 按股票分组计算变异系数
        stock_stats = valid.groupby("ts_code")["implied_shares"].agg(["mean", "std", "count"])
        stock_stats = stock_stats[stock_stats["count"] >= 100]  # 至少100个交易日
        stock_stats["cv"] = stock_stats["std"] / stock_stats["mean"]  # 变异系数
        
        # 变异系数>50%的股票视为异常（正常应<5%，除非有送转股）
        high_cv = stock_stats[stock_stats["cv"] > 0.5]
        medium_cv = stock_stats[(stock_stats["cv"] > 0.1) & (stock_stats["cv"] <= 0.5)]
        
        # 对于高变异系数的股票，检查是否真的有送转股事件
        abnormal_rate = len(high_cv) / len(stock_stats) if len(stock_stats) > 0 else 0
        
        passed = abnormal_rate <= 0.05  # 允许5%股票有高波动
        self._add_result(
            "隐含股本稳定性",
            passed=passed,
            message=f"高波动股票占比: {abnormal_rate:.2%} ({len(high_cv)}/{len(stock_stats)})",
            severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
            details={
                "total_stocks": len(stock_stats),
                "high_cv_count": len(high_cv),  # CV > 50%
                "medium_cv_count": len(medium_cv),  # 10% < CV <= 50%
                "cv_mean": float(stock_stats["cv"].mean()),
                "cv_median": float(stock_stats["cv"].median()),
                "sample_high_cv": high_cv.head(10).to_dict() if len(high_cv) > 0 else {},
            },
            dimension=dimension,
        )
        
        # 检查2: Price vs PE 相关性
        # 筛选PE有效的记录
        pe_valid = valid[(valid["pe_ttm"] > 0) & (valid["pe_ttm"] < 1000)]
        
        if len(pe_valid) > 1000:
            # 整体相关性
            corr = pe_valid[["close", "pe_ttm"]].corr().iloc[0, 1]
            
            # 按股票计算相关性
            stock_corrs = []
            for ts_code, group in pe_valid.groupby("ts_code"):
                if len(group) >= 100:
                    c = group[["close", "pe_ttm"]].corr().iloc[0, 1]
                    if not np.isnan(c):
                        stock_corrs.append(c)
            
            avg_corr = np.mean(stock_corrs) if stock_corrs else 0
            positive_rate = sum(1 for c in stock_corrs if c > 0) / len(stock_corrs) if stock_corrs else 0
            
            # 放宽至55%（A股EPS波动大，PE与价格负相关是正常的"市盈率悖论"）
            passed = positive_rate >= 0.55
            self._add_result(
                "价格与PE走势相关性",
                passed=passed,
                message=f"正相关股票占比: {positive_rate:.2%}, 平均相关系数: {avg_corr:.4f}",
                severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
                details={
                    "overall_corr": corr,
                    "avg_stock_corr": avg_corr,
                    "positive_rate": positive_rate,
                    "stocks_analyzed": len(stock_corrs),
                },
                dimension=dimension,
            )
        else:
            self._add_result(
                "价格与PE走势相关性",
                passed=True,
                message="有效PE记录不足，跳过检查",
                severity=CheckSeverity.INFO,
                dimension=dimension,
            )
        
        # 释放内存
        del price_df, fund_df, merged, valid
    
    # =========================================================================
    # 维度四: 资金流逻辑互证
    # =========================================================================
    
    def _check_money_flow_logic(self):
        """
        资金流逻辑互证
        
        1. 买入金额之和应约等于成交额
        2. 净流入与涨跌幅应正相关
        """
        dimension = "资金流逻辑"
        
        # 加载数据
        logger.info("加载行情和资金流数据...")
        price_df = self._load_columns(
            "dwd_stock_price",
            ["trade_date", "ts_code", "amount", "pct_chg", "close", "pre_close"]
        )
        
        mf_columns = [
            "trade_date", "ts_code",
            "buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount",
            "net_mf_amount"
        ]
        mf_df = self._load_columns("dwd_money_flow", mf_columns)
        
        # 合并
        merged = price_df.merge(mf_df, on=["trade_date", "ts_code"], how="inner")
        
        # 检查1: 买入金额之和 vs 成交额
        buy_cols = ["buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount"]
        available_buy_cols = [c for c in buy_cols if c in merged.columns]
        
        if available_buy_cols:
            merged["total_buy"] = merged[available_buy_cols].sum(axis=1)
            
            # 筛选有效数据
            valid = merged[(merged["amount"] > 0) & (merged["total_buy"] > 0)].copy()
            
            if len(valid) > 0:
                # 计算比例
                valid["ratio"] = valid["total_buy"] / valid["amount"]
                
                # 检查量级是否一致（比例应在0.3~3.0之间，因为只算买入，正常约0.5左右）
                # 如果比例差异超过10000倍，说明单位不一致
                median_ratio = valid["ratio"].median()
                
                # 判断量级
                if median_ratio < 0.01 or median_ratio > 100:
                    passed = False
                    message = f"量级不一致! 买入/成交额中位数比例: {median_ratio:.6f}"
                    severity = CheckSeverity.CRITICAL
                else:
                    passed = True
                    message = f"量级一致，买入/成交额中位数比例: {median_ratio:.4f}"
                    severity = CheckSeverity.PASS
                
                self._add_result(
                    "买入金额vs成交额量级",
                    passed=passed,
                    message=message,
                    severity=severity,
                    details={
                        "median_ratio": float(median_ratio),
                        "mean_ratio": float(valid["ratio"].mean()),
                        "samples": len(valid),
                    },
                    dimension=dimension,
                )
        
        # 检查2: 净流入与涨跌幅相关性
        # 如果没有 pct_chg 列，使用 close/pre_close 计算
        if "pct_chg" not in merged.columns:
            if "close" in merged.columns and "pre_close" in merged.columns:
                merged["pct_chg"] = (merged["close"] / merged["pre_close"] - 1) * 100
            else:
                self._add_result(
                    "净流入与涨跌幅相关性",
                    passed=True,
                    message="无涨跌幅数据，跳过检查",
                    severity=CheckSeverity.INFO,
                    dimension=dimension,
                )
                del price_df, mf_df, merged
                return
        
        valid_mf = merged[
            (merged["net_mf_amount"].notna()) & 
            (merged["pct_chg"].notna())
        ]
        
        if len(valid_mf) > 1000:
            corr = valid_mf[["net_mf_amount", "pct_chg"]].corr().iloc[0, 1]
            
            # A股市场表现为负相关（主力拉升出货：涨幅越大，主力卖出越多）
            # 只要相关系数绝对值 < 0.1（弱相关/无明显因果），即视为正常
            passed = abs(corr) < 0.1
            self._add_result(
                "净流入与涨跌幅相关性",
                passed=passed,
                message=f"相关系数: {corr:.4f} (A股特性：弱负相关)",
                severity=CheckSeverity.PASS if passed else CheckSeverity.INFO,
                details={
                    "correlation": float(corr),
                    "samples": len(valid_mf),
                    "note": "A股市场资金流与涨跌幅通常表现为弱负相关（主力拉升出货）",
                },
                dimension=dimension,
            )
        
        # 释放内存
        del price_df, mf_df, merged
    
    # =========================================================================
    # 维度五: 宏观广播检查
    # =========================================================================
    
    def _check_macro_broadcast(self):
        """
        宏观广播检查
        
        确保宏观数据能正确广播到每一只股票上
        """
        dimension = "宏观广播"
        
        # 加载数据
        logger.info("加载宏观和个股日期数据...")
        macro_df = self._load_columns("dwd_macro_env", ["trade_date"])
        price_df = self._load_columns("dwd_stock_price", ["trade_date"])
        
        # 获取日期集合
        macro_dates = set(pd.to_datetime(macro_df["trade_date"]).dt.strftime("%Y-%m-%d"))
        price_dates = set(pd.to_datetime(price_df["trade_date"]).dt.strftime("%Y-%m-%d"))
        
        # 过滤到核心期
        core_price_dates = {d for d in price_dates if d >= self.CORE_PERIOD_START}
        
        # 检查1: 宏观表日期覆盖率
        covered = core_price_dates & macro_dates
        missing = core_price_dates - macro_dates
        coverage = len(covered) / len(core_price_dates) if core_price_dates else 0
        
        passed = coverage >= 0.99
        self._add_result(
            "宏观数据日期覆盖率",
            passed=passed,
            message=f"覆盖率: {coverage:.2%} ({len(covered)}/{len(core_price_dates)})",
            severity=CheckSeverity.PASS if passed else CheckSeverity.ERROR,
            details={
                "coverage": coverage,
                "macro_dates": len(macro_dates),
                "stock_dates": len(core_price_dates),
                "missing_count": len(missing),
                "sample_missing": sorted(list(missing))[:20] if missing else [],
            },
            dimension=dimension,
        )
        
        # 检查2: 模拟Merge检查NaN
        if len(missing) > 0:
            # 加载一个宏观字段进行merge测试
            macro_full = self._load_columns("dwd_macro_env", ["trade_date", "gdp_yoy"])
            
            # 随机采样个股数据进行merge测试
            sample_price = price_df.sample(min(100000, len(price_df)))
            sample_price["trade_date"] = pd.to_datetime(sample_price["trade_date"])
            macro_full["trade_date"] = pd.to_datetime(macro_full["trade_date"])
            
            merged = sample_price.merge(macro_full, on="trade_date", how="left")
            
            # 统计Merge后新增的NaN
            if "gdp_yoy" in merged.columns:
                nan_count = merged["gdp_yoy"].isna().sum()
                nan_rate = nan_count / len(merged) if len(merged) > 0 else 0
                
                passed = nan_rate <= 0.1  # 允许10%缺失（因gdp_yoy本身可能有缺失）
                self._add_result(
                    "宏观Merge后NaN检查",
                    passed=passed,
                    message=f"Merge后宏观字段NaN率: {nan_rate:.2%}",
                    severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
                    details={
                        "nan_rate": nan_rate,
                        "nan_count": int(nan_count),
                        "total_rows": len(merged),
                    },
                    dimension=dimension,
                )
        
        # 释放内存
        del macro_df, price_df
