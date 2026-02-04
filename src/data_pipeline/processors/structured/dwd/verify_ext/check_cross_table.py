"""
DWD 扩展表数据质量检查 - 跨表一致性检查器

检查内容：
1. 量价与资金闭环 (成交额守恒、停牌一致性)
2. 事件与状态闭环 (解禁与交易、复牌事件)
3. 全局骨架一致性 (行数对齐、股票覆盖)
"""

import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

import pandas as pd
import numpy as np

from .dq_config import (
    DWD_EXT_PATHS, EXT_THRESHOLDS, TRAINING_START_DATE,
    CheckResult, TableSummary,
    setup_logging, format_number, format_percentage,
    get_file_size_mb, calculate_missing_rate,
)

logger = setup_logging(__name__)


class CrossTableChecker:
    """跨表一致性检查器"""
    
    def __init__(
        self,
        money_flow_df: Optional[pd.DataFrame] = None,
        chip_structure_df: Optional[pd.DataFrame] = None,
        industry_df: Optional[pd.DataFrame] = None,
        event_signal_df: Optional[pd.DataFrame] = None,
        macro_env_df: Optional[pd.DataFrame] = None,
        price_df: Optional[pd.DataFrame] = None,
        status_df: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            各表 DataFrame，如果不提供则从文件加载
        """
        logger.info("加载数据表...")
        
        # 扩展表
        self.money_flow_df = money_flow_df if money_flow_df is not None else \
            pd.read_parquet(DWD_EXT_PATHS.money_flow)
        
        self.chip_structure_df = chip_structure_df if chip_structure_df is not None else \
            pd.read_parquet(DWD_EXT_PATHS.chip_structure)
        
        self.industry_df = industry_df if industry_df is not None else \
            pd.read_parquet(DWD_EXT_PATHS.stock_industry)
        
        self.event_signal_df = event_signal_df if event_signal_df is not None else \
            pd.read_parquet(DWD_EXT_PATHS.event_signal)
        
        self.macro_env_df = macro_env_df if macro_env_df is not None else \
            pd.read_parquet(DWD_EXT_PATHS.macro_env)
        
        # 核心表（用于跨表检查）
        if price_df is not None:
            self.price_df = price_df
        else:
            # 尝试多个路径
            if DWD_EXT_PATHS.stock_price.exists():
                self.price_df = pd.read_parquet(DWD_EXT_PATHS.stock_price)
            elif DWD_EXT_PATHS.preprocessed_price.exists():
                self.price_df = pd.read_parquet(DWD_EXT_PATHS.preprocessed_price)
            else:
                self.price_df = None
                logger.warning("无法加载 price 表")
        
        if status_df is not None:
            self.status_df = status_df
        else:
            if DWD_EXT_PATHS.stock_status.exists():
                self.status_df = pd.read_parquet(DWD_EXT_PATHS.stock_status)
            elif DWD_EXT_PATHS.preprocessed_status.exists():
                self.status_df = pd.read_parquet(DWD_EXT_PATHS.preprocessed_status)
            else:
                self.status_df = None
                logger.warning("无法加载 status 表")
        
        self.results: List[CheckResult] = []
        self._prepare_data()
        
        logger.info("数据加载完成:")
        logger.info(f"  - money_flow: {len(self.money_flow_df):,} 行")
        logger.info(f"  - chip_structure: {len(self.chip_structure_df):,} 行")
        logger.info(f"  - industry: {len(self.industry_df):,} 行")
        logger.info(f"  - event_signal: {len(self.event_signal_df):,} 行")
        logger.info(f"  - macro_env: {len(self.macro_env_df):,} 行")
        if self.price_df is not None:
            logger.info(f"  - price: {len(self.price_df):,} 行")
    
    def _prepare_data(self):
        """预处理数据"""
        # 统一日期格式
        for df_name in ['money_flow_df', 'chip_structure_df', 'industry_df', 
                        'event_signal_df', 'macro_env_df', 'price_df', 'status_df']:
            df = getattr(self, df_name, None)
            if df is not None and 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    def check_amount_conservation(self) -> CheckResult:
        """
        检查1: 成交额守恒
        资金流向分单买入额之和应该接近总成交额
        公式：abs(sum(buy_components) - price.amount) / price.amount < 1%
        """
        logger.info("检查成交额守恒...")
        
        issues = []
        metrics = {}
        
        if self.price_df is None:
            return CheckResult(
                name="成交额守恒",
                passed=True,
                description="验证资金流分单之和≈总成交额",
                issues=["price 表不存在，跳过检查"],
                severity="WARNING"
            )
        
        # 检查必要字段
        buy_cols = ['buy_sm_amount', 'buy_md_amount', 'buy_lg_amount', 'buy_elg_amount']
        if not all(col in self.money_flow_df.columns for col in buy_cols):
            return CheckResult(
                name="成交额守恒",
                passed=True,
                description="验证资金流分单之和≈总成交额",
                issues=["money_flow 缺少分单买入额字段"],
                severity="WARNING"
            )
        
        if 'amount' not in self.price_df.columns:
            return CheckResult(
                name="成交额守恒",
                passed=True,
                description="验证资金流分单之和≈总成交额",
                issues=["price 缺少 amount 字段"],
                severity="WARNING"
            )
        
        # 合并数据
        mf_cols = ['trade_date', 'ts_code'] + buy_cols
        price_cols = ['trade_date', 'ts_code', 'amount']
        
        merged = self.money_flow_df[mf_cols].merge(
            self.price_df[price_cols],
            on=['trade_date', 'ts_code'],
            how='inner'
        )
        
        metrics['merged_rows'] = len(merged)
        
        if len(merged) == 0:
            issues.append("money_flow 和 price 无法 Join")
            return CheckResult(
                name="成交额守恒",
                passed=False,
                description="验证资金流分单之和≈总成交额",
                issues=issues,
                metrics=metrics,
                severity="ERROR"
            )
        
        # 计算分单之和
        merged['buy_total'] = merged[buy_cols].sum(axis=1)
        
        # 计算误差（注意单位可能不同）
        # amount 通常是千元，buy_xxx_amount 通常也是千元
        merged['error_ratio'] = abs(merged['buy_total'] - merged['amount']) / (merged['amount'] + 1e-8)
        
        # 过滤掉成交额为 0 的行
        valid_merged = merged[merged['amount'] > 0]
        
        if len(valid_merged) == 0:
            return CheckResult(
                name="成交额守恒",
                passed=True,
                description="验证资金流分单之和≈总成交额",
                issues=["无有效成交数据"],
                severity="WARNING"
            )
        
        # 统计超过阈值的比例
        large_error = valid_merged[valid_merged['error_ratio'] > EXT_THRESHOLDS.AMOUNT_CONSERVATION_TOLERANCE]
        large_error_ratio = len(large_error) / len(valid_merged)
        
        metrics['valid_rows'] = len(valid_merged)
        metrics['large_error_count'] = len(large_error)
        metrics['large_error_ratio'] = large_error_ratio
        metrics['mean_error_ratio'] = float(valid_merged['error_ratio'].mean())
        metrics['median_error_ratio'] = float(valid_merged['error_ratio'].median())
        
        # 如果超过 5% 的行误差大于阈值，发出警告
        if large_error_ratio > 0.05:
            issues.append(
                f"{large_error_ratio:.2%} 的行成交额误差超过 "
                f"{EXT_THRESHOLDS.AMOUNT_CONSERVATION_TOLERANCE:.2%}"
            )
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="成交额守恒",
            passed=passed,
            description="验证资金流分单之和≈总成交额",
            issues=issues,
            metrics=metrics,
            severity="WARNING" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"成交额守恒检查: {'通过' if passed else '警告'}")
        return result
    
    def check_suspension_mf_consistency(self) -> CheckResult:
        """
        检查2: 停牌一致性
        如果 status 表中股票当日是停牌，money_flow 中资金流应为 0
        """
        logger.info("检查停牌与资金流一致性...")
        
        issues = []
        metrics = {}
        
        if self.status_df is None:
            return CheckResult(
                name="停牌资金流一致性",
                passed=True,
                description="验证停牌日资金流为 0",
                issues=["status 表不存在，跳过检查"],
                severity="WARNING"
            )
        
        # 获取停牌记录
        if 'is_trading' in self.status_df.columns:
            suspended = self.status_df[self.status_df['is_trading'] == 0][['trade_date', 'ts_code']]
        elif 'is_suspended' in self.status_df.columns:
            suspended = self.status_df[self.status_df['is_suspended'] == 1][['trade_date', 'ts_code']]
        else:
            return CheckResult(
                name="停牌资金流一致性",
                passed=True,
                description="验证停牌日资金流为 0",
                issues=["status 缺少停牌标记字段"],
                severity="WARNING"
            )
        
        metrics['suspended_records'] = len(suspended)
        
        if len(suspended) == 0:
            return CheckResult(
                name="停牌资金流一致性",
                passed=True,
                description="验证停牌日资金流为 0",
                issues=["无停牌记录"],
                metrics=metrics,
                severity="INFO"
            )
        
        # 合并资金流
        if 'net_mf_amount' not in self.money_flow_df.columns:
            return CheckResult(
                name="停牌资金流一致性",
                passed=True,
                description="验证停牌日资金流为 0",
                issues=["money_flow 缺少 net_mf_amount 字段"],
                severity="WARNING"
            )
        
        merged = suspended.merge(
            self.money_flow_df[['trade_date', 'ts_code', 'net_mf_amount']],
            on=['trade_date', 'ts_code'],
            how='inner'
        )
        
        metrics['merged_suspended_rows'] = len(merged)
        
        if len(merged) > 0:
            # 检查停牌日是否有资金流
            has_flow = merged[merged['net_mf_amount'] != 0]
            metrics['suspended_with_flow'] = len(has_flow)
            
            if len(has_flow) > 0:
                issues.append(f"发现 {len(has_flow)} 行停牌日却有非零资金流")
                sample = has_flow.head(5)[['trade_date', 'ts_code', 'net_mf_amount']]
                metrics['suspended_with_flow_sample'] = sample.to_dict('records')
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="停牌资金流一致性",
            passed=passed,
            description="验证停牌日资金流为 0",
            issues=issues,
            metrics=metrics,
            severity="WARNING" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"停牌资金流一致性检查: {'通过' if passed else '警告'}")
        return result
    
    def check_row_alignment(self) -> CheckResult:
        """
        检查3: 行数对齐
        money_flow, chip_structure, industry, event_signal 的行数应与 price 高度接近
        """
        logger.info("检查行数对齐...")
        
        issues = []
        metrics = {}
        
        # 基准行数（使用 money_flow 或 price）
        if self.price_df is not None:
            base_rows = len(self.price_df)
            base_name = 'price'
        else:
            base_rows = len(self.money_flow_df)
            base_name = 'money_flow'
        
        metrics['base_table'] = base_name
        metrics['base_rows'] = base_rows
        
        # 检查各表行数
        table_rows = {
            'money_flow': len(self.money_flow_df),
            'chip_structure': len(self.chip_structure_df),
            'industry': len(self.industry_df),
            'event_signal': len(self.event_signal_df),
        }
        
        for table_name, rows in table_rows.items():
            metrics[f'{table_name}_rows'] = rows
            
            diff_ratio = abs(rows - base_rows) / base_rows if base_rows > 0 else 0
            metrics[f'{table_name}_diff_ratio'] = diff_ratio
            
            if diff_ratio > EXT_THRESHOLDS.ROW_ALIGNMENT_TOLERANCE:
                issues.append(
                    f"{table_name} 行数 {rows:,} 与 {base_name} ({base_rows:,}) "
                    f"差异 {diff_ratio:.2%} 超过阈值"
                )
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="行数对齐",
            passed=passed,
            description="验证扩展表行数与基准表一致",
            issues=issues,
            metrics=metrics,
            severity="ERROR" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"行数对齐检查: {'通过' if passed else '失败'}")
        return result
    
    def check_stock_coverage(self) -> CheckResult:
        """
        检查4: 股票覆盖
        扩展表的 ts_code 集合应该覆盖 price 表的 ts_code 集合
        """
        logger.info("检查股票覆盖...")
        
        issues = []
        metrics = {}
        
        # 基准股票集合
        if self.price_df is not None:
            base_stocks = set(self.price_df['ts_code'].unique())
            base_name = 'price'
        else:
            base_stocks = set(self.money_flow_df['ts_code'].unique())
            base_name = 'money_flow'
        
        metrics['base_table'] = base_name
        metrics['base_stock_count'] = len(base_stocks)
        
        # 检查各表股票覆盖
        table_stocks = {
            'money_flow': set(self.money_flow_df['ts_code'].unique()),
            'industry': set(self.industry_df['ts_code'].unique()),
            'event_signal': set(self.event_signal_df['ts_code'].unique()),
            'chip_structure': set(self.chip_structure_df['ts_code'].unique()),
        }
        
        for table_name, stocks in table_stocks.items():
            metrics[f'{table_name}_stock_count'] = len(stocks)
            
            # 计算覆盖率
            covered = len(base_stocks & stocks)
            missing = base_stocks - stocks
            extra = stocks - base_stocks
            
            coverage = covered / len(base_stocks) if base_stocks else 0
            
            metrics[f'{table_name}_coverage'] = coverage
            metrics[f'{table_name}_missing_count'] = len(missing)
            metrics[f'{table_name}_extra_count'] = len(extra)
            
            if len(missing) > 0:
                issues.append(
                    f"{table_name} 缺少 {len(missing)} 只 {base_name} 中的股票"
                )
                metrics[f'{table_name}_missing_samples'] = list(missing)[:10]
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="股票覆盖",
            passed=passed,
            description="验证扩展表股票覆盖完整",
            issues=issues,
            metrics=metrics,
            severity="WARNING" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"股票覆盖检查: {'通过' if passed else '警告'}")
        return result
    
    def check_date_range_alignment(self) -> CheckResult:
        """
        检查5: 日期范围对齐
        各表的日期范围应该一致
        """
        logger.info("检查日期范围对齐...")
        
        issues = []
        metrics = {}
        
        # 收集各表日期范围
        date_ranges = {}
        
        for table_name, df in [
            ('money_flow', self.money_flow_df),
            ('chip_structure', self.chip_structure_df),
            ('industry', self.industry_df),
            ('event_signal', self.event_signal_df),
        ]:
            if 'trade_date' in df.columns:
                min_date = df['trade_date'].min()
                max_date = df['trade_date'].max()
                date_ranges[table_name] = (min_date, max_date)
                metrics[f'{table_name}_date_range'] = f"{min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}"
        
        # 检查日期范围是否一致
        if len(date_ranges) > 1:
            min_dates = [r[0] for r in date_ranges.values()]
            max_dates = [r[1] for r in date_ranges.values()]
            
            if len(set(min_dates)) > 1:
                issues.append(f"各表起始日期不一致: {[d.strftime('%Y-%m-%d') for d in set(min_dates)]}")
            
            if len(set(max_dates)) > 1:
                issues.append(f"各表结束日期不一致: {[d.strftime('%Y-%m-%d') for d in set(max_dates)]}")
        
        # macro_env 特殊处理（允许日期范围不同）
        if 'trade_date' in self.macro_env_df.columns:
            macro_min = self.macro_env_df['trade_date'].min()
            macro_max = self.macro_env_df['trade_date'].max()
            metrics['macro_env_date_range'] = f"{macro_min.strftime('%Y-%m-%d')} ~ {macro_max.strftime('%Y-%m-%d')}"
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="日期范围对齐",
            passed=passed,
            description="验证各表日期范围一致",
            issues=issues,
            metrics=metrics,
            severity="WARNING" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"日期范围对齐检查: {'通过' if passed else '警告'}")
        return result
    
    def run_all_checks(self) -> List[CheckResult]:
        """运行所有检查"""
        self.results = []
        
        self.check_amount_conservation()
        self.check_suspension_mf_consistency()
        self.check_row_alignment()
        self.check_stock_coverage()
        self.check_date_range_alignment()
        
        return self.results
