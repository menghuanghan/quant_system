"""
DWD 扩展表数据质量检查 - 事件信号宽表检查器 (dwd_event_signal)

检查内容：
1. 非零检查 (is_repurchase_ann, pledge_ratio 等字段的 Sum > 0)
2. 质押率分布 (非零占比应 > 30%)
3. 逻辑一致性 (in_repurchase_window=1 时近期 is_repurchase_ann 曾为 1)
4. 主键唯一性
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from .dq_config import (
    DWD_EXT_PATHS, EXT_THRESHOLDS, TRAINING_START_DATE,
    CheckResult, TableSummary,
    setup_logging, format_number, format_percentage,
    get_file_size_mb, calculate_missing_rate, is_training_period,
)

logger = setup_logging(__name__)


class EventSignalChecker:
    """事件信号宽表检查器"""
    
    # 关键字段列表
    KEY_COLUMNS = [
        'is_repurchase_ann', 'repurchase_amount', 'in_repurchase_window',
        'is_unlock_day', 'unlock_share', 'unlock_ratio', 'days_to_unlock', 'in_unlock_window',
        'pledge_ratio', 'pledge_ratio_high',
        'is_dividend_ann', 'cash_div', 'stk_div', 'in_dividend_window',
        'has_event', 'has_risk_event'
    ]
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Args:
            df: 可选的DataFrame，如果不提供则从文件加载
        """
        if df is not None:
            self.df = df
        else:
            logger.info(f"加载事件信号宽表: {DWD_EXT_PATHS.event_signal}")
            self.df = pd.read_parquet(DWD_EXT_PATHS.event_signal)
        
        self.results: List[CheckResult] = []
        self._prepare_data()
    
    def _prepare_data(self):
        """预处理数据"""
        # 确保日期格式统一
        if 'trade_date' in self.df.columns:
            self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
        
        # 添加训练期标记
        self.df['is_training'] = self.df['trade_date'] >= pd.to_datetime(TRAINING_START_DATE)
    
    def get_summary(self) -> TableSummary:
        """获取数据表概览"""
        df = self.df
        
        # 计算缺失率
        null_summary = {}
        for col in self.KEY_COLUMNS:
            if col in df.columns:
                null_summary[col] = calculate_missing_rate(df[col])
        
        return TableSummary(
            name="dwd_event_signal",
            rows=len(df),
            columns=len(df.columns),
            date_range=(
                df['trade_date'].min().strftime('%Y-%m-%d'),
                df['trade_date'].max().strftime('%Y-%m-%d')
            ),
            stock_count=df['ts_code'].nunique(),
            file_size_mb=get_file_size_mb(DWD_EXT_PATHS.event_signal),
            memory_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            null_summary=null_summary
        )
    
    def check_nonzero_signals(self) -> CheckResult:
        """
        检查1: 非零检查（关键）
        is_repurchase_ann, pledge_ratio 等字段的 Sum > 0
        """
        logger.info("检查事件信号非零...")
        
        df = self.df
        issues = []
        metrics = {}
        
        # 事件信号字段（应该有非零值）
        event_cols = {
            'is_repurchase_ann': '回购公告',
            'is_unlock_day': '解禁日',
            'is_dividend_ann': '分红公告',
            'has_event': '综合事件',
        }
        
        for col, name in event_cols.items():
            if col not in df.columns:
                continue
            
            sum_val = df[col].sum()
            nonzero_count = (df[col] > 0).sum()
            
            metrics[f'{col}_sum'] = float(sum_val)
            metrics[f'{col}_nonzero_count'] = int(nonzero_count)
            
            if sum_val == 0:
                issues.append(f"{name}({col}) Sum=0，事件数据可能未正确 Join")
        
        # 数值字段（应该有非零值）
        amount_cols = {
            'pledge_ratio': '质押率',
            'repurchase_amount': '回购金额',
            'cash_div': '现金分红',
        }
        
        for col, name in amount_cols.items():
            if col not in df.columns:
                continue
            
            sum_val = df[col].sum()
            nonzero_count = (df[col] > 0).sum()
            
            metrics[f'{col}_sum'] = float(sum_val)
            metrics[f'{col}_nonzero_count'] = int(nonzero_count)
            
            if sum_val == 0:
                issues.append(f"{name}({col}) Sum=0，数据可能未正确 Join")
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="事件信号非零检查",
            passed=passed,
            description="验证事件信号字段有非零数据",
            issues=issues,
            metrics=metrics,
            severity="CRITICAL" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"事件信号非零检查: {'通过' if passed else '失败'}")
        return result
    
    def check_pledge_ratio_distribution(self) -> CheckResult:
        """
        检查2: 质押率分布
        在全市场中，非零占比应 > 30%（A股常态）
        """
        logger.info("检查质押率分布...")
        
        df = self.df
        training_df = df[df['is_training']]
        
        issues = []
        metrics = {}
        
        if 'pledge_ratio' not in training_df.columns:
            return CheckResult(
                name="质押率分布",
                passed=True,
                description="验证质押率非零占比 > 30%",
                issues=["pledge_ratio 字段不存在"],
                severity="WARNING"
            )
        
        # 计算非零占比
        total_rows = len(training_df)
        nonzero_rows = (training_df['pledge_ratio'] > 0).sum()
        nonzero_ratio = nonzero_rows / total_rows if total_rows > 0 else 0
        
        metrics['training_total_rows'] = total_rows
        metrics['pledge_nonzero_rows'] = int(nonzero_rows)
        metrics['pledge_nonzero_ratio'] = nonzero_ratio
        
        if nonzero_ratio < EXT_THRESHOLDS.PLEDGE_RATIO_COVERAGE_MIN:
            issues.append(
                f"训练期质押率非零占比 {nonzero_ratio:.2%} 低于阈值 "
                f"{EXT_THRESHOLDS.PLEDGE_RATIO_COVERAGE_MIN:.2%}"
            )
        
        # 检查质押率范围
        series = training_df['pledge_ratio'].dropna()
        if len(series) > 0:
            metrics['pledge_ratio_min'] = float(series.min())
            metrics['pledge_ratio_max'] = float(series.max())
            metrics['pledge_ratio_mean'] = float(series[series > 0].mean()) if (series > 0).sum() > 0 else 0
            
            # 检查是否有超过 100% 的异常值
            over_100 = (series > EXT_THRESHOLDS.PLEDGE_RATIO_MAX).sum()
            metrics['pledge_ratio_over_100'] = int(over_100)
            
            if over_100 > 0:
                issues.append(f"质押率存在 {over_100} 行超过 100%")
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="质押率分布",
            passed=passed,
            description="验证质押率非零占比 > 30%",
            issues=issues,
            metrics=metrics,
            severity="ERROR" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"质押率分布检查: {'通过' if passed else '失败'}")
        return result
    
    def check_window_logic_consistency(self) -> CheckResult:
        """
        检查3: 逻辑一致性
        若 in_repurchase_window=1，对应的 is_repurchase_ann 在过去 N 天内曾为 1
        """
        logger.info("检查窗口逻辑一致性...")
        
        df = self.df
        issues = []
        metrics = {}
        
        # 检查 in_repurchase_window
        if 'in_repurchase_window' in df.columns and 'is_repurchase_ann' in df.columns:
            # 有窗口标记的行
            in_window = df[df['in_repurchase_window'] == 1]
            metrics['in_repurchase_window_count'] = len(in_window)
            
            # 回购公告的行
            repurchase_ann = df[df['is_repurchase_ann'] == 1]
            metrics['is_repurchase_ann_count'] = len(repurchase_ann)
            
            # 如果有窗口但没有公告，可能有问题
            if len(in_window) > 0 and len(repurchase_ann) == 0:
                issues.append("存在 in_repurchase_window=1 但全局无 is_repurchase_ann=1")
        
        # 检查 in_dividend_window
        if 'in_dividend_window' in df.columns and 'is_dividend_ann' in df.columns:
            in_div_window = df[df['in_dividend_window'] == 1]
            metrics['in_dividend_window_count'] = len(in_div_window)
            
            dividend_ann = df[df['is_dividend_ann'] == 1]
            metrics['is_dividend_ann_count'] = len(dividend_ann)
            
            if len(in_div_window) > 0 and len(dividend_ann) == 0:
                issues.append("存在 in_dividend_window=1 但全局无 is_dividend_ann=1")
        
        # 检查 in_unlock_window
        if 'in_unlock_window' in df.columns and 'is_unlock_day' in df.columns:
            in_unlock_window = df[df['in_unlock_window'] == 1]
            metrics['in_unlock_window_count'] = len(in_unlock_window)
            
            unlock_day = df[df['is_unlock_day'] == 1]
            metrics['is_unlock_day_count'] = len(unlock_day)
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="窗口逻辑一致性",
            passed=passed,
            description="验证事件窗口与事件信号逻辑一致",
            issues=issues,
            metrics=metrics,
            severity="WARNING" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"窗口逻辑一致性检查: {'通过' if passed else '警告'}")
        return result
    
    def check_primary_key_unique(self) -> CheckResult:
        """
        检查4: 主键唯一性
        """
        logger.info("检查主键唯一性...")
        
        df = self.df
        issues = []
        metrics = {}
        
        # 检查重复
        duplicates = df.duplicated(subset=['trade_date', 'ts_code'], keep=False)
        duplicate_count = duplicates.sum()
        
        metrics['total_rows'] = len(df)
        metrics['duplicate_count'] = int(duplicate_count)
        
        if duplicate_count > 0:
            issues.append(f"发现 {duplicate_count} 行主键重复")
        
        passed = duplicate_count == 0
        
        result = CheckResult(
            name="主键唯一性",
            passed=passed,
            description="验证 (trade_date, ts_code) 主键唯一",
            issues=issues,
            metrics=metrics,
            severity="CRITICAL" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"主键唯一性检查: {'通过' if passed else '失败'}")
        return result
    
    def run_all_checks(self) -> List[CheckResult]:
        """运行所有检查"""
        self.results = []
        
        self.check_nonzero_signals()
        self.check_pledge_ratio_distribution()
        self.check_window_logic_consistency()
        self.check_primary_key_unique()
        
        return self.results
