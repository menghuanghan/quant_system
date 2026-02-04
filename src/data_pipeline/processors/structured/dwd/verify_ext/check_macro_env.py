"""
DWD 扩展表数据质量检查 - 宏观环境宽表检查器 (dwd_macro_env)

检查内容：
1. 日期连续性 (训练期内交易日无断点)
2. 字段纯净度 (必须剔除 gdp 绝对值，只保留 gdp_yoy)
3. 数据滞后性 PIT (CPI/PPI 变动日在每月 10-15 号左右)
4. 缺失值检查 (训练期无 NaN)
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


class MacroEnvChecker:
    """宏观环境宽表检查器"""
    
    # 关键字段列表
    KEY_COLUMNS = [
        'gdp_yoy', 'cpi_yoy', 'cpi_mom', 'pmi', 'pmi_prod', 'pmi_new_order',
        'm2', 'm2_yoy', 'lpr_1y', 'lpr_5y',
        'shibor_on', 'shibor_1w', 'shibor_1m', 'shibor_3m', 'shibor_6m', 'shibor_1y',
        'market_congestion', 'stock_bond_spread', 'pmi_regime'
    ]
    
    # 禁止出现的字段（可能导致锯齿陷阱）
    FORBIDDEN_COLUMNS = ['gdp']  # GDP绝对值
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Args:
            df: 可选的DataFrame，如果不提供则从文件加载
        """
        if df is not None:
            self.df = df
        else:
            logger.info(f"加载宏观环境宽表: {DWD_EXT_PATHS.macro_env}")
            self.df = pd.read_parquet(DWD_EXT_PATHS.macro_env)
        
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
        for col in df.columns:
            null_summary[col] = calculate_missing_rate(df[col])
        
        return TableSummary(
            name="dwd_macro_env",
            rows=len(df),
            columns=len(df.columns),
            date_range=(
                df['trade_date'].min().strftime('%Y-%m-%d'),
                df['trade_date'].max().strftime('%Y-%m-%d')
            ),
            stock_count=0,  # 宏观数据无股票代码
            file_size_mb=get_file_size_mb(DWD_EXT_PATHS.macro_env),
            memory_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            null_summary=null_summary
        )
    
    def check_date_continuity(self) -> CheckResult:
        """
        检查1: 日期连续性
        检查 trade_date 是否连续（交易日）, 训练期内无断点
        """
        logger.info("检查日期连续性...")
        
        df = self.df
        training_df = df[df['is_training']].sort_values('trade_date')
        
        issues = []
        metrics = {}
        
        metrics['total_dates'] = len(df)
        metrics['training_dates'] = len(training_df)
        
        if len(training_df) < 2:
            return CheckResult(
                name="日期连续性",
                passed=True,
                description="验证训练期交易日无断点",
                issues=["训练期数据不足"],
                severity="WARNING"
            )
        
        # 计算日期间隔
        dates = training_df['trade_date'].sort_values()
        date_diffs = dates.diff().dropna()
        
        # 正常交易日间隔应该是 1-3 天（周末）或最多 7-10 天（长假）
        # 超过 10 天可能存在问题
        long_gaps = date_diffs[date_diffs > pd.Timedelta(days=10)]
        
        metrics['max_gap_days'] = int(date_diffs.max().days)
        metrics['long_gaps_count'] = len(long_gaps)
        
        if len(long_gaps) > 0:
            gap_dates = [(dates[dates.diff() == gap].iloc[0].strftime('%Y-%m-%d'), gap.days) 
                        for gap in long_gaps.unique()][:5]
            metrics['long_gap_samples'] = gap_dates
            issues.append(f"发现 {len(long_gaps)} 处超过 10 天的日期间隔")
        
        passed = len(long_gaps) == 0
        
        result = CheckResult(
            name="日期连续性",
            passed=passed,
            description="验证训练期交易日无断点",
            issues=issues,
            metrics=metrics,
            severity="WARNING" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"日期连续性检查: {'通过' if passed else '警告'}")
        return result
    
    def check_field_purity(self) -> CheckResult:
        """
        检查2: 字段纯净度
        必须剔除绝对值 gdp，只保留 gdp_yoy
        """
        logger.info("检查字段纯净度...")
        
        df = self.df
        issues = []
        metrics = {}
        
        metrics['total_columns'] = len(df.columns)
        metrics['column_list'] = list(df.columns)
        
        # 检查禁止字段
        for col in self.FORBIDDEN_COLUMNS:
            if col in df.columns:
                issues.append(f"发现禁止字段 '{col}'（可能导致锯齿陷阱）")
                metrics[f'forbidden_{col}_present'] = True
            else:
                metrics[f'forbidden_{col}_present'] = False
        
        # 检查必要字段
        required_cols = ['gdp_yoy', 'cpi_yoy', 'pmi']
        missing_required = [col for col in required_cols if col not in df.columns]
        
        if missing_required:
            issues.append(f"缺少必要字段: {missing_required}")
            metrics['missing_required_cols'] = missing_required
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="字段纯净度",
            passed=passed,
            description="验证无锯齿陷阱字段(如 gdp 绝对值)",
            issues=issues,
            metrics=metrics,
            severity="CRITICAL" if 'gdp' in df.columns else ("ERROR" if not passed else "INFO")
        )
        
        self.results.append(result)
        logger.info(f"字段纯净度检查: {'通过' if passed else '失败'}")
        return result
    
    def check_pit_lag(self) -> CheckResult:
        """
        检查3: 数据滞后性 PIT
        CPI/PMI 变动日应在每月 10-15 号左右，而非 1 号
        """
        logger.info("检查 PIT 数据滞后性...")
        
        df = self.df.sort_values('trade_date')
        training_df = df[df['is_training']]
        
        issues = []
        metrics = {}
        
        # 检查 CPI 变动日
        if 'cpi_yoy' in training_df.columns:
            cpi_series = training_df[['trade_date', 'cpi_yoy']].dropna()
            
            if len(cpi_series) > 1:
                # 找出 CPI 变动的日期
                cpi_series['cpi_changed'] = cpi_series['cpi_yoy'].diff() != 0
                change_dates = cpi_series[cpi_series['cpi_changed']]['trade_date']
                
                if len(change_dates) > 0:
                    # 检查变动日期的月份日
                    change_days = change_dates.dt.day
                    avg_change_day = change_days.mean()
                    
                    metrics['cpi_change_count'] = len(change_dates)
                    metrics['cpi_avg_change_day'] = float(avg_change_day)
                    metrics['cpi_change_day_min'] = int(change_days.min())
                    metrics['cpi_change_day_max'] = int(change_days.max())
                    
                    # CPI 应该在每月 9-18 号更新
                    if avg_change_day < EXT_THRESHOLDS.CPI_UPDATE_DAY_MIN:
                        issues.append(
                            f"CPI 平均变动日 {avg_change_day:.1f} 过早"
                            f"（预期 {EXT_THRESHOLDS.CPI_UPDATE_DAY_MIN}-{EXT_THRESHOLDS.CPI_UPDATE_DAY_MAX}），"
                            f"可能存在 look-ahead bias"
                        )
        
        # 检查 PMI 变动日
        if 'pmi' in training_df.columns:
            pmi_series = training_df[['trade_date', 'pmi']].dropna()
            
            if len(pmi_series) > 1:
                pmi_series['pmi_changed'] = pmi_series['pmi'].diff() != 0
                change_dates = pmi_series[pmi_series['pmi_changed']]['trade_date']
                
                if len(change_dates) > 0:
                    change_days = change_dates.dt.day
                    avg_change_day = change_days.mean()
                    
                    metrics['pmi_change_count'] = len(change_dates)
                    metrics['pmi_avg_change_day'] = float(avg_change_day)
                    metrics['pmi_change_day_min'] = int(change_days.min())
                    metrics['pmi_change_day_max'] = int(change_days.max())
                    
                    # PMI 应该在每月 1-5 号更新
                    if avg_change_day > EXT_THRESHOLDS.PMI_UPDATE_DAY_MAX + 2:
                        issues.append(
                            f"PMI 平均变动日 {avg_change_day:.1f} 过晚"
                            f"（预期 {EXT_THRESHOLDS.PMI_UPDATE_DAY_MIN}-{EXT_THRESHOLDS.PMI_UPDATE_DAY_MAX}）"
                        )
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="PIT 数据滞后性",
            passed=passed,
            description="验证月度指标变动日符合公布规律",
            issues=issues,
            metrics=metrics,
            severity="ERROR" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"PIT 数据滞后性检查: {'通过' if passed else '失败'}")
        return result
    
    def check_training_nan(self) -> CheckResult:
        """
        检查4: 训练期缺失值检查
        2021-01-01 之后的关键列无 NaN
        """
        logger.info("检查训练期缺失值...")
        
        df = self.df
        training_df = df[df['is_training']]
        
        issues = []
        metrics = {
            'training_rows': len(training_df),
        }
        
        nan_columns = []
        for col in self.KEY_COLUMNS:
            if col not in training_df.columns:
                continue
            
            nan_count = training_df[col].isna().sum()
            nan_rate = nan_count / len(training_df) if len(training_df) > 0 else 0
            
            metrics[f'{col}_nan_count'] = int(nan_count)
            metrics[f'{col}_nan_rate'] = nan_rate
            
            if nan_count > 0:
                nan_columns.append((col, nan_count, nan_rate))
        
        if nan_columns:
            for col, count, rate in nan_columns:
                issues.append(f"训练期 {col} 存在 {count} 个 NaN ({rate:.2%})")
        
        # 如果缺失率低于 1%，视为可接受
        critical_nans = [c for c, _, r in nan_columns if r > EXT_THRESHOLDS.TRAINING_MISSING_TOLERANCE]
        
        passed = len(critical_nans) == 0
        
        result = CheckResult(
            name="训练期缺失值检查",
            passed=passed,
            description="验证训练期(2021+)关键字段无 NaN",
            issues=issues,
            metrics=metrics,
            severity="ERROR" if not passed else ("WARNING" if nan_columns else "INFO")
        )
        
        self.results.append(result)
        logger.info(f"训练期缺失值检查: {'通过' if passed else '失败'}")
        return result
    
    def run_all_checks(self) -> List[CheckResult]:
        """运行所有检查"""
        self.results = []
        
        self.check_date_continuity()
        self.check_field_purity()
        self.check_pit_lag()
        self.check_training_nan()
        
        return self.results
