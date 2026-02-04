"""
DWD 扩展表数据质量检查 - 筹码结构宽表检查器 (dwd_chip_structure)

检查内容：
1. 数据覆盖率 (holder_num, top10_hold_ratio 训练期缺失率 < 1%)
2. 比例越界 (top10_hold_ratio, top1_hold_ratio <= 100%)
3. 股东户数异常 (holder_num > 0)
4. 冷启动过渡 (2019Q1 可空，2021-01-01 必须有值)
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


class ChipStructureChecker:
    """筹码结构宽表检查器"""
    
    # 关键字段列表
    KEY_COLUMNS = [
        'holder_num', 'holder_num_chg', 'holder_num_chg_pct',
        'top10_hold_ratio', 'top10_hold_amount', 'top1_hold_ratio',
        'top10_inst_ratio', 'chip_concentration', 'holder_decrease'
    ]
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Args:
            df: 可选的DataFrame，如果不提供则从文件加载
        """
        if df is not None:
            self.df = df
        else:
            logger.info(f"加载筹码结构宽表: {DWD_EXT_PATHS.chip_structure}")
            self.df = pd.read_parquet(DWD_EXT_PATHS.chip_structure)
        
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
            name="dwd_chip_structure",
            rows=len(df),
            columns=len(df.columns),
            date_range=(
                df['trade_date'].min().strftime('%Y-%m-%d'),
                df['trade_date'].max().strftime('%Y-%m-%d')
            ),
            stock_count=df['ts_code'].nunique(),
            file_size_mb=get_file_size_mb(DWD_EXT_PATHS.chip_structure),
            memory_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            null_summary=null_summary
        )
    
    def check_data_coverage(self) -> CheckResult:
        """
        检查1: 数据覆盖率
        holder_num 和 top10_hold_ratio 训练期(2021+)缺失率 < 1%
        """
        logger.info("检查数据覆盖率...")
        
        df = self.df
        training_df = df[df['is_training']]
        
        issues = []
        metrics = {
            'training_rows': len(training_df),
            'total_rows': len(df),
        }
        
        coverage_check_cols = ['holder_num', 'top10_hold_ratio']
        
        for col in coverage_check_cols:
            if col not in training_df.columns:
                continue
            
            # 计算非零/非空覆盖率
            non_null_count = training_df[col].notna().sum()
            non_zero_count = (training_df[col] > 0).sum() if col in training_df.columns else 0
            
            missing_rate = 1 - (non_null_count / len(training_df)) if len(training_df) > 0 else 1
            
            metrics[f'{col}_missing_rate'] = missing_rate
            metrics[f'{col}_non_null_count'] = int(non_null_count)
            metrics[f'{col}_non_zero_count'] = int(non_zero_count)
            
            if missing_rate > EXT_THRESHOLDS.TRAINING_MISSING_TOLERANCE:
                issues.append(
                    f"训练期 {col} 缺失率 {missing_rate:.2%} 超过阈值 "
                    f"{EXT_THRESHOLDS.TRAINING_MISSING_TOLERANCE:.2%}"
                )
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="数据覆盖率",
            passed=passed,
            description="验证训练期(2021+)关键字段缺失率 < 1%",
            issues=issues,
            metrics=metrics,
            severity="ERROR" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"数据覆盖率检查: {'通过' if passed else '失败'}")
        return result
    
    def check_ratio_bounds(self) -> CheckResult:
        """
        检查2: 比例越界
        top10_hold_ratio 和 top1_hold_ratio <= 100%
        """
        logger.info("检查比例越界...")
        
        df = self.df
        issues = []
        metrics = {}
        
        ratio_cols = ['top10_hold_ratio', 'top1_hold_ratio', 'top10_inst_ratio']
        
        for col in ratio_cols:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            
            metrics[f'{col}_min'] = float(series.min()) if len(series) > 0 else None
            metrics[f'{col}_max'] = float(series.max()) if len(series) > 0 else None
            
            # 检查是否超过 100%
            over_100 = (series > EXT_THRESHOLDS.HOLD_RATIO_MAX).sum()
            metrics[f'{col}_over_100_count'] = int(over_100)
            
            if over_100 > 0:
                issues.append(f"{col} 存在 {over_100} 行超过 100%")
                # 记录样本
                over_100_df = df[df[col] > EXT_THRESHOLDS.HOLD_RATIO_MAX][['trade_date', 'ts_code', col]].head(5)
                metrics[f'{col}_over_100_samples'] = over_100_df.to_dict('records')
            
            # 检查负值
            negative = (series < 0).sum()
            metrics[f'{col}_negative_count'] = int(negative)
            
            if negative > 0:
                issues.append(f"{col} 存在 {negative} 行负值")
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="比例越界",
            passed=passed,
            description="验证持股比例在 [0, 100%] 范围内",
            issues=issues,
            metrics=metrics,
            severity="ERROR" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"比例越界检查: {'通过' if passed else '失败'}")
        return result
    
    def check_holder_num_validity(self) -> CheckResult:
        """
        检查3: 股东户数异常
        holder_num > 0（不能为 0 或 负数）
        """
        logger.info("检查股东户数有效性...")
        
        df = self.df
        issues = []
        metrics = {}
        
        if 'holder_num' not in df.columns:
            return CheckResult(
                name="股东户数有效性",
                passed=True,
                description="验证 holder_num > 0",
                issues=["holder_num 字段不存在"],
                severity="WARNING"
            )
        
        series = df['holder_num'].dropna()
        
        metrics['holder_num_min'] = float(series.min()) if len(series) > 0 else None
        metrics['holder_num_max'] = float(series.max()) if len(series) > 0 else None
        
        # 检查零值和负值
        zero_count = (series == 0).sum()
        negative_count = (series < 0).sum()
        
        metrics['holder_num_zero_count'] = int(zero_count)
        metrics['holder_num_negative_count'] = int(negative_count)
        
        if zero_count > 0:
            issues.append(f"holder_num 存在 {zero_count} 行零值")
        
        if negative_count > 0:
            issues.append(f"holder_num 存在 {negative_count} 行负值")
        
        passed = zero_count == 0 and negative_count == 0
        
        result = CheckResult(
            name="股东户数有效性",
            passed=passed,
            description="验证 holder_num > 0",
            issues=issues,
            metrics=metrics,
            severity="ERROR" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"股东户数有效性检查: {'通过' if passed else '失败'}")
        return result
    
    def check_coldstart_transition(self) -> CheckResult:
        """
        检查4: 冷启动过渡
        2019Q1 可全空，但 2021-01-01 当天必须有值
        """
        logger.info("检查冷启动过渡...")
        
        df = self.df
        issues = []
        metrics = {}
        
        # 2019Q1 的数据
        q1_2019 = df[(df['trade_date'] >= '2019-01-01') & (df['trade_date'] < '2019-04-01')]
        
        # 2021-01-01 附近的数据（第一个交易日可能是 1/4）
        first_trading_2021 = df[
            (df['trade_date'] >= '2021-01-01') & (df['trade_date'] <= '2021-01-10')
        ]
        
        metrics['q1_2019_rows'] = len(q1_2019)
        metrics['first_2021_rows'] = len(first_trading_2021)
        
        # 检查 2021-01-01 附近的数据
        if len(first_trading_2021) > 0:
            for col in ['holder_num', 'top10_hold_ratio']:
                if col not in first_trading_2021.columns:
                    continue
                
                non_null_ratio = first_trading_2021[col].notna().sum() / len(first_trading_2021)
                metrics[f'{col}_2021_01_coverage'] = non_null_ratio
                
                if non_null_ratio < 0.5:  # 至少 50% 的股票应该有数据
                    issues.append(
                        f"2021-01 初 {col} 覆盖率仅 {non_null_ratio:.2%}，冷启动数据可能不足"
                    )
        else:
            issues.append("2021-01 初无数据")
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="冷启动过渡",
            passed=passed,
            description="验证 2021-01-01 当天有足够数据",
            issues=issues,
            metrics=metrics,
            severity="WARNING" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"冷启动过渡检查: {'通过' if passed else '警告'}")
        return result
    
    def run_all_checks(self) -> List[CheckResult]:
        """运行所有检查"""
        self.results = []
        
        self.check_data_coverage()
        self.check_ratio_bounds()
        self.check_holder_num_validity()
        self.check_coldstart_transition()
        
        return self.results
