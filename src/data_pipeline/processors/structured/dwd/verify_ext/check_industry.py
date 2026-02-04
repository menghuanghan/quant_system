"""
DWD 扩展表数据质量检查 - 行业分类宽表检查器 (dwd_stock_industry)

检查内容：
1. 全覆盖检查 (industry_idx 训练期覆盖率 100%)
2. 无效分类清洗 (无 sw_l1_name == '未分类' 或 industry_idx == -1)
3. 索引安全性 (industry_idx 在合理范围内)
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


class IndustryChecker:
    """行业分类宽表检查器"""
    
    # 关键字段列表
    KEY_COLUMNS = [
        'industry', 'industry_idx', 'sw_l1_code', 'sw_l1_name', 'sw_l1_idx',
        'sw_l2_code', 'sw_l2_name', 'sw_l2_idx', 'sw_l3_code', 'sw_l3_name',
        'industry_changed'
    ]
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Args:
            df: 可选的DataFrame，如果不提供则从文件加载
        """
        if df is not None:
            self.df = df
        else:
            logger.info(f"加载行业分类宽表: {DWD_EXT_PATHS.stock_industry}")
            self.df = pd.read_parquet(DWD_EXT_PATHS.stock_industry)
        
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
            name="dwd_stock_industry",
            rows=len(df),
            columns=len(df.columns),
            date_range=(
                df['trade_date'].min().strftime('%Y-%m-%d'),
                df['trade_date'].max().strftime('%Y-%m-%d')
            ),
            stock_count=df['ts_code'].nunique(),
            file_size_mb=get_file_size_mb(DWD_EXT_PATHS.stock_industry),
            memory_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            null_summary=null_summary
        )
    
    def check_full_coverage(self) -> CheckResult:
        """
        检查1: 全覆盖检查
        industry 或 industry_idx 字段训练期(2021+)覆盖率应为 100%
        """
        logger.info("检查行业覆盖率...")
        
        df = self.df
        training_df = df[df['is_training']]
        
        issues = []
        metrics = {
            'training_rows': len(training_df),
            'total_rows': len(df),
        }
        
        # 检查 industry_idx 覆盖率
        if 'industry_idx' in training_df.columns:
            # 有效分类（非 -1）
            valid_industry = (training_df['industry_idx'] >= 0).sum()
            coverage = valid_industry / len(training_df) if len(training_df) > 0 else 0
            
            metrics['industry_idx_valid_count'] = int(valid_industry)
            metrics['industry_idx_coverage'] = coverage
            
            if coverage < EXT_THRESHOLDS.INDUSTRY_COVERAGE_MIN:
                issues.append(
                    f"训练期行业覆盖率 {coverage:.2%} 低于阈值 "
                    f"{EXT_THRESHOLDS.INDUSTRY_COVERAGE_MIN:.2%}"
                )
        
        # 检查 industry 字段覆盖率
        if 'industry' in training_df.columns:
            non_null = training_df['industry'].notna().sum()
            non_empty = (training_df['industry'] != '').sum()
            
            metrics['industry_non_null'] = int(non_null)
            metrics['industry_non_empty'] = int(non_empty)
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="行业覆盖率",
            passed=passed,
            description="验证训练期(2021+)行业覆盖率 >= 99%",
            issues=issues,
            metrics=metrics,
            severity="ERROR" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"行业覆盖率检查: {'通过' if passed else '失败'}")
        return result
    
    def check_invalid_classification(self) -> CheckResult:
        """
        检查2: 无效分类清洗
        检查是否存在 sw_l1_name == '未分类' 或 industry_idx == -1
        """
        logger.info("检查无效分类...")
        
        df = self.df
        training_df = df[df['is_training']]
        
        issues = []
        metrics = {}
        
        # 检查 industry_idx == -1
        if 'industry_idx' in training_df.columns:
            invalid_idx_count = (training_df['industry_idx'] == EXT_THRESHOLDS.INVALID_INDUSTRY_IDX).sum()
            metrics['invalid_idx_count'] = int(invalid_idx_count)
            
            if invalid_idx_count > 0:
                issues.append(f"训练期存在 {invalid_idx_count} 行 industry_idx=-1 (未分类)")
        
        # 检查 sw_l1_name == '未分类'
        if 'sw_l1_name' in training_df.columns:
            unclassified_count = (training_df['sw_l1_name'] == '未分类').sum()
            metrics['unclassified_sw_l1_count'] = int(unclassified_count)
            
            if unclassified_count > 0:
                issues.append(f"训练期存在 {unclassified_count} 行 sw_l1_name='未分类'")
        
        # 分析未分类的股票
        if 'industry_idx' in training_df.columns:
            invalid_df = training_df[training_df['industry_idx'] == EXT_THRESHOLDS.INVALID_INDUSTRY_IDX]
            if len(invalid_df) > 0:
                invalid_stocks = invalid_df['ts_code'].unique()
                metrics['invalid_stock_count'] = len(invalid_stocks)
                metrics['invalid_stock_samples'] = list(invalid_stocks[:10])
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="无效分类清洗",
            passed=passed,
            description="验证训练期无未分类记录",
            issues=issues,
            metrics=metrics,
            severity="ERROR" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"无效分类检查: {'通过' if passed else '失败'}")
        return result
    
    def check_index_safety(self) -> CheckResult:
        """
        检查3: 索引安全性
        industry_idx 和 sw_l1_idx 应在合理范围内
        """
        logger.info("检查索引安全性...")
        
        df = self.df
        issues = []
        metrics = {}
        
        index_cols = ['industry_idx', 'sw_l1_idx', 'sw_l2_idx']
        
        for col in index_cols:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            min_idx = int(series.min())
            max_idx = int(series.max())
            unique_count = series.nunique()
            
            metrics[f'{col}_min'] = min_idx
            metrics[f'{col}_max'] = max_idx
            metrics[f'{col}_unique_count'] = unique_count
            
            # 检查负值（除了 -1 可能用于 padding）
            negative_count = (series < -1).sum()
            metrics[f'{col}_negative_count'] = int(negative_count)
            
            if negative_count > 0:
                issues.append(f"{col} 存在 {negative_count} 个非法负值 (< -1)")
            
            # 检查异常大值（不应超过 1000）
            if max_idx > 1000:
                issues.append(f"{col} 最大值 {max_idx} 异常大")
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="索引安全性",
            passed=passed,
            description="验证行业索引在合理范围内",
            issues=issues,
            metrics=metrics,
            severity="WARNING" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"索引安全性检查: {'通过' if passed else '警告'}")
        return result
    
    def check_industry_distribution(self) -> CheckResult:
        """
        检查4: 行业分布合理性
        检查行业分布是否均匀合理
        """
        logger.info("检查行业分布...")
        
        df = self.df
        training_df = df[df['is_training']]
        
        issues = []
        metrics = {}
        
        if 'industry' in training_df.columns:
            # 行业分布
            industry_dist = training_df.groupby('industry')['ts_code'].nunique()
            
            metrics['industry_count'] = len(industry_dist)
            metrics['top_5_industries'] = industry_dist.nlargest(5).to_dict()
            metrics['bottom_5_industries'] = industry_dist.nsmallest(5).to_dict()
            
            # 检查是否有行业占比过高（单一行业超过 20%）
            total_stocks = training_df['ts_code'].nunique()
            max_industry_ratio = industry_dist.max() / total_stocks if total_stocks > 0 else 0
            
            metrics['max_industry_ratio'] = max_industry_ratio
            
            if max_industry_ratio > 0.20:
                max_industry = industry_dist.idxmax()
                issues.append(
                    f"行业 '{max_industry}' 占比 {max_industry_ratio:.2%} 过高，可能存在分类问题"
                )
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="行业分布合理性",
            passed=passed,
            description="验证行业分布均匀合理",
            issues=issues,
            metrics=metrics,
            severity="WARNING" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"行业分布检查: {'通过' if passed else '警告'}")
        return result
    
    def run_all_checks(self) -> List[CheckResult]:
        """运行所有检查"""
        self.results = []
        
        self.check_full_coverage()
        self.check_invalid_classification()
        self.check_index_safety()
        self.check_industry_distribution()
        
        return self.results
