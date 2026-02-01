"""
DWD 数据质量检查 - PIT基本面宽表检查器 (dwd_stock_fundamental)

检查内容：
1. 数据覆盖率检查（特别是冷启动期）
2. PIT逻辑验证 (Report_Date < Announce_Date <= Trade_Date)
3. TTM数值逻辑检查（突变检测）
4. 估值指标一致性验证
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from .dq_config import (
    DWD_PATHS, THRESHOLDS,
    CheckResult, TableSummary,
    setup_logging, format_number, format_percentage,
    get_file_size_mb, calculate_missing_rate,
)

logger = setup_logging(__name__)


class FundamentalTableChecker:
    """PIT基本面宽表检查器"""
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Args:
            df: 可选的DataFrame，如果不提供则从文件加载
        """
        if df is not None:
            self.df = df
        else:
            logger.info(f"加载基本面宽表: {DWD_PATHS.stock_fundamental}")
            self.df = pd.read_parquet(DWD_PATHS.stock_fundamental)
        
        self.results: List[CheckResult] = []
        self._prepare_data()
    
    def _prepare_data(self):
        """预处理数据"""
        # 确保日期格式统一
        date_cols = ['trade_date', 'ann_date', 'report_date']
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
    
    def get_summary(self) -> TableSummary:
        """获取数据表概览"""
        df = self.df
        
        # 计算缺失率
        null_summary = {}
        for col in df.columns:
            null_summary[col] = calculate_missing_rate(df[col])
        
        return TableSummary(
            name="dwd_stock_fundamental",
            rows=len(df),
            columns=len(df.columns),
            date_range=(
                df['trade_date'].min().strftime('%Y-%m-%d'),
                df['trade_date'].max().strftime('%Y-%m-%d')
            ),
            stock_count=df['ts_code'].nunique(),
            file_size_mb=get_file_size_mb(DWD_PATHS.stock_fundamental),
            memory_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            null_summary=null_summary
        )
    
    def check_data_coverage(self) -> CheckResult:
        """
        检查1: 数据覆盖率
        特别关注冷启动日期（如2021-01-04）的关键字段非空率
        """
        logger.info("检查数据覆盖率...")
        
        df = self.df
        issues = []
        metrics = {}
        
        # 关键TTM字段
        ttm_fields = [col for col in df.columns if 'ttm' in col.lower()]
        sq_fields = [col for col in df.columns if '_sq' in col.lower()]
        key_fields = ttm_fields + sq_fields + ['total_mv', 'circ_mv', 'pe_ttm', 'pb', 'ps_ttm']
        key_fields = [f for f in key_fields if f in df.columns]
        
        # 整体覆盖率
        overall_coverage = {}
        for field in key_fields:
            coverage = 1 - calculate_missing_rate(df[field])
            overall_coverage[field] = coverage
        
        metrics['overall_coverage'] = overall_coverage
        
        # 检查低覆盖率字段
        low_coverage_fields = {k: v for k, v in overall_coverage.items() if v < THRESHOLDS.FUNDAMENTAL_COVERAGE_MIN}
        if low_coverage_fields:
            issues.append(f"以下字段覆盖率低于{THRESHOLDS.FUNDAMENTAL_COVERAGE_MIN*100}%: {list(low_coverage_fields.keys())}")
        
        # 冷启动期覆盖率（前几个交易日）
        df_sorted = df.sort_values('trade_date')
        first_date = df_sorted['trade_date'].min()
        cold_start_days = 5
        cold_start_end = first_date + pd.Timedelta(days=10)  # 大约前5个交易日
        
        cold_start_df = df_sorted[df_sorted['trade_date'] <= cold_start_end]
        cold_start_coverage = {}
        
        for field in key_fields:
            if field in cold_start_df.columns:
                coverage = 1 - calculate_missing_rate(cold_start_df[field])
                cold_start_coverage[field] = coverage
        
        metrics['cold_start_coverage'] = cold_start_coverage
        metrics['cold_start_period'] = f"{first_date.strftime('%Y-%m-%d')} ~ {cold_start_end.strftime('%Y-%m-%d')}"
        metrics['cold_start_rows'] = len(cold_start_df)
        
        # 检查冷启动期低覆盖率
        cold_low_coverage = {k: v for k, v in cold_start_coverage.items() if v < THRESHOLDS.FUNDAMENTAL_COVERAGE_MIN}
        if cold_low_coverage:
            # 冷启动期覆盖率低是可以接受的（财报数据需要时间累积）
            issues.append(f"冷启动期以下字段覆盖率低于{THRESHOLDS.FUNDAMENTAL_COVERAGE_MIN*100}%: {list(cold_low_coverage.keys())}")
        
        # 按月统计覆盖率变化（监控数据质量趋势）
        df['year_month'] = df['trade_date'].dt.to_period('M')
        monthly_coverage = df.groupby('year_month').apply(
            lambda x: 1 - x[key_fields[:3]].isnull().mean() if key_fields else pd.Series()
        )
        metrics['monthly_coverage_sample'] = monthly_coverage.head(12).to_dict() if len(monthly_coverage) > 0 else {}
        
        # 综合判定（整体覆盖率决定是否通过，冷启动期不作为硬性要求）
        avg_coverage = np.mean(list(overall_coverage.values())) if overall_coverage else 0
        passed = avg_coverage >= THRESHOLDS.FUNDAMENTAL_COVERAGE_MIN * 0.9  # 允许10%的宽容度
        
        severity = "ERROR" if not passed else ("WARNING" if issues else "INFO")
        
        result = CheckResult(
            name="数据覆盖率",
            passed=passed,
            description=f"检查关键字段非空率是否>={THRESHOLDS.FUNDAMENTAL_COVERAGE_MIN*100}%",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"覆盖率检查完成: {'通过' if passed else '失败'}, 平均覆盖率={avg_coverage:.2%}")
        return result
    
    def check_pit_logic(self) -> CheckResult:
        """
        检查2: PIT逻辑验证
        确保满足 Report_Date < Announce_Date <= Trade_Date
        防止未来函数
        """
        logger.info("检查PIT逻辑...")
        
        df = self.df
        issues = []
        metrics = {}
        
        required_cols = ['trade_date', 'ann_date', 'report_date']
        available_cols = [c for c in required_cols if c in df.columns]
        
        if len(available_cols) < 3:
            # 可能没有report_date字段，尝试使用end_date
            if 'end_date' in df.columns:
                df = df.copy()
                df['report_date'] = pd.to_datetime(df['end_date'], errors='coerce')
                available_cols.append('report_date')
        
        if len(available_cols) < 3:
            return CheckResult(
                name="PIT逻辑验证",
                passed=True,
                description="验证 Report_Date < Announce_Date <= Trade_Date",
                issues=["缺少必要字段（report_date或end_date, ann_date, trade_date）进行PIT验证"],
                severity="WARNING"
            )
        
        # 过滤掉有财报数据的行（有ann_date的行）
        df_with_report = df.dropna(subset=['ann_date'])
        metrics['rows_with_ann_date'] = len(df_with_report)
        
        if len(df_with_report) == 0:
            return CheckResult(
                name="PIT逻辑验证",
                passed=True,
                description="验证 Report_Date < Announce_Date <= Trade_Date",
                issues=["所有行的ann_date为空，可能是日频ffill后的结果"],
                severity="INFO"
            )
        
        # 检查 Report_Date < Announce_Date
        if 'report_date' in df_with_report.columns:
            invalid_report_ann = df_with_report[
                df_with_report['report_date'] >= df_with_report['ann_date']
            ]
            metrics['invalid_report_ann_count'] = len(invalid_report_ann)
            
            if len(invalid_report_ann) > 0:
                issues.append(f"发现 {len(invalid_report_ann)} 行 report_date >= ann_date")
                sample = invalid_report_ann.head(5)[['trade_date', 'ts_code', 'report_date', 'ann_date']]
                metrics['invalid_report_ann_sample'] = sample.to_dict('records')
        
        # 检查 Announce_Date <= Trade_Date（核心的PIT检查）
        invalid_ann_trade = df_with_report[
            df_with_report['ann_date'] > df_with_report['trade_date']
        ]
        metrics['invalid_ann_trade_count'] = len(invalid_ann_trade)
        
        if len(invalid_ann_trade) > 0:
            issues.append(f"[CRITICAL] 发现 {len(invalid_ann_trade)} 行 ann_date > trade_date（未来函数！）")
            sample = invalid_ann_trade.head(10)[['trade_date', 'ts_code', 'ann_date']]
            metrics['invalid_ann_trade_sample'] = sample.to_dict('records')
        
        # 检查公告滞后天数分布（合理性检查）
        lag_days = (df_with_report['trade_date'] - df_with_report['ann_date']).dt.days
        metrics['announce_lag_stats'] = {
            'mean': float(lag_days.mean()),
            'median': float(lag_days.median()),
            'min': int(lag_days.min()),
            'max': int(lag_days.max()),
            'quantile_01': float(lag_days.quantile(0.01)),
            'quantile_99': float(lag_days.quantile(0.99)),
        }
        
        # 负的滞后天数是严重问题
        passed = metrics['invalid_ann_trade_count'] == 0
        severity = "CRITICAL" if not passed else "INFO"
        
        result = CheckResult(
            name="PIT逻辑验证",
            passed=passed,
            description="验证 Report_Date < Announce_Date <= Trade_Date，杜绝未来函数",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"PIT逻辑检查完成: {'通过' if passed else '失败'}")
        return result
    
    def check_ttm_logic(self) -> CheckResult:
        """
        检查3: TTM数值逻辑检查
        通过pct_change监控是否存在超过500%的剧烈突变
        """
        logger.info("检查TTM数值逻辑...")
        
        df = self.df
        issues = []
        metrics = {}
        extreme_changes = []
        
        # 找到所有TTM字段
        ttm_fields = [col for col in df.columns if 'ttm' in col.lower() or '_ttm' in col.lower()]
        metrics['ttm_fields_found'] = ttm_fields
        
        if not ttm_fields:
            return CheckResult(
                name="TTM数值逻辑",
                passed=True,
                description="检测TTM指标是否存在超过500%的突变",
                issues=["未找到TTM字段"],
                severity="WARNING"
            )
        
        # 对每个TTM字段检查突变
        df_sorted = df.sort_values(['ts_code', 'trade_date'])
        
        for field in ttm_fields[:5]:  # 限制检查前5个TTM字段
            if field not in df_sorted.columns:
                continue
            
            # 计算按股票分组的变化率
            pct_change = df_sorted.groupby('ts_code')[field].pct_change()
            
            # 找出超过阈值的突变
            extreme = pct_change.abs() > THRESHOLDS.TTM_CHANGE_EXTREME
            extreme_count = extreme.sum()
            
            field_metrics = {
                'field': field,
                'extreme_change_count': int(extreme_count),
                'extreme_rate': float(extreme_count / len(pct_change)) if len(pct_change) > 0 else 0,
            }
            
            if extreme_count > 0:
                # 获取极端变化的样本
                extreme_idx = pct_change[extreme].index[:10]
                sample = df_sorted.loc[extreme_idx, ['trade_date', 'ts_code', field]]
                sample = sample.copy()
                sample['pct_change'] = pct_change.loc[extreme_idx]
                extreme_changes.append({
                    'field': field,
                    'count': int(extreme_count),
                    'sample': sample.to_dict('records')
                })
            
            metrics[f'{field}_stats'] = field_metrics
        
        # 汇总
        total_extreme = sum(e['count'] for e in extreme_changes)
        metrics['total_extreme_changes'] = total_extreme
        metrics['extreme_changes_detail'] = extreme_changes
        
        if total_extreme > 0:
            issues.append(f"发现 {total_extreme} 处TTM指标突变超过{THRESHOLDS.TTM_CHANGE_EXTREME*100}%")
        
        # 判定（允许一定比例的突变，可能是业绩拐点）
        total_rows = len(df) * len(ttm_fields)
        extreme_rate = total_extreme / total_rows if total_rows > 0 else 0
        passed = extreme_rate < 0.01  # 允许1%的突变
        
        severity = "WARNING" if total_extreme > 0 else "INFO"
        
        result = CheckResult(
            name="TTM数值逻辑",
            passed=passed,
            description=f"检测TTM指标是否存在超过{THRESHOLDS.TTM_CHANGE_EXTREME*100}%的突变",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"TTM逻辑检查完成: {'通过' if passed else '告警'}")
        return result
    
    def check_valuation_consistency(self) -> CheckResult:
        """
        检查4: 估值指标一致性验证
        验证 Total_MV / Net_Income_TTM ≈ PE_TTM
        """
        logger.info("检查估值指标一致性...")
        
        df = self.df
        issues = []
        metrics = {}
        
        # 需要的字段
        required_fields = ['total_mv', 'pe_ttm']
        income_fields = ['n_income_attr_p_ttm', 'net_income_ttm', 'n_income_ttm']
        
        # 找到可用的净利润字段
        net_income_field = None
        for f in income_fields:
            if f in df.columns:
                net_income_field = f
                break
        
        if not all(f in df.columns for f in required_fields) or net_income_field is None:
            return CheckResult(
                name="估值指标一致性",
                passed=True,
                description="验证 PE_TTM ≈ Total_MV / Net_Income_TTM",
                issues=[f"缺少必要字段进行估值一致性验证（需要total_mv, pe_ttm, {income_fields}）"],
                severity="WARNING"
            )
        
        # 选取有效数据
        df_valid = df.dropna(subset=['total_mv', 'pe_ttm', net_income_field])
        df_valid = df_valid[(df_valid[net_income_field] != 0) & (df_valid['total_mv'] > 0)]
        
        metrics['valid_rows_for_check'] = len(df_valid)
        
        if len(df_valid) == 0:
            return CheckResult(
                name="估值指标一致性",
                passed=True,
                description="验证 PE_TTM ≈ Total_MV / Net_Income_TTM",
                issues=["没有有效数据进行估值一致性验证"],
                severity="WARNING"
            )
        
        # 计算PE并对比
        # 注意：total_mv单位可能是万元，需要根据实际情况调整
        # PE = 市值 / 净利润
        calc_pe = df_valid['total_mv'] / df_valid[net_income_field]
        existing_pe = df_valid['pe_ttm']
        
        # 计算相对误差
        # 排除极端值（PE可能为负或非常大）
        valid_mask = (existing_pe.abs() > 0.1) & (existing_pe.abs() < 10000)
        calc_pe_valid = calc_pe[valid_mask]
        existing_pe_valid = existing_pe[valid_mask]
        
        if len(calc_pe_valid) > 0:
            relative_error = ((calc_pe_valid - existing_pe_valid) / existing_pe_valid.abs()).abs()
            
            # 统计误差
            metrics['valid_comparisons'] = len(relative_error)
            metrics['error_stats'] = {
                'mean': float(relative_error.mean()),
                'median': float(relative_error.median()),
                'std': float(relative_error.std()),
                'max': float(relative_error.max()),
                'quantile_90': float(relative_error.quantile(0.90)),
            }
            
            # 找出误差过大的
            tolerance = THRESHOLDS.PE_TTM_TOLERANCE
            large_error = relative_error > tolerance
            large_error_count = large_error.sum()
            metrics['large_error_count'] = int(large_error_count)
            metrics['large_error_rate'] = float(large_error_count / len(relative_error))
            
            if large_error_count > 0:
                issues.append(f"发现 {large_error_count} 行PE_TTM与计算值误差超过{tolerance*100}%")
                # 采样
                sample_idx = relative_error[large_error].nlargest(10).index
                sample = df_valid.loc[sample_idx, ['trade_date', 'ts_code', 'total_mv', net_income_field, 'pe_ttm']]
                sample = sample.copy()
                sample['calc_pe'] = calc_pe.loc[sample_idx]
                sample['error'] = relative_error.loc[sample_idx]
                metrics['large_error_sample'] = sample.to_dict('records')
            
            # 判定
            passed = metrics['large_error_rate'] < 0.1  # 允许10%的不一致
        else:
            passed = True
            issues.append("没有足够的有效数据进行PE一致性验证")
        
        severity = "WARNING" if not passed else "INFO"
        
        result = CheckResult(
            name="估值指标一致性",
            passed=passed,
            description="验证 PE_TTM ≈ Total_MV / Net_Income_TTM",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"估值一致性检查完成: {'通过' if passed else '告警'}")
        return result
    
    def run_all_checks(self) -> List[CheckResult]:
        """运行所有检查"""
        logger.info("=" * 60)
        logger.info("开始PIT基本面宽表数据质量检查")
        logger.info("=" * 60)
        
        self.check_data_coverage()
        self.check_pit_logic()
        self.check_ttm_logic()
        self.check_valuation_consistency()
        
        passed_count = sum(1 for r in self.results if r.passed)
        logger.info(f"检查完成: {passed_count}/{len(self.results)} 项通过")
        
        return self.results


def main():
    """独立运行测试"""
    checker = FundamentalTableChecker()
    
    # 获取概览
    summary = checker.get_summary()
    print(f"\n数据概览:")
    print(f"  - 行数: {summary.rows:,}")
    print(f"  - 股票数: {summary.stock_count:,}")
    print(f"  - 日期范围: {summary.date_range[0]} ~ {summary.date_range[1]}")
    print(f"  - 文件大小: {summary.file_size_mb:.2f} MB")
    
    # 运行检查
    results = checker.run_all_checks()
    
    # 打印结果
    print(f"\n检查结果:")
    for r in results:
        status = "✓" if r.passed else "✗"
        print(f"  {status} {r.name}: {r.severity}")
        for issue in r.issues:
            print(f"      - {issue}")
    
    return results


if __name__ == "__main__":
    main()
