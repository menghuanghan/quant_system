"""
DWD 数据质量检查 - 状态与风险掩码表检查器 (dwd_stock_status)

检查内容：
1. 涨跌停标记验证 (is_limit_up/is_limit_down)
2. ST状态与新股标记检查
3. 涨跌停阈值按板块核对
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
    get_file_size_mb, calculate_missing_rate, get_market_from_ts_code,
)

logger = setup_logging(__name__)


class StatusTableChecker:
    """状态与风险掩码表检查器"""
    
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        price_df: Optional[pd.DataFrame] = None
    ):
        """
        Args:
            df: 状态表DataFrame，如果不提供则从文件加载
            price_df: 量价表DataFrame，用于交叉验证
        """
        if df is not None:
            self.df = df
        else:
            logger.info(f"加载状态表: {DWD_PATHS.stock_status}")
            self.df = pd.read_parquet(DWD_PATHS.stock_status)
        
        self.price_df = price_df
        self.results: List[CheckResult] = []
        self._prepare_data()
    
    def _prepare_data(self):
        """预处理数据"""
        # 确保日期格式统一
        if 'trade_date' in self.df.columns:
            self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
        if 'list_date' in self.df.columns:
            self.df['list_date'] = pd.to_datetime(self.df['list_date'])
        
        # 如果需要但没有price_df，尝试加载
        if self.price_df is None and DWD_PATHS.stock_price.exists():
            try:
                logger.info("加载量价表用于交叉验证...")
                self.price_df = pd.read_parquet(
                    DWD_PATHS.stock_price,
                    columns=['trade_date', 'ts_code', 'close', 'pre_close', 'return_1d']
                )
                self.price_df['trade_date'] = pd.to_datetime(self.price_df['trade_date'])
            except Exception as e:
                logger.warning(f"无法加载量价表: {e}")
    
    def get_summary(self) -> TableSummary:
        """获取数据表概览"""
        df = self.df
        
        # 计算缺失率
        null_summary = {}
        for col in df.columns:
            null_summary[col] = calculate_missing_rate(df[col])
        
        return TableSummary(
            name="dwd_stock_status",
            rows=len(df),
            columns=len(df.columns),
            date_range=(
                df['trade_date'].min().strftime('%Y-%m-%d'),
                df['trade_date'].max().strftime('%Y-%m-%d')
            ),
            stock_count=df['ts_code'].nunique(),
            file_size_mb=get_file_size_mb(DWD_PATHS.stock_status),
            memory_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            null_summary=null_summary
        )
    
    def check_limit_up_down(self) -> CheckResult:
        """
        检查1: 涨跌停标记验证
        验证 is_limit_up=1 时实际收益率是否接近 limit_ratio
        """
        logger.info("检查涨跌停标记...")
        
        df = self.df
        issues = []
        metrics = {}
        
        # 统计涨跌停数量
        if 'is_limit_up' in df.columns:
            limit_up_count = df['is_limit_up'].sum()
            metrics['is_limit_up_count'] = int(limit_up_count)
        else:
            metrics['is_limit_up_count'] = 0
        
        if 'is_limit_down' in df.columns:
            limit_down_count = df['is_limit_down'].sum()
            metrics['is_limit_down_count'] = int(limit_down_count)
        else:
            metrics['is_limit_down_count'] = 0
        
        # 如果有price_df，进行交叉验证
        if self.price_df is not None and 'limit_ratio' in df.columns:
            # 合并数据
            merged = df.merge(
                self.price_df[['trade_date', 'ts_code', 'return_1d']],
                on=['trade_date', 'ts_code'],
                how='left'
            )
            metrics['merged_rows'] = len(merged)
            
            # 检查涨停标记
            if 'is_limit_up' in merged.columns:
                limit_up = merged[merged['is_limit_up'] == 1]
                if len(limit_up) > 0:
                    # 计算实际收益率与limit_ratio的差距
                    expected_return = limit_up['limit_ratio']
                    actual_return = limit_up['return_1d']
                    
                    # 允许一定的滑点
                    tolerance = THRESHOLDS.LIMIT_UP_TOLERANCE
                    
                    # 涨停时收益率应该 >= limit_ratio - tolerance
                    invalid_limit_up = limit_up[actual_return < (expected_return - tolerance)]
                    metrics['invalid_limit_up_count'] = len(invalid_limit_up)
                    
                    if len(invalid_limit_up) > 0:
                        issues.append(f"发现 {len(invalid_limit_up)} 行标记涨停但实际收益率不符")
                        sample = invalid_limit_up.head(10)[['trade_date', 'ts_code', 'limit_ratio', 'return_1d']]
                        metrics['invalid_limit_up_sample'] = sample.to_dict('records')
                    
                    # 统计涨停时的收益率分布
                    metrics['limit_up_return_stats'] = {
                        'mean': float(actual_return.mean()),
                        'min': float(actual_return.min()),
                        'max': float(actual_return.max()),
                    }
            
            # 检查跌停标记
            if 'is_limit_down' in merged.columns:
                limit_down = merged[merged['is_limit_down'] == 1]
                if len(limit_down) > 0:
                    expected_return = -limit_down['limit_ratio']
                    actual_return = limit_down['return_1d']
                    
                    # 跌停时收益率应该 <= -limit_ratio + tolerance
                    invalid_limit_down = limit_down[actual_return > (expected_return + tolerance)]
                    metrics['invalid_limit_down_count'] = len(invalid_limit_down)
                    
                    if len(invalid_limit_down) > 0:
                        issues.append(f"发现 {len(invalid_limit_down)} 行标记跌停但实际收益率不符")
                        sample = invalid_limit_down.head(10)[['trade_date', 'ts_code', 'limit_ratio', 'return_1d']]
                        metrics['invalid_limit_down_sample'] = sample.to_dict('records')
                    
                    metrics['limit_down_return_stats'] = {
                        'mean': float(actual_return.mean()),
                        'min': float(actual_return.min()),
                        'max': float(actual_return.max()),
                    }
        else:
            issues.append("缺少price_df或limit_ratio字段，无法进行涨跌停交叉验证")
        
        # 综合判定
        invalid_count = metrics.get('invalid_limit_up_count', 0) + metrics.get('invalid_limit_down_count', 0)
        total_limit = metrics.get('is_limit_up_count', 0) + metrics.get('is_limit_down_count', 0)
        invalid_rate = invalid_count / total_limit if total_limit > 0 else 0
        
        # 允许少量不一致（可能是数据源差异、复权因子等）
        passed = invalid_rate < 0.05  # 允许5%的不一致
        severity = "WARNING" if not passed else "INFO"
        
        result = CheckResult(
            name="涨跌停标记验证",
            passed=passed,
            description="验证涨跌停标记与实际收益率的一致性",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"涨跌停检查完成: {'通过' if passed else '告警'}")
        return result
    
    def check_st_and_new_stock(self) -> CheckResult:
        """
        检查2: ST状态与新股标记检查
        - 确认is_new=1的记录严格限制在上市60天内
        - ST标记逻辑性检查
        """
        logger.info("检查ST状态与新股标记...")
        
        df = self.df
        issues = []
        metrics = {}
        
        # ST状态统计
        if 'is_st' in df.columns:
            st_count = df['is_st'].sum()
            st_stocks = df[df['is_st'] == 1]['ts_code'].nunique()
            metrics['is_st_count'] = int(st_count)
            metrics['st_stocks'] = int(st_stocks)
            
            # ST比例（ST交易日/总交易日）
            st_rate = st_count / len(df) if len(df) > 0 else 0
            metrics['st_rate'] = st_rate
            
            # 合理性检查：ST比例不应过高或过低
            if st_rate > 0.2:
                issues.append(f"ST比例异常高: {st_rate:.2%}")
            elif st_rate < 0.001 and len(df) > 10000:
                issues.append(f"ST比例异常低: {st_rate:.2%}，可能ST状态数据缺失")
        
        # 新股标记检查
        if 'is_new' in df.columns and 'list_date' in df.columns:
            new_stocks = df[df['is_new'] == 1]
            metrics['is_new_count'] = len(new_stocks)
            
            if len(new_stocks) > 0:
                # 计算上市天数
                days_since_listing = (new_stocks['trade_date'] - new_stocks['list_date']).dt.days
                
                # 检查是否都在60天内
                invalid_new = days_since_listing > THRESHOLDS.NEW_STOCK_DAYS
                invalid_new_count = invalid_new.sum()
                metrics['invalid_new_stock_count'] = int(invalid_new_count)
                
                if invalid_new_count > 0:
                    issues.append(f"发现 {invalid_new_count} 行 is_new=1 但上市已超过{THRESHOLDS.NEW_STOCK_DAYS}天")
                    sample = new_stocks[invalid_new].head(10)[['trade_date', 'ts_code', 'list_date']]
                    sample = sample.copy()
                    sample['days_since_listing'] = days_since_listing[invalid_new].head(10)
                    metrics['invalid_new_sample'] = sample.to_dict('records')
                
                # 新股天数分布
                metrics['new_stock_days_stats'] = {
                    'mean': float(days_since_listing.mean()),
                    'min': int(days_since_listing.min()),
                    'max': int(days_since_listing.max()),
                }
        elif 'is_new' in df.columns:
            metrics['is_new_count'] = int(df['is_new'].sum())
            if 'list_date' not in df.columns:
                issues.append("缺少list_date字段，无法验证新股天数")
        
        # 新股无涨跌停限制期检查
        if 'is_new_no_limit' in df.columns:
            no_limit_count = df['is_new_no_limit'].sum()
            metrics['is_new_no_limit_count'] = int(no_limit_count)
            
            # is_new_no_limit应该是is_new的子集
            if 'is_new' in df.columns:
                invalid_no_limit = df[(df['is_new_no_limit'] == 1) & (df['is_new'] == 0)]
                if len(invalid_no_limit) > 0:
                    issues.append(f"发现 {len(invalid_no_limit)} 行 is_new_no_limit=1 但 is_new=0")
        
        # 综合判定
        invalid_count = metrics.get('invalid_new_stock_count', 0)
        passed = invalid_count == 0
        severity = "ERROR" if not passed else "INFO"
        
        result = CheckResult(
            name="ST状态与新股标记",
            passed=passed,
            description=f"验证is_new=1限制在上市{THRESHOLDS.NEW_STOCK_DAYS}天内，ST标记合理性",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"ST与新股检查完成: {'通过' if passed else '失败'}")
        return result
    
    def check_limit_ratio_by_market(self) -> CheckResult:
        """
        检查3: 按板块核对涨跌停阈值
        创业板/科创板应为0.20，北交所应为0.30
        """
        logger.info("检查涨跌停阈值按板块分布...")
        
        df = self.df
        issues = []
        metrics = {}
        
        if 'limit_ratio' not in df.columns:
            return CheckResult(
                name="涨跌停阈值板块核对",
                passed=True,
                description="验证不同板块的涨跌停阈值设置",
                issues=["缺少limit_ratio字段"],
                severity="WARNING"
            )
        
        if 'market' not in df.columns:
            # 尝试从ts_code推断市场
            df = df.copy()
            df['market_inferred'] = df['ts_code'].apply(get_market_from_ts_code)
            market_col = 'market_inferred'
        else:
            market_col = 'market'
        
        # 按市场统计limit_ratio
        market_limit_stats = df.groupby(market_col)['limit_ratio'].agg(['mean', 'min', 'max', 'count'])
        metrics['market_limit_stats'] = market_limit_stats.to_dict()
        
        # 验证各板块的limit_ratio
        expected_ratios = THRESHOLDS.LIMIT_RATIOS
        
        for market, expected in expected_ratios.items():
            market_data = df[df[market_col] == market]
            if len(market_data) == 0:
                continue
            
            # 检查非ST、非新股的limit_ratio
            normal_data = market_data
            if 'is_st' in df.columns:
                normal_data = normal_data[normal_data['is_st'] == 0]
            if 'is_new_no_limit' in df.columns:
                normal_data = normal_data[normal_data['is_new_no_limit'] == 0]
            
            if len(normal_data) > 0:
                actual_ratio = normal_data['limit_ratio'].mode()
                if len(actual_ratio) > 0:
                    actual = actual_ratio.iloc[0]
                    if abs(actual - expected) > 0.001:
                        issues.append(f"{market}板块: limit_ratio={actual}，期望={expected}")
                    
                    metrics[f'{market}_ratio'] = {
                        'expected': expected,
                        'actual_mode': float(actual),
                        'count': len(normal_data),
                    }
        
        # 检查ST股票的limit_ratio
        if 'is_st' in df.columns:
            st_data = df[df['is_st'] == 1]
            if len(st_data) > 0:
                # 使用mean和median代替mode（pandas groupby不直接支持mode）
                st_ratios = st_data.groupby(market_col)['limit_ratio'].agg(['mean', 'median', 'count'])
                metrics['st_limit_ratios'] = st_ratios.to_dict() if hasattr(st_ratios, 'to_dict') else {}
                
                # 主板ST应该是5%
                main_st = st_data[st_data[market_col].isin(['主板', '中小板'])]
                if len(main_st) > 0:
                    main_st_ratio = main_st['limit_ratio'].mode()
                    if len(main_st_ratio) > 0 and abs(main_st_ratio.iloc[0] - 0.05) > 0.001:
                        issues.append(f"主板ST股票limit_ratio={main_st_ratio.iloc[0]}，期望=0.05")
        
        # 综合判定
        passed = len(issues) == 0
        severity = "WARNING" if not passed else "INFO"
        
        result = CheckResult(
            name="涨跌停阈值板块核对",
            passed=passed,
            description="验证创业板/科创板20%，北交所30%，主板10%/ST 5%",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"涨跌停阈值检查完成: {'通过' if passed else '告警'}")
        return result
    
    def run_all_checks(self) -> List[CheckResult]:
        """运行所有检查"""
        logger.info("=" * 60)
        logger.info("开始状态与风险掩码表数据质量检查")
        logger.info("=" * 60)
        
        self.check_limit_up_down()
        self.check_st_and_new_stock()
        self.check_limit_ratio_by_market()
        
        passed_count = sum(1 for r in self.results if r.passed)
        logger.info(f"检查完成: {passed_count}/{len(self.results)} 项通过")
        
        return self.results


def main():
    """独立运行测试"""
    checker = StatusTableChecker()
    
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
