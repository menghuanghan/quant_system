"""
DWD 数据质量检查 - 量价宽表检查器 (dwd_stock_price)

检查内容：
1. OHLC价格逻辑验证
2. 收益率一致性检查
3. 极值与异常检测
4. 停牌与时间连续性检查
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


class PriceTableChecker:
    """量价宽表检查器"""
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Args:
            df: 可选的DataFrame，如果不提供则从文件加载
        """
        if df is not None:
            self.df = df
        else:
            logger.info(f"加载量价宽表: {DWD_PATHS.stock_price}")
            self.df = pd.read_parquet(DWD_PATHS.stock_price)
        
        self.results: List[CheckResult] = []
        self._prepare_data()
    
    def _prepare_data(self):
        """预处理数据"""
        # 确保日期格式统一
        if 'trade_date' in self.df.columns:
            self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
        
        # 计算辅助字段
        if 'close_hfq' in self.df.columns and 'pre_close' in self.df.columns:
            # 这里我们需要 pre_close_hfq 来计算验证收益率
            self.df = self.df.sort_values(['ts_code', 'trade_date'])
            self.df['pre_close_hfq_calc'] = self.df.groupby('ts_code')['close_hfq'].shift(1)
    
    def get_summary(self) -> TableSummary:
        """获取数据表概览"""
        df = self.df
        
        # 计算缺失率
        null_summary = {}
        for col in df.columns:
            null_summary[col] = calculate_missing_rate(df[col])
        
        return TableSummary(
            name="dwd_stock_price",
            rows=len(df),
            columns=len(df.columns),
            date_range=(
                df['trade_date'].min().strftime('%Y-%m-%d'),
                df['trade_date'].max().strftime('%Y-%m-%d')
            ),
            stock_count=df['ts_code'].nunique(),
            file_size_mb=get_file_size_mb(DWD_PATHS.stock_price),
            memory_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            null_summary=null_summary
        )
    
    def check_ohlc_logic(self) -> CheckResult:
        """
        检查1: OHLC价格逻辑验证
        - High应为当日最高价 (High >= Open, High >= Close, High >= Low)
        - Low应为当日最低价 (Low <= Open, Low <= Close, Low <= High)
        - Close_hfq应全为正值
        """
        logger.info("检查 OHLC 价格逻辑...")
        
        df = self.df
        issues = []
        metrics = {}
        
        # 检查原始OHLC
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High >= Open, Close, Low
            invalid_high = df[(df['high'] < df['open']) | 
                             (df['high'] < df['close']) | 
                             (df['high'] < df['low'])]
            
            # Low <= Open, Close, High
            invalid_low = df[(df['low'] > df['open']) | 
                            (df['low'] > df['close']) | 
                            (df['low'] > df['high'])]
            
            metrics['invalid_high_count'] = len(invalid_high)
            metrics['invalid_low_count'] = len(invalid_low)
            
            if len(invalid_high) > 0:
                issues.append(f"发现 {len(invalid_high)} 行 High 不是最高价")
                sample = invalid_high.head(5)[['trade_date', 'ts_code', 'open', 'high', 'low', 'close']]
                metrics['invalid_high_sample'] = sample.to_dict('records')
            
            if len(invalid_low) > 0:
                issues.append(f"发现 {len(invalid_low)} 行 Low 不是最低价")
                sample = invalid_low.head(5)[['trade_date', 'ts_code', 'open', 'high', 'low', 'close']]
                metrics['invalid_low_sample'] = sample.to_dict('records')
        
        # 检查后复权OHLC
        if all(col in df.columns for col in ['open_hfq', 'high_hfq', 'low_hfq', 'close_hfq']):
            invalid_hfq_high = df[(df['high_hfq'] < df['open_hfq']) | 
                                  (df['high_hfq'] < df['close_hfq']) | 
                                  (df['high_hfq'] < df['low_hfq'])]
            
            invalid_hfq_low = df[(df['low_hfq'] > df['open_hfq']) | 
                                 (df['low_hfq'] > df['close_hfq']) | 
                                 (df['low_hfq'] > df['high_hfq'])]
            
            metrics['invalid_hfq_high_count'] = len(invalid_hfq_high)
            metrics['invalid_hfq_low_count'] = len(invalid_hfq_low)
            
            if len(invalid_hfq_high) > 0:
                issues.append(f"发现 {len(invalid_hfq_high)} 行 High_hfq 异常")
            
            if len(invalid_hfq_low) > 0:
                issues.append(f"发现 {len(invalid_hfq_low)} 行 Low_hfq 异常")
        
        # 检查 close_hfq 正值
        if 'close_hfq' in df.columns:
            negative_close_hfq = df[df['close_hfq'] <= 0]
            metrics['negative_close_hfq_count'] = len(negative_close_hfq)
            
            if len(negative_close_hfq) > 0:
                issues.append(f"发现 {len(negative_close_hfq)} 行 close_hfq <= 0")
                sample = negative_close_hfq.head(5)[['trade_date', 'ts_code', 'close_hfq', 'adj_factor']]
                metrics['negative_close_hfq_sample'] = sample.to_dict('records')
        
        # 综合判定
        total_invalid = (metrics.get('invalid_high_count', 0) + 
                        metrics.get('invalid_low_count', 0) + 
                        metrics.get('negative_close_hfq_count', 0))
        
        passed = total_invalid == 0
        severity = "ERROR" if not passed else "INFO"
        
        result = CheckResult(
            name="OHLC价格逻辑",
            passed=passed,
            description="验证High为最高价、Low为最低价、Close_hfq为正值",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"OHLC检查完成: {'通过' if passed else '失败'}")
        return result
    
    def check_return_consistency(self) -> CheckResult:
        """
        检查2: 收益率一致性检查
        计算 (Close_hfq - Pre_Close_hfq) / Pre_Close_hfq 并与现有 return_1d 对比
        误差需小于 1e-6
        """
        logger.info("检查收益率一致性...")
        
        df = self.df
        issues = []
        metrics = {}
        
        required_cols = ['close_hfq', 'return_1d', 'pre_close_hfq_calc']
        if not all(col in df.columns for col in required_cols):
            return CheckResult(
                name="收益率一致性",
                passed=False,
                description="检查return_1d与复权价格计算的一致性",
                issues=["缺少必要字段进行验证"],
                severity="WARNING"
            )
        
        # 计算验证收益率
        df_valid = df.dropna(subset=['close_hfq', 'pre_close_hfq_calc'])
        df_valid = df_valid[df_valid['pre_close_hfq_calc'] > 0]  # 避免除零
        
        calc_return = (df_valid['close_hfq'] - df_valid['pre_close_hfq_calc']) / df_valid['pre_close_hfq_calc']
        existing_return = df_valid['return_1d']
        
        # 计算误差
        diff = (calc_return - existing_return).abs()
        tolerance = THRESHOLDS.RETURN_TOLERANCE
        
        inconsistent = diff > tolerance
        inconsistent_count = inconsistent.sum()
        
        metrics['total_checked'] = len(df_valid)
        metrics['inconsistent_count'] = int(inconsistent_count)
        metrics['inconsistent_rate'] = float(inconsistent_count / len(df_valid)) if len(df_valid) > 0 else 0
        metrics['max_diff'] = float(diff.max()) if len(diff) > 0 else 0
        metrics['mean_diff'] = float(diff.mean()) if len(diff) > 0 else 0
        
        if inconsistent_count > 0:
            issues.append(f"发现 {inconsistent_count} 行收益率计算不一致（容差={tolerance}）")
            # 采样
            sample_idx = diff[inconsistent].nlargest(10).index
            sample_df = df_valid.loc[sample_idx, ['trade_date', 'ts_code', 'close_hfq', 'pre_close_hfq_calc', 'return_1d']]
            sample_df = sample_df.copy()
            sample_df['calc_return'] = calc_return.loc[sample_idx]
            sample_df['diff'] = diff.loc[sample_idx]
            metrics['inconsistent_sample'] = sample_df.head(10).to_dict('records')
        
        # 允许少量误差（可能是浮点精度问题）
        passed = metrics['inconsistent_rate'] < 0.001  # 允许0.1%的不一致
        severity = "ERROR" if not passed else ("WARNING" if inconsistent_count > 0 else "INFO")
        
        result = CheckResult(
            name="收益率一致性",
            passed=passed,
            description="验证return_1d = (close_hfq - pre_close_hfq) / pre_close_hfq",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"收益率一致性检查完成: {'通过' if passed else '失败'}")
        return result
    
    def check_extreme_values(self) -> CheckResult:
        """
        检查3: 极值与异常检测
        - 非首日涨跌幅超过200%或低于-30%
        - 非新股return绝对值超过0.3
        """
        logger.info("检查极值与异常...")
        
        df = self.df
        issues = []
        metrics = {}
        extreme_samples = []
        
        if 'return_1d' not in df.columns:
            return CheckResult(
                name="极值与异常检测",
                passed=False,
                description="检测异常涨跌幅",
                issues=["缺少return_1d字段"],
                severity="WARNING"
            )
        
        # 识别首日（每只股票的第一条记录）
        df = df.sort_values(['ts_code', 'trade_date'])
        df['is_first_day'] = ~df.duplicated(subset=['ts_code'], keep='first')
        
        # 非首日数据
        df_not_first = df[~df['is_first_day']]
        
        # 检查非首日涨幅 > 200%
        extreme_up = df_not_first[df_not_first['return_1d'] > THRESHOLDS.RETURN_EXTREME_UPPER]
        metrics['extreme_up_count'] = len(extreme_up)
        
        if len(extreme_up) > 0:
            issues.append(f"发现 {len(extreme_up)} 行非首日涨幅超过 {THRESHOLDS.RETURN_EXTREME_UPPER*100}%")
            sample = extreme_up.nlargest(10, 'return_1d')[['trade_date', 'ts_code', 'return_1d', 'close', 'pre_close']]
            extreme_samples.extend(sample.to_dict('records'))
        
        # 检查非首日跌幅 < -30%
        extreme_down = df_not_first[df_not_first['return_1d'] < THRESHOLDS.RETURN_EXTREME_LOWER]
        metrics['extreme_down_count'] = len(extreme_down)
        
        if len(extreme_down) > 0:
            issues.append(f"发现 {len(extreme_down)} 行非首日跌幅超过 {abs(THRESHOLDS.RETURN_EXTREME_LOWER)*100}%")
            sample = extreme_down.nsmallest(10, 'return_1d')[['trade_date', 'ts_code', 'return_1d', 'close', 'pre_close']]
            extreme_samples.extend(sample.to_dict('records'))
        
        # 检查非新股的大幅波动（这里假设新股为上市60天内，但需要list_date信息）
        # 简化处理：检查所有超过30%的涨跌幅
        high_volatility = df_not_first[df_not_first['return_1d'].abs() > THRESHOLDS.RETURN_NORMAL_THRESHOLD]
        metrics['high_volatility_count'] = len(high_volatility)
        metrics['high_volatility_rate'] = len(high_volatility) / len(df_not_first) if len(df_not_first) > 0 else 0
        
        # 收益率分布统计
        metrics['return_stats'] = {
            'mean': float(df['return_1d'].mean()),
            'std': float(df['return_1d'].std()),
            'min': float(df['return_1d'].min()),
            'max': float(df['return_1d'].max()),
            'quantile_01': float(df['return_1d'].quantile(0.01)),
            'quantile_99': float(df['return_1d'].quantile(0.99)),
        }
        
        if extreme_samples:
            metrics['extreme_samples'] = extreme_samples[:20]
        
        # 判定：极端异常应该非常少（可能是北交所、新股等特殊情况）
        total_extreme = metrics['extreme_up_count'] + metrics['extreme_down_count']
        extreme_rate = total_extreme / len(df_not_first) if len(df_not_first) > 0 else 0
        
        # 这里设定一个合理的阈值，允许一定比例的极端值（北交所、新股等）
        passed = extreme_rate < 0.01  # 允许1%的极端值
        severity = "WARNING" if not passed else "INFO"
        
        result = CheckResult(
            name="极值与异常检测",
            passed=passed,
            description=f"检测非首日涨幅>{THRESHOLDS.RETURN_EXTREME_UPPER*100}%或跌幅<{THRESHOLDS.RETURN_EXTREME_LOWER*100}%",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"极值检查完成: {'通过' if passed else '告警'}")
        return result
    
    def check_suspension_and_continuity(self) -> CheckResult:
        """
        检查4: 停牌与时间连续性检查
        - is_trading=0时成交量应为0且收益率为0（或NaN）
        - 对比交易所日历，检查是否有缺失
        """
        logger.info("检查停牌与时间连续性...")
        
        df = self.df
        issues = []
        metrics = {}
        
        # 检查停牌日逻辑
        if 'is_trading' in df.columns and 'vol' in df.columns:
            suspended = df[df['is_trading'] == 0]
            metrics['suspended_count'] = len(suspended)
            
            # 停牌日应该成交量为0
            suspended_with_vol = suspended[suspended['vol'] > 0]
            metrics['suspended_with_vol_count'] = len(suspended_with_vol)
            
            if len(suspended_with_vol) > 0:
                issues.append(f"发现 {len(suspended_with_vol)} 行停牌日有成交量")
                sample = suspended_with_vol.head(5)[['trade_date', 'ts_code', 'is_trading', 'vol']]
                metrics['suspended_with_vol_sample'] = sample.to_dict('records')
            
            # 停牌日的收益率应该为0
            if 'return_1d' in df.columns:
                suspended_with_return = suspended[suspended['return_1d'] != 0]
                metrics['suspended_with_return_count'] = len(suspended_with_return)
                
                if len(suspended_with_return) > 0:
                    issues.append(f"发现 {len(suspended_with_return)} 行停牌日收益率非0")
        
        # 检查时间连续性（按股票检查是否有日期缺失）
        # 加载交易日历
        try:
            trade_cal = pd.read_parquet(DWD_PATHS.trade_calendar)
            trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'])
            
            # 只保留开市日
            if 'is_open' in trade_cal.columns:
                trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].unique()
            else:
                trade_dates = trade_cal['cal_date'].unique()
            
            # 过滤到数据日期范围内
            date_min, date_max = df['trade_date'].min(), df['trade_date'].max()
            trade_dates = pd.Series(trade_dates)
            trade_dates = trade_dates[(trade_dates >= date_min) & (trade_dates <= date_max)]
            
            expected_trade_days = len(trade_dates)
            metrics['expected_trade_days'] = expected_trade_days
            
            # 统计每只股票的记录数
            stock_counts = df.groupby('ts_code').size()
            metrics['stock_count'] = len(stock_counts)
            metrics['avg_records_per_stock'] = float(stock_counts.mean())
            metrics['min_records_per_stock'] = int(stock_counts.min())
            metrics['max_records_per_stock'] = int(stock_counts.max())
            
            # 检查记录数明显不足的股票（低于预期的80%）
            threshold = expected_trade_days * 0.8
            incomplete_stocks = stock_counts[stock_counts < threshold]
            metrics['incomplete_stocks_count'] = len(incomplete_stocks)
            
            # 这些不一定是问题（可能是中途上市或退市的股票）
            if len(incomplete_stocks) > 0:
                sample = incomplete_stocks.head(10).to_dict()
                metrics['incomplete_stocks_sample'] = sample
            
        except Exception as e:
            issues.append(f"无法加载交易日历进行连续性检查: {str(e)}")
            metrics['calendar_check_error'] = str(e)
        
        # 综合判定
        problem_count = (metrics.get('suspended_with_vol_count', 0) + 
                        metrics.get('suspended_with_return_count', 0))
        
        passed = problem_count == 0
        severity = "ERROR" if problem_count > 0 else "INFO"
        
        result = CheckResult(
            name="停牌与时间连续性",
            passed=passed,
            description="验证停牌日无成交且收益率为0，检查时间序列完整性",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"停牌与连续性检查完成: {'通过' if passed else '失败'}")
        return result
    
    def run_all_checks(self) -> List[CheckResult]:
        """运行所有检查"""
        logger.info("=" * 60)
        logger.info("开始量价宽表数据质量检查")
        logger.info("=" * 60)
        
        self.check_ohlc_logic()
        self.check_return_consistency()
        self.check_extreme_values()
        self.check_suspension_and_continuity()
        
        passed_count = sum(1 for r in self.results if r.passed)
        logger.info(f"检查完成: {passed_count}/{len(self.results)} 项通过")
        
        return self.results


def main():
    """独立运行测试"""
    checker = PriceTableChecker()
    
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
