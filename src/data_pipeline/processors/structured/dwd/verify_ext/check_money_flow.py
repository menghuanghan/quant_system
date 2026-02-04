"""
DWD 扩展表数据质量检查 - 资金博弈宽表检查器 (dwd_money_flow)

检查内容：
1. 主键唯一性 (trade_date, ts_code)
2. 缺失值闭环 (训练期关键字段 NaN 为 0)
3. 市场广播一致性 (hsgt_north 同日所有股票相同)
4. 北交所特例 (is_bj_stock=1 时允许部分字段为 0)
5. 极值异常 (无 inf，数值在合理量级)
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
    get_file_size_mb, calculate_missing_rate, get_market_from_ts_code, is_training_period,
)

logger = setup_logging(__name__)


class MoneyFlowChecker:
    """资金博弈宽表检查器"""
    
    # 关键字段列表
    KEY_COLUMNS = [
        'net_mf_amount', 'net_main_amount', 'rzye', 'rqye',
        'top_net_amount', 'top_l_buy', 'top_l_sell',
        'hsgt_north', 'is_bj_stock'
    ]
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Args:
            df: 可选的DataFrame，如果不提供则从文件加载
        """
        if df is not None:
            self.df = df
        else:
            logger.info(f"加载资金博弈宽表: {DWD_EXT_PATHS.money_flow}")
            self.df = pd.read_parquet(DWD_EXT_PATHS.money_flow)
        
        self.results: List[CheckResult] = []
        self._prepare_data()
    
    def _prepare_data(self):
        """预处理数据"""
        # 确保日期格式统一
        if 'trade_date' in self.df.columns:
            self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
        
        # 添加训练期标记
        self.df['is_training'] = self.df['trade_date'] >= pd.to_datetime(TRAINING_START_DATE)
        
        # 添加北交所标记（如果不存在）
        if 'is_bj_stock' not in self.df.columns:
            self.df['is_bj_stock'] = self.df['ts_code'].apply(
                lambda x: 1 if x.endswith('.BJ') or x.startswith(('8', '4')) else 0
            )
    
    def get_summary(self) -> TableSummary:
        """获取数据表概览"""
        df = self.df
        
        # 计算缺失率
        null_summary = {}
        for col in self.KEY_COLUMNS:
            if col in df.columns:
                null_summary[col] = calculate_missing_rate(df[col])
        
        return TableSummary(
            name="dwd_money_flow",
            rows=len(df),
            columns=len(df.columns),
            date_range=(
                df['trade_date'].min().strftime('%Y-%m-%d'),
                df['trade_date'].max().strftime('%Y-%m-%d')
            ),
            stock_count=df['ts_code'].nunique(),
            file_size_mb=get_file_size_mb(DWD_EXT_PATHS.money_flow),
            memory_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            null_summary=null_summary
        )
    
    def check_primary_key_unique(self) -> CheckResult:
        """
        检查1: 主键唯一性
        (trade_date, ts_code) 是否唯一
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
            # 找出重复的样本
            dup_samples = df[duplicates].head(10)[['trade_date', 'ts_code']]
            metrics['duplicate_samples'] = dup_samples.to_dict('records')
        
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
    
    def check_training_nan_closure(self) -> CheckResult:
        """
        检查2: 训练期缺失值闭环
        rzye, top_net_amount 等关键列在训练期(2021+) NaN 数应为 0
        """
        logger.info("检查训练期缺失值闭环...")
        
        df = self.df
        training_df = df[df['is_training']]
        
        issues = []
        metrics = {
            'training_rows': len(training_df),
            'total_rows': len(df),
        }
        
        nan_columns = []
        for col in self.KEY_COLUMNS:
            if col in training_df.columns:
                nan_count = training_df[col].isna().sum()
                metrics[f'{col}_nan_count'] = int(nan_count)
                if nan_count > 0:
                    nan_columns.append((col, nan_count))
        
        if nan_columns:
            for col, count in nan_columns:
                issues.append(f"训练期 {col} 存在 {count} 个 NaN")
        
        passed = len(nan_columns) == 0
        
        result = CheckResult(
            name="训练期缺失值闭环",
            passed=passed,
            description="验证训练期(2021+)关键字段无 NaN",
            issues=issues,
            metrics=metrics,
            severity="CRITICAL" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"训练期缺失值检查: {'通过' if passed else '失败'}")
        return result
    
    def check_market_broadcast_consistency(self) -> CheckResult:
        """
        检查3: 市场广播一致性
        同一日期的 hsgt_north 对所有股票是否相同 (std == 0)
        """
        logger.info("检查市场广播一致性...")
        
        df = self.df
        issues = []
        metrics = {}
        
        if 'hsgt_north' not in df.columns:
            return CheckResult(
                name="市场广播一致性",
                passed=True,
                description="验证 hsgt_north 同日所有股票相同",
                issues=["hsgt_north 字段不存在"],
                severity="WARNING"
            )
        
        # 计算每日 hsgt_north 的标准差
        daily_std = df.groupby('trade_date')['hsgt_north'].std()
        inconsistent_days = daily_std[daily_std > 0.001]  # 允许微小误差
        
        metrics['total_days'] = len(daily_std)
        metrics['inconsistent_days'] = len(inconsistent_days)
        
        if len(inconsistent_days) > 0:
            issues.append(f"发现 {len(inconsistent_days)} 天 hsgt_north 广播不一致")
            # 样本
            sample_dates = inconsistent_days.head(5).index.tolist()
            metrics['inconsistent_sample_dates'] = [str(d) for d in sample_dates]
        
        passed = len(inconsistent_days) == 0
        
        result = CheckResult(
            name="市场广播一致性",
            passed=passed,
            description="验证 hsgt_north 同日所有股票相同",
            issues=issues,
            metrics=metrics,
            severity="ERROR" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"市场广播一致性检查: {'通过' if passed else '失败'}")
        return result
    
    def check_bj_stock_special_case(self) -> CheckResult:
        """
        检查4: 北交所特例
        若 is_bj_stock=1，允许 buy_lg_amount=0 但 net_mf_amount 可以 !=0
        """
        logger.info("检查北交所特例...")
        
        df = self.df
        issues = []
        metrics = {}
        
        if 'is_bj_stock' not in df.columns:
            return CheckResult(
                name="北交所特例",
                passed=True,
                description="验证北交所股票资金流特殊性",
                issues=["is_bj_stock 字段不存在"],
                severity="WARNING"
            )
        
        bj_df = df[df['is_bj_stock'] == 1]
        non_bj_df = df[df['is_bj_stock'] == 0]
        
        metrics['bj_stock_count'] = bj_df['ts_code'].nunique()
        metrics['bj_stock_rows'] = len(bj_df)
        metrics['non_bj_stock_rows'] = len(non_bj_df)
        
        # 北交所股票的大单金额应多为 0（因为北交所没有龙虎榜数据）
        if 'buy_lg_amount' in bj_df.columns:
            bj_lg_nonzero = (bj_df['buy_lg_amount'] != 0).sum()
            metrics['bj_lg_nonzero_count'] = int(bj_lg_nonzero)
            
            # 计算非零比例
            bj_lg_nonzero_ratio = bj_lg_nonzero / len(bj_df) if len(bj_df) > 0 else 0
            metrics['bj_lg_nonzero_ratio'] = bj_lg_nonzero_ratio
            
            # 北交所大单非零比例应该很低
            if bj_lg_nonzero_ratio > 0.01:  # 超过 1% 可能有问题
                issues.append(f"北交所股票 buy_lg_amount 非零比例 {bj_lg_nonzero_ratio:.2%} 超预期")
        
        # 检查北交所股票是否有净流入数据
        if 'net_mf_amount' in bj_df.columns:
            bj_mf_nonzero = (bj_df['net_mf_amount'] != 0).sum()
            metrics['bj_mf_nonzero_count'] = int(bj_mf_nonzero)
            metrics['bj_mf_nonzero_ratio'] = bj_mf_nonzero / len(bj_df) if len(bj_df) > 0 else 0
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="北交所特例",
            passed=passed,
            description="验证北交所股票资金流特殊性",
            issues=issues,
            metrics=metrics,
            severity="WARNING" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"北交所特例检查: {'通过' if passed else '警告'}")
        return result
    
    def check_extreme_values(self) -> CheckResult:
        """
        检查5: 极值异常
        检查 net_main_amount 是否出现天文数字或 inf
        """
        logger.info("检查极值异常...")
        
        df = self.df
        issues = []
        metrics = {}
        
        # 检查的数值列
        amount_cols = [
            'net_mf_amount', 'net_main_amount', 'rzye', 'rqye',
            'top_net_amount', 'top_l_buy', 'top_l_sell'
        ]
        
        for col in amount_cols:
            if col not in df.columns:
                continue
            
            series = df[col]
            
            # 检查 inf
            inf_count = np.isinf(series).sum()
            metrics[f'{col}_inf_count'] = int(inf_count)
            if inf_count > 0:
                issues.append(f"{col} 存在 {inf_count} 个 inf 值")
            
            # 检查极值（超过 100 亿）
            extreme_count = (series.abs() > EXT_THRESHOLDS.MONEY_FLOW_MAX_AMOUNT).sum()
            metrics[f'{col}_extreme_count'] = int(extreme_count)
            if extreme_count > 0:
                issues.append(f"{col} 存在 {extreme_count} 个超过 100 亿的极值")
            
            # 记录范围
            metrics[f'{col}_min'] = float(series.min())
            metrics[f'{col}_max'] = float(series.max())
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="极值异常",
            passed=passed,
            description="检查数值字段无 inf 且在合理量级",
            issues=issues,
            metrics=metrics,
            severity="ERROR" if not passed else "INFO"
        )
        
        self.results.append(result)
        logger.info(f"极值异常检查: {'通过' if passed else '失败'}")
        return result
    
    def run_all_checks(self) -> List[CheckResult]:
        """运行所有检查"""
        self.results = []
        
        self.check_primary_key_unique()
        self.check_training_nan_closure()
        self.check_market_broadcast_consistency()
        self.check_bj_stock_special_case()
        self.check_extreme_values()
        
        return self.results
