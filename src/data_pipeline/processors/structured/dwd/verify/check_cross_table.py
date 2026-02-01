"""
DWD 数据质量检查 - 跨表一致性检查 (Golden Check)

检查内容：
1. 停牌状态一致性：Status表停牌时，Price表不应有成交量
2. 退市后数据一致性：退市后Fundamental数据应停止更新
3. 北交所股票代码规则验证
4. 主键一致性：三表的 (trade_date, ts_code) 覆盖范围
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
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


class CrossTableChecker:
    """跨表一致性检查器"""
    
    def __init__(
        self,
        price_df: Optional[pd.DataFrame] = None,
        fundamental_df: Optional[pd.DataFrame] = None,
        status_df: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            price_df: 量价表DataFrame
            fundamental_df: 基本面表DataFrame
            status_df: 状态表DataFrame
        """
        logger.info("加载三张DWD宽表...")
        
        # 加载数据
        if price_df is not None:
            self.price_df = price_df
        else:
            self.price_df = pd.read_parquet(DWD_PATHS.stock_price)
        
        if fundamental_df is not None:
            self.fundamental_df = fundamental_df
        else:
            self.fundamental_df = pd.read_parquet(DWD_PATHS.stock_fundamental)
        
        if status_df is not None:
            self.status_df = status_df
        else:
            self.status_df = pd.read_parquet(DWD_PATHS.stock_status)
        
        self.results: List[CheckResult] = []
        self._prepare_data()
        
        logger.info(f"数据加载完成:")
        logger.info(f"  - Price表: {len(self.price_df):,} 行")
        logger.info(f"  - Fundamental表: {len(self.fundamental_df):,} 行")
        logger.info(f"  - Status表: {len(self.status_df):,} 行")
    
    def _prepare_data(self):
        """预处理数据"""
        # 统一日期格式
        for df in [self.price_df, self.fundamental_df, self.status_df]:
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    def check_suspension_consistency(self) -> CheckResult:
        """
        检查1: 停牌状态一致性
        Status表显示停牌时，Price表不应有成交量
        """
        logger.info("检查停牌状态一致性...")
        
        issues = []
        metrics = {}
        
        # 获取Status表中停牌的记录
        if 'is_trading' in self.status_df.columns:
            suspended_status = self.status_df[self.status_df['is_trading'] == 0][['trade_date', 'ts_code']]
        else:
            return CheckResult(
                name="停牌状态一致性",
                passed=True,
                description="验证Status表停牌时Price表无成交",
                issues=["Status表缺少is_trading字段"],
                severity="WARNING"
            )
        
        metrics['suspended_days_in_status'] = len(suspended_status)
        
        if len(suspended_status) == 0:
            return CheckResult(
                name="停牌状态一致性",
                passed=True,
                description="验证Status表停牌时Price表无成交",
                issues=["Status表中没有停牌记录"],
                metrics=metrics,
                severity="INFO"
            )
        
        # 合并Price表
        if 'vol' in self.price_df.columns or 'is_trading' in self.price_df.columns:
            merged = suspended_status.merge(
                self.price_df[['trade_date', 'ts_code', 'vol'] + 
                             (['is_trading'] if 'is_trading' in self.price_df.columns else [])],
                on=['trade_date', 'ts_code'],
                how='inner',
                suffixes=('_status', '_price')
            )
            
            metrics['merged_suspended_rows'] = len(merged)
            
            if len(merged) > 0:
                # 检查停牌日是否有成交量
                if 'vol' in merged.columns:
                    has_volume = merged[merged['vol'] > 0]
                    metrics['suspended_with_volume'] = len(has_volume)
                    
                    if len(has_volume) > 0:
                        issues.append(f"发现 {len(has_volume)} 行Status表停牌但Price表有成交量")
                        sample = has_volume.head(10)[['trade_date', 'ts_code', 'vol']]
                        metrics['suspended_with_volume_sample'] = sample.to_dict('records')
                
                # 检查is_trading一致性
                if 'is_trading_price' in merged.columns:
                    inconsistent = merged[merged['is_trading_price'] == 1]
                    metrics['is_trading_inconsistent'] = len(inconsistent)
                    
                    if len(inconsistent) > 0:
                        issues.append(f"发现 {len(inconsistent)} 行两表is_trading状态不一致")
        else:
            issues.append("Price表缺少vol或is_trading字段")
        
        # 综合判定
        problem_count = metrics.get('suspended_with_volume', 0) + metrics.get('is_trading_inconsistent', 0)
        passed = problem_count == 0
        severity = "ERROR" if not passed else "INFO"
        
        result = CheckResult(
            name="停牌状态一致性",
            passed=passed,
            description="验证Status表停牌时Price表无成交",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"停牌一致性检查完成: {'通过' if passed else '失败'}")
        return result
    
    def check_delisting_consistency(self) -> CheckResult:
        """
        检查2: 退市后数据一致性
        退市日期之后Fundamental数据应停止更新或为空
        """
        logger.info("检查退市后数据一致性...")
        
        issues = []
        metrics = {}
        
        # 尝试加载股票列表获取退市日期
        try:
            stock_list = pd.read_parquet(DWD_PATHS.stock_list)
            if 'delist_date' in stock_list.columns:
                stock_list['delist_date'] = pd.to_datetime(stock_list['delist_date'], errors='coerce')
                delisted = stock_list[stock_list['delist_date'].notna()]
                metrics['delisted_stocks_count'] = len(delisted)
                
                if len(delisted) > 0:
                    # 检查退市后的数据
                    delisted_codes = delisted['ts_code'].tolist()
                    
                    # 在Fundamental表中检查
                    fundamental_delisted = self.fundamental_df[
                        self.fundamental_df['ts_code'].isin(delisted_codes)
                    ]
                    
                    if len(fundamental_delisted) > 0:
                        # 合并退市日期
                        fund_with_delist = fundamental_delisted.merge(
                            delisted[['ts_code', 'delist_date']],
                            on='ts_code',
                            how='left'
                        )
                        
                        # 找出退市后的记录
                        after_delist = fund_with_delist[
                            fund_with_delist['trade_date'] > fund_with_delist['delist_date']
                        ]
                        
                        metrics['records_after_delist_fundamental'] = len(after_delist)
                        
                        if len(after_delist) > 0:
                            issues.append(f"Fundamental表有 {len(after_delist)} 行退市后的数据")
                            sample = after_delist.head(10)[['trade_date', 'ts_code', 'delist_date']]
                            metrics['after_delist_sample'] = sample.to_dict('records')
                    
                    # 在Price表中也检查
                    price_delisted = self.price_df[
                        self.price_df['ts_code'].isin(delisted_codes)
                    ]
                    
                    if len(price_delisted) > 0:
                        price_with_delist = price_delisted.merge(
                            delisted[['ts_code', 'delist_date']],
                            on='ts_code',
                            how='left'
                        )
                        
                        after_delist_price = price_with_delist[
                            price_with_delist['trade_date'] > price_with_delist['delist_date']
                        ]
                        
                        metrics['records_after_delist_price'] = len(after_delist_price)
                        
                        # 这个不一定是问题，可能是退市整理期
                        if len(after_delist_price) > 0:
                            issues.append(f"Price表有 {len(after_delist_price)} 行退市后的数据（可能是退市整理期）")
                else:
                    issues.append("没有退市股票数据")
            else:
                issues.append("股票列表缺少delist_date字段")
        except Exception as e:
            issues.append(f"无法加载股票列表进行退市检查: {str(e)}")
        
        # 综合判定（退市后有数据不一定是错误，可能是业务需要）
        passed = True  # 这个检查主要是提供信息，不作为硬性失败条件
        severity = "WARNING" if issues else "INFO"
        
        result = CheckResult(
            name="退市后数据一致性",
            passed=passed,
            description="检查退市后Fundamental数据是否停止更新",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"退市一致性检查完成")
        return result
    
    def check_bse_stock_codes(self) -> CheckResult:
        """
        检查3: 北交所股票代码规则验证
        北交所股票代码应以 8 或 4 开头
        """
        logger.info("检查北交所股票代码...")
        
        issues = []
        metrics = {}
        
        # 从Status表获取北交所股票
        if 'market' in self.status_df.columns:
            bse_stocks = self.status_df[
                self.status_df['market'].isin(['北交所', 'bse', 'BSE'])
            ]['ts_code'].unique()
        else:
            # 通过代码推断
            all_codes = self.status_df['ts_code'].unique()
            bse_stocks = [c for c in all_codes if c[:1] in ['8', '4']]
        
        metrics['bse_stocks_count'] = len(bse_stocks)
        
        if len(bse_stocks) == 0:
            return CheckResult(
                name="北交所股票代码验证",
                passed=True,
                description="验证北交所股票代码符合编码规则",
                issues=["未发现北交所股票"],
                metrics=metrics,
                severity="INFO"
            )
        
        # 验证代码规则
        invalid_codes = []
        for code in bse_stocks:
            # 北交所代码：8开头（新三板精选层/北交所）或 4开头（老三板）
            # 现在北交所主要是8开头的代码
            if not (code.startswith('8') or code.startswith('4')):
                invalid_codes.append(code)
        
        metrics['invalid_bse_codes'] = invalid_codes
        metrics['invalid_bse_codes_count'] = len(invalid_codes)
        
        if invalid_codes:
            issues.append(f"发现 {len(invalid_codes)} 个北交所股票代码不符合规则")
        
        # 反向检查：8开头的代码是否都被标记为北交所
        if 'market' in self.status_df.columns:
            code_8_stocks = self.status_df[
                self.status_df['ts_code'].str.startswith('8')
            ]
            non_bse_code_8 = code_8_stocks[
                ~code_8_stocks['market'].isin(['北交所', 'bse', 'BSE'])
            ]['ts_code'].unique()
            
            if len(non_bse_code_8) > 0:
                metrics['code_8_not_bse'] = list(non_bse_code_8[:10])
                issues.append(f"发现 {len(non_bse_code_8)} 个8开头代码未标记为北交所")
        
        # 综合判定
        passed = len(invalid_codes) == 0
        severity = "WARNING" if not passed else "INFO"
        
        result = CheckResult(
            name="北交所股票代码验证",
            passed=passed,
            description="验证北交所股票代码符合8或4开头的编码规则",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"北交所代码检查完成: {'通过' if passed else '告警'}")
        return result
    
    def check_primary_key_coverage(self) -> CheckResult:
        """
        检查4: 主键一致性检查
        三表的 (trade_date, ts_code) 覆盖范围应一致
        """
        logger.info("检查主键覆盖范围...")
        
        issues = []
        metrics = {}
        
        # 获取各表的主键集合
        price_keys = set(zip(
            self.price_df['trade_date'].dt.strftime('%Y-%m-%d'),
            self.price_df['ts_code']
        ))
        
        fundamental_keys = set(zip(
            self.fundamental_df['trade_date'].dt.strftime('%Y-%m-%d'),
            self.fundamental_df['ts_code']
        ))
        
        status_keys = set(zip(
            self.status_df['trade_date'].dt.strftime('%Y-%m-%d'),
            self.status_df['ts_code']
        ))
        
        metrics['price_keys_count'] = len(price_keys)
        metrics['fundamental_keys_count'] = len(fundamental_keys)
        metrics['status_keys_count'] = len(status_keys)
        
        # 计算交集和差集
        all_keys = price_keys | fundamental_keys | status_keys
        common_keys = price_keys & fundamental_keys & status_keys
        
        metrics['union_keys_count'] = len(all_keys)
        metrics['intersection_keys_count'] = len(common_keys)
        
        # 各表独有的键
        price_only = price_keys - fundamental_keys - status_keys
        fundamental_only = fundamental_keys - price_keys - status_keys
        status_only = status_keys - price_keys - fundamental_keys
        
        metrics['price_only_count'] = len(price_only)
        metrics['fundamental_only_count'] = len(fundamental_only)
        metrics['status_only_count'] = len(status_only)
        
        # 两两比较
        metrics['in_price_not_fundamental'] = len(price_keys - fundamental_keys)
        metrics['in_price_not_status'] = len(price_keys - status_keys)
        metrics['in_fundamental_not_price'] = len(fundamental_keys - price_keys)
        metrics['in_status_not_price'] = len(status_keys - price_keys)
        
        # 检查覆盖率
        # Price表应该是基准
        coverage_fundamental = len(price_keys & fundamental_keys) / len(price_keys) if len(price_keys) > 0 else 0
        coverage_status = len(price_keys & status_keys) / len(price_keys) if len(price_keys) > 0 else 0
        
        metrics['fundamental_coverage_of_price'] = coverage_fundamental
        metrics['status_coverage_of_price'] = coverage_status
        
        if coverage_fundamental < 0.95:
            issues.append(f"Fundamental表只覆盖Price表的 {coverage_fundamental:.1%} 主键")
        
        if coverage_status < 0.95:
            issues.append(f"Status表只覆盖Price表的 {coverage_status:.1%} 主键")
        
        # 采样差异
        if len(price_only) > 0:
            metrics['price_only_sample'] = list(price_only)[:10]
        if len(fundamental_only) > 0:
            metrics['fundamental_only_sample'] = list(fundamental_only)[:10]
        if len(status_only) > 0:
            metrics['status_only_sample'] = list(status_only)[:10]
        
        # 综合判定
        # 允许一定的差异（可能是数据源差异）
        passed = coverage_fundamental >= 0.90 and coverage_status >= 0.90
        severity = "WARNING" if not passed else "INFO"
        
        result = CheckResult(
            name="主键覆盖范围",
            passed=passed,
            description="验证三表的(trade_date, ts_code)覆盖范围一致性",
            issues=issues,
            metrics=metrics,
            severity=severity
        )
        
        self.results.append(result)
        logger.info(f"主键覆盖检查完成: {'通过' if passed else '告警'}")
        return result
    
    def run_all_checks(self) -> List[CheckResult]:
        """运行所有跨表检查"""
        logger.info("=" * 60)
        logger.info("开始跨表一致性检查 (Golden Check)")
        logger.info("=" * 60)
        
        self.check_suspension_consistency()
        self.check_delisting_consistency()
        self.check_bse_stock_codes()
        self.check_primary_key_coverage()
        
        passed_count = sum(1 for r in self.results if r.passed)
        logger.info(f"检查完成: {passed_count}/{len(self.results)} 项通过")
        
        return self.results


def main():
    """独立运行测试"""
    checker = CrossTableChecker()
    
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
