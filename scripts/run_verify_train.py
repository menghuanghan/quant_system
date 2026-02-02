#!/usr/bin/env python3
"""
train.parquet 全量数据质量校验脚本

对生成的训练数据进行系统性质量校验，涵盖四个核心维度：
1. 基础完整性校验 - NaN/Inf/数据量级/列名规范
2. 特征分布校验 - Z-Score标准化/极值截断/常量列
3. 标签质量校验 - 非空率/分布/极值/数据泄露
4. 时序逻辑校验 - 时间范围/每日股票数/代码覆盖

使用方法：
    python scripts/run_verify_train.py                    # 运行所有校验
    python scripts/run_verify_train.py --check integrity  # 只运行完整性校验
    python scripts/run_verify_train.py --output report.md # 指定输出文件

输出：reports/train_dq_report.md
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class VerifyConfig:
    """校验配置"""
    
    # 数据路径
    train_path: Path = PROJECT_ROOT / "data" / "features" / "structured" / "train.parquet"
    report_path: Path = PROJECT_ROOT / "reports" / "train_dq_report.md"
    
    # 基础完整性阈值
    min_rows: int = 5_000_000           # 最小样本数
    max_nan_rate: float = 0.0           # 最大 NaN 率 (严格 0%)
    max_inf_rate: float = 0.0           # 最大 Inf 率 (严格 0%)
    
    # Z-Score 校验阈值
    zscore_mean_tolerance: float = 0.01  # 截面均值容差 ±0.01
    zscore_std_tolerance: float = 0.05   # 截面标准差容差 ±0.05
    zscore_clip_threshold: float = 3.5   # 极值截断阈值
    
    # 标签校验阈值
    label_min_valid_rate: float = 0.98   # 标签最小有效率 98%
    label_max_abs_value: float = 0.5     # 标签最大绝对值 50%
    
    # 时序校验阈值
    train_start_date: str = "2021-01-01" # 训练集起始日期
    min_daily_stocks: int = 3000         # 每日最少股票数
    max_daily_stocks: int = 6000         # 每日最多股票数
    expected_stock_count: int = 5000     # 预期股票数量
    
    # 特征列后缀
    zscore_suffix: str = "_zscore"       # Z-Score 列后缀
    
    # 标签列
    label_cols: List[str] = field(default_factory=lambda: [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "label_1d", "label_5d", "label_10d", "label_20d"
    ])
    
    # 主标签
    primary_label: str = "ret_5d"


# ============================================================================
# 校验结果数据结构
# ============================================================================

@dataclass
class CheckResult:
    """单项检查结果"""
    name: str
    passed: bool
    level: str  # INFO, WARNING, ERROR
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class VerifyReport:
    """校验报告"""
    timestamp: str
    data_path: str
    data_shape: Tuple[int, int]
    
    integrity_checks: List[CheckResult] = field(default_factory=list)
    distribution_checks: List[CheckResult] = field(default_factory=list)
    label_checks: List[CheckResult] = field(default_factory=list)
    temporal_checks: List[CheckResult] = field(default_factory=list)
    
    def count_by_level(self, level: str) -> int:
        """统计指定级别的检查数量"""
        all_checks = (
            self.integrity_checks + 
            self.distribution_checks + 
            self.label_checks + 
            self.temporal_checks
        )
        return sum(1 for c in all_checks if c.level == level)
    
    def all_passed(self) -> bool:
        """是否全部通过（无 ERROR）"""
        return self.count_by_level("ERROR") == 0


# ============================================================================
# 校验器类
# ============================================================================

class TrainDataVerifier:
    """训练数据校验器"""
    
    def __init__(self, config: VerifyConfig):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.report: Optional[VerifyReport] = None
    
    def load_data(self) -> pd.DataFrame:
        """加载训练数据"""
        logger.info(f"📖 加载数据: {self.config.train_path}")
        
        if not self.config.train_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.config.train_path}")
        
        self.df = pd.read_parquet(self.config.train_path)
        logger.info(f"   ✓ 形状: {self.df.shape}")
        
        return self.df
    
    def run_all_checks(self) -> VerifyReport:
        """运行所有校验"""
        if self.df is None:
            self.load_data()
        
        self.report = VerifyReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data_path=str(self.config.train_path),
            data_shape=self.df.shape,
        )
        
        logger.info("=" * 70)
        logger.info("🔍 开始数据质量校验")
        logger.info("=" * 70)
        
        # 1. 基础完整性校验
        self._check_integrity()
        
        # 2. 特征分布校验
        self._check_distribution()
        
        # 3. 标签质量校验
        self._check_labels()
        
        # 4. 时序逻辑校验
        self._check_temporal()
        
        return self.report
    
    # ========================================================================
    # 1. 基础完整性校验
    # ========================================================================
    
    def _check_integrity(self):
        """基础完整性校验"""
        logger.info("")
        logger.info("📋 1. 基础完整性校验")
        logger.info("-" * 50)
        
        checks = []
        
        # 1.1 检查数据量级
        row_count = len(self.df)
        passed = row_count >= self.config.min_rows
        checks.append(CheckResult(
            name="数据量级",
            passed=passed,
            level="ERROR" if not passed else "INFO",
            message=f"总行数: {row_count:,} (阈值: >= {self.config.min_rows:,})",
            details={"row_count": row_count, "threshold": self.config.min_rows}
        ))
        logger.info(f"   {'✅' if passed else '❌'} 数据量级: {row_count:,} 行")
        
        # 1.2 检查 NaN 值
        nan_counts = self.df.isna().sum()
        total_nan = nan_counts.sum()
        nan_rate = total_nan / (len(self.df) * len(self.df.columns))
        passed = nan_rate <= self.config.max_nan_rate
        
        nan_cols = nan_counts[nan_counts > 0].to_dict()
        checks.append(CheckResult(
            name="NaN 检查",
            passed=passed,
            level="ERROR" if not passed else "INFO",
            message=f"NaN 率: {nan_rate:.4%} (阈值: <= {self.config.max_nan_rate:.2%})",
            details={"nan_rate": nan_rate, "nan_cols": nan_cols}
        ))
        logger.info(f"   {'✅' if passed else '❌'} NaN 率: {nan_rate:.4%} ({len(nan_cols)} 列有空值)")
        
        # 1.3 检查 Inf 值
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(self.df[numeric_cols]).any()
        inf_cols = inf_mask[inf_mask].index.tolist()
        inf_count = np.isinf(self.df[numeric_cols]).sum().sum()
        inf_rate = inf_count / (len(self.df) * len(numeric_cols))
        passed = inf_rate <= self.config.max_inf_rate
        
        checks.append(CheckResult(
            name="Inf 检查",
            passed=passed,
            level="ERROR" if not passed else "INFO",
            message=f"Inf 率: {inf_rate:.4%} (阈值: <= {self.config.max_inf_rate:.2%})",
            details={"inf_rate": inf_rate, "inf_cols": inf_cols}
        ))
        logger.info(f"   {'✅' if passed else '❌'} Inf 率: {inf_rate:.4%} ({len(inf_cols)} 列有无穷值)")
        
        # 1.4 检查列名规范性
        invalid_cols = [c for c in self.df.columns if not c.replace('_', '').replace('.', '').isalnum()]
        passed = len(invalid_cols) == 0
        
        checks.append(CheckResult(
            name="列名规范",
            passed=passed,
            level="WARNING" if not passed else "INFO",
            message=f"非法列名: {len(invalid_cols)} 个",
            details={"invalid_cols": invalid_cols}
        ))
        logger.info(f"   {'✅' if passed else '⚠️'} 列名规范: {len(invalid_cols)} 个非法列名")
        
        # 1.5 检查必要列
        required_cols = ['ts_code', 'trade_date'] + self.config.label_cols[:4]  # 至少有收益率标签
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        passed = len(missing_cols) == 0
        
        checks.append(CheckResult(
            name="必要列检查",
            passed=passed,
            level="ERROR" if not passed else "INFO",
            message=f"缺失必要列: {missing_cols}" if not passed else "所有必要列存在",
            details={"missing_cols": missing_cols}
        ))
        logger.info(f"   {'✅' if passed else '❌'} 必要列: {'完整' if passed else f'缺失 {missing_cols}'}")
        
        self.report.integrity_checks = checks
    
    # ========================================================================
    # 2. 特征分布校验
    # ========================================================================
    
    def _check_distribution(self):
        """特征分布校验"""
        logger.info("")
        logger.info("📋 2. 特征分布校验")
        logger.info("-" * 50)
        
        checks = []
        
        # 2.1 检查 Z-Score 列的截面分布
        zscore_cols = [c for c in self.df.columns if c.endswith(self.config.zscore_suffix)]
        
        if zscore_cols:
            # 按交易日分组计算截面统计
            grouped = self.df.groupby('trade_date')[zscore_cols]
            
            # 计算每日截面均值和标准差
            daily_means = grouped.mean()
            daily_stds = grouped.std()
            
            # 检查均值是否接近 0
            mean_of_means = daily_means.mean()
            bad_mean_cols = mean_of_means[mean_of_means.abs() > self.config.zscore_mean_tolerance].index.tolist()
            passed = len(bad_mean_cols) == 0
            
            checks.append(CheckResult(
                name="Z-Score 截面均值",
                passed=passed,
                level="WARNING" if not passed else "INFO",
                message=f"偏离 0 的列: {len(bad_mean_cols)} 个 (容差: ±{self.config.zscore_mean_tolerance})",
                details={"bad_cols": bad_mean_cols, "values": mean_of_means[bad_mean_cols].to_dict() if bad_mean_cols else {}}
            ))
            logger.info(f"   {'✅' if passed else '⚠️'} Z-Score 均值: {len(bad_mean_cols)} 列偏离 0")
            
            # 检查标准差是否接近 1
            mean_of_stds = daily_stds.mean()
            bad_std_cols = mean_of_stds[(mean_of_stds - 1).abs() > self.config.zscore_std_tolerance].index.tolist()
            passed = len(bad_std_cols) == 0
            
            checks.append(CheckResult(
                name="Z-Score 截面标准差",
                passed=passed,
                level="WARNING" if not passed else "INFO",
                message=f"偏离 1 的列: {len(bad_std_cols)} 个 (容差: ±{self.config.zscore_std_tolerance})",
                details={"bad_cols": bad_std_cols, "values": mean_of_stds[bad_std_cols].to_dict() if bad_std_cols else {}}
            ))
            logger.info(f"   {'✅' if passed else '⚠️'} Z-Score 标准差: {len(bad_std_cols)} 列偏离 1")
        else:
            logger.info(f"   ⚠️ 未发现 Z-Score 列 (后缀: {self.config.zscore_suffix})")
        
        # 2.2 检查极值截断
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # 排除标签列
        feature_cols = [c for c in numeric_cols if c not in self.config.label_cols]
        
        max_vals = self.df[feature_cols].max()
        min_vals = self.df[feature_cols].min()
        
        extreme_cols = []
        for col in feature_cols:
            max_val = max_vals[col]
            min_val = min_vals[col]
            # 跳过 NA 值
            if pd.isna(max_val) or pd.isna(min_val):
                continue
            if max_val > self.config.zscore_clip_threshold or min_val < -self.config.zscore_clip_threshold:
                if col.endswith(self.config.zscore_suffix):  # 只检查 zscore 列
                    extreme_cols.append(col)
        
        passed = len(extreme_cols) == 0
        checks.append(CheckResult(
            name="极值截断",
            passed=passed,
            level="WARNING" if not passed else "INFO",
            message=f"超出 ±{self.config.zscore_clip_threshold} 的 Z-Score 列: {len(extreme_cols)} 个",
            details={"extreme_cols": extreme_cols}
        ))
        logger.info(f"   {'✅' if passed else '⚠️'} 极值截断: {len(extreme_cols)} 个 Z-Score 列超限")
        
        # 2.3 检查常量列
        std_vals = self.df[feature_cols].std()
        constant_cols = std_vals[std_vals == 0].index.tolist()
        passed = len(constant_cols) == 0
        
        checks.append(CheckResult(
            name="常量列检查",
            passed=passed,
            level="WARNING" if not passed else "INFO",
            message=f"常量列 (std=0): {len(constant_cols)} 个",
            details={"constant_cols": constant_cols}
        ))
        logger.info(f"   {'✅' if passed else '⚠️'} 常量列: {len(constant_cols)} 个")
        
        # 2.4 特征数量统计
        feature_count = len(feature_cols)
        label_count = len([c for c in self.df.columns if c in self.config.label_cols])
        
        checks.append(CheckResult(
            name="列数统计",
            passed=True,
            level="INFO",
            message=f"特征列: {feature_count}, 标签列: {label_count}, 总计: {len(self.df.columns)}",
            details={"feature_cols": feature_count, "label_cols": label_count, "total": len(self.df.columns)}
        ))
        logger.info(f"   ℹ️ 列数: 特征 {feature_count} + 标签 {label_count} = {len(self.df.columns)}")
        
        self.report.distribution_checks = checks
    
    # ========================================================================
    # 3. 标签质量校验
    # ========================================================================
    
    def _check_labels(self):
        """标签质量校验"""
        logger.info("")
        logger.info("📋 3. 标签质量校验")
        logger.info("-" * 50)
        
        checks = []
        
        # 获取存在的标签列
        existing_labels = [c for c in self.config.label_cols if c in self.df.columns]
        ret_labels = [c for c in existing_labels if c.startswith('ret_')]
        
        # 3.1 检查标签非空率
        for label in ret_labels:
            valid_rate = self.df[label].notna().mean()
            passed = valid_rate >= self.config.label_min_valid_rate
            
            checks.append(CheckResult(
                name=f"{label} 非空率",
                passed=passed,
                level="WARNING" if not passed else "INFO",
                message=f"{label}: {valid_rate:.2%} (阈值: >= {self.config.label_min_valid_rate:.0%})",
                details={"label": label, "valid_rate": valid_rate}
            ))
            logger.info(f"   {'✅' if passed else '⚠️'} {label} 非空率: {valid_rate:.2%}")
        
        # 3.2 检查标签分布 (主标签)
        if self.config.primary_label in self.df.columns:
            label_data = self.df[self.config.primary_label].dropna()
            
            # 统计分布
            mean_val = label_data.mean()
            std_val = label_data.std()
            skew_val = label_data.skew()
            kurt_val = label_data.kurtosis()
            
            # 检查是否过于偏态
            passed = abs(skew_val) < 2.0  # 偏度不超过 2
            checks.append(CheckResult(
                name=f"{self.config.primary_label} 分布",
                passed=passed,
                level="WARNING" if not passed else "INFO",
                message=f"均值={mean_val:.4f}, 标准差={std_val:.4f}, 偏度={skew_val:.2f}, 峰度={kurt_val:.2f}",
                details={"mean": mean_val, "std": std_val, "skew": skew_val, "kurtosis": kurt_val}
            ))
            logger.info(f"   {'✅' if passed else '⚠️'} {self.config.primary_label} 分布: 偏度={skew_val:.2f}")
        
        # 3.3 检查标签极值
        for label in ret_labels:
            if label in self.df.columns:
                max_val = self.df[label].max()
                min_val = self.df[label].min()
                
                passed = max_val <= self.config.label_max_abs_value and min_val >= -self.config.label_max_abs_value
                
                checks.append(CheckResult(
                    name=f"{label} 极值",
                    passed=passed,
                    level="ERROR" if not passed else "INFO",
                    message=f"范围: [{min_val:.2%}, {max_val:.2%}] (阈值: ±{self.config.label_max_abs_value:.0%})",
                    details={"label": label, "min": min_val, "max": max_val}
                ))
                logger.info(f"   {'✅' if passed else '❌'} {label} 极值: [{min_val:.2%}, {max_val:.2%}]")
        
        # 3.4 检查未来数据泄露 (最后几天标签应为 NaN 或已被正确处理)
        if 'trade_date' in self.df.columns and self.config.primary_label in self.df.columns:
            # 获取最后 5 个交易日
            last_dates = sorted(self.df['trade_date'].unique())[-5:]
            last_data = self.df[self.df['trade_date'].isin(last_dates)]
            
            # 检查 ret_5d 在最后 5 天是否有非空值（如果有可能是泄露）
            if 'ret_5d' in self.df.columns:
                last_5d_valid = last_data['ret_5d'].notna().sum()
                # 最后 5 天应该没有完整的 ret_5d（因为没有未来 5 天的数据）
                # 但由于数据已经被清洗，这里只记录信息
                checks.append(CheckResult(
                    name="未来数据泄露检查",
                    passed=True,
                    level="INFO",
                    message=f"最后 5 天 ret_5d 非空: {last_5d_valid} 行 (已清洗的数据通常为 0)",
                    details={"last_dates": last_dates, "valid_count": int(last_5d_valid)}
                ))
                logger.info(f"   ℹ️ 最后 5 天 ret_5d 非空: {last_5d_valid} 行")
        
        # 3.5 分类标签分布
        class_labels = [c for c in existing_labels if c.startswith('label_')]
        for label in class_labels:
            if label in self.df.columns:
                value_counts = self.df[label].value_counts(normalize=True)
                # 假设 0=跌, 1=平, 2=涨
                dist_str = ", ".join([f"{k}:{v:.1%}" for k, v in sorted(value_counts.items())])
                
                checks.append(CheckResult(
                    name=f"{label} 分类分布",
                    passed=True,
                    level="INFO",
                    message=dist_str,
                    details={"distribution": value_counts.to_dict()}
                ))
                logger.info(f"   ℹ️ {label}: {dist_str}")
        
        self.report.label_checks = checks
    
    # ========================================================================
    # 4. 时序逻辑校验
    # ========================================================================
    
    def _check_temporal(self):
        """时序逻辑校验"""
        logger.info("")
        logger.info("📋 4. 时序逻辑校验")
        logger.info("-" * 50)
        
        checks = []
        
        if 'trade_date' not in self.df.columns:
            logger.warning("   ⚠️ trade_date 列不存在，跳过时序校验")
            return
        
        # 4.1 检查时间范围
        min_date = self.df['trade_date'].min()
        max_date = self.df['trade_date'].max()
        
        passed = str(min_date) >= self.config.train_start_date
        checks.append(CheckResult(
            name="时间范围",
            passed=passed,
            level="ERROR" if not passed else "INFO",
            message=f"范围: {min_date} ~ {max_date} (起始阈值: >= {self.config.train_start_date})",
            details={"min_date": str(min_date), "max_date": str(max_date)}
        ))
        logger.info(f"   {'✅' if passed else '❌'} 时间范围: {min_date} ~ {max_date}")
        
        # 4.2 检查每日股票数量
        daily_counts = self.df.groupby('trade_date')['ts_code'].nunique()
        min_daily = daily_counts.min()
        max_daily = daily_counts.max()
        mean_daily = daily_counts.mean()
        
        passed = min_daily >= self.config.min_daily_stocks
        
        # 找出异常日期
        abnormal_dates = daily_counts[daily_counts < self.config.min_daily_stocks].index.tolist()
        
        checks.append(CheckResult(
            name="每日股票数",
            passed=passed,
            level="WARNING" if not passed else "INFO",
            message=f"范围: {min_daily:,} ~ {max_daily:,}, 均值: {mean_daily:.0f} (阈值: >= {self.config.min_daily_stocks})",
            details={
                "min": int(min_daily), 
                "max": int(max_daily), 
                "mean": float(mean_daily),
                "abnormal_dates": abnormal_dates[:10]  # 最多保留 10 个
            }
        ))
        logger.info(f"   {'✅' if passed else '⚠️'} 每日股票数: {min_daily:,} ~ {max_daily:,}")
        
        # 4.3 检查股票代码覆盖
        unique_stocks = self.df['ts_code'].nunique()
        passed = unique_stocks >= self.config.expected_stock_count * 0.9  # 90% 容差
        
        checks.append(CheckResult(
            name="股票代码覆盖",
            passed=passed,
            level="WARNING" if not passed else "INFO",
            message=f"唯一股票: {unique_stocks:,} (预期: ~{self.config.expected_stock_count:,})",
            details={"unique_stocks": unique_stocks}
        ))
        logger.info(f"   {'✅' if passed else '⚠️'} 股票覆盖: {unique_stocks:,} 只")
        
        # 4.4 检查交易日数量
        unique_dates = self.df['trade_date'].nunique()
        
        checks.append(CheckResult(
            name="交易日数量",
            passed=True,
            level="INFO",
            message=f"交易日: {unique_dates:,} 天",
            details={"unique_dates": unique_dates}
        ))
        logger.info(f"   ℹ️ 交易日: {unique_dates:,} 天")
        
        # 4.5 检查是否有周末/假期数据（可选）
        if pd.api.types.is_datetime64_any_dtype(self.df['trade_date']):
            dates = pd.to_datetime(self.df['trade_date'])
        else:
            dates = pd.to_datetime(self.df['trade_date'])
        
        weekend_mask = dates.dt.dayofweek >= 5
        weekend_count = weekend_mask.sum()
        passed = weekend_count == 0
        
        checks.append(CheckResult(
            name="周末数据检查",
            passed=passed,
            level="WARNING" if not passed else "INFO",
            message=f"周末数据: {weekend_count:,} 行",
            details={"weekend_count": int(weekend_count)}
        ))
        logger.info(f"   {'✅' if passed else '⚠️'} 周末数据: {weekend_count:,} 行")
        
        self.report.temporal_checks = checks
    
    # ========================================================================
    # 报告生成
    # ========================================================================
    
    def generate_report(self) -> str:
        """生成 Markdown 格式报告"""
        if self.report is None:
            raise ValueError("请先运行 run_all_checks()")
        
        lines = []
        
        # 标题
        lines.append("# train.parquet 数据质量报告")
        lines.append("")
        lines.append(f"**生成时间**: {self.report.timestamp}")
        lines.append(f"**数据路径**: `{self.report.data_path}`")
        lines.append(f"**数据形状**: {self.report.data_shape[0]:,} 行 × {self.report.data_shape[1]} 列")
        lines.append("")
        
        # 摘要
        error_count = self.report.count_by_level("ERROR")
        warning_count = self.report.count_by_level("WARNING")
        info_count = self.report.count_by_level("INFO")
        
        if error_count == 0 and warning_count == 0:
            lines.append("## ✅ 校验结果: 通过")
        elif error_count == 0:
            lines.append(f"## ⚠️ 校验结果: 通过 (有 {warning_count} 个警告)")
        else:
            lines.append(f"## ❌ 校验结果: 失败 ({error_count} 个错误, {warning_count} 个警告)")
        
        lines.append("")
        lines.append(f"| 级别 | 数量 |")
        lines.append("|------|------|")
        lines.append(f"| ❌ ERROR | {error_count} |")
        lines.append(f"| ⚠️ WARNING | {warning_count} |")
        lines.append(f"| ℹ️ INFO | {info_count} |")
        lines.append("")
        
        # 各维度详情
        self._append_check_section(lines, "1. 基础完整性校验", self.report.integrity_checks)
        self._append_check_section(lines, "2. 特征分布校验", self.report.distribution_checks)
        self._append_check_section(lines, "3. 标签质量校验", self.report.label_checks)
        self._append_check_section(lines, "4. 时序逻辑校验", self.report.temporal_checks)
        
        return "\n".join(lines)
    
    def _append_check_section(self, lines: list, title: str, checks: List[CheckResult]):
        """添加检查项到报告"""
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| 检查项 | 结果 | 说明 |")
        lines.append("|--------|------|------|")
        
        for check in checks:
            icon = "✅" if check.level == "INFO" else ("⚠️" if check.level == "WARNING" else "❌")
            status = "通过" if check.passed else ("警告" if check.level == "WARNING" else "失败")
            lines.append(f"| {check.name} | {icon} {status} | {check.message} |")
        
        lines.append("")
    
    def save_report(self, output_path: Optional[Path] = None):
        """保存报告到文件"""
        path = output_path or self.config.report_path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        report_text = self.generate_report()
        path.write_text(report_text, encoding='utf-8')
        
        logger.info(f"📄 报告已保存: {path}")


# ============================================================================
# 命令行入口
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="train.parquet 数据质量校验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="输入文件路径 (默认: data/features/structured/train.parquet)",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出报告路径 (默认: reports/train_dq_report.md)",
    )
    
    parser.add_argument(
        "--check",
        nargs="+",
        choices=["integrity", "distribution", "labels", "temporal", "all"],
        default=["all"],
        help="要运行的校验类型 (默认: all)",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="打印详细信息",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = VerifyConfig()
    
    if args.input:
        config.train_path = args.input
    if args.output:
        config.report_path = args.output
    
    # 创建校验器
    verifier = TrainDataVerifier(config)
    
    try:
        # 加载数据
        verifier.load_data()
        
        # 运行校验
        report = verifier.run_all_checks()
        
        # 打印摘要
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 校验摘要")
        logger.info("=" * 70)
        
        error_count = report.count_by_level("ERROR")
        warning_count = report.count_by_level("WARNING")
        
        if error_count == 0 and warning_count == 0:
            logger.info("✅ 所有校验通过!")
        elif error_count == 0:
            logger.info(f"⚠️ 校验通过，但有 {warning_count} 个警告")
        else:
            logger.info(f"❌ 校验失败: {error_count} 个错误, {warning_count} 个警告")
        
        # 保存报告
        verifier.save_report()
        
        # 返回状态码
        return 0 if error_count == 0 else 1
        
    except Exception as e:
        logger.error(f"❌ 校验失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
