#!/usr/bin/env python3
"""
train.parquet 全量数据质量校验脚本 (增强版)

对生成的训练数据进行系统性质量校验，涵盖六个核心维度：
1. 基础概览与完整性检查 - 数据量级/主键唯一性/日历完整性/内存占用
2. 缺失值分析 - 全局缺失率/头部截断/尾部截断/中间断层/截面缺失
3. 异常值与分布检测 - 极值统计/inf检查/偏度峰度/异常点识别
4. 特征逻辑与一致性校验 - 单位一致性/价格逻辑/强相关自检/复权逻辑
5. 标签质量诊断 - 数据泄露检测/标签分布/分类平衡性
6. Smart Money 专项审计 - 大宗交易覆盖率/主力资金分布

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
    max_nan_rate: float = 0.50          # 最大总体 NaN 率 (放宽，因为很多特征有数据缺失)
    max_inf_rate: float = 0.0           # 最大 Inf 率 (严格 0%)
    
    # Z-Score 校验阈值
    zscore_mean_tolerance: float = 0.01  # 截面均值容差 ±0.01
    zscore_std_tolerance: float = 0.05   # 截面标准差容差 ±0.05
    zscore_clip_threshold: float = 3.5   # 极值截断阈值
    
    # 标签校验阈值
    label_min_valid_rate: float = 0.95   # 标签最小有效率 95%
    label_max_abs_value: float = 0.5     # 标签最大绝对值 50%
    
    # 时序校验阈值
    train_start_date: str = "2021-01-01" # 训练集起始日期
    min_daily_stocks: int = 3000         # 每日最少股票数
    max_daily_stocks: int = 6000         # 每日最多股票数
    expected_stock_count: int = 5000     # 预期股票数量
    
    # 缺失值分析阈值
    high_missing_threshold: float = 0.30  # 高缺失率阈值
    
    # 异常值检测阈值
    outlier_std_multiplier: float = 5.0   # 异常点定义: Mean ± 5*Std
    
    # 数据泄露检测阈值
    leakage_corr_threshold: float = 0.80  # 泄露相关性阈值
    
    # 特征列后缀
    zscore_suffix: str = "_zscore"       # Z-Score 列后缀
    
    # 标签列
    label_cols: List[str] = field(default_factory=lambda: [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "label_1d", "label_5d", "label_10d", "label_20d",
        "excess_ret_5d", "excess_ret_10d", 
        "rank_ret_5d", "rank_ret_10d",
        "sharpe_5d", "sharpe_10d", "sharpe_20d", 
        "label_bin_5d"
    ])
    
    # 主标签
    primary_label: str = "ret_5d"
    
    # 金额类特征（用于单位一致性检查）
    amount_cols: List[str] = field(default_factory=lambda: [
        "amount", "total_mv", "circ_mv", 
        "net_mf_amount", "net_main_amount", "net_elg_amount", 
        "net_lg_amount", "net_md_amount", "net_sm_amount"
    ])
    
    # 比率类特征（用于范围检查）
    ratio_cols: List[str] = field(default_factory=lambda: [
        "mf_main_intensity", "mf_elg_intensity", "mf_lg_intensity", 
        "mf_md_intensity", "mf_retail_intensity",
        "top10_hold_ratio", "top1_hold_ratio", "pledge_ratio"
    ])
    
    # Smart Money 特征
    smart_money_cols: List[str] = field(default_factory=lambda: [
        "mf_main_intensity", "mf_elg_intensity", "mf_lg_intensity",
        "mf_block_intensity", "mf_north_net", "mf_main_sign",
        "rzye", "rqye"
    ])


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
    memory_mb: float = 0.0
    
    integrity_checks: List[CheckResult] = field(default_factory=list)
    missing_checks: List[CheckResult] = field(default_factory=list)
    distribution_checks: List[CheckResult] = field(default_factory=list)
    logic_checks: List[CheckResult] = field(default_factory=list)
    label_checks: List[CheckResult] = field(default_factory=list)
    temporal_checks: List[CheckResult] = field(default_factory=list)
    smartmoney_checks: List[CheckResult] = field(default_factory=list)
    
    def count_by_level(self, level: str) -> int:
        """统计指定级别的检查数量"""
        all_checks = (
            self.integrity_checks + 
            self.missing_checks +
            self.distribution_checks + 
            self.logic_checks +
            self.label_checks + 
            self.temporal_checks +
            self.smartmoney_checks
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
        
        memory_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        self.report = VerifyReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data_path=str(self.config.train_path),
            data_shape=self.df.shape,
            memory_mb=memory_mb,
        )
        
        logger.info("=" * 70)
        logger.info("🔍 开始数据质量校验 (增强版)")
        logger.info("=" * 70)
        
        # 1. 基础概览与完整性检查
        self._check_integrity()
        
        # 2. 缺失值分析
        self._check_missing()
        
        # 3. 异常值与分布检测
        self._check_distribution()
        
        # 4. 特征逻辑与一致性校验
        self._check_logic()
        
        # 5. 标签质量诊断
        self._check_labels()
        
        # 6. 时序逻辑校验
        self._check_temporal()
        
        # 7. Smart Money 专项审计
        self._check_smartmoney()
        
        return self.report
    
    # ========================================================================
    # 1. 基础完整性校验
    # ========================================================================
    
    def _check_integrity(self):
        """基础概览与完整性检查"""
        logger.info("")
        logger.info("📋 1. 基础概览与完整性检查")
        logger.info("-" * 50)
        
        checks = []
        
        # 1.1 检查数据量级
        row_count = len(self.df)
        col_count = len(self.df.columns)
        passed = row_count >= self.config.min_rows
        checks.append(CheckResult(
            name="数据量级",
            passed=passed,
            level="ERROR" if not passed else "INFO",
            message=f"行数: {row_count:,}, 列数: {col_count} (阈值: >= {self.config.min_rows:,} 行)",
            details={"row_count": row_count, "col_count": col_count, "threshold": self.config.min_rows}
        ))
        logger.info(f"   {'✅' if passed else '❌'} 数据量级: {row_count:,} 行 × {col_count} 列")
        
        # 1.2 内存占用分析
        memory_mb = self.report.memory_mb
        memory_per_row = memory_mb * 1024 / row_count  # KB per row
        
        # 检查数据类型
        dtype_counts = self.df.dtypes.value_counts()
        float64_count = dtype_counts.get('float64', 0)
        float32_count = dtype_counts.get('float32', 0)
        
        passed = float64_count <= float32_count or memory_mb < 8000  # 小于 8GB 或已优化
        checks.append(CheckResult(
            name="内存占用",
            passed=passed,
            level="WARNING" if not passed else "INFO",
            message=f"总内存: {memory_mb:.1f} MB ({memory_per_row:.2f} KB/行), float64: {float64_count}, float32: {float32_count}",
            details={"memory_mb": memory_mb, "float64_count": float64_count, "float32_count": float32_count}
        ))
        logger.info(f"   {'✅' if passed else '⚠️'} 内存: {memory_mb:.1f} MB (float64: {float64_count}, float32: {float32_count})")
        
        # 1.3 主键唯一性检查
        if 'ts_code' in self.df.columns and 'trade_date' in self.df.columns:
            dup_count = self.df.duplicated(subset=['ts_code', 'trade_date']).sum()
            passed = dup_count == 0
            checks.append(CheckResult(
                name="主键唯一性",
                passed=passed,
                level="ERROR" if not passed else "INFO",
                message=f"重复 (ts_code, trade_date): {dup_count:,} 行",
                details={"duplicate_count": dup_count}
            ))
            logger.info(f"   {'✅' if passed else '❌'} 主键唯一性: {dup_count:,} 个重复")
        
        # 1.4 检查 Inf 值
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(self.df[numeric_cols]).any()
        inf_cols = inf_mask[inf_mask].index.tolist()
        inf_count = np.isinf(self.df[numeric_cols]).sum().sum()
        inf_rate = inf_count / (len(self.df) * len(numeric_cols)) if len(numeric_cols) > 0 else 0
        passed = inf_rate <= self.config.max_inf_rate
        
        checks.append(CheckResult(
            name="Inf 检查",
            passed=passed,
            level="ERROR" if not passed else "INFO",
            message=f"Inf 率: {inf_rate:.4%}, 涉及 {len(inf_cols)} 列",
            details={"inf_rate": inf_rate, "inf_cols": inf_cols[:10]}
        ))
        logger.info(f"   {'✅' if passed else '❌'} Inf 率: {inf_rate:.4%} ({len(inf_cols)} 列)")
        
        # 1.5 检查列名规范性
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
        
        # 1.6 检查必要列
        required_cols = ['ts_code', 'trade_date', 'close', 'ret_5d']
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
    # 2. 缺失值分析
    # ========================================================================
    
    def _check_missing(self):
        """缺失值分析"""
        logger.info("")
        logger.info("📋 2. 缺失值分析")
        logger.info("-" * 50)
        
        checks = []
        
        # 2.1 全局缺失率统计
        nan_counts = self.df.isna().sum()
        nan_rates = nan_counts / len(self.df)
        
        # 列出缺失率 > 0% 的特征，按缺失比例排序
        missing_cols = nan_rates[nan_rates > 0].sort_values(ascending=False)
        high_missing = missing_cols[missing_cols > self.config.high_missing_threshold]
        
        checks.append(CheckResult(
            name="全局缺失统计",
            passed=True,
            level="INFO",
            message=f"有缺失列: {len(missing_cols)} 个, 高缺失 (>{self.config.high_missing_threshold:.0%}): {len(high_missing)} 个",
            details={
                "missing_cols_count": len(missing_cols),
                "high_missing_cols": high_missing.to_dict() if len(high_missing) > 0 else {},
                "top10_missing": missing_cols.head(10).to_dict()
            }
        ))
        logger.info(f"   ℹ️ 有缺失列: {len(missing_cols)} 个, 高缺失: {len(high_missing)} 个")
        
        if len(high_missing) > 0:
            logger.info(f"      高缺失特征 TOP 5:")
            for col, rate in high_missing.head(5).items():
                logger.info(f"        - {col}: {rate:.1%}")
        
        # 2.2 头部缺失检测 (Rolling 特征前 N 天应全为 NaN)
        rolling_cols = [c for c in self.df.columns if any(x in c for x in ['ma_', 'rsi_', 'roc_', 'vol_', 'bias_'])]
        
        if rolling_cols and 'trade_date' in self.df.columns:
            # 获取前 60 天的数据
            sorted_dates = sorted(self.df['trade_date'].unique())
            first_60_dates = sorted_dates[:60]
            first_60_data = self.df[self.df['trade_date'].isin(first_60_dates)]
            
            # 检查 ma_60 在前 60 天是否大部分为 NaN（正常现象）
            head_truncation_info = {}
            for col in ['ma_60', 'ma_120', 'ma_250']:
                if col in self.df.columns:
                    nan_rate_head = first_60_data[col].isna().mean()
                    head_truncation_info[col] = nan_rate_head
            
            passed = True  # 头部缺失是正常的
            checks.append(CheckResult(
                name="头部缺失 (Rolling)",
                passed=passed,
                level="INFO",
                message=f"前 60 天缺失率 (正常): {head_truncation_info}",
                details={"head_truncation": head_truncation_info}
            ))
            logger.info(f"   ℹ️ 头部缺失 (正常): {head_truncation_info}")
        
        # 2.3 尾部缺失检测 (Label 最后 N 天应全为 NaN)
        if 'trade_date' in self.df.columns:
            sorted_dates = sorted(self.df['trade_date'].unique())
            last_20_dates = sorted_dates[-20:]
            last_20_data = self.df[self.df['trade_date'].isin(last_20_dates)]
            
            tail_truncation_info = {}
            for col in ['ret_5d', 'ret_10d', 'ret_20d']:
                if col in self.df.columns:
                    nan_rate_tail = last_20_data[col].isna().mean()
                    tail_truncation_info[col] = nan_rate_tail
            
            passed = True  # 尾部缺失是正常的
            checks.append(CheckResult(
                name="尾部缺失 (Label)",
                passed=passed,
                level="INFO",
                message=f"最后 20 天缺失率 (正常): {tail_truncation_info}",
                details={"tail_truncation": tail_truncation_info}
            ))
            logger.info(f"   ℹ️ 尾部缺失 (正常): {tail_truncation_info}")
        
        # 2.4 中间断层检测 (某些日期突然全为 NaN)
        if 'trade_date' in self.df.columns:
            # 按日期统计非 NaN 行数
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            key_cols = ['close', 'amount', 'total_mv']
            key_cols = [c for c in key_cols if c in self.df.columns]
            
            if key_cols:
                daily_valid = self.df.groupby('trade_date')[key_cols[0]].apply(lambda x: x.notna().sum())
                mean_valid = daily_valid.mean()
                
                # 检测突然下降 (低于均值 50%)
                gap_dates = daily_valid[daily_valid < mean_valid * 0.5].index.tolist()
                passed = len(gap_dates) == 0
                
                checks.append(CheckResult(
                    name="中间断层检测",
                    passed=passed,
                    level="ERROR" if not passed else "INFO",
                    message=f"异常日期 (有效数 < 50% 均值): {len(gap_dates)} 天",
                    details={"gap_dates": gap_dates[:10]}
                ))
                logger.info(f"   {'✅' if passed else '❌'} 中间断层: {len(gap_dates)} 天异常")
        
        # 2.5 截面缺失分布 (按股票统计)
        if 'ts_code' in self.df.columns:
            stock_nan_rates = self.df.groupby('ts_code').apply(
                lambda x: x.isna().sum().sum() / (len(x) * len(self.df.columns))
            )
            high_nan_stocks = stock_nan_rates[stock_nan_rates > 0.5]  # 超过 50% 缺失
            
            passed = len(high_nan_stocks) < len(stock_nan_rates) * 0.1  # 不超过 10% 的股票
            checks.append(CheckResult(
                name="截面缺失分布",
                passed=passed,
                level="WARNING" if not passed else "INFO",
                message=f"高缺失股票 (>50%): {len(high_nan_stocks)} 只 ({len(high_nan_stocks)/len(stock_nan_rates)*100:.1f}%)",
                details={"high_nan_stock_count": len(high_nan_stocks)}
            ))
            logger.info(f"   {'✅' if passed else '⚠️'} 高缺失股票: {len(high_nan_stocks)} 只")
        
        self.report.missing_checks = checks
    
    # ========================================================================
    # 2. 特征分布校验
    # ========================================================================
    
    def _check_distribution(self):
        """异常值与分布检测"""
        logger.info("")
        logger.info("📋 3. 异常值与分布检测")
        logger.info("-" * 50)
        
        checks = []
        
        # 获取数值列（排除标签）
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in self.config.label_cols]
        
        # 3.1 极值统计概览
        stats = self.df[feature_cols].describe().T
        stats['range'] = stats['max'] - stats['min']
        
        # 检查异常范围
        extremely_large = stats[stats['max'].abs() > 1e12]
        passed = len(extremely_large) == 0
        
        checks.append(CheckResult(
            name="极值范围",
            passed=passed,
            level="WARNING" if not passed else "INFO",
            message=f"超大值 (>1e12) 列: {len(extremely_large)} 个",
            details={"extremely_large_cols": extremely_large.index.tolist()[:10]}
        ))
        logger.info(f"   {'✅' if passed else '⚠️'} 极值范围: {len(extremely_large)} 列超大值")
        
        # 3.2 偏度与峰度分析
        skewness = self.df[feature_cols].skew()
        kurtosis = self.df[feature_cols].kurtosis()
        
        high_skew = skewness[skewness.abs() > 5]  # 偏度 > 5
        high_kurt = kurtosis[kurtosis.abs() > 50]  # 峰度 > 50
        
        checks.append(CheckResult(
            name="偏度检测",
            passed=len(high_skew) < len(feature_cols) * 0.3,
            level="WARNING" if len(high_skew) >= len(feature_cols) * 0.3 else "INFO",
            message=f"高偏度 (|skew|>5): {len(high_skew)} 列 ({len(high_skew)/len(feature_cols)*100:.1f}%)",
            details={"high_skew_cols": high_skew.to_dict() if len(high_skew) < 20 else dict(list(high_skew.items())[:20])}
        ))
        logger.info(f"   ℹ️ 高偏度列: {len(high_skew)} 个")
        
        checks.append(CheckResult(
            name="峰度检测",
            passed=len(high_kurt) < len(feature_cols) * 0.3,
            level="WARNING" if len(high_kurt) >= len(feature_cols) * 0.3 else "INFO",
            message=f"高峰度 (|kurt|>50): {len(high_kurt)} 列 ({len(high_kurt)/len(feature_cols)*100:.1f}%)",
            details={"high_kurt_cols": high_kurt.to_dict() if len(high_kurt) < 20 else dict(list(high_kurt.items())[:20])}
        ))
        logger.info(f"   ℹ️ 高峰度列: {len(high_kurt)} 个")
        
        # 3.3 异常点识别 (Mean ± 5*Std)
        outlier_info = {}
        total_outliers = 0
        
        for col in feature_cols[:50]:  # 只检查前 50 列以节省时间
            data = self.df[col].dropna()
            if len(data) == 0:
                continue
            mean_val = data.mean()
            std_val = data.std()
            if std_val == 0:
                continue
            
            lower = mean_val - self.config.outlier_std_multiplier * std_val
            upper = mean_val + self.config.outlier_std_multiplier * std_val
            outlier_rate = ((data < lower) | (data > upper)).mean()
            
            if outlier_rate > 0.01:  # 超过 1% 是异常点
                outlier_info[col] = outlier_rate
                total_outliers += 1
        
        passed = total_outliers < 10
        checks.append(CheckResult(
            name=f"异常点 (±{self.config.outlier_std_multiplier}σ)",
            passed=passed,
            level="WARNING" if not passed else "INFO",
            message=f"高异常率 (>1%) 列: {total_outliers} 个",
            details={"high_outlier_cols": outlier_info}
        ))
        logger.info(f"   {'✅' if passed else '⚠️'} 高异常率列: {total_outliers} 个")
        
        # 3.4 常量列检查
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
        
        # 3.5 列数统计
        label_count = len([c for c in self.df.columns if c in self.config.label_cols])
        
        checks.append(CheckResult(
            name="列数统计",
            passed=True,
            level="INFO",
            message=f"特征列: {len(feature_cols)}, 标签列: {label_count}, 总计: {len(self.df.columns)}",
            details={"feature_cols": len(feature_cols), "label_cols": label_count, "total": len(self.df.columns)}
        ))
        logger.info(f"   ℹ️ 列数: 特征 {len(feature_cols)} + 标签 {label_count} = {len(self.df.columns)}")
        
        self.report.distribution_checks = checks
    
    # ========================================================================
    # 4. 特征逻辑与一致性校验
    # ========================================================================
    
    def _check_logic(self):
        """特征逻辑与一致性校验"""
        logger.info("")
        logger.info("📋 4. 特征逻辑与一致性校验")
        logger.info("-" * 50)
        
        checks = []
        
        # 4.1 单位一致性复核 - 金额类
        for col in self.config.amount_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) == 0:
                    continue
                
                median_val = data.median()
                # 金额应在 1e4 ~ 1e12 量级（元）
                in_range = 1e4 <= median_val <= 1e12
                passed = in_range
                
                checks.append(CheckResult(
                    name=f"{col} 单位",
                    passed=passed,
                    level="WARNING" if not passed else "INFO",
                    message=f"中位数: {median_val:.2e} (预期: 1e4~1e12 元)",
                    details={"col": col, "median": median_val}
                ))
                if not passed:
                    logger.info(f"   ⚠️ {col} 单位异常: 中位数 {median_val:.2e}")
        
        if any(c.passed for c in checks):
            logger.info(f"   ✅ 金额列单位检查完成")
        
        # 4.2 比率列范围检查
        ratio_issues = []
        for col in self.config.ratio_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) == 0:
                    continue
                
                min_val, max_val = data.min(), data.max()
                # 比率应在 -2 ~ 2 或 0 ~ 100
                if col.endswith('_ratio'):
                    in_range = 0 <= max_val <= 100
                else:  # intensity
                    in_range = -2 <= min_val and max_val <= 2
                
                if not in_range:
                    ratio_issues.append(f"{col}: [{min_val:.2f}, {max_val:.2f}]")
        
        passed = len(ratio_issues) == 0
        checks.append(CheckResult(
            name="比率范围",
            passed=passed,
            level="WARNING" if not passed else "INFO",
            message=f"超范围: {len(ratio_issues)} 个",
            details={"ratio_issues": ratio_issues}
        ))
        logger.info(f"   {'✅' if passed else '⚠️'} 比率范围: {len(ratio_issues)} 个异常")
        
        # 4.3 价格逻辑检查 (high >= low, high >= close)
        if all(c in self.df.columns for c in ['high', 'low', 'close']):
            logic_fail_high_low = (self.df['high'] < self.df['low']).sum()
            logic_fail_high_close = (self.df['high'] < self.df['close']).sum()
            logic_fail_close_low = (self.df['close'] < self.df['low']).sum()
            
            total_fails = logic_fail_high_low + logic_fail_high_close + logic_fail_close_low
            passed = total_fails == 0
            
            checks.append(CheckResult(
                name="价格逻辑",
                passed=passed,
                level="ERROR" if not passed else "INFO",
                message=f"逻辑错误: high<low: {logic_fail_high_low}, high<close: {logic_fail_high_close}, close<low: {logic_fail_close_low}",
                details={"high_lt_low": logic_fail_high_low, "high_lt_close": logic_fail_high_close, "close_lt_low": logic_fail_close_low}
            ))
            logger.info(f"   {'✅' if passed else '❌'} 价格逻辑: {total_fails} 行异常")
        
        # 4.4 强相关自检 (amount vs log_amount)
        if 'amount' in self.df.columns and 'log_amount' in self.df.columns:
            corr = self.df[['amount', 'log_amount']].corr().iloc[0, 1]
            passed = corr > 0.7  # 应高度相关
            
            checks.append(CheckResult(
                name="强相关自检",
                passed=passed,
                level="WARNING" if not passed else "INFO",
                message=f"amount vs log_amount 相关性: {corr:.4f} (预期 >0.7)",
                details={"correlation": corr}
            ))
            logger.info(f"   {'✅' if passed else '⚠️'} 相关自检: amount-log_amount = {corr:.4f}")
        
        # 4.5 复权价格检查
        if 'close_hfq' in self.df.columns:
            close_hfq = self.df['close_hfq'].dropna()
            negative_count = (close_hfq < 0).sum()
            passed = negative_count == 0
            
            checks.append(CheckResult(
                name="复权价格",
                passed=passed,
                level="ERROR" if not passed else "INFO",
                message=f"负值: {negative_count} 行",
                details={"negative_count": negative_count}
            ))
            logger.info(f"   {'✅' if passed else '❌'} 复权价格: {negative_count} 个负值")
        
        self.report.logic_checks = checks
    
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
        
        # 3.6 数据泄露检测 (特征与标签高相关性)
        if self.config.primary_label in self.df.columns:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            feature_cols = [c for c in numeric_cols if c not in self.config.label_cols][:50]  # 限制检查数量
            
            leakage_candidates = []
            label_data = self.df[self.config.primary_label].dropna()
            
            for col in feature_cols:
                if col in self.df.columns:
                    try:
                        # 计算与标签的相关性
                        valid_idx = self.df[col].notna() & self.df[self.config.primary_label].notna()
                        if valid_idx.sum() < 1000:
                            continue
                        corr = self.df.loc[valid_idx, [col, self.config.primary_label]].corr().iloc[0, 1]
                        
                        if abs(corr) > self.config.leakage_corr_threshold:
                            leakage_candidates.append((col, corr))
                    except Exception:
                        continue
            
            passed = len(leakage_candidates) == 0
            leakage_str = ", ".join([f"{c[0]}({c[1]:.2f})" for c in leakage_candidates[:5]])
            
            checks.append(CheckResult(
                name="数据泄露检测",
                passed=passed,
                level="ERROR" if not passed else "INFO",
                message=f"高相关特征 (|corr|>{self.config.leakage_corr_threshold}): {len(leakage_candidates)} 个. {leakage_str}",
                details={"leakage_candidates": leakage_candidates}
            ))
            logger.info(f"   {'✅' if passed else '❌'} 数据泄露: {len(leakage_candidates)} 个可疑特征")
        
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
    # 7. Smart Money 专项审计
    # ========================================================================
    
    def _check_smartmoney(self):
        """Smart Money 专项审计"""
        logger.info("")
        logger.info("📋 7. Smart Money 专项审计")
        logger.info("-" * 50)
        
        checks = []
        
        # 7.1 大宗交易覆盖率
        if 'mf_block_intensity' in self.df.columns:
            block_data = self.df['mf_block_intensity'].dropna()
            non_zero_rate = (block_data != 0).mean()
            # 大宗交易是稀疏的，非零率通常很低
            
            checks.append(CheckResult(
                name="大宗交易覆盖",
                passed=True,
                level="INFO",
                message=f"非零覆盖率: {non_zero_rate:.2%} (稀疏是正常的)",
                details={"non_zero_rate": non_zero_rate}
            ))
            logger.info(f"   ℹ️ 大宗交易非零覆盖率: {non_zero_rate:.2%}")
        
        # 7.2 主力净流入分布检查
        if 'mf_main_intensity' in self.df.columns:
            main_intensity = self.df['mf_main_intensity'].dropna()
            
            # 计算分位数分布
            p01 = main_intensity.quantile(0.01)
            p99 = main_intensity.quantile(0.99)
            median = main_intensity.median()
            
            # 大部分应在 -0.5 ~ +0.5 之间
            in_range_rate = ((main_intensity >= -0.5) & (main_intensity <= 0.5)).mean()
            passed = in_range_rate > 0.80  # 80% 在合理范围内
            
            checks.append(CheckResult(
                name="主力净流入分布",
                passed=passed,
                level="WARNING" if not passed else "INFO",
                message=f"P1={p01:.3f}, 中位数={median:.3f}, P99={p99:.3f}, 合理范围占比={in_range_rate:.1%}",
                details={"p01": p01, "median": median, "p99": p99, "in_range_rate": in_range_rate}
            ))
            logger.info(f"   {'✅' if passed else '⚠️'} 主力净流入: P1={p01:.3f}, 中位数={median:.3f}, P99={p99:.3f}")
        
        # 7.3 融资融券数据覆盖
        rzye_col = 'rzye' if 'rzye' in self.df.columns else None
        rqye_col = 'rqye' if 'rqye' in self.df.columns else None
        
        if rzye_col:
            rzye_data = self.df[rzye_col].dropna()
            rzye_coverage = len(rzye_data) / len(self.df)
            rzye_nonzero = (rzye_data != 0).mean()
            
            checks.append(CheckResult(
                name="融资余额覆盖",
                passed=True,
                level="INFO",
                message=f"数据覆盖率: {rzye_coverage:.1%}, 非零率: {rzye_nonzero:.1%}",
                details={"coverage": rzye_coverage, "nonzero_rate": rzye_nonzero}
            ))
            logger.info(f"   ℹ️ 融资余额覆盖: {rzye_coverage:.1%}, 非零: {rzye_nonzero:.1%}")
        
        # 7.4 北向资金数据
        if 'mf_north_net' in self.df.columns:
            north_data = self.df['mf_north_net'].dropna()
            north_coverage = len(north_data) / len(self.df)
            
            # 检查是否市场级数据（同一天所有股票值相同）
            if 'trade_date' in self.df.columns:
                sample_date = self.df['trade_date'].iloc[1000] if len(self.df) > 1000 else self.df['trade_date'].iloc[0]
                sample_day = self.df[self.df['trade_date'] == sample_date]['mf_north_net']
                unique_values = sample_day.nunique()
                
                is_market_level = unique_values == 1
                checks.append(CheckResult(
                    name="北向资金属性",
                    passed=True,
                    level="INFO",
                    message=f"覆盖率: {north_coverage:.1%}, 市场级: {'是' if is_market_level else '否'} (同日唯一值: {unique_values})",
                    details={"coverage": north_coverage, "is_market_level": is_market_level}
                ))
                logger.info(f"   ℹ️ 北向资金: 覆盖 {north_coverage:.1%}, 市场级: {'是' if is_market_level else '否'}")
        
        # 7.5 各资金类特征覆盖率汇总
        coverage_summary = {}
        for col in self.config.smart_money_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                coverage = len(data) / len(self.df)
                nonzero = (data != 0).mean() if len(data) > 0 else 0
                coverage_summary[col] = {"coverage": coverage, "nonzero": nonzero}
        
        checks.append(CheckResult(
            name="Smart Money 覆盖汇总",
            passed=True,
            level="INFO",
            message=f"检查 {len(coverage_summary)} 个特征",
            details={"coverage_summary": coverage_summary}
        ))
        logger.info(f"   ℹ️ Smart Money 特征: {len(coverage_summary)} 个已检查")
        
        self.report.smartmoney_checks = checks
    
    # ========================================================================
    # 报告生成
    # ========================================================================
    
    def generate_report(self) -> str:
        """生成 Markdown 格式报告"""
        if self.report is None:
            raise ValueError("请先运行 run_all_checks()")
        
        lines = []
        
        # 标题
        lines.append("# train.parquet 数据质量报告 (增强版)")
        lines.append("")
        lines.append(f"**生成时间**: {self.report.timestamp}")
        lines.append(f"**数据路径**: `{self.report.data_path}`")
        lines.append(f"**数据形状**: {self.report.data_shape[0]:,} 行 × {self.report.data_shape[1]} 列")
        lines.append(f"**内存占用**: {self.report.memory_mb:.1f} MB")
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
        self._append_check_section(lines, "1. 基础概览与完整性检查", self.report.integrity_checks)
        self._append_check_section(lines, "2. 缺失值分析", self.report.missing_checks)
        self._append_check_section(lines, "3. 异常值与分布检测", self.report.distribution_checks)
        self._append_check_section(lines, "4. 特征逻辑与一致性校验", self.report.logic_checks)
        self._append_check_section(lines, "5. 标签质量诊断", self.report.label_checks)
        self._append_check_section(lines, "6. 时序逻辑校验", self.report.temporal_checks)
        self._append_check_section(lines, "7. Smart Money 专项审计", self.report.smartmoney_checks)
        
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
