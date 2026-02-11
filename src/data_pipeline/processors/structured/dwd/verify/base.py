"""
DWD数据质量检查器基类

定义检查结果结构、严重程度枚举、基类检查器方法
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _convert_to_native(value: Any) -> Any:
    """将numpy类型转换为Python原生类型，用于JSON序列化"""
    if value is None:
        return None
    if isinstance(value, (np.bool_, np.integer)):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _convert_to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert_to_native(v) for v in value]
    return value


class CheckSeverity(Enum):
    """检查结果严重程度"""
    
    PASS = "PASS"          # 通过
    INFO = "INFO"          # 信息（需关注）
    WARNING = "WARNING"    # 警告（需审查）
    ERROR = "ERROR"        # 错误（需修复）
    CRITICAL = "CRITICAL"  # 严重（阻断性问题）


@dataclass
class CheckResult:
    """单项检查结果"""
    
    check_name: str                    # 检查项名称
    severity: CheckSeverity            # 严重程度
    passed: bool                       # 是否通过
    message: str                       # 结果描述
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息
    metric_name: Optional[str] = None  # 指标名称（用于报告）
    metric_value: Optional[Any] = None # 指标值
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典（支持JSON序列化）"""
        return {
            "check_name": self.check_name,
            "severity": self.severity.value,
            "passed": bool(self.passed),
            "message": self.message,
            "details": _convert_to_native(self.details),
            "metric_name": self.metric_name,
            "metric_value": _convert_to_native(self.metric_value),
        }


@dataclass
class ColumnStats:
    """列统计信息"""
    
    column_name: str
    dtype: str
    total_count: int
    null_count: int
    null_rate: float
    unique_count: int
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    median_value: Optional[float] = None
    q25_value: Optional[float] = None
    q75_value: Optional[float] = None
    sample_values: Optional[List[Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_name": self.column_name,
            "dtype": self.dtype,
            "total_count": int(self.total_count),
            "null_count": int(self.null_count),
            "null_rate": f"{float(self.null_rate):.4%}",
            "unique_count": int(self.unique_count),
            "min": _convert_to_native(self.min_value),
            "max": _convert_to_native(self.max_value),
            "mean": _convert_to_native(self.mean_value),
            "std": _convert_to_native(self.std_value),
            "median": _convert_to_native(self.median_value),
            "q25": _convert_to_native(self.q25_value),
            "q75": _convert_to_native(self.q75_value),
        }


@dataclass
class TableCheckReport:
    """单表检查报告"""
    
    table_name: str
    check_time: datetime
    total_rows: int
    total_columns: int
    date_range: Tuple[str, str]
    check_results: List[CheckResult] = field(default_factory=list)
    column_stats: List[ColumnStats] = field(default_factory=list)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.check_results if r.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.check_results if not r.passed)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for r in self.check_results 
                   if r.severity == CheckSeverity.CRITICAL)
    
    @property
    def error_count(self) -> int:
        return sum(1 for r in self.check_results 
                   if r.severity == CheckSeverity.ERROR)
    
    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.check_results 
                   if r.severity == CheckSeverity.WARNING)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "check_time": self.check_time.isoformat(),
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "date_range": self.date_range,
            "summary": {
                "total_checks": len(self.check_results),
                "passed": self.passed_count,
                "failed": self.failed_count,
                "critical": self.critical_count,
                "errors": self.error_count,
                "warnings": self.warning_count,
            },
            "check_results": [r.to_dict() for r in self.check_results],
            "column_stats": [s.to_dict() for s in self.column_stats],
        }


class BaseChecker(ABC):
    """
    DWD数据质量检查器基类
    
    子类需要实现:
    - table_name: 表名属性
    - run_specific_checks: 表特定的检查逻辑
    """
    
    # 数据目录
    DWD_DATA_DIR = Path("/home/menghuanghan/quant_system/data/processed/structured/dwd")
    
    # 预热期日期范围（2019-2020）
    WARMUP_START = "2019-01-01"
    WARMUP_END = "2020-12-31"
    
    # 训练期日期范围（2021-2025）
    TRAIN_START = "2021-01-01"
    TRAIN_END = "2025-12-31"
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        初始化检查器
        
        Args:
            df: 待检查的DataFrame，如果为None则从文件加载
        """
        self._df = df
        self._results: List[CheckResult] = []
        self._column_stats: List[ColumnStats] = []
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """表名"""
        raise NotImplementedError
    
    @property
    def file_path(self) -> Path:
        """parquet文件路径"""
        return self.DWD_DATA_DIR / f"{self.table_name}.parquet"
    
    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        if self._df is not None:
            return self._df
        
        logger.info(f"加载数据: {self.file_path}")
        self._df = pd.read_parquet(self.file_path)
        return self._df
    
    @property
    def df(self) -> pd.DataFrame:
        """获取数据DataFrame"""
        if self._df is None:
            self.load_data()
        return self._df
    
    def add_result(
        self,
        check_name: str,
        passed: bool,
        message: str,
        severity: Optional[CheckSeverity] = None,
        details: Optional[Dict[str, Any]] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[Any] = None,
    ):
        """添加检查结果"""
        if severity is None:
            severity = CheckSeverity.PASS if passed else CheckSeverity.ERROR
        
        result = CheckResult(
            check_name=check_name,
            severity=severity,
            passed=passed,
            message=message,
            details=details or {},
            metric_name=metric_name,
            metric_value=metric_value,
        )
        self._results.append(result)
        
        # 日志输出
        log_level = logging.INFO if passed else logging.WARNING
        if severity == CheckSeverity.CRITICAL:
            log_level = logging.ERROR
        logger.log(log_level, f"[{self.table_name}] {check_name}: {message}")
    
    def compute_column_stats(self, exclude_cols: Optional[List[str]] = None):
        """计算所有列的统计信息"""
        exclude_cols = exclude_cols or []
        
        for col in self.df.columns:
            if col in exclude_cols:
                continue
            
            series = self.df[col]
            total = len(series)
            null_count = series.isna().sum()
            dtype_str = str(series.dtype)
            
            stats = ColumnStats(
                column_name=col,
                dtype=dtype_str,
                total_count=total,
                null_count=int(null_count),
                null_rate=null_count / total if total > 0 else 0,
                unique_count=int(series.nunique()),
            )
            
            # 数值类型计算统计量
            if pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 0:
                    stats.min_value = float(non_null.min())
                    stats.max_value = float(non_null.max())
                    stats.mean_value = float(non_null.mean())
                    stats.std_value = float(non_null.std())
                    stats.median_value = float(non_null.median())
                    stats.q25_value = float(non_null.quantile(0.25))
                    stats.q75_value = float(non_null.quantile(0.75))
            else:
                # 非数值类型取样本值
                sample_size = min(5, stats.unique_count)
                if sample_size > 0:
                    stats.sample_values = series.dropna().unique()[:sample_size].tolist()
            
            self._column_stats.append(stats)
    
    @abstractmethod
    def run_specific_checks(self):
        """运行表特定的检查逻辑（子类实现）"""
        raise NotImplementedError
    
    def run(self) -> TableCheckReport:
        """运行完整检查"""
        logger.info(f"开始检查表: {self.table_name}")
        
        # 加载数据
        df = self.load_data()
        
        # 基础信息
        date_col = "trade_date"
        if date_col in df.columns:
            date_range = (df[date_col].min(), df[date_col].max())
        else:
            date_range = ("N/A", "N/A")
        
        # 运行特定检查
        self.run_specific_checks()
        
        # 计算列统计
        self.compute_column_stats()
        
        # 生成报告
        report = TableCheckReport(
            table_name=self.table_name,
            check_time=datetime.now(),
            total_rows=len(df),
            total_columns=len(df.columns),
            date_range=date_range,
            check_results=self._results,
            column_stats=self._column_stats,
        )
        
        logger.info(
            f"表 {self.table_name} 检查完成: "
            f"{report.passed_count} 通过, {report.failed_count} 失败, "
            f"{report.critical_count} 严重, {report.error_count} 错误, "
            f"{report.warning_count} 警告"
        )
        
        return report
    
    # ==========================================================================
    # 通用检查辅助方法
    # ==========================================================================
    
    def check_column_exists(self, columns: List[str]) -> bool:
        """检查列是否存在"""
        missing = [c for c in columns if c not in self.df.columns]
        if missing:
            self.add_result(
                check_name="列存在性检查",
                passed=False,
                message=f"缺少列: {missing}",
                severity=CheckSeverity.ERROR,
            )
            return False
        return True
    
    def check_null_rate(
        self,
        column: str,
        max_rate: float = 0.0,
        period: str = "all"
    ) -> CheckResult:
        """检查列缺失率"""
        if period == "warmup":
            mask = (self.df["trade_date"] >= self.WARMUP_START) & \
                   (self.df["trade_date"] <= self.WARMUP_END)
            period_desc = "预热期"
        elif period == "train":
            mask = (self.df["trade_date"] >= self.TRAIN_START) & \
                   (self.df["trade_date"] <= self.TRAIN_END)
            period_desc = "训练期"
        else:
            mask = pd.Series([True] * len(self.df))
            period_desc = "全时段"
        
        subset = self.df.loc[mask, column]
        null_rate = subset.isna().sum() / len(subset) if len(subset) > 0 else 0
        passed = null_rate <= max_rate
        
        self.add_result(
            check_name=f"{column} 缺失率检查 ({period_desc})",
            passed=passed,
            message=f"缺失率 {null_rate:.4%}, 阈值 {max_rate:.4%}",
            severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
            metric_name=f"{column}_null_rate_{period}",
            metric_value=null_rate,
        )
        return self._results[-1]
    
    def check_range(
        self,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        allow_nan: bool = True,
    ) -> CheckResult:
        """检查数值范围"""
        series = self.df[column].dropna() if allow_nan else self.df[column]
        
        violations = 0
        if min_val is not None:
            violations += (series < min_val).sum()
        if max_val is not None:
            violations += (series > max_val).sum()
        
        passed = violations == 0
        range_desc = []
        if min_val is not None:
            range_desc.append(f">= {min_val}")
        if max_val is not None:
            range_desc.append(f"<= {max_val}")
        
        self.add_result(
            check_name=f"{column} 范围检查",
            passed=passed,
            message=f"范围 [{', '.join(range_desc)}], 违规行数: {violations}",
            severity=CheckSeverity.PASS if passed else CheckSeverity.ERROR,
            details={"violations": int(violations)},
            metric_name=f"{column}_range_violations",
            metric_value=violations,
        )
        return self._results[-1]
    
    def check_non_negative(self, column: str) -> CheckResult:
        """检查非负"""
        return self.check_range(column, min_val=0)
    
    def check_positive(self, column: str) -> CheckResult:
        """检查正数"""
        series = self.df[column].dropna()
        violations = (series <= 0).sum()
        passed = violations == 0
        
        self.add_result(
            check_name=f"{column} 正数检查",
            passed=passed,
            message=f"非正数行数: {violations}",
            severity=CheckSeverity.PASS if passed else CheckSeverity.ERROR,
            metric_name=f"{column}_non_positive_count",
            metric_value=violations,
        )
        return self._results[-1]
    
    def check_mean_order_of_magnitude(
        self,
        column: str,
        expected_log10: float,
        tolerance: float = 1.0,
        description: str = "",
    ) -> CheckResult:
        """检查均值数量级"""
        mean_val = self.df[column].dropna().mean()
        actual_log10 = np.log10(abs(mean_val) + 1e-10)
        diff = abs(actual_log10 - expected_log10)
        passed = diff <= tolerance
        
        self.add_result(
            check_name=f"{column} 单位/量级检查",
            passed=passed,
            message=f"{description} 均值={mean_val:.2e}, 预期量级=1e{expected_log10:.0f}, "
                    f"实际量级=1e{actual_log10:.1f}",
            severity=CheckSeverity.PASS if passed else CheckSeverity.ERROR,
            details={
                "mean": float(mean_val),
                "expected_log10": expected_log10,
                "actual_log10": actual_log10,
            },
            metric_name=f"{column}_mean",
            metric_value=mean_val,
        )
        return self._results[-1]
    
    def check_unique_values(
        self,
        column: str,
        expected_values: Optional[List[Any]] = None,
        allow_extra: bool = False,
    ) -> CheckResult:
        """检查唯一值"""
        actual_values = set(self.df[column].dropna().unique())
        
        if expected_values is None:
            self.add_result(
                check_name=f"{column} 唯一值检查",
                passed=True,
                message=f"唯一值数量: {len(actual_values)}",
                severity=CheckSeverity.INFO,
                details={"unique_values": list(actual_values)[:20]},
            )
        else:
            expected_set = set(expected_values)
            missing = expected_set - actual_values
            extra = actual_values - expected_set
            
            passed = len(missing) == 0 and (allow_extra or len(extra) == 0)
            
            self.add_result(
                check_name=f"{column} 唯一值检查",
                passed=passed,
                message=f"预期值缺失: {missing}, 额外值: {extra}",
                severity=CheckSeverity.PASS if passed else CheckSeverity.WARNING,
                details={
                    "expected": list(expected_set),
                    "actual": list(actual_values),
                    "missing": list(missing),
                    "extra": list(extra),
                },
            )
        return self._results[-1]
