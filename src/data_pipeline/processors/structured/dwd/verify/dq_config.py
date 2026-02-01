"""
DWD 数据质量检查 - 基础配置和工具类

提供所有检查模块共用的配置、阈值和工具函数
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
import logging


# ============================================================================
# 路径配置
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]  # 项目根目录
DWD_DIR = BASE_DIR / "data" / "processed" / "structured" / "dwd"
RAW_DIR = BASE_DIR / "data" / "raw" / "structured"
REPORTS_DIR = BASE_DIR / "reports"

# 确保reports目录存在
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 数据文件路径
# ============================================================================
@dataclass
class DWDFilePaths:
    """DWD宽表文件路径"""
    stock_price: Path = DWD_DIR / "dwd_stock_price.parquet"
    stock_fundamental: Path = DWD_DIR / "dwd_stock_fundamental.parquet"
    stock_status: Path = DWD_DIR / "dwd_stock_status.parquet"
    
    # 原始数据参考
    trade_calendar: Path = RAW_DIR / "metadata" / "trade_calendar.parquet"
    stock_list: Path = RAW_DIR / "metadata" / "stock_list_a.parquet"


# ============================================================================
# 检查阈值配置
# ============================================================================
@dataclass
class QualityThresholds:
    """数据质量检查阈值"""
    
    # 价格检查
    RETURN_EXTREME_UPPER: float = 2.0      # 非首日涨幅上限200%
    RETURN_EXTREME_LOWER: float = -0.30    # 非首日跌幅下限-30%
    RETURN_NEW_STOCK_UPPER: float = 10.0   # 新股涨幅上限1000%
    RETURN_NORMAL_THRESHOLD: float = 0.30  # 正常涨跌幅阈值30%
    PRICE_CLOSE_HFQ_MIN: float = 0.0       # 后复权收盘价最小值
    RETURN_TOLERANCE: float = 1e-6         # 收益率验证容差
    
    # 基本面检查
    FUNDAMENTAL_COVERAGE_MIN: float = 0.95  # 基本面覆盖率最小值95%
    TTM_CHANGE_EXTREME: float = 5.0         # TTM指标突变阈值500%
    PE_TTM_TOLERANCE: float = 0.10          # PE_TTM验证容差10%
    
    # 状态检查
    NEW_STOCK_DAYS: int = 60               # 新股定义天数
    LIMIT_UP_TOLERANCE: float = 0.005      # 涨停判定容差0.5%
    
    # 涨跌停比例配置
    LIMIT_RATIOS: Dict[str, float] = field(default_factory=lambda: {
        '主板': 0.10,
        '中小板': 0.10,
        '创业板': 0.20,
        '科创板': 0.20,
        '北交所': 0.30,
    })
    
    ST_LIMIT_RATIOS: Dict[str, float] = field(default_factory=lambda: {
        '主板': 0.05,
        '中小板': 0.05,
        '创业板': 0.20,
        '科创板': 0.20,
        '北交所': 0.30,
    })


# ============================================================================
# 检查结果数据结构
# ============================================================================
@dataclass
class CheckResult:
    """单项检查结果"""
    name: str                           # 检查项名称
    passed: bool                        # 是否通过
    description: str                    # 检查描述
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息
    issues: List[str] = field(default_factory=list)        # 发现的问题
    metrics: Dict[str, Any] = field(default_factory=dict)  # 指标数据
    severity: str = "INFO"              # 严重程度: INFO, WARNING, ERROR, CRITICAL


@dataclass
class TableSummary:
    """数据表概览"""
    name: str
    rows: int
    columns: int
    date_range: Tuple[str, str]
    stock_count: int
    file_size_mb: float
    memory_mb: float
    null_summary: Dict[str, float]


@dataclass
class QualityReport:
    """完整的数据质量报告"""
    generated_at: str
    summary: Dict[str, TableSummary]
    checks: Dict[str, List[CheckResult]]
    overall_passed: bool
    critical_issues: List[str]
    warnings: List[str]


# ============================================================================
# 工具函数
# ============================================================================
def setup_logging(name: str, verbose: bool = False) -> logging.Logger:
    """配置日志"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = logging.DEBUG if verbose else logging.INFO
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def format_number(n: float, decimals: int = 2) -> str:
    """格式化数字显示"""
    if abs(n) >= 1e9:
        return f"{n/1e9:.{decimals}f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.{decimals}f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.{decimals}f}K"
    else:
        return f"{n:.{decimals}f}"


def format_percentage(p: float, decimals: int = 2) -> str:
    """格式化百分比"""
    return f"{p * 100:.{decimals}f}%"


def get_file_size_mb(path: Path) -> float:
    """获取文件大小(MB)"""
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def calculate_missing_rate(series) -> float:
    """计算缺失率"""
    if len(series) == 0:
        return 1.0
    return series.isnull().sum() / len(series)


def get_market_from_ts_code(ts_code: str) -> str:
    """从股票代码推断市场板块"""
    if ts_code.startswith('688'):
        return '科创板'
    elif ts_code.startswith('30'):
        return '创业板'
    elif ts_code.startswith(('8', '4')):
        return '北交所'
    elif ts_code.startswith('00') or ts_code.startswith('60'):
        return '主板'
    else:
        return '主板'


# ============================================================================
# 实例化配置
# ============================================================================
DWD_PATHS = DWDFilePaths()
THRESHOLDS = QualityThresholds()
