"""
DWD 扩展宽表数据质量检查 - 配置模块

定义：
1. 文件路径
2. 检查阈值
3. 数据结构
4. 工具函数
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
import logging


# ============================================================================
# 路径配置
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[6]  # 项目根目录
DWD_DIR = BASE_DIR / "data" / "processed" / "structured" / "dwd"
RAW_DIR = BASE_DIR / "data" / "raw" / "structured"
PREPROCESSED_DIR = BASE_DIR / "data" / "features" / "preprocessed"
REPORTS_DIR = BASE_DIR / "reports"

# 确保reports目录存在
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 训练期起始日期（2021-01-01之后要求严格数据质量）
TRAINING_START_DATE = "2021-01-01"


# ============================================================================
# 数据文件路径
# ============================================================================
@dataclass
class DWDExtFilePaths:
    """DWD扩展宽表文件路径"""
    # 5张扩展表
    money_flow: Path = DWD_DIR / "dwd_money_flow.parquet"
    chip_structure: Path = DWD_DIR / "dwd_chip_structure.parquet"
    stock_industry: Path = DWD_DIR / "dwd_stock_industry.parquet"
    event_signal: Path = DWD_DIR / "dwd_event_signal.parquet"
    macro_env: Path = DWD_DIR / "dwd_macro_env.parquet"
    
    # 核心表（用于跨表检查）
    stock_price: Path = DWD_DIR / "dwd_stock_price.parquet"
    stock_fundamental: Path = DWD_DIR / "dwd_stock_fundamental.parquet"
    stock_status: Path = DWD_DIR / "dwd_stock_status.parquet"
    
    # 预处理表（备选路径）
    preprocessed_price: Path = PREPROCESSED_DIR / "preprocessed_stock_price.parquet"
    preprocessed_fundamental: Path = PREPROCESSED_DIR / "preprocessed_stock_fundamental.parquet"
    preprocessed_status: Path = PREPROCESSED_DIR / "preprocessed_stock_status.parquet"
    
    # 原始数据参考
    trade_calendar: Path = RAW_DIR / "metadata" / "trade_calendar.parquet"
    stock_list: Path = RAW_DIR / "metadata" / "stock_list_a.parquet"


# ============================================================================
# 检查阈值配置
# ============================================================================
@dataclass
class ExtQualityThresholds:
    """扩展表数据质量检查阈值"""
    
    # ========== 通用配置 ==========
    TRAINING_MISSING_TOLERANCE: float = 0.01  # 训练期缺失率容忍度1%
    COLDSTART_MISSING_TOLERANCE: float = 0.50  # 冷启动期缺失率容忍度50%
    
    # ========== dwd_money_flow 配置 ==========
    # 极值检查
    MONEY_FLOW_MAX_AMOUNT: float = 100e8  # 单日资金流最大值 100亿
    MONEY_FLOW_TOLERANCE: float = 0.01    # 成交额守恒误差容忍度 1%
    
    # ========== dwd_chip_structure 配置 ==========
    # 比例越界
    HOLD_RATIO_MAX: float = 100.0  # 持股比例最大值 100%
    HOLDER_NUM_MIN: int = 1        # 股东户数最小值
    
    # ========== dwd_stock_industry 配置 ==========
    # 行业覆盖
    INDUSTRY_COVERAGE_MIN: float = 0.99  # 行业覆盖率最小值 99%
    INVALID_INDUSTRY_IDX: int = -1       # 无效行业索引
    
    # ========== dwd_event_signal 配置 ==========
    # 质押率阈值
    PLEDGE_RATIO_COVERAGE_MIN: float = 0.30  # 质押率非零占比最小值 30%
    PLEDGE_RATIO_MAX: float = 100.0          # 质押率最大值 100%
    
    # ========== dwd_macro_env 配置 ==========
    # PIT 月度数据变动日期
    CPI_UPDATE_DAY_MIN: int = 9    # CPI 最早更新日
    CPI_UPDATE_DAY_MAX: int = 18   # CPI 最晚更新日
    PMI_UPDATE_DAY_MIN: int = 1    # PMI 最早更新日
    PMI_UPDATE_DAY_MAX: int = 5    # PMI 最晚更新日
    
    # ========== 跨表检查 ==========
    AMOUNT_CONSERVATION_TOLERANCE: float = 0.01  # 成交额守恒误差 1%
    ROW_ALIGNMENT_TOLERANCE: float = 0.01        # 行数对齐误差 1%


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
    elif ts_code.endswith('.BJ') or ts_code.startswith(('8', '4')):
        return '北交所'
    elif ts_code.startswith('00') or ts_code.startswith('60'):
        return '主板'
    else:
        return '主板'


def is_training_period(trade_date) -> bool:
    """判断是否在训练期内"""
    import pandas as pd
    if isinstance(trade_date, str):
        trade_date = pd.to_datetime(trade_date)
    return trade_date >= pd.to_datetime(TRAINING_START_DATE)


# ============================================================================
# 实例化配置
# ============================================================================
DWD_EXT_PATHS = DWDExtFilePaths()
EXT_THRESHOLDS = ExtQualityThresholds()
