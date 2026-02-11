"""
DWD数据质量检查模块

提供对8张DWD宽表的全面数据质量检查，包括：
- 通用核验维度（主键、时间连续性、缺失值、数据类型）
- 表级详细检查（业务逻辑、单位核验、分布统计）
- 报告生成
"""

from .base import BaseChecker, CheckResult, CheckSeverity
from .common_checks import CommonChecker
from .stock_price_checker import StockPriceChecker
from .fundamental_checker import FundamentalChecker
from .money_flow_checker import MoneyFlowChecker
from .chip_structure_checker import ChipStructureChecker
from .industry_checker import IndustryChecker
from .event_signal_checker import EventSignalChecker
from .status_checker import StatusChecker
from .macro_env_checker import MacroEnvChecker
from .report_generator import DWDReportGenerator
from .cross_table_checker import CrossTableChecker, CrossTableCheckReport

__all__ = [
    "BaseChecker",
    "CheckResult",
    "CheckSeverity",
    "CommonChecker",
    "StockPriceChecker",
    "FundamentalChecker",
    "MoneyFlowChecker",
    "ChipStructureChecker",
    "IndustryChecker",
    "EventSignalChecker",
    "StatusChecker",
    "MacroEnvChecker",
    "DWDReportGenerator",
    "CrossTableChecker",
    "CrossTableCheckReport",
]
