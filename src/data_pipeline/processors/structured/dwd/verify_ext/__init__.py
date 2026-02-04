# DWD扩展宽表数据质量检查模块
"""
验证 5 张 DWD 扩展宽表的数据质量：
- dwd_money_flow (资金博弈)
- dwd_chip_structure (筹码结构)
- dwd_stock_industry (行业分类)
- dwd_event_signal (事件信号)
- dwd_macro_env (宏观环境)

检查维度：
1. Schema 层 - 主键唯一性、列类型
2. 完整性层 - 训练期 NaN、覆盖率
3. 逻辑层 - 业务规则合理性
4. 跨表层 - 量价资金闭环、骨架对齐
"""

from .dq_config import (
    DWD_EXT_PATHS,
    EXT_THRESHOLDS,
    TRAINING_START_DATE,
    CheckResult,
    TableSummary,
    setup_logging,
    format_number,
    format_percentage,
)

__all__ = [
    "DWD_EXT_PATHS",
    "EXT_THRESHOLDS",
    "TRAINING_START_DATE",
    "CheckResult",
    "TableSummary",
    "setup_logging",
    "format_number",
    "format_percentage",
]
