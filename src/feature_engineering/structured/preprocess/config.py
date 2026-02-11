"""
特征工程预处理配置

定义各类预处理操作的参数和阈值。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# 项目根目录
# preprocess/config.py -> preprocess -> structured -> feature_engineering -> src -> quant_system
BASE_DIR = Path(__file__).resolve().parents[4]  # 向上4级到项目根目录

# 数据路径
DWD_INPUT_DIR = BASE_DIR / "data" / "processed" / "structured" / "dwd"
FEATURE_OUTPUT_DIR = BASE_DIR / "data" / "features" / "preprocessed"  # 预处理后数据输出


@dataclass
class WinsorizeConfig:
    """去极值配置"""
    
    # 收益率裁剪范围：[-30%, +100%]（允许跌停但不允许超过涨停太多）
    return_1d_lower: float = -0.30
    return_1d_upper: float = 1.00
    
    # 使用分位数去极值的字段及其分位数范围
    quantile_clip_fields: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        # (lower_quantile, upper_quantile)
        "pe_ttm": (0.01, 0.99),
        "pb": (0.01, 0.99),
        "ps_ttm": (0.01, 0.99),
        "roe": (0.01, 0.99),
        "roa": (0.01, 0.99),
        "revenue_yoy": (0.01, 0.99),
        "net_profit_yoy": (0.01, 0.99),
    })


@dataclass
class LogTransformConfig:
    """对数变换配置"""
    
    # 价格表需要取对数的字段
    price_log_fields: List[str] = field(default_factory=lambda: [
        "vol",           # 成交量
        "amount",        # 成交额
    ])
    
    # 基本面表需要取对数的字段（绝对值大的财务数据）
    fundamental_log_fields: List[str] = field(default_factory=lambda: [
        "total_mv",           # 总市值
        "circ_mv",            # 流通市值
        "revenue_ttm",        # 营收TTM
        "operate_profit_ttm", # 营业利润TTM
        "total_profit_ttm",   # 利润总额TTM
        "n_income_attr_p_ttm",# 归母净利润TTM
        "revenue_sq",         # 单季度营收
        "n_income_attr_p_sq", # 单季度归母净利润
        "total_assets",       # 总资产
        "total_liab",         # 总负债
        "total_equity",       # 净资产
    ])
    
    # 对数变换时的最小值（避免 log(0) 或 log(负数)）
    log_epsilon: float = 1.0


@dataclass
class InverseTransformConfig:
    """倒数变换配置（处理估值指标缺失）"""
    
    # 倒数变换映射：原字段 -> 新字段
    inverse_fields: Dict[str, str] = field(default_factory=lambda: {
        "pe_ttm": "ep",   # 盈利收益率 = 1 / PE
        "pb": "bp",       # Book-to-Price = 1 / PB
        "ps_ttm": "sp",   # Sales-to-Price = 1 / PS
    })
    
    # 倒数变换时的最小分母（避免除以0或极小值）
    inverse_epsilon: float = 0.01


@dataclass
class DataLagConfig:
    """数据时滞过滤配置"""
    
    # 最大允许的数据时滞（天）
    max_lag_days: int = 180
    
    # 超过时滞后需要置为 NaN 的字段
    lag_sensitive_fields: List[str] = field(default_factory=lambda: [
        "pe_ttm", "pb", "ps_ttm",
        "ep", "bp", "sp",  # 新增的倒数字段
        "revenue_ttm", "operate_profit_ttm", "total_profit_ttm", "n_income_attr_p_ttm",
        "revenue_sq", "n_income_attr_p_sq",
        "roe", "roa", "gross_margin", "netprofit_margin",
        "debt_to_assets", "revenue_yoy", "net_profit_yoy",
        "total_assets", "total_liab", "total_equity",
    ])


@dataclass
class TradingStatusConfig:
    """交易状态修正配置"""
    
    # 是否使用 "信资金，不信标签" 策略
    trust_volume_over_label: bool = True


# ============================================================================
# 扩展表预处理配置
# ============================================================================

@dataclass
class MoneyFlowConfig:
    """资金流预处理配置"""
    
    # 大宗交易成交量：万股 → 手 (1万股 = 100手)
    block_trade_vol_multiplier: float = 100.0
    
    # 需要在停牌日清零的字段
    suspend_zero_fields: List[str] = field(default_factory=lambda: [
        "net_mf_amount", "net_main_amount",
        "buy_sm_amount", "sell_sm_amount",
        "buy_md_amount", "sell_md_amount",
        "buy_lg_amount", "sell_lg_amount",
        "buy_elg_amount", "sell_elg_amount",
    ])


@dataclass
class ChipConfig:
    """筹码结构预处理配置"""
    
    # 持股比例上限 (Clip)
    hold_ratio_max: float = 100.0
    
    # 需要 Clip 的比例字段
    ratio_clip_fields: List[str] = field(default_factory=lambda: [
        "top10_hold_ratio",
        "top1_hold_ratio",
        "top10_inst_ratio",
    ])
    
    # 需要前向填充的字段
    ffill_fields: List[str] = field(default_factory=lambda: [
        "holder_num",
        "top10_hold_ratio",
        "top10_hold_amount",
    ])


@dataclass
class IndustryConfig:
    """行业分类预处理配置"""
    
    # 未分类的默认索引
    unknown_idx: int = -1
    
    # 需要确保为 int 类型的索引字段
    index_fields: List[str] = field(default_factory=lambda: [
        "industry_idx",
        "sw_l1_idx",
        "sw_l2_idx",
    ])


@dataclass
class MacroConfig:
    """宏观预处理配置"""
    
    # 是否对宏观数据做 shift(1)（模拟"开盘前已知"）
    apply_shift: bool = True
    shift_days: int = 1
    
    # 需要 shift 的字段前缀（排除 trade_date）
    shift_field_prefixes: List[str] = field(default_factory=lambda: [
        "gdp", "cpi", "ppi", "pmi", "m2", "lpr", "shibor",
        "market_", "stock_bond", "buffett", "pb_", "break_net",
        "sh300", "zz500", "cyb", "sz50", "zz1000", "kc50",
        "gc001", "r001", "if_", "ic_", "ih_", "im_",
        "macro_", "risk_", "money_regime", "lpr_trend",
    ])


@dataclass
class EventConfig:
    """事件信号预处理配置"""
    
    # 质押率上限
    pledge_ratio_max: float = 100.0
    
    # 0/1 信号列（NaN 应填 0）
    signal_columns: List[str] = field(default_factory=lambda: [
        "is_repurchase_ann",
        "in_repurchase_window",
        "is_unlock_day",
        "in_unlock_window",
        "is_dividend_ann",
        "in_dividend_window",
        "has_event",
        "has_risk_event",
    ])


@dataclass
class PreprocessConfig:
    """预处理总配置"""
    
    # 输入输出路径
    input_dir: Path = DWD_INPUT_DIR
    output_dir: Path = FEATURE_OUTPUT_DIR
    
    # 各类预处理配置
    winsorize: WinsorizeConfig = field(default_factory=WinsorizeConfig)
    log_transform: LogTransformConfig = field(default_factory=LogTransformConfig)
    inverse_transform: InverseTransformConfig = field(default_factory=InverseTransformConfig)
    data_lag: DataLagConfig = field(default_factory=DataLagConfig)
    trading_status: TradingStatusConfig = field(default_factory=TradingStatusConfig)
    
    # 扩展表预处理配置
    money_flow: MoneyFlowConfig = field(default_factory=MoneyFlowConfig)
    chip: ChipConfig = field(default_factory=ChipConfig)
    industry: IndustryConfig = field(default_factory=IndustryConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)
    event: EventConfig = field(default_factory=EventConfig)
    
    # 输出文件名 - 核心表
    output_price_file: str = "preprocessed_stock_price.parquet"
    output_fundamental_file: str = "preprocessed_stock_fundamental.parquet"
    output_status_file: str = "preprocessed_stock_status.parquet"
    
    # 输出文件名 - 扩展表
    output_money_flow_file: str = "preprocessed_money_flow.parquet"
    output_chip_file: str = "preprocessed_chip_structure.parquet"
    output_industry_file: str = "preprocessed_stock_industry.parquet"
    output_macro_file: str = "preprocessed_macro_env.parquet"
    output_event_file: str = "preprocessed_event_signal.parquet"
    
    # 处理选项
    use_gpu: bool = True  # 是否使用 GPU 加速
    verbose: bool = True  # 是否打印详细日志
    
    def __post_init__(self):
        """确保输出目录存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
