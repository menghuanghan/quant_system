"""
结构化增量数据采集调度器配置

目标：
- 面向“单日增量”采集
- 输出目录按 date 分区：data/raw/structured_increment/date=YYYY-MM-DD/
- 与 data/raw/structured 的文件命名与数据结构保持一致
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal


DateGranularity = Literal["day", "month", "quarter", "none"]
ParamStyle = Literal[
    "trade_date",
    "start_end",
    "ann_date",
    "float_date",
    "end_date",
    "month",
    "quarter",
    "trade_calendar_day",
    "none",
]
OutputMode = Literal["single", "by_code"]


@dataclass(frozen=True)
class IncrementCollectionTask:
    """增量采集任务定义"""

    name: str
    domain: str
    collector_func: str
    description: str
    param_style: ParamStyle
    date_field: str
    date_granularity: DateGranularity = "day"
    output_mode: OutputMode = "single"
    code_param: str = "ts_code"          # by_code 时使用 ts_code/index_code
    code_source: str = ""                # by_code 时使用: core_index / etf
    enabled: bool = True
    optional: bool = False
    requires_stock_list: bool = False     # 仅用于少数必须按股票循环的任务
    extra_params: Dict[str, str] = field(default_factory=dict)


CORE_INDEX_CODES: List[str] = [
    "000300.SH",  # 沪深300
    "000905.SH",  # 中证500
    "000852.SH",  # 中证1000
]


FUNDAMENTAL_MANDATORY_TSCODE_BUNDLE_TASK = "fundamental_mandatory_ts_bundle"
FUNDAMENTAL_MANDATORY_TSCODE_SUBTASKS: List[str] = [
    "balance_sheet",
    "income_statement",
    "cash_flow",
    "financial_indicator",
    "top10_holders",
]

FUNDAMENTAL_MANDATORY_TSCODE_SUBTASK_TASKS: List["IncrementCollectionTask"] = [
    IncrementCollectionTask(
        "balance_sheet",
        "fundamental",
        "get_balance_sheet",
        "资产负债表（公告维度）",
        "start_end",
        "ann_date",
    ),
    IncrementCollectionTask(
        "income_statement",
        "fundamental",
        "get_income_statement",
        "利润表（公告维度）",
        "start_end",
        "ann_date",
    ),
    IncrementCollectionTask(
        "cash_flow",
        "fundamental",
        "get_cash_flow",
        "现金流量表（公告维度）",
        "start_end",
        "ann_date",
    ),
    IncrementCollectionTask(
        "financial_indicator",
        "fundamental",
        "get_financial_indicator",
        "财务指标（公告维度）",
        "start_end",
        "ann_date",
    ),
    IncrementCollectionTask(
        "top10_holders",
        "fundamental",
        "get_top10_holders",
        "前十大股东（公告维度，全市场并发）",
        "ann_date",
        "ann_date",
    ),
]


# ===== 时间相关增量任务（默认启用） =====
TIME_DEPENDENT_TASKS: List[IncrementCollectionTask] = [
    # metadata
    IncrementCollectionTask("trade_calendar", "metadata", "get_trade_calendar", "交易日历（上交所）", "trade_calendar_day", "cal_date"),
    IncrementCollectionTask("suspend_info", "metadata", "get_suspend_info", "停复牌信息", "trade_date", "trade_date"),
    IncrementCollectionTask("st_status", "metadata", "get_st_status", "ST状态变更", "start_end", "st_start_date"),

    # market_data
    IncrementCollectionTask("stock_daily", "market_data", "get_stock_daily", "股票日线", "trade_date", "trade_date"),
    IncrementCollectionTask("daily_basic", "market_data", "get_daily_basic", "每日基本面", "trade_date", "trade_date"),
    IncrementCollectionTask("adj_factor", "market_data", "get_adj_factor", "复权因子", "trade_date", "trade_date"),
    IncrementCollectionTask(
        "index_daily", "market_data", "get_index_daily", "指数日线（核心指数）", "start_end", "trade_date",
        output_mode="by_code", code_param="ts_code", code_source="core_index"
    ),

    # fundamental
    IncrementCollectionTask(
        FUNDAMENTAL_MANDATORY_TSCODE_BUNDLE_TASK,
        "fundamental",
        "__bundle__",
        "财务四表+前十大股东（5进程并行）",
        "none",
        "",
        "none",
    ),
    IncrementCollectionTask("share_structure", "fundamental", "get_share_structure", "股本结构", "start_end", "ann_date"),
    IncrementCollectionTask("pledge", "fundamental", "get_pledge", "股权质押", "end_date", "end_date"),
    IncrementCollectionTask("share_float", "fundamental", "get_share_float", "限售解禁", "float_date", "float_date"),
    IncrementCollectionTask("repurchase", "fundamental", "get_repurchase", "股票回购", "start_end", "ann_date"),
    IncrementCollectionTask("dividend", "fundamental", "get_dividend", "分红送股", "ann_date", "ann_date"),

    # trading_behavior
    IncrementCollectionTask("money_flow", "trading_behavior", "get_money_flow", "个股资金流向", "trade_date", "trade_date"),
    IncrementCollectionTask("margin_detail", "trading_behavior", "get_margin_detail", "融资融券明细", "trade_date", "trade_date"),
    IncrementCollectionTask("top_list", "trading_behavior", "get_top_list", "龙虎榜", "trade_date", "trade_date"),
    IncrementCollectionTask("top_inst", "trading_behavior", "get_top_inst", "龙虎榜营业部", "trade_date", "trade_date"),
    IncrementCollectionTask("hsgt_flow", "trading_behavior", "get_hsgt_flow", "沪深港通资金流", "trade_date", "trade_date"),
    IncrementCollectionTask("block_trade", "trading_behavior", "get_block_trade", "大宗交易", "trade_date", "trade_date"),
    IncrementCollectionTask("margin_summary", "trading_behavior", "get_margin_summary", "融资融券汇总", "trade_date", "trade_date"),

    # macro_exogenous（仅保留日频）
    IncrementCollectionTask("shibor", "macro_exogenous", "get_shibor", "Shibor利率", "start_end", "date"),

    # deep_risk_quality
    IncrementCollectionTask("market_congestion", "deep_risk_quality", "get_market_congestion", "市场拥挤度", "start_end", "date"),
    IncrementCollectionTask("stock_bond_spread", "deep_risk_quality", "get_stock_bond_spread", "股债利差", "start_end", "date"),
    IncrementCollectionTask("a_pe_pb_ew_median", "deep_risk_quality", "get_a_pe_pb_ew_median", "全市场估值扩散", "start_end", "date"),
    IncrementCollectionTask("buffett_indicator", "deep_risk_quality", "get_buffett_indicator", "巴菲特指标", "start_end", "date"),
    IncrementCollectionTask("break_net_stock", "deep_risk_quality", "get_break_net_stock", "破净股统计", "start_end", "date"),

    # derivatives
    IncrementCollectionTask("repo_daily", "derivatives", "get_repo_daily", "回购利率日线", "trade_date", "trade_date"),
    IncrementCollectionTask("fut_daily", "derivatives", "get_fut_daily", "期货日线", "trade_date", "trade_date"),

    # index_benchmark
    IncrementCollectionTask(
        "index_weight", "index_benchmark", "get_index_weight", "指数成分权重（核心指数）", "start_end", "trade_date",
        date_granularity="none",
        output_mode="by_code", code_param="index_code", code_source="core_index"
    ),
]


# ETF 为可选项，默认不采集
ETF_OPTIONAL_TASK = IncrementCollectionTask(
    name="etf_daily",
    domain="market_data",
    collector_func="get_etf_daily",
    description="ETF日线（可选）",
    param_style="start_end",
    date_field="trade_date",
    output_mode="by_code",
    code_param="ts_code",
    code_source="etf",
    optional=True,
)


# DataSourcePaths 中存在但偏静态的参考表（默认关闭，仅按需启用）
OPTIONAL_STATIC_TASKS: List[IncrementCollectionTask] = [
    IncrementCollectionTask("stock_list_a", "metadata", "get_stock_list_a", "A股股票列表（静态）", "none", "", "none", optional=True),
    IncrementCollectionTask("name_change", "metadata", "get_name_change", "股票曾用名（静态）", "none", "", "none", optional=True),
    IncrementCollectionTask("cn_gdp", "macro_exogenous", "get_cn_gdp", "中国GDP（静态刷新）", "none", "", "none", optional=True),
    IncrementCollectionTask("cn_cpi", "macro_exogenous", "get_cn_cpi", "中国CPI（静态刷新）", "none", "", "none", optional=True),
    IncrementCollectionTask("cn_ppi", "macro_exogenous", "get_cn_ppi", "中国PPI（静态刷新）", "none", "", "none", optional=True),
    IncrementCollectionTask("cn_pmi", "macro_exogenous", "get_cn_pmi", "中国PMI（静态刷新）", "none", "", "none", optional=True),
    IncrementCollectionTask("cn_m2", "macro_exogenous", "get_cn_m2", "中国M2（静态刷新）", "none", "", "none", optional=True),
    IncrementCollectionTask("lpr", "macro_exogenous", "get_lpr", "LPR利率（静态刷新）", "none", "", "none", optional=True),
    IncrementCollectionTask("sw_index_classify", "cross_sectional", "get_sw_index_classify", "申万行业分类（静态）", "none", "", "none", optional=True),
    IncrementCollectionTask("sw_index_member", "cross_sectional", "get_sw_index_member", "申万行业成分（静态）", "none", "", "none", optional=True),
    IncrementCollectionTask("opt_basic", "derivatives", "get_opt_basic", "期权基础信息（静态）", "none", "", "none", optional=True),
]


DOMAIN_NAMES: Dict[str, str] = {
    "metadata": "基础元数据",
    "market_data": "市场行情",
    "fundamental": "公司基本面",
    "trading_behavior": "资金与交易行为",
    "cross_sectional": "板块行业",
    "derivatives": "衍生品",
    "index_benchmark": "指数基准",
    "macro_exogenous": "宏观外生",
    "deep_risk_quality": "深度风险质量",
}


def get_increment_tasks(include_etf: bool = False, include_static: bool = False) -> List[IncrementCollectionTask]:
    """获取增量任务列表（不包含静态参考表）

    说明：
    - include_static 参数仅为兼容保留，不再将 OPTIONAL_STATIC_TASKS 纳入增量落盘任务。
    - 静态参考表由调度器在 run() 时自动执行“采集 -> 对比 -> 必要时覆盖全量表”。
    """
    tasks = list(TIME_DEPENDENT_TASKS)
    if include_etf:
        tasks.append(ETF_OPTIONAL_TASK)
    return tasks
