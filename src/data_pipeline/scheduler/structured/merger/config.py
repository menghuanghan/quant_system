"""
数据合并器配置模块

定义合并任务的配置，包括日期字段、排序方式、去重策略等
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class MergeMode(Enum):
    """合并模式"""
    APPEND = "append"              # 追加模式（直接追加新数据）
    MERGE_DEDUP = "merge_dedup"    # 合并去重模式（基于关键字段去重）
    REPLACE = "replace"            # 替换模式（用增量数据替换全量数据）


class DateSortOrder(Enum):
    """日期排序方式"""
    ASC = "asc"     # 升序（从早到晚）
    DESC = "desc"   # 降序（从晚到早）
    NONE = "none"   # 不排序（时间无关数据）


@dataclass
class MergeTask:
    """合并任务定义"""
    name: str                                      # 任务名称
    domain: str                                    # 所属数据域
    output_file: str                               # 输出文件名/模式
    date_field: Optional[str] = None              # 日期字段名（用于排序和去重）
    sort_order: DateSortOrder = DateSortOrder.ASC  # 排序方式
    dedup_keys: Optional[List[str]] = None        # 去重关键字段
    merge_mode: MergeMode = MergeMode.MERGE_DEDUP # 合并模式
    is_directory: bool = False                     # 是否为目录结构（包含多个子文件）
    enabled: bool = True                           # 是否启用
    description: str = ""                          # 任务描述


@dataclass
class MergeConfig:
    """合并配置"""
    inc_data_dir: str = "data/raw/inc_structured"   # 增量数据目录
    full_data_dir: str = "data/raw/structured"      # 全量数据目录
    backup_enabled: bool = False                     # 是否启用备份
    backup_dir: str = "data/raw/structured_backup"  # 备份目录
    use_gpu: bool = True                            # 是否使用GPU加速
    batch_size: int = 1000                          # 批处理大小（用于目录合并）
    dry_run: bool = False                           # 是否干运行（不实际写入）


# ==================== 各数据域合并任务配置 ====================

# ---------- 1. 基础元数据域 (Metadata) ----------
METADATA_MERGE_TASKS = [
    # 时间无关数据 - 直接追加（但需要去重）
    MergeTask(
        name="stock_list_a",
        domain="metadata",
        output_file="stock_list_a.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["ts_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="A股股票列表",
    ),
    MergeTask(
        name="stock_list_hk",
        domain="metadata",
        output_file="stock_list_hk.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["ts_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="港股股票列表",
    ),
    MergeTask(
        name="ah_stock",
        domain="metadata",
        output_file="ah_stock.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["a_ts_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="A+H股票列表",
    ),
    MergeTask(
        name="price_limit_rule",
        domain="metadata",
        output_file="price_limit_rule.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["market", "exchange"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="涨跌停规则",
    ),
    MergeTask(
        name="auction_time",
        domain="metadata",
        output_file="auction_time.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["market", "exchange"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="集合竞价时间",
    ),
    # 时间相关数据 - 需要按日期排序
    MergeTask(
        name="name_change",
        domain="metadata",
        output_file="name_change.parquet",
        date_field="start_date",
        sort_order=DateSortOrder.DESC,  # 降序排列
        dedup_keys=["ts_code", "start_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="股票曾用名记录",
    ),
    MergeTask(
        name="st_status",
        domain="metadata",
        output_file="st_status.parquet",
        date_field="st_start_date",
        sort_order=DateSortOrder.DESC,  # 降序排列
        dedup_keys=["ts_code", "st_start_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="ST状态记录",
    ),
    MergeTask(
        name="trade_calendar",
        domain="metadata",
        output_file="trade_calendar.parquet",
        date_field="cal_date",
        sort_order=DateSortOrder.DESC,  # 降序排列（从新到旧）
        dedup_keys=["exchange", "cal_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="上交所交易日历",
    ),
    MergeTask(
        name="trade_calendar_szse",
        domain="metadata",
        output_file="trade_calendar_szse.parquet",
        date_field="cal_date",
        sort_order=DateSortOrder.DESC,  # 降序排列（从新到旧）
        dedup_keys=["exchange", "cal_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="深交所交易日历",
    ),
    MergeTask(
        name="suspend_info",
        domain="metadata",
        output_file="suspend_info.parquet",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,  # 升序排列
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="停复牌信息",
    ),
]

# ---------- 2. 市场行情数据域 (Market Data) ----------
MARKET_DATA_MERGE_TASKS = [
    MergeTask(
        name="stock_daily",
        domain="market_data",
        output_file="stock_daily",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="股票日K线",
    ),
    MergeTask(
        name="stock_weekly",
        domain="market_data",
        output_file="stock_weekly",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="股票周K线",
    ),
    MergeTask(
        name="stock_monthly",
        domain="market_data",
        output_file="stock_monthly",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="股票月K线",
    ),
    MergeTask(
        name="index_daily",
        domain="market_data",
        output_file="index_daily",
        date_field="trade_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="指数日K线",
    ),
    MergeTask(
        name="etf_daily",
        domain="market_data",
        output_file="etf_daily",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="ETF日K线",
    ),
    MergeTask(
        name="daily_basic",
        domain="market_data",
        output_file="daily_basic",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="每日基本指标",
    ),
    MergeTask(
        name="stk_factor",
        domain="market_data",
        output_file="stk_factor",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="技术因子",
    ),
    MergeTask(
        name="adj_factor",
        domain="market_data",
        output_file="adj_factor",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="复权因子",
    ),
]

# ---------- 3. 公司基本面数据域 (Fundamental) ----------
FUNDAMENTAL_MERGE_TASKS = [
    MergeTask(
        name="balance_sheet",
        domain="fundamental",
        output_file="balance_sheet",
        date_field="end_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "end_date", "report_type"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="资产负债表",
    ),
    MergeTask(
        name="income_statement",
        domain="fundamental",
        output_file="income_statement",
        date_field="end_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "end_date", "report_type"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="利润表",
    ),
    MergeTask(
        name="cash_flow",
        domain="fundamental",
        output_file="cash_flow",
        date_field="end_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "end_date", "report_type"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="现金流量表",
    ),
    MergeTask(
        name="financial_indicator",
        domain="fundamental",
        output_file="financial_indicator",
        date_field="end_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "end_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="财务指标",
    ),
    MergeTask(
        name="share_structure",
        domain="fundamental",
        output_file="share_structure",
        date_field="end_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "end_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="股本结构",
    ),
    # 单文件数据
    MergeTask(
        name="company_info",
        domain="fundamental",
        output_file="company_info.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["ts_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="公司基本信息",
    ),
    MergeTask(
        name="industry_class",
        domain="fundamental",
        output_file="industry_class.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["ts_code", "src"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="行业分类",
    ),
    MergeTask(
        name="pledge",
        domain="fundamental",
        output_file="pledge",
        date_field="end_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "end_date", "pledge_amount"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="股权质押",
    ),
    MergeTask(
        name="share_float",
        domain="fundamental",
        output_file="share_float.parquet",
        date_field="float_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "float_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="限售解禁",
    ),
    MergeTask(
        name="repurchase",
        domain="fundamental",
        output_file="repurchase.parquet",
        date_field="ann_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "ann_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="股票回购",
    ),
]

# ---------- 4. 资金与交易行为数据域 (Trading Behavior) ----------
TRADING_BEHAVIOR_MERGE_TASKS = [
    MergeTask(
        name="money_flow",
        domain="trading_behavior",
        output_file="money_flow",
        date_field="trade_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="个股资金流向",
    ),
    MergeTask(
        name="margin_detail",
        domain="trading_behavior",
        output_file="margin_detail",
        date_field="trade_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="融资融券明细",
    ),
    MergeTask(
        name="top_inst",
        domain="trading_behavior",
        output_file="top_inst",
        date_field="trade_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "trade_date", "exalter"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="龙虎榜机构",
    ),
    # 单文件数据
    MergeTask(
        name="hsgt_flow",
        domain="trading_behavior",
        output_file="hsgt_flow.parquet",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="沪深港通资金流向",
    ),
    MergeTask(
        name="margin_summary",
        domain="trading_behavior",
        output_file="margin_summary.parquet",
        date_field="trade_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["trade_date", "exchange_id"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="融资融券汇总",
    ),
    MergeTask(
        name="margin_target",
        domain="trading_behavior",
        output_file="margin_target.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["ts_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="两融标的",
    ),
    MergeTask(
        name="top_list",
        domain="trading_behavior",
        output_file="top_list.parquet",
        date_field="trade_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "trade_date", "reason"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="龙虎榜",
    ),
    MergeTask(
        name="block_trade",
        domain="trading_behavior",
        output_file="block_trade.parquet",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "trade_date", "buyer", "seller"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="大宗交易",
    ),
]

# ---------- 5. 衍生品与多资产数据域 (Derivatives) ----------
DERIVATIVES_MERGE_TASKS = [
    MergeTask(
        name="fund_nav",
        domain="derivatives",
        output_file="fund_nav",
        date_field="nav_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "nav_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="基金净值",
    ),
    MergeTask(
        name="fund_share",
        domain="derivatives",
        output_file="fund_share",
        date_field="trade_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="基金规模",
    ),
    MergeTask(
        name="fund_adj",
        domain="derivatives",
        output_file="fund_adj",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="基金复权因子",
    ),
    MergeTask(
        name="fut_daily",
        domain="derivatives",
        output_file="fut_daily",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="期货日线",
    ),
    MergeTask(
        name="opt_daily",
        domain="derivatives",
        output_file="opt_daily",
        date_field="trade_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="期权日线",
    ),
    MergeTask(
        name="cb_daily",
        domain="derivatives",
        output_file="cb_daily",
        date_field="trade_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="可转债日线",
    ),
    MergeTask(
        name="repo_daily",
        domain="derivatives",
        output_file="repo_daily",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["ts_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="回购日线",
    ),
    # 单文件数据
    MergeTask(
        name="fund_basic",
        domain="derivatives",
        output_file="fund_basic.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["ts_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="基金基本信息",
    ),
    MergeTask(
        name="fut_basic",
        domain="derivatives",
        output_file="fut_basic.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["ts_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="期货基本信息",
    ),
    MergeTask(
        name="opt_basic",
        domain="derivatives",
        output_file="opt_basic.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["ts_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="期权基本信息",
    ),
    MergeTask(
        name="cb_basic",
        domain="derivatives",
        output_file="cb_basic.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["ts_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="可转债基本信息",
    ),
]

# ---------- 6. 指数与基准数据域 (Index & Benchmark) ----------
INDEX_BENCHMARK_MERGE_TASKS = [
    MergeTask(
        name="index_weight",
        domain="index_benchmark",
        output_file="index_weight",
        date_field="trade_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["index_code", "con_code", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=True,
        description="指数成分权重",
    ),
    MergeTask(
        name="index_basic",
        domain="index_benchmark",
        output_file="index_basic.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["ts_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="指数基本信息",
    ),
    MergeTask(
        name="index_member",
        domain="index_benchmark",
        output_file="index_member.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["index_code", "con_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="指数成分股",
    ),
]

# ---------- 7. 宏观与外生变量域 (Macro & Exogenous) ----------
MACRO_EXOGENOUS_MERGE_TASKS = [
    MergeTask(
        name="cn_gdp",
        domain="macro_exogenous",
        output_file="cn_gdp.parquet",
        date_field="quarter",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["quarter"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="中国GDP",
    ),
    MergeTask(
        name="cn_cpi",
        domain="macro_exogenous",
        output_file="cn_cpi.parquet",
        date_field="month",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["month"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="中国CPI",
    ),
    MergeTask(
        name="cn_ppi",
        domain="macro_exogenous",
        output_file="cn_ppi.parquet",
        date_field="month",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["month"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="中国PPI",
    ),
    MergeTask(
        name="cn_pmi",
        domain="macro_exogenous",
        output_file="cn_pmi.parquet",
        date_field="month",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["month"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="中国PMI",
    ),
    MergeTask(
        name="cn_m2",
        domain="macro_exogenous",
        output_file="cn_m2.parquet",
        date_field="month",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["month"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="货币供应量",
    ),
    MergeTask(
        name="shibor",
        domain="macro_exogenous",
        output_file="shibor.parquet",
        date_field="date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="Shibor利率",
    ),
    MergeTask(
        name="lpr",
        domain="macro_exogenous",
        output_file="lpr.parquet",
        date_field="date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="LPR利率",
    ),
    MergeTask(
        name="us_treasury",
        domain="macro_exogenous",
        output_file="us_treasury.parquet",
        date_field="date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="美国国债收益率",
    ),
]

# ---------- 8. 板块/行业/主题数据域 (Cross-sectional) ----------
CROSS_SECTIONAL_MERGE_TASKS = [
    MergeTask(
        name="sw_index_classify",
        domain="cross_sectional",
        output_file="sw_index_classify.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["index_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="申万行业分类",
    ),
    MergeTask(
        name="sw_index_member",
        domain="cross_sectional",
        output_file="sw_index_member.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["index_code", "con_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="申万行业成分股",
    ),
    MergeTask(
        name="ths_index",
        domain="cross_sectional",
        output_file="ths_index.parquet",
        date_field=None,
        sort_order=DateSortOrder.NONE,
        dedup_keys=["ts_code"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="同花顺概念指数",
    ),
    MergeTask(
        name="concept_board_em",
        domain="cross_sectional",
        output_file="concept_board_em.parquet",
        date_field="trade_date",
        sort_order=DateSortOrder.DESC,
        dedup_keys=["name", "trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="东方财富概念板块",
    ),
]

# ---------- 9. 深度风险与质量因子域 (Deep Risk & Quality) ----------
DEEP_RISK_QUALITY_MERGE_TASKS = [
    MergeTask(
        name="a_pe_pb_ew_median",
        domain="deep_risk_quality",
        output_file="a_pe_pb_ew_median.parquet",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="A股等权重中位数PE/PB",
    ),
    MergeTask(
        name="market_congestion",
        domain="deep_risk_quality",
        output_file="market_congestion.parquet",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="大盘拥挤度",
    ),
    MergeTask(
        name="stock_bond_spread",
        domain="deep_risk_quality",
        output_file="stock_bond_spread.parquet",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="股债利差",
    ),
    MergeTask(
        name="buffett_indicator",
        domain="deep_risk_quality",
        output_file="buffett_indicator.parquet",
        date_field="trade_date",
        sort_order=DateSortOrder.ASC,
        dedup_keys=["trade_date"],
        merge_mode=MergeMode.MERGE_DEDUP,
        is_directory=False,
        description="巴菲特指标",
    ),
]


# ==================== 汇总所有合并任务 ====================

# 按数据域汇总
MERGE_TASKS_BY_DOMAIN = {
    "metadata": METADATA_MERGE_TASKS,
    "market_data": MARKET_DATA_MERGE_TASKS,
    "fundamental": FUNDAMENTAL_MERGE_TASKS,
    "trading_behavior": TRADING_BEHAVIOR_MERGE_TASKS,
    "derivatives": DERIVATIVES_MERGE_TASKS,
    "index_benchmark": INDEX_BENCHMARK_MERGE_TASKS,
    "macro_exogenous": MACRO_EXOGENOUS_MERGE_TASKS,
    "cross_sectional": CROSS_SECTIONAL_MERGE_TASKS,
    "deep_risk_quality": DEEP_RISK_QUALITY_MERGE_TASKS,
}


def get_merge_tasks_by_domain(domain: str) -> List[MergeTask]:
    """获取指定数据域的合并任务列表"""
    return MERGE_TASKS_BY_DOMAIN.get(domain, [])


def get_all_merge_tasks() -> List[MergeTask]:
    """获取所有合并任务"""
    all_tasks = []
    for tasks in MERGE_TASKS_BY_DOMAIN.values():
        all_tasks.extend(tasks)
    return all_tasks


def get_merge_task_by_name(domain: str, task_name: str) -> Optional[MergeTask]:
    """根据数据域和任务名获取合并任务"""
    tasks = get_merge_tasks_by_domain(domain)
    for task in tasks:
        if task.name == task_name:
            return task
    return None


__all__ = [
    "MergeMode",
    "DateSortOrder",
    "MergeTask",
    "MergeConfig",
    "MERGE_TASKS_BY_DOMAIN",
    "get_merge_tasks_by_domain",
    "get_all_merge_tasks",
    "get_merge_task_by_name",
]
