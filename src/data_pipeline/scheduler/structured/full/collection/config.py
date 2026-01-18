"""
原始数据采集调度器配置模块

定义所有数据域的采集任务配置，包括：
- 时间相关数据：需要指定日期范围的数据类型
- 时间无关数据：不需要日期参数的基础数据类型
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum


class DataCategory(Enum):
    """数据类别"""
    TIME_DEPENDENT = "time_dependent"      # 时间相关数据（需指定日期范围）
    TIME_INDEPENDENT = "time_independent"  # 时间无关数据（不需日期参数）


class StockScope(Enum):
    """股票范围"""
    ALL_A = "all_a"              # 全A股
    SINGLE = "single"           # 单只股票
    INDEX = "index"             # 指数成分
    NONE = "none"               # 不需要股票代码


class CollectionFrequency(Enum):
    """采集频率"""
    DAILY = "daily"             # 日频
    WEEKLY = "weekly"           # 周频
    MONTHLY = "monthly"         # 月频
    QUARTERLY = "quarterly"     # 季频
    YEARLY = "yearly"           # 年频
    ONCE = "once"               # 一次性


@dataclass
class CollectionTask:
    """采集任务定义"""
    name: str                                      # 任务名称
    description: str                               # 任务描述
    domain: str                                    # 所属数据域
    category: DataCategory                         # 数据类别
    collector_func: str                            # 采集器函数名
    stock_scope: StockScope = StockScope.NONE      # 股票范围
    frequency: CollectionFrequency = CollectionFrequency.DAILY  # 采集频率
    priority: int = 10                             # 优先级（1-100，数字越小越先执行）
    output_file: str = ""                          # 输出文件名
    batch_size: int = 100                          # 批处理大小（按股票）
    params: Dict[str, Any] = field(default_factory=dict)  # 额外参数
    date_field: str = "trade_date"                 # 日期字段名
    enabled: bool = True                           # 是否启用
    realtime: bool = False                         # 是否为实时数据（排除）


# ==================== 数据域任务配置 ====================

# ---------- 1. 基础元数据域 (Metadata) ----------
METADATA_TASKS = [
    # 证券与标的基础信息
    CollectionTask(
        name="stock_list_a",
        description="A股股票列表（主板/中小板/创业板/科创板/北交所）",
        domain="metadata",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_stock_list_a",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=1,
        output_file="stock_list_a.parquet",
    ),
    CollectionTask(
        name="stock_list_hk",
        description="港股股票列表",
        domain="metadata",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_stock_list_hk",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=1,
        output_file="stock_list_hk.parquet",
    ),
    CollectionTask(
        name="stock_list_us",
        description="美股股票列表",
        domain="metadata",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_stock_list_us",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=1,
        output_file="stock_list_us.parquet",
    ),
    CollectionTask(
        name="name_change",
        description="股票曾用名及代码变更记录",
        domain="metadata",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_name_change",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=2,
        output_file="name_change.parquet",
    ),
    CollectionTask(
        name="st_status",
        description="ST/*ST/风险警示标识",
        domain="metadata",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_st_status",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=2,
        output_file="st_status.parquet",
    ),
    CollectionTask(
        name="ah_stock",
        description="A+H股票列表",
        domain="metadata",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_ah_stock",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=2,
        output_file="ah_stock.parquet",
    ),
    
    # 交易日历与制度信息
    CollectionTask(
        name="trade_calendar",
        description="股票交易日历",
        domain="metadata",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_trade_calendar",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=1,
        output_file="trade_calendar.parquet",
        params={"exchange": "SSE"},
    ),
    CollectionTask(
        name="trade_calendar_szse",
        description="深交所交易日历",
        domain="metadata",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_trade_calendar",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=1,
        output_file="trade_calendar_szse.parquet",
        params={"exchange": "SZSE"},
    ),
    CollectionTask(
        name="suspend_info",
        description="股票停复牌信息",
        domain="metadata",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_suspend_info",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=3,
        output_file="suspend_info.parquet",
    ),
    CollectionTask(
        name="price_limit_rule",
        description="涨跌停规则",
        domain="metadata",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_price_limit_rule",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=2,
        output_file="price_limit_rule.parquet",
    ),
    CollectionTask(
        name="auction_time",
        description="集合竞价时间",
        domain="metadata",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_auction_time",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=2,
        output_file="auction_time.parquet",
    ),
]

# ---------- 2. 市场行情数据域 (Market Data) ----------
MARKET_DATA_TASKS = [
    # K线与价格序列（时间相关）
    CollectionTask(
        name="stock_daily",
        description="股票日K线（前复权）",
        domain="market_data",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_stock_daily",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="stock_daily/{ts_code}.parquet",
        batch_size=50,
        params={"adj": "qfq"},
    ),
    CollectionTask(
        name="stock_weekly",
        description="股票周K线",
        domain="market_data",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_stock_weekly",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.WEEKLY,
        priority=6,
        output_file="stock_weekly/{ts_code}.parquet",
        batch_size=100,
    ),
    CollectionTask(
        name="stock_monthly",
        description="股票月K线",
        domain="market_data",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_stock_monthly",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.MONTHLY,
        priority=6,
        output_file="stock_monthly/{ts_code}.parquet",
        batch_size=100,
    ),
    CollectionTask(
        name="index_daily",
        description="指数日K线",
        domain="market_data",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_index_daily",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="index_daily.parquet",
    ),
    CollectionTask(
        name="etf_daily",
        description="ETF日K线",
        domain="market_data",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_etf_daily",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="etf_daily.parquet",
    ),
    
    # 技术指标（时间相关）
    CollectionTask(
        name="daily_basic",
        description="每日基本指标（PE/PB/换手率等）",
        domain="market_data",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_daily_basic",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="daily_basic/{ts_code}.parquet",
        batch_size=50,
    ),
    CollectionTask(
        name="stk_factor",
        description="官方技术因子",
        domain="market_data",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_stk_factor",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.DAILY,
        priority=7,
        output_file="stk_factor/{ts_code}.parquet",
        batch_size=50,
    ),
    
    # 实时行情标记为实时数据，排除在全量采集外
    CollectionTask(
        name="realtime_quote",
        description="实时行情（排除）",
        domain="market_data",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_realtime_quote",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.DAILY,
        priority=99,
        output_file="realtime_quote.parquet",
        realtime=True,  # 标记为实时数据，排除
        enabled=False,
    ),
]

# ---------- 3. 公司基本面数据域 (Fundamental) ----------
FUNDAMENTAL_TASKS = [
    # 公司静态画像（时间无关）
    CollectionTask(
        name="company_info",
        description="上市公司基本信息",
        domain="fundamental",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_company_info",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="company_info.parquet",
    ),
    CollectionTask(
        name="industry_class",
        description="行业分类（申万/中信/巨潮）",
        domain="fundamental",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_industry_class",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="industry_class.parquet",
    ),
    CollectionTask(
        name="main_business",
        description="主营业务介绍",
        domain="fundamental",
        category=DataCategory.TIME_INDEPENDENT,  # 该接口不接受日期参数
        collector_func="get_main_business",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.QUARTERLY,
        priority=5,
        output_file="main_business/{ts_code}.parquet",  # 按股票分文件存储
    ),
    CollectionTask(
        name="management",
        description="管理层信息",
        domain="fundamental",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_management",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.ONCE,
        priority=5,
        output_file="management/{ts_code}.parquet",
        batch_size=100,
    ),
    
    # 财务报表体系（时间相关）
    CollectionTask(
        name="balance_sheet",
        description="资产负债表",
        domain="fundamental",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_balance_sheet",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.QUARTERLY,
        priority=4,
        output_file="balance_sheet/{ts_code}.parquet",
        date_field="end_date",
        batch_size=50,
    ),
    CollectionTask(
        name="income_statement",
        description="利润表",
        domain="fundamental",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_income_statement",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.QUARTERLY,
        priority=4,
        output_file="income_statement/{ts_code}.parquet",
        date_field="end_date",
        batch_size=50,
    ),
    CollectionTask(
        name="cash_flow",
        description="现金流量表",
        domain="fundamental",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_cash_flow",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.QUARTERLY,
        priority=4,
        output_file="cash_flow/{ts_code}.parquet",
        date_field="end_date",
        batch_size=50,
    ),
    CollectionTask(
        name="financial_indicator",
        description="财务指标（ROE/毛利率/杜邦）",
        domain="fundamental",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_financial_indicator",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.QUARTERLY,
        priority=4,
        output_file="financial_indicator/{ts_code}.parquet",
        date_field="end_date",
        batch_size=50,
    ),
    
    # 股权与资本结构（混合）
    CollectionTask(
        name="share_structure",
        description="股本结构",
        domain="fundamental",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_share_structure",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.QUARTERLY,
        priority=5,
        output_file="share_structure.parquet",
        date_field="ann_date",
    ),
    CollectionTask(
        name="top10_holders",
        description="前十大股东/流通股东",
        domain="fundamental",
        category=DataCategory.TIME_INDEPENDENT,  # 该接口不接受日期参数
        collector_func="get_top10_holders",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.QUARTERLY,
        priority=5,
        output_file="top10_holders/{ts_code}.parquet",
        batch_size=50,
    ),
    CollectionTask(
        name="pledge",
        description="股权质押",
        domain="fundamental",
        category=DataCategory.TIME_INDEPENDENT,  # 该接口不接受日期参数
        collector_func="get_pledge",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="pledge/{ts_code}.parquet",
    ),
    CollectionTask(
        name="share_float",
        description="限售解禁",
        domain="fundamental",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_share_float",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="share_float.parquet",
        date_field="ann_date",
    ),
    CollectionTask(
        name="repurchase",
        description="股票回购",
        domain="fundamental",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_repurchase",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="repurchase.parquet",
        date_field="ann_date",
    ),
    CollectionTask(
        name="dividend",
        description="分红送股",
        domain="fundamental",
        category=DataCategory.TIME_INDEPENDENT,  # 该接口不接受日期参数
        collector_func="get_dividend",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.YEARLY,
        priority=5,
        output_file="dividend/{ts_code}.parquet",
        batch_size=100,
    ),
]

# ---------- 4. 资金与交易行为数据域 (Trading Behavior) ----------
TRADING_BEHAVIOR_TASKS = [
    # 资金流向
    CollectionTask(
        name="money_flow",
        description="个股资金流向",
        domain="trading_behavior",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_money_flow",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="money_flow/{ts_code}.parquet",
        batch_size=50,
    ),
    CollectionTask(
        name="money_flow_industry",
        description="行业/板块资金流向",
        domain="trading_behavior",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_money_flow_industry",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="money_flow_industry.parquet",
    ),
    CollectionTask(
        name="money_flow_market",
        description="大盘资金流向",
        domain="trading_behavior",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_money_flow_market",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="money_flow_market.parquet",
    ),
    CollectionTask(
        name="hsgt_flow",
        description="沪深港通资金流向",
        domain="trading_behavior",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_hsgt_flow",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="hsgt_flow.parquet",
    ),
    
    # 融资融券
    CollectionTask(
        name="margin_summary",
        description="融资融券汇总",
        domain="trading_behavior",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_margin_summary",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="margin_summary.parquet",
    ),
    CollectionTask(
        name="margin_detail",
        description="融资融券明细",
        domain="trading_behavior",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_margin_detail",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="margin_detail/{ts_code}.parquet",
        batch_size=50,
    ),
    CollectionTask(
        name="margin_target",
        description="两融标的",
        domain="trading_behavior",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_margin_target",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=4,
        output_file="margin_target.parquet",
    ),
    CollectionTask(
        name="slb",
        description="转融通",
        domain="trading_behavior",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_slb",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="slb.parquet",
    ),
    
    # 特殊交易行为
    CollectionTask(
        name="top_list",
        description="龙虎榜",
        domain="trading_behavior",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_top_list",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="top_list.parquet",
    ),
    CollectionTask(
        name="top_inst",
        description="龙虎榜机构/营业部",
        domain="trading_behavior",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_top_inst",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="top_inst.parquet",
    ),
    CollectionTask(
        name="block_trade",
        description="大宗交易",
        domain="trading_behavior",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_block_trade",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="block_trade.parquet",
    ),
]

# ---------- 5. 板块/行业/主题数据域 (Cross-sectional) ----------
CROSS_SECTIONAL_TASKS = [
    # 行业体系
    CollectionTask(
        name="sw_index_classify",
        description="申万行业分类",
        domain="cross_sectional",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_sw_index_classify",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="sw_index_classify.parquet",
    ),
    CollectionTask(
        name="sw_index_member",
        description="申万行业成分股",
        domain="cross_sectional",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_sw_index_member",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="sw_index_member.parquet",
    ),
    CollectionTask(
        name="sw_daily",
        description="申万行业指数行情",
        domain="cross_sectional",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_sw_daily",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="sw_daily.parquet",
    ),
    
    # 概念与主题板块
    CollectionTask(
        name="ths_index",
        description="同花顺概念指数列表",
        domain="cross_sectional",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_ths_index",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="ths_index.parquet",
    ),
    CollectionTask(
        name="ths_member",
        description="同花顺概念成分股",
        domain="cross_sectional",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_ths_member",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="ths_member.parquet",
    ),
    CollectionTask(
        name="ths_daily",
        description="同花顺概念指数行情",
        domain="cross_sectional",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_ths_daily",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="ths_daily.parquet",
    ),
    CollectionTask(
        name="dc_index",
        description="东方财富概念板块",
        domain="cross_sectional",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_dc_index",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="dc_index.parquet",
    ),
    CollectionTask(
        name="dc_member",
        description="东方财富概念成分股",
        domain="cross_sectional",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_dc_member",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="dc_member.parquet",
    ),
    CollectionTask(
        name="kpl_concept",
        description="开盘啦热点概念",
        domain="cross_sectional",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_kpl_concept",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=4,
        output_file="kpl_concept.parquet",
    ),
    CollectionTask(
        name="kpl_concept_cons",
        description="开盘啦概念成分股",
        domain="cross_sectional",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_kpl_concept_cons",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=4,
        output_file="kpl_concept_cons.parquet",
    ),
    
    # 板块行情与强弱
    CollectionTask(
        name="sector_performance",
        description="板块涨跌幅表现",
        domain="cross_sectional",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_sector_performance",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="sector_performance.parquet",
    ),
    CollectionTask(
        name="industry_board_em",
        description="东方财富行业板块行情",
        domain="cross_sectional",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_industry_board_em",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="industry_board_em.parquet",
    ),
    CollectionTask(
        name="concept_board_em",
        description="东方财富概念板块行情",
        domain="cross_sectional",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_concept_board_em",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="concept_board_em.parquet",
    ),
    CollectionTask(
        name="sector_hist",
        description="板块历史行情",
        domain="cross_sectional",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_sector_hist",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="sector_hist.parquet",
    ),
    CollectionTask(
        name="sector_rank",
        description="板块热度排行",
        domain="cross_sectional",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_sector_rank",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="sector_rank.parquet",
    ),
    CollectionTask(
        name="limit_up_pool",
        description="涨停板池",
        domain="cross_sectional",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_limit_up_pool",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="limit_up_pool.parquet",
    ),
]

# ---------- 6. 衍生品与多资产数据域 (Derivatives) ----------
DERIVATIVES_TASKS = [
    # ETF与基金
    CollectionTask(
        name="fund_basic",
        description="ETF/基金基本信息",
        domain="derivatives",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_fund_basic",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="fund_basic.parquet",
    ),
    CollectionTask(
        name="fund_daily",
        description="ETF/LOF日线行情",
        domain="derivatives",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_fund_daily",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="fund_daily.parquet",
    ),
    CollectionTask(
        name="fund_nav",
        description="基金净值",
        domain="derivatives",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_fund_nav",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="fund_nav.parquet",
        date_field="ann_date",
    ),
    CollectionTask(
        name="fund_portfolio",
        description="基金持仓",
        domain="derivatives",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_fund_portfolio",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.QUARTERLY,
        priority=6,
        output_file="fund_portfolio.parquet",
        date_field="ann_date",
    ),
    CollectionTask(
        name="fund_share",
        description="基金规模",
        domain="derivatives",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_fund_share",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.QUARTERLY,
        priority=6,
        output_file="fund_share.parquet",
        date_field="ann_date",
    ),
    
    # 期货与期权
    CollectionTask(
        name="fut_basic",
        description="期货合约信息",
        domain="derivatives",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_fut_basic",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="fut_basic.parquet",
    ),
    CollectionTask(
        name="fut_daily",
        description="期货日线行情",
        domain="derivatives",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_fut_daily",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="fut_daily.parquet",
    ),
    CollectionTask(
        name="fut_holding",
        description="期货持仓排名",
        domain="derivatives",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_fut_holding",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="fut_holding.parquet",
    ),
    CollectionTask(
        name="fut_wsr",
        description="期货仓单",
        domain="derivatives",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_fut_wsr",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="fut_wsr.parquet",
    ),
    CollectionTask(
        name="opt_basic",
        description="期权合约信息",
        domain="derivatives",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_opt_basic",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="opt_basic.parquet",
    ),
    CollectionTask(
        name="opt_daily",
        description="期权日线行情",
        domain="derivatives",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_opt_daily",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="opt_daily.parquet",
    ),
    
    # 债券与可转债
    CollectionTask(
        name="yield_curve",
        description="国债收益率曲线",
        domain="derivatives",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_yield_curve",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=4,
        output_file="yield_curve.parquet",
    ),
    CollectionTask(
        name="cb_basic",
        description="可转债基本信息",
        domain="derivatives",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_cb_basic",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="cb_basic.parquet",
    ),
    CollectionTask(
        name="cb_daily",
        description="可转债日线行情",
        domain="derivatives",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_cb_daily",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="cb_daily.parquet",
    ),
    CollectionTask(
        name="repo_daily",
        description="回购利率",
        domain="derivatives",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_repo_daily",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="repo_daily.parquet",
    ),
    CollectionTask(
        name="cb_premium",
        description="可转债溢价率",
        domain="derivatives",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_cb_premium",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="cb_premium.parquet",
    ),
]

# ---------- 7. 指数与基准数据域 (Index & Benchmark) ----------
INDEX_BENCHMARK_TASKS = [
    CollectionTask(
        name="index_basic",
        description="指数基本信息",
        domain="index_benchmark",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_index_basic",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=2,
        output_file="index_basic.parquet",
    ),
    CollectionTask(
        name="index_daily",
        description="指数日线行情",
        domain="index_benchmark",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_index_daily",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=4,
        output_file="index_daily.parquet",
    ),
    CollectionTask(
        name="index_weight",
        description="指数成分权重",
        domain="index_benchmark",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_index_weight",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=5,
        output_file="index_weight.parquet",
    ),
    CollectionTask(
        name="index_member",
        description="指数成分股",
        domain="index_benchmark",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_index_member",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=3,
        output_file="index_member.parquet",
    ),
    CollectionTask(
        name="index_global",
        description="全球主要指数",
        domain="index_benchmark",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_index_global",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="index_global.parquet",
    ),
]

# ---------- 8. 宏观与外生变量域 (Macro & Exogenous) ----------
MACRO_EXOGENOUS_TASKS = [
    # 国内宏观
    CollectionTask(
        name="cn_gdp",
        description="中国GDP数据",
        domain="macro_exogenous",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_cn_gdp",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.QUARTERLY,
        priority=4,
        output_file="cn_gdp.parquet",
    ),
    CollectionTask(
        name="cn_cpi",
        description="中国CPI数据",
        domain="macro_exogenous",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_cn_cpi",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=4,
        output_file="cn_cpi.parquet",
    ),
    CollectionTask(
        name="cn_ppi",
        description="中国PPI数据",
        domain="macro_exogenous",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_cn_ppi",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=4,
        output_file="cn_ppi.parquet",
    ),
    CollectionTask(
        name="cn_pmi",
        description="中国PMI数据",
        domain="macro_exogenous",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_cn_pmi",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=4,
        output_file="cn_pmi.parquet",
    ),
    CollectionTask(
        name="cn_m2",
        description="货币供应量（M0/M1/M2）",
        domain="macro_exogenous",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_cn_m2",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=4,
        output_file="cn_m2.parquet",
    ),
    CollectionTask(
        name="shibor",
        description="Shibor利率",
        domain="macro_exogenous",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_shibor",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=4,
        output_file="shibor.parquet",
    ),
    CollectionTask(
        name="lpr",
        description="LPR利率",
        domain="macro_exogenous",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_lpr",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=4,
        output_file="lpr.parquet",
    ),
    CollectionTask(
        name="social_finance",
        description="社会融资规模",
        domain="macro_exogenous",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_sf",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=4,
        output_file="social_finance.parquet",
    ),
    
    # 国际宏观
    CollectionTask(
        name="us_treasury",
        description="美国国债收益率",
        domain="macro_exogenous",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_us_treasury",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="us_treasury.parquet",
    ),
    CollectionTask(
        name="eco_calendar",
        description="经济数据日历",
        domain="macro_exogenous",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_eco_calendar",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="eco_calendar.parquet",
    ),
    
    # 行业与现实经济映射
    CollectionTask(
        name="box_office",
        description="电影票房数据",
        domain="macro_exogenous",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_box_office",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="box_office.parquet",
    ),
    CollectionTask(
        name="car_sales",
        description="汽车销量数据",
        domain="macro_exogenous",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_car_sales",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=6,
        output_file="car_sales.parquet",
    ),
]

# ---------- 9. 预期与预测分析域 (Expectations & Forecasts) ----------
EXPECTATIONS_TASKS = [
    # 盈利预测
    CollectionTask(
        name="earnings_forecast",
        description="业绩预告（公司官方）",
        domain="expectations",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_earnings_forecast",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.QUARTERLY,
        priority=5,
        output_file="earnings_forecast.parquet",
        date_field="ann_date",
    ),
    CollectionTask(
        name="broker_forecast",
        description="券商盈利预测",
        domain="expectations",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_broker_forecast",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.MONTHLY,
        priority=6,
        output_file="broker_forecast/{ts_code}.parquet",
        date_field="ann_date",
        batch_size=100,
    ),
    CollectionTask(
        name="consensus_forecast",
        description="一致预期数据（EPS/营收/净利润预测）",
        domain="expectations",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_consensus_forecast",
        stock_scope=StockScope.ALL_A,
        frequency=CollectionFrequency.MONTHLY,
        priority=6,
        output_file="consensus_forecast/{ts_code}.parquet",
        date_field="ann_date",
        batch_size=100,
    ),
    
    # 机构评级
    CollectionTask(
        name="inst_rating",
        description="机构投资评级",
        domain="expectations",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_inst_rating",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="inst_rating.parquet",
        date_field="ann_date",
    ),
    CollectionTask(
        name="rating_summary",
        description="评级汇总统计（买入/增持评级统计）",
        domain="expectations",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_rating_summary",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=5,
        output_file="rating_summary.parquet",
    ),
    CollectionTask(
        name="inst_survey",
        description="机构调研记录",
        domain="expectations",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_inst_survey",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="inst_survey.parquet",
        date_field="ann_date",
    ),
    
    # 研究员指数
    CollectionTask(
        name="analyst_rank",
        description="分析师指数排行",
        domain="expectations",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_analyst_rank",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=6,
        output_file="analyst_rank.parquet",
    ),
    CollectionTask(
        name="analyst_detail",
        description="分析师详情",
        domain="expectations",
        category=DataCategory.TIME_INDEPENDENT,
        collector_func="get_analyst_detail",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.ONCE,
        priority=5,
        output_file="analyst_detail.parquet",
    ),
    CollectionTask(
        name="broker_gold_stock",
        description="券商月度金股组合",
        domain="expectations",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_broker_gold_stock",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=6,
        output_file="broker_gold_stock.parquet",
    ),
    CollectionTask(
        name="forecast_revision",
        description="预测修正数据",
        domain="expectations",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_forecast_revision",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=6,
        output_file="forecast_revision.parquet",
        date_field="ann_date",
    ),
]

# ---------- 10. 深度风险与质量因子域 (Deep Risk & Quality) ----------
DEEP_RISK_QUALITY_TASKS = [
    # 估值扩散与拥挤度
    CollectionTask(
        name="a_pe_pb_ew_median",
        description="A股等权重与中位数PE/PB",
        domain="deep_risk_quality",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_a_pe_pb_ew_median",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="a_pe_pb_ew_median.parquet",
    ),
    CollectionTask(
        name="market_congestion",
        description="大盘拥挤度",
        domain="deep_risk_quality",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_market_congestion",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="market_congestion.parquet",
    ),
    CollectionTask(
        name="stock_bond_spread",
        description="股债利差",
        domain="deep_risk_quality",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_stock_bond_spread",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="stock_bond_spread.parquet",
    ),
    CollectionTask(
        name="buffett_indicator",
        description="巴菲特指标",
        domain="deep_risk_quality",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_buffett_indicator",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=5,
        output_file="buffett_indicator.parquet",
    ),
    
    # 资产质量异常
    CollectionTask(
        name="stock_goodwill",
        description="个股商誉明细",
        domain="deep_risk_quality",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_stock_goodwill",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.QUARTERLY,
        priority=5,
        output_file="stock_goodwill.parquet",
        date_field="ann_date",
    ),
    CollectionTask(
        name="goodwill_impairment",
        description="商誉减值预期明细",
        domain="deep_risk_quality",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_goodwill_impairment",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.QUARTERLY,
        priority=5,
        output_file="goodwill_impairment.parquet",
        date_field="ann_date",
    ),
    CollectionTask(
        name="break_net_stock",
        description="破净股统计",
        domain="deep_risk_quality",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_break_net_stock",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.DAILY,
        priority=5,
        output_file="break_net_stock.parquet",
    ),
    
    # ESG评价
    CollectionTask(
        name="esg_msci",
        description="MSCI-ESG评级",
        domain="deep_risk_quality",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_esg_msci",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=5,
        output_file="esg_msci.parquet",
    ),
    CollectionTask(
        name="esg_hz",
        description="华证指数-ESG评级",
        domain="deep_risk_quality",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_esg_hz",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=5,
        output_file="esg_hz.parquet",
    ),
    CollectionTask(
        name="esg_refinitiv",
        description="路孚特-ESG评级",
        domain="deep_risk_quality",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_esg_refinitiv",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=5,
        output_file="esg_refinitiv.parquet",
    ),
    CollectionTask(
        name="esg_zhiding",
        description="秩鼎-ESG评级",
        domain="deep_risk_quality",
        category=DataCategory.TIME_DEPENDENT,
        collector_func="get_esg_zhiding",
        stock_scope=StockScope.NONE,
        frequency=CollectionFrequency.MONTHLY,
        priority=5,
        output_file="esg_zhiding.parquet",
    ),
]


# ==================== 汇总所有任务 ====================

ALL_TASKS = (
    METADATA_TASKS +
    MARKET_DATA_TASKS +
    FUNDAMENTAL_TASKS +
    TRADING_BEHAVIOR_TASKS +
    CROSS_SECTIONAL_TASKS +
    DERIVATIVES_TASKS +
    INDEX_BENCHMARK_TASKS +
    MACRO_EXOGENOUS_TASKS +
    EXPECTATIONS_TASKS +
    DEEP_RISK_QUALITY_TASKS
)

# 按数据域分组
TASKS_BY_DOMAIN = {
    "metadata": METADATA_TASKS,
    "market_data": MARKET_DATA_TASKS,
    "fundamental": FUNDAMENTAL_TASKS,
    "trading_behavior": TRADING_BEHAVIOR_TASKS,
    "cross_sectional": CROSS_SECTIONAL_TASKS,
    "derivatives": DERIVATIVES_TASKS,
    "index_benchmark": INDEX_BENCHMARK_TASKS,
    "macro_exogenous": MACRO_EXOGENOUS_TASKS,
    "expectations": EXPECTATIONS_TASKS,
    "deep_risk_quality": DEEP_RISK_QUALITY_TASKS,
}

# 数据域名称映射
DOMAIN_NAMES = {
    "metadata": "基础元数据域",
    "market_data": "市场行情数据域",
    "fundamental": "公司基本面数据域",
    "trading_behavior": "资金与交易行为数据域",
    "cross_sectional": "板块/行业/主题数据域",
    "derivatives": "衍生品与多资产数据域",
    "index_benchmark": "指数与基准数据域",
    "macro_exogenous": "宏观与外生变量域",
    "expectations": "预期与预测分析域",
    "deep_risk_quality": "深度风险与质量因子域",
}


def get_tasks_by_category(category: DataCategory) -> List[CollectionTask]:
    """按数据类别获取任务"""
    return [task for task in ALL_TASKS if task.category == category and task.enabled and not task.realtime]


def get_enabled_tasks() -> List[CollectionTask]:
    """获取所有启用的非实时任务"""
    return [task for task in ALL_TASKS if task.enabled and not task.realtime]


def get_tasks_by_domain(domain: str) -> List[CollectionTask]:
    """按数据域获取任务"""
    return TASKS_BY_DOMAIN.get(domain, [])


def get_tasks_sorted_by_priority() -> List[CollectionTask]:
    """按优先级排序获取任务"""
    return sorted(get_enabled_tasks(), key=lambda x: x.priority)
