"""
DWD层（明细数据层）配置文件

定义数据路径、处理参数和字段映射
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================================
# 路径配置
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[5]  # 项目根目录
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "structured"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed" / "structured"
DWD_OUTPUT_DIR = PROCESSED_DATA_DIR / "dwd"


# ============================================================================
# 数据源路径配置
# ============================================================================
@dataclass
class DataSourcePaths:
    """原始数据源路径配置"""
    # 元数据
    stock_list_a: Path = RAW_DATA_DIR / "metadata" / "stock_list_a.parquet"
    trade_calendar: Path = RAW_DATA_DIR / "metadata" / "trade_calendar.parquet"
    suspend_info: Path = RAW_DATA_DIR / "metadata" / "suspend_info.parquet"
    st_status: Path = RAW_DATA_DIR / "metadata" / "st_status.parquet"
    
    # 市场行情
    stock_daily_dir: Path = RAW_DATA_DIR / "market_data" / "stock_daily"
    adj_factor_dir: Path = RAW_DATA_DIR / "market_data" / "adj_factor"
    daily_basic_dir: Path = RAW_DATA_DIR / "market_data" / "daily_basic"
    
    # 基本面数据
    balance_sheet_dir: Path = RAW_DATA_DIR / "fundamental" / "balance_sheet"
    income_statement_dir: Path = RAW_DATA_DIR / "fundamental" / "income_statement"
    cash_flow_dir: Path = RAW_DATA_DIR / "fundamental" / "cash_flow"
    financial_indicator_dir: Path = RAW_DATA_DIR / "fundamental" / "financial_indicator"
    
    # ========== 资金流向与筹码 ==========
    # 资金流向（日频）
    money_flow_dir: Path = RAW_DATA_DIR / "trading_behavior" / "money_flow"
    margin_detail_dir: Path = RAW_DATA_DIR / "trading_behavior" / "margin_detail"
    top_list: Path = RAW_DATA_DIR / "trading_behavior" / "top_list.parquet"
    top_inst_dir: Path = RAW_DATA_DIR / "trading_behavior" / "top_inst"
    hsgt_flow: Path = RAW_DATA_DIR / "trading_behavior" / "hsgt_flow.parquet"
    block_trade: Path = RAW_DATA_DIR / "trading_behavior" / "block_trade.parquet"  # 大宗交易
    margin_summary: Path = RAW_DATA_DIR / "trading_behavior" / "margin_summary.parquet"  # 全市场两融余额
    
    # 筹码结构（低频）
    top10_holders_dir: Path = RAW_DATA_DIR / "fundamental" / "top10_holders"
    share_structure_dir: Path = RAW_DATA_DIR / "fundamental" / "share_structure"
    
    # ========== 行业分类 ==========
    sw_index_member: Path = RAW_DATA_DIR / "cross_sectional" / "sw_index_member.parquet"
    sw_index_classify: Path = RAW_DATA_DIR / "cross_sectional" / "sw_index_classify.parquet"
    
    # ========== 事件数据 ==========
    repurchase: Path = RAW_DATA_DIR / "fundamental" / "repurchase.parquet"
    share_float: Path = RAW_DATA_DIR / "fundamental" / "share_float.parquet"
    pledge_dir: Path = RAW_DATA_DIR / "fundamental" / "pledge"
    dividend_dir: Path = RAW_DATA_DIR / "fundamental" / "dividend"
    
    # ========== 宏观数据 ==========
    cn_gdp: Path = RAW_DATA_DIR / "macro_exogenous" / "cn_gdp.parquet"
    cn_cpi: Path = RAW_DATA_DIR / "macro_exogenous" / "cn_cpi.parquet"
    cn_pmi: Path = RAW_DATA_DIR / "macro_exogenous" / "cn_pmi.parquet"
    cn_m2: Path = RAW_DATA_DIR / "macro_exogenous" / "cn_m2.parquet"
    lpr: Path = RAW_DATA_DIR / "macro_exogenous" / "lpr.parquet"
    shibor: Path = RAW_DATA_DIR / "macro_exogenous" / "shibor.parquet"
    
    # 市场风险指标
    market_congestion: Path = RAW_DATA_DIR / "deep_risk_quality" / "market_congestion.parquet"
    stock_bond_spread: Path = RAW_DATA_DIR / "deep_risk_quality" / "stock_bond_spread.parquet"
    
    # ========== 深度风险因子 ==========
    a_pe_pb_ew_median: Path = RAW_DATA_DIR / "deep_risk_quality" / "a_pe_pb_ew_median.parquet"
    buffett_indicator: Path = RAW_DATA_DIR / "deep_risk_quality" / "buffett_indicator.parquet"
    break_net_stock: Path = RAW_DATA_DIR / "deep_risk_quality" / "break_net_stock.parquet"
    
    # ========== 指数与基准 ==========
    index_daily_dir: Path = RAW_DATA_DIR / "market_data" / "index_daily"
    
    # ========== 衍生品数据 ==========
    repo_daily_dir: Path = RAW_DATA_DIR / "derivatives" / "repo_daily"
    fut_daily_dir: Path = RAW_DATA_DIR / "derivatives" / "fut_daily"
    opt_basic: Path = RAW_DATA_DIR / "derivatives" / "opt_basic.parquet"


# ============================================================================
# 输出宽表配置
# ============================================================================
@dataclass
class DWDOutputConfig:
    """DWD输出配置"""
    # 输出文件名 - 核心宽表
    stock_price: str = "dwd_stock_price.parquet"
    stock_fundamental: str = "dwd_stock_fundamental.parquet"
    stock_status: str = "dwd_stock_status.parquet"
    
    # 输出文件名 - 扩展宽表
    money_flow: str = "dwd_money_flow.parquet"
    chip_structure: str = "dwd_chip_structure.parquet"
    stock_industry: str = "dwd_stock_industry.parquet"
    event_signal: str = "dwd_event_signal.parquet"
    macro_env: str = "dwd_macro_env.parquet"
    
    # 输出目录
    output_dir: Path = DWD_OUTPUT_DIR
    
    def get_output_path(self, table_name: str) -> Path:
        """获取输出表的完整路径"""
        return self.output_dir / getattr(self, table_name)


# ============================================================================
# 板块/市场配置
# ============================================================================
@dataclass
class MarketConfig:
    """市场板块配置"""
    # 板块涨跌停限制
    LIMIT_RATIOS: Dict[str, float] = field(default_factory=lambda: {
        '主板': 0.10,      # 主板 10%
        'main': 0.10,
        '中小板': 0.10,    # 中小板 10%
        'sme': 0.10,
        '创业板': 0.20,    # 创业板 20%
        'gem': 0.20,
        '科创板': 0.20,    # 科创板 20%
        'star': 0.20,
        '北交所': 0.30,    # 北交所 30%
        'bse': 0.30,
    })
    
    # ST股票涨跌停限制
    ST_LIMIT_RATIOS: Dict[str, float] = field(default_factory=lambda: {
        '主板': 0.05,      # 主板ST 5%
        'main': 0.05,
        '中小板': 0.05,    # 中小板ST 5%
        'sme': 0.05,
        '创业板': 0.20,    # 创业板ST 20%
        'gem': 0.20,
        '科创板': 0.20,    # 科创板ST 20%
        'star': 0.20,
        '北交所': 0.30,    # 北交所ST 30%
        'bse': 0.30,
    })
    
    # 新股上市不设涨跌停限制的天数
    NEW_STOCK_NO_LIMIT_DAYS: Dict[str, int] = field(default_factory=lambda: {
        '主板': 5,
        'main': 5,
        '中小板': 5,
        'sme': 5,
        '创业板': 5,
        'gem': 5,
        '科创板': 5,
        'star': 5,
        '北交所': 5,
        'bse': 5,
    })
    
    # 新股定义：上市不满N天
    NEW_STOCK_DAYS: int = 60


# ============================================================================
# 财务数据配置
# ============================================================================
@dataclass
class FundamentalConfig:
    """基本面数据配置"""
    # 需要单季拆分的利润表字段（累计值字段）
    INCOME_CUMULATIVE_FIELDS: List[str] = field(default_factory=lambda: [
        'total_revenue',       # 营业总收入
        'revenue',             # 营业收入
        'int_income',          # 利息收入
        'prem_earned',         # 已赚保费
        'total_cogs',          # 营业总成本
        'oper_cost',           # 营业成本
        'int_exp',             # 利息支出
        'biz_tax_surchg',      # 营业税金及附加
        'sell_exp',            # 销售费用
        'admin_exp',           # 管理费用
        'fin_exp',             # 财务费用
        'rd_exp',              # 研发费用
        'assets_impair_loss',  # 资产减值损失
        'operate_profit',      # 营业利润
        'non_oper_income',     # 营业外收入
        'non_oper_exp',        # 营业外支出
        'total_profit',        # 利润总额
        'income_tax',          # 所得税费用
        'n_income',            # 净利润
        'n_income_attr_p',     # 归属母公司股东的净利润
        'minority_gain',       # 少数股东损益
    ])
    
    # 需要单季拆分的现金流量表字段（累计值字段）
    CASHFLOW_CUMULATIVE_FIELDS: List[str] = field(default_factory=lambda: [
        'n_cashflow_act',          # 经营活动产生的现金流量净额
        'c_fr_sale_sg',            # 销售商品、提供劳务收到的现金
        'c_pay_for_goods',         # 购买商品、接受劳务支付的现金
        'c_pay_to_for_empl',       # 支付给职工以及为职工支付的现金
        'c_pay_for_tax',           # 支付的各项税费
        'n_cashflow_inv_act',      # 投资活动产生的现金流量净额
        'c_fr_disp_fix_ast',       # 处置固定资产等收回的现金净额
        'c_pay_acq_fix_ast',       # 购建固定资产等支付的现金
        'c_pay_acq_stock',         # 投资支付的现金
        'n_cash_flows_fnc_act',    # 筹资活动产生的现金流量净额
        'c_fr_short_loan',         # 取得借款收到的现金
        'c_fr_issue_share',        # 发行债券收到的现金
        'c_pay_repmt_debt',        # 偿还债务支付的现金
        'c_pay_div_profit',        # 分配股利、利润或偿付利息支付的现金
        'n_incr_cash_cash_equ',    # 现金及现金等价物净增加额
    ])
    
    # 资产负债表字段（期末时点值，不需要单季拆分）
    BALANCE_SHEET_FIELDS: List[str] = field(default_factory=lambda: [
        'total_cur_assets',     # 流动资产合计
        'money_cap',            # 货币资金
        'accounts_receiv',      # 应收账款
        'inventories',          # 存货
        'total_nca',            # 非流动资产合计
        'fix_assets',           # 固定资产
        'total_assets',         # 总资产
        'total_cur_liab',       # 流动负债合计
        'st_borr',              # 短期借款
        'total_ncl',            # 非流动负债合计
        'lt_borr',              # 长期借款
        'total_liab',           # 负债合计
        'total_hldr_eqy_exc_min_int',  # 股东权益合计（不含少数股东权益）
        'total_hldr_eqy_inc_min_int',  # 股东权益合计（含少数股东权益）
        'minority_int',         # 少数股东权益
    ])
    
    # 财务指标字段
    FINANCIAL_INDICATOR_FIELDS: List[str] = field(default_factory=lambda: [
        'eps',                  # 每股收益
        'bps',                  # 每股净资产
        'roe',                  # 净资产收益率
        'roe_waa',              # 加权平均净资产收益率
        'roa',                  # 总资产收益率
        'gross_margin',         # 毛利率
        'netprofit_margin',     # 净利率
        'current_ratio',        # 流动比率
        'quick_ratio',          # 速动比率
        'debt_to_assets',       # 资产负债率
        'netprofit_yoy',        # 净利润同比增长率
        'or_yoy',               # 营收同比增长率
    ])


# ============================================================================
# DWD宽表字段定义
# ============================================================================
@dataclass
class DWDFieldsConfig:
    """DWD宽表字段定义"""
    
    # dwd_stock_price 基础量价宽表字段
    STOCK_PRICE_FIELDS: List[str] = field(default_factory=lambda: [
        # 主键
        'trade_date',          # 交易日期 YYYY-MM-DD
        'ts_code',             # 证券代码
        # 不复权价格
        'open',                # 开盘价
        'high',                # 最高价
        'low',                 # 最低价
        'close',               # 收盘价
        'pre_close',           # 昨收价
        # 后复权价格 (HFQ)
        'open_hfq',            # 后复权开盘价
        'high_hfq',            # 后复权最高价
        'low_hfq',             # 后复权最低价
        'close_hfq',           # 后复权收盘价
        # 成交数据
        'vol',                 # 成交量（手）
        'amount',              # 成交额（千元）
        'adj_factor',          # 复权因子
        # 衍生指标
        'vwap',                # 不复权成交均价
        'vwap_hfq',            # 后复权成交均价
        'return_1d',           # 日涨跌幅（基于后复权价格）
        'turnover',            # 换手率
        # 状态标记
        'is_trading',          # 是否交易 (1=交易, 0=停牌)
    ])
    
    # dwd_stock_fundamental 基本面宽表字段
    STOCK_FUNDAMENTAL_FIELDS: List[str] = field(default_factory=lambda: [
        # 主键
        'trade_date',          # 交易日期 YYYY-MM-DD
        'ts_code',             # 证券代码
        # 市值指标
        'total_mv',            # 总市值
        'circ_mv',             # 流通市值
        'pe_ttm',              # 市盈率TTM
        'pb',                  # 市净率
        'ps_ttm',              # 市销率TTM
        # 利润表TTM (单季加总后的滚动4季)
        'revenue_ttm',         # 营业收入TTM
        'operate_profit_ttm',  # 营业利润TTM
        'total_profit_ttm',    # 利润总额TTM
        'n_income_attr_p_ttm', # 归母净利润TTM
        # 单季度数据
        'revenue_sq',          # 营业收入单季
        'n_income_attr_p_sq',  # 归母净利润单季
        # 财务指标
        'roe',                 # 净资产收益率
        'roa',                 # 总资产收益率
        'gross_margin',        # 毛利率
        'netprofit_margin',    # 净利率
        'debt_to_assets',      # 资产负债率
        # 增长率
        'revenue_yoy',         # 营收同比
        'net_profit_yoy',      # 净利润同比
        # 资产负债表关键项目
        'total_assets',        # 总资产
        'total_liab',          # 总负债
        'total_equity',        # 股东权益
        # PIT元数据
        'report_date',         # 财报期（YYYYMMDD）
        'ann_date',            # 公告日期
    ])
    
    # dwd_stock_status 状态与风险掩码表字段
    STOCK_STATUS_FIELDS: List[str] = field(default_factory=lambda: [
        # 主键
        'trade_date',          # 交易日期 YYYY-MM-DD
        'ts_code',             # 证券代码
        # ST状态
        'is_st',               # 是否ST (包括*ST、ST)
        # 涨跌停状态
        'is_limit_up',         # 是否涨停（无法买入）
        'is_limit_down',       # 是否跌停（无法卖出）
        # 上市状态
        'is_new',              # 是否新股（上市不满60天）
        'is_new_no_limit',     # 是否新股无涨跌停限制期
        # 交易状态
        'is_trading',          # 是否交易（非停牌）
        # 辅助字段
        'list_date',           # 上市日期
        'market',              # 市场板块
        'limit_ratio',         # 涨跌停比例
    ])


# ============================================================================
# 处理参数配置
# ============================================================================
@dataclass
class ProcessingConfig:
    """处理参数配置"""
    # 数据时间范围
    start_date: str = "2021-01-01"
    end_date: str = "2025-12-31"
    
    # 日期格式
    date_format: str = "%Y-%m-%d"
    date_format_compact: str = "%Y%m%d"
    
    # 是否使用GPU加速
    use_gpu: bool = True
    
    # 批处理大小（处理股票数量）
    batch_size: int = 500
    
    # 是否覆盖已存在的输出文件
    overwrite: bool = True
    
    # 缺失值处理
    fill_na_price: float = 0.0
    fill_na_volume: float = 0.0
    
    # Parquet压缩算法
    compression: str = "snappy"


# ============================================================================
# 全局配置实例
# ============================================================================
DATA_SOURCE_PATHS = DataSourcePaths()
DWD_OUTPUT_CONFIG = DWDOutputConfig()
MARKET_CONFIG = MarketConfig()
FUNDAMENTAL_CONFIG = FundamentalConfig()
DWD_FIELDS_CONFIG = DWDFieldsConfig()
PROCESSING_CONFIG = ProcessingConfig()
