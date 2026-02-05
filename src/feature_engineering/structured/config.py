"""
特征工程流水线配置

统一配置数据路径、特征计算参数、标签定义等。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import date


# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[3]  # 向上3级到项目根目录

# 数据路径
DWD_INPUT_DIR = BASE_DIR / "data" / "processed" / "structured" / "dwd"
PREPROCESSED_DIR = BASE_DIR / "data" / "features" / "preprocessed"  # 预处理后数据
FEATURE_OUTPUT_DIR = BASE_DIR / "data" / "features" / "structured"


@dataclass
class DataConfig:
    """数据路径配置"""
    
    # 输入：DWD 三张核心宽表（原始数据，流水线内部会进行预处理）
    input_dir: Path = DWD_INPUT_DIR
    price_table: str = "dwd_stock_price.parquet"
    fundamental_table: str = "dwd_stock_fundamental.parquet"
    status_table: str = "dwd_stock_status.parquet"
    
    # 输入：DWD 五张扩展宽表
    money_flow_table: str = "dwd_money_flow.parquet"
    chip_table: str = "dwd_chip_structure.parquet"
    industry_table: str = "dwd_stock_industry.parquet"
    event_table: str = "dwd_event_signal.parquet"
    macro_table: str = "dwd_macro_env.parquet"
    
    # 参考数据（用于特征生成阶段，不在 Merger 中合并）
    raw_data_dir: Path = BASE_DIR / "data" / "raw" / "structured"
    index_weight_dir: Path = raw_data_dir / "cross_sectional" / "index_weight"
    etf_daily_dir: Path = raw_data_dir / "market_data" / "etf_daily"
    index_daily_dir: Path = raw_data_dir / "market_data" / "index_daily"
    
    # 输出
    output_dir: Path = FEATURE_OUTPUT_DIR
    train_file: str = "train.parquet"
    
    # 数据时间范围
    # 预热期：2019-06-01 ~ 2020-12-31（用于计算 MA250 等长周期指标）
    warmup_start: str = "2019-06-01"
    warmup_end: str = "2020-12-31"
    
    # 正式期：2021-01-01 ~ 2025-12-31（用于训练/验证/测试）
    train_start: str = "2021-01-01"
    train_end: str = "2025-12-31"
    
    # 便捷属性：获取完整路径
    @property
    def price_path(self) -> Path:
        return self.input_dir / self.price_table
    
    @property
    def fundamental_path(self) -> Path:
        return self.input_dir / self.fundamental_table
    
    @property
    def status_path(self) -> Path:
        return self.input_dir / self.status_table
    
    @property
    def money_flow_path(self) -> Path:
        return self.input_dir / self.money_flow_table
    
    @property
    def chip_path(self) -> Path:
        return self.input_dir / self.chip_table
    
    @property
    def industry_path(self) -> Path:
        return self.input_dir / self.industry_table
    
    @property
    def event_path(self) -> Path:
        return self.input_dir / self.event_table
    
    @property
    def macro_path(self) -> Path:
        return self.input_dir / self.macro_table
    
    @property
    def merged_path(self) -> Path:
        """合并后的中间文件路径"""
        return self.output_dir / "merged.parquet"


@dataclass
class FeatureGroups:
    """
    字段组配置
    
    将 DWD 宽表字段按业务域分组，便于 FeatureGenerator 按组取用。
    不在白名单内的字段将被丢弃，避免无用字段进入模型。
    """
    
    # 资金流字段组 (dwd_money_flow)
    money_flow_fields: List[str] = field(default_factory=lambda: [
        # 分单资金流
        "net_mf_amount", "net_main_amount", "net_retail_amount",
        "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount",
        # 两融
        "rzye", "rqye", "rzmre", "rzche",
        # 龙虎榜
        "top_net_amount", "top_l_buy", "top_l_sell", "top_inst_net_buy",
        "is_top_list",
        # 北向资金
        "hsgt_north", "hsgt_north_ma5", "hsgt_north_ma20",
        # 大宗交易
        "block_trade_amount", "block_trade_vol", "block_trade_count", "block_trade_avg_price",
        # 北交所标记
        "is_bj_stock",
    ])
    
    # 筹码结构字段组 (dwd_chip_structure)
    chip_fields: List[str] = field(default_factory=lambda: [
        "top10_hold_ratio", "top1_hold_ratio", "top10_inst_ratio",
        "holder_num", "holder_num_chg", "holder_num_chg_pct",
        "chip_concentration", "holder_decrease",
    ])
    
    # 行业分类字段组 (dwd_stock_industry)
    industry_fields: List[str] = field(default_factory=lambda: [
        "industry", "industry_idx",
        "sw_l1_code", "sw_l1_name", "sw_l1_idx",
        "sw_l2_code", "sw_l2_name", "sw_l2_idx",
        "industry_changed",
    ])
    
    # 事件驱动字段组 (dwd_event_signal)
    event_fields: List[str] = field(default_factory=lambda: [
        # 回购
        "is_repurchase_ann", "repurchase_amount", "in_repurchase_window",
        # 解禁
        "is_unlock_day", "unlock_share", "unlock_ratio", "days_to_unlock", "in_unlock_window",
        # 质押
        "pledge_ratio", "pledge_ratio_high",
        # 分红
        "is_dividend_ann", "cash_div", "stk_div", "in_dividend_window",
        # 事件聚合
        "has_event", "has_risk_event",
    ])
    
    # 宏观环境字段组 (dwd_macro_env) - 注意：已剔除 gdp 绝对值，只保留 gdp_yoy
    macro_fields: List[str] = field(default_factory=lambda: [
        # GDP
        "gdp_yoy",
        # CPI/PPI
        "cpi_yoy", "cpi_mom",
        # PMI
        "pmi", "pmi_prod", "pmi_new_order",
        # 货币供应
        "m2", "m2_yoy",
        # 利率
        "lpr_1y", "lpr_5y",
        "shibor_on", "shibor_1w", "shibor_1m", "shibor_3m", "shibor_6m", "shibor_1y",
        # 市场风险
        "market_congestion", "stock_bond_spread",
        # 深度风险因子
        "pb_median", "pb_ew", "pb_quantile_10y", "pb_quantile_all",
        "buffett_indicator", "buffett_quantile_10y", "buffett_quantile_all",
        "break_net_ratio",
        # 指数基准
        "sh300_pct_chg", "sh300_amplitude", "sh300_turnover", "sh300_close", "sh300_vol", "sh300_amount",
        "zz500_pct_chg", "zz500_amplitude", "zz500_turnover", "zz500_close", "zz500_vol", "zz500_amount",
        "cyb_pct_chg", "cyb_amplitude", "cyb_turnover", "cyb_close", "cyb_vol", "cyb_amount",
        "sz50_pct_chg", "sz50_amplitude", "sz50_turnover", "sz50_close", "sz50_vol", "sz50_amount",
        "zz1000_pct_chg", "zz1000_amplitude", "zz1000_turnover", "zz1000_close", "zz1000_vol", "zz1000_amount",
        "kc50_pct_chg", "kc50_amplitude", "kc50_turnover", "kc50_close", "kc50_vol", "kc50_amount",
        # 回购利率
        "liquidity_gc001_close", "liquidity_gc001_weight", "liquidity_gc001_high", "liquidity_gc001_low", "liquidity_gc001_amount",
        "liquidity_r001_close", "liquidity_r001_weight", "liquidity_r001_high", "liquidity_r001_low", "liquidity_r001_amount",
        # 股指期货
        "if_total_oi", "if_close", "if_basis_rate",
        "ic_total_oi", "ic_close", "ic_basis_rate",
        "ih_total_oi", "ih_close", "ih_basis_rate",
        "im_total_oi", "im_close", "im_basis_rate",
        # 全市场两融
        "market_total_rzye", "market_total_rqye", "market_total_rzrqye",
        # PMI regime
        "pmi_regime",
        # 高阶宏观状态字段
        "lpr_trend",       # 利率趋势 (降息/加息周期)
        "money_regime",    # 货币宽紧状态
        "risk_appetite",   # 市场风险偏好
        "macro_score",     # 宏观综合评分
        "macro_regime",    # 宏观综合状态
    ])


@dataclass
class UniverseFilterConfig:
    """股票池过滤配置"""
    
    # 是否剔除停牌股（vol = 0）
    exclude_suspended: bool = True
    
    # 是否剔除 ST 股
    exclude_st: bool = True
    
    # 是否剔除次新股（上市不满 N 天）
    exclude_new: bool = True
    new_days_threshold: int = 60  # 上市不满60天为次新股
    
    # 是否剔除涨跌停股 (当日涨跌停难以买入/卖出)
    exclude_limit: bool = False  # 默认不剔除，让模型学习涨跌停信号
    
    # 北交所特殊处理
    bj_limit_pct: float = 0.30  # 北交所涨跌幅限制 30%


@dataclass
class TechnicalFeatureConfig:
    """技术指标配置"""
    
    # 移动平均线周期
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60, 120, 250])
    
    # 乖离率周期 (基于 MA)
    bias_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    
    # ROC (变化率) 周期
    roc_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    
    # RSI 周期
    rsi_periods: List[int] = field(default_factory=lambda: [6, 12, 24])
    
    # MACD 参数 (fast, slow, signal)
    macd_params: Tuple[int, int, int] = (12, 26, 9)
    
    # 波动率周期
    volatility_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    
    # 振幅周期
    amplitude_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # 换手率变化周期
    turnover_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # 量比周期
    volume_ratio_periods: List[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class DropColumnsConfig:
    """需要丢弃的字段配置"""
    
    # 估值指标：保留倒数形式 (ep/bp/sp)，丢弃原始形式 (pe_ttm/pb/ps_ttm)
    drop_raw_valuations: List[str] = field(default_factory=lambda: [
        "pe_ttm",  # 以 ep 代替
        "pb",      # 以 bp 代替
        "ps_ttm",  # 以 sp 代替
    ])
    
    # 其他无用字段
    drop_misc: List[str] = field(default_factory=lambda: [
        "report_date",  # 更新日期，已用于计算 lag_days
        "list_date",    # 上市日期，已用于计算 list_days
    ])


@dataclass
class FundamentalFeatureConfig:
    """基本面特征配置"""
    
    # 直接使用的基本面指标
    direct_features: List[str] = field(default_factory=lambda: [
        "ep", "bp", "sp",           # 估值指标（已倒数化）
        "roe", "roa",               # 盈利能力
        "gross_margin",             # 毛利率
        "debt_to_assets",           # 资产负债率
        "revenue_yoy",              # 营收增长率
        "net_profit_yoy",           # 净利润增长率
    ])
    
    # 对数化后的基本面指标
    log_features: List[str] = field(default_factory=lambda: [
        "total_mv_log",             # 市值对数
        "circ_mv_log",              # 流通市值对数
        "revenue_ttm_log",          # 营收对数
    ])


@dataclass
class LabelConfig:
    """标签配置"""
    
    # 预测目标：未来 N 日收益率
    forward_days: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # 主要标签（用于训练）
    primary_label_days: int = 5  # 默认预测未来5日收益
    
    # 标签去极值范围
    label_clip_lower: float = -0.5   # -50%
    label_clip_upper: float = 0.5    # +50%
    
    # 停牌处理：未来 N 日内停牌超过 M 天则标记为无效
    max_suspended_days: int = 2      # 未来5日内最多允许2天停牌
    
    # 是否生成分类标签
    generate_class_labels: bool = True
    # 分类阈值（涨/跌/平）
    class_threshold: float = 0.02    # ±2% 为平盘


@dataclass
class NormalizationConfig:
    """标准化配置"""
    
    # 是否进行截面标准化 (每日 Z-Score)
    cross_sectional_zscore: bool = True
    
    # Z-Score 去极值范围
    zscore_clip: float = 3.0  # clip to [-3, 3]
    
    # 需要做截面标准化的字段
    zscore_fields: List[str] = field(default_factory=lambda: [
        # 技术指标
        "bias_5", "bias_10", "bias_20", "bias_60",
        "roc_5", "roc_10", "roc_20", "roc_60",
        "rsi_6", "rsi_12", "rsi_24",
        "macd", "macd_signal", "macd_hist",
        "volatility_5", "volatility_10", "volatility_20", "volatility_60",
        "amplitude_5", "amplitude_10", "amplitude_20",
        "volume_ratio_5", "volume_ratio_10", "volume_ratio_20",
        # 基本面
        "ep", "bp", "sp",
        "roe", "roa", "gross_margin",
        "debt_to_assets", "revenue_yoy", "net_profit_yoy",
        "total_mv_log", "circ_mv_log", "revenue_ttm_log",
    ])


@dataclass
class PipelineConfig:
    """流水线总配置"""
    
    # GPU 加速
    use_gpu: bool = True
    
    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseFilterConfig = field(default_factory=UniverseFilterConfig)
    technical: TechnicalFeatureConfig = field(default_factory=TechnicalFeatureConfig)
    fundamental: FundamentalFeatureConfig = field(default_factory=FundamentalFeatureConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    drop_columns: DropColumnsConfig = field(default_factory=DropColumnsConfig)
    feature_groups: FeatureGroups = field(default_factory=FeatureGroups)  # 新增：字段组配置
    
    # 日志级别
    verbose: bool = True
    
    @classmethod
    def default(cls) -> "PipelineConfig":
        """创建默认配置"""
        return cls()
    
    def __post_init__(self):
        """验证配置"""
        # 确保 MA250 在预热期内有足够数据
        warmup_days = 250  # 至少需要250个交易日
        # 这里可以添加更多验证逻辑
