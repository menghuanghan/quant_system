"""
后处理模块配置

定义 LightGBM 和 GRU 各自的处理参数。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[4]


@dataclass
class CommonCleanConfig:
    """公共清洗配置"""
    
    # 主要回归标签（用于删除 NaN 行）
    primary_label: str = "ret_5d"
    
    # 需要检查 std 是否为 0 的列类型
    check_constant_dtypes: List[str] = field(default_factory=lambda: [
        "float32", "float64", "Float32", "Float64", "int32", "int64", "Int32", "Int64"
    ])
    
    # ============ 类别编码配置 ============
    # 需要删除的字符串列（有对应的 *_idx 索引列）
    drop_string_cols: List[str] = field(default_factory=lambda: [
        # 行业分类字符串列（使用对应的 *_idx）
        "industry",       # 使用 industry_idx
        "sw_l1_code",     # 使用 sw_l1_idx
        "sw_l1_name",     # 使用 sw_l1_idx
        "sw_l2_code",     # 使用 sw_l2_idx
        "sw_l2_name",     # 使用 sw_l2_idx
        "sw_l3_code",     # 三级行业过细，直接删除
        "sw_l3_name",     # 三级行业过细，直接删除
        # 日期字符串列（模型训练不需要）
        "list_date",
        "chip_report_date",
        "holder_report_date",
        # datetime 列（模型训练不直接使用）
        "report_date",
    ])
    
    # 需要手动编码的类别列（无对应索引）
    # 格式: {列名: {值: 编码}}
    manual_encode_mapping: dict = field(default_factory=lambda: {
        "market": {
            "主板": 0,
            "创业板": 1,
            "科创板": 2,
            "北交所": 3,
        },
    })


@dataclass
class LGBConfig:
    """LightGBM 专用处理配置"""
    
    # 输出文件名
    output_file: str = "train_lgb.parquet"
    
    # 数据切分：剔除 [2019-01-01, 2020-12-31] 的数据
    cut_start: str = "2019-01-01"
    cut_end: str = "2020-12-31"
    
    # NaN 填充策略
    # fill_value: -1 表示特殊填充值，None 表示保留 NaN
    fill_nan_value: Optional[float] = None  # LightGBM 原生支持 NaN
    
    # 类别特征列（需要转换为 int 或 category）
    category_columns: List[str] = field(default_factory=lambda: [
        "industry_idx", "sw_l1_idx", "sw_l2_idx",
    ])
    
    # 排序字段
    sort_by: List[str] = field(default_factory=lambda: ["trade_date", "ts_code"])


@dataclass
class GRUConfig:
    """GRU 专用处理配置"""
    
    # 输出文件名
    output_file: str = "train_gru.parquet"
    
    # 数据切分：剔除 [2019-01-01, 2020-06-30] 的数据
    cut_start: str = "2019-01-01"
    cut_end: str = "2020-06-30"
    
    # [防泄露] Clip 分位数仅使用训练集计算
    # 设为 None 则使用全量数据（有轻微泄露风险）
    clip_train_end: Optional[str] = "2023-12-31"
    
    # ============ 第一类：需要 DROP 的非平稳列 ============
    # GRU 应该学习"涨跌幅"和"波动率"，而不是"股价是 10 元还是 100 元"
    drop_cols: List[str] = field(default_factory=lambda: [
        # 原始价格列（非平稳，会随大盘点位漂移）
        "open", "high", "low", "close", "pre_close", "vwap", "adj_factor",
        "open_hfq", "high_hfq", "low_hfq", "close_hfq", "vwap_hfq",
        # 均线列（本质还是价格）
        "ma_5", "ma_10", "ma_20", "ma_60", "ma_120", "ma_250",
        # 指数/期货原始价格（保留 _pct_chg / _return 即可）
        "sh300_close", "zz500_close", "cyb_close", "sz50_close", "kc50_close", "zz1000_close",
        "if_close", "ic_close", "ih_close", "im_close",
        # 冗余/中间列
        "lag_days",
    ])
    
    # ============ 第二类：Log1p 变换的个股级特征 ============
    # 高偏度、长尾分布，需要 Log1p 后再做截面 Z-Score
    log1p_features: List[str] = field(default_factory=lambda: [
        # 成交量/金额类
        "vol", "amount",
        "buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount",
        "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount",
        "buy_sm_vol", "sell_sm_vol", "buy_md_vol", "sell_md_vol",
        "buy_lg_vol", "sell_lg_vol", "buy_elg_vol", "sell_elg_vol",
        # 净资金流
        "net_mf_amount", "net_mf_vol", "net_main_amount", "net_retail_amount",
        "net_sm_amount", "net_md_amount", "net_lg_amount", "net_elg_amount",
        "buy_main_amount", "sell_main_amount", "buy_retail_amount", "sell_retail_amount",
        # 市值类
        "total_mv", "circ_mv",
        # 两融（个股级）
        "rzye", "rqye", "rzmre", "rzche", "rqyl", "rqchl", "rqmcl", "rzrqye",
        # 龙虎榜
        "top_amount", "top_l_buy", "top_l_sell", "top_inst_buy", "top_inst_sell",
        "top_net_amount", "top_inst_net_buy",
        # 大宗交易
        "block_trade_amount", "block_trade_vol",
        # 股东人数/持股
        "holder_num", "top10_hold_amount",
        # 财务数据 (绝对值)
        "revenue_ttm", "operate_profit_ttm", "total_profit_ttm", "n_income_attr_p_ttm",
        "revenue_sq", "n_income_attr_p_sq",
        "total_assets", "total_liab", "total_equity",
        # 分红/回购/解禁
        "repurchase_amount", "unlock_share", "cash_div",
    ])
    
    # ============ 第二类：市场级数据（需要滚动 Z-Score）============
    # 同一天截面上是常数，不能做截面标准化，需要滚动窗口标准化
    rolling_zscore_features: List[str] = field(default_factory=lambda: [
        # 北向/南向资金
        "hsgt_north", "hsgt_south", "hsgt_hgt", "hsgt_sgt",
        "hsgt_north_ma5", "hsgt_north_ma20", "hsgt_ggt_ss", "hsgt_ggt_sz",
        # 流动性指标
        "liquidity_gc001_amount", "liquidity_r001_amount",
        "liquidity_gc001_close", "liquidity_gc001_high", "liquidity_gc001_low", "liquidity_gc001_weight",
        "liquidity_r001_close", "liquidity_r001_high", "liquidity_r001_low", "liquidity_r001_weight",
        # 指数成交量/成交额
        "sh300_amount", "zz500_amount", "cyb_amount", "sz50_amount", "kc50_amount", "zz1000_amount",
        "sh300_vol", "zz500_vol", "cyb_vol", "sz50_vol", "kc50_vol", "zz1000_vol",
        # 指数涨跌幅/换手/振幅（市场级）
        "sh300_pct_chg", "zz500_pct_chg", "cyb_pct_chg", "sz50_pct_chg", "kc50_pct_chg", "zz1000_pct_chg",
        "sh300_turnover", "zz500_turnover", "cyb_turnover", "sz50_turnover", "kc50_turnover", "zz1000_turnover",
        "sh300_amplitude", "zz500_amplitude", "cyb_amplitude", "sz50_amplitude", "kc50_amplitude", "zz1000_amplitude",
        # 期货基差/持仓
        "if_basis_rate", "ic_basis_rate", "ih_basis_rate", "im_basis_rate",
        "if_total_oi", "ic_total_oi", "ih_total_oi", "im_total_oi",
        # 市场融资融券
        "market_total_rzye", "market_total_rqye", "market_total_rzrqye", "rzye_chg",
        # 宏观指标
        "gdp_yoy", "cpi_yoy", "cpi_mom", "ppi_yoy", "m2", "m2_yoy",
        "shibor_on", "shibor_1w", "shibor_1m", "shibor_3m", "shibor_6m", "shibor_1y",
        "lpr_1y", "lpr_5y", "lpr_trend",
        "pmi", "pmi_prod", "pmi_new_order", "pmi_regime",
        "stock_bond_spread",
        # 宏观合成指标
        "macro_amount_shibor", "macro_bp_sbs", "macro_ep_sbs", "macro_growth_ex_ppi",
        "macro_real_growth", "macro_regime", "macro_score", "macro_vol_m2",
        "money_regime", "risk_appetite",
        "buffett_indicator", "buffett_quantile_10y", "buffett_quantile_all",
        # 市场估值
        "pb_ew", "pb_median", "pb_quantile_10y", "pb_quantile_all",
        # 市场拥挤度
        "market_congestion",
    ])
    
    # 滚动 Z-Score 窗口（约一年）
    rolling_window: int = 250
    
    # 时序填充的特征（先 ffill）
    # 注意：ma_* 列已被 drop，不再需要填充
    ffill_features: List[str] = field(default_factory=lambda: [
        # 宏观指标（低频更新）
        "gdp_yoy", "cpi_yoy", "cpi_mom", "pmi", "pmi_prod", "pmi_new_order",
        "m2", "m2_yoy", "lpr_1y", "lpr_5y",
        "shibor_on", "shibor_1w", "shibor_1m", "shibor_3m", "shibor_6m", "shibor_1y",
        "ppi_yoy", "stock_bond_spread",
        # 基本面（季度更新，用上一期填充）
        "roe", "roa", "gross_margin", "netprofit_margin", "debt_to_assets",
        "revenue_yoy", "net_profit_yoy",
        # 筹码结构
        "top10_hold_ratio", "top1_hold_ratio", "holder_num",
        "pledge_ratio", "pledge_ratio_high",
    ])
    
    # Clip 上下界 (用于去极值)
    clip_lower_percentile: float = 0.01  # 1%
    clip_upper_percentile: float = 0.99  # 99%
    
    # ============ 第三类：截面 Z-Score 的个股级特征 ============
    # 这些列数值范围各异（百分比、比率、计数），需要统一到 N(0,1) 分布
    zscore_features: List[str] = field(default_factory=lambda: [
        # 价格收益率
        "return_1d", "ret_1d",
        # 换手率
        "turnover",
        # 技术指标 - 偏离度
        "bias_5", "bias_10", "bias_20", "bias_60",
        # 技术指标 - 动量
        "roc_5", "roc_10", "roc_20", "roc_60",
        # 技术指标 - RSI
        "rsi_6", "rsi_12", "rsi_24",
        # 技术指标 - MACD
        "macd", "macd_signal", "macd_hist",
        # 技术指标 - 波动率
        "volatility_5", "volatility_10", "volatility_20", "volatility_60",
        "amplitude", "amplitude_5", "amplitude_10", "amplitude_20",
        # 技术指标 - 量比
        "volume_ratio_5", "volume_ratio_10", "volume_ratio_20",
        # 技术指标 - 夏普
        "sharpe_5d", "sharpe_10d", "sharpe_20d",
        # 技术指标 - 超额收益
        "excess_ret_5d", "excess_ret_10d",
        "rank_ret_5d", "rank_ret_10d",
        # 相对强弱
        "rs_hs300", "rs_csi500",
        # 财务比率 - 估值
        "ep", "bp", "sp", "ep_growth",
        # 财务比率 - 盈利能力
        "roe", "roa", "gross_margin", "netprofit_margin",
        # 财务比率 - 杠杆/风险
        "debt_to_assets",
        # 财务比率 - 成长性
        "revenue_yoy", "net_profit_yoy",
        # 筹码/事件特征
        "pledge_ratio", "pledge_ratio_high", "unlock_ratio", "days_to_unlock",
        "top10_hold_ratio", "top1_hold_ratio", "holder_num_chg_pct", "holder_num_chg",
        "chip_concentration", "chip_top10_ratio", "chip_top1_dominance",
        "chip_holder_chg", "chip_stability_score", "chip_ln_holder_num",
        # 资金流强度（已是比率）
        "mf_main_intensity", "mf_retail_intensity", "mf_lg_intensity", "mf_elg_intensity",
        "mf_md_intensity", "mf_block_intensity", "mf_north_net",
        "mf_main_sign", "mf_retail_buy_ratio", "mf_retail_sell_ratio",
        "net_main_amount_pct",
        # 大宗交易（非绝对值）
        "block_trade_count", "block_trade_avg_price",
        # 涨跌停
        "limit_ratio", "break_net_ratio",
        # 权重
        "weight_hs300", "weight_csi500", "weight_csi1000",
        # 股息/分红
        "stk_div",
        # ============ Log1p 变换后也需要截面 Z-Score 的特征 ============
        # （让不同股票的成交量、市值等在同一天可比）
        # 成交量/金额类
        "vol", "amount", "log_vol", "log_amount",
        "buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount",
        "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount",
        "buy_sm_vol", "sell_sm_vol", "buy_md_vol", "sell_md_vol",
        "buy_lg_vol", "sell_lg_vol", "buy_elg_vol", "sell_elg_vol",
        # 净资金流
        "net_mf_amount", "net_mf_vol", "net_main_amount", "net_retail_amount",
        "net_sm_amount", "net_md_amount", "net_lg_amount", "net_elg_amount",
        "buy_main_amount", "sell_main_amount", "buy_retail_amount", "sell_retail_amount",
        # 市值类
        "total_mv", "circ_mv", "log_total_mv", "log_circ_mv",
        # 两融（个股级）
        "rzye", "rqye", "rzmre", "rzche", "rqyl", "rqchl", "rqmcl", "rzrqye",
        # 龙虎榜
        "top_amount", "top_l_buy", "top_l_sell", "top_inst_buy", "top_inst_sell",
        "top_net_amount", "top_inst_net_buy",
        # 大宗交易绝对值
        "block_trade_amount", "block_trade_vol",
        # 股东人数/持股
        "holder_num", "top10_hold_amount", "holder_decrease",
        # 财务数据绝对值
        "revenue_ttm", "operate_profit_ttm", "total_profit_ttm", "n_income_attr_p_ttm",
        "revenue_sq", "n_income_attr_p_sq",
        "total_assets", "total_liab", "total_equity",
        # 分红/回购/解禁
        "repurchase_amount", "unlock_share", "cash_div",
    ])
    
    # Z-Score clip 范围
    zscore_clip: float = 3.0
    
    # 排序字段（用于时序连续性）
    sort_by: List[str] = field(default_factory=lambda: ["ts_code", "trade_date"])


@dataclass
class PostprocessConfig:
    """后处理总配置"""
    
    # 输出目录
    output_dir: Path = BASE_DIR / "data" / "features" / "structured"
    
    # 子配置
    common: CommonCleanConfig = field(default_factory=CommonCleanConfig)
    lgb: LGBConfig = field(default_factory=LGBConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    
    @classmethod
    def default(cls) -> "PostprocessConfig":
        """创建默认配置"""
        return cls()
