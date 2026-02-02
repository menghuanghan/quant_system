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
