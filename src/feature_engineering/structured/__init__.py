"""
结构化特征工程模块

提供完整的特征工程流水线，包括：
- 数据合并与股票池过滤
- 特征预处理
- 技术指标计算
- 标签生成
- 后处理与标准化
"""

from .config import (
    PipelineConfig,
    DataConfig,
    UniverseFilterConfig,
    TechnicalFeatureConfig,
    FundamentalFeatureConfig,
    LabelConfig,
    NormalizationConfig,
)

from .pipeline import FeaturePipeline

__all__ = [
    # 配置
    "PipelineConfig",
    "DataConfig",
    "UniverseFilterConfig",
    "TechnicalFeatureConfig",
    "FundamentalFeatureConfig",
    "LabelConfig",
    "NormalizationConfig",
    # 流水线
    "FeaturePipeline",
]
