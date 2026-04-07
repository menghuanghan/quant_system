"""
非结构化 DWD 构建模块

将 data/processed/unstructured 下的处理后结果，
对齐到 data/processed/structured/dwd/dwd_stock_price.parquet 骨架，
产出以 (trade_date, ts_code) 为主键的 dwd_unstructured.parquet。
"""

from .builder import (
    UnstructuredDWDConfig,
    UnstructuredDWDBuilder,
    build_dwd_unstructured,
)
from .increment_duckdb_merger import DuckDBIncrementProcessedProvider

__all__ = [
    "UnstructuredDWDConfig",
    "UnstructuredDWDBuilder",
    "build_dwd_unstructured",
    "DuckDBIncrementProcessedProvider",
]
