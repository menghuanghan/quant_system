"""
结构化原始数据合并模块

将 data/raw/structured 下按 ts_code 拆分存储的目录合并为单个 parquet 文件。
"""

from .merger import RawDataMerger

__all__ = ['RawDataMerger']
