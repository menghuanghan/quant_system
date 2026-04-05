"""
Parquet 逐列流式读取工具。
"""

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class ParquetColumnStreamReader:
    """按列按 row-group 读取 parquet，避免一次性宽表入内存。"""

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.parquet_file = pq.ParquetFile(str(self.data_path))
        self.num_row_groups = self.parquet_file.num_row_groups
        self.num_rows = self.parquet_file.metadata.num_rows

    @property
    def columns(self) -> List[str]:
        return self.parquet_file.schema_arrow.names

    def read_columns_as_pandas(self, columns: List[str]) -> pd.DataFrame:
        table = pq.read_table(str(self.data_path), columns=columns)
        return table.to_pandas()

    def iter_column_chunks(
        self,
        column: str,
        dtype: Optional[np.dtype] = None,
    ) -> Iterable[np.ndarray]:
        for rg_idx in range(self.num_row_groups):
            table = self.parquet_file.read_row_group(rg_idx, columns=[column])
            chunk = table[column].to_numpy(zero_copy_only=False)
            if dtype is not None:
                chunk = chunk.astype(dtype, copy=False)
            yield chunk

    def read_column_as_numpy(
        self,
        column: str,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        chunks = list(self.iter_column_chunks(column=column, dtype=dtype))
        if not chunks:
            out_dtype = dtype if dtype is not None else np.float32
            return np.array([], dtype=out_dtype)
        return np.concatenate(chunks, axis=0)
