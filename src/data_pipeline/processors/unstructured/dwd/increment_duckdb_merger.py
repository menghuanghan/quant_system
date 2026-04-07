"""
非结构化 DWD 增量输入合并（DuckDB）

职责：
1. 基于 latest_date 收集 data/processed/unstructured_increment/date=YYYY-MM-DD 下全部 date<=latest_date 分区
2. 将 full processed 与 increment processed 按同构 schema 合并
3. 仅去除“完全重复行”（全列去重）
4. 按需返回 pandas DataFrame，供 UnstructuredDWDBuilder 复用

说明：
- 不负责写 DWD 输出文件。
- 不改变 Builder 业务计算逻辑，仅提供输入数据覆写。
"""

from __future__ import annotations

import importlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)


def _import_duckdb():
    """运行时导入 duckdb，避免全量模式下硬依赖。"""
    try:
        return importlib.import_module("duckdb")
    except Exception as e:
        raise ImportError("未安装 duckdb，请先安装 requirements.txt 中的 duckdb 依赖") from e


def _normalize_iso_date(date_str: str) -> str:
    """归一化日期为 YYYY-MM-DD。"""
    s = str(date_str).strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
    return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")


class DuckDBIncrementProcessedProvider:
    """
    非结构化 DWD 输入覆写提供器（DuckDB 合并 full + increment）。

    使用方式：
        provider = DuckDBIncrementProcessedProvider(
            full_processed_root="data/processed/unstructured",
            increment_processed_root="data/processed/unstructured_increment",
            latest_date="2026-01-07",
        )
        df = provider.get("data/processed/unstructured/announcements.parquet")
    """

    def __init__(
        self,
        full_processed_root: Union[str, Path] = "data/processed/unstructured",
        increment_processed_root: Union[str, Path] = "data/processed/unstructured_increment",
        latest_date: str = "",
        cache_enabled: bool = True,
    ):
        if not latest_date:
            raise ValueError("latest_date 不能为空")

        self.full_processed_root = Path(full_processed_root).resolve()
        self.increment_processed_root = Path(increment_processed_root).resolve()
        self.latest_date = _normalize_iso_date(latest_date)
        self.latest_date_obj = datetime.strptime(self.latest_date, "%Y-%m-%d").date()

        self.cache_enabled = cache_enabled
        self._cache: Dict[str, pd.DataFrame] = {}

        if not self.full_processed_root.exists():
            raise FileNotFoundError(f"全量处理目录不存在: {self.full_processed_root}")
        if not self.increment_processed_root.exists():
            logger.warning("增量处理目录不存在，将不会启用增量覆写: %s", self.increment_processed_root)

        self.partition_dirs = self._discover_increment_partitions()
        self.partition_dates = [p.name.split("=", 1)[1] for p in self.partition_dirs]

        logger.info(
            "DuckDB非结构化增量输入提供器初始化完成: latest_date=%s, 分区数=%s",
            self.latest_date,
            len(self.partition_dirs),
        )

    def clear_cache(self) -> None:
        """清理已缓存的合并结果。"""
        self._cache.clear()

    def get(self, source_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        按 source_path 返回合并后的覆写 DataFrame。

        规则：
        - 若 source_path 在 date<=latest_date 的增量分区里没有任何文件，则返回 None（保持原始 full 读取路径）
        - 若存在增量文件，则执行 full + increment 合并，并做全列去重
        """
        source_path = Path(source_path)

        if source_path.is_absolute():
            try:
                relative_path = source_path.resolve().relative_to(self.full_processed_root)
            except ValueError:
                return None
        else:
            relative_path = source_path

        cache_key = relative_path.as_posix()
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key].copy(deep=True)

        full_files = self._collect_full_files(relative_path)
        increment_files = self._collect_increment_files(relative_path)

        # 无增量变更，不覆写
        if not increment_files:
            return None

        merged_pdf = self._merge_with_duckdb(full_files=full_files, increment_files=increment_files)
        if merged_pdf is None:
            return None

        if self.cache_enabled:
            self._cache[cache_key] = merged_pdf
            return merged_pdf.copy(deep=True)

        return merged_pdf

    # -------------------- internals --------------------

    def _discover_increment_partitions(self) -> List[Path]:
        """发现全部 date<=latest_date 的增量分区目录。"""
        if not self.increment_processed_root.exists():
            return []

        dirs: List[Path] = []
        for p in self.increment_processed_root.iterdir():
            if not p.is_dir() or not p.name.startswith("date="):
                continue
            date_part = p.name.split("=", 1)[1]
            try:
                dt = datetime.strptime(date_part, "%Y-%m-%d").date()
            except ValueError:
                logger.warning("跳过非法增量分区目录: %s", p)
                continue
            if dt <= self.latest_date_obj:
                dirs.append(p)

        dirs.sort(key=lambda x: x.name)
        return dirs

    @staticmethod
    def _uniq_sorted_paths(paths: Sequence[Path]) -> List[Path]:
        seen = set()
        out = []
        for p in paths:
            sp = str(p.resolve())
            if sp in seen:
                continue
            seen.add(sp)
            out.append(Path(sp))
        out.sort(key=lambda x: str(x))
        return out

    def _collect_parquet_candidates(self, candidate: Path) -> List[Path]:
        """从候选路径收集 parquet 文件，兼容文件/目录两种形态。"""
        files: List[Path] = []

        if candidate.is_file() and candidate.suffix == ".parquet":
            files.append(candidate)

        if candidate.is_dir():
            files.extend(sorted(candidate.rglob("*.parquet")))

        # 给的是 xxx.parquet，实际落盘可能是同名目录
        if candidate.suffix == ".parquet":
            alt_dir = candidate.with_suffix("")
            if alt_dir.is_dir():
                files.extend(sorted(alt_dir.rglob("*.parquet")))

        return self._uniq_sorted_paths(files)

    def _collect_full_files(self, relative_path: Path) -> List[Path]:
        candidate = self.full_processed_root / relative_path
        return self._collect_parquet_candidates(candidate)

    def _collect_increment_files(self, relative_path: Path) -> List[Path]:
        files: List[Path] = []
        for partition in self.partition_dirs:
            candidate = partition / relative_path
            files.extend(self._collect_parquet_candidates(candidate))
        return self._uniq_sorted_paths(files)

    @staticmethod
    def _sql_parquet_list(files: Sequence[Path]) -> str:
        escaped = [str(f).replace("'", "''") for f in files]
        return "[" + ", ".join(f"'{p}'" for p in escaped) + "]"

    def _merge_with_duckdb(
        self,
        full_files: Sequence[Path],
        increment_files: Sequence[Path],
    ) -> Optional[pd.DataFrame]:
        """使用 DuckDB 执行 full + increment 合并并全列去重。"""
        if not full_files and not increment_files:
            return None

        duckdb_module = _import_duckdb()
        with duckdb_module.connect(database=":memory:") as con:
            con.execute("PRAGMA threads=4")

            all_files = self._uniq_sorted_paths([*full_files, *increment_files])
            all_sql = f"SELECT * FROM read_parquet({self._sql_parquet_list(all_files)}, union_by_name=true)"
            merged_pdf = con.execute(f"SELECT DISTINCT * FROM ({all_sql}) t").fetch_df()

            # 按 full schema 对齐列顺序，保留新增列在尾部
            if full_files:
                full_sql = f"SELECT * FROM read_parquet({self._sql_parquet_list(full_files)}, union_by_name=true)"
                full_cols = con.execute(f"SELECT * FROM ({full_sql}) t LIMIT 0").fetch_df().columns.tolist()
                for col in full_cols:
                    if col not in merged_pdf.columns:
                        merged_pdf[col] = pd.NA
                extra_cols = [c for c in merged_pdf.columns if c not in full_cols]
                merged_pdf = merged_pdf[full_cols + extra_cols]

            return merged_pdf
