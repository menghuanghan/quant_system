"""
结构化DWD增量输入合并（DuckDB）

职责：
1. 基于 latest_date 收集 data/raw/structured_increment/date=YYYY-MM-DD 下全部 date<=latest_date 分区
2. 将 full raw 与 increment raw 按同构 schema 合并
3. 仅去除“完全重复行”（全列去重）
4. 按需（lazy）返回 cuDF DataFrame，供 DWD Processor 复用

说明：
- 不负责写 DWD 输出文件。
- 不改变 Processor 业务计算逻辑，仅提供输入数据覆写。
"""

from __future__ import annotations

import logging
import importlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cudf
import pandas as pd

logger = logging.getLogger(__name__)


def _import_duckdb():
    """运行时导入 duckdb，避免环境未安装时在导入阶段直接失败。"""
    try:
        return importlib.import_module("duckdb")
    except Exception as e:
        raise ImportError(
            "未安装 duckdb，请先安装 requirements.txt 中的 duckdb 依赖"
        ) from e


def _normalize_iso_date(date_str: str) -> str:
    """归一化日期为 YYYY-MM-DD。"""
    s = str(date_str).strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
    return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")


class DuckDBIncrementInputProvider:
    """
    DWD 输入覆写提供器（DuckDB 合并 full + increment）。

    使用方式：
        provider = DuckDBIncrementInputProvider(
            full_raw_root="data/raw/structured",
            increment_raw_root="data/raw/structured_increment",
            latest_date="2026-01-05",
        )
        df = provider.get("data/raw/structured/market_data/stock_daily.parquet")
    """

    def __init__(
        self,
        full_raw_root: Union[str, Path] = "data/raw/structured",
        increment_raw_root: Union[str, Path] = "data/raw/structured_increment",
        latest_date: str = "",
        cache_enabled: bool = True,
    ):
        if not latest_date:
            raise ValueError("latest_date 不能为空")

        self.full_raw_root = Path(full_raw_root).resolve()
        self.increment_raw_root = Path(increment_raw_root).resolve()
        self.latest_date = _normalize_iso_date(latest_date)
        self.latest_date_obj = datetime.strptime(self.latest_date, "%Y-%m-%d").date()

        self.cache_enabled = cache_enabled
        self._cache: Dict[str, cudf.DataFrame] = {}

        if not self.full_raw_root.exists():
            raise FileNotFoundError(f"全量原始目录不存在: {self.full_raw_root}")
        if not self.increment_raw_root.exists():
            logger.warning("增量原始目录不存在，将不会启用增量覆写: %s", self.increment_raw_root)

        self.partition_dirs = self._discover_increment_partitions()
        self.partition_dates = [p.name.split("=", 1)[1] for p in self.partition_dirs]

        logger.info(
            "DuckDB增量输入提供器初始化完成: latest_date=%s, 分区数=%s",
            self.latest_date,
            len(self.partition_dirs),
        )

    def clear_cache(self) -> None:
        """清理已缓存的合并结果。"""
        self._cache.clear()

    def get(self, source_path: Union[str, Path], expect_dir: bool = False) -> Optional[cudf.DataFrame]:
        """
        按 source_path 返回合并后的覆写 DataFrame（cuDF）。

        规则：
        - 若该 source_path 在 date<=latest_date 的增量分区里没有任何文件，则返回 None（保持原始 full 读取路径）
        - 若存在增量文件，则执行 full + increment 合并，并做全列去重
        """
        source_path = Path(source_path).resolve()

        try:
            relative_path = source_path.relative_to(self.full_raw_root)
        except ValueError:
            # 非 full_raw_root 下路径，不覆写
            return None

        cache_key = f"{relative_path.as_posix()}|dir={int(expect_dir)}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key].copy(deep=True)

        full_files = self._collect_full_files(source_path=source_path, expect_dir=expect_dir)
        increment_files = self._collect_increment_files(relative_path=relative_path, expect_dir=expect_dir)

        # 无增量变更，不覆写
        if not increment_files:
            return None

        if not full_files and not increment_files:
            return None

        merged_pdf = self._merge_with_duckdb(full_files=full_files, increment_files=increment_files)
        if merged_pdf is None:
            return None

        # 转 cuDF
        merged_cudf = cudf.from_pandas(merged_pdf)

        if self.cache_enabled:
            self._cache[cache_key] = merged_cudf
            return merged_cudf.copy(deep=True)

        return merged_cudf

    # -------------------- internals --------------------

    def _discover_increment_partitions(self) -> List[Path]:
        """发现全部 date<=latest_date 的增量分区目录。"""
        if not self.increment_raw_root.exists():
            return []

        dirs: List[Path] = []
        for p in self.increment_raw_root.iterdir():
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

    def _collect_parquet_candidates(self, candidate: Path, expect_dir: bool) -> List[Path]:
        """
        从候选路径收集 parquet 文件，兼容目录/文件两种形态与回退形态。
        """
        files: List[Path] = []

        if candidate.is_file() and candidate.suffix == ".parquet":
            files.append(candidate)

        if candidate.is_dir():
            files.extend(sorted(candidate.rglob("*.parquet")))

        # 目录模式下，兼容回退到同名 .parquet 文件
        if expect_dir:
            fallback_file = candidate.with_suffix(".parquet")
            if fallback_file.is_file():
                files.append(fallback_file)
        else:
            # 文件模式下，兼容“给的是 .parquet 但实际是同名目录”
            if candidate.suffix == ".parquet":
                alt_dir = candidate.with_suffix("")
                if alt_dir.is_dir():
                    files.extend(sorted(alt_dir.rglob("*.parquet")))
            else:
                fallback_file = candidate.with_suffix(".parquet")
                if fallback_file.is_file():
                    files.append(fallback_file)

        return self._uniq_sorted_paths(files)

    def _collect_full_files(self, source_path: Path, expect_dir: bool) -> List[Path]:
        """收集 full raw 侧文件集合。"""
        return self._collect_parquet_candidates(source_path, expect_dir=expect_dir)

    def _collect_increment_files(self, relative_path: Path, expect_dir: bool) -> List[Path]:
        """收集 increment raw（date<=latest_date）文件集合。"""
        files: List[Path] = []
        for partition in self.partition_dirs:
            candidate = partition / relative_path
            files.extend(self._collect_parquet_candidates(candidate, expect_dir=expect_dir))
        return self._uniq_sorted_paths(files)

    @staticmethod
    def _sql_parquet_list(files: Sequence[Path]) -> str:
        """将文件列表转换为 DuckDB read_parquet 的 SQL 字面量数组。"""
        escaped = [str(f).replace("'", "''") for f in files]
        return "[" + ", ".join(f"'{p}'" for p in escaped) + "]"

    def _merge_with_duckdb(self, full_files: Sequence[Path], increment_files: Sequence[Path]) -> Optional[pd.DataFrame]:
        """
        使用 DuckDB 执行 full + increment 合并并去重。

        去重策略：仅去除“完全重复行”（全列 DISTINCT）。
        """
        if not full_files and not increment_files:
            return None

        duckdb_module = _import_duckdb()
        with duckdb_module.connect(database=":memory:") as con:
            con.execute("PRAGMA threads=4")

            def rel_sql(files: Sequence[Path]) -> str:
                return f"SELECT * FROM read_parquet({self._sql_parquet_list(files)}, union_by_name=true)"

            all_files = self._uniq_sorted_paths([*full_files, *increment_files])
            all_sql = rel_sql(all_files)

            # 直接基于全文件集合按列名并集读取，避免 full/inc 分开 UNION 时列数不一致导致失败
            merged_sql = f"SELECT DISTINCT * FROM ({all_sql}) t"
            merged_pdf = con.execute(merged_sql).fetch_df()

            # 按 full schema 对齐列顺序（保证与既有 full 输入结构一致）
            if full_files:
                full_sql = rel_sql(full_files)
                full_cols = con.execute(f"SELECT * FROM ({full_sql}) t LIMIT 0").fetch_df().columns.tolist()
                for col in full_cols:
                    if col not in merged_pdf.columns:
                        merged_pdf[col] = pd.NA
                extra_cols = [c for c in merged_pdf.columns if c not in full_cols]
                merged_pdf = merged_pdf[full_cols + extra_cols]

            return merged_pdf
