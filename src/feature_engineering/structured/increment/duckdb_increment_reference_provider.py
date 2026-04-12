"""
结构化特征工程 - 参考数据增量提供器（DuckDB）

覆盖三类参考数据：
- index_daily
- index_weight
- etf_daily

规则：
1. 增量侧：仅使用 date<=latest_date 的分区
2. 全量侧：补齐到目标窗口（默认 300 交易日）
3. 合并去重：增量优先
4. 返回 pandas.DataFrame，供 ReferenceDataLoader 注入
"""

from __future__ import annotations

import importlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .duckdb_increment_data_provider import normalize_iso_date

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[4]


def _import_duckdb():
    try:
        return importlib.import_module("duckdb")
    except Exception as e:
        raise ImportError("未安装 duckdb，请先安装 requirements.txt 中 duckdb 依赖") from e


class DuckDBIncrementReferenceProvider:
    """参考数据增量提供器。"""

    def __init__(
        self,
        latest_date: str,
        lookback_trade_days: int = 300,
        date_column: str = "trade_date",
        full_raw_root: Union[str, Path] = BASE_DIR / "data" / "raw" / "structured",
        increment_raw_root: Union[str, Path] = BASE_DIR / "data" / "raw" / "structured_increment",
        cache_enabled: bool = True,
    ):
        if not latest_date:
            raise ValueError("latest_date 不能为空")

        self.latest_date = normalize_iso_date(latest_date)
        self.latest_date_obj = datetime.strptime(self.latest_date, "%Y-%m-%d").date()
        self.lookback_trade_days = int(lookback_trade_days)
        self.date_column = date_column
        self.cache_enabled = cache_enabled

        self.full_raw_root = Path(full_raw_root).resolve()
        self.increment_raw_root = Path(increment_raw_root).resolve()

        self._cache: Dict[str, pd.DataFrame] = {}
        self._dataset_specs: Dict[str, Dict[str, Path]] = {
            "index_daily": {
                "full_dir": self.full_raw_root / "market_data" / "index_daily",
                "increment_relative": Path("market_data") / "index_daily",
            },
            "index_weight": {
                "full_dir": self.full_raw_root / "index_benchmark" / "index_weight",
                "increment_relative": Path("index_benchmark") / "index_weight",
            },
            "etf_daily": {
                "full_dir": self.full_raw_root / "market_data" / "etf_daily",
                "increment_relative": Path("market_data") / "etf_daily",
            },
        }

        self._partitions = self._discover_partitions()

        logger.info(
            "DuckDBIncrementReferenceProvider 初始化完成: latest_date=%s, lookback_trade_days=%s, partitions=%s",
            self.latest_date,
            self.lookback_trade_days,
            len(self._partitions),
        )

    def clear_cache(self) -> None:
        self._cache.clear()

    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """加载并合并指定参考数据集。"""
        if dataset_name not in self._dataset_specs:
            raise KeyError(f"未知参考数据集: {dataset_name}")

        if self.cache_enabled and dataset_name in self._cache:
            return self._cache[dataset_name].copy(deep=True)

        spec = self._dataset_specs[dataset_name]
        full_files = self._collect_parquet_files(spec["full_dir"])
        increment_files = self._collect_increment_files(spec["increment_relative"])

        if not full_files and not increment_files:
            logger.warning("参考数据无可用文件: %s", dataset_name)
            return None

        merged, increment_days, output_days = self._merge_with_duckdb(full_files, increment_files)
        if merged is None:
            return None

        logger.info(
            "参考数据增量合并完成: dataset=%s, rows=%s, inc_trade_days=%s, out_trade_days=%s",
            dataset_name,
            f"{len(merged):,}",
            increment_days,
            output_days,
        )

        if self.cache_enabled:
            self._cache[dataset_name] = merged
            return merged.copy(deep=True)
        return merged

    # -------------------- internals --------------------

    def _discover_partitions(self) -> List[Path]:
        if not self.increment_raw_root.exists():
            logger.warning("参考数据增量根目录不存在: %s", self.increment_raw_root)
            return []

        partitions: List[Path] = []
        for p in self.increment_raw_root.iterdir():
            if not p.is_dir() or not p.name.startswith("date="):
                continue
            date_part = p.name.split("=", 1)[1]
            try:
                dt = datetime.strptime(date_part, "%Y-%m-%d").date()
            except ValueError:
                logger.warning("跳过非法分区目录: %s", p)
                continue
            if dt <= self.latest_date_obj:
                partitions.append(p)

        partitions.sort(key=lambda x: x.name)
        return partitions

    @staticmethod
    def _uniq_sorted_paths(paths: Sequence[Path]) -> List[Path]:
        seen = set()
        out: List[Path] = []
        for p in paths:
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            out.append(Path(rp))
        out.sort(key=lambda x: str(x))
        return out

    def _collect_parquet_files(self, root: Path) -> List[Path]:
        if not root.exists():
            return []
        return self._uniq_sorted_paths(sorted(root.rglob("*.parquet")))

    def _collect_increment_files(self, relative_dir: Path) -> List[Path]:
        files: List[Path] = []
        for partition in self._partitions:
            candidate = partition / relative_dir
            if candidate.exists():
                files.extend(sorted(candidate.rglob("*.parquet")))
        return self._uniq_sorted_paths(files)

    @staticmethod
    def _sql_parquet_list(files: Sequence[Path]) -> str:
        escaped = [str(f).replace("'", "''") for f in files]
        return "[" + ", ".join(f"'{p}'" for p in escaped) + "]"

    @staticmethod
    def _quote_ident(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def _infer_key_columns(self, columns: Sequence[str]) -> List[str]:
        cols = set(columns)
        d = self.date_column

        if {"ts_code", d}.issubset(cols):
            return ["ts_code", d]
        if {"index_code", "con_code", d}.issubset(cols):
            return ["index_code", "con_code", d]
        if {"ts_code", "con_code", d}.issubset(cols):
            return ["ts_code", "con_code", d]
        if {"con_code", d}.issubset(cols):
            return ["con_code", d]
        if d in cols:
            return [d]
        return []

    def _merge_with_duckdb(
        self,
        full_files: Sequence[Path],
        increment_files: Sequence[Path],
    ) -> Tuple[Optional[pd.DataFrame], int, int]:
        if not full_files and not increment_files:
            return None, 0, 0

        duckdb_module = _import_duckdb()
        date_col_q = self._quote_ident(self.date_column)

        with duckdb_module.connect(database=":memory:") as con:
            con.execute("PRAGMA threads=4")

            full_sql = None
            increment_sql = None
            if full_files:
                full_sql = f"SELECT * FROM read_parquet({self._sql_parquet_list(full_files)}, union_by_name=true)"
            if increment_files:
                increment_sql = f"SELECT * FROM read_parquet({self._sql_parquet_list(increment_files)}, union_by_name=true)"

            all_files = self._uniq_sorted_paths([*full_files, *increment_files])
            schema_sql = f"SELECT * FROM read_parquet({self._sql_parquet_list(all_files)}, union_by_name=true)"
            columns = con.execute(f"SELECT * FROM ({schema_sql}) t LIMIT 0").fetch_df().columns.tolist()
            full_columns: List[str] = []
            increment_columns: List[str] = []
            if full_sql:
                full_columns = con.execute(f"SELECT * FROM ({full_sql}) t LIMIT 0").fetch_df().columns.tolist()
            if increment_sql:
                increment_columns = con.execute(
                    f"SELECT * FROM ({increment_sql}) t LIMIT 0"
                ).fetch_df().columns.tolist()
            has_date_col = self.date_column in columns

            date_filter_sql = "TRUE"
            if has_date_col:
                date_filter_sql = f"CAST({date_col_q} AS DATE) <= DATE '{self.latest_date}'"

            if full_sql:
                con.execute(f"CREATE TEMP VIEW full_filtered AS SELECT * FROM ({full_sql}) t WHERE {date_filter_sql}")
            if increment_sql:
                con.execute(
                    f"CREATE TEMP VIEW increment_filtered AS SELECT * FROM ({increment_sql}) t WHERE {date_filter_sql}"
                )

            increment_days = 0
            if increment_sql and has_date_col:
                increment_days = int(
                    con.execute(
                        f"SELECT COUNT(DISTINCT CAST({date_col_q} AS DATE)) FROM increment_filtered"
                    ).fetchone()[0]
                    or 0
                )

            if full_sql:
                if not has_date_col:
                    con.execute("CREATE TEMP VIEW full_selected AS SELECT * FROM full_filtered")
                else:
                    full_needed = max(0, self.lookback_trade_days - increment_days)

                    if full_needed == 0:
                        con.execute("CREATE TEMP VIEW full_selected AS SELECT * FROM full_filtered LIMIT 0")
                    else:
                        if increment_sql and increment_days > 0:
                            keep_dates_sql = (
                                f"SELECT d FROM ("
                                f"  SELECT DISTINCT CAST({date_col_q} AS DATE) AS d "
                                f"  FROM full_filtered "
                                f"  WHERE CAST({date_col_q} AS DATE) NOT IN ("
                                f"      SELECT DISTINCT CAST({date_col_q} AS DATE) FROM increment_filtered"
                                f"  )"
                                f") t ORDER BY d DESC LIMIT {full_needed}"
                            )
                        else:
                            keep_dates_sql = (
                                f"SELECT d FROM ("
                                f"  SELECT DISTINCT CAST({date_col_q} AS DATE) AS d "
                                f"  FROM full_filtered"
                                f") t ORDER BY d DESC LIMIT {full_needed}"
                            )

                        con.execute(f"CREATE TEMP VIEW keep_full_dates AS {keep_dates_sql}")
                        con.execute(
                            f"CREATE TEMP VIEW full_selected AS "
                            f"SELECT * FROM full_filtered "
                            f"WHERE CAST({date_col_q} AS DATE) IN (SELECT d FROM keep_full_dates)"
                        )

            def build_aligned_select(view_name: str, source_columns: Sequence[str], src_flag: int) -> str:
                source_set = set(source_columns)
                select_exprs: List[str] = []
                for col in columns:
                    col_q = self._quote_ident(col)
                    if col in source_set:
                        select_exprs.append(col_q)
                    else:
                        select_exprs.append(f"NULL AS {col_q}")
                return f"SELECT {', '.join(select_exprs)}, {src_flag} AS __src FROM {view_name}"

            union_parts: List[str] = []
            if full_sql:
                union_parts.append(build_aligned_select("full_selected", full_columns, 0))
            if increment_sql:
                union_parts.append(build_aligned_select("increment_filtered", increment_columns, 1))

            if not union_parts:
                return None, increment_days, 0

            union_sql = " UNION ALL ".join(union_parts)
            merged_pdf = con.execute(f"SELECT * FROM ({union_sql}) u").fetch_df()

        key_cols = self._infer_key_columns(columns)
        if key_cols:
            merged_pdf = (
                merged_pdf.sort_values("__src")
                .drop_duplicates(subset=key_cols, keep="last")
                .drop(columns=["__src"])
            )
        else:
            merged_pdf = merged_pdf.drop_duplicates().drop(columns=["__src"], errors="ignore")

        order_cols: List[str] = []
        if self.date_column in merged_pdf.columns:
            order_cols.append(self.date_column)
        if "ts_code" in merged_pdf.columns:
            order_cols.append("ts_code")
        elif "con_code" in merged_pdf.columns:
            order_cols.append("con_code")
        if order_cols:
            merged_pdf = merged_pdf.sort_values(order_cols).reset_index(drop=True)

        output_days = 0
        if self.date_column in merged_pdf.columns and not merged_pdf.empty:
            output_days = int(pd.to_datetime(merged_pdf[self.date_column], errors="coerce").dt.normalize().nunique())

        return merged_pdf, increment_days, output_days
