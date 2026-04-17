"""
结构化 Raw + 非结构化 Processed 滚动窗口更新脚本

功能：
1. 从增量目录按 date=YYYY-MM-DD 分区识别“完整月”
2. 选择最早连续 N 个月（不足自动降级）
3. 将增量月数据并入目标输出目录，并从全量窗口中淘汰最早同月数
4. 写入成功后，自动删除已合并的增量分区（按侧独立）

目录口径：
- structured: full=data/raw/structured, increment=data/raw/structured_increment
- unstructured: full=data/processed/unstructured, increment=data/processed/unstructured_increment

说明：
- 支持分别指定 structured/unstructured 输出目录。
- 默认输出目录等于对应 full 根目录；指定其他目录时，原 full 目录不受影响。
- 仅处理“增量分区中实际出现的 parquet 数据集”。
"""

from __future__ import annotations

import argparse
import calendar
import importlib
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_duckdb():
    """运行时导入 duckdb，避免导入阶段硬依赖。"""
    try:
        return importlib.import_module("duckdb")
    except Exception as e:  # pragma: no cover - 仅在环境缺失时报错
        raise ImportError("未安装 duckdb，请先安装 requirements.txt 中的 duckdb 依赖") from e


@dataclass
class DatasetResult:
    relative_path: str
    source_full_path: str
    output_path: str
    date_field: str
    full_files: int
    increment_files: int
    full_rows_before: Optional[int]
    full_rows_after_drop: Optional[int]
    increment_rows: Optional[int]
    output_rows: Optional[int]
    action: str
    message: str = ""


@dataclass
class SideResult:
    side: str
    enabled: bool
    increment_root: str
    full_root: str
    output_root: str
    window_months: int
    requested_months: int
    full_end_month: Optional[str] = None
    complete_months: List[str] = field(default_factory=list)
    ignored_overlap_months: List[str] = field(default_factory=list)
    candidate_months: List[str] = field(default_factory=list)
    effective_months: List[str] = field(default_factory=list)
    drop_months: List[str] = field(default_factory=list)
    selected_partition_count: int = 0
    discovered_datasets: int = 0
    written_datasets: int = 0
    cleaned_partitions: List[str] = field(default_factory=list)
    status: str = "pending"
    error: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    datasets: List[DatasetResult] = field(default_factory=list)


@dataclass
class SideSpec:
    side: str
    enabled: bool
    full_root: Path
    increment_root: Path
    output_root: Path
    window_months: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按月滚动更新 structured/raw 与 unstructured/processed，并在成功后清理已合并增量分区"
    )
    parser.add_argument("--months", type=int, required=True, help="期望滚动的完整月数（1~12）")

    parser.add_argument(
        "--structured-full-root",
        type=str,
        default="data/raw/structured",
        help="结构化全量根目录（作为基线输入）",
    )
    parser.add_argument(
        "--structured-increment-root",
        type=str,
        default="data/raw/structured_increment",
        help="结构化增量分区目录（date=YYYY-MM-DD）",
    )
    parser.add_argument(
        "--structured-output-root",
        type=str,
        default=None,
        help="结构化输出根目录；默认与 --structured-full-root 相同",
    )

    parser.add_argument(
        "--unstructured-full-root",
        type=str,
        default="data/processed/unstructured",
        help="非结构化全量根目录（作为基线输入）",
    )
    parser.add_argument(
        "--unstructured-increment-root",
        type=str,
        default="data/processed/unstructured_increment",
        help="非结构化增量分区目录（date=YYYY-MM-DD）",
    )
    parser.add_argument(
        "--unstructured-output-root",
        type=str,
        default=None,
        help="非结构化输出根目录；默认与 --unstructured-full-root 相同",
    )

    parser.add_argument(
        "--structured-window-months",
        type=int,
        default=84,
        help="结构化窗口总月数（默认84）",
    )
    parser.add_argument(
        "--unstructured-window-months",
        type=int,
        default=62,
        help="非结构化窗口总月数（默认62）",
    )

    parser.add_argument("--enable-structured", dest="enable_structured", action="store_true", default=True)
    parser.add_argument("--disable-structured", dest="enable_structured", action="store_false")
    parser.add_argument("--enable-unstructured", dest="enable_unstructured", action="store_true", default=True)
    parser.add_argument("--disable-unstructured", dest="enable_unstructured", action="store_false")

    parser.add_argument("--dry-run", action="store_true", help="仅计算计划，不写文件、不删除分区")
    parser.add_argument("--log-file", type=str, default="logs/raw_window_update.log", help="日志文件")
    parser.add_argument("--report-file", type=str, default=None, help="审计报告输出路径（JSON）")

    return parser.parse_args()


def setup_logging(log_file: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def parse_month_key(m: str) -> Tuple[int, int]:
    y, mm = m.split("-")
    return int(y), int(mm)


def add_month(m: str, delta: int) -> str:
    y, mm = parse_month_key(m)
    idx = y * 12 + (mm - 1) + delta
    new_y = idx // 12
    new_m = idx % 12 + 1
    return f"{new_y:04d}-{new_m:02d}"


def list_months_ending(end_month: str, count: int) -> List[str]:
    if count <= 0:
        return []
    start = add_month(end_month, -(count - 1))
    return [add_month(start, i) for i in range(count)]


def contiguous_prefix(months: Sequence[str]) -> List[str]:
    if not months:
        return []
    out = [months[0]]
    cur = months[0]
    for m in months[1:]:
        if m == add_month(cur, 1):
            out.append(m)
            cur = m
        else:
            break
    return out


def discover_partition_dirs(root: Path) -> Dict[date, Path]:
    result: Dict[date, Path] = {}
    if not root.exists():
        return result

    for p in root.iterdir():
        if not p.is_dir() or not p.name.startswith("date="):
            continue
        raw = p.name.split("=", 1)[1]
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            logging.getLogger(__name__).warning("跳过非法分区目录: %s", p)
            continue
        result[dt] = p
    return result


def discover_complete_months(dates: Iterable[date]) -> List[str]:
    by_month: Dict[str, set[int]] = {}
    for d in dates:
        m = month_key(d)
        by_month.setdefault(m, set()).add(d.day)

    complete: List[str] = []
    for m in sorted(by_month.keys()):
        y, mm = parse_month_key(m)
        month_days = calendar.monthrange(y, mm)[1]
        if by_month[m] >= set(range(1, month_days + 1)):
            complete.append(m)
    return complete


def sql_quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_quote_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def sql_parquet_list(files: Sequence[Path]) -> str:
    escaped = [sql_quote_literal(str(p.resolve())) for p in files]
    return "[" + ", ".join(escaped) + "]"


def uniq_sorted_paths(paths: Sequence[Path]) -> List[Path]:
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


def collect_parquet_candidates(candidate: Path) -> List[Path]:
    files: List[Path] = []

    if candidate.is_file() and candidate.suffix == ".parquet":
        files.append(candidate)

    if candidate.is_dir():
        files.extend(sorted(candidate.rglob("*.parquet")))

    if candidate.suffix == ".parquet":
        alt_dir = candidate.with_suffix("")
        if alt_dir.is_dir():
            files.extend(sorted(alt_dir.rglob("*.parquet")))
    else:
        alt_file = candidate.with_suffix(".parquet")
        if alt_file.is_file():
            files.append(alt_file)

    return uniq_sorted_paths(files)


def build_parsed_date_expr(column_name: str) -> str:
    c = sql_quote_identifier(column_name)
    v = f"trim(CAST({c} AS VARCHAR))"
    return (
        "CASE "
        f"WHEN {c} IS NULL THEN NULL "
        f"WHEN regexp_full_match({v}, '^[0-9]{{8}}$') THEN strptime({v}, '%Y%m%d') "
        f"WHEN regexp_full_match({v}, '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$') THEN strptime({v}, '%Y-%m-%d') "
        f"WHEN regexp_full_match({v}, '^[0-9]{{6}}$') THEN strptime({v} || '01', '%Y%m%d') "
        f"WHEN regexp_full_match({v}, '^[0-9]{{4}}-[0-9]{{2}}$') THEN strptime({v} || '-01', '%Y-%m-%d') "
        f"WHEN regexp_full_match({v}, '^[0-9]{{4}}Q[1-4]$') "
        f"THEN make_date(CAST(substr({v}, 1, 4) AS INTEGER), (CAST(substr({v}, 6, 1) AS INTEGER)-1) * 3 + 1, 1) "
        f"ELSE try_cast({c} AS DATE) END"
    )


def parquet_relation_sql(files: Sequence[Path]) -> str:
    return (
        "SELECT * FROM read_parquet("
        f"{sql_parquet_list(files)}, "
        "union_by_name=true, "
        "hive_partitioning=false"
        ")"
    )


def read_columns(con, files: Sequence[Path]) -> List[str]:
    if not files:
        return []
    sql = f"SELECT * FROM ({parquet_relation_sql(files)}) t LIMIT 0"
    return con.execute(sql).fetch_df().columns.tolist()


def query_max_month(con, files: Sequence[Path], date_field: str) -> Optional[str]:
    if not files:
        return None

    cols = read_columns(con, files)
    if date_field not in cols:
        return None

    parsed = build_parsed_date_expr(date_field)
    sql = (
        "WITH src AS ("
        f"{parquet_relation_sql(files)}"
        "), parsed AS ("
        f"SELECT {parsed} AS __d FROM src"
        ") "
        "SELECT max(strftime(__d, '%Y-%m')) AS max_month FROM parsed WHERE __d IS NOT NULL"
    )
    row = con.execute(sql).fetchone()
    if not row:
        return None
    return row[0]


def build_structured_task_date_field_map() -> Dict[str, str]:
    from src.data_pipeline.scheduler.structured.increment.config import get_increment_tasks

    mapping: Dict[str, str] = {}
    for task in get_increment_tasks(include_etf=True, include_static=False):
        if task.date_field:
            mapping[task.name] = task.date_field
    return mapping


def infer_structured_task_name(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) >= 3:
        return parts[1]
    if len(parts) >= 2:
        return Path(parts[-1]).stem
    return Path(relative_path).stem


def infer_date_field_from_columns(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c

    lowered = {c.lower(): c for c in columns}
    for c in ["trade_date", "ann_date", "float_date", "end_date", "date", "cal_date"]:
        if c in lowered:
            return lowered[c]

    for c in columns:
        cl = c.lower()
        if cl.endswith("date") or "date" in cl:
            return c
    return None


def resolve_side_full_end_month(side: str, full_root: Path, con) -> Optional[str]:
    logger = logging.getLogger(__name__)

    if side == "structured":
        anchors = [
            ("market_data/stock_daily.parquet", "trade_date"),
            ("market_data/daily_basic.parquet", "trade_date"),
            ("fundamental/top10_holders.parquet", "ann_date"),
            ("fundamental/share_float.parquet", "float_date"),
        ]
    else:
        anchors = [
            ("announcements.parquet", "date"),
            ("events.parquet", "date"),
            ("reports.parquet", "date"),
            ("news_exchange.parquet", "date"),
            ("policy_gov.parquet", "date"),
        ]

    months: List[str] = []
    for rel_path, date_field in anchors:
        files = collect_parquet_candidates(full_root / rel_path)
        if not files:
            continue
        max_month = query_max_month(con, files, date_field)
        if max_month:
            months.append(max_month)

    if not months:
        logger.warning("%s 侧未能从锚点推断 full 末月，将不启用重叠月份剔除", side)
        return None

    return max(months)


def build_full_filter_sql(full_rel_sql: str, date_field: str, drop_months: Sequence[str]) -> str:
    if not drop_months:
        return full_rel_sql

    parsed_expr = build_parsed_date_expr(date_field)
    months_sql = ", ".join(sql_quote_literal(m) for m in drop_months)
    return (
        "SELECT * EXCLUDE(__parsed_date__) FROM ("
        "SELECT *, "
        f"{parsed_expr} AS __parsed_date__ "
        f"FROM ({full_rel_sql}) src"
        ") t "
        "WHERE __parsed_date__ IS NULL "
        f"OR strftime(__parsed_date__, '%Y-%m') NOT IN ({months_sql})"
    )


def build_aligned_relation_sql(
    relation_sql: str,
    source_columns: Sequence[str],
    target_columns: Sequence[str],
) -> str:
    """将 relation 对齐到 target_columns，缺失列补 NULL。"""
    source_set = set(source_columns)
    projections: List[str] = []
    for col in target_columns:
        q_col = sql_quote_identifier(col)
        if col in source_set:
            projections.append(q_col)
        else:
            projections.append(f"NULL AS {q_col}")

    return f"SELECT {', '.join(projections)} FROM ({relation_sql}) src"


def count_rows(con, relation_sql: Optional[str]) -> int:
    if not relation_sql:
        return 0
    row = con.execute(f"SELECT COUNT(*) FROM ({relation_sql}) t").fetchone()
    return int(row[0]) if row else 0


def collect_increment_file_map(partition_dirs: Sequence[Path]) -> Dict[str, List[Path]]:
    file_map: Dict[str, List[Path]] = {}
    for part in partition_dirs:
        for f in part.rglob("*.parquet"):
            rel = f.relative_to(part).as_posix()
            file_map.setdefault(rel, []).append(f)

    for k in list(file_map.keys()):
        file_map[k] = uniq_sorted_paths(file_map[k])
    return file_map


def merge_one_dataset(
    *,
    con,
    side: str,
    full_root: Path,
    output_root: Path,
    relative_path: str,
    increment_files: Sequence[Path],
    drop_months: Sequence[str],
    date_field: str,
    dry_run: bool,
) -> DatasetResult:
    logger = logging.getLogger(__name__)

    source_full_path = full_root / relative_path
    output_path = output_root / relative_path
    full_files = collect_parquet_candidates(source_full_path)
    inc_files = uniq_sorted_paths(increment_files)

    if not inc_files:
        return DatasetResult(
            relative_path=relative_path,
            source_full_path=str(source_full_path),
            output_path=str(output_path),
            date_field=date_field,
            full_files=len(full_files),
            increment_files=0,
            full_rows_before=0,
            full_rows_after_drop=0,
            increment_rows=0,
            output_rows=0,
            action="skipped",
            message="该数据集没有增量文件",
        )

    full_rel = parquet_relation_sql(full_files) if full_files else None
    inc_rel = parquet_relation_sql(inc_files)

    full_columns = read_columns(con, full_files) if full_files else []
    increment_columns = read_columns(con, inc_files)

    full_rows_before: Optional[int] = None
    full_rows_after_drop: Optional[int] = None
    increment_rows: Optional[int] = None
    output_rows: Optional[int] = None

    if full_rel and drop_months:
        if date_field not in full_columns:
            raise RuntimeError(
                f"数据集 {relative_path} 无法按月份淘汰：缺少日期字段 {date_field}"
            )

    full_filtered_rel = build_full_filter_sql(full_rel, date_field, drop_months) if full_rel else None

    target_columns = list(full_columns)
    for col in increment_columns:
        if col not in target_columns:
            target_columns.append(col)

    full_aligned_rel = (
        build_aligned_relation_sql(full_filtered_rel, full_columns, target_columns)
        if full_filtered_rel
        else None
    )
    increment_aligned_rel = build_aligned_relation_sql(inc_rel, increment_columns, target_columns)

    if full_aligned_rel:
        merged_rel = f"SELECT DISTINCT * FROM (({full_aligned_rel}) UNION ALL ({increment_aligned_rel})) u"
    else:
        merged_rel = f"SELECT DISTINCT * FROM ({increment_aligned_rel}) u"

    if not dry_run:
        full_rows_before = count_rows(con, full_rel)
        full_rows_after_drop = count_rows(con, full_filtered_rel)
        increment_rows = count_rows(con, inc_rel)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_output = output_path.parent / f".{output_path.name}.tmp-{uuid4().hex}.parquet"
        if tmp_output.exists():
            tmp_output.unlink()

        copy_sql = (
            f"COPY ({merged_rel}) TO {sql_quote_literal(str(tmp_output.resolve()))} "
            "(FORMAT PARQUET, COMPRESSION 'snappy')"
        )
        con.execute(copy_sql)
        output_rows = count_rows(con, parquet_relation_sql([tmp_output]))

        if output_path.exists():
            output_path.unlink()
        tmp_output.replace(output_path)

        action = "written"
        msg = "写入完成"
    else:
        action = "planned"
        msg = "dry-run，仅生成执行计划"

    logger.debug("%s %s: %s", side, relative_path, msg)

    return DatasetResult(
        relative_path=relative_path,
        source_full_path=str(source_full_path),
        output_path=str(output_path),
        date_field=date_field,
        full_files=len(full_files),
        increment_files=len(inc_files),
        full_rows_before=full_rows_before,
        full_rows_after_drop=full_rows_after_drop,
        increment_rows=increment_rows,
        output_rows=output_rows,
        action=action,
        message=msg,
    )


def partition_dirs_for_months(partitions: Dict[date, Path], months: Sequence[str]) -> List[Path]:
    month_set = set(months)
    selected = [p for d, p in partitions.items() if month_key(d) in month_set]
    selected.sort(key=lambda x: x.name)
    return selected


def resolve_dataset_date_field(
    *,
    con,
    side: str,
    relative_path: str,
    full_files: Sequence[Path],
    increment_files: Sequence[Path],
    structured_task_map: Dict[str, str],
) -> str:
    if side == "structured":
        task_name = infer_structured_task_name(relative_path)
        preferred = structured_task_map.get(task_name)
        candidates = [
            "trade_date",
            "ann_date",
            "float_date",
            "end_date",
            "date",
            "month",
            "quarter",
            "cal_date",
            "st_start_date",
        ]
        if preferred and preferred not in candidates:
            candidates.insert(0, preferred)
        elif preferred:
            candidates = [preferred] + [c for c in candidates if c != preferred]
    else:
        preferred = "date"
        candidates = ["date", "trade_date", "ann_date", "publish_date"]

    schema_files = full_files if full_files else increment_files
    cols = read_columns(con, schema_files)

    if preferred in cols:
        return preferred

    inferred = infer_date_field_from_columns(cols, candidates)
    if inferred:
        return inferred

    raise RuntimeError(f"数据集 {relative_path} 无法推断日期字段，列集合: {cols}")


def compute_drop_months(full_end_month: Optional[str], window_months: int, effective_count: int) -> List[str]:
    if not full_end_month or effective_count <= 0:
        return []
    window = list_months_ending(full_end_month, window_months)
    return window[: min(effective_count, len(window))]


def execute_side(
    *,
    spec: SideSpec,
    requested_months: int,
    dry_run: bool,
    structured_task_map: Dict[str, str],
) -> SideResult:
    logger = logging.getLogger(__name__)

    result = SideResult(
        side=spec.side,
        enabled=spec.enabled,
        increment_root=str(spec.increment_root),
        full_root=str(spec.full_root),
        output_root=str(spec.output_root),
        window_months=spec.window_months,
        requested_months=requested_months,
    )

    if not spec.enabled:
        result.status = "disabled"
        result.notes.append("该侧已禁用")
        return result

    if not spec.increment_root.exists():
        result.status = "skipped"
        result.notes.append("增量目录不存在")
        return result

    duckdb_module = _import_duckdb()
    with duckdb_module.connect(database=":memory:") as con:
        con.execute("PRAGMA threads=4")

        partitions = discover_partition_dirs(spec.increment_root)
        if not partitions:
            result.status = "skipped"
            result.notes.append("未发现增量分区")
            return result

        complete_months = discover_complete_months(partitions.keys())
        result.complete_months = complete_months

        full_end_month = resolve_side_full_end_month(spec.side, spec.full_root, con)
        result.full_end_month = full_end_month

        if full_end_month:
            result.ignored_overlap_months = [m for m in complete_months if m <= full_end_month]
            candidate_months = [m for m in complete_months if m > full_end_month]
        else:
            candidate_months = list(complete_months)

        result.candidate_months = candidate_months

        contiguous = contiguous_prefix(candidate_months)
        effective_count = min(requested_months, len(contiguous))
        effective_months = contiguous[:effective_count]
        result.effective_months = effective_months

        if requested_months > len(contiguous):
            result.notes.append(
                f"请求 {requested_months} 个月，实际可用连续完整月 {len(contiguous)} 个，已自动降级"
            )

        if not effective_months:
            result.status = "skipped"
            result.notes.append("无可用连续完整月份")
            return result

        result.drop_months = compute_drop_months(full_end_month, spec.window_months, len(effective_months))

        selected_partitions = partition_dirs_for_months(partitions, effective_months)
        result.selected_partition_count = len(selected_partitions)

        increment_map = collect_increment_file_map(selected_partitions)
        result.discovered_datasets = len(increment_map)

        if not increment_map:
            result.status = "skipped"
            result.notes.append("所选月份无可处理 parquet 数据集")
            return result

        logger.info(
            "[%s] effective_months=%s, drop_months=%s, datasets=%s",
            spec.side,
            effective_months,
            result.drop_months,
            result.discovered_datasets,
        )

        for relative_path in sorted(increment_map.keys()):
            inc_files = increment_map[relative_path]
            full_files = collect_parquet_candidates(spec.full_root / relative_path)

            date_field = resolve_dataset_date_field(
                con=con,
                side=spec.side,
                relative_path=relative_path,
                full_files=full_files,
                increment_files=inc_files,
                structured_task_map=structured_task_map,
            )

            ds = merge_one_dataset(
                con=con,
                side=spec.side,
                full_root=spec.full_root,
                output_root=spec.output_root,
                relative_path=relative_path,
                increment_files=inc_files,
                drop_months=result.drop_months,
                date_field=date_field,
                dry_run=dry_run,
            )
            result.datasets.append(ds)

            if ds.action in {"written", "planned"}:
                result.written_datasets += 1

    # 清理增量分区：仅在该侧全部成功且非 dry-run 时执行
    if not dry_run:
        selected_partition_paths = partition_dirs_for_months(discover_partition_dirs(spec.increment_root), result.effective_months)
        for p in selected_partition_paths:
            if p.exists() and p.is_dir():
                shutil.rmtree(p)
                result.cleaned_partitions.append(p.name)

    result.status = "success"
    return result


def build_report_path(args: argparse.Namespace) -> Path:
    if args.report_file:
        return Path(args.report_file)
    reports_dir = PROJECT_ROOT / "reports" / "raw_window_update"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "dry_run" if args.dry_run else "run"
    return reports_dir / f"raw_window_update_{suffix}_{ts}.json"


def validate_args(args: argparse.Namespace) -> None:
    if args.months < 1 or args.months > 12:
        raise ValueError("--months 必须在 1~12 之间")

    if not args.enable_structured and not args.enable_unstructured:
        raise ValueError("structured 与 unstructured 不能同时禁用")

    if args.structured_window_months <= 0 or args.unstructured_window_months <= 0:
        raise ValueError("窗口月数必须为正整数")


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)

    validate_args(args)

    structured_full_root = (PROJECT_ROOT / args.structured_full_root).resolve()
    structured_increment_root = (PROJECT_ROOT / args.structured_increment_root).resolve()
    structured_output_root = (
        (PROJECT_ROOT / args.structured_output_root).resolve()
        if args.structured_output_root
        else structured_full_root
    )

    unstructured_full_root = (PROJECT_ROOT / args.unstructured_full_root).resolve()
    unstructured_increment_root = (PROJECT_ROOT / args.unstructured_increment_root).resolve()
    unstructured_output_root = (
        (PROJECT_ROOT / args.unstructured_output_root).resolve()
        if args.unstructured_output_root
        else unstructured_full_root
    )

    if args.enable_structured and not structured_full_root.exists():
        raise FileNotFoundError(f"结构化 full 根目录不存在: {structured_full_root}")
    if args.enable_unstructured and not unstructured_full_root.exists():
        raise FileNotFoundError(f"非结构化 full 根目录不存在: {unstructured_full_root}")

    structured_spec = SideSpec(
        side="structured",
        enabled=args.enable_structured,
        full_root=structured_full_root,
        increment_root=structured_increment_root,
        output_root=structured_output_root,
        window_months=args.structured_window_months,
    )
    unstructured_spec = SideSpec(
        side="unstructured",
        enabled=args.enable_unstructured,
        full_root=unstructured_full_root,
        increment_root=unstructured_increment_root,
        output_root=unstructured_output_root,
        window_months=args.unstructured_window_months,
    )

    logger.info("=" * 90)
    logger.info("滚动更新开始: months=%s dry_run=%s", args.months, args.dry_run)
    logger.info(
        "structured: full=%s increment=%s output=%s",
        structured_spec.full_root,
        structured_spec.increment_root,
        structured_spec.output_root,
    )
    logger.info(
        "unstructured: full=%s increment=%s output=%s",
        unstructured_spec.full_root,
        unstructured_spec.increment_root,
        unstructured_spec.output_root,
    )
    logger.info("=" * 90)

    structured_task_map = build_structured_task_date_field_map()

    report: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(),
        "dry_run": args.dry_run,
        "requested_months": args.months,
        "config": {
            "structured_full_root": str(structured_full_root),
            "structured_increment_root": str(structured_increment_root),
            "structured_output_root": str(structured_output_root),
            "structured_window_months": args.structured_window_months,
            "unstructured_full_root": str(unstructured_full_root),
            "unstructured_increment_root": str(unstructured_increment_root),
            "unstructured_output_root": str(unstructured_output_root),
            "unstructured_window_months": args.unstructured_window_months,
            "enable_structured": args.enable_structured,
            "enable_unstructured": args.enable_unstructured,
        },
        "sides": {},
    }

    side_results: Dict[str, SideResult] = {}

    try:
        if structured_spec.enabled:
            sr = execute_side(
                spec=structured_spec,
                requested_months=args.months,
                dry_run=args.dry_run,
                structured_task_map=structured_task_map,
            )
            side_results["structured"] = sr
            report["sides"]["structured"] = asdict(sr)
            if sr.status == "success":
                logger.info(
                    "[structured] success: effective=%s, drop=%s, datasets=%s, cleaned=%s",
                    sr.effective_months,
                    sr.drop_months,
                    sr.written_datasets,
                    len(sr.cleaned_partitions),
                )
            else:
                logger.warning("[structured] status=%s notes=%s", sr.status, sr.notes)

        if unstructured_spec.enabled:
            ur = execute_side(
                spec=unstructured_spec,
                requested_months=args.months,
                dry_run=args.dry_run,
                structured_task_map=structured_task_map,
            )
            side_results["unstructured"] = ur
            report["sides"]["unstructured"] = asdict(ur)
            if ur.status == "success":
                logger.info(
                    "[unstructured] success: effective=%s, drop=%s, datasets=%s, cleaned=%s",
                    ur.effective_months,
                    ur.drop_months,
                    ur.written_datasets,
                    len(ur.cleaned_partitions),
                )
            else:
                logger.warning("[unstructured] status=%s notes=%s", ur.status, ur.notes)

    except Exception as e:
        logger.exception("执行失败")
        report["error"] = str(e)
        report_path = build_report_path(args)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.error("失败审计报告已写入: %s", report_path)
        return 1

    report_path = build_report_path(args)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("审计报告已写入: %s", report_path)
    logger.info("滚动更新结束")

    # 如果启用的侧全部非 success（比如都 skipped），返回0但给出提醒
    enabled_sides = [s for s in side_results.values() if s.enabled]
    if enabled_sides and all(s.status != "success" for s in enabled_sides):
        logger.warning("未发生实际合并，请检查完整月与候选月份条件")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
