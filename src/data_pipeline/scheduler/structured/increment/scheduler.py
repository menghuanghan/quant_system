"""
结构化数据增量采集调度器

核心能力：
1. 按指定交易日采集结构化增量数据
2. 输出到 data/raw/structured_increment/date=YYYY-MM-DD/{domain}/
3. 列结构与类型严格对齐 data/raw/structured 对应全量文件
4. 对 index_weight / index_daily 仅采集核心指数；etf_daily 可选
5. 每次运行自动刷新 OPTIONAL_STATIC_TASKS：仅对比并按需覆盖 data/raw/structured，不写入增量目录
"""

import logging
import hashlib
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .config import (
    CORE_INDEX_CODES,
    DOMAIN_NAMES,
    FUNDAMENTAL_MANDATORY_TSCODE_BUNDLE_TASK,
    FUNDAMENTAL_MANDATORY_TSCODE_SUBTASK_TASKS,
    FUNDAMENTAL_MANDATORY_TSCODE_SUBTASKS,
    IncrementCollectionTask,
    OPTIONAL_STATIC_TASKS,
    get_increment_tasks,
)

logger = logging.getLogger(__name__)


@dataclass
class IncrementTaskResult:
    task_name: str
    domain: str
    success: bool
    output_path: Optional[str] = None
    records_count: int = 0
    error_message: Optional[str] = None


@dataclass
class StaticRefreshItem:
    task_name: str
    domain: str
    target_path: str
    action: str = "unknown"  # updated / would_update / unchanged / failed
    changed: bool = False
    old_rows: int = 0
    new_rows: int = 0
    row_diff: int = 0
    old_hash: str = ""
    new_hash: str = ""
    error_message: Optional[str] = None


@dataclass
class StaticRefreshReport:
    dry_run: bool = False
    total_tasks: int = 0
    updated: int = 0
    would_update: int = 0
    unchanged: int = 0
    failed: int = 0
    items: List[StaticRefreshItem] = field(default_factory=list)


@dataclass
class IncrementCollectionReport:
    target_date: str
    total_tasks: int = 0
    success_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    results: List[IncrementTaskResult] = field(default_factory=list)
    static_refresh_report: Optional[StaticRefreshReport] = None

    def add_result(self, result: IncrementTaskResult):
        self.results.append(result)
        if result.success:
            self.success_tasks += 1
        else:
            self.failed_tasks += 1


def _run_increment_subtask_in_subprocess(payload: Dict[str, Any]) -> Dict[str, Any]:
    """子进程执行单个增量子任务，返回可序列化结果。"""
    task_name = payload["task_name"]
    try:
        scheduler = StructuredIncrementScheduler(
            date=payload["date"],
            output_dir=payload["output_dir"],
            full_raw_dir=payload["full_raw_dir"],
            include_etf=payload.get("include_etf", False),
            include_static=False,
            etf_codes=payload.get("etf_codes"),
            core_index_codes=payload.get("core_index_codes"),
            skip_existing=payload.get("skip_existing", False),
            dry_run=payload.get("dry_run", False),
            static_task_names=None,
        )
        scheduler._load_collectors()
        task = next((t for t in FUNDAMENTAL_MANDATORY_TSCODE_SUBTASK_TASKS if t.name == task_name), None)
        if task is None:
            return {
                "task_name": task_name,
                "domain": "fundamental",
                "success": False,
                "output_path": None,
                "records_count": 0,
                "error_message": f"子任务不存在: {task_name}",
            }

        result = scheduler._run_single_task(task)
        return {
            "task_name": result.task_name,
            "domain": result.domain,
            "success": result.success,
            "output_path": result.output_path,
            "records_count": result.records_count,
            "error_message": result.error_message,
        }
    except Exception as e:
        return {
            "task_name": task_name,
            "domain": "fundamental",
            "success": False,
            "output_path": None,
            "records_count": 0,
            "error_message": str(e),
        }


class StructuredIncrementScheduler:
    """结构化增量采集调度器"""

    def __init__(
        self,
        date: str,
        output_dir: str = "data/raw/structured_increment",
        full_raw_dir: str = "data/raw/structured",
        include_etf: bool = False,
        include_static: bool = False,
        etf_codes: Optional[List[str]] = None,
        core_index_codes: Optional[List[str]] = None,
        skip_existing: bool = False,
        dry_run: bool = False,
        static_task_names: Optional[List[str]] = None,
    ):
        self.target_date_compact, self.target_date_iso = self._normalize_date(date)
        self.target_month = self.target_date_compact[:6]
        self.target_quarter = self._to_quarter(self.target_date_compact)

        self.output_dir = Path(output_dir)
        self.full_raw_dir = Path(full_raw_dir)
        self.include_etf = include_etf
        self.include_static = include_static
        self.skip_existing = skip_existing
        self.dry_run = dry_run

        self.core_index_codes = core_index_codes or CORE_INDEX_CODES
        self.etf_codes = etf_codes

        # 静态参考表不再作为增量落盘任务，include_static 参数仅兼容保留
        self.tasks = get_increment_tasks(include_etf=include_etf, include_static=False)
        self.static_tasks = list(OPTIONAL_STATIC_TASKS)
        if static_task_names:
            selected = set(static_task_names)
            all_names = {t.name for t in self.static_tasks}
            missing = sorted(selected - all_names)
            if missing:
                logger.warning("静态任务过滤中存在未知任务，将忽略: %s", ", ".join(missing))
            self.static_tasks = [t for t in self.static_tasks if t.name in selected]

        self._collector_funcs: Dict[str, Callable] = {}
        self._schema_cache: Dict[str, Any] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if include_static:
            logger.warning("include_static 参数已废弃：静态参考表将自动做全量对比更新，不写入 structured_increment")
        if self.dry_run:
            logger.warning("当前为 dry-run 模式：不会写入任何增量或全量数据文件")
        logger.info(
            "增量调度器初始化完成: date=%s, output=%s",
            self.target_date_iso,
            self.output_dir,
        )

    def _collect_single_stock_task_part(
        self,
        collector: Callable,
        task: IncrementCollectionTask,
        ts_code: str,
    ) -> pd.DataFrame:
        """采集单只股票的任务分片数据。"""
        params = self._build_params(task, code=ts_code)
        part = collector(**params)
        if part is None or part.empty:
            return pd.DataFrame()
        part = self._filter_by_target_date(part, task)
        if part is None or part.empty:
            return pd.DataFrame()
        return part

    def _collect_requires_stock_list_dataframe(
        self,
        task: IncrementCollectionTask,
        collector: Callable,
    ) -> pd.DataFrame:
        """按股票列表采集（串行遍历）。"""
        stock_list = self._get_stock_list()
        if not stock_list:
            raise ValueError("无法加载 stock_list_a")

        total_codes = len(stock_list)
        logger.info(
            "任务 %s/%s 启动按股票串行采集: 股票数=%s",
            task.domain,
            task.name,
            total_codes,
        )

        frames: List[pd.DataFrame] = []

        for idx, ts_code in enumerate(stock_list, 1):
            try:
                part = self._collect_single_stock_task_part(collector, task, ts_code)
                if not part.empty:
                    frames.append(part)
            except Exception as e:
                logger.warning("任务 %s/%s 采集失败 ts_code=%s: %s", task.domain, task.name, ts_code, e)
            if idx % 500 == 0 or idx == total_codes:
                logger.info("任务 %s/%s 采集进度: %s/%s", task.domain, task.name, idx, total_codes)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _normalize_date(date_str: str) -> Tuple[str, str]:
        s = str(date_str).strip()
        if len(s) == 8 and s.isdigit():
            dt = datetime.strptime(s, "%Y%m%d")
            return dt.strftime("%Y%m%d"), dt.strftime("%Y-%m-%d")
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%Y%m%d"), dt.strftime("%Y-%m-%d")

    @staticmethod
    def _to_quarter(date_compact: str) -> str:
        year = date_compact[:4]
        month = int(date_compact[4:6])
        q = (month - 1) // 3 + 1
        return f"{year}Q{q}"

    def _load_collectors(self):
        if self._collector_funcs:
            return

        from src.data_pipeline.collectors.structured.metadata import (
            get_stock_list_a,
            get_name_change,
            get_trade_calendar,
            get_suspend_info,
            get_st_status,
        )
        from src.data_pipeline.collectors.structured.market_data import (
            get_stock_daily,
            get_daily_basic,
            get_adj_factor,
            get_index_daily,
            get_etf_daily,
        )
        from src.data_pipeline.collectors.structured.fundamental import (
            get_balance_sheet,
            get_income_statement,
            get_cash_flow,
            get_financial_indicator,
            get_share_structure,
            get_top10_holders,
            get_pledge,
            get_share_float,
            get_repurchase,
            get_dividend,
        )
        from src.data_pipeline.collectors.structured.trading_behavior import (
            get_money_flow,
            get_margin_detail,
            get_top_list,
            get_top_inst,
            get_hsgt_flow,
            get_block_trade,
            get_margin_summary,
        )
        from src.data_pipeline.collectors.structured.cross_sectional import (
            get_sw_index_classify,
            get_sw_index_member,
        )
        from src.data_pipeline.collectors.structured.macro_exogenous import (
            get_cn_gdp,
            get_cn_cpi,
            get_cn_ppi,
            get_cn_pmi,
            get_cn_m2,
            get_lpr,
            get_shibor,
        )
        from src.data_pipeline.collectors.structured.deep_risk_quality import (
            get_market_congestion,
            get_stock_bond_spread,
            get_a_pe_pb_ew_median,
            get_buffett_indicator,
            get_break_net_stock,
        )
        from src.data_pipeline.collectors.structured.derivatives import (
            get_repo_daily,
            get_fut_daily,
            get_opt_basic,
        )
        from src.data_pipeline.collectors.structured.index_benchmark import get_index_weight

        self._collector_funcs = {
            "get_stock_list_a": get_stock_list_a,
            "get_name_change": get_name_change,
            "get_trade_calendar": get_trade_calendar,
            "get_suspend_info": get_suspend_info,
            "get_st_status": get_st_status,
            "get_stock_daily": get_stock_daily,
            "get_daily_basic": get_daily_basic,
            "get_adj_factor": get_adj_factor,
            "get_index_daily": get_index_daily,
            "get_etf_daily": get_etf_daily,
            "get_balance_sheet": get_balance_sheet,
            "get_income_statement": get_income_statement,
            "get_cash_flow": get_cash_flow,
            "get_financial_indicator": get_financial_indicator,
            "get_share_structure": get_share_structure,
            "get_top10_holders": get_top10_holders,
            "get_pledge": get_pledge,
            "get_share_float": get_share_float,
            "get_repurchase": get_repurchase,
            "get_dividend": get_dividend,
            "get_money_flow": get_money_flow,
            "get_margin_detail": get_margin_detail,
            "get_top_list": get_top_list,
            "get_top_inst": get_top_inst,
            "get_hsgt_flow": get_hsgt_flow,
            "get_block_trade": get_block_trade,
            "get_margin_summary": get_margin_summary,
            "get_sw_index_classify": get_sw_index_classify,
            "get_sw_index_member": get_sw_index_member,
            "get_cn_gdp": get_cn_gdp,
            "get_cn_cpi": get_cn_cpi,
            "get_cn_ppi": get_cn_ppi,
            "get_cn_pmi": get_cn_pmi,
            "get_cn_m2": get_cn_m2,
            "get_lpr": get_lpr,
            "get_shibor": get_shibor,
            "get_market_congestion": get_market_congestion,
            "get_stock_bond_spread": get_stock_bond_spread,
            "get_a_pe_pb_ew_median": get_a_pe_pb_ew_median,
            "get_buffett_indicator": get_buffett_indicator,
            "get_break_net_stock": get_break_net_stock,
            "get_repo_daily": get_repo_daily,
            "get_fut_daily": get_fut_daily,
            "get_opt_basic": get_opt_basic,
            "get_index_weight": get_index_weight,
        }

    def _collect_task_dataframe(self, task: IncrementCollectionTask) -> pd.DataFrame:
        """仅执行采集与必要过滤，不做落盘。"""
        collector = self._collector_funcs.get(task.collector_func)
        if collector is None:
            raise ValueError(f"采集器不存在: {task.collector_func}")

        # 1) 按 code 拆分任务
        if task.output_mode == "by_code":
            codes = self._resolve_codes(task)
            if not codes:
                return pd.DataFrame()

            frames: List[pd.DataFrame] = []
            for code in codes:
                params = self._build_params(task, code=code)
                part = collector(**params)
                if part is None or part.empty:
                    continue
                part = self._filter_by_target_date(part, task)
                if not part.empty:
                    frames.append(part)

            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # 2) 需要按股票循环
        if task.requires_stock_list:
            return self._collect_requires_stock_list_dataframe(task, collector)

        # 3) 普通任务
        params = self._build_params(task)
        df = collector(**params)
        if df is None:
            return pd.DataFrame()
        if not df.empty:
            df = self._filter_by_target_date(df, task)
        return df

    @staticmethod
    def _normalize_for_compare(df: pd.DataFrame) -> pd.DataFrame:
        """将 DataFrame 规范化后用于内容比较（忽略行顺序差异）。"""
        if df is None:
            return pd.DataFrame()

        if df.empty:
            return df.reset_index(drop=True)

        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S").astype("string")
            else:
                out[col] = out[col].astype("string")
            out[col] = out[col].fillna("<NA>")

        try:
            out = out.sort_values(by=list(out.columns), kind="mergesort", na_position="last")
        except Exception:
            # 某些复杂对象列不可排序时，退化为原顺序比较
            pass

        return out.reset_index(drop=True)

    def _is_same_dataframe_content(self, old_df: pd.DataFrame, new_df: pd.DataFrame) -> bool:
        """判断两份数据内容是否一致（列顺序一致、忽略行顺序）。"""
        if list(old_df.columns) != list(new_df.columns):
            return False
        if len(old_df) != len(new_df):
            return False

        old_norm = self._normalize_for_compare(old_df)
        new_norm = self._normalize_for_compare(new_df)
        return old_norm.equals(new_norm)

    def _dataframe_hash(self, df: pd.DataFrame) -> str:
        """对 DataFrame 内容计算稳定哈希（忽略行顺序）。"""
        norm = self._normalize_for_compare(df)
        digest = hashlib.sha256()
        digest.update("|".join(norm.columns).encode("utf-8"))
        if not norm.empty:
            row_hash = pd.util.hash_pandas_object(norm, index=False)
            digest.update(row_hash.values.tobytes())
        return digest.hexdigest()

    def _save_full_raw_dataframe(self, df: pd.DataFrame, task: IncrementCollectionTask) -> Path:
        """将静态参考表写回 data/raw/structured。"""
        output_path = self.full_raw_dir / task.domain / f"{task.name}.parquet"
        if self.dry_run:
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, compression="snappy")
        return output_path

    def _refresh_static_reference_tables(self) -> StaticRefreshReport:
        """刷新静态参考表：采集最新数据，与全量表对比，不一致则覆盖全量。"""
        report = StaticRefreshReport(dry_run=self.dry_run, total_tasks=len(self.static_tasks))

        if not self.static_tasks:
            return report

        logger.info("开始刷新静态参考表（仅更新 data/raw/structured，不写入 structured_increment）")

        for task in self.static_tasks:
            logger.info("[static] 刷新 %s/%s", task.domain, task.name)
            target_path = self.full_raw_dir / task.domain / f"{task.name}.parquet"
            item = StaticRefreshItem(
                task_name=task.name,
                domain=task.domain,
                target_path=str(target_path),
            )

            try:
                new_df = self._collect_task_dataframe(task)
                if new_df is None:
                    new_df = pd.DataFrame()

                ref_path = self._get_reference_path(task)
                old_df = pd.DataFrame()
                if ref_path.exists():
                    old_df = pd.read_parquet(ref_path)
                    old_df = self._align_to_reference_schema(old_df, ref_path)
                    new_df = self._align_to_reference_schema(new_df, ref_path)

                item.old_rows = len(old_df)
                item.new_rows = len(new_df)
                item.row_diff = item.new_rows - item.old_rows
                item.old_hash = self._dataframe_hash(old_df)
                item.new_hash = self._dataframe_hash(new_df)

                changed = True
                if ref_path.exists():
                    changed = not self._is_same_dataframe_content(old_df, new_df)

                item.changed = changed

                if changed:
                    out = self._save_full_raw_dataframe(new_df, task)
                    if self.dry_run:
                        item.action = "would_update"
                        report.would_update += 1
                        logger.info(
                            "[static][dry-run] 检测到变化 %s/%s: rows %s -> %s, hash %.12s -> %.12s",
                            task.domain,
                            task.name,
                            item.old_rows,
                            item.new_rows,
                            item.old_hash,
                            item.new_hash,
                        )
                        logger.info("[static][dry-run] 将更新目标文件: %s", out)
                    else:
                        item.action = "updated"
                        report.updated += 1
                        logger.info(
                            "[static] 已更新 %s/%s -> %s (rows %s -> %s, hash %.12s -> %.12s)",
                            task.domain,
                            task.name,
                            out,
                            item.old_rows,
                            item.new_rows,
                            item.old_hash,
                            item.new_hash,
                        )
                else:
                    item.action = "unchanged"
                    report.unchanged += 1
                    logger.info(
                        "[static] 无变化 %s/%s (rows=%s, hash=%.12s)",
                        task.domain,
                        task.name,
                        item.new_rows,
                        item.new_hash,
                    )
            except Exception as e:
                item.action = "failed"
                item.error_message = str(e)
                report.failed += 1
                logger.warning("[static] 刷新失败 %s/%s: %s", task.domain, task.name, e)
                logger.debug(traceback.format_exc())
            finally:
                report.items.append(item)

        logger.info(
            "静态参考表刷新完成: updated=%s, would_update=%s, unchanged=%s, failed=%s",
            report.updated,
            report.would_update,
            report.unchanged,
            report.failed,
        )
        return report

    def _get_stock_list(self) -> List[str]:
        primary = self.full_raw_dir / "metadata" / "stock_list_a.parquet"
        fallback = Path("data/raw/structured/metadata/stock_list_a.parquet")

        candidates = [primary]
        if fallback != primary:
            candidates.append(fallback)

        for stock_file in candidates:
            if not stock_file.exists():
                continue
            try:
                df = pd.read_parquet(stock_file)
            except Exception as e:
                logger.warning("读取 stock_list_a 失败: %s (%s)", stock_file, e)
                continue

            if "ts_code" not in df.columns:
                logger.warning("stock_list_a 缺少 ts_code 字段: %s", stock_file)
                continue

            logger.info("使用股票池文件: %s", stock_file)
            return df["ts_code"].dropna().astype(str).unique().tolist()

        logger.warning("未找到可用 stock_list_a，尝试路径: %s", ", ".join(str(p) for p in candidates))
        return []

    def _get_etf_codes(self) -> List[str]:
        if self.etf_codes:
            return self.etf_codes
        etf_dir = self.full_raw_dir / "market_data" / "etf_daily"
        if not etf_dir.exists():
            return []
        codes = []
        for fp in etf_dir.glob("*.parquet"):
            code = fp.stem.replace("_", ".")
            codes.append(code)
        return sorted(set(codes))

    def _resolve_codes(self, task: IncrementCollectionTask) -> List[str]:
        if task.code_source == "core_index":
            return self.core_index_codes
        if task.code_source == "etf":
            return self._get_etf_codes()
        return []

    def _build_params(self, task: IncrementCollectionTask, code: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(task.extra_params)

        if task.param_style == "trade_date":
            params.update({"trade_date": self.target_date_compact})
        elif task.param_style == "start_end":
            params.update({"start_date": self.target_date_compact, "end_date": self.target_date_compact})
        elif task.param_style == "ann_date":
            params.update({"ann_date": self.target_date_compact})
        elif task.param_style == "float_date":
            params.update({"float_date": self.target_date_compact})
        elif task.param_style == "end_date":
            params.update({"end_date": self.target_date_compact})
        elif task.param_style == "month":
            params.update({"start_m": self.target_month, "end_m": self.target_month})
        elif task.param_style == "quarter":
            params.update({"start_q": self.target_quarter, "end_q": self.target_quarter})
        elif task.param_style == "trade_calendar_day":
            params.update({
                "exchange": "SSE",
                "start_date": self.target_date_compact,
                "end_date": self.target_date_compact,
            })

        if code:
            params[task.code_param] = code

        return params

    def _normalize_date_like(self, value: Any, granularity: str) -> Optional[str]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None

        s = str(value).strip()
        if not s:
            return None

        if granularity == "quarter":
            m = re.search(
                r"(?P<year>\d{4})\s*年?\s*第?\s*(?P<q1>[1-4])\s*(?:[-~到至]\s*(?P<q2>[1-4]))?\s*季(?:度)?",
                s,
            )
            if m:
                q = m.group("q2") or m.group("q1")
                return f"{m.group('year')}Q{q}"

            m = re.search(r"(?P<year>\d{4})\s*[Qq]\s*(?P<q>[1-4])", s)
            if m:
                return f"{m.group('year')}Q{m.group('q')}"

            s = s.upper().replace("-", "")
            if len(s) == 6 and "Q" in s:
                return s
            if len(s) == 8 and s.isdigit():
                return self._to_quarter(s)
            try:
                dt = pd.to_datetime(s, errors="coerce")
                if pd.isna(dt):
                    return None
                q = (dt.month - 1) // 3 + 1
                return f"{dt.year}Q{q}"
            except Exception:
                return None

        if granularity == "month":
            m = re.search(r"(?P<year>\d{4})\D*(?P<month>\d{1,2})\D*", s)
            if m:
                month = int(m.group("month"))
                if 1 <= month <= 12:
                    return f"{m.group('year')}{month:02d}"

            s = s.replace("-", "")
            if len(s) >= 6 and s[:6].isdigit():
                return s[:6]
            try:
                dt = pd.to_datetime(s, errors="coerce")
                if pd.isna(dt):
                    return None
                return dt.strftime("%Y%m")
            except Exception:
                return None

        # day
        m = re.search(r"(?P<year>\d{4})\D*(?P<month>\d{1,2})\D*(?P<day>\d{1,2})\D*", s)
        if m:
            month = int(m.group("month"))
            day = int(m.group("day"))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return f"{m.group('year')}{month:02d}{day:02d}"

        s = s.replace("-", "").replace("/", "")
        if len(s) >= 8 and s[:8].isdigit():
            return s[:8]
        try:
            dt = pd.to_datetime(s, errors="coerce")
            if pd.isna(dt):
                return None
            return dt.strftime("%Y%m%d")
        except Exception:
            return None

    def _filter_by_target_date(self, df: pd.DataFrame, task: IncrementCollectionTask) -> pd.DataFrame:
        if df.empty or task.date_granularity == "none" or not task.date_field:
            return df
        if task.date_field not in df.columns:
            return df

        target = self.target_date_compact
        if task.date_granularity == "month":
            target = self.target_month
        elif task.date_granularity == "quarter":
            target = self.target_quarter

        key = df[task.date_field].apply(lambda x: self._normalize_date_like(x, task.date_granularity))
        return df[key == target].copy()

    def _apply_share_float_data_rules(self, df: pd.DataFrame, task: IncrementCollectionTask) -> pd.DataFrame:
        """share_float 规则：过滤 ann_date > float_date，并对结果按全列去重。"""
        if task.name != "share_float" or df.empty:
            return df

        out = df.copy()

        if "ann_date" in out.columns and "float_date" in out.columns:
            ann = pd.to_datetime(out["ann_date"], errors="coerce")
            flt = pd.to_datetime(out["float_date"], errors="coerce")

            # 仅当两侧日期都有效且 ann_date > float_date 时过滤
            invalid_mask = ann.notna() & flt.notna() & (ann > flt)
            dropped = int(invalid_mask.sum())

            if dropped > 0:
                logger.info(
                    "share_float 规则过滤: 删除 ann_date > float_date 记录 %s 条",
                    dropped,
                )

            out = out.loc[~invalid_mask].copy()

        before_dedup = len(out)
        out = out.drop_duplicates(ignore_index=True)
        dedup_dropped = before_dedup - len(out)
        if dedup_dropped > 0:
            logger.info(
                "share_float 去重: 删除重复记录 %s 条",
                dedup_dropped,
            )

        return out

    def _apply_money_flow_stock_pool_rules(self, df: pd.DataFrame, task: IncrementCollectionTask) -> pd.DataFrame:
        """money_flow 规则：按 stock_list_a 代码池过滤，保持与 full 调度口径一致。"""
        if task.name != "money_flow" or df.empty:
            return df

        if "ts_code" not in df.columns:
            return df

        stock_list = self._get_stock_list()
        if not stock_list:
            logger.warning("money_flow 代码池过滤跳过：无法加载 stock_list_a")
            return df

        stock_set = {str(x) for x in stock_list}
        before = len(df)
        out = df[df["ts_code"].astype(str).isin(stock_set)].copy()
        dropped = before - len(out)

        if dropped > 0:
            logger.info("money_flow 代码池过滤: 删除非 stock_list_a 记录 %s 条", dropped)

        return out

    def _get_reference_path(self, task: IncrementCollectionTask, code: Optional[str] = None) -> Path:
        if task.output_mode == "by_code" and code:
            code_file = code.replace(".", "_") + ".parquet"
            return self.full_raw_dir / task.domain / task.name / code_file
        return self.full_raw_dir / task.domain / f"{task.name}.parquet"

    def _align_to_reference_schema(self, df: pd.DataFrame, reference_path: Path) -> pd.DataFrame:
        if not reference_path.exists():
            logger.warning("参考全量文件不存在，跳过schema对齐: %s", reference_path)
            return df

        schema_key = str(reference_path)
        schema = self._schema_cache.get(schema_key)

        if schema is None:
            try:
                import pyarrow.parquet as pq
                schema = pq.read_schema(reference_path)
                self._schema_cache[schema_key] = schema
            except Exception as e:
                logger.warning("读取参考schema失败，跳过类型对齐: %s (%s)", reference_path, e)
                return df

        ref_cols = [field.name for field in schema]

        # 补齐/裁剪列
        for col in ref_cols:
            if col not in df.columns:
                df[col] = pd.NA
        extra_cols = [c for c in df.columns if c not in ref_cols]
        if extra_cols:
            df = df.drop(columns=extra_cols)
        df = df[ref_cols]

        # 类型对齐（按 Arrow schema）
        for field in schema:
            col = field.name
            try:
                df[col] = self._cast_series_to_arrow(df[col], field.type)
            except Exception:
                # 保底：单列失败不影响整体
                continue

        return df

    @staticmethod
    def _cast_series_to_arrow(series: pd.Series, arrow_type: Any) -> pd.Series:
        import pyarrow as pa

        if pa.types.is_integer(arrow_type):
            return pd.to_numeric(series, errors="coerce").astype("Int64")
        if pa.types.is_floating(arrow_type) or pa.types.is_decimal(arrow_type):
            return pd.to_numeric(series, errors="coerce").astype("float64")
        if pa.types.is_boolean(arrow_type):
            return series.astype("boolean")
        if pa.types.is_timestamp(arrow_type) or pa.types.is_date(arrow_type):
            return pd.to_datetime(series, errors="coerce")
        if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type) or pa.types.is_binary(arrow_type):
            return series.astype("string")
        return series

    def _save_dataframe(self, df: pd.DataFrame, task: IncrementCollectionTask, code: Optional[str] = None) -> Path:
        date_root = self.output_dir / f"date={self.target_date_iso}" / task.domain
        if task.output_mode == "by_code" and code:
            output_path = date_root / task.name / f"{code.replace('.', '_')}.parquet"
        else:
            output_path = date_root / f"{task.name}.parquet"

        if self.dry_run:
            return output_path

        if self.skip_existing and output_path.exists():
            logger.info("跳过已存在增量文件: %s", output_path)
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, compression="snappy")
        return output_path

    def _run_fundamental_bundle_task(self, task: IncrementCollectionTask) -> IncrementTaskResult:
        """执行 fundamental 强制 ts_code 子任务打包并行采集（固定 5 进程）。"""
        subtask_names = list(FUNDAMENTAL_MANDATORY_TSCODE_SUBTASKS)
        workers = 5

        logger.info(
            "任务 %s/%s 启动打包并行采集: 子任务=%s, 进程=%s",
            task.domain,
            task.name,
            ",".join(subtask_names),
            workers,
        )

        payloads = [
            {
                "task_name": name,
                "date": self.target_date_iso,
                "output_dir": str(self.output_dir),
                "full_raw_dir": str(self.full_raw_dir),
                "include_etf": self.include_etf,
                "etf_codes": self.etf_codes,
                "core_index_codes": self.core_index_codes,
                "skip_existing": self.skip_existing,
                "dry_run": self.dry_run,
            }
            for name in subtask_names
        ]

        sub_results: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_increment_subtask_in_subprocess, payload): payload["task_name"]
                for payload in payloads
            }
            for future in as_completed(futures):
                subtask_name = futures[future]
                try:
                    sub = future.result()
                except Exception as e:
                    sub = {
                        "task_name": subtask_name,
                        "domain": "fundamental",
                        "success": False,
                        "output_path": None,
                        "records_count": 0,
                        "error_message": str(e),
                    }

                sub_results.append(sub)
                if sub.get("success"):
                    logger.info(
                        "并行子任务完成: %s, records=%s",
                        sub.get("task_name"),
                        sub.get("records_count", 0),
                    )
                else:
                    logger.error(
                        "并行子任务失败: %s, error=%s",
                        sub.get("task_name"),
                        sub.get("error_message"),
                    )

        failed = [r for r in sub_results if not r.get("success")]
        total_records = sum(int(r.get("records_count", 0) or 0) for r in sub_results)
        output_root = str(self.output_dir / f"date={self.target_date_iso}" / "fundamental")

        if failed:
            err_summary = "; ".join(
                f"{r.get('task_name')}: {r.get('error_message')}" for r in failed[:5]
            )
            return IncrementTaskResult(
                task_name=task.name,
                domain=task.domain,
                success=False,
                output_path=output_root,
                records_count=total_records,
                error_message=f"打包并行任务失败 {len(failed)}/{len(subtask_names)}: {err_summary}",
            )

        return IncrementTaskResult(
            task_name=task.name,
            domain=task.domain,
            success=True,
            output_path=output_root,
            records_count=total_records,
        )

    def _run_single_task(self, task: IncrementCollectionTask) -> IncrementTaskResult:
        if task.name == FUNDAMENTAL_MANDATORY_TSCODE_BUNDLE_TASK:
            return self._run_fundamental_bundle_task(task)

        collector = self._collector_funcs.get(task.collector_func)
        if collector is None:
            return IncrementTaskResult(task.name, task.domain, False, error_message=f"采集器不存在: {task.collector_func}")

        try:
            # 1) 按 code 拆分任务（核心指数 / ETF）
            if task.output_mode == "by_code":
                codes = self._resolve_codes(task)
                if not codes:
                    return IncrementTaskResult(task.name, task.domain, True, records_count=0)

                total = 0
                last_output = None
                for code in codes:
                    params = self._build_params(task, code=code)
                    df = collector(**params)
                    if df is None or df.empty:
                        continue

                    df = self._filter_by_target_date(df, task)
                    if df.empty:
                        continue

                    ref_path = self._get_reference_path(task, code=code)
                    df = self._align_to_reference_schema(df, ref_path)
                    out = self._save_dataframe(df, task, code=code)
                    total += len(df)
                    last_output = str(out.parent)

                return IncrementTaskResult(task.name, task.domain, True, output_path=last_output, records_count=total)

            # 2) 需要按股票循环的任务
            if task.requires_stock_list:
                df = self._collect_requires_stock_list_dataframe(task, collector)
            else:
                params = self._build_params(task)
                df = collector(**params)

            if df is None:
                df = pd.DataFrame()

            if not df.empty:
                df = self._apply_share_float_data_rules(df, task)
                df = self._filter_by_target_date(df, task)
                df = self._apply_money_flow_stock_pool_rules(df, task)

            ref_path = self._get_reference_path(task)
            df = self._align_to_reference_schema(df, ref_path)

            output_path = self._save_dataframe(df, task)
            return IncrementTaskResult(
                task_name=task.name,
                domain=task.domain,
                success=True,
                output_path=str(output_path),
                records_count=len(df),
            )

        except Exception as e:
            logger.error("任务失败 [%s]: %s", task.name, e)
            logger.debug(traceback.format_exc())
            return IncrementTaskResult(task.name, task.domain, False, error_message=str(e))

    def run(
        self,
        domains: Optional[List[str]] = None,
        task_names: Optional[List[str]] = None,
    ) -> IncrementCollectionReport:
        """执行增量采集"""
        self._load_collectors()

        report = IncrementCollectionReport(target_date=self.target_date_iso)
        report.static_refresh_report = self._refresh_static_reference_tables()

        tasks = [t for t in self.tasks if t.enabled]
        if domains:
            tasks = [t for t in tasks if t.domain in domains]
        if task_names:
            requested = set(task_names)
            if requested.intersection(FUNDAMENTAL_MANDATORY_TSCODE_SUBTASKS):
                logger.info(
                    "检测到财务四表/top10_holders 子任务名，自动映射为打包任务: %s",
                    FUNDAMENTAL_MANDATORY_TSCODE_BUNDLE_TASK,
                )
                requested -= set(FUNDAMENTAL_MANDATORY_TSCODE_SUBTASKS)
                requested.add(FUNDAMENTAL_MANDATORY_TSCODE_BUNDLE_TASK)

            tasks = [t for t in tasks if t.name in requested]

        report.total_tasks = len(tasks)

        logger.info("=" * 60)
        logger.info("结构化增量采集开始: %s", self.target_date_iso)
        logger.info("任务数: %s", len(tasks))
        logger.info("=" * 60)

        for idx, task in enumerate(tasks, 1):
            logger.info("[%s/%s] %s/%s - %s", idx, len(tasks), task.domain, task.name, task.description)
            result = self._run_single_task(task)
            report.add_result(result)

        logger.info("=" * 60)
        logger.info("结构化增量采集完成: 成功=%s, 失败=%s", report.success_tasks, report.failed_tasks)
        logger.info("=" * 60)

        return report

    def list_tasks(self) -> Dict[str, List[str]]:
        """按域列出任务"""
        result: Dict[str, List[str]] = {}
        for t in self.tasks:
            result.setdefault(t.domain, []).append(t.name)
        return result

    def list_domains(self) -> Dict[str, str]:
        """列出可用数据域"""
        used_domains = {t.domain for t in self.tasks}
        return {d: DOMAIN_NAMES.get(d, d) for d in sorted(used_domains)}
