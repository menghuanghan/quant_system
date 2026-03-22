"""
非结构化数据增量采集调度器

核心能力：
1. 接收目标日期，采集单日非结构化增量数据
2. 输出到 data/raw/unstructured_increment/date=YYYY-MM-DD/
3. 增量文件列名、字段结构、数据类型严格对齐全量 raw/unstructured 对应类型
"""

import importlib
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .config import (
    TASK_NAME_MAP,
    UnstructuredIncrementTask,
    get_increment_tasks,
)

logger = logging.getLogger(__name__)


@dataclass
class IncrementTaskResult:
    """单任务执行结果"""

    task_name: str
    success: bool
    output_path: Optional[str] = None
    records_count: int = 0
    skipped: bool = False
    error_message: Optional[str] = None
    processing_output_path: Optional[str] = None
    processing_records_count: int = 0


@dataclass
class IncrementCollectionReport:
    """增量采集汇总"""

    target_date: str
    total_tasks: int = 0
    success_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    total_records: int = 0
    total_processed_records: int = 0
    results: List[IncrementTaskResult] = field(default_factory=list)

    def add_result(self, result: IncrementTaskResult):
        self.results.append(result)
        self.total_records += int(result.records_count or 0)
        self.total_processed_records += int(result.processing_records_count or 0)
        if result.skipped:
            self.skipped_tasks += 1
        elif result.success:
            self.success_tasks += 1
        else:
            self.failed_tasks += 1


class UnstructuredIncrementScheduler:
    """非结构化增量采集调度器"""

    def __init__(
        self,
        date: str,
        output_dir: str = "data/raw/unstructured_increment",
        full_raw_dir: str = "data/raw/unstructured",
        processed_output_dir: str = "data/processed/unstructured_increment",
        full_processed_dir: str = "data/processed/unstructured",
        skip_existing: bool = False,
        processing_skip_existing: bool = False,
        enable_processing: bool = True,
        processing_model_name: str = "qwen2.5:7b-instruct",
        processing_ollama_host: str = "http://localhost:11434",
        processing_timeout: float = 60.0,
        processing_use_gpu: bool = True,
        dry_run: bool = False,
    ):
        self.target_date_compact, self.target_date_iso = self._normalize_date(date)
        self.output_dir = Path(output_dir)
        self.full_raw_dir = Path(full_raw_dir)
        self.processed_output_dir = Path(processed_output_dir)
        self.full_processed_dir = Path(full_processed_dir)
        self.skip_existing = skip_existing
        self.processing_skip_existing = processing_skip_existing
        self.enable_processing = enable_processing
        self.processing_model_name = processing_model_name
        self.processing_ollama_host = processing_ollama_host
        self.processing_timeout = processing_timeout
        self.processing_use_gpu = processing_use_gpu
        self.dry_run = dry_run

        self._collector_instances: Dict[str, Any] = {}
        self._schema_cache: Dict[str, Any] = {}
        self._processed_schema_cache: Dict[str, Any] = {}
        self._pipeline_cache: Dict[str, Any] = {}
        self._processing_config: Optional[Any] = None
        self._announcement_filter: Optional[Any] = None
        self._increment_events_df: Optional[pd.DataFrame] = None

        if not self.full_raw_dir.exists():
            raise FileNotFoundError(f"全量非结构化目录不存在: {self.full_raw_dir}")

        if self.enable_processing and not self.full_processed_dir.exists():
            raise FileNotFoundError(f"全量处理目录不存在: {self.full_processed_dir}")

        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.enable_processing:
                self.processed_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "非结构化增量调度器初始化完成: date=%s, output=%s, full_raw=%s, processed_output=%s, full_processed=%s, skip_existing=%s, processing_skip_existing=%s, enable_processing=%s, dry_run=%s",
            self.target_date_iso,
            self.output_dir,
            self.full_raw_dir,
            self.processed_output_dir,
            self.full_processed_dir,
            self.skip_existing,
            self.processing_skip_existing,
            self.enable_processing,
            self.dry_run,
        )

    @staticmethod
    def _normalize_date(date_str: str) -> Tuple[str, str]:
        """支持 YYYYMMDD / YYYY-MM-DD 输入"""
        s = str(date_str).strip()
        if len(s) == 8 and s.isdigit():
            dt = datetime.strptime(s, "%Y%m%d")
            return dt.strftime("%Y%m%d"), dt.strftime("%Y-%m-%d")
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%Y%m%d"), dt.strftime("%Y-%m-%d")

    @property
    def partition_dir(self) -> Path:
        return self.output_dir / f"date={self.target_date_iso}"

    @property
    def processed_partition_dir(self) -> Path:
        return self.processed_output_dir / f"date={self.target_date_iso}"

    def _get_processing_config(self):
        if self._processing_config is not None:
            return self._processing_config

        from src.data_pipeline.processors.unstructured.scheduler import ProcessingConfig

        self._processing_config = ProcessingConfig(
            raw_data_dir=str(self.output_dir),
            processed_data_dir=str(self.processed_output_dir),
            model_name=self.processing_model_name,
            ollama_host=self.processing_ollama_host,
            llm_timeout=self.processing_timeout,
            use_gpu=self.processing_use_gpu,
            skip_existing=self.processing_skip_existing,
        )
        return self._processing_config

    def _get_announcement_filter(self):
        if self._announcement_filter is not None:
            return self._announcement_filter

        from src.data_pipeline.processors.unstructured.filter import (
            AnnouncementFilter,
            FilterConfig,
        )

        filter_config = FilterConfig(
            use_gpu=self.processing_use_gpu,
            raw_data_dir=str(self.output_dir),
        )
        self._announcement_filter = AnnouncementFilter(config=filter_config)
        return self._announcement_filter

    @staticmethod
    def _to_processor_category(category_str: str):
        from src.data_pipeline.processors.unstructured.scheduler.base import DataCategory

        return DataCategory(category_str)

    def _get_pipeline(self, task: UnstructuredIncrementTask):
        processor_category = task.processor_category or task.full_reference_dir
        category = self._to_processor_category(processor_category)
        cache_key = category.value
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key], category

        from src.data_pipeline.processors.unstructured.scheduler.pipeline import create_pipeline

        pipeline = create_pipeline(category, self._get_processing_config())
        self._pipeline_cache[cache_key] = pipeline
        return pipeline, category

    def _get_collector(self, task: UnstructuredIncrementTask) -> Any:
        cache_key = f"{task.collector_module}.{task.collector_class}"
        if cache_key not in self._collector_instances:
            module = importlib.import_module(task.collector_module)
            cls = getattr(module, task.collector_class)
            self._collector_instances[cache_key] = cls()
        return self._collector_instances[cache_key]

    def _get_collect_func(self, task: UnstructuredIncrementTask) -> Callable:
        collector = self._get_collector(task)
        if not hasattr(collector, task.collector_func):
            raise AttributeError(
                f"采集器 {task.collector_class} 不存在函数 {task.collector_func}"
            )
        return getattr(collector, task.collector_func)

    def _get_events_task(self) -> Optional[UnstructuredIncrementTask]:
        task = TASK_NAME_MAP.get("events")
        if task and task.enabled:
            return task
        return None

    def _build_params(self, task: UnstructuredIncrementTask) -> Dict[str, Any]:
        params = dict(task.extra_params)
        if task.date_param_style == "compact":
            params["start_date"] = self.target_date_compact
            params["end_date"] = self.target_date_compact
        else:
            params["start_date"] = self.target_date_iso
            params["end_date"] = self.target_date_iso
        return params

    @staticmethod
    def _reference_sort_key(file_path: Path, root_dir: Path) -> Tuple[int, int, str]:
        """按 year/month 排序，选择最近月份作为 schema 参考。"""
        year_num = 0
        month_num = 0

        try:
            rel = file_path.relative_to(root_dir)
            # 典型结构: root/year/month.parquet
            if len(rel.parts) >= 2 and rel.parts[0].isdigit():
                year_num = int(rel.parts[0])
            stem = file_path.stem
            if stem.isdigit():
                month_num = int(stem)
        except Exception:
            pass

        return year_num, month_num, str(file_path)

    def _find_reference_file(self, task: UnstructuredIncrementTask) -> Path:
        ref_dir = self.full_raw_dir / task.full_reference_dir
        if not ref_dir.exists():
            raise FileNotFoundError(
                f"任务 {task.name} 对应全量参考目录不存在: {ref_dir}"
            )

        files = list(ref_dir.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"任务 {task.name} 对应全量参考文件不存在: {ref_dir}"
            )

        return max(files, key=lambda p: self._reference_sort_key(p, ref_dir))

    def _find_processed_reference_file(self, task: UnstructuredIncrementTask) -> Path:
        merged_name = task.processed_output_file or task.output_file
        merged_file = self.full_processed_dir / merged_name
        if merged_file.exists():
            return merged_file

        processed_ref_dir = task.full_processed_reference_dir or task.full_reference_dir
        ref_dir = self.full_processed_dir / processed_ref_dir
        if not ref_dir.exists():
            raise FileNotFoundError(
                f"任务 {task.name} 对应全量处理参考目录不存在: {ref_dir}"
            )

        files = list(ref_dir.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"任务 {task.name} 对应全量处理参考文件不存在: {ref_dir}"
            )

        return max(files, key=lambda p: self._reference_sort_key(p, ref_dir))

    def _get_reference_schema(self, reference_file: Path):
        schema_key = str(reference_file)
        if schema_key in self._schema_cache:
            return self._schema_cache[schema_key]

        import pyarrow.parquet as pq

        schema = pq.read_schema(reference_file)
        self._schema_cache[schema_key] = schema
        return schema

    def _get_processed_reference_schema(self, reference_file: Path):
        schema_key = str(reference_file)
        if schema_key in self._processed_schema_cache:
            return self._processed_schema_cache[schema_key]

        import pyarrow.parquet as pq

        schema = pq.read_schema(reference_file)
        self._processed_schema_cache[schema_key] = schema
        return schema

    def _filter_target_date(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        if df.empty or date_column not in df.columns:
            return df

        parsed = pd.to_datetime(df[date_column], errors="coerce")
        mask = parsed.dt.strftime("%Y-%m-%d") == self.target_date_iso
        return df.loc[mask].copy()

    @staticmethod
    def _align_columns(df: pd.DataFrame, reference_schema: Any) -> pd.DataFrame:
        ref_cols = [field.name for field in reference_schema]

        out = df.copy()
        for col in ref_cols:
            if col not in out.columns:
                out[col] = pd.NA

        extra_cols = [c for c in out.columns if c not in ref_cols]
        if extra_cols:
            out = out.drop(columns=extra_cols)

        return out[ref_cols]

    @staticmethod
    def _write_with_schema(df: pd.DataFrame, output_path: Path, reference_schema: Any):
        """按参考 schema 强制写 parquet，保证类型严格一致。"""
        import pyarrow as pa
        import pyarrow.parquet as pq

        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(
            df,
            schema=reference_schema,
            preserve_index=False,
            safe=False,
        )
        pq.write_table(table, output_path, compression="snappy")

    @staticmethod
    def _validate_schema(output_path: Path, reference_schema: Any):
        """校验输出 schema 与参考 schema 的字段名和类型完全一致。"""
        import pyarrow.parquet as pq

        out_schema = pq.read_schema(output_path)
        if len(out_schema) != len(reference_schema):
            raise ValueError("输出 schema 字段数量与参考 schema 不一致")

        for out_field, ref_field in zip(out_schema, reference_schema):
            if out_field.name != ref_field.name:
                raise ValueError(
                    f"输出字段名不一致: {out_field.name} != {ref_field.name}"
                )
            if out_field.type != ref_field.type:
                raise ValueError(
                    f"输出字段类型不一致: {out_field.name} -> {out_field.type} != {ref_field.type}"
                )

    def _collect_dataframe(self, task: UnstructuredIncrementTask) -> pd.DataFrame:
        collect_func = self._get_collect_func(task)
        params = self._build_params(task)

        df = collect_func(**params)
        if df is None:
            return pd.DataFrame()

        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                raise TypeError(f"任务 {task.name} 返回类型不是 DataFrame 且不可转换: {e}")

        return self._filter_target_date(df, task.date_column)

    def _ensure_increment_events_dataframe(self) -> pd.DataFrame:
        """获取公告过滤所需的当日events数据；优先读取当日events.parquet，不存在时按需生成。"""
        if self._increment_events_df is not None:
            return self._increment_events_df.copy()

        events_output_path = self.partition_dir / "events.parquet"
        if events_output_path.exists():
            self._increment_events_df = pd.read_parquet(events_output_path)
            return self._increment_events_df.copy()

        events_task = self._get_events_task()
        if events_task is None:
            logger.warning("公告增量过滤未找到可用events任务，将跳过事件过滤")
            self._increment_events_df = pd.DataFrame()
            return self._increment_events_df.copy()

        try:
            reference_file = self._find_reference_file(events_task)
            reference_schema = self._get_reference_schema(reference_file)

            events_df = self._collect_dataframe(events_task)
            events_df = self._align_columns(events_df, reference_schema)

            if not self.dry_run:
                self._write_with_schema(events_df, events_output_path, reference_schema)
                self._validate_schema(events_output_path, reference_schema)

            self._increment_events_df = events_df
            logger.info(
                "公告增量过滤依赖准备完成: %s, records=%s",
                events_output_path,
                len(events_df),
            )
            return self._increment_events_df.copy()
        except Exception as e:
            logger.warning("准备当日events.parquet失败，公告仅执行标题过滤: %s", e)
            self._increment_events_df = pd.DataFrame()
            return self._increment_events_df.copy()

    def _filter_announcements_for_processing(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        announcement_filter = self._get_announcement_filter()
        events_df = self._ensure_increment_events_dataframe()
        filtered_df, stats = announcement_filter.filter_increment_dataframe(
            announcements_df=raw_df,
            events_df=events_df,
        )

        logger.info(
            "公告增量预过滤完成: original=%s, after_event=%s, after_title=%s, final=%s, filter_rate=%.2f%%",
            stats.get("original_count", 0),
            stats.get("after_event_filter", 0),
            stats.get("after_title_filter", 0),
            stats.get("final_count", 0),
            float(stats.get("filter_rate", 0.0)) * 100,
        )
        return filtered_df

    def _build_processed_dataframe(
        self,
        task: UnstructuredIncrementTask,
        raw_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        pipeline, category = self._get_pipeline(task)
        records = raw_df.to_dict("records")

        from src.data_pipeline.processors.unstructured.scheduler.base import DataCategory

        if category == DataCategory.EXCHANGE:
            return pipeline.process_dataframe_gpu(raw_df, category)

        if category == DataCategory.CCTV:
            cctv_results = pipeline.process_batch(records, category)
            success_results = [r for r in cctv_results if r.success]
            if not success_results:
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    {
                        "date": r.date,
                        "id": r.record_id,
                        "market_sentiment": float(r.market_sentiment),
                        "beta_signal": float(r.beta_signal),
                        "keywords": ",".join(r.keywords) if r.keywords else "",
                        "tone_analysis": r.tone_analysis,
                    }
                    for r in success_results
                ]
            )

        if category in (DataCategory.POLICY_GOV, DataCategory.POLICY_NDRC):
            policy_results = pipeline.process_batch(records, category)
            success_results = [r for r in policy_results if r.success]
            if not success_results:
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    {
                        "date": r.date,
                        "id": r.record_id,
                        "summary": r.summary,
                        "benefited_industries": ",".join(r.benefited_industries) if r.benefited_industries else "",
                        "harmed_industries": ",".join(r.harmed_industries) if r.harmed_industries else "",
                        "industry_scores": json.dumps(r.industry_scores, ensure_ascii=False) if r.industry_scores else "{}",
                    }
                    for r in success_results
                ]
            )

        pdf_results = pipeline.process_batch(records, category)
        success_results = [r for r in pdf_results if r.success]
        if not success_results:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "id": r.record_id,
                    "ts_code": r.ts_code,
                    "date": r.date,
                    "score": float(r.score) if r.score is not None else 0.0,
                    "reason": r.reason,
                }
                for r in success_results
            ]
        )

    def _run_processing_for_task(
        self,
        task: UnstructuredIncrementTask,
        raw_df: pd.DataFrame,
    ) -> Tuple[str, int]:
        output_name = task.processed_output_file or task.output_file
        output_path = self.processed_partition_dir / output_name

        if self.processing_skip_existing and output_path.exists() and not self.dry_run:
            return str(output_path), 0

        reference_file = self._find_processed_reference_file(task)
        reference_schema = self._get_processed_reference_schema(reference_file)

        processing_input_df = raw_df
        if task.name == "announcements":
            processing_input_df = self._filter_announcements_for_processing(raw_df)

        processed_df = self._build_processed_dataframe(task, processing_input_df)
        processed_df = self._align_columns(processed_df, reference_schema)

        if self.dry_run:
            return str(output_path), len(processed_df)

        self._write_with_schema(processed_df, output_path, reference_schema)
        self._validate_schema(output_path, reference_schema)

        return str(output_path), len(processed_df)

    def _run_single_task(self, task: UnstructuredIncrementTask) -> IncrementTaskResult:
        output_path = self.partition_dir / task.output_file
        processed_output_path = self.processed_partition_dir / (task.processed_output_file or task.output_file)

        if self.skip_existing and output_path.exists() and not self.dry_run:
            if not self.enable_processing:
                return IncrementTaskResult(
                    task_name=task.name,
                    success=True,
                    output_path=str(output_path),
                    processing_output_path=None,
                    records_count=0,
                    skipped=True,
                )

            try:
                existing_df = pd.read_parquet(output_path)
                if task.name == "events":
                    self._increment_events_df = existing_df.copy()

                processing_path, processing_records = self._run_processing_for_task(task, existing_df)

                return IncrementTaskResult(
                    task_name=task.name,
                    success=True,
                    output_path=str(output_path),
                    processing_output_path=processing_path,
                    records_count=0,
                    processing_records_count=processing_records,
                    skipped=False,
                )
            except Exception as e:
                logger.error("任务失败 [%s]: %s", task.name, e)
                logger.debug(traceback.format_exc())
                return IncrementTaskResult(
                    task_name=task.name,
                    success=False,
                    output_path=str(output_path),
                    error_message=str(e),
                )

        try:
            reference_file = self._find_reference_file(task)
            reference_schema = self._get_reference_schema(reference_file)

            if task.name == "events" and self._increment_events_df is not None:
                df = self._increment_events_df.copy()
            else:
                df = self._collect_dataframe(task)
            df = self._align_columns(df, reference_schema)

            if task.name == "events":
                self._increment_events_df = df.copy()

            if self.dry_run:
                return IncrementTaskResult(
                    task_name=task.name,
                    success=True,
                    output_path=str(output_path),
                    records_count=len(df),
                )

            self._write_with_schema(df, output_path, reference_schema)
            self._validate_schema(output_path, reference_schema)

            processing_path = None
            processing_records = 0
            if self.enable_processing:
                processing_path, processing_records = self._run_processing_for_task(task, df)

            return IncrementTaskResult(
                task_name=task.name,
                success=True,
                output_path=str(output_path),
                records_count=len(df),
                processing_output_path=processing_path,
                processing_records_count=processing_records,
            )
        except Exception as e:
            logger.error("任务失败 [%s]: %s", task.name, e)
            logger.debug(traceback.format_exc())
            return IncrementTaskResult(
                task_name=task.name,
                success=False,
                output_path=str(output_path),
                error_message=str(e),
            )

    def run(self, task_names: Optional[List[str]] = None) -> IncrementCollectionReport:
        tasks = get_increment_tasks(task_names)

        # 过滤掉无效任务名
        if task_names:
            unknown = [n for n in task_names if n not in TASK_NAME_MAP]
            if unknown:
                logger.warning("存在未知任务名，将忽略: %s", ", ".join(unknown))

        report = IncrementCollectionReport(target_date=self.target_date_iso, total_tasks=len(tasks))

        logger.info("=" * 60)
        logger.info("非结构化增量采集开始: %s", self.target_date_iso)
        logger.info("任务数: %s", len(tasks))
        logger.info("输出分区: %s", self.partition_dir)
        logger.info("=" * 60)

        for idx, task in enumerate(tasks, 1):
            logger.info("[%s/%s] %s - %s", idx, len(tasks), task.name, task.description)
            result = self._run_single_task(task)
            report.add_result(result)

            if result.success and not result.skipped:
                if self.enable_processing:
                    logger.info(
                        "  ✓ 完成: %s, raw_records=%s, processed_records=%s",
                        task.name,
                        result.records_count,
                        result.processing_records_count,
                    )
                else:
                    logger.info("  ✓ 完成: %s, records=%s", task.name, result.records_count)
            elif result.skipped:
                logger.info("  ⏭ 跳过: %s", task.name)
            else:
                logger.error("  ✗ 失败: %s, error=%s", task.name, result.error_message)

        logger.info("=" * 60)
        logger.info(
            "非结构化增量采集完成: success=%s, failed=%s, skipped=%s, total_records=%s, total_processed_records=%s",
            report.success_tasks,
            report.failed_tasks,
            report.skipped_tasks,
            report.total_records,
            report.total_processed_records,
        )
        logger.info("=" * 60)

        return report

    @staticmethod
    def list_tasks() -> Dict[str, str]:
        """列出可用增量任务"""
        from .config import UNSTRUCTURED_INCREMENT_TASKS

        return {t.name: t.description for t in UNSTRUCTURED_INCREMENT_TASKS if t.enabled}
