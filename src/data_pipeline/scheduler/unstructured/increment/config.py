"""
非结构化数据增量采集配置

目标：
- 面向单日增量采集（给定 date）
- 输出目录按 date 分区：data/raw/unstructured_increment/date=YYYY-MM-DD/
- 增量产物与全量 raw/unstructured 对应类型保持字段与数据类型一致
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class UnstructuredIncrementTask:
    """非结构化增量采集任务定义"""

    name: str
    description: str
    collector_module: str
    collector_class: str
    collector_func: str
    output_file: str
    full_reference_dir: str
    processor_category: str = ""
    full_processed_reference_dir: str = ""
    processed_output_file: Optional[str] = None
    date_column: str = "date"
    date_param_style: str = "iso"  # iso: YYYY-MM-DD, compact: YYYYMMDD
    extra_params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


UNSTRUCTURED_INCREMENT_TASKS: List[UnstructuredIncrementTask] = [
    UnstructuredIncrementTask(
        name="announcements",
        description="全市场公告增量",
        collector_module="src.data_pipeline.collectors.unstructured.announcements",
        collector_class="AnnouncementCollector",
        collector_func="collect_announcements",
        output_file="announcements.parquet",
        full_reference_dir="announcements",
        processor_category="announcements",
        full_processed_reference_dir="announcements",
        processed_output_file="announcements.parquet",
        date_column="date",
        date_param_style="iso",
    ),
    UnstructuredIncrementTask(
        name="events",
        description="全市场事件增量",
        collector_module="src.data_pipeline.collectors.unstructured.events",
        collector_class="CninfoEventCollector",
        collector_func="collect",
        output_file="events.parquet",
        full_reference_dir="events",
        processor_category="events",
        full_processed_reference_dir="events",
        processed_output_file="events.parquet",
        date_column="date",
        date_param_style="compact",
    ),
    UnstructuredIncrementTask(
        name="news_cctv",
        description="CCTV 新闻增量",
        collector_module="src.data_pipeline.collectors.unstructured.news",
        collector_class="CCTVNewsCollector",
        collector_func="collect",
        output_file="news_cctv.parquet",
        full_reference_dir="news/cctv",
        processor_category="news/cctv",
        full_processed_reference_dir="news/cctv",
        processed_output_file="news_cctv.parquet",
        date_column="date",
        date_param_style="iso",
    ),
    UnstructuredIncrementTask(
        name="news_exchange",
        description="交易所公告增量",
        collector_module="src.data_pipeline.collectors.unstructured.news",
        collector_class="OfficialExchangeNewsCrawler",
        collector_func="collect",
        output_file="news_exchange.parquet",
        full_reference_dir="news/exchange",
        processor_category="news/exchange",
        full_processed_reference_dir="news/exchange",
        processed_output_file="news_exchange.parquet",
        date_column="date",
        date_param_style="iso",
    ),
    UnstructuredIncrementTask(
        name="policy_gov",
        description="国务院政策增量",
        collector_module="src.data_pipeline.collectors.unstructured.policy",
        collector_class="GovCouncilCollector",
        collector_func="collect",
        output_file="policy_gov.parquet",
        full_reference_dir="policy/gov",
        processor_category="policy/gov",
        full_processed_reference_dir="policy/gov",
        processed_output_file="policy_gov.parquet",
        date_column="date",
        date_param_style="compact",
    ),
    UnstructuredIncrementTask(
        name="policy_ndrc",
        description="发改委政策增量",
        collector_module="src.data_pipeline.collectors.unstructured.policy",
        collector_class="NDRCCollector",
        collector_func="collect",
        output_file="policy_ndrc.parquet",
        full_reference_dir="policy/ndrc",
        processor_category="policy/ndrc",
        full_processed_reference_dir="policy/ndrc",
        processed_output_file="policy_ndrc.parquet",
        date_column="date",
        date_param_style="compact",
    ),
    UnstructuredIncrementTask(
        name="reports",
        description="全市场研报增量",
        collector_module="src.data_pipeline.collectors.unstructured.reports",
        collector_class="EastMoneyReportCollector",
        collector_func="collect",
        output_file="reports.parquet",
        full_reference_dir="reports",
        processor_category="reports",
        full_processed_reference_dir="reports",
        processed_output_file="reports.parquet",
        date_column="date",
        date_param_style="iso",
    ),
]


TASK_NAME_MAP: Dict[str, UnstructuredIncrementTask] = {
    t.name: t for t in UNSTRUCTURED_INCREMENT_TASKS
}


def get_increment_tasks(task_names: Optional[List[str]] = None) -> List[UnstructuredIncrementTask]:
    """获取增量任务列表（不传 task_names 时返回全部启用任务）"""
    tasks = [t for t in UNSTRUCTURED_INCREMENT_TASKS if t.enabled]
    if not task_names:
        return tasks

    selected = []
    for name in task_names:
        task = TASK_NAME_MAP.get(name)
        if task and task.enabled:
            selected.append(task)
    return selected
