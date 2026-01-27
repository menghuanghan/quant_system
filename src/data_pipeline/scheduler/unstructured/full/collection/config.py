"""
非结构化数据全量采集配置

定义五大非结构化数据类型的采集任务配置：
- announcements: 上市公司公告
- events: 事件驱动型数据（并购重组、处罚公告、实控人变更、重大合同）
- news: 新闻（CCTV新闻、交易所公告）
- policy: 政策（国务院、发改委）
- reports: 券商研报
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable


class DataType(Enum):
    """非结构化数据类型"""
    ANNOUNCEMENTS = "announcements"
    EVENTS = "events"
    NEWS = "news"
    POLICY = "policy"
    REPORTS = "reports"


class StockScope(Enum):
    """股票范围"""
    ALL_A = "all_a"           # 全A股
    NONE = "none"             # 不涉及股票
    CUSTOM = "custom"         # 自定义


class StoragePattern(Enum):
    """存储模式"""
    BY_STOCK = "by_stock"     # 按股票代码存储: year/month/stock_code.parquet
    BY_DATE = "by_date"       # 按日期存储: year/month.parquet
    SINGLE_FILE = "single"    # 单文件存储


@dataclass
class CollectionTask:
    """采集任务配置"""
    name: str                          # 任务名称
    data_type: DataType               # 数据类型
    description: str                   # 任务描述
    collector_module: str              # 采集器模块路径
    collector_class: str               # 采集器类名
    collector_func: str                # 采集函数名
    storage_pattern: StoragePattern    # 存储模式
    stock_scope: StockScope = StockScope.NONE  # 股票范围
    needs_stock_list: bool = False    # 采集器是否需要股票列表参数（即使按月份存储）
    output_subdir: str = ""           # 输出子目录
    params: Dict[str, Any] = field(default_factory=dict)  # 额外参数
    enabled: bool = True              # 是否启用
    priority: int = 5                 # 优先级（1-10，越小越高）
    batch_size: int = 30              # 每批股票数量（适用于ALL_A）
    rate_limit_delay: float = 0.5     # 请求间隔（秒）


# ====================== 任务定义 ======================

# 公告采集任务
ANNOUNCEMENT_TASKS = [
    CollectionTask(
        name="cninfo_announcements",
        data_type=DataType.ANNOUNCEMENTS,
        description="巨潮资讯上市公司公告采集",
        collector_module="src.data_pipeline.collectors.unstructured.announcements",
        collector_class="AnnouncementCollector",
        collector_func="collect_announcements",
        storage_pattern=StoragePattern.BY_DATE,  # 按月份存储
        stock_scope=StockScope.NONE,
        needs_stock_list=False,  # 采集器自行处理股票遍历
        output_subdir="",
        priority=3,
        batch_size=20,
        rate_limit_delay=1.0,
    ),
]

# 事件采集任务
EVENT_TASKS = [
    CollectionTask(
        name="cninfo_events",
        data_type=DataType.EVENTS,
        description="巨潮资讯事件驱动型数据采集（并购重组、处罚、实控人变更、重大合同）",
        collector_module="src.data_pipeline.collectors.unstructured.events",
        collector_class="CninfoEventCollector",
        collector_func="collect",
        storage_pattern=StoragePattern.BY_DATE,  # 改为按月份存储
        stock_scope=StockScope.NONE,  # 事件采集器自行处理股票遍历
        output_subdir="",
        priority=4,
        rate_limit_delay=0.5,
    ),
]

# 新闻采集任务
NEWS_TASKS = [
    CollectionTask(
        name="cctv_news",
        data_type=DataType.NEWS,
        description="央视新闻联播采集",
        collector_module="src.data_pipeline.collectors.unstructured.news",
        collector_class="CCTVNewsCollector",
        collector_func="collect",
        storage_pattern=StoragePattern.BY_DATE,
        stock_scope=StockScope.NONE,
        output_subdir="cctv",
        priority=2,
        rate_limit_delay=0.3,
        enabled=True,
    ),
    CollectionTask(
        name="exchange_news",
        data_type=DataType.NEWS,
        description="交易所官方公告采集（上交所、深交所）",
        collector_module="src.data_pipeline.collectors.unstructured.news",
        collector_class="OfficialExchangeNewsCrawler",
        collector_func="collect",
        storage_pattern=StoragePattern.BY_DATE,
        stock_scope=StockScope.NONE,
        output_subdir="exchange",
        priority=2,
        rate_limit_delay=0.5,
    ),
]

# 政策采集任务
POLICY_TASKS = [
    CollectionTask(
        name="gov_council_policy",
        data_type=DataType.POLICY,
        description="国务院政策文件采集",
        collector_module="src.data_pipeline.collectors.unstructured.policy",
        collector_class="GovCouncilCollector",
        collector_func="collect",
        storage_pattern=StoragePattern.BY_DATE,
        stock_scope=StockScope.NONE,
        output_subdir="gov",
        priority=2,
        rate_limit_delay=0.5,
    ),
    CollectionTask(
        name="ndrc_policy",
        data_type=DataType.POLICY,
        description="发改委政策文件采集",
        collector_module="src.data_pipeline.collectors.unstructured.policy",
        collector_class="NDRCCollector",
        collector_func="collect",
        storage_pattern=StoragePattern.BY_DATE,
        stock_scope=StockScope.NONE,
        output_subdir="ndrc",
        priority=2,
        rate_limit_delay=0.5,
    ),
]

# 研报采集任务
REPORT_TASKS = [
    CollectionTask(
        name="eastmoney_reports",
        data_type=DataType.REPORTS,
        description="东方财富券商研报采集",
        collector_module="src.data_pipeline.collectors.unstructured.reports",
        collector_class="EastMoneyReportCollector",
        collector_func="collect",
        storage_pattern=StoragePattern.BY_DATE,  # 按月份存储
        stock_scope=StockScope.NONE,
        needs_stock_list=True,  # 采集器需要股票列表参数
        output_subdir="",
        priority=3,
        batch_size=30,
        rate_limit_delay=0.5,
    ),
]

# 所有任务
ALL_TASKS = ANNOUNCEMENT_TASKS + EVENT_TASKS + NEWS_TASKS + POLICY_TASKS + REPORT_TASKS

# 按数据类型分组
TASKS_BY_TYPE: Dict[DataType, List[CollectionTask]] = {
    DataType.ANNOUNCEMENTS: ANNOUNCEMENT_TASKS,
    DataType.EVENTS: EVENT_TASKS,
    DataType.NEWS: NEWS_TASKS,
    DataType.POLICY: POLICY_TASKS,
    DataType.REPORTS: REPORT_TASKS,
}

# 数据类型名称映射
TYPE_NAMES: Dict[DataType, str] = {
    DataType.ANNOUNCEMENTS: "上市公司公告",
    DataType.EVENTS: "事件驱动型数据",
    DataType.NEWS: "新闻数据",
    DataType.POLICY: "政策文件",
    DataType.REPORTS: "券商研报",
}


def get_enabled_tasks() -> List[CollectionTask]:
    """获取所有启用的任务"""
    return [t for t in ALL_TASKS if t.enabled]


def get_tasks_by_type(data_type: DataType) -> List[CollectionTask]:
    """按数据类型获取任务"""
    return TASKS_BY_TYPE.get(data_type, [])


def get_tasks_sorted_by_priority() -> List[CollectionTask]:
    """获取按优先级排序的任务"""
    enabled = get_enabled_tasks()
    return sorted(enabled, key=lambda x: x.priority)
