"""
调度配置模块 (Scheduler Configuration)

提供：
- 时间槽生成（月度切片，倒序）
- 策略路由（热数据 vs 冷数据）
- 采集器注册与配置
- 资源限制参数

配置策略：
- 热数据 (2023-2026): 全详情，保存 PDF，高并发
- 冷数据 (2021-2022): 仅文本模式，跳过图片，低并发
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


# ============== 数据温度分级 ==============

class DataTemperature(str, Enum):
    """数据温度分级"""
    HOT = "hot"       # 热数据：最近2年，高优先级，保留原始文件
    WARM = "warm"     # 温数据：3-4年前，中等优先级
    COLD = "cold"     # 冷数据：5年以上，低优先级，仅保留文本


@dataclass
class TemperaturePolicy:
    """温度策略配置"""
    temperature: DataTemperature
    
    # 采集配置
    save_pdf: bool = False           # 是否保存 PDF 原文件
    text_only_mode: bool = False     # 仅提取文本
    skip_images: bool = False        # 跳过图片处理
    ocr_enabled: bool = True         # 是否启用 OCR
    
    # 资源配置
    max_workers: int = 4             # 最大并发数
    batch_size: int = 100            # 批处理大小
    request_delay: float = 1.0       # 请求间隔（秒）
    timeout: int = 30                # 超时时间（秒）
    
    # 重试配置
    max_retries: int = 3             # 最大重试次数
    retry_delay: float = 60.0        # 重试延迟（秒）


# 预定义温度策略
TEMPERATURE_POLICIES: Dict[DataTemperature, TemperaturePolicy] = {
    DataTemperature.HOT: TemperaturePolicy(
        temperature=DataTemperature.HOT,
        save_pdf=True,
        text_only_mode=False,
        skip_images=False,
        ocr_enabled=True,
        max_workers=6,
        batch_size=50,
        request_delay=0.5,
        timeout=60,
        max_retries=3,
        retry_delay=30.0
    ),
    DataTemperature.WARM: TemperaturePolicy(
        temperature=DataTemperature.WARM,
        save_pdf=False,
        text_only_mode=False,
        skip_images=False,
        ocr_enabled=True,
        max_workers=4,
        batch_size=100,
        request_delay=1.0,
        timeout=45,
        max_retries=3,
        retry_delay=60.0
    ),
    DataTemperature.COLD: TemperaturePolicy(
        temperature=DataTemperature.COLD,
        save_pdf=False,
        text_only_mode=True,
        skip_images=True,
        ocr_enabled=False,
        max_workers=2,
        batch_size=200,
        request_delay=2.0,
        timeout=30,
        max_retries=2,
        retry_delay=120.0
    )
}


def get_temperature_for_date(dt: date) -> DataTemperature:
    """
    根据日期确定数据温度
    
    规则：
    - 最近2年: HOT
    - 3-4年前: WARM
    - 5年以上: COLD
    """
    now = datetime.now().date()
    years_ago = (now - dt).days / 365.25
    
    if years_ago <= 2:
        return DataTemperature.HOT
    elif years_ago <= 4:
        return DataTemperature.WARM
    else:
        return DataTemperature.COLD


def get_policy_for_month(month: str) -> TemperaturePolicy:
    """
    根据月份获取温度策略
    
    Args:
        month: 月份字符串 (YYYY-MM)
    
    Returns:
        对应的温度策略
    """
    dt = datetime.strptime(month, '%Y-%m').date()
    temp = get_temperature_for_date(dt)
    return TEMPERATURE_POLICIES[temp]


# ============== 时间槽生成 ==============

@dataclass
class TimeSlot:
    """时间槽"""
    month: str               # YYYY-MM 格式
    start_date: str          # YYYYMMDD 格式
    end_date: str            # YYYYMMDD 格式
    temperature: DataTemperature
    policy: TemperaturePolicy
    
    @property
    def year(self) -> int:
        return int(self.month.split('-')[0])
    
    @property
    def month_num(self) -> int:
        return int(self.month.split('-')[1])


def generate_time_slots(
    start_year: int = 2021,
    end_year: int = 2025,
    end_month: Optional[int] = None,
    reverse: bool = True
) -> List[TimeSlot]:
    """
    生成月度时间槽
    
    默认倒序生成（从最新到最旧），优先采集热数据。
    
    Args:
        start_year: 开始年份
        end_year: 结束年份
        end_month: 结束月份（默认当前月或12月）
        reverse: 是否倒序
    
    Returns:
        时间槽列表
    
    Example:
        >>> slots = generate_time_slots(2021, 2025)
        >>> print(slots[0].month)  # 2025-12 或当前月
        >>> print(slots[-1].month)  # 2021-01
    """
    slots = []
    now = datetime.now()
    
    # 确定结束月份
    if end_year == now.year:
        final_month = end_month or now.month
    else:
        final_month = end_month or 12
    
    # 生成所有月份
    current = datetime(start_year, 1, 1)
    end = datetime(end_year, final_month, 1)
    
    while current <= end:
        month_str = current.strftime('%Y-%m')
        
        # 计算月末日期
        next_month = current + relativedelta(months=1)
        month_end = next_month - relativedelta(days=1)
        
        start_date = current.strftime('%Y%m%d')
        end_date = month_end.strftime('%Y%m%d')
        
        temp = get_temperature_for_date(current.date())
        policy = TEMPERATURE_POLICIES[temp]
        
        slots.append(TimeSlot(
            month=month_str,
            start_date=start_date,
            end_date=end_date,
            temperature=temp,
            policy=policy
        ))
        
        current = next_month
    
    if reverse:
        slots.reverse()
    
    logger.info(f"生成时间槽: {len(slots)} 个月 ({start_year}-01 ~ {end_year}-{final_month:02d})")
    
    return slots


# ============== 采集器注册表 ==============

@dataclass
class CollectorConfig:
    """采集器配置"""
    name: str                          # 采集器名称
    collector_class: type              # 采集器类
    enabled: bool = True               # 是否启用
    priority: int = 100                # 优先级（数值小优先）
    
    # 特殊处理标志
    requires_market_data: bool = False  # 需要市场行情数据（如情感分析）
    memory_intensive: bool = False      # 内存密集型（如公告）
    rate_limit_sensitive: bool = False  # 对频率限制敏感
    
    # 覆盖配置
    override_policy: Optional[Dict[str, Any]] = None  # 覆盖温度策略
    
    def get_effective_policy(self, base_policy: TemperaturePolicy) -> Dict[str, Any]:
        """获取有效策略（合并覆盖配置）"""
        policy_dict = {
            'save_pdf': base_policy.save_pdf,
            'text_only_mode': base_policy.text_only_mode,
            'skip_images': base_policy.skip_images,
            'ocr_enabled': base_policy.ocr_enabled,
            'max_workers': base_policy.max_workers,
            'batch_size': base_policy.batch_size,
            'request_delay': base_policy.request_delay,
            'timeout': base_policy.timeout,
            'max_retries': base_policy.max_retries,
            'retry_delay': base_policy.retry_delay
        }
        
        if self.override_policy:
            policy_dict.update(self.override_policy)
        
        return policy_dict


class CollectorRegistry:
    """
    采集器注册表
    
    管理所有可用的非结构化数据采集器，提供统一的配置和获取接口。
    
    Example:
        >>> registry = CollectorRegistry()
        >>> 
        >>> # 注册采集器
        >>> registry.register(CollectorConfig(
        ...     name='news_sina',
        ...     collector_class=StreamingNewsCollector,
        ...     priority=10
        ... ))
        >>> 
        >>> # 获取采集器
        >>> configs = registry.get_enabled_collectors()
    """
    
    _instance: Optional['CollectorRegistry'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._collectors: Dict[str, CollectorConfig] = {}
        return cls._instance
    
    def register(self, config: CollectorConfig):
        """注册采集器"""
        self._collectors[config.name] = config
        logger.debug(f"注册采集器: {config.name}")
    
    def unregister(self, name: str):
        """注销采集器"""
        if name in self._collectors:
            del self._collectors[name]
    
    def get(self, name: str) -> Optional[CollectorConfig]:
        """获取采集器配置"""
        return self._collectors.get(name)
    
    def get_enabled_collectors(self) -> List[CollectorConfig]:
        """获取所有启用的采集器（按优先级排序）"""
        enabled = [c for c in self._collectors.values() if c.enabled]
        return sorted(enabled, key=lambda c: c.priority)
    
    def get_all(self) -> Dict[str, CollectorConfig]:
        """获取所有采集器配置"""
        return self._collectors.copy()
    
    def enable(self, name: str):
        """启用采集器"""
        if name in self._collectors:
            self._collectors[name].enabled = True
    
    def disable(self, name: str):
        """禁用采集器"""
        if name in self._collectors:
            self._collectors[name].enabled = False


# ============== 调度配置 ==============

@dataclass
class SchedulerConfig:
    """调度器全局配置"""
    
    # 时间范围
    start_year: int = 2021
    end_year: int = 2025
    
    # 并发控制
    global_max_workers: int = 8       # 全局最大并发
    per_source_workers: int = 4       # 单数据源最大并发
    
    # 存储配置
    output_base_dir: Path = Path("data/raw/unstructured")
    state_dir: Path = Path("data/state")
    temp_dir: Path = Path("data/temp")
    
    # 限制配置
    max_storage_gb: float = 500.0     # 最大存储（GB）
    max_memory_gb: float = 8.0        # 最大内存（GB）
    
    # 熔断配置
    circuit_break_threshold: int = 5  # 连续失败次数触发熔断
    circuit_break_duration: int = 300 # 熔断持续时间（秒）
    
    # 验证配置
    validate_output: bool = True      # 是否验证输出
    min_records_per_month: int = 10   # 每月最小记录数
    
    # 日志配置
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: Path = Path("logs/scheduler.log")
    
    # 依赖配置
    market_data_path: Path = Path("data/raw/structured/market_data/daily_prices.parquet")
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保目录存在
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        if self.log_to_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


# ============== 预定义采集器配置 ==============

def get_default_collector_configs() -> List[CollectorConfig]:
    """
    获取默认采集器配置
    
    注：实际的采集器类需要在使用时动态导入，避免循环依赖
    """
    # 延迟导入避免循环依赖
    configs = []
    
    # 新闻采集器
    configs.append(CollectorConfig(
        name='news_sina',
        collector_class=None,  # 运行时设置
        enabled=True,
        priority=10,
        requires_market_data=False,
        memory_intensive=False,
        rate_limit_sensitive=True
    ))
    
    # 公告采集器
    configs.append(CollectorConfig(
        name='announcement_cninfo',
        collector_class=None,
        enabled=True,
        priority=20,
        requires_market_data=False,
        memory_intensive=True,
        rate_limit_sensitive=True,
        override_policy={
            'max_workers': 2,  # 降低并发，避免内存溢出
            'batch_size': 30
        }
    ))
    
    # 研报采集器（如果有）
    configs.append(CollectorConfig(
        name='research_report',
        collector_class=None,
        enabled=False,  # 默认禁用
        priority=30,
        requires_market_data=False,
        memory_intensive=True,
        rate_limit_sensitive=True
    ))
    
    # 情感分析采集器（依赖市场数据）
    configs.append(CollectorConfig(
        name='sentiment',
        collector_class=None,
        enabled=False,  # 默认禁用
        priority=100,  # 最后执行
        requires_market_data=True,
        memory_intensive=False,
        rate_limit_sensitive=False
    ))
    
    return configs


def register_default_collectors():
    """注册默认采集器到全局注册表"""
    registry = CollectorRegistry()
    for config in get_default_collector_configs():
        registry.register(config)


# ============== 辅助函数 ==============

def get_output_path(
    base_dir: Path,
    source_name: str,
    month: str,
    format: str = 'parquet'
) -> Path:
    """
    获取输出文件路径
    
    结构: {base_dir}/{source_name}/{year}/{source_name}_{month}.parquet
    
    Example:
        >>> path = get_output_path(Path('data'), 'news', '2024-12')
        >>> # data/news/2024/news_2024-12.parquet
    """
    year = month.split('-')[0]
    filename = f"{source_name}_{month}.{format}"
    return base_dir / source_name / year / filename


def validate_parquet_file(
    file_path: Path,
    min_records: int = 10,
    required_columns: Optional[List[str]] = None
) -> tuple:
    """
    验证 Parquet 文件
    
    Args:
        file_path: 文件路径
        min_records: 最小记录数
        required_columns: 必需的列
    
    Returns:
        (is_valid, error_message)
    """
    try:
        import pyarrow.parquet as pq
        
        if not file_path.exists():
            return False, f"文件不存在: {file_path}"
        
        # 读取元数据
        parquet_file = pq.ParquetFile(file_path)
        num_rows = parquet_file.metadata.num_rows
        
        if num_rows < min_records:
            return False, f"记录数不足: {num_rows} < {min_records}"
        
        # 检查必需列
        if required_columns:
            schema_names = [f.name for f in parquet_file.schema_arrow]
            missing = set(required_columns) - set(schema_names)
            if missing:
                return False, f"缺少列: {missing}"
        
        # 检查文件大小
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "文件为空"
        
        return True, None
        
    except Exception as e:
        return False, f"验证失败: {e}"


def estimate_storage_usage(base_dir: Path) -> Dict[str, Any]:
    """
    估算存储使用情况
    
    Returns:
        {
            'total_size_gb': 10.5,
            'file_count': 120,
            'by_source': {'news': 5.2, 'announcement': 5.3}
        }
    """
    stats = {
        'total_size_gb': 0.0,
        'file_count': 0,
        'by_source': {}
    }
    
    if not base_dir.exists():
        return stats
    
    for source_dir in base_dir.iterdir():
        if not source_dir.is_dir():
            continue
        
        source_name = source_dir.name
        source_size = 0.0
        
        for file_path in source_dir.rglob('*.parquet'):
            source_size += file_path.stat().st_size
            stats['file_count'] += 1
        
        source_size_gb = source_size / (1024 ** 3)
        stats['by_source'][source_name] = source_size_gb
        stats['total_size_gb'] += source_size_gb
    
    return stats
