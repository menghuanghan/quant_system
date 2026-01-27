"""
原始数据采集调度器

提供FullCollectionScheduler类，统一调度十大数据域的原始数据采集工作
"""

import os
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import pandas as pd

from .config import (
    CollectionTask,
    DataCategory,
    StockScope,
    CollectionFrequency,
    ALL_TASKS,
    TASKS_BY_DOMAIN,
    DOMAIN_NAMES,
    get_enabled_tasks,
    get_tasks_sorted_by_priority,
    get_tasks_by_domain,
)

# 配置日志
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """任务执行结果"""
    task_name: str
    domain: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    records_count: int = 0
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    @property
    def duration_seconds(self) -> float:
        """执行耗时（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_name": self.task_name,
            "domain": self.domain,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "records_count": self.records_count,
            "output_path": self.output_path,
            "error_message": self.error_message,
        }


@dataclass
class CollectionProgress:
    """采集进度"""
    total_tasks: int = 0
    completed_tasks: int = 0
    success_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    current_task: Optional[str] = None
    current_domain: Optional[str] = None
    start_time: Optional[datetime] = None
    results: List[TaskResult] = field(default_factory=list)
    
    @property
    def progress_percent(self) -> float:
        """完成百分比"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    def add_result(self, result: TaskResult):
        """添加任务结果"""
        self.results.append(result)
        self.completed_tasks += 1
        if result.status == TaskStatus.SUCCESS:
            self.success_tasks += 1
        elif result.status == TaskStatus.FAILED:
            self.failed_tasks += 1
        elif result.status == TaskStatus.SKIPPED:
            self.skipped_tasks += 1


class FullCollectionScheduler:
    """
    全量数据采集调度器
    
    负责统一调度十大数据域的原始数据采集，支持：
    - 按数据域分批采集
    - 按优先级执行任务
    - 断点续采（记录采集进度）
    - 采集结果存储为parquet格式
    
    使用示例：
        scheduler = FullCollectionScheduler(
            start_date='20230101',
            end_date='20251231',
            output_dir='data/raw/structured'
        )
        
        # 采集全部数据域
        scheduler.run_all()
        
        # 仅采集指定数据域
        scheduler.run_domain('metadata')
        scheduler.run_domain('market_data')
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        output_dir: str = "data/raw/structured",
        max_workers: int = 1,
        retry_failed: bool = True,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        skip_existing: bool = False,
        progress_file: Optional[str] = None,
    ):
        """
        初始化全量采集调度器
        
        Args:
            start_date: 数据采集开始日期（YYYYMMDD格式）
            end_date: 数据采集结束日期（YYYYMMDD格式），默认为今天
            output_dir: 数据输出目录
            max_workers: 最大并发数（建议保持1，避免API限流）
            retry_failed: 是否重试失败的任务
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            skip_existing: 是否跳过已存在的文件
            progress_file: 进度文件路径（用于断点续采）
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y%m%d')
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.skip_existing = skip_existing
        self.progress_file = progress_file
        
        # 采集进度
        self.progress = CollectionProgress()
        
        # 采集器函数映射（懒加载）
        self._collector_funcs: Dict[str, Callable] = {}
        
        # 证券列表缓存
        self._stock_list_cache: Optional[pd.DataFrame] = None
        self._fund_list_cache: Optional[pd.DataFrame] = None
        self._index_list_cache: Optional[pd.DataFrame] = None
        self._option_list_cache: Optional[pd.DataFrame] = None
        self._bond_list_cache: Optional[pd.DataFrame] = None
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"全量采集调度器初始化完成: "
            f"日期范围={start_date}~{self.end_date}, "
            f"输出目录={output_dir}"
        )
    
    def _load_collector_funcs(self):
        """加载所有采集器函数"""
        if self._collector_funcs:
            return
        
        logger.info("加载数据采集器...")
        
        # 导入所有数据域的采集器
        try:
            # 元数据域
            from src.data_pipeline.collectors.structured.metadata import (
                get_stock_list_a, get_stock_list_hk, get_stock_list_us,
                get_name_change, get_st_status, get_ah_stock,
                get_trade_calendar, get_suspend_info, 
                get_price_limit_rule, get_auction_time,
            )
            self._collector_funcs.update({
                "get_stock_list_a": get_stock_list_a,
                "get_stock_list_hk": get_stock_list_hk,
                "get_stock_list_us": get_stock_list_us,
                "get_name_change": get_name_change,
                "get_st_status": get_st_status,
                "get_ah_stock": get_ah_stock,
                "get_trade_calendar": get_trade_calendar,
                "get_suspend_info": get_suspend_info,
                "get_price_limit_rule": get_price_limit_rule,
                "get_auction_time": get_auction_time,
            })
        except ImportError as e:
            logger.warning(f"元数据域采集器导入失败: {e}")
        
        try:
            # 市场行情域
            from src.data_pipeline.collectors.structured.market_data import (
                get_stock_daily, get_stock_weekly, get_stock_monthly,
                get_index_daily, get_etf_daily,
                get_daily_basic, get_stk_factor,
                get_realtime_quote, get_top_list as get_top_list_realtime,
                get_adj_factor,
            )
            self._collector_funcs.update({
                "get_stock_daily": get_stock_daily,
                "get_stock_weekly": get_stock_weekly,
                "get_stock_monthly": get_stock_monthly,
                "get_index_daily": get_index_daily,
                "get_etf_daily": get_etf_daily,
                "get_daily_basic": get_daily_basic,
                "get_stk_factor": get_stk_factor,
                "get_realtime_quote": get_realtime_quote,
                "get_adj_factor": get_adj_factor,
            })
        except ImportError as e:
            logger.warning(f"市场行情域采集器导入失败: {e}")
        
        try:
            # 基本面域
            from src.data_pipeline.collectors.structured.fundamental import (
                get_company_info, get_industry_class, 
                get_main_business, get_management,
                get_balance_sheet, get_income_statement, 
                get_cash_flow, get_financial_indicator,
                get_share_structure, get_top10_holders,
                get_pledge, get_share_float, get_repurchase, get_dividend,
            )
            self._collector_funcs.update({
                "get_company_info": get_company_info,
                "get_industry_class": get_industry_class,
                "get_main_business": get_main_business,
                "get_management": get_management,
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
            })
        except ImportError as e:
            logger.warning(f"基本面域采集器导入失败: {e}")
        
        try:
            # 资金与交易行为域
            from src.data_pipeline.collectors.structured.trading_behavior import (
                get_money_flow, get_money_flow_industry, 
                get_money_flow_market, get_hsgt_flow,
                get_margin_summary, get_margin_detail, 
                get_margin_target, get_slb,
                get_top_list, get_top_inst, get_block_trade,
            )
            self._collector_funcs.update({
                "get_money_flow": get_money_flow,
                "get_money_flow_industry": get_money_flow_industry,
                "get_money_flow_market": get_money_flow_market,
                "get_hsgt_flow": get_hsgt_flow,
                "get_margin_summary": get_margin_summary,
                "get_margin_detail": get_margin_detail,
                "get_margin_target": get_margin_target,
                "get_slb": get_slb,
                "get_top_list": get_top_list,
                "get_top_inst": get_top_inst,
                "get_block_trade": get_block_trade,
            })
        except ImportError as e:
            logger.warning(f"交易行为域采集器导入失败: {e}")
        
        try:
            # 板块/行业/主题域
            from src.data_pipeline.collectors.structured.cross_sectional import (
                get_sw_index_classify, get_sw_index_member, get_sw_daily,
                get_ths_index, get_ths_member, get_ths_daily,
                get_dc_index, get_dc_member,
                get_kpl_concept, get_kpl_concept_cons,
                get_sector_performance, get_industry_board_em, get_concept_board_em,
                get_sector_hist, get_sector_rank, get_limit_up_pool,
            )
            self._collector_funcs.update({
                "get_sw_index_classify": get_sw_index_classify,
                "get_sw_index_member": get_sw_index_member,
                "get_sw_daily": get_sw_daily,
                "get_ths_index": get_ths_index,
                "get_ths_member": get_ths_member,
                "get_ths_daily": get_ths_daily,
                "get_dc_index": get_dc_index,
                "get_dc_member": get_dc_member,
                "get_kpl_concept": get_kpl_concept,
                "get_kpl_concept_cons": get_kpl_concept_cons,
                "get_sector_performance": get_sector_performance,
                "get_industry_board_em": get_industry_board_em,
                "get_concept_board_em": get_concept_board_em,
                "get_sector_hist": get_sector_hist,
                "get_sector_rank": get_sector_rank,
                "get_limit_up_pool": get_limit_up_pool,
            })
        except ImportError as e:
            logger.warning(f"板块/行业域采集器导入失败: {e}")
        
        try:
            # 衍生品域
            from src.data_pipeline.collectors.structured.derivatives import (
                get_fund_basic, get_fund_daily, get_fund_nav,
                get_fund_portfolio, get_fund_share,
                get_fund_adj,
                get_fut_basic, get_fut_daily, get_fut_holding, get_fut_wsr,
                get_opt_basic, get_opt_daily,
                get_yield_curve, get_cb_basic, get_cb_daily, 
                get_repo_daily, get_cb_premium,
            )
            self._collector_funcs.update({
                "get_fund_basic": get_fund_basic,
                "get_fund_daily": get_fund_daily,
                "get_fund_nav": get_fund_nav,
                "get_fund_portfolio": get_fund_portfolio,
                "get_fund_share": get_fund_share,
                "get_fund_adj": get_fund_adj,
                "get_fut_basic": get_fut_basic,
                "get_fut_daily": get_fut_daily,
                "get_fut_holding": get_fut_holding,
                "get_fut_wsr": get_fut_wsr,
                "get_opt_basic": get_opt_basic,
                "get_opt_daily": get_opt_daily,
                "get_yield_curve": get_yield_curve,
                "get_cb_basic": get_cb_basic,
                "get_cb_daily": get_cb_daily,
                "get_repo_daily": get_repo_daily,
                "get_cb_premium": get_cb_premium,
            })
        except ImportError as e:
            logger.warning(f"衍生品域采集器导入失败: {e}")
        
        try:
            # 指数与基准域
            from src.data_pipeline.collectors.structured.index_benchmark import (
                get_index_basic, get_index_daily, 
                get_index_weight, get_index_member,
                get_index_global,
            )
            self._collector_funcs.update({
                "get_index_basic": get_index_basic,
                "get_index_daily": get_index_daily,
                "get_index_weight": get_index_weight,
                "get_index_member": get_index_member,
                "get_index_global": get_index_global,
            })
        except ImportError as e:
            logger.warning(f"指数与基准域采集器导入失败: {e}")
        
        try:
            # 宏观与外生变量域
            from src.data_pipeline.collectors.structured.macro_exogenous import (
                get_cn_gdp, get_cn_cpi, get_cn_ppi, get_cn_pmi,
                get_cn_m2, get_shibor, get_lpr, get_sf,
                get_us_treasury, get_eco_calendar,
                get_box_office, get_car_sales,
            )
            self._collector_funcs.update({
                "get_cn_gdp": get_cn_gdp,
                "get_cn_cpi": get_cn_cpi,
                "get_cn_ppi": get_cn_ppi,
                "get_cn_pmi": get_cn_pmi,
                "get_cn_m2": get_cn_m2,
                "get_shibor": get_shibor,
                "get_lpr": get_lpr,
                "get_sf": get_sf,
                "get_us_treasury": get_us_treasury,
                "get_eco_calendar": get_eco_calendar,
                "get_box_office": get_box_office,
                "get_car_sales": get_car_sales,
            })
        except ImportError as e:
            logger.warning(f"宏观与外生变量域采集器导入失败: {e}")
        
        # try:
        #     # 预期与预测分析域
        #     from src.data_pipeline.collectors.structured.expectations import (
        #         get_earnings_forecast, get_consensus_forecast,
        #         get_inst_rating, get_rating_summary, get_inst_survey,
        #         get_analyst_rank, get_analyst_detail, 
        #         get_broker_gold_stock, get_forecast_revision,
        #     )
        #     self._collector_funcs.update({
        #         "get_earnings_forecast": get_earnings_forecast,
        #         "get_consensus_forecast": get_consensus_forecast,
        #         "get_inst_rating": get_inst_rating,
        #         "get_rating_summary": get_rating_summary,
        #         "get_inst_survey": get_inst_survey,
        #         "get_analyst_rank": get_analyst_rank,
        #         "get_analyst_detail": get_analyst_detail,
        #         "get_broker_gold_stock": get_broker_gold_stock,
        #         "get_forecast_revision": get_forecast_revision,
        #     })
        # except ImportError as e:
        #     logger.warning(f"预期与预测分析域采集器导入失败: {e}")
        
        try:
            # 深度风险与质量因子域
            from src.data_pipeline.collectors.structured.deep_risk_quality import (
                get_a_pe_pb_ew_median, get_market_congestion,
                get_stock_bond_spread, get_buffett_indicator,
                get_stock_goodwill, get_goodwill_impairment, get_break_net_stock,
                get_esg_msci, get_esg_hz, get_esg_refinitiv, get_esg_zhiding,
            )
            self._collector_funcs.update({
                "get_a_pe_pb_ew_median": get_a_pe_pb_ew_median,
                "get_market_congestion": get_market_congestion,
                "get_stock_bond_spread": get_stock_bond_spread,
                "get_buffett_indicator": get_buffett_indicator,
                "get_stock_goodwill": get_stock_goodwill,
                "get_goodwill_impairment": get_goodwill_impairment,
                "get_break_net_stock": get_break_net_stock,
                "get_esg_msci": get_esg_msci,
                "get_esg_hz": get_esg_hz,
                "get_esg_refinitiv": get_esg_refinitiv,
                "get_esg_zhiding": get_esg_zhiding,
            })
        except ImportError as e:
            logger.warning(f"深度风险与质量因子域采集器导入失败: {e}")
        
        logger.info(f"成功加载 {len(self._collector_funcs)} 个采集器函数")
    
    
    
    def _get_stock_list(self) -> List[str]:
        """获取全A股股票列表"""
        if self._stock_list_cache is not None:
            return self._stock_list_cache['ts_code'].tolist()
        
        logger.info("获取全A股股票列表...")
        
        # 首先尝试从已采集的数据中读取
        stock_list_path = self.output_dir / "metadata" / "stock_list_a.parquet"
        if stock_list_path.exists():
            try:
                self._stock_list_cache = pd.read_parquet(stock_list_path)
                logger.info(f"从本地文件加载股票列表: {len(self._stock_list_cache)} 只")
                return self._stock_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.warning(f"读取本地股票列表失败: {e}")
        
        # 使用采集器获取
        if "get_stock_list_a" in self._collector_funcs:
            try:
                self._stock_list_cache = self._collector_funcs["get_stock_list_a"]()
                if not self._stock_list_cache.empty:
                    logger.info(f"从API获取股票列表: {len(self._stock_list_cache)} 只")
                    return self._stock_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.error(f"获取股票列表失败: {e}")
        
        logger.error("无法获取股票列表")
        return []

    def _get_fund_list(self) -> List[str]:
        """获取全量基金列表"""
        if self._fund_list_cache is not None:
            return self._fund_list_cache['ts_code'].tolist()
        
        logger.info("获取全量基金列表...")
        
        # 尝试从本地文件加载
        fund_list_path = self.output_dir / "derivatives" / "fund_basic.parquet"
        if fund_list_path.exists():
            try:
                self._fund_list_cache = pd.read_parquet(fund_list_path)
                logger.info(f"从本地文件加载基金列表: {len(self._fund_list_cache)} 只")
                return self._fund_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.warning(f"读取本地基金列表失败: {e}")
        
        # 使用采集器获取
        if "get_fund_basic" in self._collector_funcs:
            try:
                self._fund_list_cache = self._collector_funcs["get_fund_basic"]()
                if not self._fund_list_cache.empty:
                    logger.info(f"从API获取基金列表: {len(self._fund_list_cache)} 只")
                    return self._fund_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.error(f"获取基金列表失败: {e}")
        
        logger.error("无法获取基金列表")
        return []

    def _get_index_list(self) -> List[str]:
        """获取全量指数列表"""
        if self._index_list_cache is not None:
            return self._index_list_cache['ts_code'].tolist()
        
        logger.info("获取全量指数列表...")
        
        # 尝试从本地文件加载
        index_list_path = self.output_dir / "index_benchmark" / "index_basic.parquet"
        if index_list_path.exists():
            try:
                self._index_list_cache = pd.read_parquet(index_list_path)
                logger.info(f"从本地文件加载指数列表: {len(self._index_list_cache)} 只")
                return self._index_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.warning(f"读取本地指数列表失败: {e}")
        
        # 使用采集器获取
        if "get_index_basic" in self._collector_funcs:
            try:
                self._index_list_cache = self._collector_funcs["get_index_basic"]()
                if not self._index_list_cache.empty:
                    logger.info(f"从API获取指数列表: {len(self._index_list_cache)} 只")
                    return self._index_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.error(f"获取指数列表失败: {e}")
        
        logger.error("无法获取指数列表")
        return []

    def _get_option_list(self) -> List[str]:
        """获取全量期权列表"""
        if self._option_list_cache is not None:
            return self._option_list_cache['ts_code'].tolist()
        
        logger.info("获取全量期权列表...")
        
        # 尝试从本地文件加载
        opt_list_path = self.output_dir / "derivatives" / "opt_basic.parquet"
        if opt_list_path.exists():
            try:
                self._option_list_cache = pd.read_parquet(opt_list_path)
                logger.info(f"从本地文件加载期权列表: {len(self._option_list_cache)} 只")
                return self._option_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.warning(f"读取本地期权列表失败: {e}")
        
        # 使用采集器获取
        if "get_opt_basic" in self._collector_funcs:
            try:
                self._option_list_cache = self._collector_funcs["get_opt_basic"]()
                if not self._option_list_cache.empty:
                    logger.info(f"从API获取期权列表: {len(self._option_list_cache)} 只")
                    return self._option_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.error(f"获取期权列表失败: {e}")
        
        logger.error("无法获取期权列表")
        return []

    def _get_bond_list(self) -> List[str]:
        """获取全量债券（转债）列表"""
        if self._bond_list_cache is not None:
            return self._bond_list_cache['ts_code'].tolist()
        
        logger.info("获取全量可转债列表...")
        
        # 尝试从本地文件加载
        bond_list_path = self.output_dir / "derivatives" / "cb_basic.parquet"
        if bond_list_path.exists():
            try:
                self._bond_list_cache = pd.read_parquet(bond_list_path)
                logger.info(f"从本地文件加载转债列表: {len(self._bond_list_cache)} 只")
                return self._bond_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.warning(f"读取本地转债列表失败: {e}")
        
        # 使用采集器获取
        if "get_cb_basic" in self._collector_funcs:
            try:
                self._bond_list_cache = self._collector_funcs["get_cb_basic"]()
                if not self._bond_list_cache.empty:
                    logger.info(f"从API获取转债列表: {len(self._bond_list_cache)} 只")
                    return self._bond_list_cache['ts_code'].tolist()
            except Exception as e:
                logger.error(f"获取转债列表失败: {e}")
        
        logger.error("无法获取转债列表")
        return []
    
    def _save_to_parquet(
        self, 
        df: pd.DataFrame, 
        output_path: Path,
        append: bool = False
    ) -> bool:
        """保存数据到parquet文件"""
        if df.empty:
            logger.debug(f"跳过保存空数据: {output_path}")
            return True
            
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if append and output_path.exists():
                # 追加模式：读取现有数据并合并
                existing_df = pd.read_parquet(output_path)
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates()
            
            df.to_parquet(output_path, index=False, compression='snappy')
            return True
        except Exception as e:
            logger.error(f"保存parquet失败 [{output_path}]: {e}")
            return False
    
    def _execute_single_task(
        self, 
        task: CollectionTask,
        ts_code: Optional[str] = None,
    ) -> TaskResult:
        """执行单个采集任务"""
        # 检查是否跳过已存在的文件
        output_file = task.output_file
        if ts_code and '{ts_code}' in output_file:
            output_file = output_file.replace('{ts_code}', ts_code.replace('.', '_'))
        
        output_path = self.output_dir / task.domain / output_file
        
        if self.skip_existing and output_path.exists():
            logger.info(f"跳过已存在的文件: {output_path}")
            return TaskResult(
                task_name=task.name,
                domain=task.domain,
                status=TaskStatus.SKIPPED,
                output_path=str(output_path),
            )

        result = TaskResult(
            task_name=task.name,
            domain=task.domain,
            status=TaskStatus.RUNNING,
            start_time=datetime.now(),
        )
        
        try:
            # 获取采集器函数
            collector_func = self._collector_funcs.get(task.collector_func)
            if not collector_func:
                raise ValueError(f"采集器函数不存在: {task.collector_func}")
            
            # 构建参数
            params = dict(task.params)
            
            # 为所有任务提供基础时间上下文参数 (start_date/end_date)
            # 这有助于采集器在没有显式参数时尊重全局调度范围，避免出现未来日期的硬编码
            if 'start_date' not in params:
                params['start_date'] = self.start_date
            if 'end_date' not in params:
                params['end_date'] = self.end_date
            
            # 时间相关任务的特定逻辑（如果需要显式覆盖或基于分类的其他处理）
            if task.category == DataCategory.TIME_DEPENDENT:
                # 这种情况下一般已通过 params 传递了日期范围
                pass
            
            # 需要股票代码的任务
            if ts_code:
                params['ts_code'] = ts_code
            
            # 执行采集
            logger.debug(f"执行采集: {task.name}, 参数: {params}")
            df = collector_func(**params)
            
            if df is not None and not df.empty:
                # 强制日期过滤，确保数据不超出指定范围
                date_field = task.date_field or 'trade_date'
                if date_field in df.columns:
                    try:
                        # 统一转换为字符串进行比较
                        date_series = df[date_field].astype(str).str.replace('-', '').str.replace('/', '')
                        # 核心逻辑：在指定日期范围内的，或者日期确实（None/NaN）的允许通过
                        # 对于元数据列表，缺失日期不应导致整行被过滤
                        mask = (date_series >= self.start_date) & (date_series <= self.end_date) | (date_series.isin(['None', 'nan', '', 'NaT']))
                        df = df[mask].copy()
                        if df.empty:
                            logger.warning(f"任务 {task.name} 经过日期过滤后返回空数据")
                    except Exception as e:
                        logger.warning(f"日期过滤执行失败: {e}")
            
            if df is None or df.empty:
                logger.warning(f"任务 {task.name} 返回空数据")
                result.status = TaskStatus.SUCCESS
                result.records_count = 0
            else:
                # 检查是否需要拆分保存
                if task.split_by and task.split_by in df.columns:
                    split_col = task.split_by
                    logger.info(f"任务 {task.name} 开启拆分保存模式 (split_by={split_col})")
                    
                    unique_entities = df[split_col].unique()
                    for entity in unique_entities:
                        entity_df = df[df[split_col] == entity]
                        
                        # 构建该实体的输出路径
                        output_file = task.output_file
                        # 假设实体是代码，支持 {ts_code} 替换
                        if '{ts_code}' in output_file:
                            output_file = output_file.replace('{ts_code}', str(entity).replace('.', '_'))
                        elif '{' in output_file and '}' in output_file:
                            # 通用替换，如 {industry_name}
                            import re
                            output_file = re.sub(r'\{.*?\}', str(entity).replace('.', '_'), output_file)
                        
                        output_path = self.output_dir / task.domain / output_file
                        self._save_to_parquet(entity_df, output_path)
                    
                    result.status = TaskStatus.SUCCESS
                    result.records_count = len(df)
                    result.output_path = str(self.output_dir / task.domain / task.output_file.split('/')[0])
                    logger.info(f"任务 {task.name} 拆分保存完成，共 {len(unique_entities)} 个实体")
                else:
                    # 确定输出路径
                    output_file = task.output_file
                    if ts_code and '{ts_code}' in output_file:
                        output_file = output_file.replace('{ts_code}', ts_code.replace('.', '_'))
                    
                    output_path = self.output_dir / task.domain / output_file
                    
                    # 保存数据
                    if self._save_to_parquet(df, output_path):
                        result.status = TaskStatus.SUCCESS
                        result.records_count = len(df)
                        result.output_path = str(output_path)
                        logger.info(
                            f"任务 {task.name} 完成: "
                            f"{result.records_count} 条记录 -> {output_path}"
                        )
                    else:
                        raise Exception(f"保存数据失败: {output_path}")
        
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()
            logger.error(f"任务 {task.name} 失败: {e}")
        
        result.end_time = datetime.now()
        return result
    
    def _execute_batch_task(
        self, 
        task: CollectionTask,
        stock_list: List[str],
    ) -> List[TaskResult]:
        """执行批量采集任务（按股票代码遍历）"""
        results = []
        total = len(stock_list)
        
        for i, ts_code in enumerate(stock_list):
            # 检查是否跳过已存在的文件
            if self.skip_existing and '{ts_code}' in task.output_file:
                output_file = task.output_file.replace(
                    '{ts_code}', ts_code.replace('.', '_')
                )
                output_path = self.output_dir / task.domain / output_file
                if output_path.exists():
                    logger.debug(f"跳过已存在的文件: {output_path}")
                    continue
            
            logger.info(f"采集 {task.name} [{i+1}/{total}]: {ts_code}")
            
            # 执行采集
            result = self._execute_single_task(task, ts_code=ts_code)
            results.append(result)
            
            # 失败重试
            if result.status == TaskStatus.FAILED and self.retry_failed:
                for retry in range(self.max_retries):
                    logger.warning(
                        f"任务 {task.name} ({ts_code}) 重试 {retry+1}/{self.max_retries}"
                    )
                    time.sleep(self.retry_delay)
                    result = self._execute_single_task(task, ts_code=ts_code)
                    if result.status == TaskStatus.SUCCESS:
                        break
                results[-1] = result
            
            # 控制请求频率
            time.sleep(0.3)  # 避免API限流
        
        return results
    
    def run_task(self, task: CollectionTask) -> List[TaskResult]:
        """执行单个采集任务"""
        self._load_collector_funcs()
        
        results = []
        
        if task.stock_scope == StockScope.ALL_A:
            # 需要遍历全A股
            stock_list = self._get_stock_list()
            if not stock_list:
                logger.error(f"任务 {task.name} 无法获取股票列表，跳过")
                result = TaskResult(
                    task_name=task.name,
                    domain=task.domain,
                    status=TaskStatus.SKIPPED,
                    error_message="无法获取股票列表"
                )
                return [result]
            
            results = self._execute_batch_task(task, stock_list)
        elif task.stock_scope == StockScope.ALL_FUND:
            # 需要遍历全量基金
            fund_list = self._get_fund_list()
            if not fund_list:
                logger.error(f"任务 {task.name} 无法获取基金列表，跳过")
                result = TaskResult(
                    task_name=task.name,
                    domain=task.domain,
                    status=TaskStatus.SKIPPED,
                    error_message="无法获取基金列表"
                )
                return [result]
            results = self._execute_batch_task(task, fund_list)
        elif task.stock_scope == StockScope.ALL_INDEX:
            # 需要遍历全量指数
            index_list = self._get_index_list()
            if not index_list:
                logger.error(f"任务 {task.name} 无法获取指数列表，跳过")
                result = TaskResult(
                    task_name=task.name,
                    domain=task.domain,
                    status=TaskStatus.SKIPPED,
                    error_message="无法获取指数列表"
                )
                return [result]
            results = self._execute_batch_task(task, index_list)
        elif task.stock_scope == StockScope.ALL_OPTION:
            # 需要遍历全量期权
            option_list = self._get_option_list()
            if not option_list:
                logger.error(f"任务 {task.name} 无法获取期权列表，跳过")
                result = TaskResult(
                    task_name=task.name,
                    domain=task.domain,
                    status=TaskStatus.SKIPPED,
                    error_message="无法获取期权列表"
                )
                return [result]
            results = self._execute_batch_task(task, option_list)
        elif task.stock_scope == StockScope.ALL_BOND:
            # 需要遍历全量可转债
            bond_list = self._get_bond_list()
            if not bond_list:
                logger.error(f"任务 {task.name} 无法获取转债列表，跳过")
                result = TaskResult(
                    task_name=task.name,
                    domain=task.domain,
                    status=TaskStatus.SKIPPED,
                    error_message="无法获取转债列表"
                )
                return [result]
            results = self._execute_batch_task(task, bond_list)
        else:
            # 单次采集
            result = self._execute_single_task(task)
            results = [result]
            
            # 失败重试
            if result.status == TaskStatus.FAILED and self.retry_failed:
                for retry in range(self.max_retries):
                    logger.warning(
                        f"任务 {task.name} 重试 {retry+1}/{self.max_retries}"
                    )
                    time.sleep(self.retry_delay)
                    result = self._execute_single_task(task)
                    if result.status == TaskStatus.SUCCESS:
                        break
                results = [result]
        
        return results
    
    def run_domain(
        self, 
        domain: str, 
        task_names: Optional[List[str]] = None
    ) -> List[TaskResult]:
        """
        执行指定数据域的采集任务
        
        Args:
            domain: 数据域名称
            task_names: 指定要执行的任务名称列表，None表示执行全部
        
        Returns:
            任务结果列表
        """
        self._load_collector_funcs()
        
        domain_tasks = get_tasks_by_domain(domain)
        if not domain_tasks:
            logger.warning(f"数据域 {domain} 没有配置采集任务")
            return []
        
        # 过滤任务
        if task_names:
            domain_tasks = [t for t in domain_tasks if t.name in task_names]
        
        # 过滤已禁用和实时任务
        domain_tasks = [t for t in domain_tasks if t.enabled and not t.realtime]
        
        # 按优先级排序
        domain_tasks = sorted(domain_tasks, key=lambda x: x.priority)
        
        logger.info(
            f"开始采集数据域 [{DOMAIN_NAMES.get(domain, domain)}], "
            f"共 {len(domain_tasks)} 个任务"
        )
        
        all_results = []
        for i, task in enumerate(domain_tasks):
            logger.info(
                f"[{i+1}/{len(domain_tasks)}] 执行任务: {task.name} - {task.description}"
            )
            self.progress.current_task = task.name
            self.progress.current_domain = domain
            
            results = self.run_task(task)
            all_results.extend(results)
            
            for result in results:
                self.progress.add_result(result)
        
        # 统计结果
        success_count = sum(1 for r in all_results if r.status == TaskStatus.SUCCESS)
        failed_count = sum(1 for r in all_results if r.status == TaskStatus.FAILED)
        
        logger.info(
            f"数据域 [{DOMAIN_NAMES.get(domain, domain)}] 采集完成: "
            f"成功={success_count}, 失败={failed_count}"
        )
        
        return all_results
    
    def run_all(
        self,
        domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        task_names: Optional[List[str]] = None,
    ) -> CollectionProgress:
        """
        执行全部数据域的采集任务
        
        Args:
            domains: 指定要执行的数据域列表，None表示执行全部
            exclude_domains: 要排除的数据域列表
            task_names: 指定要执行的任务名称列表 (可选)
        
        Returns:
            采集进度对象
        """
        self._load_collector_funcs()
        
        # 确定要执行的数据域
        all_domains = list(TASKS_BY_DOMAIN.keys())
        if domains:
            all_domains = [d for d in domains if d in TASKS_BY_DOMAIN]
        if exclude_domains:
            all_domains = [d for d in all_domains if d not in exclude_domains]
        
        # 初始化进度
        total_tasks = 0
        for d in all_domains:
            tasks = [t for t in TASKS_BY_DOMAIN[d] if t.enabled and not t.realtime]
            if task_names:
                tasks = [t for t in tasks if t.name in task_names]
            total_tasks += len(tasks)
        self.progress = CollectionProgress(
            total_tasks=total_tasks,
            start_time=datetime.now()
        )
        
        logger.info("=" * 60)
        logger.info(f"全量数据采集开始")
        logger.info(f"日期范围: {self.start_date} ~ {self.end_date}")
        logger.info(f"数据域: {', '.join(all_domains)}")
        logger.info(f"预计任务数: {total_tasks}")
        logger.info("=" * 60)
        
        # 按数据域依次执行
        for domain in all_domains:
            logger.info("-" * 40)
            self.run_domain(domain, task_names=task_names)
        
        # 保存采集报告
        self._save_collection_report()
        
        logger.info("=" * 60)
        logger.info(f"全量数据采集完成")
        logger.info(
            f"总计: {self.progress.completed_tasks} 个任务, "
            f"成功: {self.progress.success_tasks}, "
            f"失败: {self.progress.failed_tasks}, "
            f"跳过: {self.progress.skipped_tasks}"
        )
        logger.info("=" * 60)
        
        return self.progress
    
    def _save_collection_report(self):
        """保存采集报告"""
        report_path = self.output_dir / "collection_report.parquet"
        
        if not self.progress.results:
            return
        
        report_df = pd.DataFrame([r.to_dict() for r in self.progress.results])
        report_df.to_parquet(report_path, index=False)
        logger.info(f"采集报告已保存: {report_path}")
        
        # 同时保存为CSV便于查看
        csv_path = self.output_dir / "collection_report.csv"
        report_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"采集报告(CSV)已保存: {csv_path}")
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """获取采集汇总信息"""
        summary = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "output_dir": str(self.output_dir),
            "progress": {
                "total_tasks": self.progress.total_tasks,
                "completed_tasks": self.progress.completed_tasks,
                "success_tasks": self.progress.success_tasks,
                "failed_tasks": self.progress.failed_tasks,
                "skipped_tasks": self.progress.skipped_tasks,
                "progress_percent": f"{self.progress.progress_percent:.1f}%",
            },
            "domains": {},
        }
        
        # 按域统计
        for result in self.progress.results:
            domain = result.domain
            if domain not in summary["domains"]:
                summary["domains"][domain] = {
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "records": 0,
                }
            summary["domains"][domain]["total"] += 1
            if result.status == TaskStatus.SUCCESS:
                summary["domains"][domain]["success"] += 1
                summary["domains"][domain]["records"] += result.records_count
            elif result.status == TaskStatus.FAILED:
                summary["domains"][domain]["failed"] += 1
        
        return summary
