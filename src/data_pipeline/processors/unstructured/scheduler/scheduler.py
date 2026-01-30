"""
非结构化数据处理主调度器

负责：
1. 发现和遍历原始数据文件
2. 调度合适的流水线处理
3. 断点续传支持
4. 进度跟踪和报告
5. GPU加速批量处理
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .base import (
    ProcessingConfig,
    ProcessingResult,
    BatchProcessingResult,
    DataCategory,
    ProcessingStatus,
    CheckpointData,
    FIELD_MAPPING,
    PDF_OUTPUT_COLUMNS,
    EXCHANGE_OUTPUT_COLUMNS,
    CCTV_OUTPUT_COLUMNS,
    POLICY_OUTPUT_COLUMNS,
)
from .pipeline import (
    PDFPipeline,
    ExchangePipeline,
    CCTVPipeline,
    PolicyPipeline,
    CCTVProcessingResult,
    PolicyProcessingResult,
    BasePipeline,
    create_pipeline,
)

logger = logging.getLogger(__name__)


class UnstructuredScheduler:
    """
    非结构化数据处理调度器
    
    支持：
    - 多种数据类别（announcements, reports, events, exchange）
    - GPU加速批量处理
    - 断点续传
    - 进度跟踪
    
    使用示例：
    ```python
    scheduler = UnstructuredScheduler()
    
    # 处理单个月份
    result = scheduler.process_month(
        category=DataCategory.EXCHANGE,
        year=2021,
        month=1
    )
    
    # 处理整年
    results = scheduler.process_year(
        category=DataCategory.EXCHANGE,
        year=2021
    )
    
    # 处理所有数据
    results = scheduler.process_all()
    ```
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """初始化调度器"""
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 缓存流水线实例
        self._pipelines: Dict[DataCategory, BasePipeline] = {}
        
        # GPU加速支持
        self._cudf_available = False
        if self.config.use_gpu:
            try:
                import cudf
                self._cudf_available = True
                self.logger.info("cuDF GPU加速已启用")
            except ImportError:
                self.logger.warning("cuDF不可用，将使用pandas")
        
        self.logger.info(f"调度器初始化完成 (版本 {self.VERSION})")
    
    def _get_pipeline(self, category: DataCategory) -> BasePipeline:
        """获取或创建流水线"""
        if category not in self._pipelines:
            self._pipelines[category] = create_pipeline(category, self.config)
        return self._pipelines[category]
    
    def _load_checkpoint(
        self,
        category: DataCategory,
        year: int,
        month: int
    ) -> Optional[CheckpointData]:
        """加载检查点"""
        checkpoint_path = self.config.get_checkpoint_path(category, year, month)
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return CheckpointData.from_dict(data)
            except Exception as e:
                self.logger.warning(f"加载检查点失败: {e}")
        
        return None
    
    def _save_checkpoint(
        self,
        checkpoint: CheckpointData,
        category: DataCategory,
        year: int,
        month: int
    ):
        """保存检查点"""
        checkpoint_path = self.config.get_checkpoint_path(category, year, month)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint.last_update_time = datetime.now().isoformat()
        
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
    
    def _read_raw_data(
        self,
        category: DataCategory,
        year: int,
        month: int
    ) -> Optional[pd.DataFrame]:
        """读取原始数据"""
        raw_path = self.config.get_raw_path(category, year, month)
        
        if not raw_path.exists():
            self.logger.warning(f"原始数据不存在: {raw_path}")
            return None
        
        try:
            # 尝试使用cuDF读取
            if self._cudf_available:
                import cudf
                df = cudf.read_parquet(str(raw_path)).to_pandas()
            else:
                df = pd.read_parquet(raw_path)
            
            self.logger.info(f"读取原始数据: {raw_path}, 共 {len(df)} 条")
            return df
        except Exception as e:
            self.logger.error(f"读取原始数据失败: {e}")
            return None
    
    def _save_processed_data(
        self,
        df: pd.DataFrame,
        category: DataCategory,
        year: int,
        month: int
    ) -> str:
        """保存处理后的数据"""
        output_path = self.config.get_processed_path(category, year, month)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df.to_parquet(output_path, index=False, compression='snappy')
            self.logger.info(f"保存处理结果: {output_path}, 共 {len(df)} 条")
            return str(output_path)
        except Exception as e:
            self.logger.error(f"保存处理结果失败: {e}")
            raise
    
    def process_month(
        self,
        category: DataCategory,
        year: int,
        month: int,
        force: bool = False
    ) -> BatchProcessingResult:
        """
        处理单个月份的数据
        
        Args:
            category: 数据类别
            year: 年份
            month: 月份
            force: 是否强制重新处理（忽略检查点）
            
        Returns:
            BatchProcessingResult: 处理结果
        """
        start_time = time.time()
        self.logger.info(f"开始处理: {category.value}/{year}/{month:02d}")
        
        # 检查输出是否已存在
        output_path = self.config.get_processed_path(category, year, month)
        if output_path.exists() and self.config.skip_existing and not force:
            self.logger.info(f"跳过已存在的输出: {output_path}")
            return BatchProcessingResult(
                category=category,
                year=year,
                month=month,
                total=0,
                success_count=0,
                failed_count=0,
                skipped_count=1,
                results=[],
                elapsed_time_seconds=0,
                output_path=str(output_path),
            )
        
        # 读取原始数据
        raw_df = self._read_raw_data(category, year, month)
        if raw_df is None or len(raw_df) == 0:
            return BatchProcessingResult(
                category=category,
                year=year,
                month=month,
                total=0,
                success_count=0,
                failed_count=0,
                skipped_count=0,
                results=[],
                elapsed_time_seconds=time.time() - start_time,
            )
        
        # 加载检查点
        checkpoint = None
        if not force:
            checkpoint = self._load_checkpoint(category, year, month)
        
        processed_ids = set(checkpoint.processed_ids) if checkpoint else set()
        
        # 获取流水线
        pipeline = self._get_pipeline(category)
        
        # 处理数据
        if category == DataCategory.EXCHANGE:
            # Exchange: 使用GPU加速批量处理
            result = self._process_exchange_batch(
                pipeline, raw_df, category, year, month, processed_ids
            )
        elif category == DataCategory.CCTV:
            # CCTV: 生成市场情绪和Beta信号
            result = self._process_cctv_batch(
                pipeline, raw_df, category, year, month, processed_ids
            )
        elif category in (DataCategory.POLICY_GOV, DataCategory.POLICY_NDRC):
            # Policy: 生成行业映射和打分
            result = self._process_policy_batch(
                pipeline, raw_df, category, year, month, processed_ids
            )
        else:
            # PDF类: 逐条处理（需要LLM调用）
            result = self._process_pdf_batch(
                pipeline, raw_df, category, year, month, processed_ids
            )
        
        result.elapsed_time_seconds = time.time() - start_time
        
        # 更新检查点
        if result.success_count > 0:
            new_checkpoint = CheckpointData(
                category=category.value,
                year=year,
                month=month,
                processed_ids=[r.record_id for r in result.results if r.success],
                total_records=result.total,
                success_count=result.success_count,
                failed_count=result.failed_count,
                status='completed' if result.failed_count == 0 else 'partial',
            )
            self._save_checkpoint(new_checkpoint, category, year, month)
        
        self.logger.info(result.summary())
        return result
    
    def _process_exchange_batch(
        self,
        pipeline: ExchangePipeline,
        raw_df: pd.DataFrame,
        category: DataCategory,
        year: int,
        month: int,
        processed_ids: set
    ) -> BatchProcessingResult:
        """批量处理Exchange数据（GPU加速）"""
        field_map = FIELD_MAPPING.get(category, {})
        id_col = field_map.get('id', 'news_id')
        
        # 过滤已处理的记录
        if processed_ids:
            raw_df = raw_df[~raw_df[id_col].astype(str).isin(processed_ids)]
        
        if len(raw_df) == 0:
            return BatchProcessingResult(
                category=category,
                year=year,
                month=month,
                total=0,
                success_count=0,
                failed_count=0,
                skipped_count=len(processed_ids),
                results=[],
                elapsed_time_seconds=0,
            )
        
        # 使用流水线的DataFrame处理
        result_df = pipeline.process_dataframe_gpu(raw_df, category)
        
        # 确保score列是float类型（方便模型训练）
        result_df['score'] = result_df['score'].astype(float)
        
        # 保存结果
        output_path = self._save_processed_data(result_df, category, year, month)
        
        # 构建处理结果
        results = []
        for _, row in result_df.iterrows():
            results.append(ProcessingResult(
                success=True,
                record_id=row['id'],
                ts_code=row['ts_code'],
                date=row['date'],
                score=row['score'],
                reason=None,
            ))
        
        # 统计
        scores = result_df['score'].tolist()
        bullish = sum(1 for s in scores if s > 5)
        bearish = sum(1 for s in scores if s < -5)
        neutral = sum(1 for s in scores if -5 <= s <= 5)
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return BatchProcessingResult(
            category=category,
            year=year,
            month=month,
            total=len(results),
            success_count=len(results),
            failed_count=0,
            skipped_count=len(processed_ids),
            results=results,
            elapsed_time_seconds=0,
            output_path=output_path,
            avg_score=avg_score,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
        )
    
    def _process_cctv_batch(
        self,
        pipeline: CCTVPipeline,
        raw_df: pd.DataFrame,
        category: DataCategory,
        year: int,
        month: int,
        processed_ids: set
    ) -> BatchProcessingResult:
        """
        批量处理CCTV新闻数据
        
        生成市场情绪指数和Beta信号
        """
        field_map = FIELD_MAPPING.get(category, {})
        id_col = field_map.get('id', 'news_id')
        
        # 过滤已处理的记录
        if processed_ids:
            raw_df = raw_df[~raw_df[id_col].astype(str).isin(processed_ids)]
        
        if len(raw_df) == 0:
            return BatchProcessingResult(
                category=category,
                year=year,
                month=month,
                total=0,
                success_count=0,
                failed_count=0,
                skipped_count=len(processed_ids),
                results=[],
                elapsed_time_seconds=0,
            )
        
        # 转换为记录列表
        records = raw_df.to_dict('records')
        
        # 批量处理
        cctv_results = pipeline.process_batch(records, category)
        
        # 筛选成功的结果
        success_results = [r for r in cctv_results if r.success]
        
        if success_results:
            # 构建输出DataFrame
            output_data = []
            for r in success_results:
                output_data.append({
                    'date': r.date,
                    'id': r.record_id,
                    'market_sentiment': float(r.market_sentiment),  # 确保float类型
                    'beta_signal': float(r.beta_signal),
                    'keywords': ','.join(r.keywords) if r.keywords else '',
                    'tone_analysis': r.tone_analysis,
                })
            
            result_df = pd.DataFrame(output_data)
            
            # 确保数值列是float类型（方便模型训练）
            result_df['market_sentiment'] = result_df['market_sentiment'].astype(float)
            result_df['beta_signal'] = result_df['beta_signal'].astype(float)
            
            output_path = self._save_processed_data(result_df, category, year, month)
            
            # 统计
            sentiments = [r.market_sentiment for r in success_results]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            bullish = sum(1 for s in sentiments if s > 10)
            bearish = sum(1 for s in sentiments if s < -10)
            neutral = sum(1 for s in sentiments if -10 <= s <= 10)
        else:
            output_path = None
            avg_sentiment = 0
            bullish = bearish = neutral = 0
        
        # 转换结果类型以匹配BatchProcessingResult
        converted_results = []
        for r in cctv_results:
            converted_results.append(ProcessingResult(
                success=r.success,
                record_id=r.record_id,
                ts_code='',  # CCTV没有股票代码
                date=r.date,
                score=int(r.market_sentiment) if r.success else None,
                reason=r.tone_analysis if r.success else r.error_message,
            ))
        
        return BatchProcessingResult(
            category=category,
            year=year,
            month=month,
            total=len(cctv_results),
            success_count=len(success_results),
            failed_count=len(cctv_results) - len(success_results),
            skipped_count=len(processed_ids),
            results=converted_results,
            elapsed_time_seconds=0,
            output_path=output_path,
            avg_score=avg_sentiment,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
        )
    
    def _process_policy_batch(
        self,
        pipeline: PolicyPipeline,
        raw_df: pd.DataFrame,
        category: DataCategory,
        year: int,
        month: int,
        processed_ids: set
    ) -> BatchProcessingResult:
        """
        批量处理政策数据
        
        生成行业映射和打分
        """
        field_map = FIELD_MAPPING.get(category, {})
        id_col = field_map.get('id', 'id')
        
        # 过滤已处理的记录
        if processed_ids:
            raw_df = raw_df[~raw_df[id_col].astype(str).isin(processed_ids)]
        
        if len(raw_df) == 0:
            return BatchProcessingResult(
                category=category,
                year=year,
                month=month,
                total=0,
                success_count=0,
                failed_count=0,
                skipped_count=len(processed_ids),
                results=[],
                elapsed_time_seconds=0,
            )
        
        # 转换为记录列表
        records = raw_df.to_dict('records')
        
        # 批量处理
        policy_results = pipeline.process_batch(records, category)
        
        # 筛选成功的结果
        success_results = [r for r in policy_results if r.success]
        
        if success_results:
            import json
            
            # 构建输出DataFrame
            output_data = []
            for r in success_results:
                output_data.append({
                    'date': r.date,
                    'id': r.record_id,
                    'summary': r.summary,
                    'benefited_industries': ','.join(r.benefited_industries) if r.benefited_industries else '',
                    'harmed_industries': ','.join(r.harmed_industries) if r.harmed_industries else '',
                    'industry_scores': json.dumps(r.industry_scores, ensure_ascii=False) if r.industry_scores else '{}',
                })
            
            result_df = pd.DataFrame(output_data)
            output_path = self._save_processed_data(result_df, category, year, month)
            
            # 统计受益/受损行业数量
            benefited_count = sum(len(r.benefited_industries) for r in success_results)
            harmed_count = sum(len(r.harmed_industries) for r in success_results)
        else:
            output_path = None
            benefited_count = harmed_count = 0
        
        # 转换结果类型
        converted_results = []
        for r in policy_results:
            converted_results.append(ProcessingResult(
                success=r.success,
                record_id=r.record_id,
                ts_code='',  # Policy没有股票代码
                date=r.date,
                score=len(r.benefited_industries) - len(r.harmed_industries) if r.success else None,
                reason=r.summary[:100] if r.success and r.summary else r.error_message,
            ))
        
        return BatchProcessingResult(
            category=category,
            year=year,
            month=month,
            total=len(policy_results),
            success_count=len(success_results),
            failed_count=len(policy_results) - len(success_results),
            skipped_count=len(processed_ids),
            results=converted_results,
            elapsed_time_seconds=0,
            output_path=output_path,
            avg_score=0,  # Policy用行业数量代替
            bullish_count=benefited_count,  # 受益行业数
            bearish_count=harmed_count,     # 受损行业数
            neutral_count=0,
        )
    
    def _process_pdf_batch(
        self,
        pipeline: PDFPipeline,
        raw_df: pd.DataFrame,
        category: DataCategory,
        year: int,
        month: int,
        processed_ids: set
    ) -> BatchProcessingResult:
        """批量处理PDF数据（逐条LLM调用）"""
        field_map = FIELD_MAPPING.get(category, {})
        id_col = field_map.get('id', 'id')
        
        # 过滤已处理的记录
        if processed_ids:
            raw_df = raw_df[~raw_df[id_col].astype(str).isin(processed_ids)]
        
        if len(raw_df) == 0:
            return BatchProcessingResult(
                category=category,
                year=year,
                month=month,
                total=0,
                success_count=0,
                failed_count=0,
                skipped_count=len(processed_ids),
                results=[],
                elapsed_time_seconds=0,
            )
        
        # 转换为记录列表
        records = raw_df.to_dict('records')
        
        # 批量处理
        results = pipeline.process_batch(records, category)
        
        # 筛选成功的结果
        success_results = [r for r in results if r.success]
        
        if success_results:
            # 构建输出DataFrame
            output_data = []
            for r in success_results:
                output_data.append({
                    'id': r.record_id,
                    'ts_code': r.ts_code,
                    'date': r.date,
                    'score': float(r.score) if r.score is not None else 0.0,  # 确保float类型
                    'reason': r.reason,
                })
            
            result_df = pd.DataFrame(output_data)
            
            # 确保score列是float类型（方便模型训练）
            result_df['score'] = result_df['score'].astype(float)
            
            output_path = self._save_processed_data(result_df, category, year, month)
            
            # 统计
            scores = [r.score for r in success_results if r.score is not None]
            bullish = sum(1 for s in scores if s > 5)
            bearish = sum(1 for s in scores if s < -5)
            neutral = sum(1 for s in scores if -5 <= s <= 5)
            avg_score = sum(scores) / len(scores) if scores else 0
        else:
            output_path = None
            avg_score = 0
            bullish = bearish = neutral = 0
        
        return BatchProcessingResult(
            category=category,
            year=year,
            month=month,
            total=len(results),
            success_count=len(success_results),
            failed_count=len(results) - len(success_results),
            skipped_count=len(processed_ids),
            results=results,
            elapsed_time_seconds=0,
            output_path=output_path,
            avg_score=avg_score,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
        )
    
    def process_year(
        self,
        category: DataCategory,
        year: int,
        force: bool = False
    ) -> List[BatchProcessingResult]:
        """
        处理整年的数据
        
        Args:
            category: 数据类别
            year: 年份
            force: 是否强制重新处理
            
        Returns:
            List[BatchProcessingResult]: 各月的处理结果
        """
        self.logger.info(f"开始处理年度数据: {category.value}/{year}")
        
        results = []
        for month in range(1, 13):
            # 检查原始数据是否存在
            raw_path = self.config.get_raw_path(category, year, month)
            if not raw_path.exists():
                continue
            
            result = self.process_month(category, year, month, force)
            results.append(result)
        
        # 汇总统计
        total = sum(r.total for r in results)
        success = sum(r.success_count for r in results)
        failed = sum(r.failed_count for r in results)
        
        self.logger.info(
            f"年度处理完成: {category.value}/{year}, "
            f"总计 {total}, 成功 {success}, 失败 {failed}"
        )
        
        return results
    
    def process_all(
        self,
        categories: Optional[List[DataCategory]] = None,
        years: Optional[List[int]] = None,
        force: bool = False
    ) -> Dict[str, List[BatchProcessingResult]]:
        """
        处理所有数据
        
        Args:
            categories: 要处理的类别列表（默认全部）
            years: 要处理的年份列表（默认自动发现）
            force: 是否强制重新处理
            
        Returns:
            Dict[str, List[BatchProcessingResult]]: 按类别组织的结果
        """
        if categories is None:
            categories = [
                DataCategory.ANNOUNCEMENTS,
                DataCategory.REPORTS,
                DataCategory.EVENTS,
                DataCategory.EXCHANGE,
                DataCategory.CCTV,
                DataCategory.POLICY_GOV,
                DataCategory.POLICY_NDRC,
            ]
        
        all_results = {}
        
        for category in categories:
            self.logger.info(f"开始处理类别: {category.value}")
            
            # 发现可用年份
            category_dir = Path(self.config.raw_data_dir) / category.value
            if not category_dir.exists():
                self.logger.warning(f"类别目录不存在: {category_dir}")
                continue
            
            # 自动发现年份
            if years is None:
                available_years = sorted([
                    int(d.name) for d in category_dir.iterdir()
                    if d.is_dir() and d.name.isdigit()
                ])
            else:
                available_years = years
            
            category_results = []
            for year in available_years:
                year_results = self.process_year(category, year, force)
                category_results.extend(year_results)
            
            all_results[category.value] = category_results
        
        return all_results
    
    def discover_data(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        发现所有可用的原始数据
        
        Returns:
            Dict[str, List[Tuple[int, int]]]: 按类别组织的(年,月)列表
        """
        discovery = {}
        
        for category in DataCategory:
            category_dir = Path(self.config.raw_data_dir) / category.value
            if not category_dir.exists():
                continue
            
            data_files = []
            for year_dir in sorted(category_dir.iterdir()):
                if not year_dir.is_dir() or not year_dir.name.isdigit():
                    continue
                
                year = int(year_dir.name)
                for parquet_file in sorted(year_dir.glob("*.parquet")):
                    month = int(parquet_file.stem)
                    data_files.append((year, month))
            
            if data_files:
                discovery[category.value] = data_files
        
        return discovery
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前处理状态"""
        status = {
            'discovery': self.discover_data(),
            'config': {
                'raw_data_dir': self.config.raw_data_dir,
                'processed_data_dir': self.config.processed_data_dir,
                'use_gpu': self.config.use_gpu,
                'cudf_available': self._cudf_available,
            },
            'pipelines': list(self._pipelines.keys()),
        }
        return status


# ========== 便捷函数 ==========

def process_month(
    category: str,
    year: int,
    month: int,
    config: Optional[ProcessingConfig] = None,
    force: bool = False
) -> BatchProcessingResult:
    """
    便捷函数：处理单个月份
    
    Args:
        category: 数据类别（字符串）
        year: 年份
        month: 月份
        config: 配置
        force: 是否强制重新处理
        
    Returns:
        BatchProcessingResult
    """
    scheduler = UnstructuredScheduler(config)
    cat = DataCategory(category)
    return scheduler.process_month(cat, year, month, force)


def process_year(
    category: str,
    year: int,
    config: Optional[ProcessingConfig] = None,
    force: bool = False
) -> List[BatchProcessingResult]:
    """
    便捷函数：处理整年数据
    
    Args:
        category: 数据类别（字符串）
        year: 年份
        config: 配置
        force: 是否强制重新处理
        
    Returns:
        List[BatchProcessingResult]
    """
    scheduler = UnstructuredScheduler(config)
    cat = DataCategory(category)
    return scheduler.process_year(cat, year, force)


def process_all(
    categories: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    config: Optional[ProcessingConfig] = None,
    force: bool = False
) -> Dict[str, List[BatchProcessingResult]]:
    """
    便捷函数：处理所有数据
    
    Args:
        categories: 数据类别列表
        years: 年份列表
        config: 配置
        force: 是否强制重新处理
        
    Returns:
        Dict[str, List[BatchProcessingResult]]
    """
    scheduler = UnstructuredScheduler(config)
    cats = [DataCategory(c) for c in categories] if categories else None
    return scheduler.process_all(cats, years, force)
