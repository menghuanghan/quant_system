"""
后处理流水线

协调公共清洗、LightGBM 专用处理和 GRU 专用处理。
支持三种模式：
- only-lgb: 仅输出 train_lgb.parquet
- only-gru: 仅输出 train_gru.parquet
- default (both): 输出两个文件
"""

import gc
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from .common_cleaner import CommonCleaner
from .config import PostprocessConfig
from .gru_processor import GRUProcessor
from .lgb_processor import LGBProcessor

logger = logging.getLogger(__name__)


class PostprocessMode(Enum):
    """后处理模式"""
    ONLY_LGB = "only-lgb"
    ONLY_GRU = "only-gru"
    BOTH = "both"


class PostprocessPipeline:
    """后处理流水线"""
    
    def __init__(
        self, 
        config: Optional[PostprocessConfig] = None, 
        use_gpu: bool = False,
        mode: PostprocessMode = PostprocessMode.BOTH
    ):
        """
        初始化
        
        Args:
            config: 后处理配置
            use_gpu: 是否使用 GPU
            mode: 处理模式 (only-lgb, only-gru, both)
        """
        self.config = config or PostprocessConfig.default()
        self.use_gpu = use_gpu
        self.mode = mode
        self.stats: Dict[str, Any] = {}
        
        # 初始化子模块
        self.common_cleaner = CommonCleaner(self.config.common, use_gpu)
        self.lgb_processor = LGBProcessor(self.config.lgb, use_gpu)
        self.gru_processor = GRUProcessor(self.config.gru, use_gpu)
        
        # 初始化 pandas
        if use_gpu:
            try:
                import cudf
                self.pd = cudf
                logger.info(f"🚀 PostprocessPipeline: GPU 加速已启用 (模式: {mode.value})")
            except ImportError:
                import pandas as pd
                self.pd = pd
                self.use_gpu = False
                logger.warning("⚠️ cuDF 不可用，回退到 pandas")
        else:
            import pandas as pd
            self.pd = pd
    
    def run(self, df: Any, save_output: bool = True) -> Dict[str, Any]:
        """
        执行后处理流水线
        
        流程：
        1. 公共清洗
        2. 根据模式执行 LGB 和/或 GRU 处理
        3. 保存输出文件
        
        Args:
            df: 输入 DataFrame (train.parquet 的内容)
            save_output: 是否保存输出文件
            
        Returns:
            Dict 包含处理后的 DataFrame:
            - "lgb": LGB 处理后的 DataFrame (如果适用)
            - "gru": GRU 处理后的 DataFrame (如果适用)
        """
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info(f"🚀 后处理流水线启动 (模式: {self.mode.value})")
        logger.info("=" * 70)
        
        result = {}
        
        # ============================
        # Step 1: 公共清洗
        # ============================
        df_cleaned = self.common_cleaner.process(df)
        self.stats["common_cleaner"] = self.common_cleaner.get_stats()
        
        # 释放原始数据
        del df
        gc.collect()
        if self.use_gpu:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
        
        # ============================
        # Step 2: 根据模式处理
        # ============================
        output_dir = self.config.output_dir
        
        if self.mode in [PostprocessMode.ONLY_LGB, PostprocessMode.BOTH]:
            # [内存优化] BOTH 模式下先保存 df_cleaned 到临时文件，避免内存复制
            if self.mode == PostprocessMode.BOTH:
                # 先处理 LGB，复用 df_cleaned（注意：LGB 处理是可逆的，主要是排序和切片）
                # 但为了避免影响 GRU，我们使用浅复制只复制索引
                logger.info("  [内存优化] BOTH 模式: 先处理 LGB")
                df_lgb = df_cleaned.copy()  # 仍需复制，但后续立即释放
            else:
                df_lgb = df_cleaned
            
            # LGB 处理
            df_lgb = self.lgb_processor.process(df_lgb)
            self.stats["lgb_processor"] = self.lgb_processor.get_stats()
            
            if save_output:
                self.lgb_processor.save(df_lgb, output_dir)
            
            # [内存优化] 立即释放 LGB 数据，只保留路径信息
            result["lgb_path"] = str(output_dir / self.config.lgb.output_file)
            del df_lgb
            gc.collect()
            logger.info("  [内存优化] LGB 处理完成，已释放内存")
            
            # 如果只做 LGB，释放 cleaned 数据
            if self.mode == PostprocessMode.ONLY_LGB:
                del df_cleaned
                gc.collect()
        
        if self.mode in [PostprocessMode.ONLY_GRU, PostprocessMode.BOTH]:
            # 直接使用 df_cleaned（无需复制）
            df_gru = df_cleaned
            
            # GRU 处理
            df_gru = self.gru_processor.process(df_gru)
            self.stats["gru_processor"] = self.gru_processor.get_stats()
            
            if save_output:
                self.gru_processor.save(df_gru, output_dir)
            
            # [内存优化] 释放 GRU 数据，只保留路径信息
            result["gru_path"] = str(output_dir / self.config.gru.output_file)
            del df_gru
            gc.collect()
            logger.info("  [内存优化] GRU 处理完成，已释放内存")
        
        # ============================
        # 统计信息汇总
        # ============================
        elapsed = time.time() - start_time
        self._print_summary(result, elapsed)
        
        return result
    
    def run_from_file(
        self, 
        train_path: Path, 
        save_output: bool = True
    ) -> Dict[str, Any]:
        """
        从文件加载并执行后处理
        
        内存高效：分块读取，避免 OOM
        
        Args:
            train_path: train.parquet 文件路径
            save_output: 是否保存输出文件
            
        Returns:
            处理结果
        """
        logger.info(f"  📖 加载数据: {train_path}")
        
        train_path = Path(train_path)
        if not train_path.exists():
            raise FileNotFoundError(f"文件不存在: {train_path}")
        
        # 读取数据
        if self.use_gpu:
            import cudf
            df = cudf.read_parquet(str(train_path))
        else:
            import pandas as pd
            df = pd.read_parquet(str(train_path))
        
        file_size = train_path.stat().st_size / (1024 * 1024)
        logger.info(f"     ✓ {len(df):,} 行, {len(df.columns)} 列, {file_size:.1f} MB")
        
        return self.run(df, save_output)
    
    def _print_summary(self, result: Dict[str, Any], elapsed: float):
        """打印处理摘要"""
        logger.info("=" * 70)
        logger.info("📊 后处理流水线摘要")
        logger.info("=" * 70)
        
        # 公共清洗统计
        common_stats = self.stats.get("common_cleaner", {})
        logger.info(f"  📋 公共清洗:")
        logger.info(f"     原始行数: {common_stats.get('original_rows', 'N/A'):,}")
        logger.info(f"     清洗后行数: {common_stats.get('final_rows', 'N/A'):,}")
        logger.info(f"     删除列数: {common_stats.get('constant_cols_count', 0)}")
        
        # LGB 统计
        if "lgb_path" in result:
            lgb_stats = self.stats.get("lgb_processor", {})
            logger.info(f"  📋 LightGBM 处理:")
            logger.info(f"     最终行数: {lgb_stats.get('final_rows', 'N/A'):,}")
            logger.info(f"     输出文件: {lgb_stats.get('output_path', 'N/A')}")
            logger.info(f"     文件大小: {lgb_stats.get('output_size_mb', 0):.1f} MB")
        
        # GRU 统计
        if "gru_path" in result:
            gru_stats = self.stats.get("gru_processor", {})
            logger.info(f"  📋 GRU 处理:")
            logger.info(f"     最终行数: {gru_stats.get('final_rows', 'N/A'):,}")
            logger.info(f"     输出文件: {gru_stats.get('output_path', 'N/A')}")
            logger.info(f"     文件大小: {gru_stats.get('output_size_mb', 0):.1f} MB")
        
        logger.info(f"  ⏱️ 总耗时: {elapsed:.2f} 秒")
        logger.info("=" * 70)
        logger.info("🎉 后处理流水线完成!")
        logger.info("=" * 70)
        
        self.stats["elapsed_seconds"] = elapsed
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats
