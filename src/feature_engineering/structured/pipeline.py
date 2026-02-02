"""
特征工程主流水线

完整流程：读取DWD -> 预处理 -> 合并 -> 特征计算 -> 标签生成 -> 标准化 -> 切分 -> 输出

输入：data/processed/structured/dwd/ 下的三张核心宽表
输出：data/features/structured/train.parquet
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """特征工程主流水线"""
    
    def __init__(self, config):
        """
        初始化流水线
        
        Args:
            config: PipelineConfig 配置
        """
        self.config = config
        self.use_gpu = config.use_gpu
        self.stats: Dict[str, Any] = {}
        
        # 初始化 DataFrame 库
        if self.use_gpu:
            try:
                import cudf
                self.pd = cudf
                logger.info("🚀 FeaturePipeline: GPU 加速已启用 (RAPIDS cuDF)")
            except ImportError:
                import pandas as pd
                self.pd = pd
                self.use_gpu = False
                logger.warning("⚠️ cuDF 不可用，回退到 pandas")
        else:
            import pandas as pd
            self.pd = pd
        
        # 初始化子模块
        self._init_modules()
    
    def _init_modules(self):
        """初始化各处理模块"""
        from .merger.merger import DataMerger
        from .features.feature_generator import FeatureGenerator
        from .labels.label_generator import LabelGenerator
        from .postprocess.post_processor import PostProcessor
        from .preprocess import (
            PreprocessConfig,
            PricePreprocessor,
            FundamentalPreprocessor,
            StatusPreprocessor,
        )
        
        # 预处理器
        preprocess_config = PreprocessConfig(use_gpu=self.use_gpu)
        self.price_preprocessor = PricePreprocessor(preprocess_config)
        self.fundamental_preprocessor = FundamentalPreprocessor(preprocess_config)
        self.status_preprocessor = StatusPreprocessor(preprocess_config)
        
        # 其他处理器
        self.merger = DataMerger(self.config.universe, self.use_gpu)
        self.feature_generator = FeatureGenerator(self.config.technical, self.use_gpu)
        self.label_generator = LabelGenerator(self.config.label, self.use_gpu)
        self.post_processor = PostProcessor(
            self.config.normalization,
            self.config.data,
            self.use_gpu
        )
    
    def run(self, save_output: bool = True) -> Any:
        """
        执行完整流水线
        
        流程：
        1. Load: 读取三张 DWD 宽表
        2. Preprocess: 预处理（倒数变换、对数变换、去极值等）
        3. Merge & Filter: 连表，剔除 ST、停牌、次新股
        4. Feature Eng: 计算 MA, MACD, Volatility 等
        5. Labeling: 生成未来 N 日收益率标签
        6. Slice: 只保留 2021.01.01 之后的数据
        7. Normalize: 截面 Z-Score 标准化
        8. Clean: 丢弃含 NaN 的行
        9. Save: 输出 train.parquet
        
        Args:
            save_output: 是否保存输出文件
            
        Returns:
            最终的 DataFrame
        """
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info("🚀 特征工程流水线启动")
        logger.info("=" * 70)
        
        # Step 0: 加载 DWD 数据
        price_df, fundamental_df, status_df = self._load_data()
        
        # Step 1: 预处理三张表
        price_df, fundamental_df, status_df = self._preprocess_tables(
            price_df, fundamental_df, status_df
        )
        
        # 收集需要删除的列
        drop_columns = []
        if hasattr(self.config, 'drop_columns'):
            drop_columns = (
                self.config.drop_columns.drop_raw_valuations +
                self.config.drop_columns.drop_misc
            )
        
        # Step 2: 合并与过滤
        df = self.merger.process(price_df, fundamental_df, status_df, drop_columns=drop_columns)
        
        # 释放原始数据内存
        del price_df, fundamental_df, status_df
        
        # Step 3: 特征生成
        df = self.feature_generator.generate_all(df)
        
        # Step 4: 标签生成
        df = self.label_generator.generate_labels(df)
        
        # Step 5-7: 后处理（切分、标准化、清洗）
        df = self.post_processor.process(df)
        
        # Step 8: 保存输出
        if save_output:
            self._save_output(df)
        
        # 统计信息汇总
        elapsed = time.time() - start_time
        self._print_summary(df, elapsed)
        
        return df
    
    def _load_data(self):
        """加载三张 DWD 宽表"""
        logger.info("=" * 60)
        logger.info("📋 Step 0: 加载数据")
        logger.info("=" * 60)
        
        input_dir = self.config.data.input_dir
        
        # 价格表
        price_path = input_dir / self.config.data.price_table
        logger.info(f"  📖 读取价格表: {price_path}")
        price_df = self.pd.read_parquet(str(price_path))
        logger.info(f"     ✓ {len(price_df):,} 行, {len(price_df.columns)} 列")
        
        # 基本面表
        fundamental_path = input_dir / self.config.data.fundamental_table
        logger.info(f"  📖 读取基本面表: {fundamental_path}")
        fundamental_df = self.pd.read_parquet(str(fundamental_path))
        logger.info(f"     ✓ {len(fundamental_df):,} 行, {len(fundamental_df.columns)} 列")
        
        # 状态表
        status_path = input_dir / self.config.data.status_table
        logger.info(f"  📖 读取状态表: {status_path}")
        status_df = self.pd.read_parquet(str(status_path))
        logger.info(f"     ✓ {len(status_df):,} 行, {len(status_df.columns)} 列")
        
        return price_df, fundamental_df, status_df
    
    def _preprocess_tables(self, price_df: Any, fundamental_df: Any, status_df: Any):
        """
        预处理三张表
        
        对 DWD 原始数据进行清洗和预处理：
        - 价格表：收益率裁剪、成交量对数变换
        - 基本面表：倒数变换（PE→EP）、对数变换、时滞过滤
        - 状态表：交易状态修正、风险掩码生成
        
        Args:
            price_df: 价格表
            fundamental_df: 基本面表
            status_df: 状态表
            
        Returns:
            预处理后的三张表
        """
        logger.info("=" * 60)
        logger.info("📋 Step 1: 数据预处理")
        logger.info("=" * 60)
        
        # 1. 预处理价格表
        logger.info("  📊 处理价格表...")
        price_df = self.price_preprocessor.process(price_df)
        
        # 2. 预处理基本面表
        logger.info("  📊 处理基本面表...")
        fundamental_df = self.fundamental_preprocessor.process(fundamental_df)
        
        # 3. 预处理状态表
        logger.info("  📊 处理状态表...")
        status_df = self.status_preprocessor.process(status_df)
        
        # 记录预处理统计
        self.stats["preprocess"] = {
            "price": self.price_preprocessor.get_stats(),
            "fundamental": self.fundamental_preprocessor.get_stats(),
            "status": self.status_preprocessor.get_stats(),
        }
        
        return price_df, fundamental_df, status_df
    
    def _save_output(self, df: Any):
        """保存输出文件"""
        output_dir = self.config.data.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / self.config.data.train_file
        
        logger.info(f"  💾 保存输出: {output_path}")
        df.to_parquet(str(output_path), index=False)
        
        # 获取文件大小
        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"     ✓ 文件大小: {file_size:.1f} MB")
        
        self.stats["output_path"] = str(output_path)
        self.stats["output_size_mb"] = file_size
    
    def _print_summary(self, df: Any, elapsed: float):
        """打印处理摘要"""
        logger.info("=" * 70)
        logger.info("📊 流水线执行摘要")
        logger.info("=" * 70)
        
        if self.use_gpu:
            row_count = int(len(df))
            col_count = int(len(df.columns))
            stock_count = int(df['ts_code'].nunique())
        else:
            row_count = len(df)
            col_count = len(df.columns)
            stock_count = df['ts_code'].nunique()
        
        logger.info(f"  📊 最终数据集:")
        logger.info(f"     样本数: {row_count:,}")
        logger.info(f"     特征数: {col_count}")
        logger.info(f"     股票数: {stock_count:,}")
        
        logger.info(f"  ⏱️ 总耗时: {elapsed:.2f} 秒")
        
        # 特征列统计
        feature_cols = [c for c in df.columns if not c.startswith(('ret_', 'label_', 'ts_code', 'trade_date'))]
        label_cols = [c for c in df.columns if c.startswith(('ret_', 'label_'))]
        
        logger.info(f"  📋 列统计:")
        logger.info(f"     特征列: {len(feature_cols)}")
        logger.info(f"     标签列: {len(label_cols)}")
        
        logger.info("=" * 70)
        logger.info("🎉 特征工程流水线完成!")
        logger.info("=" * 70)
        
        self.stats["final_rows"] = row_count
        self.stats["final_cols"] = col_count
        self.stats["final_stocks"] = stock_count
        self.stats["elapsed_seconds"] = elapsed
    
    def get_stats(self) -> Dict[str, Any]:
        """获取所有模块的统计信息"""
        return {
            "pipeline": self.stats,
            "merger": self.merger.get_stats(),
            "feature_generator": self.feature_generator.get_stats(),
            "label_generator": self.label_generator.get_stats(),
            "post_processor": self.post_processor.get_stats(),
        }
