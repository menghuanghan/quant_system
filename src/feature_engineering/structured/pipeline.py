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
            # 核心3表预处理器
            PricePreprocessor,
            FundamentalPreprocessor,
            StatusPreprocessor,
            # 扩展5表预处理器
            MoneyFlowPreprocessor,
            ChipPreprocessor,
            IndustryPreprocessor,
            MacroPreprocessor,
            EventPreprocessor,
        )
        
        # 预处理器 - 8表完整配置
        preprocess_config = PreprocessConfig(use_gpu=self.use_gpu)
        
        # 核心3表预处理器
        self.price_preprocessor = PricePreprocessor(preprocess_config)
        self.fundamental_preprocessor = FundamentalPreprocessor(preprocess_config)
        self.status_preprocessor = StatusPreprocessor(preprocess_config)
        
        # 扩展5表预处理器
        self.money_flow_preprocessor = MoneyFlowPreprocessor(preprocess_config)
        self.chip_preprocessor = ChipPreprocessor(preprocess_config)
        self.industry_preprocessor = IndustryPreprocessor(preprocess_config)
        self.macro_preprocessor = MacroPreprocessor(preprocess_config)
        self.event_preprocessor = EventPreprocessor(preprocess_config)
        
        # 预处理器字典（供流式合并使用）
        self.preprocessors = {
            'price': self.price_preprocessor,
            'fundamental': self.fundamental_preprocessor,
            'status': self.status_preprocessor,
            'money_flow': self.money_flow_preprocessor,
            'chip': self.chip_preprocessor,
            'industry': self.industry_preprocessor,
            'macro': self.macro_preprocessor,
            'event': self.event_preprocessor,
        }
        
        # 参考数据加载器
        from .features.reference_data_loader import ReferenceDataLoader
        self.ref_loader = ReferenceDataLoader(self.config.data, self.use_gpu)
        
        # 其他处理器
        self.merger = DataMerger(self.config.data, use_gpu=self.use_gpu)
        # 特征生成器（参考数据延迟加载）
        self.feature_generator = FeatureGenerator(self.config.technical, self.use_gpu, ref_data=None)
        # 标签生成器（参考数据延迟加载，用于超额收益等高级标签）
        self.label_generator = LabelGenerator(self.config.label, self.use_gpu, ref_data=None)
        self.post_processor = PostProcessor(
            self.config.normalization,
            self.config.data,
            self.use_gpu
        )
    
    def run(self, save_output: bool = True, streaming_mode: bool = True, memory_efficient: bool = False) -> Any:
        """
        执行完整流水线
        
        流程（流式模式）：
        1. 流式预处理+合并：读取→预处理→合并 循环（内存高效）
        2. Feature Eng: 计算 MA, MACD, Volatility 等
        3. Labeling: 生成未来 N 日收益率标签
        4. Slice: 只保留 2021.01.01 之后的数据
        5. Normalize: 截面 Z-Score 标准化
        6. Clean: 丢弃含 NaN 的行
        7. Save: 输出 train.parquet
        
        Args:
            save_output: 是否保存输出文件
            streaming_mode: 是否使用流式预处理+合并（推荐，内存高效）
            memory_efficient: 是否使用内存高效模式（逐列计算，中间结果暂存磁盘）
            
        Returns:
            最终的 DataFrame
        """
        start_time = time.time()
        
        logger.info("=" * 70)
        mode_desc = "[内存高效模式]" if memory_efficient else ("[流式模式]" if streaming_mode else "[标准模式]")
        logger.info(f"🚀 特征工程流水线启动 {mode_desc}")
        logger.info("=" * 70)
        
        if memory_efficient:
            # 内存高效模式：中间结果暂存磁盘，逐列计算特征/标签
            df = self._run_memory_efficient_mode()
            
            # Step 8: 保存输出
            if save_output:
                self._save_output(df)
            
            # 统计信息汇总
            elapsed = time.time() - start_time
            self._print_summary(df, elapsed)
            
            return df
        elif streaming_mode:
            # 流式模式：预处理+合并一体化
            df = self._run_streaming_mode()
        else:
            # 标准模式：分步加载预处理合并（向后兼容）
            price_df, fundamental_df, status_df = self._load_data()
            price_df, fundamental_df, status_df = self._preprocess_tables(
                price_df, fundamental_df, status_df
            )
            drop_columns = []
            if hasattr(self.config, 'drop_columns'):
                drop_columns = (
                    self.config.drop_columns.drop_raw_valuations +
                    self.config.drop_columns.drop_misc
                )
            df = self.merger.process(price_df, fundamental_df, status_df, drop_columns=drop_columns)
            del price_df, fundamental_df, status_df
        
        # Step 2.5: 加载参考数据（用于相对强弱和指数成分特征）
        ref_data = self.ref_loader.load_all()
        self.feature_generator.set_ref_data(ref_data)
        
        # Step 3: 特征生成
        df = self.feature_generator.generate_all(df)
        
        # Step 4: 标签生成（需要 ref_data 用于超额收益标签）
        self.label_generator.ref_data = ref_data
        df = self.label_generator.generate_labels(df)
        
        # Step 5-7: 后处理（切分、标准化、清洗）
        # 暂时注销后处理模块，避免 OOM
        # TODO: 后续需要在内存高效模式下重新实现
        # df = self.post_processor.process(df)
        logger.info("⚠️ 后处理模块已暂时注销")
        
        # Step 8: 保存输出
        if save_output:
            self._save_output(df)
        
        # 统计信息汇总
        elapsed = time.time() - start_time
        self._print_summary(df, elapsed)
        
        return df
    
    def _run_streaming_mode(self) -> Any:
        """
        流式预处理+合并模式
        
        内存高效：读取一张表 → 预处理 → 合并 → 释放 → 下一张
        避免8张表同时驻留显存导致OOM
        
        Returns:
            预处理并合并后的DataFrame
        """
        logger.info("=" * 60)
        logger.info("📋 Step 0-2: 流式 加载+预处理+合并 (8表)")
        logger.info("=" * 60)
        
        # 使用 merger 的 process_with_preprocessing 方法
        df = self.merger.process_with_preprocessing(
            preprocessors=self.preprocessors,
            filter_universe=True,
            drop_unnecessary=False,  # 保留所有列供后续处理
            save_result=False  # 不单独保存中间结果
        )
        
        logger.info(f"  ✓ 流式处理完成: {len(df):,} 行, {len(df.columns)} 列")
        return df
    
    def _run_memory_efficient_mode(self) -> Any:
        """
        内存高效模式：避免 OOM
        
        核心策略：
        1. 合并+预处理后暂存磁盘，释放内存/显存
        2. 特征生成逐列计算：每次只读取需要的列，计算后立即释放
        3. 标签生成同理
        4. 新生成的特征/标签列驻留显存
        5. 最后合并保存
        
        Returns:
            最终的 DataFrame (特征 + 标签)
        """
        import gc
        
        # 确保临时目录存在
        temp_dir = self.config.data.temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        merger_preprocess_path = self.config.data.merger_preprocess_path
        
        # ============================
        # Step 1: 流式合并+预处理 → 暂存磁盘
        # ============================
        logger.info("=" * 60)
        logger.info("📋 Step 1: 流式 加载+预处理+合并 (8表) → 暂存磁盘")
        logger.info("=" * 60)
        
        df = self.merger.process_with_preprocessing(
            preprocessors=self.preprocessors,
            filter_universe=True,
            drop_unnecessary=False,
            save_result=False
        )
        
        # 保存到磁盘
        logger.info(f"  💾 暂存到: {merger_preprocess_path}")
        
        # 如果是 cuDF，需要转换为 pandas 才能保存（避免 cuDF parquet 写入问题）
        if self.use_gpu:
            df_pd = df.to_pandas()
            df_pd.to_parquet(str(merger_preprocess_path), index=False)
            file_size = merger_preprocess_path.stat().st_size / (1024 * 1024)
            logger.info(f"     ✓ 已保存 {len(df_pd):,} 行, {len(df_pd.columns)} 列, {file_size:.1f} MB")
            # 释放内存
            del df_pd, df
        else:
            df.to_parquet(str(merger_preprocess_path), index=False)
            file_size = merger_preprocess_path.stat().st_size / (1024 * 1024)
            logger.info(f"     ✓ 已保存 {len(df):,} 行, {len(df.columns)} 列, {file_size:.1f} MB")
            del df
        
        gc.collect()
        if self.use_gpu:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                logger.info("     ✓ 显存已释放")
            except:
                pass
        
        # ============================
        # Step 2: 加载参考数据
        # ============================
        logger.info("=" * 60)
        logger.info("📋 Step 2: 加载参考数据")
        logger.info("=" * 60)
        
        ref_data = self.ref_loader.load_all()
        self.feature_generator.set_ref_data(ref_data)
        
        # ============================
        # Step 3: 逐列计算特征
        # ============================
        logger.info("=" * 60)
        logger.info("📋 Step 3: 逐列计算特征 (内存高效模式)")
        logger.info("=" * 60)
        
        feature_columns = self.feature_generator.generate_column_by_column(
            parquet_path=merger_preprocess_path,
            ref_data=ref_data,
            use_gpu=self.use_gpu
        )
        
        logger.info(f"  ✅ 特征生成完成: {len(feature_columns)} 列")
        
        # ============================
        # Step 4: 逐列计算标签
        # ============================
        logger.info("=" * 60)
        logger.info("📋 Step 4: 逐列计算标签 (内存高效模式)")
        logger.info("=" * 60)
        
        self.label_generator.ref_data = ref_data
        label_columns = self.label_generator.generate_labels_column_by_column(
            parquet_path=merger_preprocess_path,
            use_gpu=self.use_gpu
        )
        
        logger.info(f"  ✅ 标签生成完成: {len(label_columns)} 列")
        
        # ============================
        # Step 5: 合并所有列
        # ============================
        logger.info("=" * 60)
        logger.info("📋 Step 5: 合并特征+标签到最终表")
        logger.info("=" * 60)
        
        # 读取原始表（只读取主键和必要列）
        if self.use_gpu:
            import cudf
            df = cudf.read_parquet(str(merger_preprocess_path))
        else:
            import pandas as pd
            df = pd.read_parquet(str(merger_preprocess_path))
        
        # 合并特征列
        for col_name, col_data in feature_columns.items():
            df[col_name] = col_data
        
        # 合并标签列
        for col_name, col_data in label_columns.items():
            df[col_name] = col_data
        
        logger.info(f"  ✓ 最终表: {len(df):,} 行, {len(df.columns)} 列")
        
        # 释放中间变量
        del feature_columns, label_columns
        gc.collect()
        
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
        """保存输出文件（内存高效模式）"""
        import gc
        
        output_dir = self.config.data.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / self.config.data.train_file
        
        logger.info(f"  💾 保存输出: {output_path}")
        
        # 如果是 GPU DataFrame，先转到 CPU 再保存，避免 GPU OOM
        if self.use_gpu:
            logger.info(f"     📤 转换到 CPU 内存...")
            pdf = df.to_pandas()
            
            # 释放 GPU DataFrame
            del df
            gc.collect()
            if self.use_gpu:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            
            logger.info(f"     💾 写入 parquet...")
            pdf.to_parquet(str(output_path), index=False, engine='pyarrow')
            
            # 释放 pandas DataFrame
            del pdf
            gc.collect()
        else:
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
        result = {"pipeline": self.stats}
        
        # 安全获取各模块统计信息
        if hasattr(self.merger, 'get_stats'):
            result["merger"] = self.merger.get_stats()
        if hasattr(self.feature_generator, 'get_stats'):
            result["feature_generator"] = self.feature_generator.get_stats()
        if hasattr(self.label_generator, 'get_stats'):
            result["label_generator"] = self.label_generator.get_stats()
        if hasattr(self.post_processor, 'get_stats'):
            result["post_processor"] = self.post_processor.get_stats()
        
        return result
