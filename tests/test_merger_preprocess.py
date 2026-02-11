#!/usr/bin/env python3
"""
测试脚本：仅运行特征工程 Pipeline 的合并+预处理阶段

复用 src/feature_engineering/structured/pipeline.py 逻辑
输出：data/features/temp/merger_preprocess.parquet
"""

import gc
import sys
import time
import logging
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def run_merger_preprocess(use_gpu: bool = False) -> None:
    """
    执行合并+预处理阶段
    
    Args:
        use_gpu: 是否使用 GPU 加速
    """
    from src.feature_engineering.structured.config import PipelineConfig
    from src.feature_engineering.structured.preprocess import (
        PreprocessConfig,
        PricePreprocessor,
        FundamentalPreprocessor,
        StatusPreprocessor,
        MoneyFlowPreprocessor,
        ChipPreprocessor,
        IndustryPreprocessor,
        MacroPreprocessor,
        EventPreprocessor,
    )
    from src.feature_engineering.structured.merger.merger import DataMerger
    
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("🚀 合并+预处理阶段启动")
    logger.info(f"   模式: {'GPU' if use_gpu else 'CPU'}")
    logger.info("=" * 70)
    
    # 初始化配置
    config = PipelineConfig(use_gpu=use_gpu)
    
    # 初始化预处理器
    preprocess_config = PreprocessConfig(use_gpu=use_gpu)
    
    preprocessors = {
        'price': PricePreprocessor(preprocess_config),
        'fundamental': FundamentalPreprocessor(preprocess_config),
        'status': StatusPreprocessor(preprocess_config),
        'money_flow': MoneyFlowPreprocessor(preprocess_config),
        'chip': ChipPreprocessor(preprocess_config),
        'industry': IndustryPreprocessor(preprocess_config),
        'macro': MacroPreprocessor(preprocess_config),
        'event': EventPreprocessor(preprocess_config),
    }
    
    # 初始化合并器
    merger = DataMerger(config.data, use_gpu=use_gpu)
    
    # 执行流式合并+预处理
    logger.info("=" * 60)
    logger.info("📋 Step 1: 流式 加载+预处理+合并 (8表)")
    logger.info("=" * 60)
    
    df = merger.process_with_preprocessing(
        preprocessors=preprocessors,
        filter_universe=True,
        drop_unnecessary=False,
        save_result=False
    )
    
    logger.info(f"  ✓ 流式处理完成: {len(df):,} 行, {len(df.columns)} 列")
    
    # 保存到指定目录
    output_dir = PROJECT_ROOT / "data" / "features" / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "merger_preprocess.parquet"
    
    logger.info("=" * 60)
    logger.info("📋 Step 2: 保存到磁盘")
    logger.info("=" * 60)
    logger.info(f"  💾 输出路径: {output_path}")
    
    # 保存
    if use_gpu:
        df_pd = df.to_pandas()
        df_pd.to_parquet(str(output_path), index=False)
        file_size = output_path.stat().st_size / (1024 * 1024)
        n_rows, n_cols = len(df_pd), len(df_pd.columns)
        del df_pd, df
    else:
        df.to_parquet(str(output_path), index=False)
        file_size = output_path.stat().st_size / (1024 * 1024)
        n_rows, n_cols = len(df), len(df.columns)
        del df
    
    # 清理内存
    gc.collect()
    if use_gpu:
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
    
    elapsed = time.time() - start_time
    
    logger.info("=" * 70)
    logger.info("✅ 合并+预处理完成!")
    logger.info(f"   样本数: {n_rows:,}")
    logger.info(f"   特征数: {n_cols}")
    logger.info(f"   文件大小: {file_size:.2f} MB")
    logger.info(f"   耗时: {elapsed:.2f} 秒")
    logger.info(f"   输出路径: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行特征工程合并+预处理阶段")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU 加速")
    parser.add_argument("--cpu", action="store_true", help="仅使用 CPU（默认）")
    
    args = parser.parse_args()
    
    # 默认使用 CPU
    use_gpu = args.gpu and not args.cpu
    
    run_merger_preprocess(use_gpu=use_gpu)
