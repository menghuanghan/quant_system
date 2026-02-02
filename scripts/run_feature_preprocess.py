#!/usr/bin/env python3
"""
特征工程预处理入口脚本

对 DWD 宽表数据进行清洗和预处理，为下游特征工程层提供高质量输入。

使用方法：
    python scripts/run_feature_preprocess.py                    # 处理所有表
    python scripts/run_feature_preprocess.py --tables price     # 只处理价格表
    python scripts/run_feature_preprocess.py --tables price fundamental  # 处理多张表
    python scripts/run_feature_preprocess.py --no-gpu          # 禁用GPU加速
    python scripts/run_feature_preprocess.py --dry-run         # 只打印配置，不执行

输出目录：data/features/preprocessed/
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.processors.structured.feature_preprocess import (
    PreprocessConfig,
    PricePreprocessor,
    FundamentalPreprocessor,
    StatusPreprocessor,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "feature_preprocess.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="特征工程预处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    python scripts/run_feature_preprocess.py                    # 处理所有表
    python scripts/run_feature_preprocess.py --tables price     # 只处理价格表
    python scripts/run_feature_preprocess.py --no-gpu          # 禁用GPU加速
        """,
    )
    
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=["price", "fundamental", "status", "all"],
        default=["all"],
        help="要处理的表（默认: all）",
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="禁用GPU加速，使用CPU模式",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印配置，不实际执行",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="打印详细日志",
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="输入目录（默认: data/processed/structured/dwd）",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认: data/features/preprocessed）",
    )
    
    return parser.parse_args()


def print_config(config: PreprocessConfig):
    """打印配置信息"""
    logger.info("=" * 70)
    logger.info("📋 预处理配置")
    logger.info("=" * 70)
    logger.info(f"  输入目录: {config.input_dir}")
    logger.info(f"  输出目录: {config.output_dir}")
    logger.info(f"  使用GPU: {config.use_gpu}")
    logger.info("")
    
    logger.info("📊 去极值配置 (Winsorization):")
    logger.info(f"  收益率裁剪范围: [{config.winsorize.return_1d_lower:.0%}, {config.winsorize.return_1d_upper:.0%}]")
    logger.info(f"  分位数去极值字段: {len(config.winsorize.quantile_clip_fields)} 个")
    logger.info("")
    
    logger.info("📊 对数变换配置:")
    logger.info(f"  价格表字段: {config.log_transform.price_log_fields}")
    logger.info(f"  基本面表字段: {len(config.log_transform.fundamental_log_fields)} 个")
    logger.info("")
    
    logger.info("📊 倒数变换配置:")
    for src, dst in config.inverse_transform.inverse_fields.items():
        logger.info(f"  {src} -> {dst}")
    logger.info("")
    
    logger.info("📊 数据时滞配置:")
    logger.info(f"  最大允许时滞: {config.data_lag.max_lag_days} 天")
    logger.info(f"  受影响字段: {len(config.data_lag.lag_sensitive_fields)} 个")
    logger.info("")
    
    logger.info("📊 交易状态配置:")
    logger.info(f"  信资金不信标签: {config.trading_status.trust_volume_over_label}")
    logger.info("=" * 70)


def process_price_table(config: PreprocessConfig) -> dict:
    """处理价格表"""
    logger.info("")
    logger.info("🚀 开始处理价格表...")
    
    start_time = time.time()
    
    # 创建处理器
    processor = PricePreprocessor(config)
    
    # 读取数据
    input_path = config.input_dir / "dwd_stock_price.parquet"
    if not input_path.exists():
        logger.error(f"❌ 输入文件不存在: {input_path}")
        return {"success": False, "error": "文件不存在"}
    
    df = processor.read_parquet(input_path)
    logger.info(f"📖 读取完成: {len(df):,} 行")
    
    # 执行预处理
    df = processor.process(df)
    
    # 保存结果
    output_path = config.output_dir / config.output_price_file
    processor.to_parquet(df, output_path)
    
    elapsed = time.time() - start_time
    stats = processor.get_stats()
    stats["elapsed_time"] = elapsed
    
    logger.info(f"⏱️ 耗时: {elapsed:.2f} 秒")
    
    return {"success": True, "stats": stats, "df": df}


def process_fundamental_table(config: PreprocessConfig) -> dict:
    """处理基本面表"""
    logger.info("")
    logger.info("🚀 开始处理基本面表...")
    
    start_time = time.time()
    
    # 创建处理器
    processor = FundamentalPreprocessor(config)
    
    # 读取数据
    input_path = config.input_dir / "dwd_stock_fundamental.parquet"
    if not input_path.exists():
        logger.error(f"❌ 输入文件不存在: {input_path}")
        return {"success": False, "error": "文件不存在"}
    
    df = processor.read_parquet(input_path)
    logger.info(f"📖 读取完成: {len(df):,} 行")
    
    # 执行预处理
    df = processor.process(df)
    
    # 保存结果
    output_path = config.output_dir / config.output_fundamental_file
    processor.to_parquet(df, output_path)
    
    elapsed = time.time() - start_time
    stats = processor.get_stats()
    stats["elapsed_time"] = elapsed
    
    logger.info(f"⏱️ 耗时: {elapsed:.2f} 秒")
    
    return {"success": True, "stats": stats, "df": df}


def process_status_table(config: PreprocessConfig, price_df=None) -> dict:
    """处理状态表"""
    logger.info("")
    logger.info("🚀 开始处理状态表...")
    
    start_time = time.time()
    
    # 创建处理器
    processor = StatusPreprocessor(config)
    
    # 读取数据
    input_path = config.input_dir / "dwd_stock_status.parquet"
    if not input_path.exists():
        logger.error(f"❌ 输入文件不存在: {input_path}")
        return {"success": False, "error": "文件不存在"}
    
    df = processor.read_parquet(input_path)
    logger.info(f"📖 读取完成: {len(df):,} 行")
    
    # 执行预处理（需要价格表来判定交易状态）
    df = processor.process(df, price_df=price_df)
    
    # 保存结果
    output_path = config.output_dir / config.output_status_file
    processor.to_parquet(df, output_path)
    
    elapsed = time.time() - start_time
    stats = processor.get_stats()
    stats["elapsed_time"] = elapsed
    
    logger.info(f"⏱️ 耗时: {elapsed:.2f} 秒")
    
    return {"success": True, "stats": stats}


def print_summary(results: dict, total_time: float):
    """打印处理摘要"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 处理摘要")
    logger.info("=" * 70)
    
    for table_name, result in results.items():
        if result["success"]:
            stats = result.get("stats", {})
            original = stats.get("original_shape", ("?", "?"))
            final = stats.get("final_shape", ("?", "?"))
            elapsed = stats.get("elapsed_time", 0)
            
            logger.info(f"  ✅ {table_name}:")
            logger.info(f"     原始shape: {original}")
            logger.info(f"     最终shape: {final}")
            logger.info(f"     耗时: {elapsed:.2f}s")
            
            # 打印特定统计
            if "return_1d_above_upper" in stats:
                logger.info(f"     收益率截断: {stats.get('return_1d_below_lower', 0):,} 下/{stats.get('return_1d_above_upper', 0):,} 上")
            if "lag_gt_180" in stats:
                logger.info(f"     时滞过滤: {stats.get('lag_gt_180', 0):,} 行 (>180天)")
            if "tradable_pct" in stats:
                logger.info(f"     可交易比例: {stats.get('tradable_pct', 0):.1f}%")
        else:
            logger.info(f"  ❌ {table_name}: {result.get('error', '未知错误')}")
    
    logger.info("")
    logger.info(f"⏱️ 总耗时: {total_time:.2f} 秒")
    logger.info("=" * 70)


def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = PreprocessConfig(
        use_gpu=not args.no_gpu,
        verbose=args.verbose,
    )
    
    # 覆盖路径配置
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # 确保输出目录存在
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打印配置
    print_config(config)
    
    # 确定要处理的表
    tables = args.tables
    if "all" in tables:
        tables = ["price", "fundamental", "status"]
    
    logger.info(f"📋 待处理表: {tables}")
    
    if args.dry_run:
        logger.info("🔍 Dry run 模式，不执行实际处理")
        return
    
    # 开始处理
    start_time = time.time()
    results = {}
    price_df = None
    
    # 价格表（最先处理，为状态表提供 vol 信息）
    if "price" in tables:
        result = process_price_table(config)
        results["price"] = result
        if result["success"]:
            price_df = result.get("df")
    
    # 基本面表
    if "fundamental" in tables:
        result = process_fundamental_table(config)
        results["fundamental"] = result
    
    # 状态表（需要价格表的 is_trading_final）
    if "status" in tables:
        # 如果之前没有处理价格表，需要读取已处理的价格表
        if price_df is None and (config.output_dir / config.output_price_file).exists():
            try:
                import cudf
                price_df = cudf.read_parquet(str(config.output_dir / config.output_price_file))
            except ImportError:
                import pandas as pd
                price_df = pd.read_parquet(str(config.output_dir / config.output_price_file))
        
        result = process_status_table(config, price_df=price_df)
        results["status"] = result
    
    # 打印摘要
    total_time = time.time() - start_time
    print_summary(results, total_time)
    
    # 记录完成时间
    logger.info(f"🎉 预处理完成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📁 输出目录: {config.output_dir}")


if __name__ == "__main__":
    main()
