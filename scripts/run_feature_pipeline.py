#!/usr/bin/env python
"""
特征工程流水线运行脚本

使用方法：
    python scripts/run_feature_pipeline.py
    python scripts/run_feature_pipeline.py --no-gpu
    python scripts/run_feature_pipeline.py --dry-run
    python scripts/run_feature_pipeline.py --latest-date 2026-01-07
    python scripts/run_feature_pipeline.py --postprocess only-lgb
    python scripts/run_feature_pipeline.py --postprocess only-gru
    python scripts/run_feature_pipeline.py --postprocess both
    python scripts/run_feature_pipeline.py --postprocess none
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = True):
    """配置日志"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"feature_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="特征工程流水线")
    parser.add_argument("--no-gpu", action="store_true", help="禁用 GPU 加速")
    parser.add_argument("--dry-run", action="store_true", help="只打印配置，不执行")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="详细输出")
    parser.add_argument("--output", "-o", type=str, help="自定义输出文件名")
    parser.add_argument("--memory-efficient", "-m", action="store_true", default=True, 
                        help="使用内存高效模式（逐列计算，中间结果暂存磁盘）")
    parser.add_argument("--no-memory-efficient", action="store_true", 
                        help="禁用内存高效模式")
    parser.add_argument("--postprocess", "-p", type=str, default="both",
                        choices=["both", "only-lgb", "only-gru", "none"],
                        help="后处理模式: both(默认)=输出train_lgb和train_gru, "
                             "only-lgb=仅LightGBM, only-gru=仅GRU, none=不做后处理")
    parser.add_argument(
        "--latest-date",
        "--latest_date",
        dest="latest_date",
        type=str,
        default=None,
        help="启用增量模式并指定 latest_date（支持 YYYYMMDD 或 YYYY-MM-DD）",
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    logger.info("=" * 70)
    logger.info("🚀 特征工程流水线启动")
    logger.info(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # 导入配置和流水线
    from src.feature_engineering.structured import (
        FeaturePipeline,
        FeaturePipelineIncrement,
        PipelineConfig,
    )
    
    # 创建配置
    config = PipelineConfig.default()
    config.use_gpu = not args.no_gpu
    config.verbose = args.verbose
    
    if args.output:
        config.data.train_file = args.output
    
    is_increment_mode = bool(args.latest_date)

    # 初始化流水线并预先推导自动日期边界（全量模式）
    pipeline = None
    auto_bounds = {}
    if is_increment_mode:
        pipeline = FeaturePipelineIncrement(config=config, latest_date=args.latest_date)
    else:
        try:
            pipeline = FeaturePipeline(config)
            auto_bounds = pipeline.get_auto_date_boundaries(emit_log=False)
        except Exception as e:
            logger.warning(f"⚠️ 自动日期边界推导失败，回退配置默认值打印: {e}")

    # 打印配置
    memory_efficient = args.memory_efficient and not args.no_memory_efficient
    postprocess_mode = args.postprocess
    if is_increment_mode and postprocess_mode != "both":
        logger.warning("⚠️ 增量模式固定输出 both，忽略 --postprocess=%s", postprocess_mode)
        postprocess_mode = "both"
    
    logger.info("📋 流水线配置:")
    logger.info(f"   运行模式: {'increment' if is_increment_mode else 'full'}")
    logger.info(f"   GPU 加速: {config.use_gpu}")
    logger.info(f"   内存高效模式: {memory_efficient}")
    logger.info(f"   后处理模式: {postprocess_mode}")
    logger.info(f"   输入目录: {config.data.input_dir}")
    logger.info(f"   输出目录: {config.data.output_dir}")
    if is_increment_mode:
        logger.info("   中间落盘: disabled (内存态)")
        logger.info(f"   latest_date: {args.latest_date}")
        logger.info(f"   增量输出根目录: {config.data.increment_output_root}")
        logger.info(
            f"   增量输入窗口: lookback={config.data.increment_lookback_trade_days} 交易日"
        )
        logger.info(
            f"   GRU 输出窗口: recent={config.data.increment_gru_trade_days} 交易日"
        )
        logger.info("   输出文件: predict_lgb.parquet + predict_gru.parquet")
    elif auto_bounds:
        logger.info(f"   临时目录: {config.data.temp_dir}")
        logger.info(f"   输出文件: {config.data.train_file}")
        logger.info(
            f"   原始交易日范围: {auto_bounds['raw_min_date']} ~ {auto_bounds['raw_max_date']}"
        )
        logger.info(
            f"   全量月份窗口: {auto_bounds['full_start']} ~ {auto_bounds['full_end']} "
            f"({auto_bounds['total_months']} 个月)"
        )
        logger.info(
            f"   预热期(LGB剔除): {auto_bounds['lgb_cut_start']} ~ {auto_bounds['lgb_cut_end']}"
        )
        logger.info(
            f"   预热期(GRU剔除): {auto_bounds['gru_cut_start']} ~ {auto_bounds['gru_cut_end']}"
        )
        logger.info(f"   正式期: {auto_bounds['train_start']} ~ {auto_bounds['train_end']}")
    else:
        logger.info(f"   临时目录: {config.data.temp_dir}")
        logger.info(f"   输出文件: {config.data.train_file}")
        logger.info(f"   预热期: {config.data.warmup_start} ~ {config.data.warmup_end}")
        logger.info(f"   正式期: {config.data.train_start} ~ {config.data.train_end}")
    logger.info("")
    logger.info(f"📋 股票池过滤:")
    logger.info(f"   剔除停牌: {config.universe.exclude_suspended}")
    logger.info(f"   剔除 ST: {config.universe.exclude_st}")
    logger.info(f"   剔除次新股: {config.universe.exclude_new} (< {config.universe.new_days_threshold} 天)")
    logger.info("")
    logger.info(f"📋 特征配置:")
    logger.info(f"   MA 周期: {config.technical.ma_periods}")
    logger.info(f"   RSI 周期: {config.technical.rsi_periods}")
    logger.info(f"   MACD 参数: {config.technical.macd_params}")
    logger.info("")
    logger.info(f"📋 标签配置:")
    logger.info(f"   预测周期: {config.label.forward_days} 日")
    logger.info(f"   主标签: {config.label.primary_label_days} 日")
    logger.info(f"   标签裁剪: [{config.label.label_clip_lower:.0%}, {config.label.label_clip_upper:.0%}]")
    logger.info("")
    logger.info(f"📋 后处理配置:")
    if is_increment_mode:
        logger.info("   输出: predict_lgb.parquet + predict_gru.parquet")
    elif postprocess_mode == "both":
        logger.info("   输出: train_lgb.parquet + train_gru.parquet")
    elif postprocess_mode == "only-lgb":
        logger.info("   输出: train_lgb.parquet (仅 LightGBM)")
    elif postprocess_mode == "only-gru":
        logger.info("   输出: train_gru.parquet (仅 GRU)")
    else:
        logger.info("   输出: train.parquet (无后处理)")
    logger.info("")
    
    if args.dry_run:
        logger.info("🏃 Dry Run 模式，不执行实际处理")
        return 0
    
    # 执行流水线
    try:
        if pipeline is None:
            pipeline = FeaturePipeline(config)

        if is_increment_mode:
            result = pipeline.run(save_output=True)
        else:
            result = pipeline.run(
                save_output=True,
                memory_efficient=memory_efficient,
                postprocess_mode=postprocess_mode,
            )
        
        logger.info("")
        logger.info("🎉 流水线执行成功!")

        if is_increment_mode and isinstance(result, dict):
            logger.info("   增量输出目录: %s", result.get("output_dir"))
            logger.info("   predict_lgb: %s", result.get("predict_lgb_path"))
            logger.info("   predict_gru: %s", result.get("predict_gru_path"))
        
        # 打印统计信息
        stats = pipeline.get_stats()
        logger.info("")
        logger.info("📊 详细统计:")
        for module, module_stats in stats.items():
            if module_stats:
                logger.info(f"   {module}:")
                if isinstance(module_stats, dict):
                    for key, value in module_stats.items():
                        if isinstance(value, dict):
                            logger.info(f"      {key}:")
                            for k2, v2 in value.items():
                                logger.info(f"         {k2}: {v2}")
                        else:
                            logger.info(f"      {key}: {value}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 流水线执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
