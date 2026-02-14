#!/usr/bin/env python
"""
后处理数据质量检查入口脚本

运行 LGB 和 GRU 数据质量检查，生成独立报告

使用方法：
    python scripts/run_postprocess_verify.py                    # 检查两个数据集
    python scripts/run_postprocess_verify.py --only-lgb         # 只检查 LGB
    python scripts/run_postprocess_verify.py --only-gru         # 只检查 GRU
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering.verify.postprocess_lgb_checker import run_lgb_check
from src.feature_engineering.verify.postprocess_gru_checker import run_gru_check

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="后处理数据质量检查",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--lgb-path",
        type=str,
        default="data/features/structured/train_lgb.parquet",
        help="LGB 数据路径",
    )
    parser.add_argument(
        "--gru-path",
        type=str,
        default="data/features/structured/train_gru.parquet",
        help="GRU 数据路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="报告输出目录",
    )
    parser.add_argument(
        "--only-lgb",
        action="store_true",
        help="只检查 LGB 数据",
    )
    parser.add_argument(
        "--only-gru",
        action="store_true",
        help="只检查 GRU 数据",
    )
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    
    results = {}
    overall_pass = True
    
    # 检查 LGB
    if not args.only_gru:
        logger.info("=" * 70)
        logger.info("🔍 开始 LightGBM 数据质量检查")
        logger.info("=" * 70)
        
        lgb_result = run_lgb_check(
            lgb_path=args.lgb_path,
            gru_path=args.gru_path if not args.only_lgb else None,
            output_dir=args.output_dir,
        )
        results['lgb'] = lgb_result
        
        lgb_pass_rate = lgb_result['summary']['summary']['pass_rate']
        if lgb_pass_rate < 70:
            overall_pass = False
        
        logger.info("")
        logger.info(f"📊 LGB 检查完成: 通过率 {lgb_pass_rate:.1f}%")
        logger.info(f"   📄 JSON: {lgb_result['json_path']}")
        logger.info(f"   📄 Markdown: {lgb_result['md_path']}")
    
    # 检查 GRU
    if not args.only_lgb:
        logger.info("")
        logger.info("=" * 70)
        logger.info("🔍 开始 GRU 数据质量检查")
        logger.info("=" * 70)
        
        gru_result = run_gru_check(
            gru_path=args.gru_path,
            lgb_path=args.lgb_path if not args.only_gru else None,
            output_dir=args.output_dir,
        )
        results['gru'] = gru_result
        
        gru_pass_rate = gru_result['summary']['summary']['pass_rate']
        gru_critical_failed = any(
            c['level'] == 'CRITICAL' and not c['passed'] 
            for c in gru_result['summary']['checks']
        )
        
        if gru_pass_rate < 70 or gru_critical_failed:
            overall_pass = False
        
        logger.info("")
        logger.info(f"📊 GRU 检查完成: 通过率 {gru_pass_rate:.1f}%")
        logger.info(f"   📄 JSON: {gru_result['json_path']}")
        logger.info(f"   📄 Markdown: {gru_result['md_path']}")
    
    # 汇总
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 检查汇总")
    logger.info("=" * 70)
    
    if 'lgb' in results:
        lgb_summary = results['lgb']['summary']['summary']
        logger.info(f"   LGB: {lgb_summary['passed']}/{lgb_summary['total_checks']} 通过 ({lgb_summary['pass_rate']:.1f}%)")
    
    if 'gru' in results:
        gru_summary = results['gru']['summary']['summary']
        logger.info(f"   GRU: {gru_summary['passed']}/{gru_summary['total_checks']} 通过 ({gru_summary['pass_rate']:.1f}%)")
    
    if overall_pass:
        logger.info("")
        logger.info("✅ 数据质量检查通过，可以进入模型训练阶段")
    else:
        logger.info("")
        logger.info("❌ 数据质量检查未通过，请检查报告中的失败项")
    
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
