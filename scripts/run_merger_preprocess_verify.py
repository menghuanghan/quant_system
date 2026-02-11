#!/usr/bin/env python3
"""
运行 Merger 预处理数据质量检查

用法:
    python scripts/run_merger_preprocess_verify.py
    python scripts/run_merger_preprocess_verify.py --merger-path /path/to/merger.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.verify.checker import MergerPreprocessChecker
from src.feature_engineering.verify.report import QualityReportGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Merger 预处理数据质量检查')
    parser.add_argument(
        '--merger-path',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'features' / 'temp' / 'merger_preprocess.parquet',
        help='merger_preprocess.parquet 路径'
    )
    parser.add_argument(
        '--dwd-price-path',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'processed' / 'structured' / 'dwd' / 'dwd_stock_price.parquet',
        help='dwd_stock_price.parquet 路径'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'reports',
        help='报告输出目录'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("🔍 Merger 预处理数据质量检查")
    logger.info("=" * 70)
    logger.info(f"  数据文件: {args.merger_path}")
    logger.info(f"  对比基准: {args.dwd_price_path}")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info("")
    
    # 创建检查器
    checker = MergerPreprocessChecker(
        merger_path=args.merger_path,
        dwd_price_path=args.dwd_price_path,
        output_dir=args.output_dir,
    )
    
    # 运行检查
    results = checker.run_all_checks()
    
    # 生成报告
    generator = QualityReportGenerator(checker)
    md_content, json_report = generator.generate_all()
    
    # 打印结果
    summary = checker.get_summary()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("📝 报告已生成")
    logger.info("=" * 70)
    
    # 返回状态码
    if summary['critical'] > 0 or summary['errors'] > 0:
        logger.error("❌ 检查未通过，存在 CRITICAL 或 ERROR 级别问题")
        return 1
    else:
        logger.info("✅ 检查通过")
        return 0


if __name__ == '__main__':
    sys.exit(main())
