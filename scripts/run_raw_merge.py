"""
结构化原始数据合并脚本

将 data/raw/structured 下按 ts_code 拆分存储的目录合并为单个 parquet 文件。

使用方法:
    # 扫描待合并目录（dry-run，不执行实际合并）
    python scripts/run_raw_merge.py --dry-run

    # 合并所有数据域
    python scripts/run_raw_merge.py

    # 合并指定数据域
    python scripts/run_raw_merge.py --domains market_data fundamental
"""

import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.scheduler.structured.full.merge import RawDataMerger


def parse_args():
    parser = argparse.ArgumentParser(
        description="结构化原始数据合并器：将按 ts_code 拆分存储的目录合并为单个 parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/structured",
        help="结构化原始数据根目录，默认 data/raw/structured",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        type=str,
        default=None,
        help="指定要合并的数据域（如 market_data fundamental），不指定则合并全部",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅扫描并列出待合并目录，不执行实际合并",
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/raw_merge.log", encoding="utf-8"),
        ],
    )


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("结构化原始数据合并 开始")
    logger.info(f"数据目录: {args.raw_dir}")
    logger.info(f"指定域: {args.domains or '全部'}")
    logger.info(f"dry-run: {args.dry_run}")
    logger.info("=" * 60)

    merger = RawDataMerger(raw_dir=args.raw_dir, dry_run=args.dry_run)

    if args.domains:
        for domain in args.domains:
            logger.info(f"\n>>> 合并数据域: {domain}")
            report = merger.merge_domain(domain)
            print(report.summary())
    else:
        report = merger.merge_all()
        print(report.summary())

    logger.info("结构化原始数据合并 完成")


if __name__ == "__main__":
    main()
