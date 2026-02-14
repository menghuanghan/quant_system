#!/usr/bin/env python
"""
Train.parquet 数据质量检查脚本

运行示例:
    python scripts/run_train_data_verify.py
    python scripts/run_train_data_verify.py --train-path data/features/structured/train.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering.verify.train_data_checker import TrainDataChecker
from src.feature_engineering.verify.train_data_report import TrainDataReportGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/train_data_verify.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train.parquet 数据质量检查')
    parser.add_argument(
        '--train-path',
        type=str,
        default='data/features/structured/train.parquet',
        help='训练数据路径'
    )
    parser.add_argument(
        '--merger-path',
        type=str,
        default='data/features/temp/merger_preprocess.parquet',
        help='合并预处理数据路径（用于对比）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='报告输出目录'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("🚀 开始 Train.parquet 数据质量检查")
    logger.info("=" * 70)
    
    try:
        # 创建检查器
        checker = TrainDataChecker(
            train_path=args.train_path,
            merger_path=args.merger_path,
            output_dir=args.output_dir,
        )
        
        # 运行检查
        summary = checker.run_all_checks()
        
        # 生成报告
        report_generator = TrainDataReportGenerator(checker)
        md_content, json_data = report_generator.generate_all()
        
        # 输出结果
        logger.info("\n" + "=" * 70)
        logger.info("📊 检查完成")
        logger.info("=" * 70)
        
        passed = summary['check_results']['passed']
        total = summary['check_results']['total']
        failed = summary['check_results']['failed']
        
        logger.info(f"✅ 通过: {passed}/{total}")
        if failed > 0:
            logger.info(f"❌ 失败: {failed}/{total}")
        
        # 输出报告路径
        logger.info(f"\n📄 报告已生成:")
        logger.info(f"  - Markdown: reports/train_dq_report_*.md")
        logger.info(f"  - JSON: reports/train_dq_report_*.json")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        logger.error(f"❌ 检查失败: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
