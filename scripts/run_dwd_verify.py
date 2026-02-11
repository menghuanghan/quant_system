#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DWD数据质量检查运行脚本

运行8张DWD宽表的全面数据质量检查并生成报告

使用方法:
    python scripts/run_dwd_verify.py
    python scripts/run_dwd_verify.py --tables dwd_stock_price dwd_stock_fundamental
    python scripts/run_dwd_verify.py --output-dir reports/dwd_quality
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.processors.structured.dwd.verify import (
    StockPriceChecker,
    FundamentalChecker,
    MoneyFlowChecker,
    ChipStructureChecker,
    IndustryChecker,
    EventSignalChecker,
    StatusChecker,
    MacroEnvChecker,
    DWDReportGenerator,
    CrossTableChecker,
    CrossTableCheckReport,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "dwd_verify.log"),
    ],
)
logger = logging.getLogger(__name__)


# 所有检查器映射
CHECKER_CLASSES = {
    "dwd_stock_price": StockPriceChecker,
    "dwd_stock_fundamental": FundamentalChecker,
    "dwd_money_flow": MoneyFlowChecker,
    "dwd_chip_structure": ChipStructureChecker,
    "dwd_stock_industry": IndustryChecker,
    "dwd_event_signal": EventSignalChecker,
    "dwd_stock_status": StatusChecker,
    "dwd_macro_env": MacroEnvChecker,
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DWD数据质量检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=list(CHECKER_CLASSES.keys()),
        default=list(CHECKER_CLASSES.keys()),
        help="要检查的表名列表，默认检查所有表",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "reports"),
        help="报告输出目录",
    )
    
    parser.add_argument(
        "--report-name",
        type=str,
        default=f"dwd_data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="报告文件名（不含扩展名）",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出模式",
    )
    
    parser.add_argument(
        "--cross-table",
        action="store_true",
        help="执行跨表一致性检查",
    )
    
    parser.add_argument(
        "--cross-table-only",
        action="store_true",
        help="仅执行跨表一致性检查（跳过单表检查）",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("DWD数据质量检查开始")
    
    # 创建报告生成器
    report_generator = DWDReportGenerator(
        output_dir=Path(args.output_dir),
        report_name=args.report_name,
    )
    
    success_count = 0
    fail_count = 0
    
    # 单表检查（除非指定 --cross-table-only）
    if not args.cross_table_only:
        logger.info(f"待检查表: {args.tables}")
        logger.info("=" * 60)
    
    # 逐表检查
    success_count = 0
    fail_count = 0
    
    for table_name in args.tables:
        logger.info(f"\n{'='*40}")
        logger.info(f"检查表: {table_name}")
        logger.info(f"{'='*40}")
        
        try:
            # 获取检查器类
            checker_class = CHECKER_CLASSES[table_name]
            
            # 创建检查器实例并运行
            checker = checker_class()
            report = checker.run()
            
            # 添加到报告生成器
            report_generator.add_report(report)
            
            if report.critical_count == 0 and report.error_count == 0:
                success_count += 1
                logger.info(f"✅ {table_name} 检查通过")
            else:
                fail_count += 1
                logger.warning(f"❌ {table_name} 检查发现问题")
                
        except FileNotFoundError as e:
            logger.error(f"❌ {table_name} 文件不存在: {e}")
            fail_count += 1
        except Exception as e:
            logger.error(f"❌ {table_name} 检查失败: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    # 跨表一致性检查（如果指定了 --cross-table 或 --cross-table-only）
    cross_table_report = None
    if args.cross_table or args.cross_table_only:
        logger.info("\n" + "=" * 60)
        logger.info("开始跨表一致性检查")
        logger.info("=" * 60)
        
        try:
            cross_checker = CrossTableChecker()
            cross_table_report = cross_checker.run()
            
            # 添加到报告生成器
            report_generator.add_cross_table_report(cross_table_report)
            
            if cross_table_report.critical_count == 0 and cross_table_report.error_count == 0:
                success_count += 1
                logger.info("✅ 跨表检查通过")
            else:
                fail_count += 1
                logger.warning("❌ 跨表检查发现问题")
                
        except Exception as e:
            logger.error(f"❌ 跨表检查失败: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    # 生成并保存报告
    logger.info("\n" + "=" * 60)
    logger.info("生成检查报告...")
    
    try:
        report_path = report_generator.save()
        logger.info(f"报告已保存: {report_path}")
    except Exception as e:
        logger.error(f"报告保存失败: {e}")
    
    # 打印摘要
    report_generator.print_summary()
    
    # 结果统计
    logger.info("\n" + "=" * 60)
    logger.info(f"检查完成: {success_count} 成功, {fail_count} 失败")
    logger.info("=" * 60)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
