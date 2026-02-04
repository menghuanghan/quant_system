#!/usr/bin/env python
"""
DWD全量数据处理脚本 - 一键生成八张DWD宽表

核心表 (core):
1. dwd_stock_price - 基础量价宽表（后复权价格、VWAP、收益率等）
2. dwd_stock_fundamental - PIT基本面宽表（财务指标、估值等）
3. dwd_stock_status - 状态与风险掩码表（ST、涨跌停、交易状态等）

扩展表 (extended):
4. dwd_money_flow - 资金博弈宽表（资金流向、融资融券、龙虎榜、沪深港通等）
5. dwd_chip_structure - 筹码结构宽表（十大股东、股本结构、解禁等）
6. dwd_stock_industry - 行业分类宽表（申万行业分类）
7. dwd_event_signal - 事件信号宽表（回购、分红、股权质押等）
8. dwd_macro_env - 宏观环境宽表（GDP、CPI、利率、情绪指标等）

特性：
- 全程cuDF GPU加速
- 支持自定义日期范围
- 详细的进度和性能统计

用法：
    # 处理全部八张表（默认2021-2025）
    python scripts/run_dwd_full.py
    
    # 仅处理3张核心表
    python scripts/run_dwd_full.py --tables core
    
    # 仅处理5张扩展表
    python scripts/run_dwd_full.py --tables extended
    
    # 处理指定表
    python scripts/run_dwd_full.py --tables price money_flow industry
    
    # 自定义日期范围
    python scripts/run_dwd_full.py --start-date 2024-01-01 --end-date 2024-12-31
    
    # 使用CPU模式（不推荐）
    python scripts/run_dwd_full.py --no-gpu
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.processors.structured.dwd import (
    MarketDataProcessor,
    FundamentalProcessor,
    StatusProcessor,
    MoneyFlowProcessor,
    ChipStructureProcessor,
    IndustryProcessor,
    EventSignalProcessor,
    MacroEnvProcessor,
)


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/dwd_full_processing.log'),
        ]
    )


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.2f}分钟"
    else:
        return f"{seconds/3600:.2f}小时"


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def process_price_table(
    use_gpu: bool,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """处理价格宽表"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("处理 dwd_stock_price（基础量价宽表）")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    processor = MarketDataProcessor(
        use_gpu=use_gpu,
        start_date=start_date,
        end_date=end_date,
    )
    
    df = processor.run()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # 获取输出文件大小
    output_file = processor.output_path
    file_size = output_file.stat().st_size if output_file.exists() else 0
    
    result = {
        'name': 'dwd_stock_price',
        'rows': len(df),
        'elapsed': elapsed,
        'file_size': file_size,
        'success': True,
    }
    
    logger.info(f"✓ dwd_stock_price 处理完成")
    logger.info(f"  - 行数: {result['rows']:,}")
    logger.info(f"  - 耗时: {format_time(elapsed)}")
    logger.info(f"  - 文件大小: {format_size(file_size)}")
    logger.info("")
    
    return result


def process_fundamental_table(
    use_gpu: bool,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """处理基本面宽表"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("处理 dwd_stock_fundamental（PIT基本面宽表）")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    processor = FundamentalProcessor(
        use_gpu=use_gpu,
        start_date=start_date,
        end_date=end_date,
    )
    
    df = processor.run()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # 获取输出文件大小
    output_file = processor.output_path
    file_size = output_file.stat().st_size if output_file.exists() else 0
    
    result = {
        'name': 'dwd_stock_fundamental',
        'rows': len(df),
        'elapsed': elapsed,
        'file_size': file_size,
        'success': True,
    }
    
    logger.info(f"✓ dwd_stock_fundamental 处理完成")
    logger.info(f"  - 行数: {result['rows']:,}")
    logger.info(f"  - 耗时: {format_time(elapsed)}")
    logger.info(f"  - 文件大小: {format_size(file_size)}")
    logger.info("")
    
    return result


def process_status_table(
    use_gpu: bool,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """处理状态宽表"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("处理 dwd_stock_status（状态与风险掩码表）")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    processor = StatusProcessor(
        use_gpu=use_gpu,
        start_date=start_date,
        end_date=end_date,
    )
    
    df = processor.run()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # 获取输出文件大小
    output_file = processor.output_path
    file_size = output_file.stat().st_size if output_file.exists() else 0
    
    result = {
        'name': 'dwd_stock_status',
        'rows': len(df),
        'elapsed': elapsed,
        'file_size': file_size,
        'success': True,
    }
    
    logger.info(f"✓ dwd_stock_status 处理完成")
    logger.info(f"  - 行数: {result['rows']:,}")
    logger.info(f"  - 耗时: {format_time(elapsed)}")
    logger.info(f"  - 文件大小: {format_size(file_size)}")
    logger.info("")
    
    return result


# ============== 扩展宽表处理函数 ==============

def process_money_flow_table(
    use_gpu: bool,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """处理资金博弈宽表"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("处理 dwd_money_flow（资金博弈宽表）")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    processor = MoneyFlowProcessor(
        use_gpu=use_gpu,
        start_date=start_date,
        end_date=end_date,
    )
    
    df = processor.run()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    output_file = processor.output_path
    file_size = output_file.stat().st_size if output_file.exists() else 0
    
    result = {
        'name': 'dwd_money_flow',
        'rows': len(df),
        'elapsed': elapsed,
        'file_size': file_size,
        'success': True,
    }
    
    logger.info(f"✓ dwd_money_flow 处理完成")
    logger.info(f"  - 行数: {result['rows']:,}")
    logger.info(f"  - 耗时: {format_time(elapsed)}")
    logger.info(f"  - 文件大小: {format_size(file_size)}")
    logger.info("")
    
    return result


def process_chip_structure_table(
    use_gpu: bool,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """处理筹码结构宽表"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("处理 dwd_chip_structure（筹码结构宽表）")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    processor = ChipStructureProcessor(
        use_gpu=use_gpu,
        start_date=start_date,
        end_date=end_date,
    )
    
    df = processor.run()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    output_file = processor.output_path
    file_size = output_file.stat().st_size if output_file.exists() else 0
    
    result = {
        'name': 'dwd_chip_structure',
        'rows': len(df),
        'elapsed': elapsed,
        'file_size': file_size,
        'success': True,
    }
    
    logger.info(f"✓ dwd_chip_structure 处理完成")
    logger.info(f"  - 行数: {result['rows']:,}")
    logger.info(f"  - 耗时: {format_time(elapsed)}")
    logger.info(f"  - 文件大小: {format_size(file_size)}")
    logger.info("")
    
    return result


def process_industry_table(
    use_gpu: bool,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """处理行业分类宽表"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("处理 dwd_stock_industry（行业分类宽表）")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    processor = IndustryProcessor(
        use_gpu=use_gpu,
        start_date=start_date,
        end_date=end_date,
    )
    
    df = processor.run()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    output_file = processor.output_path
    file_size = output_file.stat().st_size if output_file.exists() else 0
    
    result = {
        'name': 'dwd_stock_industry',
        'rows': len(df),
        'elapsed': elapsed,
        'file_size': file_size,
        'success': True,
    }
    
    logger.info(f"✓ dwd_stock_industry 处理完成")
    logger.info(f"  - 行数: {result['rows']:,}")
    logger.info(f"  - 耗时: {format_time(elapsed)}")
    logger.info(f"  - 文件大小: {format_size(file_size)}")
    logger.info("")
    
    return result


def process_event_signal_table(
    use_gpu: bool,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """处理事件信号宽表"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("处理 dwd_event_signal（事件信号宽表）")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    processor = EventSignalProcessor(
        use_gpu=use_gpu,
        start_date=start_date,
        end_date=end_date,
    )
    
    df = processor.run()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    output_file = processor.output_path
    file_size = output_file.stat().st_size if output_file.exists() else 0
    
    result = {
        'name': 'dwd_event_signal',
        'rows': len(df),
        'elapsed': elapsed,
        'file_size': file_size,
        'success': True,
    }
    
    logger.info(f"✓ dwd_event_signal 处理完成")
    logger.info(f"  - 行数: {result['rows']:,}")
    logger.info(f"  - 耗时: {format_time(elapsed)}")
    logger.info(f"  - 文件大小: {format_size(file_size)}")
    logger.info("")
    
    return result


def process_macro_env_table(
    use_gpu: bool,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """处理宏观环境宽表"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("处理 dwd_macro_env（宏观环境宽表）")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    processor = MacroEnvProcessor(
        use_gpu=use_gpu,
        start_date=start_date,
        end_date=end_date,
    )
    
    df = processor.run()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    output_file = processor.output_path
    file_size = output_file.stat().st_size if output_file.exists() else 0
    
    result = {
        'name': 'dwd_macro_env',
        'rows': len(df),
        'elapsed': elapsed,
        'file_size': file_size,
        'success': True,
    }
    
    logger.info(f"✓ dwd_macro_env 处理完成")
    logger.info(f"  - 行数: {result['rows']:,}")
    logger.info(f"  - 耗时: {format_time(elapsed)}")
    logger.info(f"  - 文件大小: {format_size(file_size)}")
    logger.info("")
    
    return result


def print_summary(results: list):
    """打印处理摘要"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("处理摘要")
    logger.info("=" * 70)
    
    total_rows = sum(r['rows'] for r in results)
    total_time = sum(r['elapsed'] for r in results)
    total_size = sum(r['file_size'] for r in results)
    
    logger.info(f"\n{'表名':<30} {'行数':>15} {'耗时':>15} {'文件大小':>15}")
    logger.info("-" * 80)
    
    for result in results:
        logger.info(
            f"{result['name']:<30} "
            f"{result['rows']:>15,} "
            f"{format_time(result['elapsed']):>15} "
            f"{format_size(result['file_size']):>15}"
        )
    
    logger.info("-" * 80)
    logger.info(
        f"{'总计':<30} "
        f"{total_rows:>15,} "
        f"{format_time(total_time):>15} "
        f"{format_size(total_size):>15}"
    )
    
    logger.info("")
    logger.info(f"✓ 全部处理完成！")
    logger.info(f"  平均速度: {total_rows/total_time:,.0f} 行/秒")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description='DWD全量数据处理 - 一键生成八张DWD宽表（3核心+5扩展）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        '--tables',
        nargs='+',
        choices=[
            'price', 'fundamental', 'status',  # 核心表
            'money_flow', 'chip_structure', 'industry', 'event_signal', 'macro_env',  # 扩展表
            'all', 'core', 'extended'  # 快捷选项
        ],
        default=['all'],
        help='要处理的表（默认: all）。可选: core(3张核心表), extended(5张扩展表), all(全部8张表)'
    )
    
    parser.add_argument(
        '--start-date',
        default='2021-01-01',
        help='开始日期 (YYYY-MM-DD，默认: 2021-01-01)'
    )
    
    parser.add_argument(
        '--end-date',
        default='2025-12-31',
        help='结束日期 (YYYY-MM-DD，默认: 2025-12-31)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='使用CPU模式（不推荐）'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细日志'
    )
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # 确定要处理的表
    CORE_TABLES = ['price', 'fundamental', 'status']
    EXTENDED_TABLES = ['money_flow', 'chip_structure', 'industry', 'event_signal', 'macro_env']
    ALL_TABLES = CORE_TABLES + EXTENDED_TABLES
    
    if 'all' in args.tables:
        tables_to_process = ALL_TABLES
    elif 'core' in args.tables:
        tables_to_process = CORE_TABLES
    elif 'extended' in args.tables:
        tables_to_process = EXTENDED_TABLES
    else:
        tables_to_process = args.tables
    
    # 打印配置
    use_gpu = not args.no_gpu
    logger.info("=" * 70)
    logger.info("DWD全量数据处理")
    logger.info("=" * 70)
    logger.info(f"日期范围: {args.start_date} 至 {args.end_date}")
    logger.info(f"处理表: {', '.join(tables_to_process)}")
    logger.info(f"加速模式: {'GPU (cuDF)' if use_gpu else 'CPU (pandas)'}")
    logger.info("")
    
    # 开始处理
    overall_start = datetime.now()
    results = []
    
    try:
        # 处理价格表
        if 'price' in tables_to_process:
            result = process_price_table(use_gpu, args.start_date, args.end_date)
            results.append(result)
        
        # 处理基本面表
        if 'fundamental' in tables_to_process:
            result = process_fundamental_table(use_gpu, args.start_date, args.end_date)
            results.append(result)
        
        # 处理状态表
        if 'status' in tables_to_process:
            result = process_status_table(use_gpu, args.start_date, args.end_date)
            results.append(result)
        
        # ========== 扩展表 ==========
        
        # 处理资金博弈表
        if 'money_flow' in tables_to_process:
            result = process_money_flow_table(use_gpu, args.start_date, args.end_date)
            results.append(result)
        
        # 处理筹码结构表
        if 'chip_structure' in tables_to_process:
            result = process_chip_structure_table(use_gpu, args.start_date, args.end_date)
            results.append(result)
        
        # 处理行业分类表
        if 'industry' in tables_to_process:
            result = process_industry_table(use_gpu, args.start_date, args.end_date)
            results.append(result)
        
        # 处理事件信号表
        if 'event_signal' in tables_to_process:
            result = process_event_signal_table(use_gpu, args.start_date, args.end_date)
            results.append(result)
        
        # 处理宏观环境表
        if 'macro_env' in tables_to_process:
            result = process_macro_env_table(use_gpu, args.start_date, args.end_date)
            results.append(result)
        
        # 打印摘要
        overall_elapsed = (datetime.now() - overall_start).total_seconds()
        logger.info(f"总耗时: {format_time(overall_elapsed)}")
        print_summary(results)
        
        return 0
        
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
