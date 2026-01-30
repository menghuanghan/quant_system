#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据合并脚本

将增量数据合并到全量数据中，直接覆盖全量数据目录，支持GPU加速

功能：
- 合并全部数据域
- 合并指定数据域
- 合并指定数据域的指定任务
- 干运行模式（不实际写入）
- 备份功能（建议使用）

使用示例：
    # 合并全部数据域（建议先备份）
    python scripts/run_data_merge.py --backup
    
    # 合并指定数据域
    python scripts/run_data_merge.py --domains metadata --backup
    python scripts/run_data_merge.py --domains metadata market_data
    
    # 合并指定任务
    python scripts/run_data_merge.py --domains metadata --tasks trade_calendar suspend_info
    
    # 列出所有数据域和任务
    python scripts/run_data_merge.py --list-domains
    python scripts/run_data_merge.py --list-tasks
    python scripts/run_data_merge.py --list-tasks --domains metadata
    
    # 干运行（不实际写入）
    python scripts/run_data_merge.py --domains metadata --dry-run
    
    # 禁用GPU
    python scripts/run_data_merge.py --no-gpu
    
    # 输出到自定义目录（不覆盖原始数据）
    python scripts/run_data_merge.py --output-dir data/temp
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.scheduler.structured.merger import (
    DataMerger,
    MergeConfig,
    MERGE_TASKS_BY_DOMAIN,
    get_merge_tasks_by_domain,
    get_all_merge_tasks,
)


def setup_logging(log_file: str = None, verbose: bool = False):
    """配置日志"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # 创建日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def list_domains():
    """列出所有数据域"""
    print("\n可用的数据域:")
    print("=" * 50)
    for domain, tasks in MERGE_TASKS_BY_DOMAIN.items():
        enabled_tasks = [t for t in tasks if t.enabled]
        print(f"  {domain}: {len(enabled_tasks)} 个任务")
    print("=" * 50)


def list_tasks(domains: list = None):
    """列出所有任务"""
    if domains is None:
        domains = list(MERGE_TASKS_BY_DOMAIN.keys())
    
    print("\n合并任务列表:")
    print("=" * 80)
    
    for domain in domains:
        tasks = get_merge_tasks_by_domain(domain)
        if not tasks:
            continue
        
        print(f"\n【{domain}】")
        print("-" * 40)
        
        for task in tasks:
            status = "✓" if task.enabled else "✗"
            dir_flag = "[目录]" if task.is_directory else "[文件]"
            print(f"  {status} {task.name:<25} {dir_flag:<8} {task.description}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='数据合并脚本 - 将增量数据合并到全量数据中',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 基本参数
    parser.add_argument(
        '--domains', '-d',
        nargs='+',
        help='要合并的数据域列表（默认合并全部）'
    )
    parser.add_argument(
        '--tasks', '-t',
        nargs='+',
        help='要合并的任务列表（需配合 --domains 使用，仅支持单个数据域）'
    )
    
    # 路径配置
    parser.add_argument(
        '--inc-dir',
        default='data/raw/inc_structured',
        help='增量数据目录（默认: data/raw/inc_structured）'
    )
    parser.add_argument(
        '--full-dir',
        default='data/raw/structured',
        help='全量数据目录（默认: data/raw/structured）'
    )
    parser.add_argument(
        '--output-dir',
        default='data/raw/structured',
        help='输出目录（默认: data/raw/structured，直接覆盖全量数据）'
    )
    
    # 功能开关
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='干运行模式（不实际写入数据）'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='禁用GPU加速'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='启用备份（合并前备份全量数据）'
    )
    parser.add_argument(
        '--backup-dir',
        default='data/raw/structured_backup',
        help='备份目录（默认: data/raw/structured_backup）'
    )
    
    # 列表命令
    parser.add_argument(
        '--list-domains',
        action='store_true',
        help='列出所有数据域'
    )
    parser.add_argument(
        '--list-tasks',
        action='store_true',
        help='列出所有合并任务'
    )
    
    # 日志配置
    parser.add_argument(
        '--log-file',
        default='logs/data_merge.log',
        help='日志文件路径（默认: logs/data_merge.log）'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细日志'
    )
    
    args = parser.parse_args()
    
    # 处理列表命令
    if args.list_domains:
        list_domains()
        return 0
    
    if args.list_tasks:
        list_tasks(args.domains)
        return 0
    
    # 配置日志
    setup_logging(args.log_file, args.verbose)
    logger = logging.getLogger(__name__)
    
    # 显示配置信息
    print("\n" + "=" * 60)
    print("数据合并脚本")
    print("=" * 60)
    print(f"增量数据目录: {args.inc_dir}")
    print(f"全量数据目录: {args.full_dir}")
    print(f"输出数据目录: {args.output_dir}")
    print(f"GPU加速: {'禁用' if args.no_gpu else '启用'}")
    print(f"干运行模式: {'是' if args.dry_run else '否'}")
    print(f"备份功能: {'启用' if args.backup else '禁用'}")
    
    if args.domains:
        print(f"数据域: {', '.join(args.domains)}")
    else:
        print("数据域: 全部")
    
    if args.tasks:
        print(f"任务: {', '.join(args.tasks)}")
    
    print("=" * 60 + "\n")
    
    # 检查任务参数
    if args.tasks and (not args.domains or len(args.domains) != 1):
        logger.error("使用 --tasks 参数时，必须通过 --domains 指定单个数据域")
        return 1
    
    try:
        # 检查是否要求禁用GPU
        if args.no_gpu:
            logger.warning("当前合并器必须使用GPU加速，--no-gpu 参数被忽略")
        
        # 创建GPU合并器
        merger = DataMerger(
            inc_data_dir=args.inc_dir,
            full_data_dir=args.full_dir,
            output_dir=args.output_dir,
            backup_enabled=args.backup,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run,
        )
        
        # 执行合并
        start_time = datetime.now()
        
        if args.tasks:
            # 合并指定任务
            domain = args.domains[0]
            results = []
            for task_name in args.tasks:
                result = merger.merge_task(domain, task_name)
                results.append(result)
        elif args.domains:
            # 合并指定数据域
            results = merger.merge_all(domains=args.domains)
        else:
            # 合并全部
            results = merger.merge_all()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 生成报告
        report_df = merger.generate_report(results)
        
        # 显示统计
        print("\n" + "=" * 60)
        print("合并完成统计")
        print("=" * 60)
        
        success_count = len([r for r in results if r.status.value == 'success'])
        failed_count = len([r for r in results if r.status.value == 'failed'])
        skipped_count = len([r for r in results if r.status.value == 'skipped'])
        
        total_records_before = sum(r.records_before for r in results)
        total_records_increment = sum(r.records_increment for r in results)
        total_records_after = sum(r.records_after for r in results)
        
        print(f"总任务数: {len(results)}")
        print(f"  成功: {success_count}")
        print(f"  失败: {failed_count}")
        print(f"  跳过: {skipped_count}")
        print(f"\n数据统计:")
        print(f"  合并前总记录数: {total_records_before:,}")
        print(f"  增量数据记录数: {total_records_increment:,}")
        print(f"  合并后总记录数: {total_records_after:,}")
        print(f"\n耗时: {duration:.2f} 秒")
        
        # 显示失败任务
        if failed_count > 0:
            print("\n失败任务:")
            for r in results:
                if r.status.value == 'failed':
                    print(f"  - {r.domain}/{r.task_name}: {r.error_message}")
        
        print("=" * 60 + "\n")
        
        # 保存报告
        report_dir = Path(args.output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = report_dir / f'merge_report_{timestamp}.csv'
        report_df.to_csv(report_path, index=False)
        logger.info(f"合并报告已保存: {report_path}")
        
        return 0 if failed_count == 0 else 1
        
    except Exception as e:
        logger.exception(f"合并过程发生错误: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
