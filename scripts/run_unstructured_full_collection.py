"""
非结构化数据全量采集脚本

本脚本用于启动非结构化数据全量采集调度器，执行以下五类数据的采集：
- announcements: 上市公司公告
- events: 事件驱动型数据（并购重组、处罚公告、实控人变更、重大合同）
- news: 新闻（CCTV新闻、交易所公告）
- policy: 政策（国务院、发改委）
- reports: 券商研报

使用方法:
    # 采集全部类型数据（默认日期范围 2021.01.01-2025.12.31）
    python scripts/run_unstructured_full_collection.py
    
    # 采集指定类型数据
    python scripts/run_unstructured_full_collection.py --types announcements events
    
    # 指定日期范围
    python scripts/run_unstructured_full_collection.py --start-date 2021-01-01 --end-date 2025-12-31
    
    # 断点续采（跳过已存在的文件）
    python scripts/run_unstructured_full_collection.py --skip-existing
    
    # 列出所有可用任务
    python scripts/run_unstructured_full_collection.py --list-tasks
    
    # 列出所有数据类型
    python scripts/run_unstructured_full_collection.py --list-types
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import ssl

# 忽略SSL证书验证（解决部分网站的SSL握手报错）
ssl._create_default_https_context = ssl._create_unverified_context

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='非结构化数据全量采集调度器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据类型说明:
  announcements  上市公司公告（巨潮资讯）
  events         事件驱动型数据（并购重组、处罚、实控人变更、重大合同）
  news           新闻（CCTV新闻联播、交易所公告）
  policy         政策文件（国务院、发改委）
  reports        券商研报（东方财富）

存储结构:
  公告/事件/研报: data/raw/unstructured/{type}/{year}/{month}/{stock_code}.parquet
  新闻/政策:      data/raw/unstructured/{type}/{subdir}/{year}/{month}.parquet

示例:
  # 采集2024年全部数据
  python scripts/run_unstructured_full_collection.py --start-date 2024-01-01 --end-date 2024-12-31

  # 仅采集公告和研报
  python scripts/run_unstructured_full_collection.py --types announcements reports

  # 断点续采
  python scripts/run_unstructured_full_collection.py --skip-existing
        """
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2021-01-01',
        help='开始日期（YYYY-MM-DD格式），默认 2021-01-01'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-12-31',
        help='结束日期（YYYY-MM-DD格式），默认 2025-12-31'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/unstructured',
        help='输出目录，默认 data/raw/unstructured'
    )
    
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        default=None,
        choices=['announcements', 'events', 'news', 'policy', 'reports'],
        help='要采集的数据类型列表，不指定则采集全部'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='跳过已存在的文件（断点续采）'
    )
    
    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help='禁用检查点保存'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='失败重试次数，默认 3'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=30,
        help='每批股票数量（用于股票级别采集），默认 30'
    )
    
    parser.add_argument(
        '--stock-limit',
        type=int,
        default=None,
        help='限制股票数量（如 100=前100只，500=前500只，不指定则采集全部）'
    )
    
    parser.add_argument(
        '--list-tasks',
        action='store_true',
        help='列出所有可用任务并退出'
    )
    
    parser.add_argument(
        '--list-types',
        action='store_true',
        help='列出所有数据类型并退出'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/unstructured_full_collection.log',
        help='日志文件路径'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志（DEBUG级别）'
    )
    
    return parser.parse_args()


def list_all_types():
    """列出所有数据类型"""
    from src.data_pipeline.scheduler.unstructured.full import (
        TYPE_NAMES,
        TASKS_BY_TYPE,
    )
    
    print("\n" + "=" * 60)
    print("可用数据类型列表")
    print("=" * 60)
    
    for data_type, name in TYPE_NAMES.items():
        task_count = len(TASKS_BY_TYPE.get(data_type, []))
        print(f"  {data_type.value:15s} - {name} ({task_count} 个任务)")
    
    print("=" * 60)


def list_all_tasks():
    """列出所有可用任务"""
    from src.data_pipeline.scheduler.unstructured.full import (
        list_all_tasks as _list_all_tasks,
        get_task_count,
    )
    
    print("\n" + "=" * 60)
    print("可用采集任务列表")
    print("=" * 60)
    
    tasks = _list_all_tasks()
    for data_type, info in tasks.items():
        print(f"\n【{info['type_name']}】({data_type})")
        print("-" * 40)
        for task in info['tasks']:
            if not task['enabled']:
                continue
            stock_scope = "全A股" if task['stock_scope'] == 'all_a' else "无"
            storage = "按股票" if task['storage_pattern'] == 'by_stock' else "按月份"
            print(f"  {task['name']:30s} [{storage}] [{stock_scope}]")
            print(f"    └─ {task['description']}")
    
    stats = get_task_count()
    print("\n" + "=" * 60)
    print(f"任务统计: 总计 {stats['total_tasks']} 个, 启用 {stats['enabled_tasks']} 个")
    print(f"  - 需遍历股票: {stats['stock_related_tasks']} 个")
    print("各类型任务数:")
    for dtype, count in stats['by_type'].items():
        print(f"    {dtype}: {count} 个")
    print("=" * 60)


def main():
    """主函数"""
    args = parse_args()
    
    # 配置日志
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file, encoding='utf-8'),
        ],
        force=True
    )
    
    # 确保日志目录存在
    Path('logs').mkdir(exist_ok=True)
    
    # 列出类型/任务并退出
    if args.list_types:
        list_all_types()
        return
    
    if args.list_tasks:
        list_all_tasks()
        return
    
    # 导入调度器
    from src.data_pipeline.scheduler.unstructured.full import (
        UnstructuredFullCollectionScheduler,
        TaskStatus,
    )
    
    logger.info("=" * 60)
    logger.info("非结构化数据全量采集开始")
    logger.info("=" * 60)
    logger.info(f"日期范围: {args.start_date} ~ {args.end_date}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"数据类型: {args.types or '全部'}")
    logger.info(f"股票范围: {'前' + str(args.stock_limit) + '只' if args.stock_limit else '全部'}")
    logger.info(f"跳过已存在: {args.skip_existing}")
    logger.info(f"检查点: {'禁用' if args.no_checkpoint else '启用'}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info("=" * 60)
    
    # 创建调度器
    scheduler = UnstructuredFullCollectionScheduler(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        max_retries=args.max_retries,
        batch_size=args.batch_size,
        enable_checkpoint=not args.no_checkpoint,
        stock_limit=args.stock_limit,
    )
    
    # 执行采集
    try:
        progress = scheduler.run_all(data_types=args.types)
        
        # 输出汇总
        summary = scheduler.get_collection_summary()
        logger.info("\n采集结果汇总:")
        logger.info(f"  日期范围: {summary['start_date']} ~ {summary['end_date']}")
        logger.info(f"  进度: {summary['progress']['progress_percent']}")
        logger.info(
            f"  结果: 成功={summary['progress']['success_count']}, "
            f"失败={summary['progress']['failed_count']}, "
            f"跳过={summary['progress']['skipped_count']}"
        )
        logger.info(f"  总记录数: {summary['progress']['total_records']}")
        
        # 返回退出码
        if progress.failed_count > 0:
            logger.warning(f"存在 {progress.failed_count} 个失败任务")
            sys.exit(1)
        else:
            logger.info("全部采集任务完成!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.warning("用户中断采集")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"采集过程发生异常: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
