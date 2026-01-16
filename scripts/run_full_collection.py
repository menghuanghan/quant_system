"""
全量数据采集示例脚本

本脚本演示如何使用全量数据采集调度器采集原始数据。

使用方法:
    # 采集全部数据域（近3年）
    python scripts/run_full_collection.py
    
    # 采集指定数据域
    python scripts/run_full_collection.py --domains metadata market_data
    
    # 指定日期范围
    python scripts/run_full_collection.py --start-date 20210101 --end-date 20251231
    
    # 断点续采
    python scripts/run_full_collection.py --skip-existing
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/full_collection.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='全量数据采集调度器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d'),
        help='开始日期（YYYYMMDD格式），默认近3年'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y%m%d'),
        help='结束日期（YYYYMMDD格式），默认今天'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/structured',
        help='输出目录'
    )
    
    parser.add_argument(
        '--domains',
        type=str,
        nargs='+',
        default=None,
        help='要采集的数据域列表，不指定则采集全部'
    )
    
    parser.add_argument(
        '--exclude-domains',
        type=str,
        nargs='+',
        default=None,
        help='要排除的数据域列表'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='跳过已存在的文件（断点续采）'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='失败重试次数'
    )
    
    parser.add_argument(
        '--list-tasks',
        action='store_true',
        help='列出所有可用任务并退出'
    )
    
    parser.add_argument(
        '--list-domains',
        action='store_true',
        help='列出所有数据域并退出'
    )
    
    return parser.parse_args()


def list_all_domains():
    """列出所有数据域"""
    from src.data_pipeline.scheduler.structured.full import (
        DOMAIN_NAMES,
        TASKS_BY_DOMAIN,
    )
    
    print("\n" + "=" * 60)
    print("可用数据域列表")
    print("=" * 60)
    
    for domain, name in DOMAIN_NAMES.items():
        task_count = len(TASKS_BY_DOMAIN.get(domain, []))
        print(f"  {domain:25s} - {name} ({task_count} 个任务)")
    
    print("=" * 60)


def list_all_available_tasks():
    """列出所有可用任务"""
    from src.data_pipeline.scheduler.structured.full import (
        list_all_tasks,
        get_task_count,
    )
    
    print("\n" + "=" * 60)
    print("可用采集任务列表")
    print("=" * 60)
    
    tasks = list_all_tasks()
    for domain, info in tasks.items():
        print(f"\n【{info['domain_name']}】({domain})")
        print("-" * 40)
        for task in info['tasks']:
            if not task['enabled'] or task['realtime']:
                continue
            category = "时间相关" if task['category'] == 'time_dependent' else "时间无关"
            scope = "全A股" if task['stock_scope'] == 'all_a' else "无"
            print(f"  {task['name']:30s} [{category}] [{scope}]")
            print(f"    └─ {task['description']}")
    
    stats = get_task_count()
    print("\n" + "=" * 60)
    print(f"任务统计: 总计 {stats['total_tasks']} 个, 启用 {stats['enabled_tasks']} 个")
    print(f"  - 时间相关任务: {stats['time_dependent_tasks']} 个")
    print(f"  - 时间无关任务: {stats['time_independent_tasks']} 个")
    print(f"  - 需遍历股票: {stats['stock_related_tasks']} 个")
    print("=" * 60)


def main():
    """主函数"""
    args = parse_args()
    
    # 确保日志目录存在
    Path('logs').mkdir(exist_ok=True)
    
    # 列出任务/域并退出
    if args.list_domains:
        list_all_domains()
        return
    
    if args.list_tasks:
        list_all_available_tasks()
        return
    
    # 导入调度器
    from src.data_pipeline.scheduler.structured.full import (
        FullCollectionScheduler,
        TaskStatus,
    )
    
    logger.info("=" * 60)
    logger.info("全量数据采集开始")
    logger.info("=" * 60)
    logger.info(f"日期范围: {args.start_date} ~ {args.end_date}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"数据域: {args.domains or '全部'}")
    logger.info(f"排除域: {args.exclude_domains or '无'}")
    logger.info(f"跳过已存在: {args.skip_existing}")
    logger.info("=" * 60)
    
    # 创建调度器
    scheduler = FullCollectionScheduler(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        max_retries=args.max_retries,
    )
    
    # 执行采集
    try:
        progress = scheduler.run_all(
            domains=args.domains,
            exclude_domains=args.exclude_domains,
        )
        
        # 输出汇总
        logger.info("\n" + "=" * 60)
        logger.info("采集完成汇总")
        logger.info("=" * 60)
        
        summary = scheduler.get_collection_summary()
        logger.info(f"日期范围: {summary['start_date']} ~ {summary['end_date']}")
        logger.info(f"输出目录: {summary['output_dir']}")
        logger.info(f"进度: {summary['progress']['progress_percent']}")
        logger.info(
            f"结果: 成功={summary['progress']['success_tasks']}, "
            f"失败={summary['progress']['failed_tasks']}, "
            f"跳过={summary['progress']['skipped_tasks']}"
        )
        
        logger.info("\n各数据域采集情况:")
        for domain, stats in summary['domains'].items():
            logger.info(
                f"  {domain}: 成功={stats['success']}/{stats['total']}, "
                f"记录数={stats['records']}"
            )
        
        # 输出失败任务
        failed_results = [
            r for r in progress.results 
            if r.status == TaskStatus.FAILED
        ]
        if failed_results:
            logger.warning(f"\n失败任务 ({len(failed_results)} 个):")
            for r in failed_results[:10]:  # 只显示前10个
                logger.warning(f"  - {r.task_name}: {r.error_message}")
            if len(failed_results) > 10:
                logger.warning(f"  ... 还有 {len(failed_results)-10} 个失败任务")
        
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("用户中断采集")
    except Exception as e:
        logger.exception(f"采集过程发生异常: {e}")
        raise


if __name__ == '__main__':
    main()
