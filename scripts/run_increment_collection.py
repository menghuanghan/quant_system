"""
增量数据采集示例脚本

本脚本演示如何使用增量数据采集调度器采集原始数据。

使用方法:
    # 采集全部非实时数据
    python scripts/run_increment_collection.py --start-date 20190101 --end-date 20201231
    
    # 只采集时间相关数据
    python scripts/run_increment_collection.py --start-date 20190101 --end-date 20201231 --mode time_dependent
    
    # 只采集时间无关数据
    python scripts/run_increment_collection.py --mode time_independent
    
    # 采集指定数据域
    python scripts/run_increment_collection.py --start-date 20190101 --end-date 20201231 --domains metadata market_data
    
    # 断点续采
    python scripts/run_increment_collection.py --start-date 20190101 --end-date 20201231 --skip-existing
    
    # 列出所有可用任务
    python scripts/run_increment_collection.py --list-tasks
    
    # 按模式列出任务
    python scripts/run_increment_collection.py --list-tasks --mode time_dependent
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import ssl

# 忽略SSL证书验证（解决MOFCOM等网站及AkShare的SSL握手报错）
ssl._create_default_https_context = ssl._create_unverified_context

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='增量数据采集调度器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
采集模式说明:
  all             - 采集所有非实时数据（默认）
  time_dependent  - 只采集时间相关非实时数据（如日K线、财报等）
  time_independent - 只采集时间无关非实时数据（如股票列表、公司信息等）

示例:
  # 采集2019-2020年的所有时间相关数据
  python scripts/run_increment_collection.py --start-date 20190101 --end-date 20201231 --mode time_dependent
  
  # 只采集时间无关的基础数据
  python scripts/run_increment_collection.py --mode time_independent
  
  # 采集指定域的数据
  python scripts/run_increment_collection.py --start-date 20190101 --end-date 20201231 --domains market_data fundamental
"""
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='20190101',
        help='开始日期（YYYYMMDD格式），默认 20190101'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='20201231',
        help='结束日期（YYYYMMDD格式），默认 20201231'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/inc_structured',
        help='输出目录，默认 data/raw/inc_structured'
    )
    
    parser.add_argument(
        '--full-data-dir',
        type=str,
        default='data/raw/structured',
        help='全量数据目录（用于获取股票列表等元数据），默认 data/raw/structured'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'time_dependent', 'time_independent'],
        default='all',
        help='采集模式: all=全部, time_dependent=仅时间相关, time_independent=仅时间无关'
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
    
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=None,
        help='指定具体要执行的任务名称列表'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/increment_collection.log',
        help='日志文件路径'
    )
    
    return parser.parse_args()


def get_mode_enum(mode_str: str):
    """将字符串转换为CollectionMode枚举"""
    from src.data_pipeline.scheduler.structured.increment import CollectionMode
    
    mode_map = {
        'all': CollectionMode.ALL,
        'time_dependent': CollectionMode.TIME_DEPENDENT,
        'time_independent': CollectionMode.TIME_INDEPENDENT,
    }
    return mode_map.get(mode_str, CollectionMode.ALL)


def list_all_domains(mode_str: str = 'all'):
    """列出所有数据域"""
    from src.data_pipeline.scheduler.structured.increment import (
        DOMAIN_NAMES,
        TASKS_BY_DOMAIN,
        DataCategory,
    )
    
    mode = get_mode_enum(mode_str)
    mode_desc = {
        'all': "全部",
        'time_dependent': "时间相关",
        'time_independent': "时间无关",
    }
    
    print("\n" + "=" * 60)
    print(f"可用数据域列表 (采集模式: {mode_desc.get(mode_str, mode_str)})")
    print("=" * 60)
    
    for domain, name in DOMAIN_NAMES.items():
        tasks = TASKS_BY_DOMAIN.get(domain, [])
        enabled_tasks = [t for t in tasks if t.enabled and not t.realtime]
        
        if mode_str == 'time_dependent':
            enabled_tasks = [t for t in enabled_tasks if t.category == DataCategory.TIME_DEPENDENT]
        elif mode_str == 'time_independent':
            enabled_tasks = [t for t in enabled_tasks if t.category == DataCategory.TIME_INDEPENDENT]
        
        task_count = len(enabled_tasks)
        if task_count > 0:
            print(f"  {domain:25s} - {name} ({task_count} 个任务)")
    
    print("=" * 60)


def list_all_available_tasks(mode_str: str = 'all'):
    """列出所有可用任务"""
    from src.data_pipeline.scheduler.structured.increment import (
        list_all_tasks,
        get_task_count,
    )
    
    mode = get_mode_enum(mode_str)
    mode_desc = {
        'all': "全部",
        'time_dependent': "时间相关",
        'time_independent': "时间无关",
    }
    
    print("\n" + "=" * 60)
    print(f"可用采集任务列表 (采集模式: {mode_desc.get(mode_str, mode_str)})")
    print("=" * 60)
    
    tasks = list_all_tasks(mode)
    for domain, info in tasks.items():
        print(f"\n【{info['domain_name']}】({domain})")
        print("-" * 40)
        for task in info['tasks']:
            category = "时间相关" if task['category'] == 'time_dependent' else "时间无关"
            scope = "全A股" if task['stock_scope'] == 'all_a' else "无"
            print(f"  {task['name']:30s} [{category}] [{scope}]")
            print(f"    └─ {task['description']}")
    
    stats = get_task_count(mode)
    print("\n" + "=" * 60)
    print(f"任务统计 (模式: {mode_desc.get(mode_str, mode_str)})")
    print(f"  - 总计任务数: {stats['total_tasks']} 个")
    print(f"  - 时间相关任务: {stats['time_dependent_tasks']} 个")
    print(f"  - 时间无关任务: {stats['time_independent_tasks']} 个")
    print(f"  - 需遍历股票: {stats['stock_related_tasks']} 个")
    print("=" * 60)


def main():
    """主函数"""
    args = parse_args()
    
    # 配置日志
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file, encoding='utf-8'),
        ],
        force=True  # 允许重新配置
    )
    
    # 确保日志目录存在
    Path('logs').mkdir(exist_ok=True)
    
    # 列出任务/域并退出
    if args.list_domains:
        list_all_domains(args.mode)
        return
    
    if args.list_tasks:
        list_all_available_tasks(args.mode)
        return
    
    # 导入调度器
    from src.data_pipeline.scheduler.structured.increment import (
        IncrementCollectionScheduler,
        TaskStatus,
        CollectionMode,
    )
    
    # 获取采集模式
    mode = get_mode_enum(args.mode)
    mode_desc = {
        CollectionMode.ALL: "全部非实时数据",
        CollectionMode.TIME_DEPENDENT: "时间相关非实时数据",
        CollectionMode.TIME_INDEPENDENT: "时间无关非实时数据",
    }
    
    logger.info("=" * 60)
    logger.info("增量数据采集开始")
    logger.info("=" * 60)
    logger.info(f"日期范围: {args.start_date} ~ {args.end_date}")
    logger.info(f"采集模式: {mode_desc.get(mode, args.mode)}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"全量数据目录: {args.full_data_dir}")
    logger.info(f"数据域: {args.domains or '全部'}")
    logger.info(f"排除域: {args.exclude_domains or '无'}")
    logger.info(f"跳过已存在: {args.skip_existing}")
    logger.info("=" * 60)
    
    # 创建调度器
    scheduler = IncrementCollectionScheduler(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        mode=mode,
        skip_existing=args.skip_existing,
        max_retries=args.max_retries,
        full_data_dir=args.full_data_dir,
    )
    
    # 执行采集
    try:
        progress = scheduler.run_all(
            domains=args.domains,
            exclude_domains=args.exclude_domains,
            task_names=args.tasks
        )
        
        # 输出汇总
        logger.info("\n" + "=" * 60)
        logger.info("采集完成汇总")
        logger.info("=" * 60)
        
        summary = scheduler.get_collection_summary()
        logger.info(f"日期范围: {summary['start_date']} ~ {summary['end_date']}")
        logger.info(f"采集模式: {summary['mode']}")
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
