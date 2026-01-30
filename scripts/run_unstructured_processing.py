"""
非结构化数据处理启动器

一键启动对 data/raw/unstructured 下所有非结构化原始数据的pipeline处理

功能：
- 处理全部数据或部分数据（按类型筛选）
- 断点恢复（基于已处理的文件）
- 进度跟踪和报告
- 多种数据类型支持：
  * announcements: 公告（PDF）
  * reports: 研报（PDF）
  * events: 事件（PDF）
  * news/exchange: 交易所新闻
  * news/cctv: CCTV新闻（市场情绪分析）
  * policy/gov: 国务院政策（行业映射）
  * policy/ndrc: 发改委政策（行业映射）

使用示例：
    # 列出所有可处理的数据
    python scripts/run_unstructured_processing.py --list
    
    # 处理所有数据
    python scripts/run_unstructured_processing.py --all --year 2021
    
    # 只处理CCTV新闻
    python scripts/run_unstructured_processing.py --categories news/cctv --year 2021
    
    # 处理多个类型
    python scripts/run_unstructured_processing.py --categories announcements reports --year 2021
    
    # 强制重新处理（忽略已存在的文件）
    python scripts/run_unstructured_processing.py --categories news/cctv --year 2021 --force
    
    # 指定月份范围
    python scripts/run_unstructured_processing.py --categories news/cctv --year 2021 --start-month 1 --end-month 6
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.processors.unstructured.scheduler import (
    UnstructuredScheduler,
    ProcessingConfig,
    DataCategory,
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """配置日志"""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"unstructured_processing_{timestamp}.log"
    
    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    
    return logger


def list_available_data(raw_data_dir: Path) -> dict:
    """列出所有可处理的数据"""
    categories_info = {}
    
    # 映射：数据类别 -> 文件系统路径
    category_paths = {
        DataCategory.ANNOUNCEMENTS: "announcements",
        DataCategory.REPORTS: "reports",
        DataCategory.EVENTS: "events",
        DataCategory.EXCHANGE: "news/exchange",
        DataCategory.CCTV: "news/cctv",
        DataCategory.POLICY_GOV: "policy/gov",
        DataCategory.POLICY_NDRC: "policy/ndrc",
    }
    
    for category, rel_path in category_paths.items():
        category_path = raw_data_dir / rel_path
        if not category_path.exists():
            continue
        
        # 统计年月文件
        files_info = []
        for year_dir in sorted(category_path.glob("*")):
            if not year_dir.is_dir():
                continue
            year = year_dir.name
            for month_file in sorted(year_dir.glob("*.parquet")):
                month = month_file.stem
                file_size = month_file.stat().st_size / (1024 * 1024)  # MB
                
                # 读取记录数
                try:
                    df = pd.read_parquet(month_file)
                    record_count = len(df)
                except Exception:
                    record_count = 0
                
                files_info.append({
                    'year': year,
                    'month': month,
                    'file': str(month_file.relative_to(raw_data_dir)),
                    'size_mb': round(file_size, 2),
                    'records': record_count,
                })
        
        if files_info:
            categories_info[category.value] = {
                'category': category,
                'path': rel_path,
                'files': files_info,
                'total_files': len(files_info),
                'total_records': sum(f['records'] for f in files_info),
                'total_size_mb': round(sum(f['size_mb'] for f in files_info), 2),
            }
    
    return categories_info


def print_available_data(categories_info: dict):
    """打印可处理的数据列表"""
    if not categories_info:
        print("未发现可处理的数据文件")
        return
    
    print("=" * 80)
    print("可处理的非结构化数据")
    print("=" * 80)
    
    for cat_value, info in categories_info.items():
        category = info['category']
        print(f"\n📁 {cat_value}")
        print(f"   路径: data/raw/unstructured/{info['path']}")
        print(f"   文件数: {info['total_files']}")
        print(f"   记录数: {info['total_records']:,}")
        print(f"   总大小: {info['total_size_mb']:.2f} MB")
        
        # 特性说明
        features = []
        if category.requires_pdf:
            features.append("PDF提取")
        if category.requires_html:
            features.append("HTML解析")
        if category.requires_llm:
            features.append("LLM处理")
        if category.is_cctv:
            features.append("市场情绪分析")
        if category.is_policy:
            features.append("行业映射")
        
        if features:
            print(f"   特性: {', '.join(features)}")
        
        # 显示前3个文件
        if len(info['files']) <= 5:
            files_to_show = info['files']
        else:
            files_to_show = info['files'][:3]
        
        for file_info in files_to_show:
            print(f"      • {file_info['year']}/{file_info['month']}.parquet "
                  f"({file_info['records']:,} 条, {file_info['size_mb']:.2f} MB)")
        
        if len(info['files']) > 5:
            print(f"      ... 还有 {len(info['files']) - 3} 个文件")
    
    print("\n" + "=" * 80)
    print("使用 --categories 参数选择要处理的数据类型")
    print("例如: --categories news/cctv policy/gov")
    print("=" * 80)


def check_processed_status(categories_info: dict, processed_data_dir: Path) -> dict:
    """检查处理状态"""
    status_info = {}
    
    for cat_value, info in categories_info.items():
        category = info['category']
        
        # 映射到输出路径
        category_paths = {
            DataCategory.ANNOUNCEMENTS: "announcements",
            DataCategory.REPORTS: "reports",
            DataCategory.EVENTS: "events",
            DataCategory.EXCHANGE: "news/exchange",
            DataCategory.CCTV: "news/cctv",
            DataCategory.POLICY_GOV: "policy/gov",
            DataCategory.POLICY_NDRC: "policy/ndrc",
        }
        
        output_path = processed_data_dir / category_paths[category]
        
        processed_files = []
        if output_path.exists():
            for year_dir in output_path.glob("*"):
                if not year_dir.is_dir():
                    continue
                for month_file in year_dir.glob("*.parquet"):
                    processed_files.append({
                        'year': year_dir.name,
                        'month': month_file.stem,
                        'file': str(month_file),
                    })
        
        status_info[cat_value] = {
            'total_raw': info['total_files'],
            'processed': len(processed_files),
            'pending': info['total_files'] - len(processed_files),
            'processed_files': processed_files,
        }
    
    return status_info


def parse_category(cat_str: str) -> Optional[DataCategory]:
    """解析数据类别字符串"""
    category_map = {
        'announcements': DataCategory.ANNOUNCEMENTS,
        'announcement': DataCategory.ANNOUNCEMENTS,
        'reports': DataCategory.REPORTS,
        'report': DataCategory.REPORTS,
        'events': DataCategory.EVENTS,
        'event': DataCategory.EVENTS,
        'exchange': DataCategory.EXCHANGE,
        'news/exchange': DataCategory.EXCHANGE,
        'cctv': DataCategory.CCTV,
        'news/cctv': DataCategory.CCTV,
        'gov': DataCategory.POLICY_GOV,
        'policy/gov': DataCategory.POLICY_GOV,
        'ndrc': DataCategory.POLICY_NDRC,
        'policy/ndrc': DataCategory.POLICY_NDRC,
    }
    return category_map.get(cat_str.lower())


def main():
    parser = argparse.ArgumentParser(
        description="非结构化数据处理启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 数据选择
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可处理的数据'
    )
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='检查处理状态'
    )
    parser.add_argument(
        '--categories', '-c',
        nargs='+',
        help='要处理的数据类别（可多选）：announcements, reports, events, news/exchange, news/cctv, policy/gov, policy/ndrc'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='处理所有类别'
    )
    
    # 时间范围
    parser.add_argument(
        '--year', '-y',
        type=int,
        help='要处理的年份（必需，除非使用--list）'
    )
    parser.add_argument(
        '--start-month',
        type=int,
        default=1,
        help='起始月份（默认：1）'
    )
    parser.add_argument(
        '--end-month',
        type=int,
        default=12,
        help='结束月份（默认：12）'
    )
    parser.add_argument(
        '--month', '-m',
        type=int,
        help='只处理指定月份'
    )
    
    # 处理选项
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='强制重新处理（忽略已存在的文件）'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='跳过已存在的输出文件（默认）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批处理大小（默认：32）'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='启用GPU加速（需要cuDF）'
    )
    
    # LLM配置
    parser.add_argument(
        '--model',
        default='qwen2.5:7b-instruct',
        help='LLM模型名称（默认：qwen2.5:7b-instruct）'
    )
    parser.add_argument(
        '--ollama-host',
        default='http://localhost:11434',
        help='Ollama服务地址（默认：http://localhost:11434）'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='LLM调用超时时间（秒，默认：60）'
    )
    
    # 其他
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别（默认：INFO）'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='生成处理报告'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    # 路径配置
    raw_data_dir = project_root / "data" / "raw" / "unstructured"
    processed_data_dir = project_root / "data" / "processed" / "unstructured"
    
    # 列出可用数据
    categories_info = list_available_data(raw_data_dir)
    
    if args.list:
        print_available_data(categories_info)
        return 0
    
    if args.status:
        status_info = check_processed_status(categories_info, processed_data_dir)
        print("=" * 80)
        print("处理状态")
        print("=" * 80)
        for cat_value, status in status_info.items():
            print(f"\n{cat_value}:")
            print(f"   原始文件: {status['total_raw']}")
            print(f"   已处理: {status['processed']}")
            print(f"   待处理: {status['pending']}")
            if status['pending'] > 0:
                print(f"   进度: {status['processed']/status['total_raw']*100:.1f}%")
        print("=" * 80)
        return 0
    
    # 必须指定年份（除非只是list或status）
    if not args.year:
        parser.error("必须指定 --year 参数")
    
    # 确定要处理的类别
    if args.all:
        categories = [info['category'] for info in categories_info.values()]
        logger.info(f"将处理所有 {len(categories)} 个数据类别")
    elif args.categories:
        categories = []
        for cat_str in args.categories:
            category = parse_category(cat_str)
            if category is None:
                logger.error(f"无效的数据类别: {cat_str}")
                return 1
            if category.value not in categories_info:
                logger.warning(f"未找到数据: {category.value}")
                continue
            categories.append(category)
        
        if not categories:
            logger.error("未找到可处理的数据")
            return 1
    else:
        parser.error("必须指定 --categories 或 --all")
    
    # 确定月份范围
    if args.month:
        months = [args.month]
    else:
        months = list(range(args.start_month, args.end_month + 1))
    
    logger.info(f"处理配置:")
    logger.info(f"  年份: {args.year}")
    logger.info(f"  月份: {months}")
    logger.info(f"  类别: {[c.value for c in categories]}")
    logger.info(f"  强制重新处理: {args.force}")
    logger.info(f"  LLM模型: {args.model}")
    logger.info(f"  批处理大小: {args.batch_size}")
    logger.info(f"  GPU加速: {args.use_gpu}")
    
    # 创建处理配置
    config = ProcessingConfig(
        raw_data_dir=str(raw_data_dir),
        processed_data_dir=str(processed_data_dir),
        model_name=args.model,
        ollama_host=args.ollama_host,
        llm_timeout=args.timeout,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu,
        skip_existing=args.skip_existing and not args.force,
    )
    
    # 创建调度器
    scheduler = UnstructuredScheduler(config=config)
    
    # 统计信息
    total_tasks = len(categories) * len(months)
    completed_tasks = 0
    all_results = []
    
    print("\n" + "=" * 80)
    print(f"开始处理 {total_tasks} 个任务")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # 处理每个类别
    for category in categories:
        logger.info(f"\n处理类别: {category.value}")
        print(f"\n{'=' * 80}")
        print(f"📊 {category.value}")
        print(f"{'=' * 80}")
        
        for month in months:
            task_start = datetime.now()
            
            try:
                logger.info(f"处理: {args.year}/{month:02d}")
                print(f"\n⏳ {args.year}/{month:02d} ...", end=' ', flush=True)
                
                result = scheduler.process_month(
                    category=category,
                    year=args.year,
                    month=month,
                    force=args.force
                )
                
                all_results.append(result)
                completed_tasks += 1
                
                # 显示结果
                if result.skipped_count > 0 and result.total == 0:
                    print(f"⏭️  已跳过")
                elif result.total == 0:
                    print(f"📭 无数据")
                else:
                    success_rate = result.success_count / result.total * 100 if result.total > 0 else 0
                    print(f"✅ {result.success_count}/{result.total} "
                          f"({success_rate:.1f}%) "
                          f"耗时 {result.elapsed_time_seconds:.1f}s")
                    
                    if hasattr(result, 'avg_score') and result.avg_score is not None:
                        print(f"      平均得分: {result.avg_score:.2f}")
                
            except Exception as e:
                logger.error(f"处理失败: {category.value}/{args.year}/{month:02d}: {e}", exc_info=True)
                print(f"❌ 失败: {e}")
                completed_tasks += 1
            
            # 进度
            progress = completed_tasks / total_tasks * 100
            elapsed = (datetime.now() - start_time).total_seconds()
            eta = elapsed / completed_tasks * (total_tasks - completed_tasks) if completed_tasks > 0 else 0
            
            print(f"      进度: {completed_tasks}/{total_tasks} ({progress:.1f}%) "
                  f"| 已用时: {elapsed:.0f}s | 预计剩余: {eta:.0f}s")
    
    # 总结
    total_time = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("处理完成")
    print("=" * 80)
    
    total_records = sum(r.total for r in all_results)
    total_success = sum(r.success_count for r in all_results)
    total_failed = sum(r.failed_count for r in all_results)
    total_skipped = sum(r.skipped_count for r in all_results)
    
    print(f"\n总记录数: {total_records:,}")
    print(f"成功: {total_success:,} ({total_success/total_records*100:.1f}%)" if total_records > 0 else "成功: 0")
    print(f"失败: {total_failed:,}")
    print(f"跳过: {total_skipped:,}")
    print(f"总耗时: {total_time:.1f}s ({total_time/60:.1f} 分钟)")
    
    if total_success > 0:
        print(f"平均处理速度: {total_success/total_time:.1f} 条/秒")
    
    # 生成报告
    if args.report and all_results:
        report_dir = project_root / "reports"
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"unstructured_processing_{timestamp}.csv"
        
        report_data = []
        for result in all_results:
            report_data.append({
                'category': result.category.value,
                'year': result.year,
                'month': result.month,
                'total': result.total,
                'success': result.success_count,
                'failed': result.failed_count,
                'skipped': result.skipped_count,
                'elapsed_seconds': result.elapsed_time_seconds,
                'output_path': result.output_path or '',
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(report_file, index=False, encoding='utf-8')
        print(f"\n报告已保存: {report_file}")
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
