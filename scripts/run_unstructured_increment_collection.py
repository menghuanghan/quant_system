"""
非结构化数据增量采集启动脚本

示例：
    python scripts/run_unstructured_increment_collection.py --date 20260320
    python scripts/run_unstructured_increment_collection.py --date 2026-03-20 --tasks announcements reports
    python scripts/run_unstructured_increment_collection.py --date 20260320 --skip-existing
"""

import argparse
import logging
import ssl
import sys
from datetime import datetime
from pathlib import Path

# 忽略SSL证书验证（与全量脚本保持一致）
ssl._create_default_https_context = ssl._create_unverified_context

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="非结构化增量采集调度器")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="目标日期，支持 YYYYMMDD 或 YYYY-MM-DD，默认当天",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/unstructured_increment",
        help="增量输出目录，默认 data/raw/unstructured_increment",
    )
    parser.add_argument(
        "--full-raw-dir",
        type=str,
        default="data/raw/unstructured",
        help="全量非结构化目录（用于schema对齐），默认 data/raw/unstructured",
    )
    parser.add_argument(
        "--processed-output-dir",
        type=str,
        default="data/processed/unstructured_increment",
        help="增量处理后输出目录，默认 data/processed/unstructured_increment",
    )
    parser.add_argument(
        "--full-processed-dir",
        type=str,
        default="data/processed/unstructured",
        help="全量非结构化处理目录（用于处理后schema对齐），默认 data/processed/unstructured",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=None,
        help="仅执行指定任务（announcements events news_cctv news_exchange policy_gov policy_ndrc reports）",
    )
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在增量文件")
    parser.add_argument(
        "--processing-skip-existing",
        action="store_true",
        help="跳过已存在的增量处理文件",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="跳过增量处理流水线，仅执行原始增量采集",
    )
    parser.add_argument(
        "--processing-model",
        type=str,
        default="qwen2.5:7b-instruct",
        help="处理流水线LLM模型名称，默认 qwen2.5:7b-instruct",
    )
    parser.add_argument(
        "--processing-ollama-host",
        type=str,
        default="http://localhost:11434",
        help="处理流水线Ollama服务地址，默认 http://localhost:11434",
    )
    parser.add_argument(
        "--processing-timeout",
        type=float,
        default=60.0,
        help="处理流水线LLM超时时间（秒），默认 60",
    )
    parser.add_argument(
        "--processing-use-gpu",
        action="store_true",
        default=True,
        help="处理流水线启用GPU加速（默认启用）",
    )
    parser.add_argument(
        "--processing-no-gpu",
        action="store_true",
        help="处理流水线禁用GPU加速",
    )
    parser.add_argument("--dry-run", action="store_true", help="干跑模式：不写入任何文件")
    parser.add_argument("--list-tasks", action="store_true", help="列出可用任务并退出")
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/unstructured_increment_collection.log",
        help="日志文件路径",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="启用DEBUG日志")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.processing_no_gpu:
        args.processing_use_gpu = False

    Path("logs").mkdir(exist_ok=True)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file, encoding="utf-8"),
        ],
        force=True,
    )

    logger = logging.getLogger(__name__)

    from src.data_pipeline.scheduler.unstructured.increment import UnstructuredIncrementScheduler

    scheduler = UnstructuredIncrementScheduler(
        date=args.date,
        output_dir=args.output_dir,
        full_raw_dir=args.full_raw_dir,
        processed_output_dir=args.processed_output_dir,
        full_processed_dir=args.full_processed_dir,
        skip_existing=args.skip_existing,
        processing_skip_existing=args.processing_skip_existing,
        enable_processing=not args.skip_processing,
        processing_model_name=args.processing_model,
        processing_ollama_host=args.processing_ollama_host,
        processing_timeout=args.processing_timeout,
        processing_use_gpu=args.processing_use_gpu,
        dry_run=args.dry_run,
    )

    if args.list_tasks:
        task_map = scheduler.list_tasks()
        print("\n可用任务:")
        for name, desc in task_map.items():
            print(f"  - {name:14s} : {desc}")
        return

    if args.dry_run:
        logger.warning("当前为 --dry-run 模式：不会写入增量文件")

    report = scheduler.run(task_names=args.tasks)

    logger.info(
        "增量采集结果: date=%s, success=%s, failed=%s, skipped=%s, total_records=%s, total_processed_records=%s",
        report.target_date,
        report.success_tasks,
        report.failed_tasks,
        report.skipped_tasks,
        report.total_records,
        report.total_processed_records,
    )

    if report.failed_tasks > 0:
        logger.warning("存在失败任务: %s", report.failed_tasks)
        for r in [x for x in report.results if not x.success][:20]:
            logger.warning("  - %s: %s", r.task_name, r.error_message)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
