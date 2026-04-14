"""
结构化增量采集启动脚本

示例：
    python scripts/run_increment_collection.py --date 20260317
    python scripts/run_increment_collection.py --date 2026-03-17 --domains market_data trading_behavior
    python scripts/run_increment_collection.py --date 20260317 --include-etf
"""

import argparse
import json
import logging
import ssl
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# 忽略SSL证书验证（与全量脚本保持一致）
ssl._create_default_https_context = ssl._create_unverified_context

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="结构化增量采集调度器")
    parser.add_argument("--date", type=str, required=True, help="目标日期，支持 YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="data/raw/structured_increment", help="增量输出目录")
    parser.add_argument("--full-raw-dir", type=str, default="data/raw/structured", help="全量结构化目录（用于schema对齐）")
    parser.add_argument("--domains", nargs="+", type=str, default=None, help="仅执行指定数据域")
    parser.add_argument("--tasks", nargs="+", type=str, default=None, help="仅执行指定任务")
    parser.add_argument("--include-etf", action="store_true", help="是否采集 ETF 日线（默认关闭）")
    parser.add_argument("--etf-codes", nargs="+", type=str, default=None, help="指定ETF代码列表（仅在 --include-etf 时生效）")
    parser.add_argument(
        "--include-static",
        action="store_true",
        help="[已废弃] 静态参考表会在每次运行时自动比对并按需覆盖全量目录，无需显式开启",
    )
    parser.add_argument("--core-index-codes", nargs="+", type=str, default=None, help="核心指数代码列表")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在的增量文件")
    parser.add_argument("--dry-run", action="store_true", help="干跑模式：执行采集与对比逻辑，但不写入任何数据文件")
    parser.add_argument(
        "--static-tasks",
        nargs="+",
        type=str,
        default=None,
        help="仅刷新指定静态任务（默认刷新全部 OPTIONAL_STATIC_TASKS）",
    )
    parser.add_argument(
        "--static-report-file",
        type=str,
        default=None,
        help="静态刷新审计报告输出路径（JSON）；默认写入 reports/static_refresh_report_*.json",
    )
    parser.add_argument("--list-domains", action="store_true", help="列出可用数据域并退出")
    parser.add_argument("--list-tasks", action="store_true", help="列出可用任务并退出")
    parser.add_argument("--log-file", type=str, default="logs/increment_collection.log", help="日志文件")
    return parser.parse_args()


def main():
    args = parse_args()

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file, encoding="utf-8"),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    from src.data_pipeline.scheduler.structured.increment import StructuredIncrementScheduler

    scheduler = StructuredIncrementScheduler(
        date=args.date,
        output_dir=args.output_dir,
        full_raw_dir=args.full_raw_dir,
        include_etf=args.include_etf,
        include_static=args.include_static,
        etf_codes=args.etf_codes,
        core_index_codes=args.core_index_codes,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        static_task_names=args.static_tasks,
    )

    if args.include_static:
        logger.warning("--include-static 参数已废弃：静态参考表已默认自动刷新，不会写入 structured_increment")
    if args.dry_run:
        logger.warning("当前为 --dry-run 模式：不会写入 structured_increment 或 data/raw/structured")

    if args.list_domains:
        domains = scheduler.list_domains()
        print("\n可用数据域:")
        for k, v in domains.items():
            print(f"  {k:20s} - {v}")
        return

    if args.list_tasks:
        tasks = scheduler.list_tasks()
        print("\n可用任务:")
        for d, names in tasks.items():
            print(f"\n[{d}]")
            for n in names:
                print(f"  - {n}")
        return

    report = scheduler.run(domains=args.domains, task_names=args.tasks)

    static_report = report.static_refresh_report
    if static_report is not None:
        logger.info(
            "静态刷新摘要: total=%s, updated=%s, would_update=%s, unchanged=%s, failed=%s",
            static_report.total_tasks,
            static_report.updated,
            static_report.would_update,
            static_report.unchanged,
            static_report.failed,
        )

        changed_actions = {"updated", "would_update"}
        changed_items = [
            item for item in static_report.items if item.action in changed_actions
        ]
        if changed_items:
            logger.info("静态表发生变化（含dry-run拟更新）: %s", len(changed_items))
            for item in changed_items:
                logger.info(
                    "  - %s/%s: action=%s, rows %s->%s, hash %.12s->%.12s",
                    item.domain,
                    item.task_name,
                    item.action,
                    item.old_rows,
                    item.new_rows,
                    item.old_hash,
                    item.new_hash,
                )

        reports_dir = Path("reports/static_refresh")
        reports_dir.mkdir(parents=True, exist_ok=True)
        default_path = reports_dir / (
            f"static_refresh_report_{report.target_date.replace('-', '')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        report_path = Path(args.static_report_file) if args.static_report_file else default_path
        report_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "target_date": report.target_date,
            "generated_at": datetime.now().isoformat(),
            "dry_run": static_report.dry_run,
            "total_tasks": static_report.total_tasks,
            "updated": static_report.updated,
            "would_update": static_report.would_update,
            "unchanged": static_report.unchanged,
            "failed": static_report.failed,
            "items": [asdict(item) for item in static_report.items],
        }

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("静态刷新审计报告已写入: %s", report_path)

    logger.info("增量采集完成: date=%s, success=%s, failed=%s", report.target_date, report.success_tasks, report.failed_tasks)
    failed = [r for r in report.results if not r.success]
    if failed:
        logger.warning("失败任务数: %s", len(failed))
        for r in failed[:10]:
            logger.warning("  - %s/%s: %s", r.domain, r.task_name, r.error_message)


if __name__ == "__main__":
    main()
