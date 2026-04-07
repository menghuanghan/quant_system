#!/usr/bin/env python
"""
DWD数据处理脚本（全量 + 增量）

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

运行模式：
1. 全量模式（默认）
   - 读取 data/raw/structured
   - 输出到 data/processed/structured/dwd

2. 增量模式（传入 --latest-date）
   - 使用 DuckDB 合并 full raw + increment raw(date<=latest-date)
   - 复用现有 Processor 计算逻辑
   - 仅截取 trade_date == latest-date 的行
   - 输出到 data/processed/structured_increment/dwd/date=YYYY-MM-DD
   - 不写入、不覆盖 data/processed/structured/dwd
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.processors.structured.dwd import (  # noqa: E402
    ChipStructureProcessor,
    EventSignalProcessor,
    FundamentalProcessor,
    IndustryProcessor,
    MacroEnvProcessor,
    MarketDataProcessor,
    MoneyFlowProcessor,
    StatusProcessor,
)


TABLE_SPECS: Dict[str, Dict[str, Any]] = {
    "price": {
        "result_name": "dwd_stock_price",
        "title": "基础量价宽表",
        "processor": MarketDataProcessor,
    },
    "fundamental": {
        "result_name": "dwd_stock_fundamental",
        "title": "PIT基本面宽表",
        "processor": FundamentalProcessor,
    },
    "status": {
        "result_name": "dwd_stock_status",
        "title": "状态与风险掩码表",
        "processor": StatusProcessor,
    },
    "money_flow": {
        "result_name": "dwd_money_flow",
        "title": "资金博弈宽表",
        "processor": MoneyFlowProcessor,
    },
    "chip_structure": {
        "result_name": "dwd_chip_structure",
        "title": "筹码结构宽表",
        "processor": ChipStructureProcessor,
    },
    "industry": {
        "result_name": "dwd_stock_industry",
        "title": "行业分类宽表",
        "processor": IndustryProcessor,
    },
    "event_signal": {
        "result_name": "dwd_event_signal",
        "title": "事件信号宽表",
        "processor": EventSignalProcessor,
    },
    "macro_env": {
        "result_name": "dwd_macro_env",
        "title": "宏观环境宽表",
        "processor": MacroEnvProcessor,
    },
}

CORE_TABLES = ["price", "fundamental", "status"]
EXTENDED_TABLES = ["money_flow", "chip_structure", "industry", "event_signal", "macro_env"]
ALL_TABLES = CORE_TABLES + EXTENDED_TABLES


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO

    log_path = PROJECT_ROOT / "logs" / "dwd_full_processing.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path), encoding="utf-8"),
        ],
        force=True,
    )


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    if seconds < 3600:
        return f"{seconds / 60:.2f}分钟"
    return f"{seconds / 3600:.2f}小时"


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def normalize_date_yyyy_mm_dd(date_str: str) -> str:
    """标准化日期为 YYYY-MM-DD。"""
    s = str(date_str).strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
    return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")


def _is_subpath(path: Path, parent: Path) -> bool:
    """判断 path 是否在 parent 下。"""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def filter_latest_trade_date(df: Any, latest_date: str) -> Any:
    """仅保留 trade_date == latest_date 的行。"""
    if "trade_date" not in df.columns:
        raise ValueError("DWD结果缺少 trade_date 列，无法截取增量数据")

    target_compact = latest_date.replace("-", "")
    date_text = df["trade_date"].astype(str)
    normalized = date_text.str.replace("-", "").str.replace("/", "").str.slice(0, 8)
    return df[normalized == target_compact]


def process_single_table(
    *,
    processor_cls: Type,
    result_name: str,
    title: str,
    use_gpu: bool,
    start_date: str,
    end_date: str,
    incremental_mode: bool,
    latest_date: Optional[str],
    increment_output_dir: Optional[Path],
    input_provider: Optional[Any],
) -> Dict[str, Any]:
    """执行单张表处理（全量或增量）。"""
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("处理 %s（%s）", result_name, title)
    logger.info("=" * 70)

    start_time = datetime.now()

    processor = processor_cls(
        use_gpu=use_gpu,
        start_date=start_date,
        end_date=end_date,
    )

    if input_provider is not None:
        processor.input_provider = input_provider

    if incremental_mode:
        if latest_date is None or increment_output_dir is None:
            raise ValueError("增量模式缺少 latest_date 或 increment_output_dir")

        full_output_root = (PROJECT_ROOT / "data" / "processed" / "structured" / "dwd").resolve()

        df_all = processor.process()
        rows_before_filter = len(df_all)
        df_out = filter_latest_trade_date(df_all, latest_date)

        output_file = (increment_output_dir / Path(processor.output_path).name).resolve()

        # 安全保护：增量模式禁止写全量目录
        if _is_subpath(output_file, full_output_root):
            raise ValueError(f"增量模式输出路径非法（指向全量目录）: {output_file}")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        processor.save_parquet(df_out, output_file, index=False)
        rows_out = len(df_out)
    else:
        df_out = processor.run()
        output_file = Path(processor.output_path).resolve()
        rows_before_filter = len(df_out)
        rows_out = rows_before_filter

    elapsed = (datetime.now() - start_time).total_seconds()
    file_size = output_file.stat().st_size if output_file.exists() else 0

    result = {
        "name": result_name,
        "rows": rows_out,
        "rows_before_filter": rows_before_filter,
        "elapsed": elapsed,
        "file_size": file_size,
        "output_path": str(output_file),
        "success": True,
    }

    logger.info("✓ %s 处理完成", result_name)
    if incremental_mode:
        logger.info("  - 行数(处理结果): %s", f"{rows_before_filter:,}")
        logger.info("  - 行数(增量截取): %s", f"{rows_out:,}")
    else:
        logger.info("  - 行数: %s", f"{rows_out:,}")
    logger.info("  - 耗时: %s", format_time(elapsed))
    logger.info("  - 文件大小: %s", format_size(file_size))
    logger.info("  - 输出文件: %s", output_file)
    logger.info("")

    return result


def print_summary(results: List[Dict[str, Any]], incremental_mode: bool = False):
    """打印处理摘要"""
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("处理摘要")
    logger.info("=" * 70)

    total_rows = sum(r["rows"] for r in results)
    total_rows_before = sum(r.get("rows_before_filter", r["rows"]) for r in results)
    total_time = sum(r["elapsed"] for r in results)
    total_size = sum(r["file_size"] for r in results)

    if incremental_mode:
        logger.info(f"\n{'表名':<24} {'处理行数':>14} {'增量行数':>14} {'耗时':>12} {'文件大小':>14}")
        logger.info("-" * 90)
        for result in results:
            logger.info(
                f"{result['name']:<24} "
                f"{result.get('rows_before_filter', result['rows']):>14,} "
                f"{result['rows']:>14,} "
                f"{format_time(result['elapsed']):>12} "
                f"{format_size(result['file_size']):>14}"
            )
        logger.info("-" * 90)
        logger.info(
            f"{'总计':<24} "
            f"{total_rows_before:>14,} "
            f"{total_rows:>14,} "
            f"{format_time(total_time):>12} "
            f"{format_size(total_size):>14}"
        )
    else:
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
    logger.info("✓ 全部处理完成！")
    if total_time > 0:
        logger.info("  平均速度: %s 行/秒", f"{total_rows / total_time:,.0f}")
    logger.info("")


def resolve_tables(tables_arg: List[str]) -> List[str]:
    """解析表参数。"""
    if "all" in tables_arg:
        return ALL_TABLES
    if "core" in tables_arg:
        return CORE_TABLES
    if "extended" in tables_arg:
        return EXTENDED_TABLES
    return tables_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DWD数据处理 - 支持全量与增量模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--tables",
        nargs="+",
        choices=[
            "price", "fundamental", "status",
            "money_flow", "chip_structure", "industry", "event_signal", "macro_env",
            "all", "core", "extended",
        ],
        default=["all"],
        help="要处理的表（默认: all）。可选: core(3张核心表), extended(5张扩展表), all(全部8张表)",
    )

    parser.add_argument(
        "--start-date",
        default="2019-01-01",
        help="开始日期 (YYYY-MM-DD，默认: 2019-01-01)",
    )

    parser.add_argument(
        "--end-date",
        default="2025-12-31",
        help="结束日期 (YYYY-MM-DD，默认: 2025-12-31)",
    )

    parser.add_argument(
        "--latest-date",
        default=None,
        help="最新增量日期（YYYYMMDD 或 YYYY-MM-DD）。传入后自动切换增量模式，并强制 end-date=latest-date",
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="使用CPU模式（不推荐）",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细日志",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    tables_to_process = resolve_tables(args.tables)
    use_gpu = not args.no_gpu

    # 运行模式
    incremental_mode = args.latest_date is not None
    latest_date: Optional[str] = None
    effective_end_date = args.end_date
    increment_output_dir: Optional[Path] = None
    input_provider: Optional[Any] = None

    if incremental_mode:
        latest_date = normalize_date_yyyy_mm_dd(args.latest_date)
        effective_end_date = latest_date

        increment_output_dir = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "structured_increment"
            / "dwd"
            / f"date={latest_date}"
        )
        increment_output_dir.mkdir(parents=True, exist_ok=True)

        # DuckDB 增量输入提供器（延迟导入，避免全量模式下硬依赖）
        from src.data_pipeline.processors.structured.dwd.increment_duckdb_merger import (  # noqa: E402
            DuckDBIncrementInputProvider,
        )

        input_provider = DuckDBIncrementInputProvider(
            full_raw_root=PROJECT_ROOT / "data" / "raw" / "structured",
            increment_raw_root=PROJECT_ROOT / "data" / "raw" / "structured_increment",
            latest_date=latest_date,
            cache_enabled=True,
        )

    logger.info("=" * 70)
    logger.info("DWD数据处理")
    logger.info("=" * 70)
    logger.info("运行模式: %s", "增量模式" if incremental_mode else "全量模式")
    logger.info("日期范围: %s 至 %s", args.start_date, effective_end_date)
    if incremental_mode and latest_date is not None:
        logger.info("latest-date: %s", latest_date)
        if args.end_date != effective_end_date:
            logger.info("end-date 已自动调整为 latest-date: %s -> %s", args.end_date, effective_end_date)
        logger.info("增量输入分区: date<=%s", latest_date)
        logger.info("增量输出目录: %s", increment_output_dir)
        logger.info("增量分区数量: %s", len(input_provider.partition_dates) if input_provider else 0)
    logger.info("处理表: %s", ", ".join(tables_to_process))
    logger.info("加速模式: %s", "GPU (cuDF)" if use_gpu else "CPU (pandas)")
    logger.info("")

    overall_start = datetime.now()
    results = []

    try:
        for table_key in tables_to_process:
            spec = TABLE_SPECS[table_key]
            result = process_single_table(
                processor_cls=spec["processor"],
                result_name=spec["result_name"],
                title=spec["title"],
                use_gpu=use_gpu,
                start_date=args.start_date,
                end_date=effective_end_date,
                incremental_mode=incremental_mode,
                latest_date=latest_date,
                increment_output_dir=increment_output_dir,
                input_provider=input_provider,
            )
            results.append(result)

            # 释放单表覆写缓存，降低峰值内存
            if input_provider is not None:
                input_provider.clear_cache()

        overall_elapsed = (datetime.now() - overall_start).total_seconds()
        logger.info("总耗时: %s", format_time(overall_elapsed))
        print_summary(results, incremental_mode=incremental_mode)
        return 0

    except Exception as e:
        logger.error("处理失败: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
