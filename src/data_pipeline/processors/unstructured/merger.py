"""
非结构化处理结果合并器

将 data/processed/unstructured 下按“类型/年份/月.parquet”分散存储的数据，
合并为“类型.parquet”的扁平化全量文件，便于后续统一读取。

示例：
    data/processed/unstructured/announcements/2021/01.parquet
    data/processed/unstructured/news/exchange/2021/01.parquet

合并后：
    data/processed/unstructured/announcements.parquet
    data/processed/unstructured/news_exchange.parquet
"""

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass
class UnstructuredMergeResult:
    """单个类型的合并结果"""

    category: str
    output_file: str
    file_count: int = 0
    total_rows: int = 0
    total_columns: int = 0
    success: bool = False
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class UnstructuredMergeReport:
    """合并任务总报告"""

    results: List[UnstructuredMergeResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def total_rows(self) -> int:
        return sum(r.total_rows for r in self.results if r.success)

    def summary(self) -> str:
        lines = [
            f"非结构化合并报告: 共 {self.total} 个类型, 成功 {self.success_count}, 失败 {self.failed_count}",
            f"总行数: {self.total_rows:,}",
            "-" * 80,
        ]
        for r in self.results:
            status = "✓" if r.success else "✗"
            lines.append(
                f"  {status} {r.category}: {r.file_count} 文件 -> {r.total_rows:,} 行"
                f" -> {Path(r.output_file).name}"
                + (f" [错误: {r.error_message}]" if r.error_message else "")
            )
        return "\n".join(lines)


class UnstructuredDataMerger:
    """非结构化处理结果合并器"""

    # 预期的类别顺序（用于稳定输出）；若发现其他类别会自动追加
    CATEGORY_ORDER = [
        "announcements",
        "events",
        "news/cctv",
        "news/exchange",
        "policy/gov",
        "policy/ndrc",
        "reports",
    ]

    def __init__(self, processed_dir: str = "data/processed/unstructured"):
        self.processed_dir = Path(processed_dir)
        if not self.processed_dir.exists():
            raise FileNotFoundError(f"处理后数据目录不存在: {self.processed_dir}")

    def scan(self) -> Dict[str, List[Path]]:
        """
        扫描待合并的月度文件。

        仅识别形如 {category}/{year}/{month}.parquet 的文件。
        """
        category_files: Dict[str, List[Path]] = {}

        for parquet_file in sorted(self.processed_dir.rglob("*.parquet")):
            # 跳过已生成的扁平化文件（如 announcements.parquet）
            rel = parquet_file.relative_to(self.processed_dir)
            if len(rel.parts) < 3:
                continue

            year_dir = parquet_file.parent.name
            if not year_dir.isdigit():
                continue

            category_path = parquet_file.parent.parent.relative_to(self.processed_dir)
            category = category_path.as_posix()

            category_files.setdefault(category, []).append(parquet_file)

        # 排序：先用预设顺序，再按字典序追加未预设类别
        ordered: Dict[str, List[Path]] = {}
        discovered = set(category_files.keys())

        for category in self.CATEGORY_ORDER:
            if category in category_files:
                ordered[category] = sorted(
                    category_files[category],
                    key=self._sort_key,
                )

        for category in sorted(discovered - set(self.CATEGORY_ORDER)):
            ordered[category] = sorted(
                category_files[category],
                key=self._sort_key,
            )

        return ordered

    def merge_all(
        self,
        categories: Optional[List[str]] = None,
        remove_source_dirs: bool = False,
    ) -> UnstructuredMergeReport:
        """
        合并所有（或指定）类型的数据。

        Args:
            categories: 指定类别列表（例如 ["announcements", "news/exchange"]），None 表示全部
            remove_source_dirs: 合并成功后是否删除源目录（按月目录）
        """
        scanned = self.scan()
        report = UnstructuredMergeReport()

        if categories is not None:
            category_set = {c.strip().strip("/") for c in categories if c and c.strip()}
            scanned = {k: v for k, v in scanned.items() if k in category_set}

        if not scanned:
            logger.warning("未发现可合并的非结构化月度文件")
            return report

        logger.info(f"发现 {len(scanned)} 个类型待合并")
        for category, files in scanned.items():
            logger.info(f"  {category}: {len(files)} 个月度文件")

        for category, files in scanned.items():
            result = self._merge_category(
                category=category,
                files=files,
                remove_source_dirs=remove_source_dirs,
            )
            report.results.append(result)

        logger.info(report.summary())
        return report

    def _merge_category(
        self,
        category: str,
        files: List[Path],
        remove_source_dirs: bool,
    ) -> UnstructuredMergeResult:
        """合并单个类别"""
        output_name = category.replace("/", "_") + ".parquet"
        output_path = self.processed_dir / output_name

        result = UnstructuredMergeResult(
            category=category,
            output_file=str(output_path),
            file_count=len(files),
        )

        start = time.time()
        try:
            tables: List[pa.Table] = []

            for file_path in files:
                try:
                    table = pq.read_table(file_path)
                    tables.append(table)
                except Exception as e:
                    logger.warning(f"读取失败，跳过文件 {file_path}: {e}")

            if tables:
                # 严格按原schema拼接，不做类型提升/转换
                try:
                    merged_table = pa.concat_tables(tables, promote_options="none")
                except TypeError:
                    # 兼容旧版pyarrow
                    merged_table = pa.concat_tables(tables, promote=False)

                pq.write_table(merged_table, output_path, compression="snappy")
                result.total_rows = merged_table.num_rows
                result.total_columns = merged_table.num_columns
            else:
                # 全为空或全部读取失败时，写空文件（结构未知）
                pq.write_table(pa.table({}), output_path, compression="snappy")
                result.total_rows = 0
                result.total_columns = 0

            if remove_source_dirs:
                self._cleanup_category_dir(category)

            result.success = True
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"合并失败 {category}: {e}")
        finally:
            result.duration_seconds = time.time() - start

        return result

    def _cleanup_category_dir(self, category: str):
        """删除类别源目录，并向上清理空目录（不删除 processed_dir 根）"""
        category_dir = self.processed_dir / category
        if category_dir.exists() and category_dir.is_dir():
            shutil.rmtree(category_dir)

        current = category_dir.parent
        while current != self.processed_dir and current.exists():
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    @staticmethod
    def _sort_key(file_path: Path) -> tuple:
        """按 year/month 排序，保证拼接顺序稳定"""
        year = file_path.parent.name
        month = file_path.stem

        try:
            year_num = int(year)
        except ValueError:
            year_num = 9999

        try:
            month_num = int(month)
        except ValueError:
            month_num = 99

        return year_num, month_num, file_path.name

def merge_unstructured_processed(
    processed_dir: str = "data/processed/unstructured",
    categories: Optional[List[str]] = None,
    remove_source_dirs: bool = False,
) -> UnstructuredMergeReport:
    """便捷函数：合并非结构化处理结果"""
    merger = UnstructuredDataMerger(processed_dir=processed_dir)
    return merger.merge_all(
        categories=categories,
        remove_source_dirs=remove_source_dirs,
    )
