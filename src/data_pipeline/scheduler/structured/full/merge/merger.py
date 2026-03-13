"""
结构化原始数据合并器

扫描 data/raw/structured 下所有数据域，将按 ts_code 拆分存储为目录的数据
合并为单个 parquet 文件，合并后删除原目录。

合并逻辑：
  - 遍历每个数据域目录（如 market_data/、fundamental/ 等）
  - 对于其中的子目录（如 stock_daily/、dividend/ 等），读取目录内所有 .parquet 文件
  - 将所有 DataFrame 纵向拼接为一个完整的 DataFrame
  - 保存为 {子目录名}.parquet（snappy 压缩）
  - 删除原子目录

不处理的情况：
  - 已经是单个 .parquet 文件的数据（如 hsgt_flow.parquet）直接跳过
"""

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """单个目录的合并结果"""
    domain: str
    name: str
    source_dir: str
    output_file: str
    file_count: int = 0
    total_rows: int = 0
    success: bool = False
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class MergeReport:
    """合并任务总报告"""
    results: List[MergeResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    def summary(self) -> str:
        lines = [
            f"合并报告: 共 {self.total} 个目录, 成功 {self.success_count}, 失败 {self.failed_count}",
            "-" * 80,
        ]
        for r in self.results:
            status = "✓" if r.success else "✗"
            lines.append(
                f"  {status} {r.domain}/{r.name}: "
                f"{r.file_count} 文件 -> {r.total_rows} 行, "
                f"耗时 {r.duration_seconds:.1f}s"
                + (f" [错误: {r.error_message}]" if r.error_message else "")
            )
        return "\n".join(lines)


class RawDataMerger:
    """结构化原始数据合并器"""

    # 不参与合并的子目录名（这些数据保持按 ts_code 分文件存储）
    SKIP_NAMES = {"index_weight", "etf_daily", "index_daily"}

    def __init__(self, raw_dir: str = "data/raw/structured", dry_run: bool = False):
        """
        Args:
            raw_dir: 结构化原始数据根目录
            dry_run: 若为 True，仅扫描并报告待合并目录，不执行实际合并和删除
        """
        self.raw_dir = Path(raw_dir)
        self.dry_run = dry_run

        if not self.raw_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.raw_dir}")

    def scan(self) -> List[dict]:
        """扫描所有需要合并的目录

        Returns:
            列表，每个元素包含 domain、name、path、file_count
        """
        targets = []
        for domain_dir in sorted(self.raw_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain_name = domain_dir.name
            for item in sorted(domain_dir.iterdir()):
                if item.is_dir() and item.name not in self.SKIP_NAMES:
                    parquet_files = list(item.glob("*.parquet"))
                    targets.append({
                        "domain": domain_name,
                        "name": item.name,
                        "path": str(item),
                        "file_count": len(parquet_files),
                    })
        return targets

    def merge_all(self) -> MergeReport:
        """合并所有按 ts_code 拆分的目录

        Returns:
            MergeReport 包含每个目录的合并结果
        """
        targets = self.scan()
        report = MergeReport()

        logger.info(f"扫描完成，共发现 {len(targets)} 个待合并目录")
        for t in targets:
            logger.info(f"  {t['domain']}/{t['name']}: {t['file_count']} 个文件")

        if self.dry_run:
            logger.info("dry_run 模式，不执行实际合并")
            for t in targets:
                report.results.append(MergeResult(
                    domain=t["domain"],
                    name=t["name"],
                    source_dir=t["path"],
                    output_file=str(Path(t["path"]).parent / f"{t['name']}.parquet"),
                    file_count=t["file_count"],
                ))
            return report

        for t in targets:
            result = self._merge_directory(t["domain"], t["name"], Path(t["path"]))
            report.results.append(result)

        logger.info(report.summary())
        return report

    def merge_domain(self, domain: str) -> MergeReport:
        """合并指定数据域下的所有拆分目录"""
        domain_dir = self.raw_dir / domain
        if not domain_dir.exists():
            raise FileNotFoundError(f"数据域目录不存在: {domain_dir}")

        report = MergeReport()
        for item in sorted(domain_dir.iterdir()):
            if item.is_dir() and item.name not in self.SKIP_NAMES:
                result = self._merge_directory(domain, item.name, item)
                report.results.append(result)

        logger.info(report.summary())
        return report

    def _merge_directory(self, domain: str, name: str, dir_path: Path) -> MergeResult:
        """合并单个目录内的所有 parquet 文件

        Args:
            domain: 数据域名称
            name: 子目录名（如 stock_daily）
            dir_path: 子目录完整路径

        Returns:
            MergeResult 合并结果
        """
        output_file = dir_path.parent / f"{name}.parquet"
        result = MergeResult(
            domain=domain,
            name=name,
            source_dir=str(dir_path),
            output_file=str(output_file),
        )

        start_time = time.time()
        try:
            parquet_files = sorted(dir_path.glob("*.parquet"))
            result.file_count = len(parquet_files)

            if not parquet_files:
                logger.warning(f"  {domain}/{name}: 目录为空，跳过")
                result.success = True
                result.duration_seconds = time.time() - start_time
                return result

            logger.info(f"  合并 {domain}/{name}: {len(parquet_files)} 个文件 ...")

            # 分批读取并合并，避免内存溢出
            dfs = []
            for pf in parquet_files:
                try:
                    df = pd.read_parquet(pf)
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"    读取 {pf.name} 失败: {e}")

            if dfs:
                merged = pd.concat(dfs, ignore_index=True)
                result.total_rows = len(merged)

                # 统一列类型，避免跨文件类型不一致导致 parquet 写入失败
                merged = self._coerce_column_types(merged)

                # 保存合并后的 parquet
                merged.to_parquet(output_file, index=False, compression="snappy")
                logger.info(
                    f"  {domain}/{name}: 合并完成, "
                    f"{result.file_count} 文件 -> {result.total_rows} 行 -> {output_file.name}"
                )
            else:
                # 所有文件都为空，写一个空 parquet
                pd.DataFrame().to_parquet(output_file, index=False, compression="snappy")
                logger.warning(f"  {domain}/{name}: 所有文件均为空，已生成空 parquet")

            # 删除原目录
            shutil.rmtree(dir_path)
            logger.info(f"  {domain}/{name}: 已删除原目录 {dir_path}")

            result.success = True

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"  {domain}/{name}: 合并失败 - {e}")

        result.duration_seconds = time.time() - start_time
        return result

    @staticmethod
    def _coerce_column_types(df: pd.DataFrame) -> pd.DataFrame:
        """统一 DataFrame 列类型，处理跨文件合并后的类型不一致问题

        - object 列中混有 datetime.date / datetime.datetime -> 统一转为字符串
        - object 列中混有空字符串和数值 -> 将空字符串替换为 NaN 后转数值
        """
        import datetime as dt

        for col in df.columns:
            if df[col].dtype == object:
                non_null = df[col].dropna()
                if non_null.empty:
                    continue

                # 从首尾各取100行采样，确保覆盖不同来源文件的类型差异
                sample = pd.concat([non_null.head(100), non_null.tail(100)]).drop_duplicates()

                # 检测是否包含 datetime.date / datetime.datetime 对象
                has_date = sample.apply(lambda x: isinstance(x, (dt.date, dt.datetime))).any()
                if has_date:
                    df[col] = df[col].astype(str).replace("None", pd.NA).replace("NaT", pd.NA)
                    continue

                # 检测是否是混有空字符串的数值列
                df[col] = df[col].replace("", pd.NA)
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass  # 非数值列，保持原样

        return df
