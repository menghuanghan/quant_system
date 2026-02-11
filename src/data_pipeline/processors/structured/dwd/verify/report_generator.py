"""
DWD数据质量检查报告生成器

生成Markdown格式的详细检查报告
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .base import CheckSeverity, TableCheckReport
from .cross_table_checker import CrossTableCheckReport

logger = logging.getLogger(__name__)


class DWDReportGenerator:
    """DWD数据质量检查报告生成器"""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        report_name: str = "dwd_data_quality_report",
    ):
        self.output_dir = output_dir or Path("/home/menghuanghan/quant_system/reports")
        self.report_name = report_name
        self.reports: List[TableCheckReport] = []
        self.cross_table_report: Optional[CrossTableCheckReport] = None
    
    def add_report(self, report: TableCheckReport):
        """添加单表检查报告"""
        self.reports.append(report)
    
    def add_cross_table_report(self, report: CrossTableCheckReport):
        """添加跨表检查报告"""
        self.cross_table_report = report
    
    def generate_markdown(self) -> str:
        """生成Markdown格式报告"""
        
        lines = []
        
        # 标题
        lines.append("# DWD数据质量检查报告")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 总体摘要
        lines.extend(self._generate_summary())
        
        # 各表详细报告
        for report in self.reports:
            lines.extend(self._generate_table_report(report))
        
        # 跨表一致性检查报告
        if self.cross_table_report:
            lines.extend(self._generate_cross_table_report())
        
        return "\n".join(lines)
    
    def _generate_summary(self) -> List[str]:
        """生成总体摘要"""
        
        lines = []
        lines.append("## 总体摘要")
        lines.append("")
        
        # 统计汇总表
        lines.append("| 表名 | 行数 | 列数 | 通过 | 失败 | 严重 | 错误 | 警告 |")
        lines.append("|------|------|------|------|------|------|------|------|")
        
        total_passed = 0
        total_failed = 0
        total_critical = 0
        total_error = 0
        total_warning = 0
        
        for report in self.reports:
            total_passed += report.passed_count
            total_failed += report.failed_count
            total_critical += report.critical_count
            total_error += report.error_count
            total_warning += report.warning_count
            
            status = "✅" if report.critical_count == 0 and report.error_count == 0 else "❌"
            
            lines.append(
                f"| {report.table_name} | {report.total_rows:,} | {report.total_columns} | "
                f"{report.passed_count} | {report.failed_count} | "
                f"{report.critical_count} | {report.error_count} | {report.warning_count} |"
            )
        
        lines.append("")
        lines.append(f"**总计**: 通过 {total_passed}, 失败 {total_failed}, "
                    f"严重 {total_critical}, 错误 {total_error}, 警告 {total_warning}")
        lines.append("")
        
        # 关键问题列表
        critical_issues = []
        for report in self.reports:
            for result in report.check_results:
                if result.severity in [CheckSeverity.CRITICAL, CheckSeverity.ERROR]:
                    critical_issues.append((report.table_name, result))
        
        if critical_issues:
            lines.append("### 🚨 关键问题")
            lines.append("")
            for table_name, result in critical_issues:
                severity_icon = "🔴" if result.severity == CheckSeverity.CRITICAL else "🟠"
                lines.append(f"- {severity_icon} **{table_name}** - {result.check_name}: {result.message}")
            lines.append("")
        
        return lines
    
    def _generate_table_report(self, report: TableCheckReport) -> List[str]:
        """生成单表报告"""
        
        lines = []
        
        # 表标题
        status_icon = "✅" if report.critical_count == 0 and report.error_count == 0 else "❌"
        lines.append(f"## {status_icon} {report.table_name}")
        lines.append("")
        
        # 基本信息
        lines.append("### 基本信息")
        lines.append("")
        lines.append(f"- **总行数**: {report.total_rows:,}")
        lines.append(f"- **总列数**: {report.total_columns}")
        lines.append(f"- **日期范围**: {report.date_range[0]} ~ {report.date_range[1]}")
        lines.append(f"- **检查时间**: {report.check_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 检查结果
        lines.append("### 检查结果")
        lines.append("")
        
        # 按严重程度分组
        grouped = {
            CheckSeverity.CRITICAL: [],
            CheckSeverity.ERROR: [],
            CheckSeverity.WARNING: [],
            CheckSeverity.INFO: [],
            CheckSeverity.PASS: [],
        }
        
        for result in report.check_results:
            grouped[result.severity].append(result)
        
        # 输出严重问题和错误
        if grouped[CheckSeverity.CRITICAL] or grouped[CheckSeverity.ERROR]:
            lines.append("#### ⚠️ 问题项")
            lines.append("")
            
            for result in grouped[CheckSeverity.CRITICAL]:
                lines.append(f"- 🔴 **[CRITICAL]** {result.check_name}: {result.message}")
            
            for result in grouped[CheckSeverity.ERROR]:
                lines.append(f"- 🟠 **[ERROR]** {result.check_name}: {result.message}")
            
            lines.append("")
        
        # 输出警告
        if grouped[CheckSeverity.WARNING]:
            lines.append("#### ⚡ 警告项")
            lines.append("")
            for result in grouped[CheckSeverity.WARNING]:
                lines.append(f"- 🟡 {result.check_name}: {result.message}")
            lines.append("")
        
        # 输出通过项（折叠）
        if grouped[CheckSeverity.PASS]:
            lines.append("<details>")
            lines.append(f"<summary>✅ 通过项 ({len(grouped[CheckSeverity.PASS])}个)</summary>")
            lines.append("")
            for result in grouped[CheckSeverity.PASS]:
                lines.append(f"- ✅ {result.check_name}: {result.message}")
            lines.append("")
            lines.append("</details>")
            lines.append("")
        
        # 输出信息项（折叠）
        if grouped[CheckSeverity.INFO]:
            lines.append("<details>")
            lines.append(f"<summary>ℹ️ 信息项 ({len(grouped[CheckSeverity.INFO])}个)</summary>")
            lines.append("")
            for result in grouped[CheckSeverity.INFO]:
                lines.append(f"- ℹ️ {result.check_name}: {result.message}")
            lines.append("")
            lines.append("</details>")
            lines.append("")
        
        # 列统计表
        lines.extend(self._generate_column_stats_table(report))
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _generate_column_stats_table(self, report: TableCheckReport) -> List[str]:
        """生成列统计表"""
        
        if not report.column_stats:
            return []
        
        lines = []
        lines.append("### 列统计信息")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>点击展开列统计详情</summary>")
        lines.append("")
        
        # 数值列统计
        numeric_stats = [s for s in report.column_stats 
                        if s.mean_value is not None]
        
        if numeric_stats:
            lines.append("#### 数值列")
            lines.append("")
            lines.append("| 列名 | 类型 | 缺失率 | 最小值 | 最大值 | 均值 | 标准差 | 中位数 |")
            lines.append("|------|------|--------|--------|--------|------|--------|--------|")
            
            for s in numeric_stats:
                lines.append(
                    f"| {s.column_name} | {s.dtype} | {s.null_rate:.2%} | "
                    f"{s.min_value:.4g} | {s.max_value:.4g} | "
                    f"{s.mean_value:.4g} | {s.std_value:.4g} | {s.median_value:.4g} |"
                )
            lines.append("")
        
        # 非数值列统计
        non_numeric_stats = [s for s in report.column_stats 
                            if s.mean_value is None]
        
        if non_numeric_stats:
            lines.append("#### 非数值列")
            lines.append("")
            lines.append("| 列名 | 类型 | 缺失率 | 唯一值数 | 样本值 |")
            lines.append("|------|------|--------|----------|--------|")
            
            for s in non_numeric_stats:
                sample = str(s.sample_values[:3]) if s.sample_values else "-"
                lines.append(
                    f"| {s.column_name} | {s.dtype} | {s.null_rate:.2%} | "
                    f"{s.unique_count} | {sample[:50]} |"
                )
            lines.append("")
        
        lines.append("</details>")
        lines.append("")
        
        return lines
    
    def _generate_cross_table_report(self) -> List[str]:
        """生成跨表一致性检查报告"""
        
        if not self.cross_table_report:
            return []
        
        report = self.cross_table_report
        lines = []
        
        # 标题
        status_icon = "✅" if report.critical_count == 0 and report.error_count == 0 else "❌"
        lines.append(f"## {status_icon} 跨表一致性检查")
        lines.append("")
        
        # 基本信息
        lines.append("### 检查概况")
        lines.append("")
        lines.append(f"- **检查时间**: {report.check_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **检查项数**: {len(report.check_results)}")
        lines.append(f"- **通过**: {report.passed_count}, **失败**: {report.failed_count}")
        lines.append(f"- **严重**: {report.critical_count}, **错误**: {report.error_count}, **警告**: {report.warning_count}")
        lines.append("")
        
        # 按维度分组
        dimensions = {}
        for result in report.check_results:
            # 从check_name中提取维度标签
            if result.check_name.startswith("["):
                dim = result.check_name.split("]")[0][1:]
                check = result.check_name.split("]")[1].strip()
            else:
                dim = "其他"
                check = result.check_name
            
            if dim not in dimensions:
                dimensions[dim] = []
            dimensions[dim].append((check, result))
        
        # 输出各维度结果
        for dim, checks in dimensions.items():
            lines.append(f"### {dim}")
            lines.append("")
            
            for check_name, result in checks:
                if result.severity == CheckSeverity.CRITICAL:
                    icon = "🔴"
                elif result.severity == CheckSeverity.ERROR:
                    icon = "🟠"
                elif result.severity == CheckSeverity.WARNING:
                    icon = "🟡"
                elif result.severity == CheckSeverity.PASS:
                    icon = "✅"
                else:
                    icon = "ℹ️"
                
                lines.append(f"- {icon} **{check_name}**: {result.message}")
                
                # 如果有详细信息且不是通过，展示关键细节
                if result.details and result.severity != CheckSeverity.PASS:
                    for key, value in list(result.details.items())[:5]:
                        if not isinstance(value, (list, dict)) or (isinstance(value, list) and len(value) <= 5):
                            lines.append(f"  - {key}: {value}")
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def save(self) -> Path:
        """保存报告"""
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存Markdown报告
        md_content = self.generate_markdown()
        md_path = self.output_dir / f"{self.report_name}.md"
        md_path.write_text(md_content, encoding="utf-8")
        logger.info(f"Markdown报告已保存: {md_path}")
        
        # 保存JSON报告
        json_data = {
            "generated_at": datetime.now().isoformat(),
            "tables": [r.to_dict() for r in self.reports],
            "cross_table": self.cross_table_report.to_dict() if self.cross_table_report else None,
        }
        json_path = self.output_dir / f"{self.report_name}.json"
        json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"JSON报告已保存: {json_path}")
        
        return md_path
    
    def print_summary(self):
        """打印摘要到控制台"""
        
        print("\n" + "=" * 60)
        print("DWD 数据质量检查结果摘要")
        print("=" * 60)
        
        for report in self.reports:
            status = "✅ PASS" if (report.critical_count == 0 and report.error_count == 0) else "❌ FAIL"
            print(f"\n{report.table_name}: {status}")
            print(f"  行数: {report.total_rows:,}, 列数: {report.total_columns}")
            print(f"  通过: {report.passed_count}, 失败: {report.failed_count}, "
                  f"严重: {report.critical_count}, 错误: {report.error_count}, 警告: {report.warning_count}")
            
            # 打印关键问题
            for result in report.check_results:
                if result.severity in [CheckSeverity.CRITICAL, CheckSeverity.ERROR]:
                    print(f"  ❌ [{result.severity.value}] {result.check_name}: {result.message}")
        
        # 跨表检查结果
        if self.cross_table_report:
            report = self.cross_table_report
            status = "✅ PASS" if (report.critical_count == 0 and report.error_count == 0) else "❌ FAIL"
            print(f"\n跨表一致性检查: {status}")
            print(f"  通过: {report.passed_count}, 失败: {report.failed_count}, "
                  f"严重: {report.critical_count}, 错误: {report.error_count}, 警告: {report.warning_count}")
            
            for result in report.check_results:
                if result.severity in [CheckSeverity.CRITICAL, CheckSeverity.ERROR]:
                    print(f"  ❌ [{result.severity.value}] {result.check_name}: {result.message}")
        
        print("\n" + "=" * 60)
