#!/usr/bin/env python
"""
DWD 扩展表数据质量检查主脚本

运行所有扩展表数据质量检查并生成 Markdown 格式的报告

用法：
    python -m src.data_pipeline.processors.structured.dwd.verify_ext.run_dq_check
    python -m src.data_pipeline.processors.structured.dwd.verify_ext.run_dq_check --verbose
    python -m src.data_pipeline.processors.structured.dwd.verify_ext.run_dq_check --output reports/custom_report.md
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from .dq_config import (
    DWD_EXT_PATHS, EXT_THRESHOLDS, REPORTS_DIR, TRAINING_START_DATE,
    CheckResult, TableSummary, QualityReport,
    setup_logging, format_number, format_percentage, get_file_size_mb,
)
from .check_money_flow import MoneyFlowChecker
from .check_chip_structure import ChipStructureChecker
from .check_industry import IndustryChecker
from .check_event_signal import EventSignalChecker
from .check_macro_env import MacroEnvChecker
from .check_cross_table import CrossTableChecker


logger = setup_logging(__name__)


class ExtDQReportGenerator:
    """扩展表数据质量报告生成器"""
    
    def __init__(self, output_path: Optional[Path] = None):
        """
        Args:
            output_path: 报告输出路径，默认为 reports/dwd_extended_dq_report.md
        """
        self.output_path = output_path or (REPORTS_DIR / "dwd_extended_dq_report.md")
        self.generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.summaries: Dict[str, TableSummary] = {}
        self.all_results: Dict[str, List[CheckResult]] = {}
        self.critical_issues: List[str] = []
        self.warnings: List[str] = []
    
    def run_all_checks(self) -> bool:
        """运行所有检查"""
        logger.info("=" * 70)
        logger.info("DWD 扩展表数据质量检查开始")
        logger.info(f"时间: {self.generated_at}")
        logger.info(f"训练期起始: {TRAINING_START_DATE}")
        logger.info("=" * 70)
        
        # 1. 资金博弈表检查
        logger.info("\n[1/6] 检查资金博弈宽表 (dwd_money_flow)...")
        try:
            checker = MoneyFlowChecker()
            self.summaries['money_flow'] = checker.get_summary()
            self.all_results['money_flow'] = checker.run_all_checks()
        except Exception as e:
            logger.error(f"资金博弈表检查失败: {e}")
            self.critical_issues.append(f"资金博弈表检查失败: {str(e)}")
        
        # 2. 筹码结构表检查
        logger.info("\n[2/6] 检查筹码结构宽表 (dwd_chip_structure)...")
        try:
            checker = ChipStructureChecker()
            self.summaries['chip_structure'] = checker.get_summary()
            self.all_results['chip_structure'] = checker.run_all_checks()
        except Exception as e:
            logger.error(f"筹码结构表检查失败: {e}")
            self.critical_issues.append(f"筹码结构表检查失败: {str(e)}")
        
        # 3. 行业分类表检查
        logger.info("\n[3/6] 检查行业分类宽表 (dwd_stock_industry)...")
        try:
            checker = IndustryChecker()
            self.summaries['industry'] = checker.get_summary()
            self.all_results['industry'] = checker.run_all_checks()
        except Exception as e:
            logger.error(f"行业分类表检查失败: {e}")
            self.critical_issues.append(f"行业分类表检查失败: {str(e)}")
        
        # 4. 事件信号表检查
        logger.info("\n[4/6] 检查事件信号宽表 (dwd_event_signal)...")
        try:
            checker = EventSignalChecker()
            self.summaries['event_signal'] = checker.get_summary()
            self.all_results['event_signal'] = checker.run_all_checks()
        except Exception as e:
            logger.error(f"事件信号表检查失败: {e}")
            self.critical_issues.append(f"事件信号表检查失败: {str(e)}")
        
        # 5. 宏观环境表检查
        logger.info("\n[5/6] 检查宏观环境宽表 (dwd_macro_env)...")
        try:
            checker = MacroEnvChecker()
            self.summaries['macro_env'] = checker.get_summary()
            self.all_results['macro_env'] = checker.run_all_checks()
        except Exception as e:
            logger.error(f"宏观环境表检查失败: {e}")
            self.critical_issues.append(f"宏观环境表检查失败: {str(e)}")
        
        # 6. 跨表一致性检查
        logger.info("\n[6/6] 执行跨表一致性检查...")
        try:
            checker = CrossTableChecker()
            self.all_results['cross_table'] = checker.run_all_checks()
        except Exception as e:
            logger.error(f"跨表检查失败: {e}")
            self.critical_issues.append(f"跨表检查失败: {str(e)}")
        
        # 汇总结果
        self._collect_issues()
        
        return self.is_overall_passed()
    
    def _collect_issues(self):
        """收集所有问题"""
        for table_name, results in self.all_results.items():
            for result in results:
                if result.severity == 'CRITICAL' and not result.passed:
                    for issue in result.issues:
                        self.critical_issues.append(f"[{table_name}] {result.name}: {issue}")
                elif result.severity in ['ERROR', 'WARNING'] and not result.passed:
                    for issue in result.issues:
                        self.warnings.append(f"[{table_name}] {result.name}: {issue}")
    
    def is_overall_passed(self) -> bool:
        """判断整体是否通过"""
        # 只要没有 CRITICAL 问题就算通过
        if self.critical_issues:
            return False
        
        # 检查各表的关键检查项
        for results in self.all_results.values():
            for result in results:
                if result.severity == 'CRITICAL' and not result.passed:
                    return False
        
        return True
    
    def generate_report(self) -> str:
        """生成 Markdown 格式的数据质量报告"""
        lines = []
        
        # 标题
        lines.append("# DWD 扩展表数据质量检查报告")
        lines.append("")
        lines.append(f"**生成时间**: {self.generated_at}")
        lines.append(f"**训练期起始**: {TRAINING_START_DATE}")
        lines.append("")
        
        # 总体结论
        overall_passed = self.is_overall_passed()
        if overall_passed:
            lines.append("## ✅ 总体结论：通过")
            lines.append("")
            lines.append("所有关键检查项均已通过，扩展表数据质量符合预期。")
        else:
            lines.append("## ❌ 总体结论：失败")
            lines.append("")
            lines.append("存在以下关键问题需要处理：")
            lines.append("")
            for issue in self.critical_issues:
                lines.append(f"- 🚨 {issue}")
        
        lines.append("")
        
        # 警告汇总
        if self.warnings:
            lines.append("### ⚠️ 警告")
            lines.append("")
            for warning in self.warnings[:20]:  # 最多显示 20 条
                lines.append(f"- {warning}")
            if len(self.warnings) > 20:
                lines.append(f"- ... 还有 {len(self.warnings) - 20} 条警告")
            lines.append("")
        
        # 数据概览
        lines.append("## 📊 数据概览")
        lines.append("")
        lines.append("| 数据表 | 行数 | 列数 | 日期范围 | 股票数 | 文件大小 |")
        lines.append("|--------|------|------|----------|--------|----------|")
        
        for name, summary in self.summaries.items():
            lines.append(
                f"| {summary.name} | {summary.rows:,} | {summary.columns} | "
                f"{summary.date_range[0]} ~ {summary.date_range[1]} | "
                f"{summary.stock_count:,} | {summary.file_size_mb:.2f} MB |"
            )
        
        lines.append("")
        
        # 详细检查结果
        lines.append("## 📋 详细检查结果")
        lines.append("")
        
        for table_name, results in self.all_results.items():
            lines.append(f"### {table_name}")
            lines.append("")
            
            # 统计通过/失败
            passed_count = sum(1 for r in results if r.passed)
            total_count = len(results)
            
            lines.append(f"**检查通过率**: {passed_count}/{total_count}")
            lines.append("")
            
            lines.append("| 检查项 | 结果 | 说明 |")
            lines.append("|--------|------|------|")
            
            for result in results:
                status = "✅" if result.passed else ("❌" if result.severity in ['CRITICAL', 'ERROR'] else "⚠️")
                desc = result.description
                if result.issues:
                    desc += f" ({result.issues[0][:50]}...)" if len(result.issues[0]) > 50 else f" ({result.issues[0]})"
                lines.append(f"| {result.name} | {status} | {desc} |")
            
            lines.append("")
            
            # 详细指标
            if any(r.metrics for r in results):
                lines.append("<details>")
                lines.append(f"<summary>展开查看 {table_name} 详细指标</summary>")
                lines.append("")
                
                for result in results:
                    if result.metrics:
                        lines.append(f"#### {result.name}")
                        lines.append("")
                        lines.append("```json")
                        # 格式化 metrics
                        import json
                        metrics_str = json.dumps(result.metrics, indent=2, default=str, ensure_ascii=False)
                        lines.append(metrics_str)
                        lines.append("```")
                        lines.append("")
                
                lines.append("</details>")
                lines.append("")
        
        # 结论
        lines.append("## 📝 结论与建议")
        lines.append("")
        
        if overall_passed:
            lines.append("数据质量检查全部通过，可以进行后续的特征工程和模型训练。")
        else:
            lines.append("以下问题需要优先处理：")
            lines.append("")
            for i, issue in enumerate(self.critical_issues, 1):
                lines.append(f"{i}. {issue}")
        
        lines.append("")
        lines.append("---")
        lines.append(f"*报告生成于 {self.generated_at}*")
        
        return "\n".join(lines)
    
    def save_report(self):
        """保存报告到文件"""
        report = self.generate_report()
        
        # 确保目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已保存到: {self.output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='DWD 扩展表数据质量检查',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='报告输出路径 (默认: reports/dwd_extended_dq_report.md)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细日志'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建报告生成器
    output_path = Path(args.output) if args.output else None
    generator = ExtDQReportGenerator(output_path=output_path)
    
    # 运行检查
    passed = generator.run_all_checks()
    
    # 生成报告
    generator.save_report()
    
    # 输出结论
    print()
    print("=" * 70)
    if passed:
        print("✅ 数据质量检查通过！")
    else:
        print("❌ 数据质量检查失败！")
        print("关键问题：")
        for issue in generator.critical_issues:
            print(f"  - {issue}")
    print("=" * 70)
    print(f"详细报告: {generator.output_path}")
    
    # 返回退出码
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
