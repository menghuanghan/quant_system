#!/usr/bin/env python
"""
DWD数据质量检查主脚本

运行所有数据质量检查并生成Markdown格式的报告

用法：
    python scripts/verify/run_dq_check.py
    python scripts/verify/run_dq_check.py --verbose
    python scripts/verify/run_dq_check.py --output reports/custom_dq_report.md
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
    DWD_PATHS, THRESHOLDS, REPORTS_DIR,
    CheckResult, TableSummary, QualityReport,
    setup_logging, format_number, format_percentage, get_file_size_mb,
)
from .check_price_table import PriceTableChecker
from .check_fundamental_table import FundamentalTableChecker
from .check_status_table import StatusTableChecker
from .check_cross_table import CrossTableChecker


logger = setup_logging(__name__)


class DQReportGenerator:
    """数据质量报告生成器"""
    
    def __init__(self, output_path: Optional[Path] = None):
        """
        Args:
            output_path: 报告输出路径，默认为 reports/dwd_dq_report.md
        """
        self.output_path = output_path or (REPORTS_DIR / "dwd_dq_report.md")
        self.generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.summaries: Dict[str, TableSummary] = {}
        self.all_results: Dict[str, List[CheckResult]] = {}
        self.critical_issues: List[str] = []
        self.warnings: List[str] = []
    
    def run_all_checks(self) -> bool:
        """运行所有检查"""
        logger.info("=" * 70)
        logger.info("DWD 数据质量检查开始")
        logger.info(f"时间: {self.generated_at}")
        logger.info("=" * 70)
        
        # 1. 量价表检查
        logger.info("\n[1/4] 检查量价宽表...")
        try:
            price_checker = PriceTableChecker()
            self.summaries['price'] = price_checker.get_summary()
            self.all_results['price'] = price_checker.run_all_checks()
        except Exception as e:
            logger.error(f"量价表检查失败: {e}")
            self.critical_issues.append(f"量价表检查失败: {str(e)}")
        
        # 2. 基本面表检查
        logger.info("\n[2/4] 检查基本面宽表...")
        try:
            fundamental_checker = FundamentalTableChecker()
            self.summaries['fundamental'] = fundamental_checker.get_summary()
            self.all_results['fundamental'] = fundamental_checker.run_all_checks()
        except Exception as e:
            logger.error(f"基本面表检查失败: {e}")
            self.critical_issues.append(f"基本面表检查失败: {str(e)}")
        
        # 3. 状态表检查
        logger.info("\n[3/4] 检查状态表...")
        try:
            status_checker = StatusTableChecker()
            self.summaries['status'] = status_checker.get_summary()
            self.all_results['status'] = status_checker.run_all_checks()
        except Exception as e:
            logger.error(f"状态表检查失败: {e}")
            self.critical_issues.append(f"状态表检查失败: {str(e)}")
        
        # 4. 跨表一致性检查
        logger.info("\n[4/4] 执行跨表一致性检查...")
        try:
            cross_checker = CrossTableChecker()
            self.all_results['cross_table'] = cross_checker.run_all_checks()
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
                if result.severity == 'CRITICAL':
                    for issue in result.issues:
                        self.critical_issues.append(f"[{table_name}] {result.name}: {issue}")
                elif result.severity in ['ERROR', 'WARNING']:
                    for issue in result.issues:
                        self.warnings.append(f"[{table_name}] {result.name}: {issue}")
    
    def is_overall_passed(self) -> bool:
        """判断整体是否通过"""
        # 只要没有CRITICAL问题就算通过
        if self.critical_issues:
            return False
        
        # 检查各表的关键检查项
        for results in self.all_results.values():
            for result in results:
                if result.severity == 'CRITICAL' and not result.passed:
                    return False
        
        return True
    
    def generate_report(self) -> str:
        """生成Markdown格式的数据质量报告"""
        lines = []
        
        # 标题
        lines.append("# DWD 数据质量检查报告")
        lines.append("")
        lines.append(f"**生成时间**: {self.generated_at}")
        lines.append("")
        
        # 总体结论
        overall_passed = self.is_overall_passed()
        if overall_passed:
            lines.append("## ✅ 总体结论：通过")
            lines.append("")
            lines.append("所有关键检查项均已通过，数据质量符合预期。")
        else:
            lines.append("## ❌ 总体结论：失败")
            lines.append("")
            lines.append("存在关键问题需要处理。")
        
        lines.append("")
        
        # 数据概览
        lines.append("## 📊 数据概览")
        lines.append("")
        lines.append("| 数据表 | 行数 | 股票数 | 日期范围 | 文件大小 |")
        lines.append("|--------|------|--------|----------|----------|")
        
        for name, summary in self.summaries.items():
            lines.append(f"| {summary.name} | {summary.rows:,} | {summary.stock_count:,} | "
                        f"{summary.date_range[0]} ~ {summary.date_range[1]} | "
                        f"{summary.file_size_mb:.1f} MB |")
        
        lines.append("")
        
        # 关键字段缺失率热力图
        lines.append("### 关键字段缺失率")
        lines.append("")
        
        for name, summary in self.summaries.items():
            lines.append(f"**{summary.name}**:")
            lines.append("")
            
            # 筛选缺失率 > 0 的字段
            missing_fields = {k: v for k, v in summary.null_summary.items() if v > 0}
            
            if missing_fields:
                lines.append("| 字段 | 缺失率 | 状态 |")
                lines.append("|------|--------|------|")
                
                for field, rate in sorted(missing_fields.items(), key=lambda x: -x[1])[:15]:
                    status = "⚠️" if rate > 0.05 else "✓"
                    lines.append(f"| {field} | {rate*100:.2f}% | {status} |")
            else:
                lines.append("所有字段完整，无缺失值。")
            
            lines.append("")
        
        # 检查结果明细
        lines.append("## 📋 检查结果明细")
        lines.append("")
        
        table_names = {
            'price': '量价宽表 (dwd_stock_price)',
            'fundamental': 'PIT基本面宽表 (dwd_stock_fundamental)',
            'status': '状态与风险掩码表 (dwd_stock_status)',
            'cross_table': '跨表一致性检查 (Golden Check)',
        }
        
        for table_key, results in self.all_results.items():
            table_name = table_names.get(table_key, table_key)
            lines.append(f"### {table_name}")
            lines.append("")
            
            passed_count = sum(1 for r in results if r.passed)
            lines.append(f"检查结果: **{passed_count}/{len(results)}** 项通过")
            lines.append("")
            
            lines.append("| 检查项 | 状态 | 严重程度 | 说明 |")
            lines.append("|--------|------|----------|------|")
            
            for result in results:
                status = "✅" if result.passed else "❌"
                severity_icon = {
                    'INFO': '🟢',
                    'WARNING': '🟡',
                    'ERROR': '🟠',
                    'CRITICAL': '🔴',
                }.get(result.severity, '⚪')
                
                lines.append(f"| {result.name} | {status} | {severity_icon} {result.severity} | "
                            f"{result.description} |")
            
            lines.append("")
            
            # 问题详情
            has_issues = any(r.issues for r in results)
            if has_issues:
                lines.append("**问题详情:**")
                lines.append("")
                
                for result in results:
                    if result.issues:
                        lines.append(f"- **{result.name}**")
                        for issue in result.issues:
                            lines.append(f"  - {issue}")
                
                lines.append("")
        
        # 异常清单
        lines.append("## 🚨 异常清单")
        lines.append("")
        
        # Top 10 收益率异常
        if 'price' in self.all_results:
            for result in self.all_results['price']:
                if result.name == "极值与异常检测" and 'extreme_samples' in result.metrics:
                    lines.append("### Top 10 收益率异常")
                    lines.append("")
                    lines.append("| 日期 | 股票代码 | 收益率 | 收盘价 | 前收盘 |")
                    lines.append("|------|----------|--------|--------|--------|")
                    
                    for sample in result.metrics['extreme_samples'][:10]:
                        trade_date = sample.get('trade_date', '')
                        if isinstance(trade_date, pd.Timestamp):
                            trade_date = trade_date.strftime('%Y-%m-%d')
                        lines.append(f"| {trade_date} | {sample.get('ts_code', '')} | "
                                    f"{sample.get('return_1d', 0)*100:.2f}% | "
                                    f"{sample.get('close', 0):.2f} | "
                                    f"{sample.get('pre_close', 0):.2f} |")
                    
                    lines.append("")
        
        # TTM突变清单
        if 'fundamental' in self.all_results:
            for result in self.all_results['fundamental']:
                if result.name == "TTM数值逻辑" and 'extreme_changes_detail' in result.metrics:
                    lines.append("### TTM指标突变清单")
                    lines.append("")
                    
                    for change_info in result.metrics['extreme_changes_detail'][:5]:
                        lines.append(f"**{change_info['field']}** (共 {change_info['count']} 处突变)")
                        lines.append("")
                        
                        if 'sample' in change_info:
                            lines.append("| 日期 | 股票代码 | 值 | 变化率 |")
                            lines.append("|------|----------|-------|--------|")
                            
                            for s in change_info['sample'][:5]:
                                trade_date = s.get('trade_date', '')
                                if isinstance(trade_date, pd.Timestamp):
                                    trade_date = trade_date.strftime('%Y-%m-%d')
                                lines.append(f"| {trade_date} | {s.get('ts_code', '')} | "
                                            f"{format_number(s.get(change_info['field'], 0))} | "
                                            f"{s.get('pct_change', 0)*100:.1f}% |")
                            
                            lines.append("")
        
        # 总结
        lines.append("## 📝 总结")
        lines.append("")
        
        if self.critical_issues:
            lines.append("### 关键问题（需立即处理）")
            lines.append("")
            for issue in self.critical_issues:
                lines.append(f"- ❌ {issue}")
            lines.append("")
        
        if self.warnings:
            lines.append("### 警告（建议检查）")
            lines.append("")
            for warning in self.warnings[:20]:  # 限制显示数量
                lines.append(f"- ⚠️ {warning}")
            if len(self.warnings) > 20:
                lines.append(f"- ... 还有 {len(self.warnings) - 20} 条警告")
            lines.append("")
        
        if not self.critical_issues and not self.warnings:
            lines.append("所有检查项均通过，数据质量良好。")
            lines.append("")
        
        # 建议
        lines.append("## 💡 建议")
        lines.append("")
        
        if overall_passed:
            lines.append("1. 数据质量检查通过，可以继续进行特征工程。")
            lines.append("2. 建议定期运行此检查脚本，监控数据质量变化。")
            lines.append("3. 对于冷启动期的数据缺失，是正常现象，特征工程时需注意处理。")
        else:
            lines.append("1. 请优先处理关键问题，这些问题可能导致模型训练出错。")
            lines.append("2. 检查数据处理流程，确保DWD层处理器逻辑正确。")
            lines.append("3. 问题修复后，重新运行此检查脚本验证。")
        
        lines.append("")
        lines.append("---")
        lines.append(f"*报告由 DWD 数据质量检查工具自动生成 | {self.generated_at}*")
        
        return '\n'.join(lines)
    
    def save_report(self) -> Path:
        """保存报告到文件"""
        report_content = self.generate_report()
        
        # 确保目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"报告已保存到: {self.output_path}")
        return self.output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DWD数据质量检查')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--output', '-o', type=str, help='报告输出路径')
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # 确定输出路径
    output_path = Path(args.output) if args.output else None
    
    # 运行检查
    generator = DQReportGenerator(output_path=output_path)
    passed = generator.run_all_checks()
    
    # 生成报告
    report_path = generator.save_report()
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("DWD 数据质量检查完成")
    print("=" * 70)
    
    if passed:
        print("✅ 总体结论: 通过")
    else:
        print("❌ 总体结论: 失败")
        print(f"\n关键问题数: {len(generator.critical_issues)}")
        for issue in generator.critical_issues[:5]:
            print(f"  - {issue}")
    
    print(f"\n警告数: {len(generator.warnings)}")
    print(f"\n报告路径: {report_path}")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
