"""
数据质量报告生成器

生成 Markdown 和 JSON 格式的数据质量报告
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .checker import CheckLevel, CheckResult, ColumnStats, MergerPreprocessChecker

logger = logging.getLogger(__name__)


class QualityReportGenerator:
    """数据质量报告生成器"""
    
    def __init__(self, checker: MergerPreprocessChecker):
        """
        初始化报告生成器
        
        Args:
            checker: 已执行检查的 MergerPreprocessChecker 实例
        """
        self.checker = checker
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def generate_markdown(self, output_path: Path = None) -> str:
        """
        生成 Markdown 格式报告
        
        Args:
            output_path: 输出路径，默认为 reports/merger_preprocess_dq_report_{timestamp}.md
            
        Returns:
            Markdown 内容
        """
        if output_path is None:
            output_path = self.checker.output_dir / f'merger_preprocess_dq_report_{self.timestamp}.md'
        
        lines = []
        
        # 标题
        lines.append('# Merger 预处理数据质量报告')
        lines.append('')
        lines.append(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        lines.append(f'**数据文件**: `{self.checker.merger_path}`')
        lines.append('')
        
        # 汇总
        lines.append('## 📊 检查结果汇总')
        lines.append('')
        
        summary = self.checker.get_summary()
        overall = '✅ PASS' if summary['critical'] == 0 and summary['errors'] == 0 else '❌ FAIL'
        
        lines.append(f'| 指标 | 值 |')
        lines.append(f'|------|-----|')
        lines.append(f'| 总体结果 | {overall} |')
        lines.append(f'| 总检查项 | {summary["total_checks"]} |')
        lines.append(f'| 通过 | {summary["passed"]} |')
        lines.append(f'| CRITICAL | {summary["critical"]} |')
        lines.append(f'| ERROR | {summary["errors"]} |')
        lines.append(f'| WARNING | {summary["warnings"]} |')
        lines.append('')
        
        # 按维度分组结果
        dimensions = {}
        for result in self.checker.results:
            dim = result.dimension
            if dim not in dimensions:
                dimensions[dim] = []
            dimensions[dim].append(result)
        
        # 各维度详情
        for dim, results in dimensions.items():
            lines.append(f'## 📋 {dim}')
            lines.append('')
            
            lines.append('| 检查项 | 级别 | 状态 | 说明 |')
            lines.append('|--------|------|------|------|')
            
            for r in results:
                icon = '✅' if r.passed else '❌'
                lines.append(f'| {r.name} | {r.level.value} | {icon} | {r.message} |')
            
            lines.append('')
            
            # 添加详情
            for r in results:
                if r.details and not r.passed:
                    lines.append(f'### {r.name} 详情')
                    lines.append('')
                    lines.append('```json')
                    lines.append(json.dumps(r.details, indent=2, ensure_ascii=False, default=str))
                    lines.append('```')
                    lines.append('')
        
        # 列统计信息
        lines.append('## 📈 列统计信息')
        lines.append('')
        
        # 数值列统计
        lines.append('### 数值列统计')
        lines.append('')
        lines.append('| 列名 | 类型 | 非空 | 缺失% | 均值 | 标准差 | 最小 | 最大 |')
        lines.append('|------|------|------|-------|------|--------|------|------|')
        
        for name, stats in self.checker.column_stats.items():
            if stats.mean is not None:
                lines.append(
                    f'| {name} | {stats.dtype} | {stats.count - stats.null_count:,} | '
                    f'{stats.null_pct:.2f}% | {stats.mean:.2e} | {stats.std:.2e} | '
                    f'{stats.min_val:.2e} | {stats.max_val:.2e} |'
                )
        
        lines.append('')
        
        # 分类列统计
        lines.append('### 分类列统计')
        lines.append('')
        lines.append('| 列名 | 类型 | 唯一值 | 缺失% | Top Values |')
        lines.append('|------|------|--------|-------|------------|')
        
        for name, stats in self.checker.column_stats.items():
            if stats.top_values is not None:
                top_str = ', '.join([f'{k}({v})' for k, v in list(stats.top_values.items())[:3]])
                lines.append(
                    f'| {name} | {stats.dtype} | {stats.unique_count} | '
                    f'{stats.null_pct:.2f}% | {top_str} |'
                )
        
        lines.append('')
        
        # 高缺失率字段
        high_null_fields = [
            (name, stats) for name, stats in self.checker.column_stats.items()
            if stats.null_pct > 20
        ]
        
        if high_null_fields:
            lines.append('### ⚠️ 高缺失率字段 (>20%)')
            lines.append('')
            lines.append('| 列名 | 缺失行数 | 缺失率 |')
            lines.append('|------|----------|--------|')
            
            for name, stats in sorted(high_null_fields, key=lambda x: -x[1].null_pct):
                lines.append(f'| {name} | {stats.null_count:,} | {stats.null_pct:.2f}% |')
            
            lines.append('')
        
        # 写入文件
        content = '\n'.join(lines)
        output_path.write_text(content, encoding='utf-8')
        logger.info(f"📝 Markdown 报告已保存: {output_path}")
        
        return content
    
    def generate_json(self, output_path: Path = None) -> Dict[str, Any]:
        """
        生成 JSON 格式报告
        
        Args:
            output_path: 输出路径
            
        Returns:
            JSON 字典
        """
        if output_path is None:
            output_path = self.checker.output_dir / f'merger_preprocess_dq_report_{self.timestamp}.json'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_file': str(self.checker.merger_path),
            'summary': self.checker.get_summary(),
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📝 JSON 报告已保存: {output_path}")
        
        return report
    
    def generate_all(self) -> tuple:
        """生成所有格式的报告"""
        md_content = self.generate_markdown()
        json_report = self.generate_json()
        return md_content, json_report
