"""
Train.parquet 数据质量报告生成器

生成详细的 Markdown 和 JSON 格式报告
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .train_data_checker import (
    CheckLevel, CheckResult, ColumnProfile, TrainDataChecker, _convert_to_serializable
)

logger = logging.getLogger(__name__)


class TrainDataReportGenerator:
    """Train.parquet 数据质量报告生成器"""
    
    def __init__(self, checker: TrainDataChecker):
        """
        初始化报告生成器
        
        Args:
            checker: 已执行检查的 TrainDataChecker 实例
        """
        self.checker = checker
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def generate_markdown(self, output_path: Path = None) -> str:
        """生成 Markdown 格式报告"""
        if output_path is None:
            output_path = self.checker.output_dir / f'train_dq_report_{self.timestamp}.md'
        
        lines = []
        
        # 标题
        lines.append('# Train.parquet 数据质量检查报告')
        lines.append('')
        lines.append(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        lines.append(f'**数据文件**: `{self.checker.train_path}`')
        lines.append('')
        
        # 数据概览
        summary = self.checker.get_summary()
        lines.append('## 📊 数据概览')
        lines.append('')
        lines.append(f'| 指标 | 值 |')
        lines.append(f'|------|------|')
        lines.append(f'| 行数 | {summary["shape"]["rows"]:,} |')
        lines.append(f'| 列数 | {summary["shape"]["cols"]} |')
        lines.append(f'| 检查项 | {summary["check_results"]["total"]} |')
        lines.append(f'| 通过 | {summary["check_results"]["passed"]} |')
        lines.append(f'| 失败 | {summary["check_results"]["failed"]} |')
        lines.append('')
        
        # 检查结果汇总
        lines.append('## 🔍 检查结果汇总')
        lines.append('')
        
        results = self.checker.get_results()
        
        # 按维度分组
        for dimension in ['基础完整性', '特征数值质量', '标签逻辑', '时序稳定性', '业务逻辑']:
            dim_results = [r for r in results if r.dimension == dimension]
            if not dim_results:
                continue
            
            lines.append(f'### {dimension}')
            lines.append('')
            lines.append('| 检查项 | 状态 | 说明 |')
            lines.append('|--------|------|------|')
            
            for r in dim_results:
                status = '✅' if r.passed else ('⚠️' if r.level == CheckLevel.WARNING else '❌')
                lines.append(f'| {r.name} | {status} {r.level.value} | {r.message} |')
            
            lines.append('')
        
        # 详细问题列表
        lines.append('## ⚠️ 问题详情')
        lines.append('')
        
        issues = [r for r in results if not r.passed or r.level in [CheckLevel.WARNING, CheckLevel.ERROR, CheckLevel.CRITICAL]]
        
        if not issues:
            lines.append('无问题发现。')
        else:
            for r in issues:
                lines.append(f'### {r.name} ({r.level.value})')
                lines.append('')
                lines.append(f'**维度**: {r.dimension}')
                lines.append(f'**状态**: {"通过" if r.passed else "未通过"}')
                lines.append(f'**说明**: {r.message}')
                lines.append('')
                
                if r.details:
                    lines.append('**详情**:')
                    lines.append('```json')
                    lines.append(json.dumps(r.details, indent=2, ensure_ascii=False, default=str)[:2000])
                    lines.append('```')
                    lines.append('')
        
        # 列档案摘要
        lines.append('## 📋 列档案摘要')
        lines.append('')
        
        profiles = self.checker.get_column_profiles()
        
        # 按类别分组
        for category, cat_name in [('meta', 'Meta 列'), ('label', 'Label 列'), ('feature', 'Feature 列')]:
            cat_profiles = [p for p in profiles.values() if p.category == category]
            if not cat_profiles:
                continue
            
            lines.append(f'### {cat_name} ({len(cat_profiles)})')
            lines.append('')
            
            # Feature 列太多，只显示摘要
            if category == 'feature':
                lines.append('| 列名 | 类型 | NaN% | Zero% | Inf | Min | Max | Mean | Std |')
                lines.append('|------|------|------|-------|-----|-----|-----|------|-----|')
                
                for p in sorted(cat_profiles, key=lambda x: x.null_pct, reverse=True)[:50]:
                    min_val = f'{p.min_val:.4g}' if p.min_val is not None else '-'
                    max_val = f'{p.max_val:.4g}' if p.max_val is not None else '-'
                    mean_val = f'{p.mean:.4g}' if p.mean is not None else '-'
                    std_val = f'{p.std:.4g}' if p.std is not None else '-'
                    inf_total = p.inf_count + p.neg_inf_count
                    
                    lines.append(f'| {p.name} | {p.dtype} | {p.null_pct:.1f}% | {p.zero_pct:.1f}% | {inf_total} | {min_val} | {max_val} | {mean_val} | {std_val} |')
                
                if len(cat_profiles) > 50:
                    lines.append(f'| ... | ... | ... | ... | ... | ... | ... | ... | ... |')
                    lines.append(f'| (共 {len(cat_profiles)} 列) | | | | | | | | |')
            else:
                lines.append('| 列名 | 类型 | NaN% | Zero% | Unique |')
                lines.append('|------|------|------|-------|--------|')
                
                for p in cat_profiles:
                    lines.append(f'| {p.name} | {p.dtype} | {p.null_pct:.1f}% | {p.zero_pct:.1f}% | {p.unique_count:,} |')
            
            lines.append('')
        
        # 高风险列（NaN > 20% 或有 inf）
        lines.append('### 🔴 高风险列')
        lines.append('')
        
        high_risk = [p for p in profiles.values() 
                     if p.null_pct > 20 or p.inf_count > 0 or p.neg_inf_count > 0]
        
        if high_risk:
            lines.append('| 列名 | NaN% | Inf | -Inf | 风险原因 |')
            lines.append('|------|------|-----|------|----------|')
            
            for p in sorted(high_risk, key=lambda x: x.null_pct, reverse=True):
                reasons = []
                if p.null_pct > 20:
                    reasons.append(f'高NaN({p.null_pct:.1f}%)')
                if p.inf_count > 0:
                    reasons.append(f'+inf({p.inf_count})')
                if p.neg_inf_count > 0:
                    reasons.append(f'-inf({p.neg_inf_count})')
                
                lines.append(f'| {p.name} | {p.null_pct:.1f}% | {p.inf_count} | {p.neg_inf_count} | {", ".join(reasons)} |')
        else:
            lines.append('无高风险列。')
        
        lines.append('')
        
        # 全列详情表（附录）
        lines.append('## 📑 附录：全列详情')
        lines.append('')
        lines.append('<details>')
        lines.append('<summary>点击展开全部 {} 列的详细统计</summary>'.format(len(profiles)))
        lines.append('')
        lines.append('| # | 列名 | 类别 | 类型 | NaN% | Zero% | Min | Max | Mean | Std | Skew | Kurt |')
        lines.append('|---|------|------|------|------|-------|-----|-----|------|-----|------|------|')
        
        for i, p in enumerate(profiles.values(), 1):
            min_val = f'{p.min_val:.4g}' if p.min_val is not None else '-'
            max_val = f'{p.max_val:.4g}' if p.max_val is not None else '-'
            mean_val = f'{p.mean:.4g}' if p.mean is not None else '-'
            std_val = f'{p.std:.4g}' if p.std is not None else '-'
            skew_val = f'{p.skew:.2f}' if p.skew is not None else '-'
            kurt_val = f'{p.kurtosis:.2f}' if p.kurtosis is not None else '-'
            
            lines.append(f'| {i} | {p.name} | {p.category} | {p.dtype} | {p.null_pct:.1f}% | {p.zero_pct:.1f}% | {min_val} | {max_val} | {mean_val} | {std_val} | {skew_val} | {kurt_val} |')
        
        lines.append('')
        lines.append('</details>')
        lines.append('')
        
        # 写入文件
        content = '\n'.join(lines)
        output_path.write_text(content, encoding='utf-8')
        logger.info(f"✅ Markdown 报告已生成: {output_path}")
        
        return content
    
    def generate_json(self, output_path: Path = None) -> Dict[str, Any]:
        """生成 JSON 格式报告"""
        if output_path is None:
            output_path = self.checker.output_dir / f'train_dq_report_{self.timestamp}.json'
        
        report = {
            'metadata': {
                'file': str(self.checker.train_path),
                'timestamp': datetime.now().isoformat(),
                'shape': self.checker.summary['shape'],
            },
            'summary': self.checker.summary['check_results'],
            'check_results': [r.to_dict() for r in self.checker.get_results()],
            'column_profiles': {k: v.to_dict() for k, v in self.checker.get_column_profiles().items()},
        }
        
        # 转换为可序列化格式
        report = _convert_to_serializable(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ JSON 报告已生成: {output_path}")
        
        return report
    
    def generate_all(self) -> Tuple[str, Dict[str, Any]]:
        """生成所有格式报告"""
        md_content = self.generate_markdown()
        json_data = self.generate_json()
        return md_content, json_data
