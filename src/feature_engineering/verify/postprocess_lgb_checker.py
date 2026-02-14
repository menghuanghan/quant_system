"""
LightGBM 后处理数据质量检查器

验收 train_lgb.parquet 的数据质量，确保：
1. 公共基础核验（与 GRU 数据对齐）
2. 保留数据的物理意义和缺失状态
3. 特征分布符合 LGB 的期望
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CheckLevel(Enum):
    """检查结果级别"""
    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def _convert_to_serializable(obj: Any) -> Any:
    """将对象转换为JSON可序列化格式"""
    if isinstance(obj, bool):  # Python bool 必须在 np.bool_ 之前检查
        return obj
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return str(obj)
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [_convert_to_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalar
        return _convert_to_serializable(obj.item())
    return obj


@dataclass
class CheckResult:
    """单项检查结果"""
    name: str
    dimension: str
    level: CheckLevel
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'dimension': self.dimension,
            'level': self.level.value,
            'passed': self.passed,
            'message': self.message,
            'details': _convert_to_serializable(self.details),
        }


@dataclass
class ColumnProfile:
    """列详细档案"""
    name: str
    dtype: str
    category: str  # feature, label, meta
    count: int
    null_count: int
    null_pct: float
    zero_count: int
    zero_pct: float
    inf_count: int
    neg_inf_count: int
    unique_count: int
    
    # 数值统计
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    skew: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return _convert_to_serializable(self.__dict__)


class LGBDataChecker:
    """LightGBM 数据质量检查器"""
    
    # 元数据列
    META_COLS = ['ts_code', 'trade_date']
    
    # 标签列
    LABEL_COLS = [
        'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d',
        'label_bin_1d', 'label_bin_2d', 'label_bin_3d', 'label_bin_5d', 'label_bin_10d'
    ]
    
    # 长尾特征（应保持原始分布，偏度应该很大）
    LONG_TAIL_FEATURES = [
        'amount', 'vol', 'total_mv', 'circ_mv',
        'buy_sm_amount', 'sell_sm_amount', 'buy_lg_amount', 'sell_lg_amount',
    ]
    
    # 技术指标（应保持原始范围）
    TECH_INDICATORS = {
        'rsi_6': (0, 100),
        'rsi_12': (0, 100),
        'rsi_24': (0, 100),
        'macd': (-100, 100),  # 原始 MACD 范围较大
        'macd_signal': (-100, 100),
        'macd_hist': (-50, 50),
    }
    
    # 量级检查特征
    MAGNITUDE_CHECKS = {
        'amount': (1e7, 1e10),  # 成交额：千万到百亿
        'total_mv': (1e9, 1e12),  # 市值：十亿到万亿
    }
    
    def __init__(
        self,
        lgb_path: str,
        gru_path: Optional[str] = None,
        output_dir: str = "reports",
    ):
        self.lgb_path = Path(lgb_path)
        self.gru_path = Path(gru_path) if gru_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[CheckResult] = []
        self.column_profiles: Dict[str, ColumnProfile] = {}
        self.lgb_df: Optional[pd.DataFrame] = None
        self.gru_df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> bool:
        """加载数据"""
        logger.info(f"📖 加载 LGB 数据: {self.lgb_path}")
        try:
            self.lgb_df = pd.read_parquet(self.lgb_path)
            logger.info(f"   ✓ LGB: {len(self.lgb_df):,} 行, {len(self.lgb_df.columns)} 列")
        except Exception as e:
            logger.error(f"   ✗ 加载失败: {e}")
            return False
        
        if self.gru_path and self.gru_path.exists():
            logger.info(f"📖 加载 GRU 数据（用于对齐检查）: {self.gru_path}")
            try:
                self.gru_df = pd.read_parquet(self.gru_path)
                logger.info(f"   ✓ GRU: {len(self.gru_df):,} 行, {len(self.gru_df.columns)} 列")
            except Exception as e:
                logger.warning(f"   ⚠ GRU 数据加载失败: {e}")
        
        return True
    
    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有检查"""
        logger.info("=" * 60)
        logger.info("🔍 LightGBM 数据质量检查")
        logger.info("=" * 60)
        
        # 1. 加载数据
        if not self.load_data():
            return {"error": "数据加载失败"}
        
        # 2. 公共基础核验
        self._check_common_baseline()
        
        # 3. LGB 专用核验
        self._check_magnitude()
        self._check_feature_distribution()
        self._check_long_tail_skewness()
        self._check_no_standardization()
        
        # 4. 生成列档案
        self._generate_column_profiles()
        
        # 5. 汇总结果
        return self._summarize_results()
    
    def _check_common_baseline(self):
        """公共基础核验"""
        logger.info("-" * 40)
        logger.info("📋 公共基础核验")
        logger.info("-" * 40)
        
        df = self.lgb_df
        
        # 1. 行数检查（与 GRU 对齐）
        if self.gru_df is not None:
            lgb_rows = len(df)
            gru_rows = len(self.gru_df)
            
            # 注意：LGB 和 GRU 的时间窗口不同，所以行数可能不同
            # LGB: 2019-2020 剔除
            # GRU: 2019.01-2020.06 剔除
            # 但在共同的时间窗口（2020.07-2025）内应该一致
            
            # 检查共同时间窗口的行数
            lgb_dates = df['trade_date'].unique()
            gru_dates = self.gru_df['trade_date'].unique()
            common_dates = set(lgb_dates) & set(gru_dates)
            
            lgb_common = df[df['trade_date'].isin(common_dates)]
            gru_common = self.gru_df[self.gru_df['trade_date'].isin(common_dates)]
            
            rows_match = len(lgb_common) == len(gru_common)
            
            self.results.append(CheckResult(
                name="行数对齐（共同时间窗口）",
                dimension="公共基础",
                level=CheckLevel.PASS if rows_match else CheckLevel.ERROR,
                passed=rows_match,
                message=f"LGB={len(lgb_common):,}, GRU={len(gru_common):,}",
                details={
                    'lgb_total_rows': lgb_rows,
                    'gru_total_rows': gru_rows,
                    'common_dates_count': len(common_dates),
                    'lgb_common_rows': len(lgb_common),
                    'gru_common_rows': len(gru_common),
                }
            ))
            logger.info(f"   行数对齐: {'✓' if rows_match else '✗'} LGB={len(lgb_common):,}, GRU={len(gru_common):,}")
        
        # 2. 标签清洗彻底性
        ret_5d_nan = df['ret_5d'].isna().sum()
        label_clean = ret_5d_nan == 0
        
        self.results.append(CheckResult(
            name="标签清洗彻底性（ret_5d）",
            dimension="公共基础",
            level=CheckLevel.PASS if label_clean else CheckLevel.CRITICAL,
            passed=label_clean,
            message=f"ret_5d NaN 数量: {ret_5d_nan}",
            details={'ret_5d_nan_count': ret_5d_nan}
        ))
        logger.info(f"   标签清洗: {'✓' if label_clean else '✗'} ret_5d NaN={ret_5d_nan}")
        
        # 3. 标签一致性（与 GRU）
        if self.gru_df is not None and len(common_dates) > 0:
            # 合并共同数据
            lgb_common = lgb_common.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
            gru_common = gru_common.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
            
            # 检查 ret_5d
            if len(lgb_common) == len(gru_common):
                ret_5d_match = np.allclose(
                    lgb_common['ret_5d'].values, 
                    gru_common['ret_5d'].values, 
                    rtol=1e-5, 
                    equal_nan=True
                )
            else:
                ret_5d_match = False
            
            self.results.append(CheckResult(
                name="标签一致性（ret_5d）",
                dimension="公共基础",
                level=CheckLevel.PASS if ret_5d_match else CheckLevel.ERROR,
                passed=ret_5d_match,
                message=f"LGB 与 GRU 的 ret_5d 是否一致: {ret_5d_match}",
                details={
                    'lgb_ret_5d_mean': lgb_common['ret_5d'].mean() if len(lgb_common) > 0 else None,
                    'gru_ret_5d_mean': gru_common['ret_5d'].mean() if len(gru_common) > 0 else None,
                }
            ))
            logger.info(f"   标签一致: {'✓' if ret_5d_match else '✗'}")
        
        # 4. 排序验证
        is_sorted_date = df['trade_date'].is_monotonic_increasing or \
                         (df['trade_date'].values == np.sort(df['trade_date'].values)).all()
        
        # 更精确的排序检查
        df_sorted_check = df.sort_values(['trade_date', 'ts_code'])
        is_sorted = (df.index == df_sorted_check.index).all() if len(df) == len(df_sorted_check) else False
        
        self.results.append(CheckResult(
            name="排序验证（trade_date, ts_code）",
            dimension="公共基础",
            level=CheckLevel.PASS if is_sorted else CheckLevel.WARNING,
            passed=is_sorted,
            message=f"按 (trade_date, ts_code) 升序: {is_sorted}",
            details={'is_sorted': is_sorted}
        ))
        logger.info(f"   排序验证: {'✓' if is_sorted else '⚠'}")
    
    def _check_magnitude(self):
        """检查数值量级"""
        logger.info("-" * 40)
        logger.info("📋 数值量级核验")
        logger.info("-" * 40)
        
        df = self.lgb_df
        
        for col, (min_expected, max_expected) in self.MAGNITUDE_CHECKS.items():
            if col not in df.columns:
                continue
            
            col_mean = df[col].mean()
            in_range = min_expected <= col_mean <= max_expected
            
            self.results.append(CheckResult(
                name=f"量级检查: {col}",
                dimension="数值量级",
                level=CheckLevel.PASS if in_range else CheckLevel.ERROR,
                passed=in_range,
                message=f"均值={col_mean:.2e}, 期望范围=[{min_expected:.0e}, {max_expected:.0e}]",
                details={
                    'mean': col_mean,
                    'expected_min': min_expected,
                    'expected_max': max_expected,
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                }
            ))
            status = '✓' if in_range else '✗'
            logger.info(f"   {col}: {status} mean={col_mean:.2e} (期望 {min_expected:.0e}~{max_expected:.0e})")
    
    def _check_feature_distribution(self):
        """检查特征分布（技术指标应保持原始范围）"""
        logger.info("-" * 40)
        logger.info("📋 特征分布核验（技术指标）")
        logger.info("-" * 40)
        
        df = self.lgb_df
        
        for col, (expected_min, expected_max) in self.TECH_INDICATORS.items():
            if col not in df.columns:
                continue
            
            col_mean = df[col].mean()
            col_std = df[col].std()
            col_min = df[col].min()
            col_max = df[col].max()
            
            # 检查是否被标准化（均值接近 0，标准差接近 1）
            is_standardized = abs(col_mean) < 0.5 and 0.5 < col_std < 1.5
            
            # 对于 RSI，期望范围是 0-100
            if col.startswith('rsi'):
                in_expected_range = 0 <= col_min and col_max <= 100
            else:
                # 对于 MACD 等，检查是否在合理范围内
                in_expected_range = expected_min <= col_mean <= expected_max or not is_standardized
            
            passed = not is_standardized  # LGB 不应该被标准化
            
            self.results.append(CheckResult(
                name=f"特征分布: {col}",
                dimension="特征分布",
                level=CheckLevel.PASS if passed else CheckLevel.ERROR,
                passed=passed,
                message=f"mean={col_mean:.4f}, std={col_std:.4f}, 是否被标准化={is_standardized}",
                details={
                    'mean': col_mean,
                    'std': col_std,
                    'min': col_min,
                    'max': col_max,
                    'is_standardized': is_standardized,
                }
            ))
            status = '✓' if passed else '✗'
            logger.info(f"   {col}: {status} mean={col_mean:.2f}, std={col_std:.2f}, 标准化={is_standardized}")
    
    def _check_long_tail_skewness(self):
        """检查长尾特征的偏度（应该很大，>5 表示未被 Log 处理）"""
        logger.info("-" * 40)
        logger.info("📋 长尾特征偏度核验")
        logger.info("-" * 40)
        
        df = self.lgb_df
        
        for col in self.LONG_TAIL_FEATURES:
            if col not in df.columns:
                continue
            
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            skewness = col_data.skew()
            
            # 偏度 > 5 表示保持原始长尾分布
            has_high_skew = skewness > 5
            
            self.results.append(CheckResult(
                name=f"偏度检查: {col}",
                dimension="长尾特征",
                level=CheckLevel.PASS if has_high_skew else CheckLevel.WARNING,
                passed=has_high_skew,
                message=f"偏度={skewness:.2f}, 期望 >5 表示保持原始分布",
                details={
                    'skewness': skewness,
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                }
            ))
            status = '✓' if has_high_skew else '⚠'
            logger.info(f"   {col}: {status} skew={skewness:.2f} (期望 >5)")
    
    def _check_no_standardization(self):
        """确认 LGB 数据没有被 Z-Score 标准化"""
        logger.info("-" * 40)
        logger.info("📋 确认未做标准化")
        logger.info("-" * 40)
        
        df = self.lgb_df
        
        # 检查几个关键特征
        key_features = ['amount', 'vol', 'total_mv', 'turnover']
        
        standardized_cols = []
        for col in key_features:
            if col not in df.columns:
                continue
            
            col_mean = df[col].mean()
            col_std = df[col].std()
            
            # 如果均值接近 0 且标准差接近 1，说明被标准化了
            is_standardized = abs(col_mean) < 0.5 and 0.5 < col_std < 1.5
            
            if is_standardized:
                standardized_cols.append(col)
        
        passed = len(standardized_cols) == 0
        
        self.results.append(CheckResult(
            name="确认未做 Z-Score 标准化",
            dimension="标准化检查",
            level=CheckLevel.PASS if passed else CheckLevel.ERROR,
            passed=passed,
            message=f"被标准化的列: {standardized_cols if standardized_cols else '无'}",
            details={'standardized_cols': standardized_cols}
        ))
        logger.info(f"   未标准化: {'✓' if passed else '✗'} 被标准化列={standardized_cols}")
    
    def _generate_column_profiles(self):
        """生成所有列的详细档案"""
        logger.info("-" * 40)
        logger.info("📋 生成列档案")
        logger.info("-" * 40)
        
        df = self.lgb_df
        
        for col in df.columns:
            # 确定列类别
            if col in self.META_COLS:
                category = 'meta'
            elif col in self.LABEL_COLS:
                category = 'label'
            else:
                category = 'feature'
            
            col_data = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(col_data)
            
            # 基础统计
            count = len(col_data)
            null_count = col_data.isna().sum()
            null_pct = null_count / count * 100 if count > 0 else 0
            unique_count = col_data.nunique()
            
            # 数值特有统计
            if is_numeric:
                zero_count = (col_data == 0).sum()
                zero_pct = zero_count / count * 100 if count > 0 else 0
                inf_count = np.isinf(col_data).sum() if pd.api.types.is_float_dtype(col_data) else 0
                neg_inf_count = (col_data == -np.inf).sum() if pd.api.types.is_float_dtype(col_data) else 0
                
                col_clean = col_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                profile = ColumnProfile(
                    name=col,
                    dtype=str(col_data.dtype),
                    category=category,
                    count=count,
                    null_count=null_count,
                    null_pct=null_pct,
                    zero_count=zero_count,
                    zero_pct=zero_pct,
                    inf_count=inf_count,
                    neg_inf_count=neg_inf_count,
                    unique_count=unique_count,
                    mean=col_clean.mean() if len(col_clean) > 0 else None,
                    std=col_clean.std() if len(col_clean) > 0 else None,
                    min_val=col_clean.min() if len(col_clean) > 0 else None,
                    max_val=col_clean.max() if len(col_clean) > 0 else None,
                    q1=col_clean.quantile(0.25) if len(col_clean) > 0 else None,
                    median=col_clean.median() if len(col_clean) > 0 else None,
                    q3=col_clean.quantile(0.75) if len(col_clean) > 0 else None,
                    skew=col_clean.skew() if len(col_clean) > 0 else None,
                )
            else:
                profile = ColumnProfile(
                    name=col,
                    dtype=str(col_data.dtype),
                    category=category,
                    count=count,
                    null_count=null_count,
                    null_pct=null_pct,
                    zero_count=0,
                    zero_pct=0,
                    inf_count=0,
                    neg_inf_count=0,
                    unique_count=unique_count,
                )
            
            self.column_profiles[col] = profile
        
        logger.info(f"   ✓ 生成 {len(self.column_profiles)} 列档案")
    
    def _summarize_results(self) -> Dict[str, Any]:
        """汇总检查结果"""
        logger.info("=" * 60)
        logger.info("📊 检查结果汇总")
        logger.info("=" * 60)
        
        # 按维度分组
        by_dimension = {}
        for r in self.results:
            if r.dimension not in by_dimension:
                by_dimension[r.dimension] = []
            by_dimension[r.dimension].append(r)
        
        # 统计
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        # 按级别统计
        by_level = {}
        for r in self.results:
            level = r.level.value
            by_level[level] = by_level.get(level, 0) + 1
        
        logger.info(f"   总检查项: {total}")
        logger.info(f"   通过: {passed} ({passed/total*100:.1f}%)")
        logger.info(f"   失败: {failed} ({failed/total*100:.1f}%)")
        
        for level, count in by_level.items():
            logger.info(f"   {level}: {count}")
        
        # 列统计
        feature_cols = [p for p in self.column_profiles.values() if p.category == 'feature']
        label_cols = [p for p in self.column_profiles.values() if p.category == 'label']
        meta_cols = [p for p in self.column_profiles.values() if p.category == 'meta']
        
        logger.info(f"   特征列: {len(feature_cols)}")
        logger.info(f"   标签列: {len(label_cols)}")
        logger.info(f"   元数据列: {len(meta_cols)}")
        
        summary = {
            'check_time': datetime.now().isoformat(),
            'data_file': str(self.lgb_path),
            'total_rows': len(self.lgb_df),
            'total_cols': len(self.lgb_df.columns),
            'summary': {
                'total_checks': total,
                'passed': passed,
                'failed': failed,
                'pass_rate': passed / total * 100 if total > 0 else 0,
                'by_level': by_level,
            },
            'column_summary': {
                'feature_cols': len(feature_cols),
                'label_cols': len(label_cols),
                'meta_cols': len(meta_cols),
            },
            'checks': [r.to_dict() for r in self.results],
            'column_profiles': {k: v.to_dict() for k, v in self.column_profiles.items()},
        }
        
        return summary
    
    def save_report(self, summary: Dict[str, Any]) -> Tuple[str, str]:
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 转换所有值为 JSON 可序列化格式
        serializable_summary = _convert_to_serializable(summary)
        
        # JSON 报告
        json_path = self.output_dir / f"lgb_dq_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, ensure_ascii=False, indent=2)
        
        # Markdown 报告
        md_path = self.output_dir / f"lgb_dq_report_{timestamp}.md"
        md_content = self._generate_markdown_report(summary)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"   📄 JSON 报告: {json_path}")
        logger.info(f"   📄 Markdown 报告: {md_path}")
        
        return str(json_path), str(md_path)
    
    def _generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """生成 Markdown 格式报告"""
        lines = [
            "# LightGBM 数据质量检查报告",
            "",
            f"**检查时间**: {summary['check_time']}",
            f"**数据文件**: `{summary['data_file']}`",
            f"**数据规模**: {summary['total_rows']:,} 行 × {summary['total_cols']} 列",
            "",
            "---",
            "",
            "## 检查结果摘要",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 总检查项 | {summary['summary']['total_checks']} |",
            f"| 通过 | {summary['summary']['passed']} |",
            f"| 失败 | {summary['summary']['failed']} |",
            f"| 通过率 | {summary['summary']['pass_rate']:.1f}% |",
            "",
        ]
        
        # 按级别统计
        lines.append("### 按级别统计")
        lines.append("")
        lines.append("| 级别 | 数量 |")
        lines.append("|------|------|")
        for level, count in summary['summary']['by_level'].items():
            lines.append(f"| {level} | {count} |")
        lines.append("")
        
        # 详细检查结果
        lines.append("---")
        lines.append("")
        lines.append("## 详细检查结果")
        lines.append("")
        
        # 按维度分组
        checks_by_dim = {}
        for check in summary['checks']:
            dim = check['dimension']
            if dim not in checks_by_dim:
                checks_by_dim[dim] = []
            checks_by_dim[dim].append(check)
        
        for dim, checks in checks_by_dim.items():
            lines.append(f"### {dim}")
            lines.append("")
            lines.append("| 检查项 | 结果 | 级别 | 说明 |")
            lines.append("|--------|------|------|------|")
            for check in checks:
                status = "✅" if check['passed'] else "❌"
                lines.append(f"| {check['name']} | {status} | {check['level']} | {check['message']} |")
            lines.append("")
        
        # 列档案摘要
        lines.append("---")
        lines.append("")
        lines.append("## 列档案摘要")
        lines.append("")
        
        # 特征列统计
        feature_profiles = [p for p in summary['column_profiles'].values() if p['category'] == 'feature']
        
        if feature_profiles:
            lines.append("### 特征列分布")
            lines.append("")
            lines.append("| 特征 | 类型 | NaN% | 零值% | 均值 | 标准差 | 最小值 | 最大值 | 偏度 |")
            lines.append("|------|------|------|-------|------|--------|--------|--------|------|")
            
            # 只显示前 50 个重要特征
            important_features = ['amount', 'vol', 'total_mv', 'circ_mv', 'turnover', 'return_1d',
                                 'rsi_6', 'rsi_12', 'macd', 'bias_5', 'bias_20']
            
            for feat in important_features:
                if feat in summary['column_profiles']:
                    p = summary['column_profiles'][feat]
                    mean = f"{p['mean']:.4g}" if p['mean'] is not None else "N/A"
                    std = f"{p['std']:.4g}" if p['std'] is not None else "N/A"
                    min_v = f"{p['min_val']:.4g}" if p['min_val'] is not None else "N/A"
                    max_v = f"{p['max_val']:.4g}" if p['max_val'] is not None else "N/A"
                    skew = f"{p['skew']:.2f}" if p['skew'] is not None else "N/A"
                    lines.append(f"| {p['name']} | {p['dtype']} | {p['null_pct']:.2f}% | {p['zero_pct']:.2f}% | {mean} | {std} | {min_v} | {max_v} | {skew} |")
            
            lines.append("")
        
        # 标签列统计
        label_profiles = [p for p in summary['column_profiles'].values() if p['category'] == 'label']
        
        if label_profiles:
            lines.append("### 标签列分布")
            lines.append("")
            lines.append("| 标签 | 类型 | NaN% | 均值 | 标准差 | 最小值 | 最大值 |")
            lines.append("|------|------|------|------|--------|--------|--------|")
            
            for p in label_profiles:
                mean = f"{p['mean']:.4g}" if p['mean'] is not None else "N/A"
                std = f"{p['std']:.4g}" if p['std'] is not None else "N/A"
                min_v = f"{p['min_val']:.4g}" if p['min_val'] is not None else "N/A"
                max_v = f"{p['max_val']:.4g}" if p['max_val'] is not None else "N/A"
                lines.append(f"| {p['name']} | {p['dtype']} | {p['null_pct']:.2f}% | {mean} | {std} | {min_v} | {max_v} |")
            
            lines.append("")
        
        # 数据质量结论
        lines.append("---")
        lines.append("")
        lines.append("## 数据质量结论")
        lines.append("")
        
        pass_rate = summary['summary']['pass_rate']
        if pass_rate >= 90:
            lines.append("✅ **数据质量良好**，可以进入模型训练阶段。")
        elif pass_rate >= 70:
            lines.append("⚠️ **数据质量一般**，建议检查失败项后再进行训练。")
        else:
            lines.append("❌ **数据质量较差**，需要重新处理数据。")
        
        lines.append("")
        
        return "\n".join(lines)


def run_lgb_check(
    lgb_path: str = "data/features/structured/train_lgb.parquet",
    gru_path: Optional[str] = "data/features/structured/train_gru.parquet",
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """运行 LGB 数据质量检查"""
    checker = LGBDataChecker(lgb_path, gru_path, output_dir)
    summary = checker.run_all_checks()
    json_path, md_path = checker.save_report(summary)
    
    return {
        'summary': summary,
        'json_path': json_path,
        'md_path': md_path,
    }


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    
    result = run_lgb_check()
    
    # 输出结果
    pass_rate = result['summary']['summary']['pass_rate']
    print(f"\n通过率: {pass_rate:.1f}%")
    print(f"JSON 报告: {result['json_path']}")
    print(f"Markdown 报告: {result['md_path']}")
    
    sys.exit(0 if pass_rate >= 70 else 1)
