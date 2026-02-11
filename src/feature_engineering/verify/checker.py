"""
Merger 预处理数据质量检查器

4 维度全面检查：
1. 结构与完整性
2. 数值与单位一致性  
3. 合并逻辑有效性
4. 预处理效果
"""

import logging
from dataclasses import dataclass, field
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
            'details': self.details,
        }


@dataclass
class ColumnStats:
    """列统计信息"""
    name: str
    dtype: str
    count: int
    null_count: int
    null_pct: float
    unique_count: int
    
    # 数值列统计
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    
    # 分类列统计
    top_values: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class MergerPreprocessChecker:
    """
    Merger 预处理数据质量检查器
    
    检查 data/features/temp/merger_preprocess.parquet 的数据质量
    """
    
    # 预期字段配置（预处理后）
    EXPECTED_CORE_FIELDS = [
        'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount',
        'return_1d', 'is_trading', 'limit_ratio',  # pct_chg 已转换为 return_1d
    ]
    
    EXPECTED_FUNDAMENTAL_FIELDS = [
        'total_mv', 'circ_mv', 'ep', 'bp', 'sp', 'roe', 'roa',
        'revenue_yoy', 'net_profit_yoy',
    ]
    
    EXPECTED_MONEY_FLOW_FIELDS = [
        'net_mf_amount', 'buy_sm_amount', 'sell_sm_amount',
        'buy_md_amount', 'sell_md_amount', 'buy_lg_amount', 'sell_lg_amount',
        'buy_elg_amount', 'sell_elg_amount',
    ]
    
    EXPECTED_MACRO_FIELDS = [
        'shibor_1w', 'gdp_yoy', 'cpi_yoy', 'pmi',
    ]
    
    # 金额字段期望量级 (均值)
    AMOUNT_MAGNITUDE_EXPECTATIONS = {
        'amount': (1e7, 1e9, '1亿~10亿'),           # 成交额
        'total_mv': (1e9, 5e11, '10亿~5000亿'),     # 总市值
        'circ_mv': (1e9, 5e11, '10亿~5000亿'),      # 流通市值
        'buy_sm_amount': (1e5, 1e8, '10万~1亿'),    # 小单买入
        'sell_sm_amount': (1e5, 1e8, '10万~1亿'),   # 小单卖出
        'buy_md_amount': (1e5, 1e8, '10万~1亿'),    # 中单买入
        'sell_md_amount': (1e5, 1e8, '10万~1亿'),   # 中单卖出
        'buy_lg_amount': (1e5, 1e8, '10万~1亿'),    # 大单买入
        'sell_lg_amount': (1e5, 1e8, '10万~1亿'),   # 大单卖出
        'buy_elg_amount': (1e5, 1e8, '10万~1亿'),   # 超大单买入
        'sell_elg_amount': (1e5, 1e8, '10万~1亿'),  # 超大单卖出
        'net_mf_amount': (-1e9, 1e9, '-10亿~10亿'), # 净流入（可负）
    }
    
    def __init__(
        self,
        merger_path: Path,
        dwd_price_path: Path,
        output_dir: Optional[Path] = None,
    ):
        """
        初始化检查器
        
        Args:
            merger_path: merger_preprocess.parquet 路径
            dwd_price_path: dwd_stock_price.parquet 路径（用于行数对比）
            output_dir: 报告输出目录
        """
        self.merger_path = Path(merger_path)
        self.dwd_price_path = Path(dwd_price_path)
        self.output_dir = Path(output_dir) if output_dir else Path('reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df: Optional[pd.DataFrame] = None
        self.dwd_price_df: Optional[pd.DataFrame] = None
        self.results: List[CheckResult] = []
        self.column_stats: Dict[str, ColumnStats] = {}
        
    def load_data(self) -> None:
        """加载数据"""
        logger.info(f"加载 merger_preprocess: {self.merger_path}")
        self.df = pd.read_parquet(self.merger_path)
        logger.info(f"  ✓ 已加载: {len(self.df):,} 行, {len(self.df.columns)} 列")
        
        logger.info(f"加载 dwd_stock_price: {self.dwd_price_path}")
        self.dwd_price_df = pd.read_parquet(self.dwd_price_path)
        logger.info(f"  ✓ 已加载: {len(self.dwd_price_df):,} 行")
        
    def run_all_checks(self) -> List[CheckResult]:
        """运行所有检查"""
        if self.df is None:
            self.load_data()
            
        logger.info("=" * 70)
        logger.info("🔍 开始数据质量检查")
        logger.info("=" * 70)
        
        # 计算列统计信息
        self._compute_column_stats()
        
        # 维度1: 结构与完整性
        self._check_structure_integrity()
        
        # 维度2: 数值与单位一致性
        self._check_value_unit_consistency()
        
        # 维度3: 合并逻辑有效性
        self._check_merge_logic()
        
        # 维度4: 预处理效果
        self._check_preprocess_effect()
        
        # 汇总结果
        self._print_summary()
        
        return self.results
    
    def _add_result(self, result: CheckResult) -> None:
        """添加检查结果"""
        self.results.append(result)
        icon = "✅" if result.passed else "❌"
        logger.info(f"  {icon} [{result.level.value}] {result.name}: {result.message}")
    
    def _compute_column_stats(self) -> None:
        """计算所有列的统计信息"""
        logger.info("📊 计算列统计信息...")
        
        for col in self.df.columns:
            series = self.df[col]
            dtype = str(series.dtype)
            count = len(series)
            null_count = int(series.isna().sum())
            null_pct = null_count / count * 100 if count > 0 else 0
            unique_count = int(series.nunique())
            
            stats = ColumnStats(
                name=col,
                dtype=dtype,
                count=count,
                null_count=null_count,
                null_pct=null_pct,
                unique_count=unique_count,
            )
            
            # 数值列统计（兼容 pandas nullable dtypes 如 Float32）
            is_numeric = False
            try:
                is_numeric = pd.api.types.is_numeric_dtype(series)
            except:
                is_numeric = np.issubdtype(series.dtype, np.number)
            
            if is_numeric:
                valid = series.dropna()
                if len(valid) > 0:
                    stats.mean = float(valid.mean())
                    stats.std = float(valid.std())
                    stats.min_val = float(valid.min())
                    stats.max_val = float(valid.max())
                    stats.q1 = float(valid.quantile(0.25))
                    stats.median = float(valid.quantile(0.50))
                    stats.q3 = float(valid.quantile(0.75))
            
            # 分类列统计（前5个值）
            elif series.dtype == 'object' or series.dtype.name == 'category':
                top = series.value_counts().head(5)
                stats.top_values = {str(k): int(v) for k, v in top.items()}
            
            self.column_stats[col] = stats
        
        logger.info(f"  ✓ 统计完成: {len(self.column_stats)} 列")
    
    # ========== 维度1: 结构与完整性 ==========
    def _check_structure_integrity(self) -> None:
        """维度1: 结构与完整性检查"""
        logger.info("-" * 50)
        logger.info("📋 维度1: 结构与完整性检查")
        logger.info("-" * 50)
        
        # 1.1 行数检查
        self._check_row_count()
        
        # 1.2 主键唯一性
        self._check_primary_key()
        
        # 1.3 核心字段存在性
        self._check_core_fields()
        
        # 1.4 重名列检查
        self._check_duplicate_columns()
        
        # 1.5 排序检查
        self._check_sorting()
    
    def _check_row_count(self) -> None:
        """检查行数是否与 DWD 一致"""
        merger_rows = len(self.df)
        dwd_rows = len(self.dwd_price_df)
        diff = merger_rows - dwd_rows
        diff_pct = abs(diff) / dwd_rows * 100 if dwd_rows > 0 else 100
        
        # 允许 5% 的差异（部分股票可能因条件过滤被排除）
        if diff_pct > 5:
            if merger_rows > dwd_rows * 1.5:
                level = CheckLevel.CRITICAL
                msg = f"行数暴增 {diff_pct:.1f}%，疑似笛卡尔积爆炸"
            else:
                level = CheckLevel.ERROR
                msg = f"行数骤减 {diff_pct:.1f}%，疑似 Inner Join 数据丢失"
            passed = False
        elif diff_pct > 1:
            level = CheckLevel.WARNING
            msg = f"行数差异 {diff_pct:.2f}%，需关注"
            passed = True
        else:
            level = CheckLevel.PASS
            msg = f"行数正常 (merger={merger_rows:,}, dwd={dwd_rows:,})"
            passed = True
        
        self._add_result(CheckResult(
            name='行数检查',
            dimension='结构完整性',
            level=level,
            passed=passed,
            message=msg,
            details={
                'merger_rows': merger_rows,
                'dwd_rows': dwd_rows,
                'diff': diff,
                'diff_pct': diff_pct,
            }
        ))
    
    def _check_primary_key(self) -> None:
        """检查主键唯一性"""
        dup_count = self.df.duplicated(subset=['ts_code', 'trade_date']).sum()
        
        if dup_count > 0:
            level = CheckLevel.CRITICAL
            msg = f"发现 {dup_count:,} 行主键重复"
            passed = False
        else:
            level = CheckLevel.PASS
            msg = f"主键 (ts_code, trade_date) 唯一"
            passed = True
        
        self._add_result(CheckResult(
            name='主键唯一性',
            dimension='结构完整性',
            level=level,
            passed=passed,
            message=msg,
            details={'duplicate_count': int(dup_count)}
        ))
    
    def _check_core_fields(self) -> None:
        """检查核心字段是否存在"""
        all_expected = (
            self.EXPECTED_CORE_FIELDS +
            self.EXPECTED_FUNDAMENTAL_FIELDS +
            self.EXPECTED_MONEY_FLOW_FIELDS +
            self.EXPECTED_MACRO_FIELDS
        )
        
        missing = [f for f in all_expected if f not in self.df.columns]
        
        if missing:
            level = CheckLevel.ERROR
            msg = f"缺失 {len(missing)} 个核心字段: {missing[:5]}..."
            passed = False
        else:
            level = CheckLevel.PASS
            msg = f"所有 {len(all_expected)} 个核心字段存在"
            passed = True
        
        self._add_result(CheckResult(
            name='核心字段存在性',
            dimension='结构完整性',
            level=level,
            passed=passed,
            message=msg,
            details={'missing_fields': missing, 'total_expected': len(all_expected)}
        ))
    
    def _check_duplicate_columns(self) -> None:
        """检查是否有重名列（带后缀）"""
        suffix_cols = [c for c in self.df.columns if c.endswith('_x') or c.endswith('_y')]
        
        if suffix_cols:
            level = CheckLevel.WARNING
            msg = f"发现 {len(suffix_cols)} 个后缀列: {suffix_cols[:5]}"
            passed = False
        else:
            level = CheckLevel.PASS
            msg = "无重名列后缀"
            passed = True
        
        self._add_result(CheckResult(
            name='重名列检查',
            dimension='结构完整性',
            level=level,
            passed=passed,
            message=msg,
            details={'suffix_columns': suffix_cols}
        ))
    
    def _check_sorting(self) -> None:
        """检查排序"""
        # 检查是否按 ts_code -> trade_date 排序
        is_sorted = (
            self.df['ts_code'].is_monotonic_increasing or
            (self.df.groupby('ts_code')['trade_date'].apply(lambda x: x.is_monotonic_increasing).all())
        )
        
        # 抽样检查
        sample_idx = [0, len(self.df) // 4, len(self.df) // 2, len(self.df) * 3 // 4, len(self.df) - 1]
        sample = self.df.iloc[sample_idx][['ts_code', 'trade_date']].to_dict('records')
        
        if is_sorted:
            level = CheckLevel.PASS
            msg = "数据已按 ts_code -> trade_date 排序"
            passed = True
        else:
            level = CheckLevel.WARNING
            msg = "数据未完全排序，可能影响 Rolling/Lag 计算"
            passed = False
        
        self._add_result(CheckResult(
            name='排序检查',
            dimension='结构完整性',
            level=level,
            passed=passed,
            message=msg,
            details={'sample': sample}
        ))
    
    # ========== 维度2: 数值与单位一致性 ==========
    def _check_value_unit_consistency(self) -> None:
        """维度2: 数值与单位一致性检查"""
        logger.info("-" * 50)
        logger.info("📋 维度2: 数值与单位一致性检查")
        logger.info("-" * 50)
        
        # 2.1 金额字段量级检查
        self._check_amount_magnitude()
        
        # 2.2 价格逻辑自洽检查
        self._check_price_consistency()
        
        # 2.3 收益率范围检查
        self._check_return_range()
    
    def _check_amount_magnitude(self) -> None:
        """检查金额字段量级"""
        issues = []
        passed_fields = []
        
        for field, (min_exp, max_exp, desc) in self.AMOUNT_MAGNITUDE_EXPECTATIONS.items():
            if field not in self.df.columns:
                continue
            
            mean_val = self.df[field].mean()
            
            # 净流入可以为负
            if field == 'net_mf_amount':
                abs_mean = abs(mean_val)
                in_range = abs_mean <= max_exp
            else:
                in_range = min_exp <= mean_val <= max_exp
            
            if not in_range:
                issues.append({
                    'field': field,
                    'mean': mean_val,
                    'expected': desc,
                    'magnitude': f"{mean_val:.2e}",
                })
            else:
                passed_fields.append(field)
        
        if issues:
            level = CheckLevel.ERROR
            msg = f"{len(issues)} 个金额字段量级异常"
            passed = False
        else:
            level = CheckLevel.PASS
            msg = f"所有 {len(passed_fields)} 个金额字段量级正常"
            passed = True
        
        self._add_result(CheckResult(
            name='金额字段量级',
            dimension='数值单位一致性',
            level=level,
            passed=passed,
            message=msg,
            details={'issues': issues, 'passed_fields': passed_fields}
        ))
    
    def _check_price_consistency(self) -> None:
        """检查价格逻辑自洽性: amount / (vol * 100) ≈ close"""
        if 'amount' not in self.df.columns or 'vol' not in self.df.columns:
            return
        
        # 过滤有效数据
        valid = self.df[(self.df['vol'] > 0) & (self.df['amount'] > 0)].copy()
        
        # 计算隐含价格 (vol 单位是手=100股)
        valid['calc_price'] = valid['amount'] / (valid['vol'] * 100)
        
        # 计算与 close 的相关性
        corr = valid['calc_price'].corr(valid['close'])
        
        # 计算相对误差
        valid['rel_error'] = abs(valid['calc_price'] - valid['close']) / valid['close']
        mean_error = valid['rel_error'].mean() * 100
        median_error = valid['rel_error'].median() * 100
        
        if corr < 0.99:
            level = CheckLevel.ERROR
            msg = f"价格相关性 {corr:.4f} < 0.99，疑似单位问题"
            passed = False
        elif mean_error > 20:
            level = CheckLevel.WARNING
            msg = f"价格误差均值 {mean_error:.1f}% > 20%"
            passed = True
        else:
            level = CheckLevel.PASS
            msg = f"价格一致 (corr={corr:.4f}, error={mean_error:.2f}%)"
            passed = True
        
        self._add_result(CheckResult(
            name='价格逻辑自洽',
            dimension='数值单位一致性',
            level=level,
            passed=passed,
            message=msg,
            details={
                'correlation': float(corr),
                'mean_error_pct': float(mean_error),
                'median_error_pct': float(median_error),
            }
        ))
    
    def _check_return_range(self) -> None:
        """检查收益率范围"""
        if 'return_1d' not in self.df.columns:
            return
        
        ret = self.df['return_1d'].dropna()
        min_ret = ret.min()
        max_ret = ret.max()
        
        # A 股涨跌停限制，合理范围 [-30%, 100%]
        if min_ret < -0.5 or max_ret > 2.0:
            level = CheckLevel.WARNING
            msg = f"收益率范围异常: [{min_ret:.2%}, {max_ret:.2%}]"
            passed = False
        else:
            level = CheckLevel.PASS
            msg = f"收益率范围正常: [{min_ret:.2%}, {max_ret:.2%}]"
            passed = True
        
        # 极值统计
        extreme_neg = (ret < -0.2).sum()
        extreme_pos = (ret > 0.5).sum()
        
        self._add_result(CheckResult(
            name='收益率范围',
            dimension='数值单位一致性',
            level=level,
            passed=passed,
            message=msg,
            details={
                'min': float(min_ret),
                'max': float(max_ret),
                'extreme_negative_count': int(extreme_neg),
                'extreme_positive_count': int(extreme_pos),
            }
        ))
    
    # ========== 维度3: 合并逻辑有效性 ==========
    def _check_merge_logic(self) -> None:
        """维度3: 合并逻辑有效性检查"""
        logger.info("-" * 50)
        logger.info("📋 维度3: 合并逻辑有效性检查")
        logger.info("-" * 50)
        
        # 3.1 宏观广播一致性
        self._check_macro_broadcast()
        
        # 3.2 状态标记有效性
        self._check_status_flags()
    
    def _check_macro_broadcast(self) -> None:
        """检查宏观数据广播一致性（同一天所有股票宏观字段应相同）"""
        macro_fields = [f for f in self.EXPECTED_MACRO_FIELDS if f in self.df.columns]
        
        if not macro_fields:
            return
        
        # 抽样检查5个日期
        sample_dates = self.df['trade_date'].drop_duplicates().sample(min(5, len(self.df['trade_date'].unique())))
        
        issues = []
        for date in sample_dates:
            day_data = self.df[self.df['trade_date'] == date]
            for field in macro_fields:
                unique_vals = day_data[field].dropna().nunique()
                if unique_vals > 1:
                    issues.append({
                        'date': str(date),
                        'field': field,
                        'unique_count': unique_vals,
                    })
        
        if issues:
            level = CheckLevel.ERROR
            msg = f"宏观广播不一致: {len(issues)} 处"
            passed = False
        else:
            level = CheckLevel.PASS
            msg = "宏观数据广播一致"
            passed = True
        
        self._add_result(CheckResult(
            name='宏观广播一致性',
            dimension='合并逻辑有效性',
            level=level,
            passed=passed,
            message=msg,
            details={'issues': issues, 'checked_dates': len(sample_dates)}
        ))
    
    def _check_status_flags(self) -> None:
        """检查状态标记有效性"""
        issues = []
        
        # 检查 is_st 字段
        if 'is_st' in self.df.columns:
            st_count = (self.df['is_st'] == 1).sum()
            st_pct = st_count / len(self.df) * 100
            if st_pct > 20:
                issues.append(f"ST 占比异常高: {st_pct:.1f}%")
        
        # 检查 is_trading 字段
        if 'is_trading' in self.df.columns:
            trading_count = (self.df['is_trading'] == 1).sum()
            trading_pct = trading_count / len(self.df) * 100
            if trading_pct < 80:
                issues.append(f"交易状态占比异常低: {trading_pct:.1f}%")
        
        if issues:
            level = CheckLevel.WARNING
            msg = "; ".join(issues)
            passed = False
        else:
            level = CheckLevel.PASS
            msg = "状态标记正常"
            passed = True
        
        self._add_result(CheckResult(
            name='状态标记有效性',
            dimension='合并逻辑有效性',
            level=level,
            passed=passed,
            message=msg,
            details={'issues': issues}
        ))
    
    # ========== 维度4: 预处理效果 ==========
    def _check_preprocess_effect(self) -> None:
        """维度4: 预处理效果检查"""
        logger.info("-" * 50)
        logger.info("📋 维度4: 预处理效果检查")
        logger.info("-" * 50)
        
        # 4.1 比例字段范围检查
        self._check_ratio_fields()
        
        # 4.2 估值指标倒数化检查
        self._check_valuation_inverse()
        
        # 4.3 极值去除检查
        self._check_extreme_values()
    
    def _check_ratio_fields(self) -> None:
        """检查比例字段范围 (0-100)"""
        ratio_fields = ['top10_hold_ratio', 'top1_hold_ratio', 'top10_inst_ratio']
        issues = []
        
        for field in ratio_fields:
            if field not in self.df.columns:
                continue
            
            max_val = self.df[field].max()
            if max_val > 100:
                issues.append({
                    'field': field,
                    'max_value': float(max_val),
                })
        
        if issues:
            level = CheckLevel.ERROR
            msg = f"{len(issues)} 个比例字段超过 100%"
            passed = False
        else:
            level = CheckLevel.PASS
            msg = "比例字段范围正常 (0-100)"
            passed = True
        
        self._add_result(CheckResult(
            name='比例字段范围',
            dimension='预处理效果',
            level=level,
            passed=passed,
            message=msg,
            details={'issues': issues}
        ))
    
    def _check_valuation_inverse(self) -> None:
        """检查估值指标倒数化是否正确"""
        inverse_pairs = [('pe_ttm', 'ep'), ('pb', 'bp'), ('ps_ttm', 'sp')]
        
        issues = []
        passed_checks = []
        
        for original, inverse in inverse_pairs:
            # 原始字段应该被删除
            if original in self.df.columns:
                issues.append(f"{original} 未被删除")
            
            # 倒数字段应该存在
            if inverse in self.df.columns:
                # 检查是否有负值（负 PE 转负 EP 是正确的）
                neg_count = (self.df[inverse] < 0).sum()
                null_count = self.df[inverse].isna().sum()
                mean_val = self.df[inverse].mean()
                
                passed_checks.append({
                    'field': inverse,
                    'mean': float(mean_val),
                    'neg_count': int(neg_count),
                    'null_count': int(null_count),
                })
            else:
                issues.append(f"{inverse} 不存在")
        
        if issues:
            level = CheckLevel.ERROR
            msg = f"估值倒数化问题: {issues}"
            passed = False
        else:
            level = CheckLevel.PASS
            msg = "估值指标倒数化正确 (ep/bp/sp)"
            passed = True
        
        self._add_result(CheckResult(
            name='估值指标倒数化',
            dimension='预处理效果',
            level=level,
            passed=passed,
            message=msg,
            details={'issues': issues, 'passed_checks': passed_checks}
        ))
    
    def _check_extreme_values(self) -> None:
        """检查极值是否被处理"""
        # 收益率极值检查
        if 'return_1d' not in self.df.columns:
            return
        
        ret = self.df['return_1d'].dropna()
        
        # 检查是否有超过 ±100% 的收益率（应该被去极值处理）
        extreme_count = ((ret < -1) | (ret > 1)).sum()
        
        if extreme_count > 0:
            level = CheckLevel.WARNING
            msg = f"收益率存在 {extreme_count} 个极端值 (>100%)"
            passed = False
        else:
            level = CheckLevel.PASS
            msg = "收益率极值已处理"
            passed = True
        
        self._add_result(CheckResult(
            name='极值处理检查',
            dimension='预处理效果',
            level=level,
            passed=passed,
            message=msg,
            details={'extreme_count': int(extreme_count)}
        ))
    
    def _print_summary(self) -> None:
        """打印检查汇总"""
        logger.info("=" * 70)
        logger.info("📊 数据质量检查汇总")
        logger.info("=" * 70)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        critical = sum(1 for r in self.results if r.level == CheckLevel.CRITICAL)
        errors = sum(1 for r in self.results if r.level == CheckLevel.ERROR)
        warnings = sum(1 for r in self.results if r.level == CheckLevel.WARNING)
        
        logger.info(f"  总计: {total} 项检查")
        logger.info(f"  通过: {passed} 项 ✅")
        logger.info(f"  CRITICAL: {critical} 项")
        logger.info(f"  ERROR: {errors} 项")
        logger.info(f"  WARNING: {warnings} 项")
        
        overall = "PASS" if critical == 0 and errors == 0 else "FAIL"
        logger.info(f"  总体结果: {overall}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取检查结果摘要"""
        return {
            'total_checks': len(self.results),
            'passed': sum(1 for r in self.results if r.passed),
            'critical': sum(1 for r in self.results if r.level == CheckLevel.CRITICAL),
            'errors': sum(1 for r in self.results if r.level == CheckLevel.ERROR),
            'warnings': sum(1 for r in self.results if r.level == CheckLevel.WARNING),
            'results': [r.to_dict() for r in self.results],
            'column_stats': {k: v.to_dict() for k, v in self.column_stats.items()},
        }
