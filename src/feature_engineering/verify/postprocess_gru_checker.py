"""
GRU 后处理数据质量检查器

验收 train_gru.parquet 的数据质量，确保：
1. 公共基础核验（与 LGB 数据对齐）
2. 缺失值：全表 NaN = 0
3. 标准化：均值 ≈ 0，标准差 ≈ 1
4. 长尾修正：Log1p 后的特征也已 Z-Score
5. 去极值：所有特征 max ≤ 10
6. 无限值：无 Inf
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


class GRUDataChecker:
    """GRU 数据质量检查器"""
    
    # 元数据列
    META_COLS = ['ts_code', 'trade_date']
    
    # 标签列
    LABEL_COLS = [
        'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
        'label_bin_1d', 'label_bin_2d', 'label_bin_3d', 'label_bin_5d', 'label_bin_10d'
    ]
    
    # 类别特征（idx 列）- 不参与 Z-Score，保持整数
    CATEGORY_COLS = [
        'industry_idx', 'sw_l1_idx', 'sw_l2_idx', 'sw_l3_idx',
        'market',  # 已编码为整数
    ]
    
    # 布尔/二值特征 - 不参与 Z-Score（值为 0/1 或 -1/0/1）
    BOOLEAN_COLS = [
        # 基础布尔特征
        'is_trading', 'is_st', 'is_suspended',
        # 涨跌停/交易状态
        'is_limit_up', 'is_limit_down', 'is_limit',
        'is_tradable', 'is_risky',
        # 新股相关
        'is_new', 'is_new_no_limit',
        # 龙虎榜/北交所
        'is_top_list', 'is_bj_stock',
        # 事件相关
        'is_repurchase_ann', 'in_repurchase_window',
        'is_unlock_day', 'in_unlock_window',
        'is_dividend_ann', 'in_dividend_window',
        'has_event', 'has_risk_event',
        # 指数成分
        'is_hs300', 'is_csi500', 'is_csi1000',
    ]
    
    # 日期/字符串列（已在后处理中删除，但保留用于向后兼容）
    EXCLUDE_FROM_NAN_CHECK = [
        'report_date', 'list_date', 'chip_report_date', 'holder_report_date',
        'industry', 'sw_l1_name', 'sw_l2_name', 'sw_l3_name',
    ]
    
    # [已废弃] 原始数值列 - 现在这些列已被 GRU 后处理器 drop 或标准化
    # 保留此列表仅用于参考/向后兼容
    _DEPRECATED_RAW_NUMERIC_COLS = [
        'open', 'high', 'low', 'close', 'pre_close', 'vwap',
        'ma_5', 'ma_10', 'ma_20', 'ma_60', 'ma_120', 'ma_250',
    ]
    
    # Log1p 变换后应该做 Z-Score 的特征（现在全量标准化，仅用于参考）
    LOG1P_ZSCORE_FEATURES = [
        'amount', 'vol', 'total_mv', 'circ_mv',
        'buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount',
        'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount',
    ]
    
    # 关键 Z-Score 特征（现在全量标准化，仅用于向后兼容）
    KEY_ZSCORE_FEATURES = [
        'return_1d', 'turnover', 'rsi_6', 'rsi_12', 'macd',
        'bias_5', 'bias_20', 'volatility_5', 'volatility_20',
    ]
    
    # 极值检查阈值
    MAX_EXTREME_VALUE = 10.0  # 推荐 [-5, 5]，最大允许 10
    
    # Z-Score 标准化阈值
    # 注：对于稀疏特征（如龙虎榜），大部分为 0，标准化后 std 会偏低
    MEAN_TOLERANCE = 0.2  # 均值允许误差
    STD_MIN = 0.2  # 标准差最小值（降低以适应稀疏特征）
    STD_MAX = 1.5  # 标准差最大值
    
    def __init__(
        self,
        gru_path: str,
        lgb_path: Optional[str] = None,
        output_dir: str = "reports",
    ):
        self.gru_path = Path(gru_path)
        self.lgb_path = Path(lgb_path) if lgb_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[CheckResult] = []
        self.column_profiles: Dict[str, ColumnProfile] = {}
        self.gru_df: Optional[pd.DataFrame] = None
        self.lgb_df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> bool:
        """加载数据"""
        logger.info(f"📖 加载 GRU 数据: {self.gru_path}")
        try:
            self.gru_df = pd.read_parquet(self.gru_path)
            logger.info(f"   ✓ GRU: {len(self.gru_df):,} 行, {len(self.gru_df.columns)} 列")
        except Exception as e:
            logger.error(f"   ✗ 加载失败: {e}")
            return False
        
        if self.lgb_path and self.lgb_path.exists():
            logger.info(f"📖 加载 LGB 数据（用于对齐检查）: {self.lgb_path}")
            try:
                self.lgb_df = pd.read_parquet(self.lgb_path)
                logger.info(f"   ✓ LGB: {len(self.lgb_df):,} 行, {len(self.lgb_df.columns)} 列")
            except Exception as e:
                logger.warning(f"   ⚠ LGB 数据加载失败: {e}")
        
        return True
    
    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有检查"""
        logger.info("=" * 60)
        logger.info("🔍 GRU 数据质量检查")
        logger.info("=" * 60)
        
        # 1. 加载数据
        if not self.load_data():
            return {"error": "数据加载失败"}
        
        # 2. 公共基础核验
        self._check_common_baseline()
        
        # 3. GRU 专用核验
        self._check_no_nan()
        self._check_no_inf()
        self._check_zscore_standardization()
        self._check_log1p_zscore()
        self._check_extreme_values()
        self._check_sequence_continuity()
        
        # 4. 生成列档案
        self._generate_column_profiles()
        
        # 5. 汇总结果
        return self._summarize_results()
    
    def _check_common_baseline(self):
        """公共基础核验"""
        logger.info("-" * 40)
        logger.info("📋 公共基础核验")
        logger.info("-" * 40)
        
        df = self.gru_df
        
        # 1. 行数检查（与 LGB 对齐）
        if self.lgb_df is not None:
            gru_rows = len(df)
            lgb_rows = len(self.lgb_df)
            
            # 检查共同时间窗口的行数
            gru_dates = df['trade_date'].unique()
            lgb_dates = self.lgb_df['trade_date'].unique()
            common_dates = set(gru_dates) & set(lgb_dates)
            
            gru_common = df[df['trade_date'].isin(common_dates)]
            lgb_common = self.lgb_df[self.lgb_df['trade_date'].isin(common_dates)]
            
            rows_match = len(gru_common) == len(lgb_common)
            
            self.results.append(CheckResult(
                name="行数对齐（共同时间窗口）",
                dimension="公共基础",
                level=CheckLevel.PASS if rows_match else CheckLevel.ERROR,
                passed=rows_match,
                message=f"GRU={len(gru_common):,}, LGB={len(lgb_common):,}",
                details={
                    'gru_total_rows': gru_rows,
                    'lgb_total_rows': lgb_rows,
                    'common_dates_count': len(common_dates),
                    'gru_common_rows': len(gru_common),
                    'lgb_common_rows': len(lgb_common),
                }
            ))
            logger.info(f"   行数对齐: {'✓' if rows_match else '✗'} GRU={len(gru_common):,}, LGB={len(lgb_common):,}")
        
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
        
        # 3. 排序验证（GRU 需要按 ts_code, trade_date 排序以保证时序连续性）
        df_sorted_check = df.sort_values(['ts_code', 'trade_date'])
        is_sorted = df['ts_code'].equals(df_sorted_check['ts_code']) and \
                    df['trade_date'].equals(df_sorted_check['trade_date'])
        
        self.results.append(CheckResult(
            name="排序验证（ts_code, trade_date）",
            dimension="公共基础",
            level=CheckLevel.PASS if is_sorted else CheckLevel.ERROR,
            passed=is_sorted,
            message=f"按 (ts_code, trade_date) 升序: {is_sorted}",
            details={'is_sorted': is_sorted}
        ))
        logger.info(f"   排序验证: {'✓' if is_sorted else '✗'} (ts_code, trade_date)")
    
    def _check_no_nan(self):
        """检查无 NaN（仅检查数值特征列）"""
        logger.info("-" * 40)
        logger.info("📋 缺失值检查（NaN）")
        logger.info("-" * 40)
        
        df = self.gru_df
        
        # 排除元数据列、标签列、日期/字符串列
        exclude_cols = set(self.META_COLS + self.LABEL_COLS + self.EXCLUDE_FROM_NAN_CHECK)
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # 只检查数值列
        numeric_feature_cols = [
            c for c in feature_cols 
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        
        total_nan = 0
        nan_cols = []
        
        for col in numeric_feature_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_cols.append({'col': col, 'nan_count': nan_count})
                total_nan += nan_count
        
        passed = total_nan == 0
        
        self.results.append(CheckResult(
            name="数值特征列无 NaN",
            dimension="缺失值",
            level=CheckLevel.PASS if passed else CheckLevel.CRITICAL,
            passed=passed,
            message=f"总 NaN 数量: {total_nan:,}, 涉及 {len(nan_cols)} 列（检查 {len(numeric_feature_cols)} 列）",
            details={
                'total_nan': total_nan,
                'nan_cols_count': len(nan_cols),
                'nan_cols_top10': nan_cols[:10] if nan_cols else [],
                'checked_cols': len(numeric_feature_cols),
            }
        ))
        
        status = '✓' if passed else '✗'
        logger.info(f"   数值特征列无 NaN: {status} 总 NaN={total_nan:,} (检查 {len(numeric_feature_cols)} 列)")
        if nan_cols:
            for item in nan_cols[:5]:
                logger.info(f"      - {item['col']}: {item['nan_count']:,}")
    
    def _check_no_inf(self):
        """检查无 Inf"""
        logger.info("-" * 40)
        logger.info("📋 无限值检查（Inf）")
        logger.info("-" * 40)
        
        df = self.gru_df
        
        # 检查所有数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        total_inf = 0
        total_neg_inf = 0
        inf_cols = []
        
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            pos_inf = (df[col] == np.inf).sum()
            neg_inf = (df[col] == -np.inf).sum()
            
            if inf_count > 0:
                inf_cols.append({
                    'col': col, 
                    'inf_count': inf_count,
                    'pos_inf': pos_inf,
                    'neg_inf': neg_inf,
                })
                total_inf += pos_inf
                total_neg_inf += neg_inf
        
        passed = (total_inf + total_neg_inf) == 0
        
        self.results.append(CheckResult(
            name="无 Inf 值",
            dimension="无限值",
            level=CheckLevel.PASS if passed else CheckLevel.CRITICAL,
            passed=passed,
            message=f"+inf={total_inf:,}, -inf={total_neg_inf:,}",
            details={
                'pos_inf_count': total_inf,
                'neg_inf_count': total_neg_inf,
                'inf_cols': inf_cols,
            }
        ))
        
        status = '✓' if passed else '✗'
        logger.info(f"   无 Inf: {status} +inf={total_inf:,}, -inf={total_neg_inf:,}")
    
    def _check_zscore_standardization(self):
        """检查全体数值列的 Z-Score 标准化
        
        [更新] 现在 GRU 后处理已对所有数值列做标准化，
        因此检查全体数值列（排除标签、类别、布尔特征）
        """
        logger.info("-" * 40)
        logger.info("📋 全量 Z-Score 标准化检查")
        logger.info("-" * 40)
        
        df = self.gru_df
        
        # 排除列：元数据、标签、类别特征、布尔特征
        exclude_cols = set(
            self.META_COLS + self.LABEL_COLS + 
            self.CATEGORY_COLS + self.BOOLEAN_COLS +
            self.EXCLUDE_FROM_NAN_CHECK
        )
        
        # 获取所有应检查的数值列
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        
        failed_features = []
        passed_features = []
        skipped_features = []
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                skipped_features.append(col)
                continue
            
            col_mean = col_data.mean()
            col_std = col_data.std()
            
            # 检查均值是否接近 0
            mean_ok = abs(col_mean) < self.MEAN_TOLERANCE
            # 检查标准差是否在合理范围（考虑 Z-Score 后 std 应接近 1）
            std_ok = self.STD_MIN < col_std < self.STD_MAX
            
            passed = mean_ok and std_ok
            
            if passed:
                passed_features.append(col)
            else:
                failed_features.append({
                    'col': col,
                    'mean': float(col_mean),
                    'std': float(col_std),
                    'mean_ok': mean_ok,
                    'std_ok': std_ok,
                })
        
        total_checked = len(passed_features) + len(failed_features)
        pass_rate = len(passed_features) / total_checked * 100 if total_checked > 0 else 0
        
        # 通过率阈值：80% 以上认为通过
        all_passed = pass_rate >= 80
        
        self.results.append(CheckResult(
            name="全量 Z-Score 标准化",
            dimension="标准化",
            level=CheckLevel.PASS if all_passed else CheckLevel.ERROR,
            passed=all_passed,
            message=f"通过 {len(passed_features)}/{total_checked} ({pass_rate:.1f}%)",
            details={
                'passed_count': len(passed_features),
                'failed_count': len(failed_features),
                'skipped_count': len(skipped_features),
                'pass_rate': pass_rate,
                'failed_features_top20': failed_features[:20],
                'mean_tolerance': self.MEAN_TOLERANCE,
                'std_range': [self.STD_MIN, self.STD_MAX],
            }
        ))
        
        status = '✓' if all_passed else '✗'
        logger.info(f"   全量 Z-Score: {status} 通过 {len(passed_features)}/{total_checked} ({pass_rate:.1f}%)")
        
        if failed_features:
            logger.info(f"   未通过列示例（显示前 5 个）：")
            for item in failed_features[:5]:
                logger.info(f"      - {item['col']}: mean={item['mean']:.4f}, std={item['std']:.4f}")
    
    def _check_log1p_zscore(self):
        """检查 Log1p 特征是否做了 Z-Score"""
        logger.info("-" * 40)
        logger.info("📋 Log1p + Z-Score 检查")
        logger.info("-" * 40)
        
        df = self.gru_df
        
        failed_features = []
        passed_features = []
        
        for col in self.LOG1P_ZSCORE_FEATURES:
            if col not in df.columns:
                continue
            
            col_mean = df[col].mean()
            col_std = df[col].std()
            
            # Log1p 后的特征如果只做了 Log 没做 Z-Score，均值会在 10-20 之间
            # 如果做了 Z-Score，均值应该接近 0
            is_only_log = col_mean > 5  # 只做了 Log，没做 Z-Score
            
            # 检查是否正确标准化
            mean_ok = abs(col_mean) < self.MEAN_TOLERANCE
            std_ok = self.STD_MIN < col_std < self.STD_MAX
            
            passed = mean_ok and std_ok and not is_only_log
            
            if passed:
                passed_features.append(col)
            else:
                failed_features.append({
                    'col': col,
                    'mean': col_mean,
                    'std': col_std,
                    'is_only_log': is_only_log,
                })
        
        all_passed = len(failed_features) == 0
        
        self.results.append(CheckResult(
            name="Log1p 特征 Z-Score 标准化",
            dimension="长尾修正",
            level=CheckLevel.PASS if all_passed else CheckLevel.CRITICAL,
            passed=all_passed,
            message=f"通过 {len(passed_features)}/{len(passed_features)+len(failed_features)}",
            details={
                'passed_features': passed_features,
                'failed_features': failed_features,
                'description': 'amount/vol 等 Log1p 特征均值应接近 0，若 >5 说明只做了 Log 没做 Z-Score',
            }
        ))
        
        status = '✓' if all_passed else '✗'
        logger.info(f"   Log1p + Z-Score: {status} 通过 {len(passed_features)}/{len(passed_features)+len(failed_features)}")
        
        for item in failed_features[:5]:
            logger.info(f"      - {item['col']}: mean={item['mean']:.4f} (is_only_log={item['is_only_log']})")
    
    def _check_extreme_values(self):
        """检查去极值 (Clipping)，检查全体标准化特征
        
        [更新] 现在 GRU 后处理已对所有数值列做标准化，
        因此检查全体数值列（排除标签、类别、布尔特征）
        """
        logger.info("-" * 40)
        logger.info("📋 全量去极值检查 (Clipping)")
        logger.info("-" * 40)
        
        df = self.gru_df
        
        # 排除列：元数据、标签、类别特征、布尔特征
        exclude_cols = set(
            self.META_COLS + self.LABEL_COLS + 
            self.CATEGORY_COLS + self.BOOLEAN_COLS +
            self.EXCLUDE_FROM_NAN_CHECK
        )
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        
        extreme_cols = []
        ok_cols = []
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
                
            col_max = col_data.abs().max()
            
            if col_max > self.MAX_EXTREME_VALUE:
                extreme_cols.append({
                    'col': col,
                    'max_abs': float(col_max),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                })
            else:
                ok_cols.append(col)
        
        passed = len(extreme_cols) == 0
        
        self.results.append(CheckResult(
            name=f"去极值 (|x| ≤ {self.MAX_EXTREME_VALUE})",
            dimension="去极值",
            level=CheckLevel.PASS if passed else CheckLevel.WARNING,
            passed=passed,
            message=f"超限列: {len(extreme_cols)}/{len(numeric_cols)}",
            details={
                'threshold': self.MAX_EXTREME_VALUE,
                'extreme_cols_count': len(extreme_cols),
                'extreme_cols_top10': extreme_cols[:10] if extreme_cols else [],
                'ok_cols_count': len(ok_cols),
            }
        ))
        
        status = '✓' if passed else '⚠'
        logger.info(f"   去极值: {status} 超限列 {len(extreme_cols)}/{len(numeric_cols)}")
        
        for item in extreme_cols[:5]:
            logger.info(f"      - {item['col']}: max_abs={item['max_abs']:.2f}")
    
    def _check_sequence_continuity(self):
        """检查时序连续性（GRU 需要）"""
        logger.info("-" * 40)
        logger.info("📋 时序连续性检查")
        logger.info("-" * 40)
        
        df = self.gru_df
        
        # 按股票分组检查时序
        sample_stocks = df['ts_code'].unique()[:100]  # 抽样检查 100 只股票
        
        gaps_found = 0
        total_checked = 0
        gap_examples = []
        
        for ts_code in sample_stocks:
            stock_data = df[df['ts_code'] == ts_code].sort_values('trade_date')
            
            if len(stock_data) < 2:
                continue
            
            total_checked += 1
            dates = pd.to_datetime(stock_data['trade_date'])
            
            # 计算日期差
            date_diffs = dates.diff().dropna()
            
            # 允许最大 10 个交易日的间隔（节假日等）
            max_gap = date_diffs.max().days if len(date_diffs) > 0 else 0
            
            if max_gap > 30:  # 超过 30 天认为有问题
                gaps_found += 1
                if len(gap_examples) < 5:
                    gap_examples.append({
                        'ts_code': ts_code,
                        'max_gap_days': max_gap,
                        'data_points': len(stock_data),
                    })
        
        # 允许少量股票有间隔（停牌等）
        passed = gaps_found <= total_checked * 0.1  # 10% 以内
        
        self.results.append(CheckResult(
            name="时序连续性",
            dimension="序列检查",
            level=CheckLevel.PASS if passed else CheckLevel.WARNING,
            passed=passed,
            message=f"有大间隔的股票: {gaps_found}/{total_checked}",
            details={
                'total_checked': total_checked,
                'gaps_found': gaps_found,
                'gap_examples': gap_examples,
                'max_allowed_gap_days': 30,
            }
        ))
        
        status = '✓' if passed else '⚠'
        logger.info(f"   时序连续性: {status} 有大间隔 {gaps_found}/{total_checked}")
    
    def _generate_column_profiles(self):
        """生成所有列的详细档案"""
        logger.info("-" * 40)
        logger.info("📋 生成列档案")
        logger.info("-" * 40)
        
        df = self.gru_df
        
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
            'data_file': str(self.gru_path),
            'total_rows': len(self.gru_df),
            'total_cols': len(self.gru_df.columns),
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
        json_path = self.output_dir / f"gru_dq_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, ensure_ascii=False, indent=2)
        
        # Markdown 报告
        md_path = self.output_dir / f"gru_dq_report_{timestamp}.md"
        md_content = self._generate_markdown_report(summary)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"   📄 JSON 报告: {json_path}")
        logger.info(f"   📄 Markdown 报告: {md_path}")
        
        return str(json_path), str(md_path)
    
    def _generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """生成 Markdown 格式报告"""
        lines = [
            "# GRU 数据质量检查报告",
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
        
        # 特征列统计（展示关键特征）
        lines.append("### 关键特征分布（Z-Score 后）")
        lines.append("")
        lines.append("| 特征 | NaN% | 零值% | 均值 | 标准差 | 最小值 | 最大值 |")
        lines.append("|------|------|-------|------|--------|--------|--------|")
        
        # 展示关键特征
        key_features = ['amount', 'vol', 'total_mv', 'return_1d', 'turnover', 
                       'rsi_6', 'macd', 'bias_5', 'volatility_5']
        
        for feat in key_features:
            if feat in summary['column_profiles']:
                p = summary['column_profiles'][feat]
                mean = f"{p['mean']:.4f}" if p['mean'] is not None else "N/A"
                std = f"{p['std']:.4f}" if p['std'] is not None else "N/A"
                min_v = f"{p['min_val']:.4f}" if p['min_val'] is not None else "N/A"
                max_v = f"{p['max_val']:.4f}" if p['max_val'] is not None else "N/A"
                lines.append(f"| {p['name']} | {p['null_pct']:.2f}% | {p['zero_pct']:.2f}% | {mean} | {std} | {min_v} | {max_v} |")
        
        lines.append("")
        
        # 标签列统计
        label_profiles = [p for p in summary['column_profiles'].values() if p['category'] == 'label']
        
        if label_profiles:
            lines.append("### 标签列分布")
            lines.append("")
            lines.append("| 标签 | NaN% | 均值 | 标准差 | 最小值 | 最大值 |")
            lines.append("|------|------|------|--------|--------|--------|")
            
            for name in ['ret_5d', 'label_bin_5d']:
                if name in summary['column_profiles']:
                    p = summary['column_profiles'][name]
                    mean = f"{p['mean']:.4f}" if p['mean'] is not None else "N/A"
                    std = f"{p['std']:.4f}" if p['std'] is not None else "N/A"
                    min_v = f"{p['min_val']:.4f}" if p['min_val'] is not None else "N/A"
                    max_v = f"{p['max_val']:.4f}" if p['max_val'] is not None else "N/A"
                    lines.append(f"| {p['name']} | {p['null_pct']:.2f}% | {mean} | {std} | {min_v} | {max_v} |")
            
            lines.append("")
        
        # 数据质量结论
        lines.append("---")
        lines.append("")
        lines.append("## 数据质量结论")
        lines.append("")
        
        pass_rate = summary['summary']['pass_rate']
        critical_failed = any(
            c['level'] == 'CRITICAL' and not c['passed'] 
            for c in summary['checks']
        )
        
        if critical_failed:
            lines.append("❌ **存在严重问题**，需要修复后才能进行 GRU 训练。")
            lines.append("")
            lines.append("### 严重问题列表")
            lines.append("")
            for check in summary['checks']:
                if check['level'] == 'CRITICAL' and not check['passed']:
                    lines.append(f"- **{check['name']}**: {check['message']}")
        elif pass_rate >= 90:
            lines.append("✅ **数据质量良好**，可以进入 GRU 模型训练阶段。")
        elif pass_rate >= 70:
            lines.append("⚠️ **数据质量一般**，建议检查 WARNING 项后再进行训练。")
        else:
            lines.append("❌ **数据质量较差**，需要重新处理数据。")
        
        lines.append("")
        
        return "\n".join(lines)


def run_gru_check(
    gru_path: str = "data/features/structured/train_gru.parquet",
    lgb_path: Optional[str] = "data/features/structured/train_lgb.parquet",
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """运行 GRU 数据质量检查"""
    checker = GRUDataChecker(gru_path, lgb_path, output_dir)
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
    
    result = run_gru_check()
    
    # 输出结果
    pass_rate = result['summary']['summary']['pass_rate']
    print(f"\n通过率: {pass_rate:.1f}%")
    print(f"JSON 报告: {result['json_path']}")
    print(f"Markdown 报告: {result['md_path']}")
    
    # 检查是否有 CRITICAL 失败
    critical_failed = any(
        c['level'] == 'CRITICAL' and not c['passed'] 
        for c in result['summary']['checks']
    )
    
    sys.exit(0 if pass_rate >= 70 and not critical_failed else 1)
