"""
Train.parquet 数据质量检查器

全面检查特征+标签数据的质量，为后处理(postprocess)提供参考

检查维度：
1. 基础完整性检查
2. 特征数值质量检查
3. 标签逻辑检查
4. 时序稳定性检查
5. 业务逻辑一致性检查
"""

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
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return str(obj)
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
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
    kurtosis: Optional[float] = None
    
    # 极值样本
    extreme_high: Optional[List[float]] = None
    extreme_low: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class TrainDataChecker:
    """
    Train.parquet 数据质量检查器
    
    检查 data/features/structured/train.parquet 的数据质量
    """
    
    # 列分类配置
    META_COLUMNS = ['ts_code', 'trade_date']
    
    LABEL_COLUMNS = [
        'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
        'label_1d', 'label_5d', 'label_10d', 'label_20d',
        'excess_ret_5d', 'excess_ret_10d',
        'rank_ret_5d', 'rank_ret_10d',
        'sharpe_5d', 'sharpe_10d', 'sharpe_20d',
        'label_bin_5d',
    ]
    
    # 技术指标范围约束
    RSI_RANGE = (0, 100)
    BETA_RANGE = (-5, 5)  # 放宽一些
    RATIO_RANGE = (0, 1)  # 比例类
    
    # Rolling 特征预热期
    ROLLING_WARMUP = {
        'ma_5': 5, 'ma_10': 10, 'ma_20': 20, 'ma_60': 60, 
        'ma_120': 120, 'ma_250': 250,
        'volatility_5': 5, 'volatility_10': 10, 'volatility_20': 20, 'volatility_60': 60,
        'rsi_6': 6, 'rsi_12': 12, 'rsi_24': 24,
        'volume_ratio_5': 5, 'volume_ratio_10': 10, 'volume_ratio_20': 20,
    }
    
    def __init__(
        self,
        train_path: str = "data/features/structured/train.parquet",
        merger_path: str = "data/features/temp/merger_preprocess.parquet",
        output_dir: str = "reports",
    ):
        self.train_path = Path(train_path)
        self.merger_path = Path(merger_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.df: Optional[pd.DataFrame] = None
        self.merger_df: Optional[pd.DataFrame] = None
        
        self.results: List[CheckResult] = []
        self.column_profiles: Dict[str, ColumnProfile] = {}
        self.summary: Dict[str, Any] = {}
        
    def load_data(self) -> None:
        """加载数据"""
        logger.info(f"📂 加载数据: {self.train_path}")
        
        if not self.train_path.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {self.train_path}")
        
        self.df = pd.read_parquet(self.train_path)
        logger.info(f"  ✓ train.parquet: {self.df.shape[0]:,} 行 × {self.df.shape[1]} 列")
        
        if self.merger_path.exists():
            self.merger_df = pd.read_parquet(self.merger_path)
            logger.info(f"  ✓ merger_preprocess.parquet: {self.merger_df.shape[0]:,} 行 × {self.merger_df.shape[1]} 列")
    
    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有检查"""
        logger.info("=" * 70)
        logger.info("🔍 开始 Train.parquet 数据质量检查")
        logger.info("=" * 70)
        
        self.load_data()
        
        # 1. 基础完整性检查
        self._check_basic_integrity()
        
        # 2. 特征数值质量检查
        self._check_feature_quality()
        
        # 3. 标签逻辑检查
        self._check_label_logic()
        
        # 4. 时序稳定性检查
        self._check_temporal_stability()
        
        # 5. 业务逻辑一致性检查
        self._check_business_logic()
        
        # 生成列档案
        self._generate_column_profiles()
        
        # 汇总
        self._generate_summary()
        
        return self.summary
    
    # ==================== 1. 基础完整性检查 ====================
    
    def _check_basic_integrity(self) -> None:
        """基础完整性检查"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 维度 1: 基础完整性检查")
        logger.info("=" * 60)
        
        # 1.1 行数核验
        self._check_row_count()
        
        # 1.2 列数核验
        self._check_column_count()
        
        # 1.3 主键唯一性
        self._check_primary_key()
        
    def _check_row_count(self) -> None:
        """行数核验"""
        train_rows = len(self.df)
        
        if self.merger_df is not None:
            merger_rows = len(self.merger_df)
            row_diff = train_rows - merger_rows
            row_diff_pct = abs(row_diff) / merger_rows * 100 if merger_rows > 0 else 0
            
            # 允许一定的行数差异（预热期剪裁）
            if row_diff_pct > 10:
                level = CheckLevel.ERROR
                passed = False
            elif row_diff_pct > 5:
                level = CheckLevel.WARNING
                passed = True
            else:
                level = CheckLevel.PASS
                passed = True
            
            self.results.append(CheckResult(
                name="行数核验",
                dimension="基础完整性",
                level=level,
                passed=passed,
                message=f"train: {train_rows:,} 行, merger: {merger_rows:,} 行, 差异: {row_diff:+,} ({row_diff_pct:.2f}%)",
                details={
                    'train_rows': train_rows,
                    'merger_rows': merger_rows,
                    'row_diff': row_diff,
                    'row_diff_pct': row_diff_pct,
                }
            ))
        else:
            self.results.append(CheckResult(
                name="行数核验",
                dimension="基础完整性",
                level=CheckLevel.PASS,
                passed=True,
                message=f"train: {train_rows:,} 行 (无 merger 文件对比)",
                details={'train_rows': train_rows}
            ))
    
    def _check_column_count(self) -> None:
        """列数核验"""
        all_cols = set(self.df.columns)
        meta_cols = set(self.META_COLUMNS) & all_cols
        label_cols = set(self.LABEL_COLUMNS) & all_cols
        feature_cols = all_cols - meta_cols - label_cols
        
        # 预期特征（列举关键特征）
        expected_features = [
            # 技术指标
            'ma_5', 'ma_20', 'rsi_6', 'rsi_12', 'macd', 'volatility_20',
            # 基本面
            'ep', 'bp', 'sp', 'roe', 'roa', 'log_total_mv',
            # 资金流
            'mf_main_intensity', 'mf_elg_intensity', 'mf_north_net',
            # 筹码
            'chip_concentration', 'chip_stability_score',
            # 相对强弱
            'rs_hs300', 'rs_csi500',
        ]
        
        missing_features = [f for f in expected_features if f not in feature_cols]
        
        if missing_features:
            level = CheckLevel.WARNING
            passed = True
        else:
            level = CheckLevel.PASS
            passed = True
        
        self.results.append(CheckResult(
            name="列数核验",
            dimension="基础完整性",
            level=level,
            passed=passed,
            message=f"Meta: {len(meta_cols)}, Feature: {len(feature_cols)}, Label: {len(label_cols)}, 总计: {len(all_cols)}",
            details={
                'meta_count': len(meta_cols),
                'feature_count': len(feature_cols),
                'label_count': len(label_cols),
                'total_count': len(all_cols),
                'missing_features': missing_features,
                'meta_columns': list(meta_cols),
                'label_columns': list(label_cols),
            }
        ))
    
    def _check_primary_key(self) -> None:
        """主键唯一性检查"""
        pk = self.df.groupby(['trade_date', 'ts_code']).size()
        dup_count = (pk > 1).sum()
        
        if dup_count > 0:
            dup_samples = pk[pk > 1].head(10).to_dict()
            level = CheckLevel.CRITICAL
            passed = False
        else:
            dup_samples = {}
            level = CheckLevel.PASS
            passed = True
        
        self.results.append(CheckResult(
            name="主键唯一性",
            dimension="基础完整性",
            level=level,
            passed=passed,
            message=f"重复主键数: {dup_count}",
            details={'duplicate_count': dup_count, 'duplicate_samples': dup_samples}
        ))
    
    # ==================== 2. 特征数值质量检查 ====================
    
    def _check_feature_quality(self) -> None:
        """特征数值质量检查"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 维度 2: 特征数值质量检查")
        logger.info("=" * 60)
        
        # 2.1 无限值检查
        self._check_infinite_values()
        
        # 2.2 缺失值模式分析
        self._check_missing_patterns()
        
        # 2.3 异常值检查
        self._check_outliers()
    
    def _check_infinite_values(self) -> None:
        """无限值检查 [最重要]"""
        logger.info("📌 检查无限值 (inf / -inf)")
        
        feature_cols = self._get_feature_columns()
        inf_stats = {}
        total_inf = 0
        total_neg_inf = 0
        
        for col in feature_cols:
            # 兼容 pandas nullable dtype (Float32Dtype 等)
            dtype = self.df[col].dtype
            is_numeric = pd.api.types.is_numeric_dtype(dtype)
            if not is_numeric:
                continue
            
            try:
                # 转换为 numpy 数组处理 inf
                col_values = self.df[col].to_numpy(dtype=float, na_value=np.nan)
                pos_inf = np.sum(col_values == np.inf)
                neg_inf = np.sum(col_values == -np.inf)
                inf_count = pos_inf + neg_inf
            except Exception:
                inf_count = 0
                pos_inf = 0
                neg_inf = 0
            
            if inf_count > 0:
                inf_stats[col] = {
                    'pos_inf': int(pos_inf),
                    'neg_inf': int(neg_inf),
                    'total': int(inf_count),
                }
                total_inf += pos_inf
                total_neg_inf += neg_inf
        
        if total_inf + total_neg_inf > 0:
            level = CheckLevel.ERROR
            passed = False
        else:
            level = CheckLevel.PASS
            passed = True
        
        self.results.append(CheckResult(
            name="无限值检查",
            dimension="特征数值质量",
            level=level,
            passed=passed,
            message=f"+inf: {total_inf:,}, -inf: {total_neg_inf:,}, 涉及列数: {len(inf_stats)}",
            details={
                'total_pos_inf': int(total_inf),
                'total_neg_inf': int(total_neg_inf),
                'affected_columns': inf_stats,
            }
        ))
    
    def _check_missing_patterns(self) -> None:
        """缺失值模式分析"""
        logger.info("📌 分析缺失值模式")
        
        feature_cols = self._get_feature_columns()
        
        # 全 NaN 列
        full_nan_cols = []
        high_nan_cols = []  # >50% NaN
        rolling_warmup_nan = {}  # Rolling 预热期 NaN
        
        for col in feature_cols:
            null_pct = self.df[col].isna().mean() * 100
            
            if null_pct >= 99.9:
                full_nan_cols.append(col)
            elif null_pct >= 50:
                high_nan_cols.append((col, null_pct))
            
            # 检查 Rolling 特征预热期
            if col in self.ROLLING_WARMUP:
                warmup = self.ROLLING_WARMUP[col]
                # 按股票分组，检查前 N 天是否为 NaN
                sample_stock = self.df.groupby('ts_code').head(warmup + 5)
                warmup_nan_rate = sample_stock[col].isna().mean() * 100
                rolling_warmup_nan[col] = {'warmup': warmup, 'warmup_nan_pct': warmup_nan_rate}
        
        if full_nan_cols:
            level = CheckLevel.ERROR
            passed = False
        elif high_nan_cols:
            level = CheckLevel.WARNING
            passed = True
        else:
            level = CheckLevel.PASS
            passed = True
        
        self.results.append(CheckResult(
            name="缺失值模式",
            dimension="特征数值质量",
            level=level,
            passed=passed,
            message=f"全NaN列: {len(full_nan_cols)}, 高NaN列(>50%): {len(high_nan_cols)}",
            details={
                'full_nan_columns': full_nan_cols,
                'high_nan_columns': high_nan_cols[:20],
                'rolling_warmup_nan': rolling_warmup_nan,
            }
        ))
    
    def _check_outliers(self) -> None:
        """异常值/极值检查"""
        logger.info("📌 检查异常值/极值")
        
        feature_cols = self._get_feature_columns()
        extreme_cols = []  # span > 1e6 且非金额
        high_skew_cols = []  # |skew| > 10
        high_kurtosis_cols = []  # kurtosis > 100
        
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(self.df[col].dtype):
                continue
            
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # 替换 inf
            col_data = col_data.replace([np.inf, -np.inf], np.nan).dropna()
            if len(col_data) == 0:
                continue
            
            min_val = col_data.min()
            max_val = col_data.max()
            span = max_val - min_val
            
            # 金额类字段跳过 span 检查
            amount_keywords = ['amount', 'mv', 'rzye', 'rqye', 'vol']
            is_amount = any(kw in col.lower() for kw in amount_keywords)
            
            if span > 1e6 and not is_amount:
                extreme_cols.append({
                    'column': col,
                    'min': float(min_val),
                    'max': float(max_val),
                    'span': float(span),
                })
            
            # 偏度和峰度
            if len(col_data) > 100:
                skew = col_data.skew()
                kurt = col_data.kurtosis()
                
                if abs(skew) > 10:
                    high_skew_cols.append({'column': col, 'skew': float(skew)})
                if kurt > 100:
                    high_kurtosis_cols.append({'column': col, 'kurtosis': float(kurt)})
        
        if extreme_cols:
            level = CheckLevel.WARNING
            passed = True
        else:
            level = CheckLevel.PASS
            passed = True
        
        self.results.append(CheckResult(
            name="异常值检查",
            dimension="特征数值质量",
            level=level,
            passed=passed,
            message=f"极端跨度列: {len(extreme_cols)}, 高偏度: {len(high_skew_cols)}, 高峰度: {len(high_kurtosis_cols)}",
            details={
                'extreme_span_columns': extreme_cols[:20],
                'high_skew_columns': high_skew_cols[:20],
                'high_kurtosis_columns': high_kurtosis_cols[:20],
            }
        ))
    
    # ==================== 3. 标签逻辑检查 ====================
    
    def _check_label_logic(self) -> None:
        """标签逻辑检查"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 维度 3: 标签逻辑检查")
        logger.info("=" * 60)
        
        # 3.1 回归标签 NaN 位置
        self._check_regression_label_nan()
        
        # 3.2 分类标签分布
        self._check_classification_label_dist()
        
        # 3.3 特征-标签相关性（检测数据穿越）
        self._check_feature_label_correlation()
    
    def _check_regression_label_nan(self) -> None:
        """回归标签 NaN 位置检查"""
        logger.info("📌 检查回归标签 NaN 位置")
        
        reg_labels = ['ret_5d', 'ret_10d', 'ret_20d']
        results = {}
        
        for label in reg_labels:
            if label not in self.df.columns:
                continue
            
            horizon = int(label.split('_')[-1].replace('d', ''))
            
            # 检查 NaN 是否集中在时间序列末尾
            nan_mask = self.df[label].isna()
            nan_count = nan_mask.sum()
            nan_pct = nan_count / len(self.df) * 100
            
            # 按股票检查末尾 NaN
            tail_nan_correct = 0
            mid_nan_count = 0
            
            for ts_code, group in self.df.groupby('ts_code'):
                group = group.sort_values('trade_date')
                nan_positions = group[nan_mask.loc[group.index]].index
                
                if len(nan_positions) > 0:
                    # 检查是否都在末尾
                    tail_positions = group.tail(horizon).index
                    tail_nan = len(set(nan_positions) & set(tail_positions))
                    mid_nan = len(nan_positions) - tail_nan
                    
                    tail_nan_correct += tail_nan
                    mid_nan_count += mid_nan
            
            results[label] = {
                'nan_count': int(nan_count),
                'nan_pct': nan_pct,
                'tail_nan': int(tail_nan_correct),
                'mid_nan': int(mid_nan_count),
            }
        
        # 评估
        total_mid_nan = sum(r.get('mid_nan', 0) for r in results.values())
        
        if total_mid_nan > len(self.df) * 0.01:  # >1% 中间 NaN
            level = CheckLevel.WARNING
            passed = True
        else:
            level = CheckLevel.PASS
            passed = True
        
        self.results.append(CheckResult(
            name="回归标签NaN位置",
            dimension="标签逻辑",
            level=level,
            passed=passed,
            message=f"中间NaN总数: {total_mid_nan:,}",
            details={'label_nan_stats': results}
        ))
    
    def _check_classification_label_dist(self) -> None:
        """分类标签分布检查"""
        logger.info("📌 检查分类标签分布")
        
        cls_label = 'label_bin_5d'
        if cls_label not in self.df.columns:
            self.results.append(CheckResult(
                name="分类标签分布",
                dimension="标签逻辑",
                level=CheckLevel.WARNING,
                passed=True,
                message=f"{cls_label} 不存在",
                details={}
            ))
            return
        
        # 统计分布
        value_counts = self.df[cls_label].value_counts(normalize=True) * 100
        
        # 检查是否存在极端不平衡
        max_pct = value_counts.max()
        min_pct = value_counts.min()
        
        if max_pct > 90:
            level = CheckLevel.ERROR
            passed = False
        elif max_pct > 70:
            level = CheckLevel.WARNING
            passed = True
        else:
            level = CheckLevel.PASS
            passed = True
        
        self.results.append(CheckResult(
            name="分类标签分布",
            dimension="标签逻辑",
            level=level,
            passed=passed,
            message=f"分布: {value_counts.to_dict()}",
            details={
                'distribution': value_counts.to_dict(),
                'max_class_pct': float(max_pct),
                'min_class_pct': float(min_pct),
            }
        ))
    
    def _check_feature_label_correlation(self) -> None:
        """特征-标签相关性检查（检测数据穿越）"""
        logger.info("📌 检查特征-标签相关性 (检测数据穿越)")
        
        label = 'ret_5d'
        if label not in self.df.columns:
            return
        
        feature_cols = self._get_feature_columns()
        suspicious_features = []
        
        # 排除自身和相关标签
        exclude_patterns = ['ret_', 'label_', 'excess_', 'rank_', 'sharpe_']
        
        for col in feature_cols:
            if any(col.startswith(p) for p in exclude_patterns):
                continue
            
            if not pd.api.types.is_numeric_dtype(self.df[col].dtype):
                continue
            
            # 计算相关系数
            try:
                valid_mask = self.df[col].notna() & self.df[label].notna()
                if valid_mask.sum() < 1000:
                    continue
                
                corr = self.df.loc[valid_mask, col].corr(self.df.loc[valid_mask, label])
                
                if abs(corr) > 0.5:  # 相关性 > 0.5 需要关注
                    suspicious_features.append({
                        'feature': col,
                        'correlation': float(corr),
                        'abs_corr': abs(float(corr)),
                    })
            except Exception:
                continue
        
        # 排序
        suspicious_features.sort(key=lambda x: x['abs_corr'], reverse=True)
        
        # 检测数据穿越（相关性 > 0.9 极大概率穿越）
        lookahead_suspects = [f for f in suspicious_features if f['abs_corr'] > 0.9]
        
        if lookahead_suspects:
            level = CheckLevel.CRITICAL
            passed = False
        elif len(suspicious_features) > 10:
            level = CheckLevel.WARNING
            passed = True
        else:
            level = CheckLevel.PASS
            passed = True
        
        self.results.append(CheckResult(
            name="特征-标签相关性",
            dimension="标签逻辑",
            level=level,
            passed=passed,
            message=f"高相关特征: {len(suspicious_features)}, 疑似穿越: {len(lookahead_suspects)}",
            details={
                'high_correlation_features': suspicious_features[:30],
                'lookahead_suspects': lookahead_suspects,
            }
        ))
    
    # ==================== 4. 时序稳定性检查 ====================
    
    def _check_temporal_stability(self) -> None:
        """时序稳定性检查"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 维度 4: 时序稳定性检查")
        logger.info("=" * 60)
        
        # 4.1 日均值趋势
        self._check_daily_mean_trend()
        
        # 4.2 覆盖度趋势
        self._check_coverage_trend()
    
    def _check_daily_mean_trend(self) -> None:
        """日均值趋势检查"""
        logger.info("📌 检查关键特征日均值趋势")
        
        key_features = ['volatility_20', 'rsi_12', 'turnover', 'mf_main_intensity']
        daily_stats = {}
        
        for feat in key_features:
            if feat not in self.df.columns:
                continue
            
            daily_mean = self.df.groupby('trade_date')[feat].mean()
            
            # 计算变异系数 (CV)
            cv = daily_mean.std() / daily_mean.mean() * 100 if daily_mean.mean() != 0 else 0
            
            # 检查是否有断崖式变化
            daily_diff = daily_mean.diff().abs()
            max_jump = daily_diff.max()
            avg_level = daily_mean.mean()
            jump_ratio = max_jump / avg_level if avg_level != 0 else 0
            
            daily_stats[feat] = {
                'mean': float(daily_mean.mean()),
                'std': float(daily_mean.std()),
                'cv': float(cv),
                'max_jump_ratio': float(jump_ratio),
            }
        
        # 评估
        high_cv_features = [f for f, s in daily_stats.items() if s['cv'] > 100]
        
        if high_cv_features:
            level = CheckLevel.WARNING
            passed = True
        else:
            level = CheckLevel.PASS
            passed = True
        
        self.results.append(CheckResult(
            name="日均值趋势",
            dimension="时序稳定性",
            level=level,
            passed=passed,
            message=f"高变异系数特征: {len(high_cv_features)}",
            details={'daily_stats': daily_stats, 'high_cv_features': high_cv_features}
        ))
    
    def _check_coverage_trend(self) -> None:
        """覆盖度趋势检查"""
        logger.info("📌 检查每日有效股票数量趋势")
        
        daily_coverage = self.df.groupby('trade_date')['ts_code'].nunique()
        
        # 计算年度平均
        self.df['year'] = pd.to_datetime(self.df['trade_date']).dt.year
        yearly_avg = self.df.groupby('year')['ts_code'].apply(lambda x: x.nunique())
        
        # 检查是否有断崖下跌
        daily_diff = daily_coverage.diff()
        max_drop = daily_diff.min()
        max_drop_date = daily_diff.idxmin() if daily_diff.min() < 0 else None
        
        # 检查 2019（预热期）vs 2025（最新）
        start_year_avg = yearly_avg.get(2019, 0)
        end_year_avg = yearly_avg.get(2025, yearly_avg.iloc[-1] if len(yearly_avg) > 0 else 0)
        
        growth_pct = (end_year_avg - start_year_avg) / start_year_avg * 100 if start_year_avg > 0 else 0
        
        if max_drop < -500:  # 单日下跌超过 500 只股票
            level = CheckLevel.WARNING
            passed = True
        else:
            level = CheckLevel.PASS
            passed = True
        
        self.results.append(CheckResult(
            name="覆盖度趋势",
            dimension="时序稳定性",
            level=level,
            passed=passed,
            message=f"日均股票数: {daily_coverage.mean():.0f}, 2019→2025 增长: {growth_pct:.1f}%",
            details={
                'daily_mean': float(daily_coverage.mean()),
                'daily_min': int(daily_coverage.min()),
                'daily_max': int(daily_coverage.max()),
                'yearly_avg': yearly_avg.to_dict(),
                'max_daily_drop': int(max_drop) if max_drop else 0,
                'max_drop_date': str(max_drop_date) if max_drop_date else None,
                'growth_pct': float(growth_pct),
            }
        ))
        
        # 清理临时列
        self.df.drop('year', axis=1, inplace=True)
    
    # ==================== 5. 业务逻辑一致性检查 ====================
    
    def _check_business_logic(self) -> None:
        """业务逻辑一致性检查"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 维度 5: 业务逻辑一致性检查")
        logger.info("=" * 60)
        
        # 5.1 技术指标范围
        self._check_indicator_ranges()
        
        # 5.2 宏观特征填充
        self._check_macro_fill()
    
    def _check_indicator_ranges(self) -> None:
        """技术指标范围检查"""
        logger.info("📌 检查技术指标范围")
        
        violations = []
        
        # RSI
        for rsi_col in ['rsi_6', 'rsi_12', 'rsi_24']:
            if rsi_col in self.df.columns:
                out_of_range = ((self.df[rsi_col] < 0) | (self.df[rsi_col] > 100)).sum()
                if out_of_range > 0:
                    violations.append({
                        'indicator': rsi_col,
                        'expected_range': '[0, 100]',
                        'violations': int(out_of_range),
                    })
        
        # 比例类
        ratio_cols = ['chip_top10_ratio', 'chip_top1_dominance', 'chip_inst_ratio', 
                      'mf_retail_buy_ratio', 'mf_retail_sell_ratio']
        for col in ratio_cols:
            if col in self.df.columns:
                out_of_range = ((self.df[col] < 0) | (self.df[col] > 1)).sum()
                if out_of_range > len(self.df) * 0.01:  # >1%
                    violations.append({
                        'indicator': col,
                        'expected_range': '[0, 1]',
                        'violations': int(out_of_range),
                    })
        
        # MACD（不应有天文数字）
        if 'macd' in self.df.columns:
            extreme_macd = (self.df['macd'].abs() > 1000).sum()
            if extreme_macd > 0:
                violations.append({
                    'indicator': 'macd',
                    'expected_range': '[-1000, 1000]',
                    'violations': int(extreme_macd),
                })
        
        if violations:
            level = CheckLevel.WARNING
            passed = True
        else:
            level = CheckLevel.PASS
            passed = True
        
        self.results.append(CheckResult(
            name="技术指标范围",
            dimension="业务逻辑",
            level=level,
            passed=passed,
            message=f"违规指标: {len(violations)}",
            details={'violations': violations}
        ))
    
    def _check_macro_fill(self) -> None:
        """宏观特征填充检查"""
        logger.info("📌 检查宏观特征填充")
        
        macro_cols = ['gdp_yoy', 'cpi_yoy', 'pmi', 'm2_yoy', 'lpr_1y']
        
        # 检查 2019 预热期
        df_2019 = self.df[pd.to_datetime(self.df['trade_date']).dt.year == 2019]
        df_2021 = self.df[pd.to_datetime(self.df['trade_date']).dt.year == 2021]
        
        fill_stats = {}
        for col in macro_cols:
            if col not in self.df.columns:
                continue
            
            # 2019 年填充率
            if len(df_2019) > 0:
                null_2019 = df_2019[col].isna().mean() * 100
                zero_2019 = (df_2019[col] == 0).mean() * 100
            else:
                null_2019 = 0
                zero_2019 = 0
            
            # 2021 年填充率
            if len(df_2021) > 0:
                null_2021 = df_2021[col].isna().mean() * 100
                zero_2021 = (df_2021[col] == 0).mean() * 100
            else:
                null_2021 = 0
                zero_2021 = 0
            
            fill_stats[col] = {
                'null_2019': null_2019,
                'zero_2019': zero_2019,
                'null_2021': null_2021,
                'zero_2021': zero_2021,
            }
        
        # 评估：2021 年作为正式训练期，应该正常
        issues_2021 = [c for c, s in fill_stats.items() if s['null_2021'] > 10 or s['zero_2021'] > 90]
        
        if issues_2021:
            level = CheckLevel.WARNING
            passed = True
        else:
            level = CheckLevel.PASS
            passed = True
        
        self.results.append(CheckResult(
            name="宏观特征填充",
            dimension="业务逻辑",
            level=level,
            passed=passed,
            message=f"2021年问题宏观特征: {len(issues_2021)}",
            details={'fill_stats': fill_stats, 'issues_2021': issues_2021}
        ))
    
    # ==================== 列档案生成 ====================
    
    def _generate_column_profiles(self) -> None:
        """生成每列详细档案"""
        logger.info("\n📊 生成列档案...")
        
        for col in self.df.columns:
            category = self._classify_column(col)
            
            col_data = self.df[col]
            count = len(col_data)
            null_count = col_data.isna().sum()
            null_pct = null_count / count * 100
            
            # 数值列特殊处理
            if pd.api.types.is_numeric_dtype(col_data.dtype):
                valid_data = col_data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
                
                zero_count = (col_data == 0).sum()
                zero_pct = zero_count / count * 100
                
                # 处理 inf
                try:
                    col_values = col_data.to_numpy(dtype=float, na_value=np.nan)
                    inf_count = int(np.sum(col_values == np.inf))
                    neg_inf_count = int(np.sum(col_values == -np.inf))
                except Exception:
                    inf_count = 0
                    neg_inf_count = 0
                
                unique_count = col_data.nunique()
                
                if len(valid_data) > 0:
                    profile = ColumnProfile(
                        name=col,
                        dtype=str(col_data.dtype),
                        category=category,
                        count=count,
                        null_count=int(null_count),
                        null_pct=null_pct,
                        zero_count=int(zero_count),
                        zero_pct=zero_pct,
                        inf_count=int(inf_count),
                        neg_inf_count=int(neg_inf_count),
                        unique_count=int(unique_count),
                        mean=float(valid_data.mean()),
                        std=float(valid_data.std()),
                        min_val=float(valid_data.min()),
                        max_val=float(valid_data.max()),
                        q1=float(valid_data.quantile(0.25)),
                        median=float(valid_data.median()),
                        q3=float(valid_data.quantile(0.75)),
                        skew=float(valid_data.skew()) if len(valid_data) > 10 else None,
                        kurtosis=float(valid_data.kurtosis()) if len(valid_data) > 10 else None,
                        extreme_high=valid_data.nlargest(5).tolist(),
                        extreme_low=valid_data.nsmallest(5).tolist(),
                    )
                else:
                    profile = ColumnProfile(
                        name=col, dtype=str(col_data.dtype), category=category,
                        count=count, null_count=int(null_count), null_pct=null_pct,
                        zero_count=int(zero_count), zero_pct=zero_pct,
                        inf_count=int(inf_count), neg_inf_count=int(neg_inf_count),
                        unique_count=int(unique_count),
                    )
            else:
                # 非数值列
                unique_count = col_data.nunique()
                profile = ColumnProfile(
                    name=col, dtype=str(col_data.dtype), category=category,
                    count=count, null_count=int(null_count), null_pct=null_pct,
                    zero_count=0, zero_pct=0, inf_count=0, neg_inf_count=0,
                    unique_count=int(unique_count),
                )
            
            self.column_profiles[col] = profile
    
    def _generate_summary(self) -> None:
        """生成汇总"""
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        level_counts = {}
        for r in self.results:
            level_counts[r.level.value] = level_counts.get(r.level.value, 0) + 1
        
        self.summary = {
            'file': str(self.train_path),
            'timestamp': datetime.now().isoformat(),
            'shape': {'rows': len(self.df), 'cols': len(self.df.columns)},
            'check_results': {
                'total': total_count,
                'passed': passed_count,
                'failed': total_count - passed_count,
                'by_level': level_counts,
            },
            'dimensions': {
                '基础完整性': [r.to_dict() for r in self.results if r.dimension == '基础完整性'],
                '特征数值质量': [r.to_dict() for r in self.results if r.dimension == '特征数值质量'],
                '标签逻辑': [r.to_dict() for r in self.results if r.dimension == '标签逻辑'],
                '时序稳定性': [r.to_dict() for r in self.results if r.dimension == '时序稳定性'],
                '业务逻辑': [r.to_dict() for r in self.results if r.dimension == '业务逻辑'],
            },
        }
    
    # ==================== 工具方法 ====================
    
    def _get_feature_columns(self) -> List[str]:
        """获取特征列"""
        all_cols = set(self.df.columns)
        meta_cols = set(self.META_COLUMNS)
        label_cols = set(self.LABEL_COLUMNS)
        return list(all_cols - meta_cols - label_cols)
    
    def _classify_column(self, col: str) -> str:
        """分类列"""
        if col in self.META_COLUMNS:
            return 'meta'
        elif col in self.LABEL_COLUMNS:
            return 'label'
        else:
            return 'feature'
    
    def get_results(self) -> List[CheckResult]:
        """获取检查结果"""
        return self.results
    
    def get_column_profiles(self) -> Dict[str, ColumnProfile]:
        """获取列档案"""
        return self.column_profiles
    
    def get_summary(self) -> Dict[str, Any]:
        """获取汇总"""
        return self.summary
