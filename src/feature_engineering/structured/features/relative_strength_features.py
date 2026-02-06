"""
相对强弱和指数成分特征生成器 (Relative Strength & Index Member Generator)

需要参考数据 (Reference Data)：
- benchmark: 基准指数行情
- weights: 指数成分权重

生成的特征：
- 超额收益 (Alpha Return)
- Beta 滚动计算
- 指数成分标记
- 指数权重因子

注意：
- 指数权重需要 Lag 1 天使用，防止未来信息泄露
- trade_date 必须已转换为 datetime64 格式以加速 Merge
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RelativeStrengthGenerator:
    """
    相对强弱特征生成器
    
    计算个股相对于基准指数的超额收益和 Beta。
    需要 ref_data['benchmark'] 参考数据。
    """
    
    def __init__(
        self, 
        config: Any = None, 
        ref_data: Optional[Dict[str, Any]] = None,
        use_gpu: bool = True
    ):
        """
        初始化
        
        Args:
            config: 配置对象
            ref_data: 参考数据字典（需包含 'benchmark'）
            use_gpu: 是否使用 GPU
        """
        self.config = config
        self.ref_data = ref_data or {}
        self.use_gpu = use_gpu
        self.stats: Dict[str, Any] = {'generated_features': []}
        
        self._cudf = None
        if use_gpu:
            try:
                import cudf
                self._cudf = cudf
                self.df_lib = cudf
                logger.info("🚀 RelativeStrengthGenerator: GPU 模式")
            except ImportError:
                self.df_lib = pd
                self.use_gpu = False
        else:
            self.df_lib = pd
    
    def fit_transform(self, df: Any) -> Any:
        """
        生成相对强弱特征
        
        Args:
            df: 输入 DataFrame（需包含 trade_date, pct_chg 或 return_1d）
            
        Returns:
            添加相对强弱特征后的 DataFrame
        """
        logger.info("  📊 相对强弱特征生成...")
        
        if 'benchmark' not in self.ref_data or self.ref_data['benchmark'] is None:
            logger.warning("    ⚠️ 缺少基准数据，跳过相对强弱特征")
            return df
        
        # 获取收益率列
        ret_col = 'return_1d' if 'return_1d' in df.columns else 'pct_chg'
        if ret_col not in df.columns:
            logger.warning(f"    ⚠️ 缺少收益率列 {ret_col}，跳过")
            return df
        
        # 1. 超额收益（相对沪深300）
        df = self._generate_alpha_return(df, ret_col, '000300.SH', 'rs_hs300')
        
        # 2. 超额收益（相对中证500）
        df = self._generate_alpha_return(df, ret_col, '000905.SH', 'rs_csi500')
        
        # 3. 滚动 Beta（可选，计算密集）
        # df = self._generate_rolling_beta(df, ret_col, '000300.SH', 20, 'beta_hs300_20')
        
        logger.info(f"    ✓ 共生成 {len(self.stats['generated_features'])} 个相对强弱特征")
        return df
    
    def _generate_alpha_return(
        self, 
        df: Any, 
        ret_col: str,
        index_code: str,
        out_col: str
    ) -> Any:
        """
        计算超额收益
        
        alpha_return = stock_return - benchmark_return
        
        Args:
            df: 主表
            ret_col: 收益率列名
            index_code: 基准指数代码
            out_col: 输出列名
        """
        benchmark = self.ref_data.get('benchmark')
        if benchmark is None:
            return df
        
        # 筛选基准指数
        if self.use_gpu and self._cudf:
            bench = benchmark[benchmark['ts_code'] == index_code][['trade_date', 'pct_chg']].copy()
        else:
            bench = benchmark[benchmark['ts_code'] == index_code][['trade_date', 'pct_chg']].copy()
        
        if len(bench) == 0:
            logger.warning(f"    ⚠️ 基准 {index_code} 数据为空")
            return df
        
        bench = bench.rename(columns={'pct_chg': f'_bench_{index_code}'})
        
        # 确保 trade_date 类型一致
        if self.use_gpu and self._cudf:
            if str(df['trade_date'].dtype) != 'datetime64[ns]':
                df['trade_date'] = self._cudf.to_datetime(df['trade_date'])
            if str(bench['trade_date'].dtype) != 'datetime64[ns]':
                bench['trade_date'] = self._cudf.to_datetime(bench['trade_date'])
        else:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            bench['trade_date'] = pd.to_datetime(bench['trade_date'])
        
        # Merge 基准收益率
        n_before = len(df)
        df = df.merge(bench, on='trade_date', how='left')
        
        # 验证行数未变
        if len(df) != n_before:
            logger.warning(f"    ⚠️ Merge 后行数变化: {n_before} -> {len(df)}")
        
        # 计算超额收益
        bench_col = f'_bench_{index_code}'
        df[out_col] = df[ret_col] - df[bench_col].fillna(0)
        
        # 删除临时列
        df = df.drop(columns=[bench_col], errors='ignore')
        
        self.stats['generated_features'].append(out_col)
        logger.info(f"    ✓ 超额收益 ({index_code}): {out_col}")
        
        return df
    
    def _generate_rolling_beta(
        self, 
        df: Any,
        ret_col: str,
        index_code: str,
        window: int,
        out_col: str
    ) -> Any:
        """
        计算滚动 Beta
        
        Beta = Cov(stock, market) / Var(market)
        
        注意：计算密集，建议只对关键基准计算。
        """
        # 实现略复杂，暂时跳过
        # 需要先 Merge 基准收益率，然后分组计算滚动协方差/方差
        pass
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats


class IndexMemberGenerator:
    """
    指数成分特征生成器
    
    标记股票是否为指数成分股，并附加权重。
    需要 ref_data['weights'] 参考数据。
    """
    
    def __init__(
        self, 
        config: Any = None, 
        ref_data: Optional[Dict[str, Any]] = None,
        use_gpu: bool = True
    ):
        """
        初始化
        
        Args:
            config: 配置对象
            ref_data: 参考数据字典（需包含 'weights'）
            use_gpu: 是否使用 GPU
        """
        self.config = config
        self.ref_data = ref_data or {}
        self.use_gpu = use_gpu
        self.stats: Dict[str, Any] = {'generated_features': []}
        
        self._cudf = None
        if use_gpu:
            try:
                import cudf
                self._cudf = cudf
                self.df_lib = cudf
                logger.info("🚀 IndexMemberGenerator: GPU 模式")
            except ImportError:
                self.df_lib = pd
                self.use_gpu = False
        else:
            self.df_lib = pd
    
    def fit_transform(self, df: Any) -> Any:
        """
        生成指数成分特征
        
        Args:
            df: 输入 DataFrame（需包含 trade_date, ts_code）
            
        Returns:
            添加指数成分特征后的 DataFrame
        """
        logger.info("  📊 指数成分特征生成...")
        
        if 'weights' not in self.ref_data or self.ref_data['weights'] is None:
            logger.warning("    ⚠️ 缺少指数权重数据，跳过成分股特征")
            return df
        
        weights = self.ref_data['weights']
        
        # 确定列名
        index_col = 'index_code' if 'index_code' in weights.columns else 'ts_code'
        con_col = 'con_code' if 'con_code' in weights.columns else 'stock_code'
        weight_col = 'weight'
        
        # **性能优化**: 只做一次 GPU → CPU 转换
        was_gpu = False
        if self.use_gpu and self._cudf and hasattr(df, 'to_pandas'):
            was_gpu = True
            df_pd = df.to_pandas()
        else:
            df_pd = df.copy() if isinstance(df, pd.DataFrame) else df
        
        # 转换日期并排序（只做一次）
        df_pd['trade_date'] = pd.to_datetime(df_pd['trade_date'])
        df_pd = df_pd.sort_values('trade_date')
        
        # 批量处理所有指数
        indices = [
            ('000300.SH', 'is_hs300', 'weight_hs300'),
            ('000905.SH', 'is_csi500', 'weight_csi500'),
            ('000852.SH', 'is_csi1000', 'weight_csi1000'),
        ]
        
        for index_code, flag_col, weight_out_col in indices:
            df_pd = self._generate_member_flag_fast(
                df_pd, weights,
                index_col, con_col, weight_col,
                index_code, flag_col, weight_out_col
            )
        
        # **性能优化**: 只做一次 CPU → GPU 转换
        if was_gpu and self._cudf:
            df = self._cudf.from_pandas(df_pd)
        else:
            df = df_pd
        
        logger.info(f"    ✓ 共生成 {len(self.stats['generated_features'])} 个指数成分特征")
        return df
    
    def _generate_member_flag_fast(
        self,
        df_pd: pd.DataFrame,
        weights: Any,
        index_col: str,
        con_col: str,
        weight_col: str,
        index_code: str,
        flag_col: str,
        weight_out_col: str
    ) -> pd.DataFrame:
        """
        生成单个指数的成分标记（已优化，输入输出均为 pandas）
        """
        # 筛选目标指数
        if self._cudf and hasattr(weights, 'to_pandas'):
            idx_weights = weights[weights[index_col] == index_code].to_pandas()
        else:
            idx_weights = weights[weights[index_col] == index_code].copy()
        
        if len(idx_weights) == 0:
            df_pd[flag_col] = 0
            df_pd[weight_out_col] = 0.0
            return df_pd
        
        # 准备权重表
        merge_cols = ['trade_date', con_col]
        if weight_col in idx_weights.columns:
            idx_weights = idx_weights[merge_cols + [weight_col]].copy()
        else:
            idx_weights = idx_weights[merge_cols].copy()
            idx_weights[weight_col] = 1.0
        
        idx_weights = idx_weights.rename(columns={
            con_col: 'ts_code',
            weight_col: weight_out_col
        })
        idx_weights[flag_col] = 1
        idx_weights['trade_date'] = pd.to_datetime(idx_weights['trade_date'])
        idx_weights = idx_weights.sort_values('trade_date')
        
        n_before = len(df_pd)
        
        result = pd.merge_asof(
            df_pd,
            idx_weights[['trade_date', 'ts_code', flag_col, weight_out_col]],
            on='trade_date',
            by='ts_code',
            direction='backward'
        )
        
        if len(result) != n_before:
            logger.warning(f"    ⚠️ merge_asof 后行数变化 ({index_code}): {n_before} -> {len(result)}")
        
        result[flag_col] = result[flag_col].fillna(0).astype('int8')
        result[weight_out_col] = result[weight_out_col].fillna(0).astype('float32')
        
        self.stats['generated_features'].extend([flag_col, weight_out_col])
        
        n_members = int((result[flag_col] == 1).sum()) if len(result) > 0 else 0
        logger.info(f"    ✓ {index_code}: {flag_col} ({n_members:,} 条成分记录)")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats
