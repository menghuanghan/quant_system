"""
参考数据加载器 (Reference Data Loader)

加载特征计算时需要的参考数据（Lookup Tables）：
- index_daily: 核心指数日线行情（用于计算 Beta/RS）
- index_weight: 指数成分权重（用于标记成分股）
- etf_daily: 行业ETF行情（可选，用于行业相对强弱）

这些数据不直接 Merge 进大宽表，而是作为计算时的查找表。

使用 GPU 加速 (cuDF) 或 CPU (pandas)。
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd

logger = logging.getLogger(__name__)


class ReferenceDataLoader:
    """
    参考数据加载器
    
    加载并缓存特征生成所需的参考数据。
    """
    
    # 核心指数代码
    CORE_INDEXES = ['000300.SH', '000905.SH', '000852.SH']  # 沪深300, 中证500, 中证1000
    
    def __init__(
        self, 
        config: Any,
        use_gpu: bool = True,
        cache_enabled: bool = True
    ):
        """
        初始化加载器
        
        Args:
            config: 数据配置对象（需包含路径）
            use_gpu: 是否使用 GPU
            cache_enabled: 是否缓存数据
        """
        self.config = config
        self.use_gpu = use_gpu
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Any] = {}
        
        # 初始化 DataFrame 库
        self._cudf = None
        if use_gpu:
            try:
                import cudf
                self._cudf = cudf
                self.df_lib = cudf
                logger.info("🚀 ReferenceDataLoader: GPU 加速已启用")
            except ImportError:
                self.df_lib = pd
                self.use_gpu = False
                logger.warning("⚠️ cuDF 不可用，回退到 pandas")
        else:
            self.df_lib = pd
    
    @property
    def index_daily_path(self) -> Path:
        """指数日线数据路径"""
        base = getattr(self.config, 'raw_data_dir', Path('data/raw/structured'))
        return base / 'market_data' / 'index_daily'
    
    @property
    def index_weight_path(self) -> Path:
        """指数权重数据路径"""
        base = getattr(self.config, 'raw_data_dir', Path('data/raw/structured'))
        return base / 'index_benchmark' / 'index_weight'
    
    @property
    def etf_daily_path(self) -> Path:
        """ETF日线数据路径"""
        base = getattr(self.config, 'raw_data_dir', Path('data/raw/structured'))
        return base / 'market_data' / 'etf_daily'
    
    def load_all(self) -> Dict[str, Any]:
        """
        加载所有参考数据
        
        Returns:
            {
                'benchmark': 基准指数行情 (trade_date, ts_code, pct_chg),
                'weights': 指数成分权重表,
                'etf': ETF行情 (可选)
            }
        """
        logger.info("=" * 50)
        logger.info("📋 加载参考数据 (Lookup Tables)")
        logger.info("=" * 50)
        
        ref_data = {}
        
        # 1. 加载基准指数行情
        ref_data['benchmark'] = self.load_benchmark_index()
        
        # 2. 加载指数权重
        ref_data['weights'] = self.load_index_weights()
        
        # 3. 加载ETF行情（可选）
        # ref_data['etf'] = self.load_etf_daily()
        
        logger.info("=" * 50)
        return ref_data
    
    def load_benchmark_index(self) -> Optional[Any]:
        """
        加载基准指数行情
        
        只保留核心指数：沪深300, 中证500, 中证1000
        返回简化表：trade_date, ts_code, pct_chg, close
        
        Returns:
            DataFrame 或 None
        """
        if 'benchmark' in self._cache and self.cache_enabled:
            return self._cache['benchmark']
        
        logger.info("  📖 加载指数日线行情...")
        
        path = self.index_daily_path
        if not path.exists():
            logger.warning(f"    ⚠️ 指数日线目录不存在: {path}")
            return None
        
        # 读取所有 parquet 文件
        dfs = []
        for f in path.glob('*.parquet'):
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                logger.debug(f"    读取失败 {f}: {e}")
        
        if not dfs:
            logger.warning("    ⚠️ 无可用指数日线数据")
            return None
        
        df = pd.concat(dfs, ignore_index=True)
        
        # 筛选核心指数
        df = df[df['ts_code'].isin(self.CORE_INDEXES)]
        
        # 只保留必要列
        required_cols = ['trade_date', 'ts_code', 'pct_chg', 'close']
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols]
        
        # 确保 trade_date 格式
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 确保数值列类型正确（解决 object 列问题）
        for col in ['pct_chg', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        
        # 转换为 GPU 格式
        if self.use_gpu and self._cudf:
            df = self._cudf.from_pandas(df)
        
        # 获取唯一指数代码（兼容 cuDF）
        if self.use_gpu and self._cudf:
            unique_codes = df['ts_code'].unique().to_arrow().to_pylist()
        else:
            unique_codes = df['ts_code'].unique().tolist()
        
        logger.info(f"    ✓ 基准指数: {unique_codes}, {len(df):,} 行")
        
        if self.cache_enabled:
            self._cache['benchmark'] = df
        
        return df
    
    def load_index_weights(self, lag_days: int = 1) -> Optional[Any]:
        """
        加载指数成分权重
        
        Args:
            lag_days: 滞后天数（防止未来信息泄露）
        
        Returns:
            DataFrame (trade_date, con_code, index_code, weight)
        """
        if 'weights' in self._cache and self.cache_enabled:
            return self._cache['weights']
        
        logger.info("  📖 加载指数权重表...")
        
        path = self.index_weight_path
        if not path.exists():
            logger.warning(f"    ⚠️ 指数权重目录不存在: {path}")
            return None
        
        # 读取所有 parquet 文件
        dfs = []
        for f in path.glob('*.parquet'):
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                logger.debug(f"    读取失败 {f}: {e}")
        
        if not dfs:
            logger.warning("    ⚠️ 无可用指数权重数据")
            return None
        
        df = pd.concat(dfs, ignore_index=True)
        
        # 筛选核心指数
        index_col = 'index_code' if 'index_code' in df.columns else 'ts_code'
        df = df[df[index_col].isin(self.CORE_INDEXES)]
        
        # 标准化列名
        if 'con_code' not in df.columns and 'stock_code' in df.columns:
            df = df.rename(columns={'stock_code': 'con_code'})
        
        # 确保 trade_date 格式
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # LAG处理：防止未来信息泄露
        if lag_days > 0 and 'trade_date' in df.columns:
            df['trade_date'] = df['trade_date'] + pd.Timedelta(days=lag_days)
        
        # 转换为 GPU 格式
        if self.use_gpu and self._cudf:
            df = self._cudf.from_pandas(df)
        
        logger.info(f"    ✓ 指数权重: {len(df):,} 行 (Lag {lag_days} 天)")
        
        if self.cache_enabled:
            self._cache['weights'] = df
        
        return df
    
    def load_etf_daily(self) -> Optional[Any]:
        """
        加载ETF日线行情
        
        用于行业相对强弱计算（可选功能）
        
        Returns:
            DataFrame 或 None
        """
        if 'etf' in self._cache and self.cache_enabled:
            return self._cache['etf']
        
        logger.info("  📖 加载ETF日线...")
        
        path = self.etf_daily_path
        if not path.exists():
            logger.warning(f"    ⚠️ ETF日线目录不存在: {path}")
            return None
        
        # 读取所有 parquet 文件
        dfs = []
        for f in path.glob('*.parquet'):
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                logger.debug(f"    读取失败 {f}: {e}")
        
        if not dfs:
            logger.warning("    ⚠️ 无可用ETF日线数据")
            return None
        
        df = pd.concat(dfs, ignore_index=True)
        
        # 确保 trade_date 格式
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 转换为 GPU 格式
        if self.use_gpu and self._cudf:
            df = self._cudf.from_pandas(df)
        
        logger.info(f"    ✓ ETF日线: {len(df):,} 行, {df['ts_code'].nunique()} 只")
        
        if self.cache_enabled:
            self._cache['etf'] = df
        
        return df
    
    def get_benchmark_returns(
        self, 
        index_code: str = '000300.SH',
        col_name: str = 'bench_pct_chg'
    ) -> Optional[Any]:
        """
        获取基准收益率序列（用于 Merge）
        
        Args:
            index_code: 基准指数代码
            col_name: 输出列名
        
        Returns:
            DataFrame (trade_date, {col_name})
        """
        benchmark = self.load_benchmark_index()
        if benchmark is None:
            return None
        
        df = benchmark[benchmark['ts_code'] == index_code][['trade_date', 'pct_chg']].copy()
        df = df.rename(columns={'pct_chg': col_name})
        
        return df
    
    def get_index_member_flags(
        self,
        index_codes: Optional[List[str]] = None
    ) -> Optional[Any]:
        """
        获取指数成分股标记表
        
        Args:
            index_codes: 要标记的指数代码列表
        
        Returns:
            DataFrame (trade_date, con_code, is_hs300, is_csi500, weight_hs300, weight_csi500)
        """
        weights = self.load_index_weights()
        if weights is None:
            return None
        
        if index_codes is None:
            index_codes = ['000300.SH', '000905.SH']
        
        # 将长表 Pivot 为宽表
        index_col = 'index_code' if 'index_code' in weights.columns else 'ts_code'
        
        # 转回 pandas 进行 pivot（cuDF 的 pivot 支持有限）
        if self.use_gpu and self._cudf:
            weights_pd = weights.to_pandas()
        else:
            weights_pd = weights
        
        # 创建成分股标记
        result_dfs = []
        for idx_code in index_codes:
            idx_df = weights_pd[weights_pd[index_col] == idx_code].copy()
            short_name = idx_code.split('.')[0]  # 000300.SH -> 000300
            
            idx_df[f'is_{short_name}'] = 1
            idx_df[f'weight_{short_name}'] = idx_df['weight']
            idx_df = idx_df[['trade_date', 'con_code', f'is_{short_name}', f'weight_{short_name}']]
            result_dfs.append(idx_df)
        
        if not result_dfs:
            return None
        
        # 合并所有指数标记
        result = result_dfs[0]
        for df in result_dfs[1:]:
            result = pd.merge(result, df, on=['trade_date', 'con_code'], how='outer')
        
        # 填充NaN为0
        is_cols = [c for c in result.columns if c.startswith('is_')]
        weight_cols = [c for c in result.columns if c.startswith('weight_')]
        result[is_cols + weight_cols] = result[is_cols + weight_cols].fillna(0)
        
        # 转换为GPU
        if self.use_gpu and self._cudf:
            result = self._cudf.from_pandas(result)
        
        return result
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        logger.info("✓ 参考数据缓存已清除")
