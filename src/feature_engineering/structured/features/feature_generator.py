"""
特征衍生计算模块

计算技术指标和基本面衍生指标。
支持 GPU 加速 (cuDF) 和 CPU (pandas)。

集成多个特征生成器：
- 技术指标：MA, Bias, ROC, RSI, MACD, 波动率, 振幅, 量比
- 基本面衍生：EP-增长率交叉等
- 资金流特征：主力强度, 大宗交易, 散户情绪, 北向资金
- 筹码特征：股东户数, 持股集中度, 机构持股
- 相对强弱：超额收益, Beta
- 指数成分：成分股标记, 权重因子
- 宏观交互：股债性价比, 流动性敏感度

注意：cuDF 的 groupby.transform 不支持 lambda 中的 rolling，
因此使用不同的实现策略。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """特征生成器（主入口）"""
    
    def __init__(
        self, 
        config, 
        use_gpu: bool = True,
        ref_data: Optional[Dict[str, Any]] = None
    ):
        """
        初始化特征生成器
        
        Args:
            config: TechnicalFeatureConfig 配置
            use_gpu: 是否使用 GPU
            ref_data: 参考数据字典（用于相对强弱和指数成分特征）
        """
        self.config = config
        self.use_gpu = use_gpu
        self.ref_data = ref_data or {}
        self.stats: Dict[str, Any] = {}
        
        if use_gpu:
            try:
                import cudf
                self.pd = cudf
                self.cudf = cudf
                logger.info("🚀 FeatureGenerator: GPU 加速已启用 (cuDF)")
            except ImportError:
                import pandas as pd
                self.pd = pd
                self.cudf = None
                self.use_gpu = False
                logger.warning("⚠️ cuDF 不可用，回退到 pandas")
        else:
            import pandas as pd
            self.pd = pd
            self.cudf = None
        
        # 初始化子生成器
        self._init_sub_generators()
    
    def _init_sub_generators(self):
        """初始化子特征生成器"""
        from .money_flow_features import MoneyFlowFeatureGenerator
        from .chip_features import ChipFeatureGenerator
        from .relative_strength_features import RelativeStrengthGenerator, IndexMemberGenerator
        from .macro_interaction_features import MacroInteractionGenerator
        
        self.money_flow_gen = MoneyFlowFeatureGenerator(self.config, self.use_gpu)
        self.chip_gen = ChipFeatureGenerator(self.config, self.use_gpu)
        self.rs_gen = RelativeStrengthGenerator(self.config, self.ref_data, self.use_gpu)
        self.index_gen = IndexMemberGenerator(self.config, self.ref_data, self.use_gpu)
        self.macro_gen = MacroInteractionGenerator(self.config, self.use_gpu)
    
    def set_ref_data(self, ref_data: Dict[str, Any]):
        """设置参考数据（用于延迟加载场景）"""
        self.ref_data = ref_data
        # 更新依赖参考数据的生成器
        self.rs_gen.ref_data = ref_data
        self.index_gen.ref_data = ref_data
    
    def generate_all(self, df: Any) -> Any:
        """
        生成所有特征
        
        执行顺序：
        1. 技术指标（MA, RSI, MACD 等）
        2. 基本面衍生
        3. 资金流特征
        4. 筹码特征
        5. 相对强弱特征（需参考数据）
        6. 指数成分特征（需参考数据）
        7. 宏观交互特征
        
        Args:
            df: 输入 DataFrame (需包含 ts_code, trade_date, close_hfq, vol 等)
            
        Returns:
            添加了所有特征的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📋 Step 3: 特征衍生计算")
        logger.info("=" * 60)
        
        original_cols = len(df.columns)
        
        # 确保按股票和日期排序
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        # 1. 技术指标计算（分组操作）
        df = self._generate_technical_features(df)
        
        # 2. 基本面衍生特征
        df = self._generate_fundamental_features(df)
        
        # 3. 资金流特征
        df = self.money_flow_gen.fit_transform(df)
        
        # 4. 筹码特征
        df = self.chip_gen.fit_transform(df)
        
        # 5. 相对强弱特征（需参考数据）
        if self.ref_data:
            df = self.rs_gen.fit_transform(df)
            
            # 6. 指数成分特征（需参考数据）
            df = self.index_gen.fit_transform(df)
        else:
            logger.info("  ⚠️ 未提供参考数据，跳过相对强弱和指数成分特征")
        
        # 7. 宏观交互特征
        df = self.macro_gen.fit_transform(df)
        
        new_cols = len(df.columns) - original_cols
        logger.info(f"  ✅ 特征生成完成: 新增 {new_cols} 列，总计 {len(df.columns)} 列")
        
        self.stats["original_cols"] = original_cols
        self.stats["new_cols"] = new_cols
        self.stats["final_cols"] = len(df.columns)
        
        return df
    
    def _generate_technical_features(self, df: Any) -> Any:
        """
        生成技术指标特征
        
        包括：MA、Bias、ROC、RSI、MACD、波动率、振幅、量比
        """
        logger.info("  📊 计算技术指标...")
        
        # 获取收盘价列
        price_col = 'close_hfq' if 'close_hfq' in df.columns else 'close'
        
        # 1. 移动平均线 (MA) - 使用 groupby rolling
        df = self._calculate_grouped_rolling_mean(df, price_col, self.config.ma_periods, prefix='ma')
        logger.info(f"    ✓ MA (周期: {self.config.ma_periods})")
        
        # 2. 乖离率 (Bias) = (Close - MA) / MA
        for period in self.config.bias_periods:
            ma_col = f'ma_{period}'
            bias_col = f'bias_{period}'
            if ma_col in df.columns:
                df[bias_col] = (df[price_col] - df[ma_col]) / df[ma_col]
        logger.info(f"    ✓ Bias (周期: {self.config.bias_periods})")
        
        # 3. ROC (变化率) - 使用 pct_change
        df = self._calculate_grouped_pct_change(df, price_col, self.config.roc_periods, prefix='roc')
        logger.info(f"    ✓ ROC (周期: {self.config.roc_periods})")
        
        # 4. RSI (相对强弱指数)
        df = self._calculate_grouped_rsi(df, price_col, self.config.rsi_periods)
        logger.info(f"    ✓ RSI (周期: {self.config.rsi_periods})")
        
        # 5. MACD
        df = self._calculate_grouped_macd(df, price_col, *self.config.macd_params)
        logger.info(f"    ✓ MACD {self.config.macd_params}")
        
        # 6. 波动率 (过去 N 日收益率标准差)
        if 'return_1d' in df.columns:
            df = self._calculate_grouped_rolling_std(df, 'return_1d', self.config.volatility_periods, prefix='volatility')
            logger.info(f"    ✓ 波动率 (周期: {self.config.volatility_periods})")
        
        # 7. 振幅 = (High - Low) / Pre_Close 的 N 日均值
        if all(col in df.columns for col in ['high', 'low', 'pre_close']):
            df['amplitude'] = (df['high'] - df['low']) / df['pre_close']
            df = self._calculate_grouped_rolling_mean(df, 'amplitude', self.config.amplitude_periods, prefix='amplitude')
            logger.info(f"    ✓ 振幅 (周期: {self.config.amplitude_periods})")
        
        # 8. 量比 = 当日成交量 / 过去 N 日均量
        if 'vol' in df.columns:
            for period in self.config.volume_ratio_periods:
                # 先计算均量
                df = self._calculate_grouped_rolling_mean(df, 'vol', [period], prefix='vol_ma')
                ma_col = f'vol_ma_{period}'
                ratio_col = f'volume_ratio_{period}'
                df[ratio_col] = df['vol'] / df[ma_col]
                # 删除中间列 (使用 del 避免 cuDF 深拷贝)
                del df[ma_col]
            logger.info(f"    ✓ 量比 (周期: {self.config.volume_ratio_periods})")
        
        return df
    
    def _calculate_grouped_rolling_mean(self, df: Any, col: str, periods: List[int], prefix: str) -> Any:
        """
        计算分组滚动均值（兼容 cuDF 和 pandas）
        """
        if self.use_gpu:
            # cuDF 方式：使用 groupby.rolling
            for period in periods:
                out_col = f'{prefix}_{period}'
                rolling = df.groupby('ts_code')[col].rolling(window=period, min_periods=period)
                df[out_col] = rolling.mean().reset_index(level=0, drop=True)
        else:
            # pandas 方式：使用 groupby.transform
            grouped = df.groupby('ts_code', sort=False)
            for period in periods:
                out_col = f'{prefix}_{period}'
                df[out_col] = grouped[col].transform(
                    lambda x: x.rolling(window=period, min_periods=period).mean()
                )
        return df
    
    def _calculate_grouped_rolling_std(self, df: Any, col: str, periods: List[int], prefix: str) -> Any:
        """
        计算分组滚动标准差
        """
        if self.use_gpu:
            for period in periods:
                out_col = f'{prefix}_{period}'
                rolling = df.groupby('ts_code')[col].rolling(window=period, min_periods=period)
                df[out_col] = rolling.std().reset_index(level=0, drop=True)
        else:
            grouped = df.groupby('ts_code', sort=False)
            for period in periods:
                out_col = f'{prefix}_{period}'
                df[out_col] = grouped[col].transform(
                    lambda x: x.rolling(window=period, min_periods=period).std()
                )
        return df
    
    def _calculate_grouped_pct_change(self, df: Any, col: str, periods: List[int], prefix: str) -> Any:
        """
        计算分组百分比变化 (ROC)
        """
        if self.use_gpu:
            # cuDF 方式：使用 shift 手动计算
            for period in periods:
                out_col = f'{prefix}_{period}'
                shifted = df.groupby('ts_code')[col].shift(period)
                df[out_col] = (df[col] - shifted) / shifted
        else:
            grouped = df.groupby('ts_code', sort=False)
            for period in periods:
                out_col = f'{prefix}_{period}'
                df[out_col] = grouped[col].transform(
                    lambda x: x.pct_change(periods=period)
                )
        return df
    
    def _calculate_grouped_rsi(self, df: Any, col: str, periods: List[int]) -> Any:
        """
        计算分组 RSI 指标
        """
        # 计算价格变动
        if self.use_gpu:
            delta = df.groupby('ts_code')[col].diff()
        else:
            delta = df.groupby('ts_code', sort=False)[col].transform(lambda x: x.diff())
        
        # 分离涨跌
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        df['_temp_gain'] = gain
        df['_temp_loss'] = loss
        
        for period in periods:
            out_col = f'rsi_{period}'
            
            if self.use_gpu:
                # cuDF 方式：使用 rolling mean
                avg_gain = df.groupby('ts_code')['_temp_gain'].rolling(window=period, min_periods=period).mean().reset_index(level=0, drop=True)
                avg_loss = df.groupby('ts_code')['_temp_loss'].rolling(window=period, min_periods=period).mean().reset_index(level=0, drop=True)
            else:
                # pandas 方式：使用 ewm
                grouped = df.groupby('ts_code', sort=False)
                avg_gain = grouped['_temp_gain'].transform(
                    lambda x: x.ewm(span=period, adjust=False).mean()
                )
                avg_loss = grouped['_temp_loss'].transform(
                    lambda x: x.ewm(span=period, adjust=False).mean()
                )
            
            rs = avg_gain / (avg_loss + 1e-10)
            df[out_col] = 100 - 100 / (1 + rs)
        
        # 删除临时列 (使用 del 避免 cuDF 深拷贝)
        del df['_temp_gain']
        del df['_temp_loss']
        
        return df
    
    def _calculate_grouped_macd(self, df: Any, col: str, fast: int, slow: int, signal: int) -> Any:
        """
        计算分组 MACD 指标
        
        注意：cuDF 的 ewm 不完全支持 groupby，需要特殊处理。
        """
        if self.use_gpu:
            import cudf
            import pandas as pd
            
            # cuDF 的 ewm groupby 支持有限，转换为 pandas 计算
            # 只提取需要的列以减少内存使用
            calc_cols = ['ts_code', col]
            df_calc = df[calc_cols].to_pandas()
            
            grouped = df_calc.groupby('ts_code', sort=False)
            
            ema_fast = grouped[col].transform(
                lambda x: x.ewm(span=fast, adjust=False).mean()
            )
            ema_slow = grouped[col].transform(
                lambda x: x.ewm(span=slow, adjust=False).mean()
            )
            macd_line = ema_fast - ema_slow
            
            df_calc['macd'] = macd_line
            macd_signal_series = df_calc.groupby('ts_code', sort=False)['macd'].transform(
                lambda x: x.ewm(span=signal, adjust=False).mean()
            )
            
            # 将结果转回 cuDF
            df['macd'] = cudf.Series(macd_line.values)
            df['macd_signal'] = cudf.Series(macd_signal_series.values)
            df['macd_hist'] = df['macd'] - df['macd_signal']
        else:
            # pandas 方式
            grouped = df.groupby('ts_code', sort=False)
            
            df['_ema_fast'] = grouped[col].transform(
                lambda x: x.ewm(span=fast, adjust=False).mean()
            )
            df['_ema_slow'] = grouped[col].transform(
                lambda x: x.ewm(span=slow, adjust=False).mean()
            )
            df['macd'] = df['_ema_fast'] - df['_ema_slow']
            df['macd_signal'] = df.groupby('ts_code', sort=False)['macd'].transform(
                lambda x: x.ewm(span=signal, adjust=False).mean()
            )
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 删除中间列 (使用 del 避免 cuDF 深拷贝)
            del df['_ema_fast']
            del df['_ema_slow']
        
        return df
    
    def _generate_fundamental_features(self, df: Any) -> Any:
        """
        生成基本面衍生特征
        """
        logger.info("  📊 基本面特征处理...")
        
        # 0. 对数变换特征（处理偏态分布）
        # 注: 使用 pandas 计算以避免 cupy 内核兼容性问题
        log_cols = []
        for col, out_col in [('amount', 'log_amount'), ('total_mv', 'log_total_mv'), 
                              ('circ_mv', 'log_circ_mv'), ('vol', 'log_vol')]:
            if col in df.columns:
                if self.use_gpu:
                    # 转为 pandas 计算，再转回 cuDF
                    val_pd = df[col].to_pandas().clip(lower=0)
                    log_pd = np.log1p(val_pd)
                    df[out_col] = self.cudf.Series(log_pd.values, index=df.index)
                else:
                    val = df[col].clip(lower=0)
                    df[out_col] = np.log1p(val)
                log_cols.append(out_col)
        if log_cols:
            logger.info(f"    ✓ 对数变换特征: {log_cols}")
        
        # 1. EP 与增长率的交叉特征 (PEG 替代)
        if 'ep' in df.columns and 'revenue_yoy' in df.columns:
            # 当增长率为正时，EP * growth_rate 类似 EP 调整
            df['ep_growth'] = df['ep'] * df['revenue_yoy'].clip(lower=0)
            logger.info("    ✓ EP-增长率交叉特征")
        
        # 2. 质量因子整合
        quality_cols = []
        if 'roe' in df.columns:
            quality_cols.append('roe')
        if 'gross_margin' in df.columns:
            quality_cols.append('gross_margin')
        if 'debt_to_assets' in df.columns:
            quality_cols.append('debt_to_assets')
        
        if len(quality_cols) >= 2:
            logger.info(f"    ✓ 质量因子字段: {quality_cols}")
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.stats
