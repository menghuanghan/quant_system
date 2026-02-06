"""
宏观交互特征生成器 (Macro Interaction Feature Generator)

利用 dwd_macro_env 中的宏观数据与个股数据做交互：
- 股债性价比交互
- 流动性敏感度
- 市场情绪交互

这类特征对深度学习和交叉特征学习特别有效。
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

EPSILON = 1e-10


class MacroInteractionGenerator:
    """
    宏观交互特征生成器
    
    计算个股数据与宏观环境的交互特征。
    """
    
    def __init__(self, config: Any = None, use_gpu: bool = True):
        """
        初始化
        
        Args:
            config: 配置对象
            use_gpu: 是否使用 GPU
        """
        self.config = config
        self.use_gpu = use_gpu
        self.stats: Dict[str, Any] = {'generated_features': []}
        
        if use_gpu:
            try:
                import cudf
                self.pd = cudf
                logger.info("🚀 MacroInteractionGenerator: GPU 模式")
            except ImportError:
                import pandas as pd
                self.pd = pd
                self.use_gpu = False
        else:
            import pandas as pd
            self.pd = pd
    
    def fit_transform(self, df: Any) -> Any:
        """
        生成宏观交互特征
        
        Args:
            df: 输入 DataFrame（应已包含宏观字段）
            
        Returns:
            添加宏观交互特征后的 DataFrame
        """
        logger.info("  📊 宏观交互特征生成...")
        
        # 1. 股债性价比交互
        df = self._generate_stock_bond_interaction(df)
        
        # 2. 流动性敏感度
        df = self._generate_liquidity_sensitivity(df)
        
        # 3. 估值-利率交互
        df = self._generate_valuation_rate_interaction(df)
        
        # 4. 成长-通胀交互
        df = self._generate_growth_inflation_interaction(df)
        
        logger.info(f"    ✓ 共生成 {len(self.stats['generated_features'])} 个宏观交互特征")
        return df
    
    def _generate_stock_bond_interaction(self, df: Any) -> Any:
        """
        股债性价比交互特征
        
        - macro_ep_sbs: EP * 股债利差 (估值 × 股债性价比)
        - macro_bp_sbs: BP * 股债利差
        
        意义：股债利差高位时，低估值股票可能更有吸引力
        """
        features = {}
        
        # 股债利差（stock_bond_spread）
        sbs_col = 'stock_bond_spread'
        
        if sbs_col in df.columns:
            # EP × 股债利差
            if 'ep' in df.columns:
                features['macro_ep_sbs'] = df['ep'] * df[sbs_col]
            
            # BP × 股债利差
            if 'bp' in df.columns:
                features['macro_bp_sbs'] = df['bp'] * df[sbs_col]
        
        # 应用特征
        for name, values in features.items():
            df[name] = values
            self.stats['generated_features'].append(name)
        
        if features:
            logger.info(f"    ✓ 股债性价比交互: {list(features.keys())}")
        
        return df
    
    def _generate_liquidity_sensitivity(self, df: Any) -> Any:
        """
        流动性敏感度特征
        
        - macro_amount_shibor: 成交额 × SHIBOR (归一化后)
        - macro_vol_m2: 成交量 × M2增速
        
        意义：流动性宽松时，高换手股票可能更活跃
        """
        features = {}
        
        # SHIBOR 交互
        shibor_col = 'shibor_1m' if 'shibor_1m' in df.columns else 'shibor_1w'
        if shibor_col in df.columns and 'amount' in df.columns:
            # 对数成交额 × SHIBOR（转 CPU 做 log 避免 CUDA kernel 编译问题）
            import numpy as np
            amount_clipped = df['amount'].clip(lower=0)
            if self.use_gpu:
                import cudf
                amount_cpu = amount_clipped.to_pandas()
                ln_cpu = np.log1p(amount_cpu)
                ln_amount = cudf.Series(ln_cpu.values, index=amount_clipped.index)
            else:
                ln_amount = np.log1p(amount_clipped)
            features['macro_amount_shibor'] = ln_amount * df[shibor_col] / 100
        
        # M2 交互
        if 'm2_yoy' in df.columns and 'vol' in df.columns:
            import numpy as np
            vol_clipped = df['vol'].clip(lower=0)
            if self.use_gpu:
                import cudf
                vol_cpu = vol_clipped.to_pandas()
                ln_cpu = np.log1p(vol_cpu)
                ln_vol = cudf.Series(ln_cpu.values, index=vol_clipped.index)
            else:
                ln_vol = np.log1p(vol_clipped)
            features['macro_vol_m2'] = ln_vol * df['m2_yoy']
        
        # 应用特征
        for name, values in features.items():
            df[name] = values
            self.stats['generated_features'].append(name)
        
        if features:
            logger.info(f"    ✓ 流动性敏感度: {list(features.keys())}")
        
        return df
    
    def _generate_valuation_rate_interaction(self, df: Any) -> Any:
        """
        估值-利率交互特征
        
        - macro_ep_rate: EP / (1 + 十年期国债收益率)
        - macro_pe_spread: PE - 利率倒数（相对估值）
        
        意义：利率下行时，高估值更容易被接受
        """
        features = {}
        
        # 十年期国债收益率
        bond_col = 'cn10y_yield' if 'cn10y_yield' in df.columns else None
        
        if bond_col and 'ep' in df.columns:
            # EP 相对利率调整
            adj_rate = 1 + df[bond_col] / 100  # 转为小数
            features['macro_ep_adj'] = df['ep'] / (adj_rate + EPSILON)
        
        # 应用特征
        for name, values in features.items():
            df[name] = values
            self.stats['generated_features'].append(name)
        
        if features:
            logger.info(f"    ✓ 估值-利率交互: {list(features.keys())}")
        
        return df
    
    def _generate_growth_inflation_interaction(self, df: Any) -> Any:
        """
        成长-通胀交互特征
        
        - macro_growth_cpi: 营收增速 / (1 + CPI)（实际增长）
        - macro_growth_ppi: 营收增速 - PPI（剔除价格影响）
        
        意义：高通胀下，名义增速需要打折
        """
        features = {}
        
        # CPI 交互
        if 'cpi_yoy' in df.columns and 'revenue_yoy' in df.columns:
            adj_cpi = 1 + df['cpi_yoy'] / 100
            features['macro_real_growth'] = df['revenue_yoy'] / (adj_cpi + EPSILON)
        
        # PPI 交互
        if 'ppi_yoy' in df.columns and 'revenue_yoy' in df.columns:
            features['macro_growth_ex_ppi'] = df['revenue_yoy'] - df['ppi_yoy']
        
        # 应用特征
        for name, values in features.items():
            df[name] = values
            self.stats['generated_features'].append(name)
        
        if features:
            logger.info(f"    ✓ 成长-通胀交互: {list(features.keys())}")
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats
