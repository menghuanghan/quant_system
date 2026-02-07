"""
资金流特征生成器 (Money Flow Feature Generator)

基于 dwd_money_flow 构建资金因子：
- 主力资金强度 (Main Force Intensity)
- 大宗交易影响 (Block Trade Impact)
- 散户情绪 (Retail Sentiment)
- 北向资金偏好 (Northbound Preference)

所有比率类特征使用 amount 作为分母，已预处理为元单位。
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 防止除零的小量
EPSILON = 1e-10

# 单位转换：万元 -> 元
# 资金流字段（net_main_amount, hsgt_north 等）为万元单位
# 成交额 amount 为元单位，需要将分子乘以 10000 对齐
UNIT_CONVERSION_FACTOR = 10000.0


class MoneyFlowFeatureGenerator:
    """
    资金流特征生成器
    
    计算资金流向相关的因子，包括主力、散户、北向等维度。
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
                logger.info("🚀 MoneyFlowFeatureGenerator: GPU 模式")
            except ImportError:
                import pandas as pd
                self.pd = pd
                self.use_gpu = False
        else:
            import pandas as pd
            self.pd = pd
    
    def fit_transform(self, df: Any) -> Any:
        """
        生成资金流特征
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            添加资金流特征后的 DataFrame
        """
        logger.info("  📊 资金流特征生成...")
        
        # 检查必要列
        if 'amount' not in df.columns:
            logger.warning("    ⚠️ 缺少 amount 列，跳过资金流特征")
            return df
        
        # 基础分母
        amount = df['amount'] + EPSILON
        
        # 1. 主力资金强度
        df = self._generate_main_force_features(df, amount)
        
        # 2. 大宗交易影响
        df = self._generate_block_trade_features(df, amount)
        
        # 3. 散户情绪
        df = self._generate_retail_features(df, amount)
        
        # 4. 北向资金偏好
        df = self._generate_northbound_features(df, amount)
        
        # 5. 资金流动态特征
        df = self._generate_flow_dynamics(df)
        
        logger.info(f"    ✓ 共生成 {len(self.stats['generated_features'])} 个资金流特征")
        return df
    
    def _generate_main_force_features(self, df: Any, amount: Any) -> Any:
        """
        主力资金特征
        
        - mf_main_intensity: 主力净流入 / 成交额
        - mf_elg_intensity: 特大单净流入 / 成交额
        - mf_lg_intensity: 大单净流入 / 成交额
        """
        features = {}
        
        # 主力净流入强度 (超大+大单)
        # 注意：net_main_amount 为万元单位，amount 为元单位，需乘以 10000 对齐
        if 'net_main_amount' in df.columns:
            features['mf_main_intensity'] = df['net_main_amount'] * UNIT_CONVERSION_FACTOR / amount
        
        # 特大单净流入强度
        if 'net_elg_amount' in df.columns:
            features['mf_elg_intensity'] = df['net_elg_amount'] * UNIT_CONVERSION_FACTOR / amount
        
        # 大单净流入强度
        if 'net_lg_amount' in df.columns:
            features['mf_lg_intensity'] = df['net_lg_amount'] * UNIT_CONVERSION_FACTOR / amount
        
        # 中单净流入强度
        if 'net_md_amount' in df.columns:
            features['mf_md_intensity'] = df['net_md_amount'] * UNIT_CONVERSION_FACTOR / amount
        
        # 应用特征
        for name, values in features.items():
            df[name] = values
            self.stats['generated_features'].append(name)
        
        if features:
            logger.info(f"    ✓ 主力资金特征: {list(features.keys())}")
        
        return df
    
    def _generate_block_trade_features(self, df: Any, amount: Any) -> Any:
        """
        大宗交易特征
        
        - mf_block_intensity: 大宗交易金额 / 成交额
        - mf_block_premium: 大宗交易溢价率（直接使用）
        """
        features = {}
        
        # 大宗交易强度
        if 'block_trade_amount' in df.columns:
            features['mf_block_intensity'] = df['block_trade_amount'] / amount
        
        # 大宗交易溢价率（已存在则直接使用）
        if 'block_trade_premium' in df.columns:
            features['mf_block_premium'] = df['block_trade_premium']
        
        # 应用特征
        for name, values in features.items():
            df[name] = values
            self.stats['generated_features'].append(name)
        
        if features:
            logger.info(f"    ✓ 大宗交易特征: {list(features.keys())}")
        
        return df
    
    def _generate_retail_features(self, df: Any, amount: Any) -> Any:
        """
        散户情绪特征
        
        - mf_retail_intensity: 散户净流入 / 成交额（小单净流入）
        - mf_retail_buy_ratio: 散户买入占比
        """
        features = {}
        
        # 散户净流入强度（小单）
        # 注意：net_sm_amount 为万元单位，需乘以 10000 对齐
        if 'net_sm_amount' in df.columns:
            features['mf_retail_intensity'] = df['net_sm_amount'] * UNIT_CONVERSION_FACTOR / amount
        
        # 散户买入占比（buy_sm_amount 已在预处理中换算为元，无需再乘）
        if 'buy_sm_amount' in df.columns:
            features['mf_retail_buy_ratio'] = df['buy_sm_amount'] / amount
        
        # 散户卖出占比（sell_sm_amount 已在预处理中换算为元，无需再乘）
        if 'sell_sm_amount' in df.columns:
            features['mf_retail_sell_ratio'] = df['sell_sm_amount'] / amount
        
        # 应用特征
        for name, values in features.items():
            df[name] = values
            self.stats['generated_features'].append(name)
        
        if features:
            logger.info(f"    ✓ 散户情绪特征: {list(features.keys())}")
        
        return df
    
    def _generate_northbound_features(self, df: Any, amount: Any) -> Any:
        """
        北向资金特征
        
        注意：hsgt_north 是市场级数据（当日北向资金总净流入），
        非个股级！用于捕捉宏观资金情绪。
        
        - mf_north_net: 北向资金净流入原始值（万元），市场级
        - mf_north_hold_ratio: 北向持股占流通市值比例（个股级，如有）
        """
        features = {}
        
        # 北向资金净流入原始值（万元）- 市场级宏观情绪
        if 'hsgt_north' in df.columns:
            features['mf_north_net'] = df['hsgt_north']
        
        # 北向持股占比（如果有）
        if 'hsgt_hold_ratio' in df.columns:
            features['mf_north_hold_ratio'] = df['hsgt_hold_ratio']
        
        # 应用特征
        for name, values in features.items():
            df[name] = values
            self.stats['generated_features'].append(name)
        
        if features:
            logger.info(f"    ✓ 北向资金特征: {list(features.keys())}")
        
        return df
    
    def _generate_flow_dynamics(self, df: Any) -> Any:
        """
        资金流动态特征（需要分组计算）
        
        - mf_main_momentum_5d: 主力净流入5日动量
        - mf_main_persistence: 主力连续流入天数
        """
        if 'net_mf_amount' not in df.columns:
            return df
        
        # 需要按股票分组，计算累积效应
        # 这里只计算简单衍生，复杂 Rolling 留给技术指标模块
        
        # 主力净流入符号（用于后续统计）
        df['mf_main_sign'] = (df['net_mf_amount'] > 0).astype('int8') - (df['net_mf_amount'] < 0).astype('int8')
        self.stats['generated_features'].append('mf_main_sign')
        
        logger.info("    ✓ 资金流动态特征: ['mf_main_sign']")
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats
