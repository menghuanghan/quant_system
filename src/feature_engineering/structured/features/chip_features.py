"""
筹码结构特征生成器 (Chip Structure Feature Generator)

基于 dwd_chip_structure 构建筹码因子：
- 筹码稳定性 (Chip Stability)
- 机构控盘度 (Institutional Control)
- 股东集中度 (Shareholder Concentration)

这些因子反映筹码分布和持股结构变化。
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EPSILON = 1e-10


class ChipFeatureGenerator:
    """
    筹码结构特征生成器
    
    计算筹码分布和持股结构相关因子。
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
                logger.info("🚀 ChipFeatureGenerator: GPU 模式")
            except ImportError:
                import pandas as pd
                self.pd = pd
                self.use_gpu = False
        else:
            import pandas as pd
            self.pd = pd
    
    def fit_transform(self, df: Any) -> Any:
        """
        生成筹码结构特征
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            添加筹码特征后的 DataFrame
        """
        logger.info("  📊 筹码结构特征生成...")
        
        # 1. 股东户数相关
        df = self._generate_holder_features(df)
        
        # 2. 持股集中度
        df = self._generate_concentration_features(df)
        
        # 3. 机构持股
        df = self._generate_institutional_features(df)
        
        # 4. 筹码稳定性
        df = self._generate_stability_features(df)
        
        logger.info(f"    ✓ 共生成 {len(self.stats['generated_features'])} 个筹码特征")
        return df
    
    def _generate_holder_features(self, df: Any) -> Any:
        """
        股东户数特征
        
        - chip_holder_growth: 股东户数增长率（已有 holder_num_chg_pct）
        - chip_holder_level: 股东户数水平（标准化后）
        """
        features = {}
        
        # 股东户数变化率（负值表示筹码集中，利好）
        if 'holder_num_chg_pct' in df.columns:
            # 原值已经是百分比，取负转为"集中度变化"
            features['chip_holder_chg'] = df['holder_num_chg_pct']
        
        # 股东户数对数（规模标准化）
        if 'holder_num' in df.columns:
            holder_clipped = df['holder_num'].clip(lower=0)
            import numpy as np
            if self.use_gpu:
                # cuDF: 转 CPU 做 log 再转回（避免 CUDA kernel 编译问题）
                import cudf
                holder_cpu = holder_clipped.to_pandas()
                ln_cpu = np.log1p(holder_cpu)
                features['chip_ln_holder_num'] = cudf.Series(ln_cpu.values, index=holder_clipped.index)
            else:
                features['chip_ln_holder_num'] = np.log1p(holder_clipped)
        
        # 应用特征
        for name, values in features.items():
            df[name] = values
            self.stats['generated_features'].append(name)
        
        if features:
            logger.info(f"    ✓ 股东户数特征: {list(features.keys())}")
        
        return df
    
    def _generate_concentration_features(self, df: Any) -> Any:
        """
        持股集中度特征
        
        - chip_top10_ratio: 前十大股东持股比例（已有 top10_hold_ratio）
        - chip_top1_dominance: 第一大股东独大程度 = top1 / top10
        - chip_top10_inst_ratio: 机构持股占前十比例
        """
        features = {}
        
        # 前十大持股比例（直接使用预处理后的值）
        if 'top10_hold_ratio' in df.columns:
            features['chip_top10_ratio'] = df['top10_hold_ratio']
        
        # 第一大股东独大程度
        if 'top1_hold_ratio' in df.columns and 'top10_hold_ratio' in df.columns:
            top10 = df['top10_hold_ratio'] + EPSILON
            features['chip_top1_dominance'] = df['top1_hold_ratio'] / top10
        
        # 机构持股占前十比例
        if 'top10_inst_ratio' in df.columns and 'top10_hold_ratio' in df.columns:
            top10 = df['top10_hold_ratio'] + EPSILON
            features['chip_inst_in_top10'] = df['top10_inst_ratio'] / top10
        
        # 应用特征
        for name, values in features.items():
            df[name] = values
            self.stats['generated_features'].append(name)
        
        if features:
            logger.info(f"    ✓ 持股集中度特征: {list(features.keys())}")
        
        return df
    
    def _generate_institutional_features(self, df: Any) -> Any:
        """
        机构持股特征
        
        - chip_inst_ratio: 机构持股比例
        - chip_fund_ratio: 基金持股比例（如有）
        """
        features = {}
        
        # 机构持股比例
        if 'top10_inst_ratio' in df.columns:
            features['chip_inst_ratio'] = df['top10_inst_ratio']
        
        # 应用特征
        for name, values in features.items():
            df[name] = values
            self.stats['generated_features'].append(name)
        
        if features:
            logger.info(f"    ✓ 机构持股特征: {list(features.keys())}")
        
        return df
    
    def _generate_stability_features(self, df: Any) -> Any:
        """
        筹码稳定性特征
        
        - chip_congestion: 市场拥挤度（已有 market_congestion）
        - chip_stability_score: 筹码稳定性得分（综合）
        """
        features = {}
        
        # 市场拥挤度
        if 'market_congestion' in df.columns:
            features['chip_congestion'] = df['market_congestion']
        
        # 筹码稳定性得分（简化版：基于持股集中度和户数变化）
        stability_components = []
        if 'top10_hold_ratio' in df.columns:
            stability_components.append(df['top10_hold_ratio'] / 100)  # 归一化到0-1
        if 'holder_num_chg_pct' in df.columns:
            # 户数减少为正面信号，取负并裁剪
            holder_signal = (-df['holder_num_chg_pct']).clip(-1, 1)
            stability_components.append(holder_signal)
        
        if len(stability_components) >= 2:
            features['chip_stability_score'] = (stability_components[0] + stability_components[1]) / 2
        
        # 应用特征
        for name, values in features.items():
            df[name] = values
            self.stats['generated_features'].append(name)
        
        if features:
            logger.info(f"    ✓ 筹码稳定性特征: {list(features.keys())}")
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats
