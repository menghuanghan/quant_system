"""
非结构化DWD预处理器

处理 dwd_unstructured 表，主要包括：
1. 缺失值填充（6个score+market_sentiment填0，beta_signal填1）
2. has_* 指示列生成（基于fillna后、log前原始值）
3. 对6个score执行 signed-log 变换（保留方向，压缩长尾）
"""

import logging
from typing import Any, Dict

from .base import BasePreprocessor
from .config import PreprocessConfig

logger = logging.getLogger(__name__)


class UnstructuredPreprocessor(BasePreprocessor):
    """非结构化DWD预处理器"""

    def __init__(self, config: PreprocessConfig):
        super().__init__(config)
        self.stats: Dict[str, Any] = {}

    def process(self, df: Any) -> Any:
        """
        执行非结构化DWD预处理

        Args:
            df: 输入的非结构化DWD表 DataFrame

        Returns:
            处理后的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("📊 开始处理非结构化表 (dwd_unstructured)")
        logger.info("=" * 60)

        original_shape = df.shape
        df = df.copy()

        # 1) 缺失值填充
        df = self._fill_missing_values(df)

        # 2) 生成 has_* 信号列（基于填充后、log前值）
        df = self._create_has_features(df)

        # 3) 6个score做 signed-log
        df = self._apply_signed_log(df)

        self.stats["original_shape"] = original_shape
        self.stats["final_shape"] = df.shape

        logger.info(f"✅ 非结构化表处理完成: {original_shape} -> {df.shape}")

        return df

    def _fill_missing_values(self, df: Any) -> Any:
        """缺失值填充规则"""
        logger.info("📌 Step 1: 缺失值填充")

        zero_fields = self.config.unstructured.fillna_zero_fields
        beta_field = self.config.unstructured.beta_field
        beta_fillna_value = self.config.unstructured.beta_fillna_value

        fill_stats: Dict[str, int] = {}

        for col in zero_fields:
            if col not in df.columns:
                continue
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                df[col] = df[col].fillna(0.0)
            fill_stats[col] = null_count

        if beta_field in df.columns:
            null_count = int(df[beta_field].isna().sum())
            if null_count > 0:
                df[beta_field] = df[beta_field].fillna(beta_fillna_value)
            fill_stats[beta_field] = null_count

        self.stats["fillna_counts"] = fill_stats
        logger.info(f"  ✓ 缺失值填充完成: {len(fill_stats)} 列")

        return df

    def _create_has_features(self, df: Any) -> Any:
        """基于原始score（填充后、log前）生成 has_* 列"""
        logger.info("📌 Step 2: 生成 has_* 指示列")

        has_map = self.config.unstructured.has_signal_map
        created = []

        for src_col, has_col in has_map.items():
            if src_col not in df.columns:
                continue

            df[has_col] = (df[src_col] != 0).astype("int8")
            created.append(has_col)

        self.stats["has_feature_count"] = len(created)
        self.stats["has_features"] = created

        if created:
            logger.info(f"  ✓ 新增 {len(created)} 个 has_* 列")
        else:
            logger.info("  ✓ 未生成 has_* 列（源列缺失）")

        return df

    def _apply_signed_log(self, df: Any) -> Any:
        """对指定列执行 signed-log 变换：sign(x) * log(|x| + epsilon)"""
        logger.info("📌 Step 3: signed-log 变换")

        fields = self.config.unstructured.signed_log_fields
        epsilon = self.config.unstructured.signed_log_epsilon

        transformed = []
        for col in fields:
            if col not in df.columns:
                continue

            df = self.log_transform(
                df,
                column=col,
                epsilon=epsilon,
                output_column=col,
                inplace=True,
            )
            transformed.append(col)

        self.stats["signed_log_features"] = transformed
        self.stats["signed_log_count"] = len(transformed)

        logger.info(f"  ✓ signed-log 完成: {len(transformed)} 列")

        return df

    def get_stats(self) -> Dict[str, Any]:
        """获取预处理统计信息"""
        return self.stats
