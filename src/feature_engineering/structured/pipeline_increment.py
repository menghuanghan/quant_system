"""
结构化特征工程增量流水线

目标：
- 以 latest_date 为锚点，使用 DuckDB 合并“增量 + 部分全量（近 300 交易日）”输入
- 复用现有特征/标签生成与后处理模块
- 输出增量预测特征文件：
  - predict_lgb.parquet（仅 latest_date）
  - predict_gru.parquet（最近 60 个交易日）
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .increment import (
    DuckDBIncrementDWDProvider,
    DuckDBIncrementReferenceProvider,
    normalize_iso_date,
)

logger = logging.getLogger(__name__)


class FeaturePipelineIncrement:
    """结构化特征工程增量流水线。"""

    def __init__(
        self,
        config,
        latest_date: str,
        lookback_trade_days: Optional[int] = None,
        gru_trade_days: Optional[int] = None,
    ):
        self.config = config
        self.latest_date = normalize_iso_date(latest_date)
        self.latest_date_ts = pd.to_datetime(self.latest_date)

        self.lookback_trade_days = int(
            lookback_trade_days
            if lookback_trade_days is not None
            else getattr(self.config.data, "increment_lookback_trade_days", 300)
        )
        self.gru_trade_days = int(
            gru_trade_days
            if gru_trade_days is not None
            else getattr(self.config.data, "increment_gru_trade_days", 60)
        )

        self.increment_output_dir = (
            Path(getattr(self.config.data, "increment_output_root")) / f"date={self.latest_date}"
        )

        self.use_gpu = bool(getattr(config, "use_gpu", False))
        self.stats: Dict[str, Any] = {}

        # 初始化 DataFrame 库
        if self.use_gpu:
            try:
                import cudf

                self.pd = cudf
                logger.info("🚀 FeaturePipelineIncrement: GPU 加速已启用 (cuDF)")
            except ImportError:
                import pandas as pandas_lib

                self.pd = pandas_lib
                self.use_gpu = False
                logger.warning("⚠️ cuDF 不可用，回退到 pandas")
        else:
            import pandas as pandas_lib

            self.pd = pandas_lib

        self._init_modules()

    def _init_modules(self):
        """初始化增量流水线各处理模块。"""
        from .features.feature_generator import FeatureGenerator
        from .features.reference_data_loader import ReferenceDataLoader
        from .labels.label_generator import LabelGenerator
        from .merger.merger import DataMerger
        from .postprocess import PostprocessConfig, PostprocessMode, PostprocessPipeline
        from .preprocess import (
            ChipPreprocessor,
            EventPreprocessor,
            FundamentalPreprocessor,
            IndustryPreprocessor,
            MacroPreprocessor,
            MoneyFlowPreprocessor,
            PreprocessConfig,
            PricePreprocessor,
            StatusPreprocessor,
            UnstructuredPreprocessor,
        )

        preprocess_config = PreprocessConfig(use_gpu=self.use_gpu)

        self.preprocessors = {
            "price": PricePreprocessor(preprocess_config),
            "fundamental": FundamentalPreprocessor(preprocess_config),
            "status": StatusPreprocessor(preprocess_config),
            "money_flow": MoneyFlowPreprocessor(preprocess_config),
            "chip": ChipPreprocessor(preprocess_config),
            "industry": IndustryPreprocessor(preprocess_config),
            "macro": MacroPreprocessor(preprocess_config),
            "event": EventPreprocessor(preprocess_config),
            "unstructured": UnstructuredPreprocessor(preprocess_config),
        }

        # 增量输入 provider（DWD + 参考数据）
        self.dwd_provider = DuckDBIncrementDWDProvider(
            latest_date=self.latest_date,
            lookback_trade_days=self.lookback_trade_days,
        )
        self.reference_provider = DuckDBIncrementReferenceProvider(
            latest_date=self.latest_date,
            lookback_trade_days=self.lookback_trade_days,
        )

        self.merger = DataMerger(
            self.config.data,
            use_gpu=self.use_gpu,
            table_provider=self.dwd_provider,
        )
        self.ref_loader = ReferenceDataLoader(
            self.config.data,
            use_gpu=self.use_gpu,
            provider=self.reference_provider,
        )

        self.feature_generator = FeatureGenerator(self.config.technical, self.use_gpu, ref_data=None)
        self.label_generator = LabelGenerator(self.config.label, self.use_gpu, ref_data=None)

        self._PostprocessPipeline = PostprocessPipeline
        self._PostprocessMode = PostprocessMode
        self._PostprocessConfig = PostprocessConfig

    def run(self, save_output: bool = True) -> Dict[str, Any]:
        """执行增量特征工程流水线。"""
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("🚀 结构化特征增量流水线启动")
        logger.info("   latest_date: %s", self.latest_date)
        logger.info("   lookback_trade_days: %s", self.lookback_trade_days)
        logger.info("   gru_trade_days: %s", self.gru_trade_days)
        logger.info("=" * 70)

        self.increment_output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: 流式预处理+合并（输入由 provider 注入）
        logger.info("📋 Step 1: 流式预处理+合并 (增量+部分全量输入)")
        df = self.merger.process_with_preprocessing(
            preprocessors=self.preprocessors,
            filter_universe=True,
            drop_unnecessary=False,
            save_result=False,
        )
        df = self._filter_to_latest_date(df)
        self.stats["merged_rows"] = int(len(df))
        self.stats["merged_cols"] = int(len(df.columns))

        # Step 2: 加载参考数据（同样走增量 provider）
        logger.info("📋 Step 2: 加载参考数据 (增量+部分全量)")
        ref_data = self.ref_loader.load_all()
        self.feature_generator.set_ref_data(ref_data)
        self.label_generator.ref_data = ref_data

        # Step 3: 逐列计算特征（内存态，不落盘）
        logger.info("📋 Step 3: 逐列计算特征 (内存态)")
        feature_columns = self.feature_generator.generate_column_by_column_from_df(
            source_df=df,
            ref_data=ref_data,
        )
        self.stats["feature_columns"] = len(feature_columns)

        # Step 4: 逐列计算标签（内存态，不落盘）
        logger.info("📋 Step 4: 逐列计算标签 (内存态)")
        label_columns = self.label_generator.generate_labels_column_by_column_from_df(
            source_df=df,
        )
        self.stats["label_columns"] = len(label_columns)

        # Step 4.5: 合并主表+特征+标签（排序对齐）
        logger.info("📋 Step 4.5: 合并特征+标签")
        df_final = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        for col_name, col_data in feature_columns.items():
            df_final[col_name] = col_data
        for col_name, col_data in label_columns.items():
            df_final[col_name] = col_data

        del df, feature_columns, label_columns
        self._cleanup_memory()

        df_final = self._filter_to_latest_date(df_final)
        self.stats["pre_postprocess_rows"] = int(len(df_final))
        self.stats["pre_postprocess_cols"] = int(len(df_final.columns))

        # Step 5: 构造增量切分配置并执行后处理
        logger.info("📋 Step 5: 后处理并输出 predict_lgb/predict_gru")
        gru_start_date = self._compute_gru_start_date(df_final)
        self.stats["gru_start_date"] = (
            gru_start_date.strftime("%Y-%m-%d") if gru_start_date is not None else None
        )

        pp_config = self._build_postprocess_config(gru_start_date)
        postprocess_pipeline = self._PostprocessPipeline(
            config=pp_config,
            use_gpu=self.use_gpu,
            mode=self._PostprocessMode.BOTH,
        )
        pp_result = postprocess_pipeline.run(df_final, save_output=save_output)
        self.stats["postprocess"] = postprocess_pipeline.get_stats()

        schema_align_stats = {}
        if save_output:
            predict_lgb_path = self.increment_output_dir / "predict_lgb.parquet"
            predict_gru_path = self.increment_output_dir / "predict_gru.parquet"
            full_lgb_path = Path(self.config.data.output_dir) / "train_lgb.parquet"
            full_gru_path = Path(self.config.data.output_dir) / "train_gru.parquet"

            schema_align_stats["lgb"] = self._align_saved_predict_schema(
                predict_path=predict_lgb_path,
                baseline_path=full_lgb_path,
                tag="LGB",
            )
            schema_align_stats["gru"] = self._align_saved_predict_schema(
                predict_path=predict_gru_path,
                baseline_path=full_gru_path,
                tag="GRU",
            )
            self.stats["schema_alignment"] = schema_align_stats

        elapsed = time.time() - start_time
        self.stats["elapsed_seconds"] = elapsed

        result = {
            "latest_date": self.latest_date,
            "output_dir": str(self.increment_output_dir),
            "predict_lgb_path": str(self.increment_output_dir / "predict_lgb.parquet"),
            "predict_gru_path": str(self.increment_output_dir / "predict_gru.parquet"),
            "postprocess_result": pp_result,
            "schema_alignment": schema_align_stats,
            "elapsed_seconds": elapsed,
        }

        logger.info("=" * 70)
        logger.info("🎉 增量流水线完成: output_dir=%s, elapsed=%.2fs", self.increment_output_dir, elapsed)
        logger.info("=" * 70)

        return result

    def get_stats(self) -> Dict[str, Any]:
        return self.stats

    def _build_postprocess_config(self, gru_start_date: Optional[pd.Timestamp]):
        """构造增量后处理配置。"""
        pp_config = self._PostprocessConfig.default()
        pp_config.output_dir = self.increment_output_dir

        # 增量产物命名
        pp_config.lgb.output_file = "predict_lgb.parquet"
        pp_config.gru.output_file = "predict_gru.parquet"

        # LGB: 仅保留 latest_date
        lgb_cut_end = (self.latest_date_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        pp_config.lgb.cut_start = "1900-01-01"
        pp_config.lgb.cut_end = lgb_cut_end

        # GRU: 保留最近 gru_trade_days 交易日（<=latest_date）
        if gru_start_date is None:
            gru_cut_end = lgb_cut_end
        else:
            gru_cut_end = (gru_start_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        pp_config.gru.cut_start = "1900-01-01"
        pp_config.gru.cut_end = gru_cut_end

        logger.info(
            "后处理切分配置: LGB keep=%s, GRU keep=[%s,%s]",
            self.latest_date,
            gru_start_date.strftime("%Y-%m-%d") if gru_start_date is not None else self.latest_date,
            self.latest_date,
        )

        return pp_config

    def _filter_to_latest_date(self, df: Any) -> Any:
        """统一过滤 trade_date<=latest_date。"""
        if "trade_date" not in df.columns:
            return df

        if self.use_gpu:
            import cudf

            if not str(df["trade_date"].dtype).startswith("datetime64"):
                df["trade_date"] = cudf.to_datetime(df["trade_date"])
            cutoff = cudf.to_datetime(self.latest_date)
            return df[df["trade_date"] <= cutoff].reset_index(drop=True)

        trade_date = pd.to_datetime(df["trade_date"], errors="coerce")
        df["trade_date"] = trade_date
        return df[df["trade_date"] <= self.latest_date_ts].reset_index(drop=True)

    def _compute_gru_start_date(self, df: Any) -> Optional[pd.Timestamp]:
        """计算最近 gru_trade_days 交易日窗口的起始日。"""
        if "trade_date" not in df.columns or len(df) == 0:
            return None

        if self.use_gpu and hasattr(df["trade_date"], "to_pandas"):
            date_series = df["trade_date"].to_pandas()
        else:
            date_series = pd.Series(df["trade_date"])

        dates = pd.to_datetime(date_series, errors="coerce").dropna().dt.normalize()
        if dates.empty:
            return None

        unique_dates = sorted(dates.unique())
        start_idx = max(0, len(unique_dates) - self.gru_trade_days)
        return pd.Timestamp(unique_dates[start_idx])

    def _cleanup_memory(self) -> None:
        gc.collect()
        if self.use_gpu:
            try:
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def _align_saved_predict_schema(
        self,
        predict_path: Path,
        baseline_path: Path,
        tag: str,
    ) -> Dict[str, Any]:
        """
        将增量产物 schema 对齐到既有全量产物（仅字段顺序/类型）。

        说明：
        - 仅作用于增量输出文件，不改全量流水线逻辑
        - 若 baseline 不存在，自动跳过
        """
        stats: Dict[str, Any] = {
            "applied": False,
            "baseline_exists": baseline_path.exists(),
            "predict_exists": predict_path.exists(),
            "baseline": str(baseline_path),
            "predict": str(predict_path),
        }

        if not baseline_path.exists() or not predict_path.exists():
            return stats

        try:
            import pandas as pd
            import pyarrow.parquet as pq

            baseline_schema = pq.read_schema(str(baseline_path))
            baseline_cols = [f.name for f in baseline_schema]
            baseline_types = {f.name: str(f.type) for f in baseline_schema}

            df = pd.read_parquet(str(predict_path))
            if len(df.columns) == 0:
                return stats

            # 仅保留 baseline 列并按顺序重排
            missing_cols = [c for c in baseline_cols if c not in df.columns]
            extra_cols = [c for c in df.columns if c not in set(baseline_cols)]
            stats["missing_cols"] = missing_cols
            stats["extra_cols"] = extra_cols

            if missing_cols:
                logger.warning("%s schema 对齐跳过：predict 缺失 baseline 列 %s", tag, missing_cols[:10])
                return stats

            df = df[baseline_cols]

            cast_count = 0
            for col in baseline_cols:
                expected = baseline_types[col]
                try:
                    if expected in {"float", "double"}:
                        target_dtype = "float32" if expected == "float" else "float64"
                        new_series = pd.to_numeric(df[col], errors="coerce").astype(target_dtype)
                        if str(df[col].dtype) != str(new_series.dtype):
                            cast_count += 1
                        df[col] = new_series
                    elif expected in {"int8", "int16", "int32", "int64"}:
                        new_series = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(expected)
                        if str(df[col].dtype) != str(new_series.dtype):
                            cast_count += 1
                        df[col] = new_series
                    elif expected == "bool":
                        if str(df[col].dtype) != "bool":
                            num = pd.to_numeric(df[col], errors="coerce").fillna(0)
                            df[col] = num.astype("int8").astype(bool)
                            cast_count += 1
                    elif expected.startswith("timestamp"):
                        if not str(df[col].dtype).startswith("datetime64"):
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                            cast_count += 1
                    elif expected.startswith("date"):
                        if not str(df[col].dtype).startswith("datetime64"):
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                            cast_count += 1
                    # string/decimal/list 等类型保持原样
                except Exception as e:
                    logger.warning("%s schema 对齐列失败: %s (%s)", tag, col, e)

            df.to_parquet(str(predict_path), index=False, engine="pyarrow")

            new_schema = pq.read_schema(str(predict_path))
            new_types = {f.name: str(f.type) for f in new_schema}
            mismatch_after = [c for c in baseline_cols if new_types.get(c) != baseline_types.get(c)]

            stats.update(
                {
                    "applied": True,
                    "cast_count": cast_count,
                    "mismatch_after": len(mismatch_after),
                    "mismatch_after_sample": mismatch_after[:20],
                }
            )

            logger.info(
                "%s schema 对齐完成: cast=%s, mismatch_after=%s",
                tag,
                cast_count,
                len(mismatch_after),
            )
            return stats
        except Exception as e:
            logger.warning("%s schema 对齐失败: %s", tag, e)
            stats["error"] = str(e)
            return stats
