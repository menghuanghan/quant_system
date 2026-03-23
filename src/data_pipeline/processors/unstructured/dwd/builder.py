"""
非结构化 DWD 构建器

核心逻辑：
1. 以 dwd_stock_price (trade_date, ts_code) 作为最终骨架
2. 非结构化数据先做 date + 1天（防未来泄露）
3. 将“自然日”映射到“下一个有效交易日”
4. 对齐并聚合到 (trade_date, ts_code)
    - 个股特征（ann/events/ex/reports）: 同键 score 求和
    - 宏观新闻（cctv）: 同日 market_sentiment/beta_signal 求均值后广播
    - 政策特征（gov/ndrc）: 同日同业求和后按行业广播
5. 生成 data/processed/unstructured/dwd_unstructured.parquet
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "ann_score",
    "events_score",
    "ex_score",
    "reports_score",
    "market_sentiment",
    "beta_signal",
    "gov_score",
    "ndrc_score",
]


@dataclass
class UnstructuredDWDConfig:
    """非结构化 DWD 构建配置"""

    unstructured_processed_dir: str = "data/processed/unstructured"
    structured_dwd_dir: str = "data/processed/structured/dwd"
    output_file: str = "data/processed/unstructured/dwd_unstructured.parquet"
    compression: str = "snappy"


class UnstructuredDWDBuilder:
    """构建非结构化 DWD 宽表"""

    def __init__(self, config: Optional[UnstructuredDWDConfig] = None):
        self.config = config or UnstructuredDWDConfig()

        self.unstructured_processed_dir = Path(self.config.unstructured_processed_dir)
        self.structured_dwd_dir = Path(self.config.structured_dwd_dir)
        self.output_path = Path(self.config.output_file)

        self.price_path = self.structured_dwd_dir / "dwd_stock_price.parquet"
        self.industry_path = self.structured_dwd_dir / "dwd_stock_industry.parquet"

        self._calendar_map: Dict[pd.Timestamp, str] = {}
        self._price_code_suffix_map: Dict[str, set[str]] = {}
        self._normalized_ts_cache: Dict[str, Optional[str]] = {}

    def run(self) -> pd.DataFrame:
        """执行构建流程并写出 parquet"""
        logger.info("=" * 80)
        logger.info("开始构建非结构化 DWD 宽表")
        logger.info(f"非结构化输入目录: {self.unstructured_processed_dir}")
        logger.info(f"结构化 DWD 目录: {self.structured_dwd_dir}")
        logger.info(f"输出文件: {self.output_path}")
        logger.info("=" * 80)

        base_keys = self._load_price_skeleton_keys()
        industry_ref = self._load_industry_reference()

        ann_df = self._build_stock_score_feature(
            source_file="announcements.parquet",
            feature_name="ann_score",
        )
        events_df = self._build_stock_score_feature(
            source_file="events.parquet",
            feature_name="events_score",
        )
        ex_df = self._build_stock_score_feature(
            source_file="news_exchange.parquet",
            feature_name="ex_score",
        )
        reports_df = self._build_stock_score_feature(
            source_file="reports.parquet",
            feature_name="reports_score",
        )

        cctv_df = self._build_cctv_feature(base_keys)

        gov_df = self._build_policy_feature(
            source_file="policy_gov.parquet",
            feature_name="gov_score",
            industry_ref=industry_ref,
        )
        ndrc_df = self._build_policy_feature(
            source_file="policy_ndrc.parquet",
            feature_name="ndrc_score",
            industry_ref=industry_ref,
        )

        result = base_keys.copy()
        for feature_df in [ann_df, events_df, ex_df, reports_df, cctv_df, gov_df, ndrc_df]:
            if feature_df.empty:
                continue
            result = result.merge(feature_df, on=["trade_date", "ts_code"], how="left")

        for col in FEATURE_COLUMNS:
            if col not in result.columns:
                result[col] = np.nan
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")

        result = result[["trade_date", "ts_code", *FEATURE_COLUMNS]]
        result = result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(self.output_path, index=False, compression=self.config.compression)

        non_null_stats = {c: int(result[c].notna().sum()) for c in FEATURE_COLUMNS}
        logger.info("非结构化 DWD 构建完成")
        logger.info(f"输出行数: {len(result):,}")
        logger.info(f"输出列数: {len(result.columns)}")
        logger.info(f"特征非空统计: {non_null_stats}")
        logger.info(f"已保存: {self.output_path}")

        return result

    def _load_price_skeleton_keys(self) -> pd.DataFrame:
        if not self.price_path.exists():
            raise FileNotFoundError(f"未找到骨架表: {self.price_path}")

        price_df = pd.read_parquet(self.price_path, columns=["trade_date", "ts_code"])
        base_keys = (
            price_df[["trade_date", "ts_code"]]
            .drop_duplicates()
            .sort_values(["trade_date", "ts_code"])
            .reset_index(drop=True)
        )

        trade_dates = pd.to_datetime(base_keys["trade_date"], errors="coerce")
        self._calendar_map = self._build_natural_to_trade_map(trade_dates.dropna())

        self._price_code_suffix_map = {}
        for code in base_keys["ts_code"].astype(str).str.upper().unique():
            m = re.match(r"^(\d{6})\.([A-Z]{2})$", code)
            if not m:
                continue
            code6, suffix = m.group(1), m.group(2)
            self._price_code_suffix_map.setdefault(code6, set()).add(suffix)

        logger.info(
            "骨架加载完成: %s 行, %s 个交易日, %s 个 ts_code",
            f"{len(base_keys):,}",
            f"{base_keys['trade_date'].nunique():,}",
            f"{base_keys['ts_code'].nunique():,}",
        )
        return base_keys

    def _load_industry_reference(self) -> pd.DataFrame:
        if not self.industry_path.exists():
            raise FileNotFoundError(f"未找到行业表: {self.industry_path}")

        industry_df = pd.read_parquet(
            self.industry_path,
            columns=["trade_date", "ts_code", "sw_l1_name"],
        )

        industry_df["trade_date"] = industry_df["trade_date"].astype(str)
        industry_df["ts_code"] = industry_df["ts_code"].astype(str).str.upper()
        industry_df["sw_l1_name"] = industry_df["sw_l1_name"].astype(str).str.strip()

        industry_df = (
            industry_df[["trade_date", "ts_code", "sw_l1_name"]]
            .dropna(subset=["trade_date", "ts_code", "sw_l1_name"])
            .drop_duplicates()
        )

        logger.info(
            "行业参考表加载完成: %s 行, %s 个行业",
            f"{len(industry_df):,}",
            f"{industry_df['sw_l1_name'].nunique():,}",
        )
        return industry_df

    def _build_stock_score_feature(self, source_file: str, feature_name: str) -> pd.DataFrame:
        source_path = self.unstructured_processed_dir / source_file
        if not source_path.exists():
            logger.warning("缺少输入文件，跳过 %s: %s", feature_name, source_path)
            return pd.DataFrame(columns=["trade_date", "ts_code", feature_name])

        df = pd.read_parquet(source_path, columns=["date", "ts_code", "score"])
        if df.empty:
            logger.warning("输入文件为空，跳过 %s: %s", feature_name, source_path)
            return pd.DataFrame(columns=["trade_date", "ts_code", feature_name])

        df["trade_date"] = self._map_to_trade_date_after_shift(df["date"])
        df["ts_code"] = self._normalize_ts_code_series(df["ts_code"])
        df[feature_name] = pd.to_numeric(df["score"], errors="coerce")

        df = df.dropna(subset=["trade_date", "ts_code", feature_name])
        if df.empty:
            return pd.DataFrame(columns=["trade_date", "ts_code", feature_name])

        # 个股事件特征采用“情感净值求和法（动量累加）”
        out = (
            df.groupby(["trade_date", "ts_code"], as_index=False)[feature_name]
            .sum()
            .astype({feature_name: "float32"})
        )

        logger.info("%s 构建完成: %s 行", feature_name, f"{len(out):,}")
        return out

    def _build_cctv_feature(self, base_keys: pd.DataFrame) -> pd.DataFrame:
        source_path = self.unstructured_processed_dir / "news_cctv.parquet"
        if not source_path.exists():
            logger.warning("缺少输入文件，跳过 market_sentiment/beta_signal: %s", source_path)
            return pd.DataFrame(columns=["trade_date", "ts_code", "market_sentiment", "beta_signal"])

        df = pd.read_parquet(source_path, columns=["date", "market_sentiment", "beta_signal"])
        if df.empty:
            return pd.DataFrame(columns=["trade_date", "ts_code", "market_sentiment", "beta_signal"])

        df["trade_date"] = self._map_to_trade_date_after_shift(df["date"])
        df["market_sentiment"] = pd.to_numeric(df["market_sentiment"], errors="coerce")
        df["beta_signal"] = pd.to_numeric(df["beta_signal"], errors="coerce")

        daily = (
            df.dropna(subset=["trade_date"])  # score 可空，由 mean 忽略
            .groupby("trade_date", as_index=False)[["market_sentiment", "beta_signal"]]
            .mean()
        )

        if daily.empty:
            return pd.DataFrame(columns=["trade_date", "ts_code", "market_sentiment", "beta_signal"])

        daily["market_sentiment"] = daily["market_sentiment"].astype("float32")
        daily["beta_signal"] = daily["beta_signal"].astype("float32")

        out = base_keys[["trade_date", "ts_code"]].merge(daily, on="trade_date", how="left")
        logger.info("market_sentiment/beta_signal 构建完成: %s 行", f"{len(out):,}")
        return out

    def _build_policy_feature(
        self,
        source_file: str,
        feature_name: str,
        industry_ref: pd.DataFrame,
    ) -> pd.DataFrame:
        source_path = self.unstructured_processed_dir / source_file
        if not source_path.exists():
            logger.warning("缺少输入文件，跳过 %s: %s", feature_name, source_path)
            return pd.DataFrame(columns=["trade_date", "ts_code", feature_name])

        df = pd.read_parquet(source_path, columns=["date", "industry_scores"])
        if df.empty:
            return pd.DataFrame(columns=["trade_date", "ts_code", feature_name])

        df["trade_date"] = self._map_to_trade_date_after_shift(df["date"])
        df = df.dropna(subset=["trade_date", "industry_scores"])
        if df.empty:
            return pd.DataFrame(columns=["trade_date", "ts_code", feature_name])

        records = []
        for trade_date, payload in zip(df["trade_date"], df["industry_scores"]):
            scores = self._parse_industry_scores(payload)
            if not scores:
                continue
            for industry_name, score in scores.items():
                records.append((trade_date, industry_name, score))

        if not records:
            return pd.DataFrame(columns=["trade_date", "ts_code", feature_name])

        industry_scores_df = pd.DataFrame(
            records,
            columns=["trade_date", "sw_l1_name", feature_name],
        )
        industry_scores_df["sw_l1_name"] = industry_scores_df["sw_l1_name"].astype(str).str.strip()
        industry_scores_df[feature_name] = pd.to_numeric(industry_scores_df[feature_name], errors="coerce")

        # 同日同一行业多条政策，先按行业做净值求和（动量累加）
        industry_scores_df = (
            industry_scores_df
            .dropna(subset=["trade_date", "sw_l1_name", feature_name])
            .groupby(["trade_date", "sw_l1_name"], as_index=False)[feature_name]
            .sum()
        )

        merged = industry_ref.merge(
            industry_scores_df,
            on=["trade_date", "sw_l1_name"],
            how="inner",
        )

        if merged.empty:
            return pd.DataFrame(columns=["trade_date", "ts_code", feature_name])

        # 广播到个股后，若同日同股仍有重复映射，继续做净值求和
        out = (
            merged.groupby(["trade_date", "ts_code"], as_index=False)[feature_name]
            .sum()
            .astype({feature_name: "float32"})
        )

        logger.info("%s 构建完成: %s 行", feature_name, f"{len(out):,}")
        return out

    def _map_to_trade_date_after_shift(self, date_series: pd.Series) -> pd.Series:
        date_ts = pd.to_datetime(date_series, errors="coerce").dt.normalize()
        shifted = date_ts + pd.Timedelta(days=1)
        return shifted.map(self._calendar_map)

    def _normalize_ts_code_series(self, code_series: pd.Series) -> pd.Series:
        return code_series.map(self._normalize_single_ts_code)

    def _normalize_single_ts_code(self, raw_code: object) -> Optional[str]:
        if pd.isna(raw_code):
            return None

        text = str(raw_code).strip().upper()
        if not text:
            return None

        cached = self._normalized_ts_cache.get(text)
        if text in self._normalized_ts_cache:
            return cached

        normalized: Optional[str] = None

        # 标准格式：000001.SZ
        m = re.match(r"^(\d{6})\.([A-Z]{2})$", text)
        if m:
            normalized = f"{m.group(1)}.{m.group(2)}"
        else:
            # 提取6位数字（兼容 SH600000 / SZ000001 / 600000）
            digits_match = re.search(r"(\d{6})", text)
            if digits_match:
                code6 = digits_match.group(1)
                suffix = self._infer_ts_suffix(code6)
                normalized = f"{code6}.{suffix}" if suffix else None

        self._normalized_ts_cache[text] = normalized
        return normalized

    def _infer_ts_suffix(self, code6: str) -> Optional[str]:
        candidate_suffix = self._price_code_suffix_map.get(code6, set())
        if len(candidate_suffix) == 1:
            return next(iter(candidate_suffix))

        # 根据A股常见编码规则兜底
        if code6.startswith(("5", "6", "9")):
            return "SH"
        if code6.startswith(("0", "1", "2", "3")):
            return "SZ"
        if code6.startswith(("4", "8")):
            return "BJ"
        return None

    @staticmethod
    def _parse_industry_scores(payload: object) -> Dict[str, float]:
        if payload is None or (isinstance(payload, float) and np.isnan(payload)):
            return {}

        data = None
        if isinstance(payload, dict):
            data = payload
        elif isinstance(payload, str):
            text = payload.strip()
            if not text:
                return {}
            try:
                data = json.loads(text)
            except Exception:
                try:
                    data = ast.literal_eval(text)
                except Exception:
                    logger.debug("industry_scores 解析失败: %s", text[:200])
                    return {}
        else:
            return {}

        if not isinstance(data, dict):
            return {}

        parsed: Dict[str, float] = {}
        for k, v in data.items():
            if k is None:
                continue
            try:
                parsed[str(k).strip()] = float(v)
            except Exception:
                continue
        return parsed

    @staticmethod
    def _build_natural_to_trade_map(trade_dates: Iterable[pd.Timestamp]) -> Dict[pd.Timestamp, str]:
        trade_idx = pd.DatetimeIndex(pd.to_datetime(list(trade_dates), errors="coerce")).dropna().unique()
        trade_idx = trade_idx.sort_values()
        if len(trade_idx) == 0:
            raise ValueError("交易日序列为空，无法构建自然日映射")

        natural_start = trade_idx.min().to_period("M").to_timestamp(how="start")
        natural_end = trade_idx.max().to_period("M").to_timestamp(how="end")
        natural_idx = pd.date_range(natural_start, natural_end, freq="D")

        pos = np.searchsorted(trade_idx.values, natural_idx.values, side="left")
        valid = pos < len(trade_idx)

        mapped = pd.Series(pd.NaT, index=natural_idx, dtype="datetime64[ns]")
        mapped.loc[valid] = trade_idx.values[pos[valid]]

        out: Dict[pd.Timestamp, str] = {}
        for natural_date, target_trade in mapped.items():
            if pd.isna(target_trade):
                continue
            out[pd.Timestamp(natural_date).normalize()] = pd.Timestamp(target_trade).strftime("%Y-%m-%d")

        logger.info(
            "自然日映射构建完成: 交易日 %s 天, 自然日 %s 天",
            f"{len(trade_idx):,}",
            f"{len(out):,}",
        )
        return out


def build_dwd_unstructured(config: Optional[UnstructuredDWDConfig] = None) -> pd.DataFrame:
    """便捷函数：构建非结构化 DWD 宽表"""
    builder = UnstructuredDWDBuilder(config=config)
    return builder.run()
