"""
GRU 多任务模型训练报告生成器

自动生成详细的训练结果报告（Markdown 格式），包括：
- 训练配置汇总（数据/切分/网络/训练超参）
- 各标签评估指标对比
- 各 Fold / 各种子训练详情（含损失曲线、早停信息）
- 结论与建议
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..config import GRUConfig
from ..metrics.evaluator import QuantEvaluator

logger = logging.getLogger(__name__)


class GRUReportGenerator:
    """
    GRU 多任务模型训练报告生成器

    生成 Markdown 格式的训练报告，自动从 OOF 和 fold_train_info 中提取统计数据。

    Example:
        >>> gen = GRUReportGenerator(config, oof_df, fold_train_info)
        >>> gen.generate_report()
    """

    def __init__(
        self,
        config: GRUConfig,
        oof_df: pd.DataFrame,
        fold_train_info: List[Dict[str, Any]],
        report_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            config: GRU 完整配置
            oof_df: OOF 预测 DataFrame（包含 trade_date, ts_code, fold,
                    y_true_{col}, y_pred_{col}, y_pred, y_true）
            fold_train_info: 每个 fold/seed 的训练信息列表，每项包含:
                - fold_idx / seed (int)
                - train_start, train_end, valid_start, valid_end (str)
                - train_samples, valid_samples (int)
                - best_epoch (int)
                - epochs_trained (int)
                - best_rank_ic (float)
                - train_time_s (float)
                - history: {train_loss: [...], valid_loss: [...], valid_rank_ic: [...]}
            report_dir: 报告保存目录
        """
        self.config = config
        self.oof_df = oof_df
        self.fold_train_info = fold_train_info
        self.report_dir = Path(report_dir) if report_dir else Path("reports")
        self.evaluator = QuantEvaluator()

        # 缓存评估结果：{target_col: metrics_dict}
        self._eval_cache: Dict[str, Dict[str, Any]] = {}

    # ================================================================
    # 内部工具
    # ================================================================

    def _evaluate_target(self, target_col: str) -> Dict[str, Any]:
        """评估单个目标标签，带缓存"""
        if target_col in self._eval_cache:
            return self._eval_cache[target_col]

        pred_col = f"y_pred_{target_col}"
        true_col = f"y_true_{target_col}"

        if pred_col not in self.oof_df.columns or true_col not in self.oof_df.columns:
            return {}

        metrics = self.evaluator.evaluate(
            self.oof_df,
            y_pred_col=pred_col,
            y_true_col=true_col,
        )
        self._eval_cache[target_col] = metrics
        return metrics

    def _evaluate_all(self) -> None:
        """评估所有目标"""
        for col in self.config.data.target_cols:
            self._evaluate_target(col)

    @staticmethod
    def _mode_label(mode: str) -> str:
        return {"rolling": "滚动窗口", "expanding": "扩展窗口",
                "single_full": "单次全量（多种子融合）"}.get(mode, mode)

    @staticmethod
    def _sig_mark(p: float) -> str:
        if np.isnan(p):
            return "-"
        if p < 0.001:
            return "★★★"
        if p < 0.01:
            return "★★"
        if p < 0.05:
            return "★"
        return "-"

    @staticmethod
    def _mono_desc(m: float) -> str:
        if np.isnan(m):
            return "N/A"
        if m > 0.9:
            return "完美单调"
        if m > 0.7:
            return "强单调"
        if m > 0.5:
            return "中等单调"
        if m > 0:
            return "弱正相关"
        if m > -0.5:
            return "无显著单调性"
        return "负相关"

    # ================================================================
    # 报告段落生成
    # ================================================================

    def _header(self) -> str:
        now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        mode = self.config.split.mode
        targets = ", ".join(self.config.data.target_cols)

        # OOF 日期范围
        if "trade_date" in self.oof_df.columns and not self.oof_df.empty:
            dates = pd.to_datetime(self.oof_df["trade_date"])
            date_range = f"{dates.min().strftime('%Y-%m-%d')} ~ {dates.max().strftime('%Y-%m-%d')}"
        else:
            date_range = "N/A"

        return (
            f"# GRU 多任务模型训练报告\n\n"
            f"**生成时间**: {now}  \n"
            f"**训练模式**: {mode.capitalize()}（{self._mode_label(mode)}）  \n"
            f"**训练标签**: {targets}  \n"
            f"**验证数据范围**: {date_range}\n\n---\n\n"
        )

    def _config_section(self) -> str:
        """配置信息"""
        sc = self.config.split
        dc = self.config.data
        nc = self.config.network
        tc = self.config.train
        mode = sc.mode

        n_folds = len(self.fold_train_info) if self.fold_train_info else 0

        s = "## 1. 训练配置\n\n"

        # 1.1 时序切分
        s += "### 1.1 时序切分参数\n\n"
        s += "| 参数 | 值 |\n| ---- | ---- |\n"
        s += f"| 训练模式 | {mode} |\n"
        if mode != "single_full":
            s += f"| 训练窗口 | {sc.train_window_months} 个月 |\n"
            s += f"| 验证窗口 | {sc.valid_window_months} 个月 |\n"
            s += f"| 滚动步长 | {sc.step_months} 个月 |\n"
        else:
            s += f"| 多种子数量 | {len(tc.multi_seeds)} (seeds={tc.multi_seeds}) |\n"
        s += f"| Fold 数量 | {n_folds} |\n"
        s += f"| 数据逻辑起始 | {sc.data_start_date} |\n"
        s += f"| 数据逻辑截止 | {sc.data_end_date} |\n"
        s += f"| 序列长度 (seq_len) | {dc.seq_len} |\n\n"

        # 1.2 网络结构
        s += "### 1.2 GRU 网络结构\n\n"
        s += "| 参数 | 值 | 说明 |\n| ---- | ---- | ---- |\n"
        s += f"| num_features | {nc.num_features} | 输入特征数 |\n"
        s += f"| hidden_size | {nc.hidden_size} | GRU 隐层维度 |\n"
        s += f"| num_layers | {nc.num_layers} | GRU 层数 |\n"
        s += f"| dropout | {nc.dropout} | Dropout 比例 |\n"
        s += f"| use_attention | {nc.use_attention} | 时间注意力 |\n"
        s += f"| num_targets | {nc.num_targets} | 多任务目标数 |\n\n"

        # 1.3 训练超参
        s += "### 1.3 训练超参数\n\n"
        s += "| 参数 | 值 | 说明 |\n| ---- | ---- | ---- |\n"
        s += f"| epochs | {tc.epochs} | 最大训练轮数 |\n"
        s += f"| batch_size | {tc.batch_size} | 批次大小 |\n"
        s += f"| learning_rate | {tc.learning_rate} | 基础学习率 |\n"
        s += f"| max_lr | {tc.max_lr} | OneCycleLR 最大学习率 |\n"
        s += f"| weight_decay | {tc.weight_decay} | L2 正则化 |\n"
        s += f"| patience | {tc.patience} | 早停耐心轮数 |\n"
        s += f"| min_epochs | {tc.min_epochs} | 最小训练轮数 |\n"
        s += f"| use_amp | {tc.use_amp} | 混合精度 |\n"

        # 损失权重
        s += "\n### 1.4 多任务损失配置\n\n"
        s += "| 目标 | 损失函数 | 权重 |\n| ---- | -------- | ---- |\n"
        for col in self.config.data.target_cols:
            lt = tc.loss_types.get(col, "mse")
            lw = tc.loss_weights.get(col, 0.0)
            s += f"| {col} | {lt} | {lw:.2f} |\n"

        s += "\n---\n\n"
        return s

    def _summary_section(self) -> str:
        """2. 结果汇总"""
        self._evaluate_all()
        target_cols = self.config.data.target_cols

        s = "## 2. 训练结果汇总\n\n"

        # 2.1 核心指标
        s += "### 2.1 核心指标对比\n\n"
        s += "| 标签 | IC Mean | IC Std | ICIR | RankIC Mean | RankIC Std | Rank ICIR |\n"
        s += "| ---- | ------- | ------ | ---- | ----------- | ---------- | --------- |\n"
        for col in target_cols:
            m = self._eval_cache.get(col, {})
            ic = m.get("ic_mean", np.nan)
            ics = m.get("ic_std", np.nan)
            icir = m.get("icir", np.nan)
            ric = m.get("rank_ic_mean", np.nan)
            rics = m.get("rank_ic_std", np.nan)
            ricir = m.get("rank_icir", np.nan)
            bold = abs(icir) > 0.5 if not np.isnan(icir) else False
            if bold:
                s += (f"| **{col}** | **{ic:.4f}** | **{ics:.4f}** | "
                      f"**{icir:.4f}** | **{ric:.4f}** | **{rics:.4f}** | "
                      f"**{ricir:.4f}** |\n")
            else:
                s += (f"| {col} | {ic:.4f} | {ics:.4f} | "
                      f"{icir:.4f} | {ric:.4f} | {rics:.4f} | "
                      f"{ricir:.4f} |\n")

        # 2.2 统计检验
        s += "\n### 2.2 统计检验\n\n"
        s += "| 标签 | IC>0 比例 | T统计量 | P值 | 显著性 |\n"
        s += "| ---- | --------- | ------- | --- | ------ |\n"
        for col in target_cols:
            m = self._eval_cache.get(col, {})
            ic_pos = m.get("ic_positive_ratio", np.nan)
            t_stat = m.get("t_stat", np.nan)
            p_val = m.get("p_value", np.nan)
            sig = self._sig_mark(p_val)
            bold = abs(m.get("icir", 0)) > 0.5
            if bold:
                s += (f"| **{col}** | **{ic_pos:.2%}** | "
                      f"**{t_stat:.4f}** | **{p_val:.4e}** | **{sig}** |\n")
            else:
                s += (f"| {col} | {ic_pos:.2%} | "
                      f"{t_stat:.4f} | {p_val:.4e} | {sig} |\n")

        # 2.3 多空收益
        s += "\n### 2.3 多空收益\n\n"
        s += "| 标签 | 日均多空收益 | 年化夏普 | 单调性 | 描述 |\n"
        s += "| ---- | ----------- | -------- | ------ | ---- |\n"
        for col in target_cols:
            m = self._eval_cache.get(col, {})
            avg_ls = m.get("avg_long_short_return", np.nan)
            sharpe = m.get("sharpe_long_short", np.nan)
            mono = m.get("monotonicity", np.nan)
            desc = self._mono_desc(mono)
            bold = abs(m.get("icir", 0)) > 0.5
            if bold:
                s += (f"| **{col}** | **{avg_ls:.4%}** | "
                      f"**{sharpe:.4f}** | **{mono:.4f}** | {desc} |\n")
            else:
                s += (f"| {col} | {avg_ls:.4%} | "
                      f"{sharpe:.4f} | {mono:.4f} | {desc} |\n")

        # 2.4 综合信号（主信号，rank 通道）
        if "y_pred" in self.oof_df.columns and "y_true" in self.oof_df.columns:
            main_m = self.evaluator.evaluate(
                self.oof_df, y_pred_col="y_pred", y_true_col="y_true",
            )
            s += "\n### 2.4 综合信号（主信号 rank 通道）\n\n"
            s += "| 指标 | 值 |\n| ---- | ---- |\n"
            s += f"| IC Mean | {main_m.get('ic_mean', np.nan):.4f} |\n"
            s += f"| ICIR | {main_m.get('icir', np.nan):.4f} |\n"
            s += f"| Rank IC Mean | {main_m.get('rank_ic_mean', np.nan):.4f} |\n"
            s += f"| Rank ICIR | {main_m.get('rank_icir', np.nan):.4f} |\n"
            s += f"| IC>0 比例 | {main_m.get('ic_positive_ratio', np.nan):.2%} |\n"
            s += f"| 单调性 | {main_m.get('monotonicity', np.nan):.4f} |\n"
            s += f"| 年化夏普 | {main_m.get('sharpe_long_short', np.nan):.4f} |\n"

        s += "\n---\n\n"
        return s

    def _fold_detail_section(self) -> str:
        """3. 各 Fold / 各种子训练详情"""
        mode = self.config.split.mode
        infos = self.fold_train_info
        if not infos:
            return ""

        if mode == "single_full":
            title = "各种子训练详情"
            idx_label = "Seed"
        else:
            title = "各 Fold 训练详情"
            idx_label = "Fold"

        s = f"## 3. {title}\n\n"

        # 概览表
        s += f"### 3.1 {title}概览\n\n"
        s += (f"| {idx_label} | 训练区间 | 验证区间 | 训练样本 | "
              f"验证样本 | 实训轮数 | 最佳轮 | Best RankIC | 耗时 |\n")
        s += "| ---- " * 9 + "|\n"

        total_time = 0.0
        for info in infos:
            idx = info.get("seed", info.get("fold_idx", "?"))
            ts = info.get("train_start", "?")
            te = info.get("train_end", "?")
            vs = info.get("valid_start", "?")
            ve = info.get("valid_end", "?")
            n_train = info.get("train_samples", 0)
            n_valid = info.get("valid_samples", 0)
            epochs = info.get("epochs_trained", 0)
            best_ep = info.get("best_epoch", "?")
            best_ic = info.get("best_rank_ic", np.nan)
            t_sec = info.get("train_time_s", 0.0)
            total_time += t_sec

            s += (f"| {idx} | {ts}~{te} | {vs}~{ve} | "
                  f"{n_train:,} | {n_valid:,} | {epochs} | "
                  f"{best_ep} | {best_ic:.4f} | {t_sec:.0f}s |\n")

        s += f"\n**总训练耗时**: {total_time:.0f}s ({total_time / 60:.1f} min)\n\n"

        # 3.2 各 fold 损失与 IC 变化
        s += f"### 3.2 训练曲线摘要\n\n"
        for info in infos:
            idx = info.get("seed", info.get("fold_idx", "?"))
            hist = info.get("history", {})
            if not hist:
                continue

            train_losses = hist.get("train_loss", [])
            valid_losses = hist.get("valid_loss", [])
            rank_ics = hist.get("valid_rank_ic", [])

            if not train_losses:
                continue

            best_ep = info.get("best_epoch", "?")
            s += f"**{idx_label} {idx}** (best epoch={best_ep})\n\n"
            s += "| Epoch | TrainLoss | ValidLoss | RankIC |\n"
            s += "| ----- | --------- | --------- | ------ |\n"

            # 显示关键节点：第1、每10轮、最佳轮、最后轮
            n_epochs = len(train_losses)
            show_indices = {0, n_epochs - 1}
            for e in range(9, n_epochs, 10):
                show_indices.add(e)
            if isinstance(best_ep, int) and 0 <= best_ep - 1 < n_epochs:
                show_indices.add(best_ep - 1)

            for e in sorted(show_indices):
                tl = train_losses[e] if e < len(train_losses) else np.nan
                vl = valid_losses[e] if e < len(valid_losses) else np.nan
                ric = rank_ics[e] if e < len(rank_ics) else np.nan
                marker = " ◀ best" if isinstance(best_ep, int) and e == best_ep - 1 else ""
                s += f"| {e + 1} | {tl:.4f} | {vl:.4f} | {ric:.4f}{marker} |\n"

            s += "\n"

        s += "---\n\n"
        return s

    def _target_detail_section(self) -> str:
        """4. 各标签详细评估"""
        self._evaluate_all()
        target_cols = self.config.data.target_cols

        s = "## 4. 各标签详细评估\n\n"

        for col in target_cols:
            m = self._eval_cache.get(col, {})
            icir = m.get("icir", 0)
            is_valid = abs(icir) > 0.5
            status = "✅ 有效因子" if is_valid else "❌ 无效因子"

            s += f"### {col}（{status}）\n\n"
            s += "```\n"
            s += f"============================================================\n"
            s += f"Factor Evaluation Report: {col}\n"
            s += f"============================================================\n\n"
            s += f"[IC Analysis]\n"
            s += f"  IC Mean:            {m.get('ic_mean', np.nan):.4f}\n"
            s += f"  IC Std:             {m.get('ic_std', np.nan):.4f}\n"
            s += f"  ICIR:               {m.get('icir', np.nan):.4f}\n"
            s += f"  Rank IC Mean:       {m.get('rank_ic_mean', np.nan):.4f}\n"
            s += f"  Rank IC Std:        {m.get('rank_ic_std', np.nan):.4f}\n"
            s += f"  Rank ICIR:          {m.get('rank_icir', np.nan):.4f}\n\n"
            s += f"[Statistical Tests]\n"
            s += f"  IC Positive Ratio:  {m.get('ic_positive_ratio', np.nan):.2%}\n"
            s += f"  T-Statistic:        {m.get('t_stat', np.nan):.4f}\n"
            s += f"  P-Value:            {m.get('p_value', np.nan):.4e}\n\n"
            s += f"[Long-Short Returns]\n"
            s += f"  Avg Daily Return:   {m.get('avg_long_short_return', np.nan):.4%}\n"
            s += f"  Sharpe Ratio:       {m.get('sharpe_long_short', np.nan):.4f}\n\n"
            s += f"[Monotonicity]\n"
            s += f"  Group Monotonicity: {m.get('monotonicity', np.nan):.4f}\n"
            s += f"============================================================\n"
            s += "```\n\n"

        s += "---\n\n"
        return s

    def _conclusion_section(self) -> str:
        """结论与建议"""
        self._evaluate_all()
        valid_targets = []
        invalid_targets = []

        for col in self.config.data.target_cols:
            m = self._eval_cache.get(col, {})
            icir = m.get("icir", 0)
            if abs(icir) > 0.5:
                valid_targets.append((col, m))
            else:
                invalid_targets.append((col, m))

        s = "## 5. 结论与建议\n\n"

        # 有效标签
        s += "### 有效标签\n\n"
        if valid_targets:
            for col, m in valid_targets:
                icir = m.get("icir", 0)
                ic_pos = m.get("ic_positive_ratio", 0)
                mono = m.get("monotonicity", 0)
                sharpe = m.get("sharpe_long_short", 0)
                s += (f"✅ **{col}**: ICIR={icir:.4f}"
                      f"（{'显著有效' if icir > 1.0 else '有效'}）, "
                      f"IC>0 比例={ic_pos:.1%}, "
                      f"单调性={mono:.4f}, 夏普={sharpe:.4f}\n\n")
        else:
            s += "无有效标签。\n\n"

        # 无效标签
        s += "### 无效标签\n\n"
        if invalid_targets:
            for col, m in invalid_targets:
                icir = m.get("icir", 0)
                s += f"❌ **{col}**: ICIR={icir:.4f}，模型无法有效预测\n\n"
        else:
            s += "无。\n\n"

        # 建议
        s += "### 后续建议\n\n"
        if valid_targets:
            s += ("1. **模型集成**: 将 Rolling 与 Single_Full 版本加权融合"
                  "（推荐 0.6:0.4）\n")
            s += "2. **扩展周期**: 尝试 10d、20d 等更长预测周期\n"
            s += ("3. **网络调优**: 可尝试增大 hidden_size 或增加 num_layers，"
                  "注意 OOM 风险\n")
        if invalid_targets:
            names = ", ".join(t[0] for t in invalid_targets)
            s += (f"4. **标签改进**: 对 {names} 考虑秩变换、"
                  "行业中性化或更长训练窗口\n")

        s += "\n---\n\n"
        return s

    def _model_files_section(self) -> str:
        """模型文件列表"""
        mode = self.config.split.mode
        model_dir = self.config.train.save_dir / mode

        s = "## 6. 模型产出\n\n"
        s += f"**模型保存目录**: `{model_dir}`\n\n"

        if model_dir.exists():
            pth_files = sorted(model_dir.glob("*.pth"))
            if pth_files:
                s += "```\n"
                for f in pth_files[:20]:
                    s += f"{f.name}\n"
                if len(pth_files) > 20:
                    s += f"... (共 {len(pth_files)} 个模型文件)\n"
                s += "```\n\n"

        oof_path = model_dir / "oof_predictions.parquet"
        s += f"**OOF 预测文件**: `{oof_path}`\n"
        return s

    # ================================================================
    # 公共接口
    # ================================================================

    def generate_report(self, filename: Optional[str] = None) -> Path:
        """
        生成完整训练报告

        Args:
            filename: 报告文件名（默认自动生成）

        Returns:
            report_path: 生成的报告路径
        """
        logger.info("生成 GRU 训练报告...")

        content = self._header()
        content += self._config_section()
        content += self._summary_section()
        content += self._fold_detail_section()
        content += self._target_detail_section()
        # content += self._conclusion_section()
        # content += self._model_files_section()

        # 保存
        self.report_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = self.config.split.mode
            filename = f"gru_training_report_{mode}_{ts}.md"

        report_path = self.report_dir / filename
        report_path.write_text(content, encoding="utf-8")
        logger.info(f"GRU 训练报告已保存: {report_path}")
        return report_path
