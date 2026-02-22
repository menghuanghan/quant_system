"""
LightGBM 训练报告生成器

自动生成详细的训练结果报告，包括：
- 训练配置汇总
- 各标签评估指标对比
- 各 Fold 训练详情
- 特征重要性分析
- 结论与建议
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..config import (
    FeatureConfig,
    LGBConfig,
    SplitConfig,
    SplitMode,
    TrainConfig,
)
from ..metrics.evaluator import QuantEvaluator

logger = logging.getLogger(__name__)


class TrainingReportGenerator:
    """
    LightGBM 训练报告生成器
    
    生成 Markdown 格式的详细训练报告
    
    Example:
        >>> generator = TrainingReportGenerator(config, oof_dict, feature_importance, fold_info_list)
        >>> generator.generate_report()
    """
    
    def __init__(
        self,
        config: TrainConfig,
        oof_dict: Dict[str, pd.DataFrame],
        feature_importance: Dict[str, pd.DataFrame],
        fold_info_dict: Dict[str, List[Dict[str, Any]]],
        model_train_info: Dict[str, List[Dict[str, Any]]],
        report_dir: Optional[Union[str, Path]] = None,
    ):
        """
        初始化报告生成器
        
        Args:
            config: 训练配置
            oof_dict: {target_col: oof_df} OOF 预测结果
            feature_importance: {target_col: importance_df} 特征重要性
            fold_info_dict: {target_col: [fold_info]} 各 Fold 的信息
            model_train_info: {target_col: [train_info]} 各模型的训练信息
            report_dir: 报告保存目录
        """
        self.config = config
        self.oof_dict = oof_dict
        self.feature_importance = feature_importance
        self.fold_info_dict = fold_info_dict
        self.model_train_info = model_train_info
        self.report_dir = Path(report_dir) if report_dir else Path("reports")
        
        # 评估器
        self.evaluator = QuantEvaluator()
        
        # 评估结果缓存
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        
    def _evaluate_all_targets(self) -> None:
        """评估所有标签"""
        for target_col, oof_df in self.oof_dict.items():
            metrics = self.evaluator.evaluate(
                oof_df,
                y_pred_col="y_pred",
                y_true_col="y_true",
            )
            self.evaluation_results[target_col] = metrics
    
    def _generate_header(self) -> str:
        """生成报告头部"""
        now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        mode = self.config.split_config.mode.value
        
        # 获取数据时间范围（从 OOF 中推断）
        all_dates = []
        for oof_df in self.oof_dict.values():
            all_dates.extend(oof_df["trade_date"].tolist())
        
        if all_dates:
            min_date = pd.Timestamp(min(all_dates)).strftime("%Y-%m-%d")
            max_date = pd.Timestamp(max(all_dates)).strftime("%Y-%m-%d")
            date_range = f"{min_date} ~ {max_date}"
        else:
            date_range = "N/A"
        
        targets = ", ".join(self.oof_dict.keys())
        
        return f"""# LightGBM 模型训练报告

**生成时间**: {now}  
**训练模式**: {mode.capitalize()}（{self._get_mode_description(mode)}）  
**训练标签**: {targets}  
**验证数据范围**: {date_range}

---
"""
    
    def _get_mode_description(self, mode: str) -> str:
        """获取模式描述"""
        descriptions = {
            "rolling": "滚动窗口",
            "expanding": "扩展窗口",
            "single_full": "单次全量",
        }
        return descriptions.get(mode, mode)
    
    def _generate_config_section(self) -> str:
        """生成配置部分"""
        split_cfg = self.config.split_config
        lgb_cfg = self.config.lgb_config
        feature_cfg = self.config.feature_config
        
        # 计算 Fold 数量
        n_folds = len(next(iter(self.fold_info_dict.values()), []))
        
        content = """## 1. 训练配置

### 1.1 时序切分参数

| 参数 | 值 |
| ---- | ---- |
"""
        content += f"| 训练窗口 | {split_cfg.train_window_months} 个月 |\n"
        content += f"| 验证窗口 | {split_cfg.valid_window_months} 个月 |\n"
        content += f"| 滚动步长 | {split_cfg.step_months} 个月 |\n"
        content += f"| Fold 数量 | {n_folds} |\n"
        content += f"| 数据起始 | {split_cfg.data_start_date} |\n"
        content += f"| 数据截止 | {split_cfg.data_end_date} |\n"
        
        content += """
### 1.2 LightGBM 超参数

| 参数 | 值 | 说明 |
| ---- | ---- | ---- |
"""
        content += f"| learning_rate | {lgb_cfg.learning_rate} | 学习率 |\n"
        content += f"| max_depth | {lgb_cfg.max_depth} | 树深度 |\n"
        content += f"| num_leaves | {lgb_cfg.num_leaves} | 叶子数 |\n"
        content += f"| min_data_in_leaf | {lgb_cfg.min_data_in_leaf} | 叶子最小样本数 |\n"
        content += f"| feature_fraction | {lgb_cfg.feature_fraction} | 列抽样比例 |\n"
        content += f"| bagging_fraction | {lgb_cfg.bagging_fraction} | 行抽样比例 |\n"
        content += f"| lambda_l1 | {lgb_cfg.lambda_l1} | L1 正则化 |\n"
        content += f"| lambda_l2 | {lgb_cfg.lambda_l2} | L2 正则化 |\n"
        content += f"| num_boost_round | {lgb_cfg.num_boost_round} | 最大迭代轮数 |\n"
        content += f"| early_stopping_rounds | {lgb_cfg.early_stopping_rounds} | 早停轮数 |\n"
        content += f"| device | {lgb_cfg.device} | 计算设备 |\n"
        
        # 特征过滤
        if hasattr(feature_cfg, 'drop_macro_prefixes') and feature_cfg.drop_macro_prefixes:
            content += """
### 1.3 特征过滤

**过滤的宏观特征前缀**（截面无区分度）:

"""
            # 分组显示
            prefixes = feature_cfg.drop_macro_prefixes
            for i in range(0, len(prefixes), 6):
                batch = prefixes[i:i+6]
                content += "- " + ", ".join(f"`{p}`" for p in batch) + "\n"
            
            # 获取实际特征数量
            if self.feature_importance:
                first_target = next(iter(self.feature_importance.keys()))
                n_features = len(self.feature_importance[first_target])
                content += f"\n**实际使用特征数**: {n_features}\n"
        
        content += "\n---\n\n"
        return content
    
    def _generate_summary_section(self) -> str:
        """生成结果汇总部分"""
        if not self.evaluation_results:
            self._evaluate_all_targets()
        
        content = """## 2. 训练结果汇总

### 2.1 核心指标对比

| 标签 | IC Mean | IC Std | ICIR | RankIC Mean | RankIC Std | Rank ICIR |
| ---- | ------- | ------ | ---- | ----------- | ---------- | --------- |
"""
        for target_col, metrics in self.evaluation_results.items():
            ic_mean = metrics.get("ic_mean", np.nan)
            ic_std = metrics.get("ic_std", np.nan)
            icir = metrics.get("icir", np.nan)
            rank_ic_mean = metrics.get("rank_ic_mean", np.nan)
            rank_ic_std = metrics.get("rank_ic_std", np.nan)
            rank_icir = metrics.get("rank_icir", np.nan)
            
            # 判断是否有效因子，加粗显示
            is_valid = abs(icir) > 0.5 if not np.isnan(icir) else False
            if is_valid:
                content += f"| **{target_col}** | **{ic_mean:.4f}** | **{ic_std:.4f}** | **{icir:.4f}** | **{rank_ic_mean:.4f}** | **{rank_ic_std:.4f}** | **{rank_icir:.4f}** |\n"
            else:
                content += f"| {target_col} | {ic_mean:.4f} | {ic_std:.4f} | {icir:.4f} | {rank_ic_mean:.4f} | {rank_ic_std:.4f} | {rank_icir:.4f} |\n"
        
        content += """
### 2.2 统计检验

| 标签 | IC>0 比例 | T统计量 | P值 | 显著性 |
| ---- | --------- | ------- | --- | ------ |
"""
        for target_col, metrics in self.evaluation_results.items():
            ic_pos = metrics.get("ic_positive_ratio", np.nan)
            t_stat = metrics.get("t_stat", np.nan)
            p_value = metrics.get("p_value", np.nan)
            
            # 显著性标记
            if np.isnan(p_value):
                sig = "-"
            elif p_value < 0.001:
                sig = "★★★"
            elif p_value < 0.01:
                sig = "★★"
            elif p_value < 0.05:
                sig = "★"
            else:
                sig = "-"
            
            is_valid = abs(metrics.get("icir", 0)) > 0.5
            if is_valid:
                content += f"| **{target_col}** | **{ic_pos:.2%}** | **{t_stat:.4f}** | **{p_value:.4f}** | **{sig}** |\n"
            else:
                content += f"| {target_col} | {ic_pos:.2%} | {t_stat:.4f} | {p_value:.4f} | {sig} |\n"
        
        content += """
### 2.3 多空收益

| 标签 | 日均多空收益 | 累计多空收益 | 年化夏普 |
| ---- | ----------- | ----------- | -------- |
"""
        for target_col, metrics in self.evaluation_results.items():
            avg_ls = metrics.get("avg_long_short_return", np.nan)
            cum_ls = metrics.get("cum_long_short_return", np.nan)
            sharpe = metrics.get("sharpe_long_short", np.nan)
            
            is_valid = abs(metrics.get("icir", 0)) > 0.5
            if is_valid:
                content += f"| **{target_col}** | **{avg_ls:.4%}** | **{cum_ls:.2%}** | **{sharpe:.4f}** |\n"
            else:
                content += f"| {target_col} | {avg_ls:.4%} | {cum_ls:.2%} | {sharpe:.4f} |\n"
        
        content += """
### 2.4 分组单调性

| 标签 | 单调性系数 | 说明 |
| ---- | --------- | ---- |
"""
        for target_col, metrics in self.evaluation_results.items():
            mono = metrics.get("monotonicity", np.nan)
            
            # 单调性说明
            if np.isnan(mono):
                desc = "N/A"
            elif mono > 0.9:
                desc = "完美单调（组号↑收益↑）"
            elif mono > 0.7:
                desc = "强单调"
            elif mono > 0.5:
                desc = "中等单调"
            elif mono > 0:
                desc = "弱正相关"
            elif mono > -0.5:
                desc = "无显著单调性"
            else:
                desc = "负相关"
            
            is_valid = abs(metrics.get("icir", 0)) > 0.5
            if is_valid:
                content += f"| **{target_col}** | **{mono:.4f}** | {desc} |\n"
            else:
                content += f"| {target_col} | {mono:.4f} | {desc} |\n"
        
        content += "\n---\n\n"
        return content
    
    def _generate_target_detail_section(self, target_col: str) -> str:
        """生成单个标签的详细结果部分"""
        metrics = self.evaluation_results.get(target_col, {})
        fold_infos = self.fold_info_dict.get(target_col, [])
        importance_df = self.feature_importance.get(target_col)
        train_infos = self.model_train_info.get(target_col, [])
        
        # 判断是否有效
        icir = metrics.get("icir", 0)
        is_valid = abs(icir) > 0.5
        status = "✅ 有效因子" if is_valid else "❌ 无效因子"
        
        content = f"""## {target_col} 详细结果 ({status})

### 评估指标

```
============================================================
Factor Evaluation Report: {target_col}
============================================================

[IC Analysis]
  IC Mean:            {metrics.get('ic_mean', np.nan):.4f}
  IC Std:             {metrics.get('ic_std', np.nan):.4f}
  ICIR:               {metrics.get('icir', np.nan):.4f}
  Rank IC Mean:       {metrics.get('rank_ic_mean', np.nan):.4f}
  Rank IC Std:        {metrics.get('rank_ic_std', np.nan):.4f}
  Rank ICIR:          {metrics.get('rank_icir', np.nan):.4f}

[Statistical Tests]
  IC Positive Ratio:  {metrics.get('ic_positive_ratio', np.nan):.2%}
  T-Statistic:        {metrics.get('t_stat', np.nan):.4f}
  P-Value:            {metrics.get('p_value', np.nan):.4f}

[Long-Short Returns]
  Avg Daily Return:   {metrics.get('avg_long_short_return', np.nan):.4%}
  Cumulative Return:  {metrics.get('cum_long_short_return', np.nan):.2%}
  Sharpe Ratio:       {metrics.get('sharpe_long_short', np.nan):.4f}

[Monotonicity]
  Group Monotonicity: {metrics.get('monotonicity', np.nan):.4f}
============================================================
```

"""
        
        # Fold 训练详情
        if fold_infos:
            content += """### 各 Fold 训练情况

| Fold | 训练区间 | 验证区间 | 训练样本 | 验证样本 | Best Iter |
| ---- | -------- | -------- | -------- | -------- | --------- |
"""
            for i, fold_info in enumerate(fold_infos):
                train_start = fold_info.get("train_start", "N/A")
                train_end = fold_info.get("train_end", "N/A")
                valid_start = fold_info.get("valid_start", "N/A")
                valid_end = fold_info.get("valid_end", "N/A")
                train_samples = fold_info.get("train_samples", 0)
                valid_samples = fold_info.get("valid_samples", 0)
                
                # 从 train_info 获取 best_iteration
                best_iter = "N/A"
                if i < len(train_infos):
                    best_iter = train_infos[i].get("best_iteration", "N/A")
                
                content += f"| {i} | {train_start} ~ {train_end} | {valid_start} ~ {valid_end} | {train_samples:,} | {valid_samples:,} | {best_iter} |\n"
            
            content += "\n"
        
        # 特征重要性
        if importance_df is not None and len(importance_df) > 0:
            content += """### 特征重要性 Top 20

| 排名 | 特征 | 重要性 | 占比 |
| ---- | ---- | ------ | ---- |
"""
            top_20 = importance_df.head(20)
            total_importance = importance_df["importance"].sum()
            
            for rank, (_, row) in enumerate(top_20.iterrows(), 1):
                feature = row["feature"]
                importance = row["importance"]
                pct = importance / total_importance * 100 if total_importance > 0 else 0
                content += f"| {rank} | {feature} | {importance:,.0f} | {pct:.2f}% |\n"
            
            content += "\n"
        
        content += "---\n\n"
        return content
    
    def _generate_conclusion_section(self) -> str:
        """生成结论部分"""
        if not self.evaluation_results:
            self._evaluate_all_targets()
        
        # 分类有效和无效标签
        valid_targets = []
        invalid_targets = []
        
        for target_col, metrics in self.evaluation_results.items():
            icir = metrics.get("icir", 0)
            if abs(icir) > 0.5:
                valid_targets.append((target_col, metrics))
            else:
                invalid_targets.append((target_col, metrics))
        
        content = """## 结论与建议

### 有效标签

"""
        if valid_targets:
            for target_col, metrics in valid_targets:
                icir = metrics.get("icir", 0)
                ic_pos = metrics.get("ic_positive_ratio", 0)
                mono = metrics.get("monotonicity", 0)
                sharpe = metrics.get("sharpe_long_short", 0)
                
                content += f"""✅ **{target_col}**:
- ICIR: {icir:.4f}（{'显著有效' if icir > 1.0 else '有效'}）
- IC > 0 比例: {ic_pos:.1%}
- 组单调性: {mono:.4f}
- 多空年化夏普: {sharpe:.4f}

"""
        else:
            content += "无有效标签。\n\n"
        
        content += """### 无效标签

"""
        if invalid_targets:
            for target_col, metrics in invalid_targets:
                icir = metrics.get("icir", 0)
                content += f"❌ **{target_col}**: ICIR = {icir:.4f}，模型无法有效预测\n\n"
        else:
            content += "无无效标签。\n\n"
        
        # 建议
        content += """### 后续建议

"""
        if valid_targets:
            content += """1. **模型集成**: 可以将有效模型的 Rolling 和 Single_Full 版本进行加权融合
2. **扩展训练**: 尝试训练其他周期的秩标签（如 rank_ret_10d, rank_ret_20d）
3. **特征优化**: 进一步剔除低重要性特征，尝试特征交叉
"""
        
        if invalid_targets:
            content += f"""4. **标签改进**: 对于无效标签（{', '.join(t[0] for t in invalid_targets)}），考虑：
   - 使用行业中性化处理
   - 尝试秩变换（rank）替代原始值
   - 增加特征工程，如动量、反转因子
"""
        
        content += "\n---\n\n"
        return content
    
    def _generate_model_files_section(self) -> str:
        """生成模型文件部分"""
        mode = self.config.split_config.mode.value
        model_dir = self.config.model_save_dir / mode
        
        content = f"""## 模型文件

**模型保存目录**: `{model_dir}`

"""
        if model_dir.exists():
            model_files = sorted(model_dir.glob("*.pkl"))
            if model_files:
                content += "```\n"
                for f in model_files[:15]:  # 最多显示15个
                    content += f"{f.name}\n"
                if len(model_files) > 15:
                    content += f"... (共 {len(model_files)} 个模型文件)\n"
                content += "```\n\n"
        
        # 【修复】根据训练模式显示正确的输出路径
        mode_dir = self.config.model_save_dir / self.config.split_config.mode.value
        content += f"""**OOF 预测文件**: `{mode_dir / 'oof_predictions.parquet'}`  
**特征重要性**: `{mode_dir / 'feature_importance.parquet'}`
"""
        return content
    
    def generate_report(self, filename: Optional[str] = None) -> Path:
        """
        生成完整训练报告
        
        Args:
            filename: 报告文件名（默认自动生成）
            
        Returns:
            report_path: 报告文件路径
        """
        logger.info("Generating training report...")
        
        # 评估所有标签
        self._evaluate_all_targets()
        
        # 生成报告内容
        content = self._generate_header()
        content += self._generate_config_section()
        content += self._generate_summary_section()
        
        # 为每个标签生成详细部分
        # 先按 ICIR 排序，有效的放前面
        sorted_targets = sorted(
            self.oof_dict.keys(),
            key=lambda t: abs(self.evaluation_results.get(t, {}).get("icir", 0)),
            reverse=True,
        )
        
        section_num = 3
        for target_col in sorted_targets:
            content += f"## {section_num}. {target_col} 详细结果\n\n"
            detail = self._generate_target_detail_section(target_col)
            # 移除重复的标题
            detail = detail.split("\n", 1)[1] if "\n" in detail else detail
            content += detail
            section_num += 1
        

        # 保存报告
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = self.config.split_config.mode.value
            filename = f"lgb_training_report_{mode}_{timestamp}.md"
        
        report_path = self.report_dir / filename
        report_path.write_text(content, encoding="utf-8")
        
        logger.info(f"Training report saved to {report_path}")
        return report_path
