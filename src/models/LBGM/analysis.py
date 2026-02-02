"""
模型分析与归因

包含:
- 特征重要性分析
- 测试集回测指标
- SHAP 可解释性分析
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from .metrics import evaluate_predictions, calculate_daily_ic, spearman_corr

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """
    模型分析器
    
    回答问题: "模型到底学到了什么？"
    """
    
    def __init__(self, model: lgb.Booster, config=None):
        """
        初始化分析器
        
        Args:
            model: 训练好的 LightGBM 模型
            config: LGBMConfig 配置对象 (可选)
        """
        self.model = model
        self.config = config
        self._analysis_results: Dict[str, Any] = {}
    
    def analyze_feature_importance(
        self,
        top_n: int = 20,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        分析特征重要性
        
        Gain (增益) 比 Split (分裂次数) 更重要，
        因为它反映了特征对模型预测精度的实际贡献。
        
        Args:
            top_n: 返回 Top N 重要特征
            importance_type: 'gain' 或 'split'
            
        Returns:
            特征重要性 DataFrame
        """
        logger.info("=" * 60)
        logger.info(f"📊 特征重要性分析 (Top {top_n})")
        logger.info("=" * 60)
        
        feature_names = self.model.feature_name()
        importance = self.model.feature_importance(importance_type=importance_type)
        
        # 构建 DataFrame
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # 排序
        df = df.sort_values('importance', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
        
        # 打印 Top N
        logger.info(f"  📋 {importance_type.upper()} 重要性 Top {top_n}:")
        for i, row in df.head(top_n).iterrows():
            logger.info(f"     {row['rank']:2d}. {row['feature']:<30} {row['importance_pct']:6.2f}%")
        
        # 检查异常: 如果某个冷门因子排第一，要警惕泄露
        top_features = df.head(5)['feature'].tolist()
        known_important = ['ma_5', 'ma_10', 'ma_20', 'vol', 'amount', 'close', 
                          'turnover', 'roc_5', 'bias_5', 'volatility_5']
        
        suspicious = [f for f in top_features if not any(k in f.lower() for k in ['ma', 'vol', 'roc', 'bias', 'close', 'open', 'high', 'low', 'turnover', 'amount', 'return', 'rsi', 'macd'])]
        
        if suspicious:
            logger.warning(f"  ⚠️ 可疑的头部特征: {suspicious}")
            logger.warning("     请检查是否存在数据泄露!")
        
        self._analysis_results['feature_importance'] = df
        
        return df.head(top_n)
    
    def evaluate_test_set(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_dates: np.ndarray,
        test_codes: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        测试集回测评估
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            test_dates: 测试集日期
            test_codes: 测试集股票代码 (可选)
            
        Returns:
            评估指标字典
        """
        logger.info("=" * 60)
        logger.info("📊 测试集回测评估 (Out-of-Sample)")
        logger.info("=" * 60)
        
        # 预测
        preds = self.model.predict(X_test)
        
        # 综合评估
        results = evaluate_predictions(preds, y_test, test_dates)
        
        # 打印结果
        logger.info(f"  📋 整体指标:")
        logger.info(f"     IC: {results.get('IC', 0):.4f}")
        logger.info(f"     RankIC: {results.get('RankIC', 0):.4f}")
        logger.info(f"     RMSE: {results.get('RMSE', 0):.6f}")
        
        logger.info(f"  📋 每日指标:")
        logger.info(f"     RankIC 均值: {results.get('DailyRankIC_Mean', 0):.4f}")
        logger.info(f"     RankIC 标准差: {results.get('DailyRankIC_Std', 0):.4f}")
        logger.info(f"     ICIR: {results.get('ICIR', 0):.4f}")
        logger.info(f"     交易日数: {results.get('DailyIC_Count', 0)}")
        
        # 判断效果
        rank_ic = results.get('DailyRankIC_Mean', 0)
        icir = results.get('ICIR', 0)
        
        logger.info(f"  📊 效果判断:")
        if rank_ic > 0.05:
            logger.info("     ✅ RankIC > 0.05，模型非常有效!")
        elif rank_ic > 0.03:
            logger.info("     ✓ RankIC > 0.03，模型有效")
        else:
            logger.warning("     ⚠️ RankIC < 0.03，模型效果欠佳")
        
        if icir > 0.5:
            logger.info("     ✅ ICIR > 0.5，发挥稳定!")
        elif icir > 0.3:
            logger.info("     ✓ ICIR > 0.3，发挥较稳定")
        else:
            logger.warning("     ⚠️ ICIR < 0.3，发挥不稳定")
        
        # 保存预测值用于后续分析
        self._analysis_results['test_predictions'] = preds
        self._analysis_results['test_labels'] = y_test
        self._analysis_results['test_dates'] = test_dates
        self._analysis_results['test_metrics'] = results
        
        return results
    
    def analyze_daily_ic(
        self,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        test_dates: np.ndarray = None
    ) -> pd.DataFrame:
        """
        分析每日 IC 分布
        
        Args:
            X_test: 测试特征 (可选，如果之前已评估则可省略)
            y_test: 测试标签
            test_dates: 测试日期
            
        Returns:
            每日 IC DataFrame
        """
        # 使用缓存的结果
        if X_test is None:
            preds = self._analysis_results.get('test_predictions')
            y_test = self._analysis_results.get('test_labels')
            test_dates = self._analysis_results.get('test_dates')
        else:
            preds = self.model.predict(X_test)
        
        if preds is None:
            raise RuntimeError("请先调用 evaluate_test_set() 或提供测试数据")
        
        # 计算每日 IC
        unique_dates = np.unique(test_dates)
        daily_data = []
        
        for date in unique_dates:
            mask = test_dates == date
            if mask.sum() < 10:
                continue
            
            day_preds = preds[mask]
            day_labels = y_test[mask]
            
            ic = spearman_corr(day_preds, day_labels)
            
            daily_data.append({
                'date': date,
                'RankIC': ic,
                'sample_count': mask.sum()
            })
        
        df = pd.DataFrame(daily_data)
        
        # 添加统计
        df['cumulative_mean'] = df['RankIC'].expanding().mean()
        
        logger.info(f"  📊 每日 IC 统计:")
        logger.info(f"     最大值: {df['RankIC'].max():.4f}")
        logger.info(f"     最小值: {df['RankIC'].min():.4f}")
        logger.info(f"     正值比例: {(df['RankIC'] > 0).mean():.1%}")
        
        self._analysis_results['daily_ic'] = df
        
        return df
    
    def explain_with_shap(
        self,
        X_sample: np.ndarray,
        feature_names: List[str],
        sample_date: str = None,
        sample_codes: List[str] = None,
        max_display: int = 20
    ) -> Dict[str, Any]:
        """
        使用 SHAP 解释模型预测
        
        Args:
            X_sample: 要解释的样本特征
            feature_names: 特征名列表
            sample_date: 样本日期 (用于日志)
            sample_codes: 样本股票代码 (用于日志)
            max_display: 展示的特征数
            
        Returns:
            SHAP 分析结果
        """
        try:
            import shap
        except ImportError:
            logger.warning("⚠️ shap 未安装，跳过 SHAP 分析")
            return {}
        
        logger.info("=" * 60)
        logger.info("📊 SHAP 可解释性分析")
        logger.info("=" * 60)
        
        if sample_date:
            logger.info(f"  📅 分析日期: {sample_date}")
        logger.info(f"  📋 样本数: {len(X_sample)}")
        
        # 创建 TreeExplainer
        explainer = shap.TreeExplainer(self.model)
        
        # 计算 SHAP 值
        shap_values = explainer.shap_values(X_sample)
        
        # 计算每个特征的平均绝对 SHAP 值
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # 排序
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        logger.info(f"  📋 SHAP 重要性 Top {max_display}:")
        for i, row in feature_importance.head(max_display).iterrows():
            logger.info(f"     {row['feature']:<30} {row['mean_abs_shap']:.6f}")
        
        result = {
            'shap_values': shap_values,
            'expected_value': explainer.expected_value,
            'feature_importance': feature_importance,
        }
        
        self._analysis_results['shap'] = result
        
        return result
    
    def generate_report(self, output_path: Path = None) -> str:
        """
        生成分析报告
        
        Args:
            output_path: 报告输出路径 (可选)
            
        Returns:
            报告内容
        """
        lines = [
            "# LightGBM 模型分析报告",
            "",
            "## 1. 测试集评估指标",
            ""
        ]
        
        metrics = self._analysis_results.get('test_metrics', {})
        if metrics:
            lines.append(f"| 指标 | 值 |")
            lines.append(f"|------|-----|")
            lines.append(f"| IC | {metrics.get('IC', 0):.4f} |")
            lines.append(f"| RankIC | {metrics.get('RankIC', 0):.4f} |")
            lines.append(f"| RMSE | {metrics.get('RMSE', 0):.6f} |")
            lines.append(f"| 每日 RankIC 均值 | {metrics.get('DailyRankIC_Mean', 0):.4f} |")
            lines.append(f"| 每日 RankIC 标准差 | {metrics.get('DailyRankIC_Std', 0):.4f} |")
            lines.append(f"| ICIR | {metrics.get('ICIR', 0):.4f} |")
            lines.append("")
        
        lines.append("## 2. 特征重要性 (Top 20)")
        lines.append("")
        
        fi = self._analysis_results.get('feature_importance')
        if fi is not None:
            lines.append(f"| 排名 | 特征 | 重要性 (%) |")
            lines.append(f"|------|------|-----------|")
            for _, row in fi.head(20).iterrows():
                lines.append(f"| {row['rank']} | {row['feature']} | {row['importance_pct']:.2f}% |")
            lines.append("")
        
        report = "\n".join(lines)
        
        if output_path:
            output_path.write_text(report)
            logger.info(f"  📝 报告已保存: {output_path}")
        
        return report
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """获取所有分析结果"""
        return self._analysis_results
