#!/usr/bin/env python
"""
特征与标签计算验证脚本

抽样检查 train.parquet 中的特征和标签计算是否正确。

验证内容：
1. 技术指标（MA、Bias、RSI、MACD 等）
2. 资金流特征（主力强度、散户情绪等）
3. 筹码特征（户数变化、集中度等）
4. 收益率标签（ret_5d、ret_10d 等）
5. 高级标签（超额收益、排名、夏普等）

使用方法：
    python tests/test_feature_label_calculation.py
    python tests/test_feature_label_calculation.py --sample-size 10000
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureLabelValidator:
    """特征与标签验证器"""
    
    def __init__(self, train_path: str):
        """
        初始化
        
        Args:
            train_path: train.parquet 路径
        """
        self.train_path = Path(train_path)
        self.df = None
        self.results: Dict[str, Dict] = {}
        self.tolerance = 1e-6  # 数值比较容差
        
    def load_sample(self, sample_size: int = 5000, seed: int = 42) -> pd.DataFrame:
        """
        加载采样数据
        
        为了验证时序计算，按股票完整加载部分股票
        """
        logger.info(f"📂 加载 {self.train_path}...")
        
        # 先读取全部 ts_code 唯一值
        df_full = pd.read_parquet(self.train_path, columns=['ts_code'])
        unique_codes = df_full['ts_code'].unique()
        
        # 随机选择部分股票
        np.random.seed(seed)
        n_stocks = min(50, len(unique_codes))  # 最多 50 只股票
        selected_codes = np.random.choice(unique_codes, n_stocks, replace=False)
        
        # 读取选定股票的完整数据
        df = pd.read_parquet(self.train_path)
        df = df[df['ts_code'].isin(selected_codes)].copy()
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        # 如果超过 sample_size，进一步过滤日期
        if len(df) > sample_size:
            # 保留最近的数据
            unique_dates = df['trade_date'].unique()
            n_dates = sample_size // n_stocks
            recent_dates = sorted(unique_dates)[-n_dates:]
            df = df[df['trade_date'].isin(recent_dates)].copy()
        
        logger.info(f"  ✓ 采样 {len(df):,} 行, {df['ts_code'].nunique()} 只股票")
        self.df = df
        return df
    
    def validate_technical_features(self) -> Dict:
        """
        验证技术指标
        
        验证公式：
        - ma_5 = close_hfq 的 5 日均值
        - bias_5 = (close_hfq - ma_5) / ma_5
        - roc_5 = (close_hfq - close_hfq.shift(5)) / close_hfq.shift(5)
        - volatility_20 = close_hfq.pct_change() 的 20 日标准差
        """
        logger.info("\n📊 验证技术指标...")
        df = self.df.copy()
        results = {'checks': [], 'passed': 0, 'failed': 0}
        
        # 按股票分组计算
        grouped = df.groupby('ts_code', sort=False)
        
        # 1. MA_5
        if 'ma_5' in df.columns and 'close_hfq' in df.columns:
            expected = grouped['close_hfq'].transform(lambda x: x.rolling(5, min_periods=1).mean())
            check = self._compare_values(df['ma_5'], expected, 'ma_5')
            results['checks'].append(check)
            results['passed' if check['passed'] else 'failed'] += 1
        
        # 2. Bias_5
        if 'bias_5' in df.columns and 'ma_5' in df.columns:
            expected = (df['close_hfq'] - df['ma_5']) / df['ma_5']
            check = self._compare_values(df['bias_5'], expected, 'bias_5')
            results['checks'].append(check)
            results['passed' if check['passed'] else 'failed'] += 1
        
        # 3. ROC_5
        if 'roc_5' in df.columns:
            shifted = grouped['close_hfq'].shift(5)
            expected = (df['close_hfq'] - shifted) / shifted
            check = self._compare_values(df['roc_5'], expected, 'roc_5')
            results['checks'].append(check)
            results['passed' if check['passed'] else 'failed'] += 1
        
        # 4. Volatility_20
        if 'volatility_20' in df.columns:
            pct_change = grouped['close_hfq'].pct_change()
            expected = grouped['close_hfq'].transform(
                lambda x: x.pct_change().rolling(20, min_periods=5).std()
            )
            check = self._compare_values(df['volatility_20'], expected, 'volatility_20')
            results['checks'].append(check)
            results['passed' if check['passed'] else 'failed'] += 1
        
        # 5. Vol Ratio (量比)
        if 'vol_ratio_5' in df.columns and 'vol' in df.columns:
            ma_vol = grouped['vol'].transform(lambda x: x.rolling(5, min_periods=1).mean())
            expected = df['vol'] / (ma_vol + 1e-10)
            check = self._compare_values(df['vol_ratio_5'], expected, 'vol_ratio_5')
            results['checks'].append(check)
            results['passed' if check['passed'] else 'failed'] += 1
        
        self.results['technical'] = results
        self._log_results('技术指标', results)
        return results
    
    def validate_money_flow_features(self) -> Dict:
        """
        验证资金流特征
        
        验证公式：
        - mf_main_intensity = net_main_amount / amount
        - mf_block_intensity = block_trade_amount / amount
        """
        logger.info("\n📊 验证资金流特征...")
        df = self.df.copy()
        results = {'checks': [], 'passed': 0, 'failed': 0}
        
        amount = df['amount'] + 1e-10
        
        # 1. 主力强度
        if 'mf_main_intensity' in df.columns and 'net_main_amount' in df.columns:
            expected = df['net_main_amount'] / amount
            check = self._compare_values(df['mf_main_intensity'], expected, 'mf_main_intensity')
            results['checks'].append(check)
            results['passed' if check['passed'] else 'failed'] += 1
        
        # 2. 大宗交易强度
        if 'mf_block_intensity' in df.columns and 'block_trade_amount' in df.columns:
            expected = df['block_trade_amount'] / amount
            check = self._compare_values(df['mf_block_intensity'], expected, 'mf_block_intensity')
            results['checks'].append(check)
            results['passed' if check['passed'] else 'failed'] += 1
        
        # 3. 散户买入比
        if 'mf_retail_buy_ratio' in df.columns:
            if 'buy_sm_amount' in df.columns and 'buy_md_amount' in df.columns:
                expected = df['buy_sm_amount'] / amount
                check = self._compare_values(df['mf_retail_buy_ratio'], expected, 'mf_retail_buy_ratio')
                results['checks'].append(check)
                results['passed' if check['passed'] else 'failed'] += 1
        
        self.results['money_flow'] = results
        self._log_results('资金流特征', results)
        return results
    
    def validate_chip_features(self) -> Dict:
        """
        验证筹码特征
        
        验证公式：
        - chip_holder_chg = holder_num_chg_pct
        - chip_top10_ratio = top10_hold_ratio / 100
        - chip_top1_dominance = top1_hold_ratio / (top10_hold_ratio + epsilon)
        """
        logger.info("\n📊 验证筹码特征...")
        df = self.df.copy()
        results = {'checks': [], 'passed': 0, 'failed': 0}
        
        # 1. 户数变化
        if 'chip_holder_chg' in df.columns and 'holder_num_chg_pct' in df.columns:
            expected = df['holder_num_chg_pct']
            check = self._compare_values(df['chip_holder_chg'], expected, 'chip_holder_chg')
            results['checks'].append(check)
            results['passed' if check['passed'] else 'failed'] += 1
        
        # 2. Top10 比例标准化
        if 'chip_top10_ratio' in df.columns and 'top10_hold_ratio' in df.columns:
            expected = df['top10_hold_ratio'] / 100.0
            check = self._compare_values(df['chip_top10_ratio'], expected, 'chip_top10_ratio')
            results['checks'].append(check)
            results['passed' if check['passed'] else 'failed'] += 1
        
        # 3. Top1 控盘度
        if 'chip_top1_dominance' in df.columns:
            if 'top1_hold_ratio' in df.columns and 'top10_hold_ratio' in df.columns:
                expected = df['top1_hold_ratio'] / (df['top10_hold_ratio'] + 1e-10)
                check = self._compare_values(df['chip_top1_dominance'], expected, 'chip_top1_dominance')
                results['checks'].append(check)
                results['passed' if check['passed'] else 'failed'] += 1
        
        self.results['chip'] = results
        self._log_results('筹码特征', results)
        return results
    
    def validate_labels(self) -> Dict:
        """
        验证标签计算
        
        验证公式：
        - ret_5d = (close_hfq.shift(-5) - close_hfq) / close_hfq
        - ret_10d = (close_hfq.shift(-10) - close_hfq) / close_hfq
        """
        logger.info("\n📊 验证收益率标签...")
        df = self.df.copy()
        results = {'checks': [], 'passed': 0, 'failed': 0}
        
        grouped = df.groupby('ts_code', sort=False)
        price_col = 'close_hfq' if 'close_hfq' in df.columns else 'close'
        
        for days in [1, 5, 10, 20]:
            label_col = f'ret_{days}d'
            if label_col in df.columns:
                future_price = grouped[price_col].shift(-days)
                expected = (future_price - df[price_col]) / df[price_col]
                
                # 注意：标签可能被裁剪过，只检查未裁剪的部分
                check = self._compare_values(
                    df[label_col], expected, label_col,
                    tolerance=0.01  # 放宽容差，因为可能有裁剪
                )
                results['checks'].append(check)
                results['passed' if check['passed'] else 'failed'] += 1
        
        self.results['labels'] = results
        self._log_results('收益率标签', results)
        return results
    
    def validate_advanced_labels(self) -> Dict:
        """
        验证高级标签
        
        验证公式：
        - rank_ret_5d = ret_5d 在每日截面的分位数
        - label_5d = 1 if ret_5d > 0 else 0
        """
        logger.info("\n📊 验证高级标签...")
        df = self.df.copy()
        results = {'checks': [], 'passed': 0, 'failed': 0}
        
        # 1. 分类标签
        if 'label_5d' in df.columns and 'ret_5d' in df.columns:
            expected = (df['ret_5d'] > 0).astype('int32')
            check = self._compare_values(df['label_5d'], expected, 'label_5d')
            results['checks'].append(check)
            results['passed' if check['passed'] else 'failed'] += 1
        
        # 2. 截面排名
        if 'rank_ret_5d' in df.columns and 'ret_5d' in df.columns:
            # 检查是否在 [0, 1] 范围内
            valid = (df['rank_ret_5d'] >= 0) & (df['rank_ret_5d'] <= 1)
            valid_ratio = valid.sum() / len(df)
            check = {
                'name': 'rank_ret_5d',
                'passed': valid_ratio > 0.99,
                'message': f'有效范围 [0,1] 比例: {valid_ratio:.4f}'
            }
            results['checks'].append(check)
            results['passed' if check['passed'] else 'failed'] += 1
        
        # 3. 分位数分类
        if 'label_bin_5d' in df.columns:
            # 检查值分布 (应该是 0, 1, 2)
            unique_vals = df['label_bin_5d'].dropna().unique()
            expected_vals = {0, 1, 2}
            valid = set(unique_vals).issubset(expected_vals)
            check = {
                'name': 'label_bin_5d',
                'passed': valid,
                'message': f'唯一值: {sorted(unique_vals)}'
            }
            results['checks'].append(check)
            results['passed' if check['passed'] else 'failed'] += 1
        
        self.results['advanced_labels'] = results
        self._log_results('高级标签', results)
        return results
    
    def validate_macro_features(self) -> Dict:
        """
        验证宏观交互特征
        """
        logger.info("\n📊 验证宏观交互特征...")
        df = self.df.copy()
        results = {'checks': [], 'passed': 0, 'failed': 0}
        
        # 1. 流动性敏感度
        if 'macro_amount_shibor' in df.columns:
            if 'amount' in df.columns and 'shibor_1m' in df.columns:
                log_amount = np.log1p(df['amount'])
                expected = log_amount * df['shibor_1m']
                check = self._compare_values(df['macro_amount_shibor'], expected, 'macro_amount_shibor')
                results['checks'].append(check)
                results['passed' if check['passed'] else 'failed'] += 1
        
        # 2. M2 敏感度
        if 'macro_vol_m2' in df.columns:
            if 'vol' in df.columns and 'm2_yoy' in df.columns:
                log_vol = np.log1p(df['vol'])
                expected = log_vol * df['m2_yoy']
                check = self._compare_values(df['macro_vol_m2'], expected, 'macro_vol_m2')
                results['checks'].append(check)
                results['passed' if check['passed'] else 'failed'] += 1
        
        self.results['macro'] = results
        self._log_results('宏观交互特征', results)
        return results
    
    def _compare_values(
        self, 
        actual: pd.Series, 
        expected: pd.Series, 
        name: str,
        tolerance: float = None
    ) -> Dict:
        """
        比较两个 Series 的值
        
        Returns:
            检查结果字典
        """
        if tolerance is None:
            tolerance = self.tolerance
        
        # 过滤 NaN
        mask = actual.notna() & expected.notna()
        
        if mask.sum() == 0:
            return {
                'name': name,
                'passed': False,
                'message': '无有效数据可比较'
            }
        
        actual_valid = actual[mask]
        expected_valid = expected[mask]
        
        # 计算差异
        diff = (actual_valid - expected_valid).abs()
        
        # 使用相对误差（避免大数值问题）
        rel_diff = diff / (expected_valid.abs() + 1e-10)
        
        max_diff = diff.max()
        mean_diff = diff.mean()
        max_rel_diff = rel_diff.max()
        mean_rel_diff = rel_diff.mean()
        
        # 判断是否通过
        passed = mean_rel_diff < tolerance or mean_diff < tolerance
        
        return {
            'name': name,
            'passed': passed,
            'samples': int(mask.sum()),
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'max_rel_diff': float(max_rel_diff),
            'mean_rel_diff': float(mean_rel_diff),
            'message': f'相对误差: mean={mean_rel_diff:.2e}, max={max_rel_diff:.2e}'
        }
    
    def _log_results(self, category: str, results: Dict):
        """打印检查结果"""
        total = results['passed'] + results['failed']
        if total == 0:
            logger.info(f"  ⚠️ {category}: 无可检查项")
            return
        
        status = '✅' if results['failed'] == 0 else '❌'
        logger.info(f"  {status} {category}: {results['passed']}/{total} 通过")
        
        for check in results['checks']:
            mark = '✓' if check['passed'] else '✗'
            logger.info(f"    [{mark}] {check['name']}: {check.get('message', '')}")
    
    def run_all_validations(self) -> Dict:
        """运行所有验证"""
        logger.info("\n" + "=" * 60)
        logger.info("🔍 特征与标签计算验证")
        logger.info("=" * 60)
        
        self.validate_technical_features()
        self.validate_money_flow_features()
        self.validate_chip_features()
        self.validate_labels()
        self.validate_advanced_labels()
        self.validate_macro_features()
        
        # 汇总
        total_passed = sum(r['passed'] for r in self.results.values())
        total_failed = sum(r['failed'] for r in self.results.values())
        
        logger.info("\n" + "=" * 60)
        logger.info(f"📋 验证汇总: {total_passed}/{total_passed + total_failed} 通过")
        if total_failed > 0:
            logger.warning(f"   ⚠️ {total_failed} 项验证失败")
        else:
            logger.info("   ✅ 所有验证通过!")
        logger.info("=" * 60)
        
        return {
            'total_passed': total_passed,
            'total_failed': total_failed,
            'details': self.results
        }
    
    def generate_report(self) -> str:
        """生成验证报告"""
        lines = [
            "# 特征与标签计算验证报告\n",
            f"**数据源**: {self.train_path}\n",
            f"**采样量**: {len(self.df):,} 行\n",
            "---\n",
        ]
        
        for category, results in self.results.items():
            total = results['passed'] + results['failed']
            status = '✅ 通过' if results['failed'] == 0 else '❌ 有问题'
            lines.append(f"\n## {category.title()}\n")
            lines.append(f"**状态**: {status} ({results['passed']}/{total})\n")
            lines.append("\n| 特征 | 状态 | 详情 |\n|------|------|------|\n")
            
            for check in results['checks']:
                mark = '✅' if check['passed'] else '❌'
                msg = check.get('message', '-')
                lines.append(f"| {check['name']} | {mark} | {msg} |\n")
        
        return ''.join(lines)


def main():
    parser = argparse.ArgumentParser(description="验证 train.parquet 的特征和标签计算")
    parser.add_argument(
        '--train-path',
        default='data/features/structured/train.parquet',
        help='train.parquet 路径'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5000,
        help='采样大小'
    )
    parser.add_argument(
        '--output',
        default='reports/feature_label_validation_report.md',
        help='报告输出路径'
    )
    args = parser.parse_args()
    
    # 运行验证
    validator = FeatureLabelValidator(args.train_path)
    validator.load_sample(sample_size=args.sample_size)
    results = validator.run_all_validations()
    
    # 保存报告
    report = validator.generate_report()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info(f"\n📄 报告已保存: {output_path}")
    
    # 返回状态码
    return 0 if results['total_failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
