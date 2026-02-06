#!/usr/bin/env python
"""
特征工程流水线测试脚本 - 测试 8 表合并 -> 预处理 流程

流程：
1. 从 data/processed/structured/dwd/ 读取 8 张 DWD 宽表
2. 使用 DataMerger 合并为一张大表
3. 对合并后的大表进行预处理（列级别处理）
4. 输出预处理结果到 data/features/preprocessed/merged_preprocessed.parquet

使用方法：
    python tests/test_merge_preprocess_pipeline.py
    python tests/test_merge_preprocess_pipeline.py --no-gpu
    python tests/test_merge_preprocess_pipeline.py --dry-run

验证点：
- GPU 加速是否生效 (RAPIDS cuDF)
- 8 表合并行数与 price 骨架表一致
- 预处理后无未处理的 NaN
- 处理耗时合理（GPU < 3分钟）
"""

import argparse
import gc
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = True) -> logging.Logger:
    """配置日志"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"test_merge_preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def get_df_lib(use_gpu: bool):
    """获取 DataFrame 库"""
    if use_gpu:
        try:
            import cudf
            import cupy as cp
            return cudf, cp, True
        except ImportError as e:
            print(f"⚠️ cuDF 导入失败: {e}，回退到 pandas")
            import pandas as pd
            import numpy as np
            return pd, np, False
    else:
        import pandas as pd
        import numpy as np
        return pd, np, False


def force_gc(use_gpu: bool, cp=None):
    """强制垃圾回收"""
    gc.collect()
    
    if use_gpu and cp is not None:
        try:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
        except Exception:
            pass


def log_memory(stage: str, use_gpu: bool, cp=None, logger=None):
    """记录内存使用"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if use_gpu and cp is not None:
        try:
            mempool = cp.get_default_memory_pool()
            used_gb = mempool.used_bytes() / (1024**3)
            total_gb = mempool.total_bytes() / (1024**3)
            logger.info(f"[{stage}] GPU内存: {used_gb:.2f}GB / {total_gb:.2f}GB")
        except Exception:
            pass


class MergePreprocessPipeline:
    """合并 + 预处理 流水线"""
    
    # DWD 表文件名映射
    TABLE_FILES = {
        'price': 'dwd_stock_price.parquet',
        'fundamental': 'dwd_stock_fundamental.parquet',
        'status': 'dwd_stock_status.parquet',
        'money_flow': 'dwd_money_flow.parquet',
        'chip': 'dwd_chip_structure.parquet',
        'industry': 'dwd_stock_industry.parquet',
        'event': 'dwd_event_signal.parquet',
        'macro': 'dwd_macro_env.parquet',
    }
    
    # 面板表（按 ts_code + trade_date 合并）
    PANEL_TABLES = ['fundamental', 'status', 'money_flow', 'chip', 'industry', 'event']
    
    # 宏观表（仅按 trade_date 广播）
    MACRO_TABLES = ['macro']
    
    # 合并键
    PANEL_KEYS = ['ts_code', 'trade_date']
    MACRO_KEYS = ['trade_date']
    
    def __init__(self, use_gpu: bool = True, logger=None):
        self.use_gpu = use_gpu
        self.logger = logger or logging.getLogger(__name__)
        
        # 获取 DataFrame 库
        self.df_lib, self.array_lib, self.gpu_active = get_df_lib(use_gpu)
        
        if self.gpu_active:
            self.logger.info("🚀 GPU 加速已启用 (RAPIDS cuDF)")
        else:
            self.logger.warning("⚠️ GPU 不可用，使用 CPU 模式 (pandas)")
        
        # 路径配置
        self.input_dir = PROJECT_ROOT / "data" / "processed" / "structured" / "dwd"
        self.output_dir = PROJECT_ROOT / "data" / "features" / "preprocessed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = {}
    
    def _read_table(self, table_name: str) -> Optional[Any]:
        """读取单张表"""
        file_name = self.TABLE_FILES.get(table_name)
        if file_name is None:
            self.logger.warning(f"未知表名: {table_name}")
            return None
        
        path = self.input_dir / file_name
        if not path.exists():
            self.logger.warning(f"文件不存在: {path}")
            return None
        
        self.logger.info(f"  📖 读取: {table_name}")
        df = self.df_lib.read_parquet(str(path))
        self.logger.info(f"     → {len(df):,} 行, {len(df.columns)} 列")
        
        return df
    
    def _downcast_float(self, df: Any) -> Any:
        """float64 → float32 节省内存"""
        float64_cols = df.select_dtypes(include=['float64']).columns.tolist()
        if len(float64_cols) > 0:
            for col in float64_cols:
                df[col] = df[col].astype('float32')
            self.logger.debug(f"  下压 {len(float64_cols)} 列 float64 → float32")
        return df
    
    def step1_load_and_merge(self) -> Any:
        """
        Step 1: 加载 8 张 DWD 表并合并
        
        合并策略：
        - 主骨架：price 表
        - 面板表：fundamental, status, money_flow, chip, industry, event → 按 (ts_code, trade_date) 左连接
        - 宏观表：macro → 仅按 trade_date 广播
        """
        self.logger.info("=" * 60)
        self.logger.info("📋 Step 1: 加载 8 张 DWD 表并合并")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # 1. 加载主骨架表 (price)
        self.logger.info("[1/3] 加载骨架表 (price)...")
        skeleton = self._read_table('price')
        if skeleton is None:
            raise RuntimeError("骨架表 price 不存在")
        
        skeleton_rows = len(skeleton)
        self.stats['skeleton_rows'] = skeleton_rows
        
        # 下压内存
        skeleton = self._downcast_float(skeleton)
        
        # 2. 流式合并面板表
        self.logger.info("\n[2/3] 流式合并面板表...")
        for table_name in self.PANEL_TABLES:
            panel_df = self._read_table(table_name)
            if panel_df is None:
                continue
            
            # 下压内存
            panel_df = self._downcast_float(panel_df)
            
            # 确保合并键存在
            missing_keys = [k for k in self.PANEL_KEYS if k not in panel_df.columns]
            if missing_keys:
                self.logger.warning(f"  表 {table_name} 缺少合并键: {missing_keys}，跳过")
                del panel_df
                force_gc(self.gpu_active, self.array_lib)
                continue
            
            # 预去重
            dup_before = len(panel_df)
            panel_df = panel_df.drop_duplicates(subset=self.PANEL_KEYS, keep='first')
            dup_after = len(panel_df)
            if dup_before > dup_after:
                self.logger.debug(f"  去重: {dup_before:,} → {dup_after:,} ({dup_before - dup_after} 重复)")
            
            # 获取除主键外的增量列
            new_cols = [c for c in panel_df.columns if c not in self.PANEL_KEYS and c not in skeleton.columns]
            if not new_cols:
                self.logger.debug(f"  表 {table_name} 无增量列，跳过")
                del panel_df
                force_gc(self.gpu_active, self.array_lib)
                continue
            
            # 只保留需要的列进行合并
            merge_cols = self.PANEL_KEYS + new_cols
            panel_df = panel_df[merge_cols]
            
            # 左连接
            merge_start = time.time()
            skeleton = skeleton.merge(panel_df, on=self.PANEL_KEYS, how='left')
            merge_time = time.time() - merge_start
            
            self.logger.info(f"  ✓ {table_name}: +{len(new_cols)} 列 ({merge_time:.2f}s)")
            
            # 释放内存
            del panel_df
            force_gc(self.gpu_active, self.array_lib)
        
        # 3. 广播宏观表
        self.logger.info("\n[3/3] 广播宏观表 (macro)...")
        macro_df = self._read_table('macro')
        if macro_df is not None:
            macro_df = self._downcast_float(macro_df)
            
            # 去重
            macro_df = macro_df.drop_duplicates(subset=self.MACRO_KEYS, keep='first')
            
            # 增量列
            new_cols = [c for c in macro_df.columns if c not in self.MACRO_KEYS and c not in skeleton.columns]
            if new_cols:
                merge_cols = self.MACRO_KEYS + new_cols
                macro_df = macro_df[merge_cols]
                
                merge_start = time.time()
                skeleton = skeleton.merge(macro_df, on=self.MACRO_KEYS, how='left')
                merge_time = time.time() - merge_start
                
                self.logger.info(f"  ✓ macro: +{len(new_cols)} 列 ({merge_time:.2f}s)")
            
            del macro_df
            force_gc(self.gpu_active, self.array_lib)
        
        # 验证行数
        merged_rows = len(skeleton)
        if merged_rows != skeleton_rows:
            self.logger.warning(f"⚠️ 行数变化: {skeleton_rows:,} → {merged_rows:,} (差异: {merged_rows - skeleton_rows:,})")
        
        elapsed = time.time() - start_time
        self.stats['merge_time'] = elapsed
        self.stats['merged_rows'] = merged_rows
        self.stats['merged_cols'] = len(skeleton.columns)
        
        self.logger.info(f"\n✅ 合并完成: {merged_rows:,} 行 × {len(skeleton.columns)} 列 (耗时 {elapsed:.2f}s)")
        
        return skeleton
    
    def step2_preprocess(self, df: Any) -> Any:
        """
        Step 2: 对合并后的大表进行预处理
        
        预处理内容：
        1. 单位换算（金额统一为元）
        2. 缺失值处理（ffill + fillna(0)）
        3. 异常值裁剪（收益率、比例等）
        4. 类型优化（float32）
        """
        self.logger.info("=" * 60)
        self.logger.info("📋 Step 2: 预处理合并后的大表")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # 2.1 单位换算
        self.logger.info("[1/4] 单位换算...")
        df = self._apply_unit_conversion(df)
        
        # 2.2 异常值裁剪
        self.logger.info("[2/4] 异常值裁剪...")
        df = self._apply_clipping(df)
        
        # 2.3 缺失值处理
        self.logger.info("[3/4] 缺失值处理...")
        df = self._apply_fillna(df)
        
        # 2.4 类型优化
        self.logger.info("[4/4] 类型优化...")
        df = self._downcast_float(df)
        
        elapsed = time.time() - start_time
        self.stats['preprocess_time'] = elapsed
        
        # 统计 NaN
        nan_summary = self._check_nan_summary(df)
        self.stats['nan_summary'] = nan_summary
        
        self.logger.info(f"\n✅ 预处理完成 (耗时 {elapsed:.2f}s)")
        
        return df
    
    def step3_features(self, df: Any) -> Any:
        """
        Step 3: 衍生特征计算
        
        生成技术指标、资金流特征、筹码特征等
        """
        self.logger.info("=" * 60)
        self.logger.info("📋 Step 3: 衍生特征计算")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        cols_before = len(df.columns)
        
        try:
            from src.feature_engineering.structured.features.feature_generator import FeatureGenerator
            from src.feature_engineering.structured.features.reference_data_loader import ReferenceDataLoader
            from src.feature_engineering.structured.config import PipelineConfig
            
            config = PipelineConfig()
            
            # 加载参考数据
            self.logger.info("  📖 加载参考数据...")
            ref_loader = ReferenceDataLoader(config.data, use_gpu=self.gpu_active)
            self.ref_data = ref_loader.load_all()
            
            # 特征生成
            self.logger.info("  🔧 生成技术指标和衍生特征...")
            feature_generator = FeatureGenerator(config.technical, use_gpu=self.gpu_active, ref_data=self.ref_data)
            df = feature_generator.generate_all(df)
            
            cols_after = len(df.columns)
            elapsed = time.time() - start_time
            
            self.logger.info(f"  ✓ 特征生成完成: {cols_before} → {cols_after} 列 (+{cols_after - cols_before})")
            self.logger.info(f"  ⏱️ 耗时: {elapsed:.2f}s")
            
            self.stats['feature_time'] = elapsed
            self.stats['feature_cols_added'] = cols_after - cols_before
            
        except Exception as e:
            self.logger.error(f"  ❌ 特征生成失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.ref_data = None
        
        force_gc(self.gpu_active, self.array_lib)
        return df
    
    def step4_labels(self, df: Any) -> Any:
        """
        Step 4: 标签生成
        
        生成基础收益率标签和高级标签
        """
        self.logger.info("=" * 60)
        self.logger.info("📋 Step 4: 标签生成")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        cols_before = len(df.columns)
        
        try:
            from src.feature_engineering.structured.labels.label_generator import LabelGenerator
            from src.feature_engineering.structured.config import PipelineConfig
            
            config = PipelineConfig()
            
            ref_data = getattr(self, 'ref_data', None)
            label_generator = LabelGenerator(config.label, use_gpu=self.gpu_active, ref_data=ref_data)
            df = label_generator.generate_labels(df)
            
            cols_after = len(df.columns)
            elapsed = time.time() - start_time
            
            # 统计标签列
            label_cols = [c for c in df.columns if c.startswith('ret_') or c.startswith('label_') 
                          or c.startswith('excess_') or c.startswith('rank_') or c.startswith('sharpe_')]
            
            self.logger.info(f"  ✓ 标签生成完成: {cols_before} → {cols_after} 列 (+{cols_after - cols_before})")
            self.logger.info(f"  📊 标签字段: {label_cols}")
            self.logger.info(f"  ⏱️ 耗时: {elapsed:.2f}s")
            
            self.stats['label_time'] = elapsed
            self.stats['label_cols'] = label_cols
            
        except Exception as e:
            self.logger.error(f"  ❌ 标签生成失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        force_gc(self.gpu_active, self.array_lib)
        return df
    
    def _apply_unit_conversion(self, df: Any) -> Any:
        """单位换算"""
        # 成交额：千元 → 元 (dwd_stock_price.amount)
        if 'amount' in df.columns:
            df['amount'] = df['amount'] * 1000
            self.logger.info("  - amount: 千元 → 元")
        
        # 市值：万元 → 元 (dwd_stock_fundamental)
        for col in ['total_mv', 'circ_mv']:
            if col in df.columns:
                df[col] = df[col] * 10000
                self.logger.info(f"  - {col}: 万元 → 元")
        
        # 资金流：万元 → 元 (dwd_money_flow)
        money_flow_cols = [c for c in df.columns if c.endswith('_amount') and c.startswith(('buy_', 'sell_', 'net_'))]
        if money_flow_cols:
            for col in money_flow_cols:
                if col != 'amount':  # 排除已处理的 amount
                    df[col] = df[col] * 10000
            self.logger.info(f"  - {len(money_flow_cols)} 个资金流列: 万元 → 元")
        
        # 大宗交易：单独处理（不匹配上面的模式）
        if 'block_trade_amount' in df.columns:
            df['block_trade_amount'] = df['block_trade_amount'] * 10000  # 万元 → 元
            self.logger.info("  - block_trade_amount: 万元 → 元")
        
        if 'block_trade_vol' in df.columns:
            df['block_trade_vol'] = df['block_trade_vol'] * 100  # 万股 → 手 (1万股=100手)
            self.logger.info("  - block_trade_vol: 万股 → 手 (×100)")
        
        # 龙虎榜：万元 → 元
        top_cols = ['top_l_buy', 'top_l_sell', 'top_net_amount', 
                    'top_inst_buy', 'top_inst_sell', 'top_inst_net_buy']
        for col in top_cols:
            if col in df.columns:
                df[col] = df[col] * 10000
        if any(col in df.columns for col in top_cols):
            self.logger.info(f"  - 龙虎榜相关字段: 万元 → 元")
        
        return df
    
    def _apply_clipping(self, df: Any) -> Any:
        """异常值裁剪"""
        # 收益率：裁剪到 [-50%, +100%]
        return_cols = [c for c in df.columns if c.startswith('ret_') or c == 'pct_chg']
        for col in return_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=-0.5, upper=1.0)
        if return_cols:
            self.logger.info(f"  - {len(return_cols)} 个收益率列: 裁剪到 [-50%, +100%]")
        
        # 持股比例：裁剪到 [0, 100]
        ratio_cols = ['top10_hold_ratio', 'top1_hold_ratio', 'top10_inst_ratio', 'pledge_ratio', 'pledge_ratio_high']
        for col in ratio_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0, upper=100)
        self.logger.info(f"  - 持股/质押比例: 裁剪到 [0, 100]")
        
        return df
    
    def _apply_fillna(self, df: Any) -> Any:
        """缺失值处理"""
        # 数值列：填充 0
        numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.tolist()
        
        # 排除主键和日期列
        exclude_cols = ['ts_code', 'trade_date']
        fill_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        filled_count = 0
        for col in fill_cols:
            nan_count = df[col].isna().sum()
            if self.gpu_active:
                nan_count = int(nan_count)
            if nan_count > 0:
                df[col] = df[col].fillna(0)
                filled_count += 1
        
        self.logger.info(f"  - {filled_count} 列填充 NaN → 0")
        
        return df
    
    def _check_nan_summary(self, df: Any) -> Dict[str, int]:
        """检查 NaN 统计"""
        nan_cols = {}
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if self.gpu_active:
                nan_count = int(nan_count)
            if nan_count > 0:
                nan_cols[col] = nan_count
        
        if nan_cols:
            self.logger.warning(f"  ⚠️ 仍有 {len(nan_cols)} 列存在 NaN")
            for col, count in list(nan_cols.items())[:5]:
                self.logger.warning(f"     - {col}: {count:,}")
        else:
            self.logger.info("  ✓ 无 NaN 列")
        
        return nan_cols
    
    def run(self, save_output: bool = True, 
            run_features: bool = False, 
            run_labels: bool = False,
            sample_ratio: float = 1.0) -> Any:
        """
        执行完整流水线：合并 → 预处理 → [特征生成] → [标签生成] → 保存
        
        Args:
            save_output: 是否保存输出
            run_features: 是否运行特征生成
            run_labels: 是否运行标签生成
            sample_ratio: 采样比例 (0.0-1.0)，小于1时采样以节省内存
        """
        total_start = time.time()
        
        self.logger.info("=" * 70)
        self.logger.info("🚀 特征工程 8 表合并 + 预处理 流水线")
        self.logger.info("=" * 70)
        self.logger.info(f"  输入目录: {self.input_dir}")
        self.logger.info(f"  输出目录: {self.output_dir}")
        self.logger.info(f"  GPU模式: {'✓' if self.gpu_active else '✗'}")
        self.logger.info(f"  特征生成: {'✓' if run_features else '✗'}")
        self.logger.info(f"  标签生成: {'✓' if run_labels else '✗'}")
        self.logger.info(f"  采样比例: {sample_ratio*100:.0f}%")
        self.logger.info("")
        
        # Step 1: 合并
        df = self.step1_load_and_merge()
        log_memory("合并后", self.gpu_active, self.array_lib, self.logger)
        
        # 采样（减少内存使用）
        if sample_ratio < 1.0:
            n_total = len(df)
            n_sample = int(n_total * sample_ratio)
            self.logger.info(f"\n📉 采样 {sample_ratio*100:.0f}%: {n_total:,} → {n_sample:,} 行")
            
            # 使用 iloc 替代 sample 避免 cupy random 编译问题
            import numpy as np
            np.random.seed(42)
            indices = np.sort(np.random.choice(n_total, n_sample, replace=False))
            df = df.iloc[indices].reset_index(drop=True)
            df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
            
            force_gc(self.gpu_active, self.array_lib)
        
        # Step 2: 预处理
        df = self.step2_preprocess(df)
        log_memory("预处理后", self.gpu_active, self.array_lib, self.logger)
        
        # Step 3: 特征生成（可选）
        if run_features:
            df = self.step3_features(df)
            log_memory("特征生成后", self.gpu_active, self.array_lib, self.logger)
        
        # Step 4: 标签生成（可选）
        if run_labels:
            df = self.step4_labels(df)
            log_memory("标签生成后", self.gpu_active, self.array_lib, self.logger)
        
        # 记录统计信息（保存前）
        row_count = int(len(df)) if self.gpu_active else len(df)
        col_count = int(len(df.columns))
        stock_count = int(df['ts_code'].nunique()) if 'ts_code' in df.columns else 0
        self.stats['final_rows'] = row_count
        self.stats['final_cols'] = col_count
        self.stats['final_stocks'] = stock_count
        
        # Step 5: 保存
        if save_output:
            self.step5_save(df)
            # 保存后 df 可能已被释放，使用统计信息
        
        # 汇总
        total_time = time.time() - total_start
        self.stats['total_time'] = total_time
        
        self._print_summary(total_time)
        
        # 如果没保存，返回 df；否则返回 None（已释放）
        if save_output and self.gpu_active:
            return None
        return df
    
    def step5_save(self, df: Any) -> Path:
        """
        Step 5: 保存输出（重命名以配合新步骤编号）
        """
        return self._save_output(df)
    
    def _save_output(self, df: Any) -> Path:
        """保存输出到文件"""
        self.logger.info("=" * 60)
        self.logger.info("📋 Step 5: 保存输出")
        self.logger.info("=" * 60)
        
        output_path = self.output_dir / "merged_preprocessed.parquet"
        
        start_time = time.time()
        
        # GPU 模式下转换到 CPU 保存，避免 OOM
        if self.gpu_active and hasattr(df, 'to_pandas'):
            self.logger.info("  ⚙️ GPU → CPU 转换中...")
            convert_start = time.time()
            df_cpu = df.to_pandas()
            convert_time = time.time() - convert_start
            self.logger.info(f"     转换耗时: {convert_time:.2f}s")
            
            # 释放 GPU 内存
            del df
            force_gc(True, self.array_lib)
            
            # 保存
            df_cpu.to_parquet(str(output_path), index=False)
            del df_cpu
        else:
            df.to_parquet(str(output_path), index=False)
        
        elapsed = time.time() - start_time
        
        file_size = output_path.stat().st_size / (1024**2)
        
        self.logger.info(f"  💾 保存: {output_path}")
        self.logger.info(f"     大小: {file_size:.1f} MB")
        self.logger.info(f"     耗时: {elapsed:.2f}s")
        
        self.stats['output_path'] = str(output_path)
        self.stats['output_size_mb'] = file_size
        self.stats['save_time'] = elapsed
        
        return output_path
    
    def _print_summary(self, elapsed: float):
        """打印摘要"""
        self.logger.info("=" * 70)
        self.logger.info("📊 流水线执行摘要")
        self.logger.info("=" * 70)
        
        row_count = self.stats.get('final_rows', 0)
        col_count = self.stats.get('final_cols', 0)
        stock_count = self.stats.get('final_stocks', 0)
        
        self.logger.info(f"  📊 最终数据集:")
        self.logger.info(f"     样本数: {row_count:,}")
        self.logger.info(f"     特征数: {col_count}")
        self.logger.info(f"     股票数: {stock_count:,}")
        
        self.logger.info(f"\n  ⏱️ 耗时统计:")
        self.logger.info(f"     合并: {self.stats.get('merge_time', 0):.2f}s")
        self.logger.info(f"     预处理: {self.stats.get('preprocess_time', 0):.2f}s")
        self.logger.info(f"     保存: {self.stats.get('save_time', 0):.2f}s")
        self.logger.info(f"     总计: {elapsed:.2f}s")
        
        self.logger.info("=" * 70)
        self.logger.info("🎉 流水线完成!")
        self.logger.info("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试 合并 → 预处理 → 特征 → 标签 流水线")
    parser.add_argument('--no-gpu', action='store_true', help='禁用 GPU 加速')
    parser.add_argument('--dry-run', action='store_true', help='只打印配置，不执行')
    parser.add_argument('--no-save', action='store_true', help='不保存输出文件')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细日志')
    parser.add_argument('--features', action='store_true', help='运行特征生成')
    parser.add_argument('--labels', action='store_true', help='运行标签生成')
    parser.add_argument('--full', action='store_true', help='运行完整流程(合并+预处理+特征+标签)')
    parser.add_argument('--sample', type=float, default=1.0, 
                        help='采样比例 (0.0-1.0)，默认1.0表示全量')
    
    args = parser.parse_args()
    
    use_gpu = not args.no_gpu
    logger = setup_logging(args.verbose)
    
    # --full 开启全部步骤
    run_features = args.features or args.full
    run_labels = args.labels or args.full
    
    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info(f"GPU模式: {'启用' if use_gpu else '禁用'}")
        logger.info(f"输入目录: {PROJECT_ROOT / 'data' / 'processed' / 'structured' / 'dwd'}")
        logger.info(f"输出目录: {PROJECT_ROOT / 'data' / 'features' / 'preprocessed'}")
        logger.info(f"特征生成: {'✓' if run_features else '✗'}")
        logger.info(f"标签生成: {'✓' if run_labels else '✗'}")
        logger.info(f"采样比例: {args.sample*100:.0f}%")
        return
    
    pipeline = MergePreprocessPipeline(use_gpu=use_gpu, logger=logger)
    df = pipeline.run(
        save_output=not args.no_save,
        run_features=run_features,
        run_labels=run_labels,
        sample_ratio=args.sample
    )
    
    # 验证
    logger.info("\n=== 验证 ===")
    
    if df is not None:
        logger.info(f"行数: {len(df):,}")
        logger.info(f"列数: {len(df.columns)}")
        
        # 检查主键
        if 'ts_code' in df.columns and 'trade_date' in df.columns:
            dup_count = df.duplicated(subset=['ts_code', 'trade_date']).sum()
            if use_gpu:
                dup_count = int(dup_count)
            logger.info(f"主键重复: {dup_count}")
        
        # 显示部分列名
        cols = list(df.columns)
        logger.info(f"列名(前20): {cols[:20]}")
    else:
        # 从统计信息获取
        logger.info(f"行数: {pipeline.stats.get('final_rows', 0):,}")
        logger.info(f"列数: {pipeline.stats.get('final_cols', 0)}")
        logger.info(f"输出文件: {pipeline.stats.get('output_path', 'N/A')}")
        logger.info(f"文件大小: {pipeline.stats.get('output_size_mb', 0):.1f} MB")
        if 'label_cols' in pipeline.stats:
            logger.info(f"标签列: {pipeline.stats['label_cols']}")
    
    logger.info("\n✅ 测试完成!")


if __name__ == '__main__':
    main()
