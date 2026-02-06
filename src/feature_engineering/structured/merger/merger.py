"""
DWD表合并器 - 支持8表合并的升级版（内存优化版）
==================================

合并策略：主骨架 + 多维左连接 + 宏观广播
- 主骨架：dwd_stock_price (ts_code, trade_date) 作为行索引
- 多维左连接：fundamental, status, money_flow, chip, industry, event 按 (ts_code, trade_date) 左连接
- 宏观广播：macro_env 仅按 trade_date 广播（截面共享）

内存优化策略（解决 WSL 内存爆炸问题）：
1. 流式合并：加载一张表 → 合并 → 立即释放，避免8表同时驻留内存
2. 显式GC：merge后调用 gc.collect() + cupy.get_default_memory_pool().free_all_blocks()
3. 内存下压：float64 → float32 减少 50% 内存占用
4. 分批模式：可选按年份分批处理，进一步降低峰值内存
5. 避免拷贝：原地操作替代 .copy()
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List

import pandas as pd

from ..config import DataConfig, PipelineConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DataMerger:
    """
    DWD表合并器 - 处理8表合并逻辑
    
    支持的表：
    - 核心3表：price, fundamental, status
    - 扩展5表：money_flow, chip_structure, stock_industry, event_signal, macro_env
    """
    
    # 面板表（按 ts_code + trade_date 合并）
    PANEL_TABLES = ['fundamental', 'status', 'money_flow', 'chip', 'industry', 'event']
    
    # 宏观表（仅按 trade_date 广播）
    MACRO_TABLES = ['macro']
    
    # 合并键
    PANEL_KEYS = ['ts_code', 'trade_date']
    MACRO_KEYS = ['trade_date']
    
    def __init__(
        self, 
        data_config: Optional[DataConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        use_gpu: bool = True
    ):
        """
        初始化合并器
        
        Args:
            data_config: 数据路径配置
            pipeline_config: 管道配置
            use_gpu: 是否使用GPU加速
        """
        self.data_config = data_config or DataConfig()
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.use_gpu = use_gpu
        
        # 延迟导入 cuDF 和 cupy（用于内存管理）
        self._cudf = None
        self._cupy = None
        if use_gpu:
            try:
                import cudf
                import cupy as cp
                self._cudf = cudf
                self._cupy = cp
                logger.info("GPU模式启用 (cuDF + cupy)")
            except ImportError:
                logger.warning("cuDF不可用，回退到pandas")
                self.use_gpu = False
        
        # 禁用表缓存（流式合并不需要缓存）
        self._table_cache: dict[str, pd.DataFrame] = {}
        self._cache_enabled = False  # 默认禁用缓存以节省内存
    
    @property
    def df_lib(self):
        """返回当前使用的DataFrame库"""
        return self._cudf if self.use_gpu and self._cudf else pd
    
    def _force_gc(self):
        """强制垃圾回收，释放 GPU 和 CPU 内存"""
        gc.collect()
        
        if self.use_gpu and self._cupy:
            try:
                # 释放 cupy 内存池
                mempool = self._cupy.get_default_memory_pool()
                mempool.free_all_blocks()
                
                # 释放 pinned memory pool
                pinned_mempool = self._cupy.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()
                
                logger.debug("GPU内存已释放")
            except Exception as e:
                logger.debug(f"GPU内存释放失败: {e}")
    
    def _log_memory_usage(self, stage: str):
        """记录当前内存使用情况"""
        if self.use_gpu and self._cupy:
            try:
                mempool = self._cupy.get_default_memory_pool()
                used_bytes = mempool.used_bytes()
                total_bytes = mempool.total_bytes()
                logger.info(f"[{stage}] GPU内存: {used_bytes/1024**3:.2f}GB / {total_bytes/1024**3:.2f}GB")
            except:
                pass
    
    def _read_parquet(self, path: Path) -> Optional[pd.DataFrame]:
        """读取parquet文件"""
        if not path.exists():
            logger.warning(f"文件不存在: {path}")
            return None
        
        try:
            if self.use_gpu and self._cudf:
                df = self._cudf.read_parquet(str(path))
            else:
                df = pd.read_parquet(path)
            return df
        except Exception as e:
            logger.error(f"读取文件失败 {path}: {e}")
            return None
    
    def _downcast_float(self, df: pd.DataFrame) -> pd.DataFrame:
        """将float64列下压为float32以节省内存"""
        if df is None:
            return None
        
        float64_cols = df.select_dtypes(include=['float64']).columns
        if len(float64_cols) > 0:
            for col in float64_cols:
                df[col] = df[col].astype('float32')
            logger.debug(f"已下压 {len(float64_cols)} 列 float64 → float32")
        
        return df
    
    def load_table(self, table_name: str, downcast: bool = True) -> Optional[pd.DataFrame]:
        """
        加载单个DWD表
        
        Args:
            table_name: 表名（price, fundamental, status, money_flow, chip, industry, event, macro）
            downcast: 是否下压float类型
        
        Returns:
            DataFrame 或 None（如果加载失败）
        """
        # 检查缓存
        if table_name in self._table_cache:
            return self._table_cache[table_name]
        
        # 获取路径
        path_map = {
            'price': self.data_config.price_path,
            'fundamental': self.data_config.fundamental_path,
            'status': self.data_config.status_path,
            'money_flow': self.data_config.money_flow_path,
            'chip': self.data_config.chip_path,
            'industry': self.data_config.industry_path,
            'event': self.data_config.event_path,
            'macro': self.data_config.macro_path,
        }
        
        path = path_map.get(table_name)
        if path is None:
            logger.error(f"未知表名: {table_name}")
            return None
        
        # 读取
        df = self._read_parquet(path)
        if df is None:
            return None
        
        # 下压
        if downcast:
            df = self._downcast_float(df)
        
        # 缓存（仅在启用时）
        if self._cache_enabled:
            self._table_cache[table_name] = df
        logger.info(f"已加载 {table_name}: {len(df):,} 行, {len(df.columns)} 列")
        
        return df
    
    def load_all_tables(self) -> dict[str, pd.DataFrame]:
        """
        加载所有8个DWD表
        
        Returns:
            {table_name: DataFrame} 字典
        """
        all_tables = ['price', 'fundamental', 'status', 'money_flow', 'chip', 'industry', 'event', 'macro']
        loaded = {}
        
        for name in all_tables:
            df = self.load_table(name)
            if df is not None:
                loaded[name] = df
            else:
                logger.warning(f"跳过未加载的表: {name}")
        
        return loaded
    
    def merge_panel_data(
        self, 
        skeleton: pd.DataFrame, 
        panel_tables: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        合并面板数据（按 ts_code + trade_date 左连接）
        
        Args:
            skeleton: 主骨架表（price）
            panel_tables: 面板表字典（不含price和macro）
        
        Returns:
            合并后的DataFrame
        """
        result = skeleton.copy()
        
        for name, df in panel_tables.items():
            if df is None or len(df) == 0:
                logger.warning(f"跳过空表: {name}")
                continue
            
            # 确保合并键存在
            missing_keys = [k for k in self.PANEL_KEYS if k not in df.columns]
            if missing_keys:
                logger.warning(f"表 {name} 缺少合并键: {missing_keys}，跳过")
                continue
            
            # 预去重：按合并键去重，保留第一条
            dup_before = len(df)
            df = df.drop_duplicates(subset=self.PANEL_KEYS, keep='first')
            if len(df) < dup_before:
                logger.debug(f"表 {name} 去重: {dup_before} → {len(df)}")
            
            # 去除与skeleton重复的列（保留合并键）
            overlap_cols = set(df.columns) & set(result.columns) - set(self.PANEL_KEYS)
            if overlap_cols:
                # 使用后缀区分，但优先保留skeleton的列
                df = df.drop(columns=list(overlap_cols), errors='ignore')
            
            # 左连接
            before_rows = len(result)
            if self.use_gpu and self._cudf:
                result = result.merge(df, on=self.PANEL_KEYS, how='left')
            else:
                result = result.merge(df, on=self.PANEL_KEYS, how='left')
            
            logger.debug(f"合并 {name}: {before_rows} → {len(result)} 行, 新增 {len(df.columns) - len(self.PANEL_KEYS)} 列")
        
        return result
    
    def merge_macro_data(
        self, 
        panel_result: pd.DataFrame, 
        macro_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        合并宏观数据（仅按 trade_date 广播）
        
        Args:
            panel_result: 已合并的面板数据
            macro_df: 宏观表
        
        Returns:
            最终合并结果
        """
        if macro_df is None or len(macro_df) == 0:
            logger.warning("宏观表为空，跳过")
            return panel_result
        
        # 确保trade_date存在
        if 'trade_date' not in macro_df.columns:
            logger.error("宏观表缺少 trade_date 列")
            return panel_result
        
        # 预去重：按 trade_date 去重，保留第一条
        dup_before = len(macro_df)
        macro_df = macro_df.drop_duplicates(subset=['trade_date'], keep='first')
        if len(macro_df) < dup_before:
            logger.debug(f"宏观表去重: {dup_before} → {len(macro_df)}")
        
        # 去除重复列（保留trade_date）
        overlap_cols = set(macro_df.columns) & set(panel_result.columns) - {'trade_date'}
        if overlap_cols:
            macro_df = macro_df.drop(columns=list(overlap_cols), errors='ignore')
        
        # 左连接
        before_rows = len(panel_result)
        if self.use_gpu and self._cudf:
            result = panel_result.merge(macro_df, on='trade_date', how='left')
        else:
            result = panel_result.merge(macro_df, on='trade_date', how='left')
        
        logger.info(f"合并宏观: {before_rows} → {len(result)} 行, 新增 {len(macro_df.columns) - 1} 列")
        
        return result
    
    def sanity_check(self, result: pd.DataFrame, skeleton_rows: int) -> bool:
        """
        合理性检查
        
        Args:
            result: 合并结果
            skeleton_rows: 原始骨架行数
        
        Returns:
            是否通过检查
        """
        passed = True
        
        # 行数检查
        if len(result) != skeleton_rows:
            logger.error(f"行数不匹配: 预期 {skeleton_rows}, 实际 {len(result)}")
            passed = False
        else:
            logger.info(f"✓ 行数检查通过: {len(result):,} 行")
        
        # 必需字段检查
        required_fields = ['ts_code', 'trade_date', 'close', 'adj_factor']
        missing = [f for f in required_fields if f not in result.columns]
        if missing:
            logger.error(f"缺少必需字段: {missing}")
            passed = False
        else:
            logger.info(f"✓ 必需字段检查通过")
        
        # 扩展字段检查（来自各扩展表的关键字段）
        extended_fields = {
            'money_flow': ['net_mf_amount', 'buy_elg_amount'],
            'chip': ['market_congestion', 'top10_hold_ratio'],
            'industry': ['sw_l1_idx', 'sw_l2_idx'],
            'event': ['pledge_ratio', 'freeze_ratio'],
            'macro': ['shibor_1m', 'cpi_yoy'],
        }
        
        for table, fields in extended_fields.items():
            found = [f for f in fields if f in result.columns]
            if found:
                logger.info(f"✓ {table} 字段已合并: {found[:3]}...")
            else:
                logger.warning(f"⚠ {table} 字段未找到")
        
        # 内存统计
        if hasattr(result, 'memory_usage'):
            mem_mb = result.memory_usage(deep=True).sum() / 1024 / 1024
            logger.info(f"合并结果内存占用: {mem_mb:.1f} MB")
        
        return passed
    
    def filter_universe(
        self, 
        df: pd.DataFrame, 
        exclude_st: bool = True,
        exclude_suspended: bool = True,
        min_list_days: int = 60
    ) -> pd.DataFrame:
        """
        过滤股票池
        
        Args:
            df: 输入DataFrame
            exclude_st: 排除ST股票
            exclude_suspended: 排除停牌股票
            min_list_days: 最小上市天数
        
        Returns:
            过滤后的DataFrame
        """
        before_rows = len(df)
        
        # ST过滤
        if exclude_st and 'is_st' in df.columns:
            df = df[df['is_st'] != 1]
        
        # 停牌过滤
        if exclude_suspended and 'is_suspended' in df.columns:
            df = df[df['is_suspended'] != 1]
        
        # 上市天数过滤
        if min_list_days > 0 and 'list_days' in df.columns:
            df = df[df['list_days'] >= min_list_days]
        
        logger.info(f"股票池过滤: {before_rows:,} → {len(df):,} 行")
        return df
    
    def drop_columns(
        self, 
        df: pd.DataFrame, 
        drop_patterns: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        删除不需要的列
        
        Args:
            df: 输入DataFrame
            drop_patterns: 要删除的列名模式列表
        
        Returns:
            处理后的DataFrame
        """
        if drop_patterns is None:
            drop_patterns = ['_id', '_name', '_code']
        
        drop_cols = []
        for pattern in drop_patterns:
            drop_cols.extend([c for c in df.columns if pattern in c.lower()])
        
        # 保护关键列
        protected = ['ts_code', 'trade_date', 'industry_code', 'sw_l1_code', 'sw_l2_code']
        drop_cols = [c for c in drop_cols if c not in protected]
        
        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')
            logger.info(f"已删除 {len(drop_cols)} 列")
        
        return df
    
    def process(
        self,
        filter_universe: bool = True,
        drop_unnecessary: bool = True,
        save_result: bool = True,
        output_path: Optional[Path] = None,
        memory_optimized: bool = True  # 新增：内存优化模式
    ) -> pd.DataFrame:
        """
        执行完整合并流程（内存优化版）
        
        Args:
            filter_universe: 是否过滤股票池
            drop_unnecessary: 是否删除不需要的列
            save_result: 是否保存结果
            output_path: 输出路径
            memory_optimized: 是否启用内存优化模式（流式合并 + 显式GC）
        
        Returns:
            合并后的DataFrame
        """
        logger.info("=" * 60)
        logger.info("开始8表合并流程" + (" [内存优化模式]" if memory_optimized else ""))
        logger.info("=" * 60)
        
        if memory_optimized:
            return self._process_memory_optimized(
                filter_universe=filter_universe,
                drop_unnecessary=drop_unnecessary,
                save_result=save_result,
                output_path=output_path
            )
        else:
            return self._process_standard(
                filter_universe=filter_universe,
                drop_unnecessary=drop_unnecessary,
                save_result=save_result,
                output_path=output_path
            )
    
    def _process_memory_optimized(
        self,
        filter_universe: bool = True,
        drop_unnecessary: bool = True,
        save_result: bool = True,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        内存优化版合并流程：流式加载 + 显式GC
        
        策略：加载一张表 → 合并 → 释放 → 下一张
        避免 8 张表同时驻留内存
        """
        # Step 1: 加载主骨架（必须保留）
        logger.info("[Step 1] 加载主骨架...")
        self._log_memory_usage("开始")
        
        result = self.load_table('price', downcast=True)
        if result is None:
            raise ValueError("主骨架表(price)加载失败，无法继续")
        
        skeleton_rows = len(result)
        logger.info(f"主骨架: {skeleton_rows:,} 行")
        self._log_memory_usage("加载骨架后")
        
        # Step 2: 流式合并面板表（一次加载一张，合并后释放）
        logger.info("[Step 2] 流式合并面板数据...")
        
        for table_name in self.PANEL_TABLES:
            logger.info(f"  合并 {table_name}...")
            
            # 加载表
            df = self.load_table(table_name, downcast=True)
            if df is None or len(df) == 0:
                logger.warning(f"  跳过空表: {table_name}")
                continue
            
            # 确保合并键存在
            missing_keys = [k for k in self.PANEL_KEYS if k not in df.columns]
            if missing_keys:
                logger.warning(f"  表 {table_name} 缺少合并键: {missing_keys}，跳过")
                del df
                self._force_gc()
                continue
            
            # 预去重
            dup_before = len(df)
            df = df.drop_duplicates(subset=self.PANEL_KEYS, keep='first')
            if len(df) < dup_before:
                logger.debug(f"  表 {table_name} 去重: {dup_before} → {len(df)}")
            
            # 去除重复列
            overlap_cols = set(df.columns) & set(result.columns) - set(self.PANEL_KEYS)
            if overlap_cols:
                df = df.drop(columns=list(overlap_cols), errors='ignore')
            
            # 合并
            cols_before = len(result.columns)
            result = result.merge(df, on=self.PANEL_KEYS, how='left')
            cols_added = len(result.columns) - cols_before
            
            logger.info(f"  ✓ {table_name}: +{cols_added} 列")
            
            # 立即释放已合并的表
            del df
            self._force_gc()
        
        self._log_memory_usage("面板合并后")
        
        # Step 3: 合并宏观数据
        logger.info("[Step 3] 合并宏观数据...")
        macro_df = self.load_table('macro', downcast=True)
        if macro_df is not None and len(macro_df) > 0:
            # 预去重
            macro_df = macro_df.drop_duplicates(subset=['trade_date'], keep='first')
            
            # 去除重复列
            overlap_cols = set(macro_df.columns) & set(result.columns) - {'trade_date'}
            if overlap_cols:
                macro_df = macro_df.drop(columns=list(overlap_cols), errors='ignore')
            
            cols_before = len(result.columns)
            result = result.merge(macro_df, on='trade_date', how='left')
            cols_added = len(result.columns) - cols_before
            
            logger.info(f"  ✓ macro: +{cols_added} 列")
            
            del macro_df
            self._force_gc()
        
        self._log_memory_usage("宏观合并后")
        
        # Step 4: 合理性检查
        logger.info("[Step 4] 合理性检查...")
        self.sanity_check(result, skeleton_rows)
        
        # Step 5: 可选过滤
        if filter_universe:
            logger.info("[Step 5] 过滤股票池...")
            result = self.filter_universe(result)
            self._force_gc()
        
        # Step 6: 可选删除列
        if drop_unnecessary:
            logger.info("[Step 6] 删除冗余列...")
            result = self.drop_columns(result)
        
        # Step 7: 保存
        if save_result:
            if output_path is None:
                output_path = self.data_config.merged_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"[Step 7] 保存结果...")
            
            # GPU模式需要转回pandas
            if self.use_gpu and self._cudf and hasattr(result, 'to_pandas'):
                result_pd = result.to_pandas()
                result_pd.to_parquet(output_path, compression='snappy')
                del result_pd
            else:
                result.to_parquet(output_path, compression='snappy')
            
            logger.info(f"  ✓ 已保存: {output_path}")
        
        self._log_memory_usage("完成")
        logger.info("=" * 60)
        logger.info(f"合并完成: {len(result):,} 行, {len(result.columns)} 列")
        logger.info("=" * 60)
        
        return result
    
    def process_with_preprocessing(
        self,
        preprocessors: Optional[dict] = None,
        filter_universe: bool = True,
        drop_unnecessary: bool = True,
        save_result: bool = True,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        流式预处理+合并：读取→预处理→合并 循环
        
        内存优化策略：
        - 每次只加载一张表到显存
        - 立即预处理，然后与中间结果合并
        - 合并后释放原始表，只保留中间结果
        - 避免8张表同时驻留显存导致OOM
        
        Args:
            preprocessors: 预处理器字典 {'price': PricePreprocessor, ...}
            filter_universe: 是否过滤股票池
            drop_unnecessary: 是否删除不需要的列
            save_result: 是否保存结果
            output_path: 输出路径
        
        Returns:
            预处理并合并后的DataFrame
        """
        logger.info("=" * 60)
        logger.info("🚀 流式预处理+合并流程 [读取→预处理→合并 循环]")
        logger.info("=" * 60)
        
        if preprocessors is None:
            preprocessors = {}
        
        # Step 1: 加载并预处理主骨架（price）
        logger.info("[Step 1] 加载并预处理主骨架(price)...")
        self._log_memory_usage("开始")
        
        result = self.load_table('price', downcast=True)
        if result is None:
            raise ValueError("主骨架表(price)加载失败")
        
        # 预处理 price
        if 'price' in preprocessors:
            logger.info("  📊 预处理 price...")
            result = preprocessors['price'].process(result)
        
        # 统一 trade_date 类型为 datetime64[ns]（确保后续合并一致）
        if 'trade_date' in result.columns:
            if self.use_gpu and self._cudf:
                if not str(result['trade_date'].dtype).startswith('datetime64'):
                    result['trade_date'] = self._cudf.to_datetime(result['trade_date'])
            else:
                # pandas: 统一转为 datetime64[ns]
                if not str(result['trade_date'].dtype).startswith('datetime64'):
                    result['trade_date'] = pd.to_datetime(result['trade_date'])
                # 确保是 datetime64[ns] 而非 datetime64[us]
                if str(result['trade_date'].dtype) != 'datetime64[ns]':
                    result['trade_date'] = result['trade_date'].astype('datetime64[ns]')
        
        skeleton_rows = len(result)
        logger.info(f"  ✓ 主骨架: {skeleton_rows:,} 行, {len(result.columns)} 列")
        self._log_memory_usage("骨架预处理后")
        
        # Step 2: 流式 读取→预处理→合并 面板表
        logger.info("[Step 2] 流式处理面板表...")
        
        for table_name in self.PANEL_TABLES:
            logger.info(f"  处理 {table_name}...")
            
            # 加载表
            df = self.load_table(table_name, downcast=True)
            if df is None or len(df) == 0:
                logger.warning(f"    跳过空表: {table_name}")
                continue
            
            # 预处理
            if table_name in preprocessors:
                logger.info(f"    📊 预处理 {table_name}...")
                df = preprocessors[table_name].process(df)
            
            # 确保合并键存在
            missing_keys = [k for k in self.PANEL_KEYS if k not in df.columns]
            if missing_keys:
                logger.warning(f"    表 {table_name} 缺少合并键: {missing_keys}，跳过")
                del df
                self._force_gc()
                continue
            
            # 统一 trade_date 类型（确保与骨架表一致：datetime64[ns]）
            if 'trade_date' in df.columns:
                if self.use_gpu and self._cudf:
                    # cuDF: 确保都是 datetime64[ns]
                    if not str(df['trade_date'].dtype).startswith('datetime64'):
                        df['trade_date'] = self._cudf.to_datetime(df['trade_date'])
                else:
                    # pandas: 统一转为 datetime64[ns]
                    if not str(df['trade_date'].dtype).startswith('datetime64'):
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                    # 确保与骨架表类型完全一致
                    skeleton_dtype = str(result['trade_date'].dtype)
                    if str(df['trade_date'].dtype) != skeleton_dtype:
                        # 统一为 datetime64[ns]
                        df['trade_date'] = pd.to_datetime(df['trade_date']).astype('datetime64[ns]')
                        if str(result['trade_date'].dtype) != 'datetime64[ns]':
                            result['trade_date'] = pd.to_datetime(result['trade_date']).astype('datetime64[ns]')
            
            # 预去重
            dup_before = len(df)
            df = df.drop_duplicates(subset=self.PANEL_KEYS, keep='first')
            if len(df) < dup_before:
                logger.debug(f"    去重 {table_name}: {dup_before} → {len(df)}")
            
            # 去除重复列
            overlap_cols = set(df.columns) & set(result.columns) - set(self.PANEL_KEYS)
            if overlap_cols:
                df = df.drop(columns=list(overlap_cols), errors='ignore')
            
            # 合并
            cols_before = len(result.columns)
            result = result.merge(df, on=self.PANEL_KEYS, how='left')
            cols_added = len(result.columns) - cols_before
            
            logger.info(f"    ✓ {table_name}: +{cols_added} 列")
            
            # 立即释放
            del df
            self._force_gc()
        
        self._log_memory_usage("面板处理后")
        
        # Step 3: 流式处理宏观表
        logger.info("[Step 3] 处理宏观表...")
        macro_df = self.load_table('macro', downcast=True)
        if macro_df is not None and len(macro_df) > 0:
            # 预处理
            if 'macro' in preprocessors:
                logger.info("    📊 预处理 macro...")
                macro_df = preprocessors['macro'].process(macro_df)
            
            # 统一 trade_date 类型（确保与骨架表一致）
            if 'trade_date' in macro_df.columns:
                if self.use_gpu and self._cudf:
                    if not str(macro_df['trade_date'].dtype).startswith('datetime64'):
                        macro_df['trade_date'] = self._cudf.to_datetime(macro_df['trade_date'])
                else:
                    # pandas: 统一为 datetime64[ns]
                    if not str(macro_df['trade_date'].dtype).startswith('datetime64'):
                        macro_df['trade_date'] = pd.to_datetime(macro_df['trade_date'])
                    if str(macro_df['trade_date'].dtype) != 'datetime64[ns]':
                        macro_df['trade_date'] = macro_df['trade_date'].astype('datetime64[ns]')
            
            # 去重和合并
            macro_df = macro_df.drop_duplicates(subset=['trade_date'], keep='first')
            
            overlap_cols = set(macro_df.columns) & set(result.columns) - {'trade_date'}
            if overlap_cols:
                macro_df = macro_df.drop(columns=list(overlap_cols), errors='ignore')
            
            cols_before = len(result.columns)
            result = result.merge(macro_df, on='trade_date', how='left')
            cols_added = len(result.columns) - cols_before
            
            logger.info(f"    ✓ macro: +{cols_added} 列")
            
            del macro_df
            self._force_gc()
        
        self._log_memory_usage("宏观处理后")
        
        # Step 4: 合理性检查
        logger.info("[Step 4] 合理性检查...")
        self.sanity_check(result, skeleton_rows)
        
        # Step 5: 可选过滤
        if filter_universe:
            logger.info("[Step 5] 过滤股票池...")
            result = self.filter_universe(result)
            self._force_gc()
        
        # Step 6: 可选删除列
        if drop_unnecessary:
            logger.info("[Step 6] 删除冗余列...")
            result = self.drop_columns(result)
        
        # Step 7: 保存
        if save_result:
            if output_path is None:
                output_path = self.data_config.merged_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"[Step 7] 保存结果...")
            
            # GPU模式需要转回pandas
            if self.use_gpu and self._cudf and hasattr(result, 'to_pandas'):
                result_pd = result.to_pandas()
                result_pd.to_parquet(output_path, compression='snappy')
                del result_pd
            else:
                result.to_parquet(output_path, compression='snappy')
            
            logger.info(f"  ✓ 已保存: {output_path}")
        
        self._log_memory_usage("完成")
        logger.info("=" * 60)
        logger.info(f"🎉 流式预处理+合并完成: {len(result):,} 行, {len(result.columns)} 列")
        logger.info("=" * 60)
        
        return result
    
    def _process_standard(
        self,
        filter_universe: bool = True,
        drop_unnecessary: bool = True,
        save_result: bool = True,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """标准合并流程（原有逻辑，保留向后兼容）"""
        # Step 1: 加载所有表
        logger.info("[Step 1] 加载DWD表...")
        self._cache_enabled = True  # 启用缓存
        tables = self.load_all_tables()
        
        if 'price' not in tables:
            raise ValueError("主骨架表(price)加载失败，无法继续")
        
        skeleton = tables['price']
        skeleton_rows = len(skeleton)
        logger.info(f"主骨架: {skeleton_rows:,} 行")
        
        # Step 2: 合并面板数据
        logger.info("[Step 2] 合并面板数据...")
        panel_tables = {
            k: v for k, v in tables.items() 
            if k in self.PANEL_TABLES and v is not None
        }
        result = self.merge_panel_data(skeleton, panel_tables)
        
        # Step 3: 合并宏观数据
        logger.info("[Step 3] 合并宏观数据...")
        macro_df = tables.get('macro')
        if macro_df is not None:
            result = self.merge_macro_data(result, macro_df)
        
        # Step 4: 合理性检查
        logger.info("[Step 4] 合理性检查...")
        self.sanity_check(result, skeleton_rows)
        
        # Step 5: 可选过滤
        if filter_universe:
            logger.info("[Step 5] 过滤股票池...")
            result = self.filter_universe(result)
        
        # Step 6: 可选删除列
        if drop_unnecessary:
            logger.info("[Step 6] 删除冗余列...")
            result = self.drop_columns(result)
        
        # Step 7: 保存
        if save_result:
            if output_path is None:
                output_path = self.data_config.merged_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # GPU模式需要转回pandas
            if self.use_gpu and self._cudf and hasattr(result, 'to_pandas'):
                result_pd = result.to_pandas()
                result_pd.to_parquet(output_path, compression='snappy')
            else:
                result.to_parquet(output_path, compression='snappy')
            
            logger.info(f"✓ 已保存: {output_path}")
        
        logger.info("=" * 60)
        logger.info(f"合并完成: {len(result):,} 行, {len(result.columns)} 列")
        logger.info("=" * 60)
        
        return result


def merge_tables(
    data_config: Optional[DataConfig] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    use_gpu: bool = True,
    memory_optimized: bool = True,  # 默认启用内存优化
    **kwargs
) -> pd.DataFrame:
    """
    便捷函数：执行DWD表合并
    
    Args:
        data_config: 数据配置
        pipeline_config: 管道配置
        use_gpu: 是否使用GPU
        memory_optimized: 是否启用内存优化模式（解决 WSL 内存溢出）
        **kwargs: 传递给 DataMerger.process() 的参数
    
    Returns:
        合并后的DataFrame
    """
    merger = DataMerger(
        data_config=data_config,
        pipeline_config=pipeline_config,
        use_gpu=use_gpu
    )
    return merger.process(memory_optimized=memory_optimized, **kwargs)


if __name__ == '__main__':
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    # 快速测试（使用内存优化模式）
    config = DataConfig()
    result = merge_tables(
        data_config=config,
        use_gpu=True,
        memory_optimized=True,  # 启用内存优化，避免 WSL OOM
        filter_universe=True,
        drop_unnecessary=False,  # 先保留所有列观察
        save_result=True
    )
    
    print(f"\n最终结果: {result.shape}")
    print(f"列: {list(result.columns)[:20]}...")
