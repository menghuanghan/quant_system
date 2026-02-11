"""
DWD层处理器基类 - 强制cuDF GPU加速版本

所有数据操作全部使用cuDF在GPU上执行，完全不依赖pandas

金额单位统一规范：
    - 所有金额字段在DWD层统一转换为"元"
    - 原始数据单位参考：
        * stock_daily.amount: 千元 → ×1000
        * daily_basic.total_mv/circ_mv: 万元 → ×10000
        * money_flow 资金流字段: 万元 → ×10000
        * margin_detail 两融余额: 已是元
        * top_list 龙虎榜: 已是元
        * hsgt_flow 沪深港通: 百万元/万元 → 代码中已处理
        * block_trade 大宗交易金额: 万元 → ×10000
        * index_daily.amount: 千元 → ×1000
        * repo_daily.amount: 万元 → ×10000

数据类型规范：
    - 从DWD层开始，所有float64字段转换为float32以节省内存
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Optional, Dict
from datetime import datetime

import cudf
import cupy as cp

from .config import (
    DATA_SOURCE_PATHS,
    DWD_OUTPUT_CONFIG,
    PROCESSING_CONFIG,
)


# ============================================================================
# 金额单位转换配置
# ============================================================================
class AmountUnitConfig:
    """金额单位转换配置"""
    
    # 千元 → 元
    QIAN_YUAN_TO_YUAN = 1000.0
    
    # 万元 → 元
    WAN_YUAN_TO_YUAN = 10000.0
    
    # 百万元 → 元
    BAIWAN_YUAN_TO_YUAN = 1000000.0

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    DWD层处理器基类 - 强制GPU加速版本
    
    所有数据操作全部使用cuDF，完全不依赖pandas
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.use_gpu = use_gpu
        if not self.use_gpu:
            raise ValueError("BaseProcessor 仅支持GPU模式，请设置 use_gpu=True")
        
        logger.info("使用cuDF进行GPU加速处理")
        
        self.start_date = start_date or PROCESSING_CONFIG.start_date
        self.end_date = end_date or PROCESSING_CONFIG.end_date
        
        self._trade_dates_cache: Optional[List[str]] = None
        self._stock_list_cache: Optional[cudf.DataFrame] = None
    
    def to_cpu(self, df: cudf.DataFrame):
        """cuDF -> pandas（仅用于最终输出/保存）"""
        if hasattr(df, 'to_arrow'):
            return df.to_arrow().to_pandas()
        return df
    
    # ========================================================================
    # 金额单位转换方法
    # ========================================================================
    def convert_amount_columns(
        self,
        df: cudf.DataFrame,
        columns: List[str],
        multiplier: float,
        unit_desc: str = "",
    ) -> cudf.DataFrame:
        """
        将金额列乘以系数，统一为元
        
        Args:
            df: DataFrame
            columns: 需要转换的列名列表
            multiplier: 乘数(如 1000 表示千元→元, 10000 表示万元→元)
            unit_desc: 单位描述(用于日志)
            
        Returns:
            转换后的 DataFrame
        """
        converted = []
        for col in columns:
            if col in df.columns:
                df[col] = df[col] * multiplier
                converted.append(col)
        
        if converted:
            logger.info(f"金额单位转换({unit_desc}→元): {converted}, ×{multiplier:.0f}")
        
        return df
    
    def convert_qian_yuan_to_yuan(
        self,
        df: cudf.DataFrame,
        columns: List[str],
    ) -> cudf.DataFrame:
        """千元 → 元 (×1000)"""
        return self.convert_amount_columns(
            df, columns, AmountUnitConfig.QIAN_YUAN_TO_YUAN, "千元"
        )
    
    def convert_wan_yuan_to_yuan(
        self,
        df: cudf.DataFrame,
        columns: List[str],
    ) -> cudf.DataFrame:
        """万元 → 元 (×10000)"""
        return self.convert_amount_columns(
            df, columns, AmountUnitConfig.WAN_YUAN_TO_YUAN, "万元"
        )
    
    # ========================================================================
    # 数据类型压缩方法
    # ========================================================================
    def convert_float64_to_float32(
        self,
        df: cudf.DataFrame,
        exclude_columns: Optional[List[str]] = None,
    ) -> cudf.DataFrame:
        """
        将所有 float64 列转换为 float32 以节省内存
        
        注意：在衍生指标计算完成后、保存前调用
        
        Args:
            df: DataFrame
            exclude_columns: 需要保持 float64 的列(可选)
            
        Returns:
            转换后的 DataFrame
        """
        exclude_set = set(exclude_columns) if exclude_columns else set()
        converted = []
        
        for col in df.columns:
            if col in exclude_set:
                continue
            if str(df[col].dtype) == 'float64':
                df[col] = df[col].astype('float32')
                converted.append(col)
        
        if converted:
            logger.info(f"float64→float32: 转换了 {len(converted)} 个字段")
        
        return df
    
    def read_parquet(
        self,
        path: Union[str, Path],
        columns: Optional[List[str]] = None,
    ) -> cudf.DataFrame:
        """读取Parquet文件（cuDF直接读取）"""
        path = Path(path)
        if not path.exists():
            logger.warning(f"文件不存在: {path}")
            return cudf.DataFrame()
        
        try:
            return cudf.read_parquet(str(path), columns=columns)
        except Exception as e:
            logger.error(f"读取文件失败 {path}: {e}")
            return cudf.DataFrame()
    
    def read_parquet_dir(
        self,
        dir_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        pattern: str = "*.parquet",
    ) -> cudf.DataFrame:
        """
        读取目录下所有Parquet文件（纯cuDF GPU加速）
        
        优化策略：
        1. 指定columns避免schema不一致
        2. 批量读取优先，失败时降级逐文件
        3. 并行处理提升性能
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.warning(f"目录不存在: {dir_path}")
            return cudf.DataFrame()
        
        files = sorted(dir_path.glob(pattern))
        if not files:
            logger.warning(f"目录下没有匹配的文件: {dir_path}")
            return cudf.DataFrame()
        
        logger.info(f"从 {dir_path} 读取 {len(files)} 个文件...")
        
        # cuDF批量读取：指定columns可以避免大部分schema不一致问题
        file_paths = [str(f) for f in files]
        
        # 尝试批量读取（当指定columns时成功率更高）
        if columns is not None:
            try:
                result = cudf.read_parquet(file_paths, columns=columns)
                logger.info(f"批量读取成功，合并后共 {len(result)} 行数据")
                return result
            except Exception as e:
                logger.debug(f"批量读取失败: {e}")
        
        # 降级：多线程并行逐文件读取（性能优化版）
        logger.info(f"使用多线程并行读取模式...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # 线程安全的结果收集
        dfs = []
        dfs_lock = threading.Lock()
        failed_count = [0]  # 使用列表以便在嵌套函数中修改
        progress = [0]
        
        def read_single_file(file_path):
            """读取单个文件（线程安全）"""
            try:
                # 使用 pandas 读取再转 cuDF（避免 GPU 多线程冲突）
                import pandas as pd
                df_pd = pd.read_parquet(str(file_path), columns=columns)
                if len(df_pd) > 0:
                    return df_pd
            except Exception as e:
                with dfs_lock:
                    failed_count[0] += 1
                    if failed_count[0] <= 3:
                        logger.debug(f"读取文件失败 {file_path.name}: {e}")
            return None
        
        # 使用线程池并行读取（8个线程）
        max_workers = min(8, len(files))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(read_single_file, f): f for f in files}
            
            for future in as_completed(futures):
                result_df = future.result()
                if result_df is not None:
                    with dfs_lock:
                        dfs.append(result_df)
                        progress[0] += 1
                        if progress[0] % 1000 == 0:
                            logger.info(f"已读取 {progress[0]}/{len(files)} 个文件...")
        
        if failed_count[0] > 0:
            logger.warning(f"共 {failed_count[0]} 个文件读取失败")
        
        if not dfs:
            return cudf.DataFrame()
        
        # 使用 pandas concat（因为多线程读取返回的是 pandas DataFrames）
        import pandas as pd
        
        # 统一列类型：pandas concat 要求所有 DataFrame 同类型
        # 收集所有列的候选类型，优先使用数值类型
        all_cols = set()
        for df in dfs:
            all_cols.update(df.columns.tolist())
        
        col_dtype_map = {}
        date_cols = set()  # 需要转换为字符串的日期列
        
        for col in all_cols:
            type_set = set()
            for df in dfs:
                if col in df.columns:
                    type_set.add(str(df[col].dtype))
            
            if len(type_set) <= 1:
                continue  # 类型一致，无需转换
            
            # datetime 类型混合 object → 标记为日期列（后续转字符串）
            has_datetime = any('datetime' in t or 'date' in t.lower() for t in type_set)
            if has_datetime:
                date_cols.add(col)
                continue
            
            # 数值类型混合 object → 统一为数值类型
            numeric_dtype = None
            for t in type_set:
                if t != 'object':
                    if numeric_dtype is None:
                        numeric_dtype = t
                    elif t.startswith('float') or numeric_dtype.startswith('int'):
                        numeric_dtype = 'float64'
            if numeric_dtype is not None:
                col_dtype_map[col] = numeric_dtype
        
        # 应用类型转换
        if col_dtype_map or date_cols:
            for i, df in enumerate(dfs):
                needs_cast = False
                for col, target_dtype in col_dtype_map.items():
                    if col in df.columns and str(df[col].dtype) != str(target_dtype):
                        needs_cast = True
                        break
                # 检查是否有日期列需要转换
                for col in date_cols:
                    if col in df.columns:
                        needs_cast = True
                        break
                
                if needs_cast:
                    df_copy = df.copy()
                    # 处理日期列：统一转为字符串
                    for col in date_cols:
                        if col in df_copy.columns:
                            try:
                                # 处理各种日期类型：datetime, date, datetime64
                                df_copy[col] = df_copy[col].apply(
                                    lambda x: str(x) if x is not None and pd.notna(x) else None
                                )
                            except Exception:
                                pass
                    
                    for col, target_dtype in col_dtype_map.items():
                        if col in df_copy.columns and str(df_copy[col].dtype) != str(target_dtype):
                            # 安全类型转换：处理 None/NaN 值
                            try:
                                if 'int' in str(target_dtype):
                                    # 整数类型：先转 float 再转 int（处理 NaN）
                                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(target_dtype)
                                else:
                                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype(target_dtype)
                            except Exception:
                                # 转换失败时保持原类型
                                pass
                    dfs[i] = df_copy
            logger.info(f"已统一 {len(col_dtype_map) + len(date_cols)} 个列的类型")
        
        # 合并前最终检查：确保所有 object 列中的 datetime.date 转为字符串
        # （避免 cuDF 无法处理 datetime.date 对象的问题）
        import datetime as dt
        for i, df in enumerate(dfs):
            converted = False
            for col in df.columns:
                if str(df[col].dtype) == 'object' and len(df) > 0:
                    sample = df[col].iloc[0]
                    if isinstance(sample, (dt.date, dt.datetime)) and not isinstance(sample, str):
                        if not converted:
                            df = df.copy()
                            converted = True
                        df[col] = df[col].apply(lambda x: str(x) if x is not None and pd.notna(x) else None)
            if converted:
                dfs[i] = df
        
        # 合并 pandas DataFrames 然后转换为 cuDF
        result_pd = pd.concat(dfs, ignore_index=True)
        result = cudf.from_pandas(result_pd)
        logger.info(f"合并后共 {len(result)} 行数据")
        return result
    
    def save_parquet(
        self,
        df: cudf.DataFrame,
        path: Union[str, Path],
        index: bool = False,
    ):
        """保存DataFrame到Parquet文件（cuDF直接保存，无需转pandas）"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # cuDF直接保存parquet，更快且避免类型转换问题
        df.to_parquet(str(path), index=index, compression=PROCESSING_CONFIG.compression)
        
        logger.info(f"数据已保存到 {path}，共 {len(df)} 行")
    
    def normalize_date_column(
        self,
        df: cudf.DataFrame,
        column: str,
    ) -> cudf.DataFrame:
        """标准化日期列为YYYY-MM-DD格式（纯cuDF操作）"""
        if column not in df.columns:
            return df
        
        date_col = df[column].astype(str)
        
        # 检测格式
        if len(df) > 0:
            sample = str(date_col.iloc[0].item() if hasattr(date_col.iloc[0], 'item') else date_col.iloc[0])
            if '-' not in sample and len(sample) == 8:
                # YYYYMMDD -> YYYY-MM-DD
                df[column] = (
                    date_col.str.slice(0, 4) + "-" +
                    date_col.str.slice(4, 6) + "-" +
                    date_col.str.slice(6, 8)
                )
        
        return df
    
    def get_trade_dates(self) -> List[str]:
        """获取时间范围内的所有交易日（纯cuDF操作）"""
        if self._trade_dates_cache is not None:
            return self._trade_dates_cache
        
        # 用cuDF读取交易日历
        cal_df = cudf.read_parquet(str(DATA_SOURCE_PATHS.trade_calendar))
        
        # 过滤交易日
        cal_df = cal_df[cal_df['is_open'] == 1]
        
        # 标准化日期格式
        cal_df = self.normalize_date_column(cal_df, 'cal_date')
        
        # 过滤日期范围
        cal_df = cal_df[
            (cal_df['cal_date'] >= self.start_date) &
            (cal_df['cal_date'] <= self.end_date)
        ]
        
        # 去重并排序
        trade_dates = sorted(cal_df['cal_date'].unique().to_arrow().to_pylist())
        
        self._trade_dates_cache = trade_dates
        logger.info(f"交易日范围: {trade_dates[0]} 至 {trade_dates[-1]}，共 {len(trade_dates)} 个交易日")
        
        return trade_dates
    
    def get_stock_list(self) -> cudf.DataFrame:
        """获取A股股票列表（纯cuDF）"""
        if self._stock_list_cache is not None:
            return self._stock_list_cache
        
        stock_df = cudf.read_parquet(str(DATA_SOURCE_PATHS.stock_list_a))
        
        stock_df = self.normalize_date_column(stock_df, 'list_date')
        stock_df = stock_df[stock_df['list_date'] <= self.end_date]
        
        self._stock_list_cache = stock_df
        logger.info(f"股票总数: {len(stock_df)}")
        
        return stock_df
    
    def build_skeleton_table(
        self,
        stock_codes: Optional[List[str]] = None,
    ) -> cudf.DataFrame:
        """构建全量骨架表（日期 × 股票）- 纯GPU笛卡尔积"""
        trade_dates = self.get_trade_dates()
        
        if stock_codes is None:
            stock_list = self.get_stock_list()
            stock_codes = stock_list['ts_code'].to_arrow().to_pylist()
        
        logger.info(f"构建骨架表: {len(trade_dates)} 交易日 × {len(stock_codes)} 股票")
        
        # cuDF笛卡尔积
        dates_df = cudf.DataFrame({'trade_date': trade_dates, '_key': 1})
        stocks_df = cudf.DataFrame({'ts_code': stock_codes, '_key': 1})
        skeleton = dates_df.merge(stocks_df, on='_key').drop(columns=['_key'])
        
        logger.info(f"骨架表大小: {len(skeleton)} 行")
        return skeleton
    
    @abstractmethod
    def process(self) -> cudf.DataFrame:
        pass
    
    @abstractmethod
    def save(self, df: cudf.DataFrame):
        pass
    
    def run(self) -> cudf.DataFrame:
        logger.info(f"开始处理 {self.__class__.__name__}...")
        start_time = datetime.now()
        
        df = self.process()
        self.save(df)
        
        elapsed = datetime.now() - start_time
        logger.info(f"{self.__class__.__name__} 处理完成，耗时 {elapsed}")
        
        return df


# ============== GPU工具函数 ==============

def calculate_vwap_gpu(
    amount: cudf.Series,
    volume: cudf.Series,
    adj_factor: Optional[cudf.Series] = None,
    amount_unit: str = "yuan",
) -> cudf.Series:
    """
    计算VWAP（成交均价）- GPU版本
    
    Args:
        amount: 成交额（单位由 amount_unit 指定）
        volume: 成交量（手，1手=100股）
        adj_factor: 复权因子（可选，用于计算后复权VWAP）
        amount_unit: 成交额单位，"yuan"=元(默认), "qian"=千元
        
    Returns:
        VWAP（元/股）
    """
    safe_volume = volume.where(volume > 0, 1)
    
    # 根据amount单位计算VWAP
    if amount_unit == "yuan":
        # amount 已是元：VWAP = amount / (vol * 100)
        vwap = amount / (safe_volume * 100)
    else:
        # amount 是千元（兼容旧逻辑）：VWAP = (amount * 1000) / (vol * 100)
        vwap = (amount * 1000) / (safe_volume * 100)
    
    vwap = vwap.where(volume > 0, cp.nan)
    
    if adj_factor is not None:
        vwap = vwap * adj_factor
    
    return vwap


def ffill_by_group_gpu(
    df: cudf.DataFrame,
    group_col: str,
    value_cols: List[str],
    sort_col: str,
) -> cudf.DataFrame:
    """
    按分组前向填充 - GPU版本
    
    cuDF支持groupby().ffill()
    """
    df = df.sort_values([group_col, sort_col])
    
    for col in value_cols:
        if col in df.columns:
            df[col] = df.groupby(group_col)[col].ffill()
    
    return df
