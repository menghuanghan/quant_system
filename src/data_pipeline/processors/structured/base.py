"""
DWD层处理器基类 - 强制cuDF GPU加速版本

所有数据操作全部使用cuDF在GPU上执行，完全不依赖pandas
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Optional
from datetime import datetime

import cudf
import cupy as cp

from .config import (
    DATA_SOURCE_PATHS,
    DWD_OUTPUT_CONFIG,
    PROCESSING_CONFIG,
)

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
        
        # 降级：逐文件读取（更稳定但稍慢）
        logger.info(f"使用逐文件读取模式...")
        dfs = []
        failed_count = 0
        
        for i, f in enumerate(files):
            try:
                df = cudf.read_parquet(str(f), columns=columns)
                if len(df) > 0:
                    dfs.append(df)
            except Exception as e:
                failed_count += 1
                if failed_count <= 3:  # 只记录前3个错误
                    logger.debug(f"读取文件失败 {f.name}: {e}")
            
            # 进度提示（每1000个文件）
            if (i + 1) % 1000 == 0:
                logger.info(f"已读取 {i+1}/{len(files)} 个文件...")
        
        if failed_count > 0:
            logger.warning(f"共 {failed_count} 个文件读取失败")
        
        if not dfs:
            return cudf.DataFrame()
        
        result = cudf.concat(dfs, ignore_index=True)
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
) -> cudf.Series:
    """计算VWAP - GPU版本"""
    safe_volume = volume.where(volume > 0, 1)
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
