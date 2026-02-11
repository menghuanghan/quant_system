"""
MarketDataProcessor - 基础量价宽表处理器（纯cuDF GPU版本）

生成 dwd_stock_price 宽表
"""

import logging
from typing import Optional, List

import cudf
import cupy as cp

from .base import BaseProcessor, calculate_vwap_gpu, ffill_by_group_gpu
from .config import (
    DATA_SOURCE_PATHS,
    DWD_OUTPUT_CONFIG,
    PROCESSING_CONFIG,
)

logger = logging.getLogger(__name__)


class MarketDataProcessor(BaseProcessor):
    """
    基础量价宽表处理器 - 纯GPU版本
    
    输出字段：
        - trade_date, ts_code: 主键
        - open, high, low, close, pre_close: 不复权价格
        - open_hfq, high_hfq, low_hfq, close_hfq: 后复权价格
        - vol, amount: 成交量/额
        - adj_factor: 复权因子
        - vwap, vwap_hfq: 成交均价
        - return_1d: 日涨跌幅
        - turnover: 换手率
        - is_trading: 交易状态
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        super().__init__(use_gpu=use_gpu, start_date=start_date, end_date=end_date)
        self.output_path = DWD_OUTPUT_CONFIG.output_dir / DWD_OUTPUT_CONFIG.stock_price
    
    def _load_stock_daily(self) -> cudf.DataFrame:
        """加载股票日线数据"""
        logger.info("加载股票日线数据...")
        
        df = self.read_parquet_dir(
            DATA_SOURCE_PATHS.stock_daily_dir,
            columns=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 
                     'pre_close', 'change', 'pct_chg', 'vol', 'amount']
        )
        
        if len(df) == 0:
            raise ValueError("无法加载股票日线数据")
        
        df = self.normalize_date_column(df, 'trade_date')
        # 注意：这里不按日期过滤，保留全部历史数据用于计算 return_1d
        # 最后输出时再按日期范围过滤
        df = df[df['trade_date'] <= self.end_date]
        
        logger.info(f"加载股票日线数据完成，共 {len(df)} 行")
        return df
    
    def _load_adj_factor(self) -> cudf.DataFrame:
        """加载复权因子数据"""
        logger.info("加载复权因子数据...")
        
        df = self.read_parquet_dir(
            DATA_SOURCE_PATHS.adj_factor_dir,
            columns=['ts_code', 'trade_date', 'adj_factor']
        )
        
        if len(df) == 0:
            raise ValueError("无法加载复权因子数据")
        
        df = self.normalize_date_column(df, 'trade_date')
        # 注意：这里不按日期过滤，保留全部历史数据用于计算后复权价格和 return_1d
        df = df[df['trade_date'] <= self.end_date]
        
        logger.info(f"加载复权因子数据完成，共 {len(df)} 行")
        return df
    
    def _load_daily_basic(self) -> cudf.DataFrame:
        """加载每日基本指标（换手率等）"""
        logger.info("加载每日基本指标数据...")
        
        df = self.read_parquet_dir(
            DATA_SOURCE_PATHS.daily_basic_dir,
            columns=['ts_code', 'trade_date', 'turnover_rate', 'turnover_rate_f']
        )
        
        if len(df) == 0:
            logger.warning("无法加载每日基本指标数据")
            return cudf.DataFrame() if self.use_gpu else None
        
        df = self.normalize_date_column(df, 'trade_date')
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载每日基本指标数据完成，共 {len(df)} 行")
        return df
    
    def _load_suspend_info(self) -> cudf.DataFrame:
        """加载停牌信息"""
        logger.info("加载停牌信息...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.suspend_info)
        
        if len(df) == 0:
            logger.warning("无法加载停牌信息")
            return cudf.DataFrame() if self.use_gpu else None
        
        df = self.normalize_date_column(df, 'trade_date')
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        df['is_suspended'] = 1
        
        logger.info(f"加载停牌信息完成，共 {len(df)} 条停牌记录")
        return df[['ts_code', 'trade_date', 'is_suspended']]
    
    def process(self) -> cudf.DataFrame:
        """执行数据处理 - 纯GPU操作
        
        修复说明：
        1. 加载全量历史数据（不截断 start_date）
        2. 在全量数据上计算 return_1d（避免第一天缺少历史数据）
        3. 最后按 start_date~end_date 过滤输出
        """
        # 1. 加载原始数据（包含 start_date 之前的历史数据）
        daily = self._load_stock_daily()
        adj_factor = self._load_adj_factor()
        daily_basic = self._load_daily_basic()
        suspend_info = self._load_suspend_info()
        
        # 获取股票列表
        stock_codes = daily['ts_code'].unique().to_arrow().to_pylist()
        logger.info(f"数据中包含 {len(stock_codes)} 只股票")
        
        # 2. 直接合并数据（不使用骨架表限制日期范围）
        # 使用 daily 作为基础，保留全部历史数据用于计算 return_1d
        logger.info("合并数据...")
        df = daily.merge(adj_factor, on=['trade_date', 'ts_code'], how='left')
        logger.info(f"合并复权因子后: {len(df)} 行")
        
        if len(daily_basic) > 0:
            df = df.merge(
                daily_basic[['ts_code', 'trade_date', 'turnover_rate', 'turnover_rate_f']],
                on=['trade_date', 'ts_code'],
                how='left'
            )
            logger.info(f"合并换手率后: {len(df)} 行")
        
        if len(suspend_info) > 0:
            df = df.merge(suspend_info, on=['trade_date', 'ts_code'], how='left')
            logger.info(f"合并停牌信息后: {len(df)} 行")
        
        # 释放内存
        del daily, adj_factor, daily_basic, suspend_info
        
        # 3. 填充缺失值（cuDF groupby ffill）
        logger.info("填充缺失值...")
        df = df.sort_values(['ts_code', 'trade_date'])
        
        # 价格和复权因子前向填充
        price_cols = ['open', 'high', 'low', 'close', 'pre_close', 'adj_factor']
        for col in price_cols:
            if col in df.columns:
                df[col] = df.groupby('ts_code')[col].ffill()
        
        # 成交量/额填充为0
        for col in ['vol', 'amount']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # 换手率填充为0
        for col in ['turnover_rate', 'turnover_rate_f']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # 停牌标记填充为0
        if 'is_suspended' in df.columns:
            df['is_suspended'] = df['is_suspended'].fillna(0)
        
        # 删除无数据的行（上市前）
        df = df.dropna(subset=['close', 'adj_factor'])
        logger.info(f"填充缺失值后: {len(df)} 行")
        
        # ====== 金额单位转换（千元 → 元）======
        # Tushare stock_daily 的 amount 单位是千元
        df = self.convert_qian_yuan_to_yuan(df, ['amount'])
        
        # 4. 计算衍生字段（在全量数据上计算，包含历史数据）
        logger.info("计算衍生字段...")
        
        # 后复权价格
        adj = df['adj_factor']
        df['open_hfq'] = df['open'] * adj
        df['high_hfq'] = df['high'] * adj
        df['low_hfq'] = df['low'] * adj
        df['close_hfq'] = df['close'] * adj
        
        # VWAP（注意：amount 已转换为元）
        df['vwap'] = calculate_vwap_gpu(df['amount'], df['vol'], adj_factor=None, amount_unit="yuan")
        df['vwap_hfq'] = calculate_vwap_gpu(df['amount'], df['vol'], adj_factor=adj, amount_unit="yuan")
        
        # 日收益率（基于后复权价格）- 在全量数据上计算
        df = df.sort_values(['ts_code', 'trade_date'])
        df['pre_close_hfq'] = df.groupby('ts_code')['close_hfq'].shift(1)
        df['return_1d'] = (df['close_hfq'] - df['pre_close_hfq']) / df['pre_close_hfq']
        df['return_1d'] = df['return_1d'].fillna(0)
        
        # 换手率
        if 'turnover_rate_f' in df.columns:
            df['turnover'] = df['turnover_rate_f']
        elif 'turnover_rate' in df.columns:
            df['turnover'] = df['turnover_rate']
        else:
            df['turnover'] = 0.0
        
        # 交易状态修复：以实际成交量 vol > 0 作为 is_trading 判定标准
        # 原因：停牌信息表可能包含"盘中临时停牌"（涨跌停触发后恢复），这种情况仍有成交
        # 因此用 vol > 0 更准确地反映"当日是否有交易"
        df['is_trading'] = (df['vol'] > 0).astype('int32')
        
        # 停牌日收益率设为0
        df['return_1d'] = df['return_1d'].where(df['is_trading'] == 1, 0)
        
        logger.info("计算衍生字段完成")
        
        # 5. 按输出日期范围过滤（关键修复点）
        logger.info(f"按日期范围过滤: {self.start_date} 至 {self.end_date}")
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        logger.info(f"过滤后: {len(df)} 行")
        
        # 6. 选择输出列
        output_columns = [
            'trade_date', 'ts_code',
            'open', 'high', 'low', 'close', 'pre_close',
            'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq',
            'vol', 'amount', 'adj_factor',
            'vwap', 'vwap_hfq', 'return_1d', 'turnover', 'is_trading',
        ]
        existing_columns = [col for col in output_columns if col in df.columns]
        df = df[existing_columns]
        df = df.sort_values(['trade_date', 'ts_code'])
        
        # 7. float64 → float32（节省内存）
        df = self.convert_float64_to_float32(df)
        
        return df
    
    def save(self, df: cudf.DataFrame):
        """保存处理结果"""
        self.save_parquet(df, self.output_path)


def main():
    """独立运行测试"""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    processor = MarketDataProcessor(use_gpu=True)
    df = processor.run()
    
    print(f"\n输出数据样例：")
    print(df.head(10).to_arrow().to_pandas())
    
    return df


if __name__ == "__main__":
    main()
