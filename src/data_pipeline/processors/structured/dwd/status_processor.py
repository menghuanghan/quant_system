"""
StatusProcessor - 状态与风险掩码表处理器（纯cuDF GPU版本）

生成 dwd_stock_status 宽表，用于过滤不可交易的样本

涨跌停判定逻辑：
1. 根据板块确定基础涨跌停比例
   - 主板/中小板: 10%（ST 5%）
   - 创业板/科创板: 20%
   - 北交所: 30%
2. 新股上市前N天不设涨跌停限制
"""

import logging
from typing import Optional

import cudf
import cupy as cp

from .base import BaseProcessor
from .config import (
    DATA_SOURCE_PATHS,
    DWD_OUTPUT_CONFIG,
    MARKET_CONFIG,
    PROCESSING_CONFIG,
)

logger = logging.getLogger(__name__)


class StatusProcessor(BaseProcessor):
    """
    状态与风险掩码表处理器 - 纯GPU版本
    
    输出字段：
        - trade_date, ts_code: 主键
        - is_st: 是否ST
        - is_limit_up: 是否涨停
        - is_limit_down: 是否跌停
        - is_new: 是否新股
        - is_new_no_limit: 新股无涨跌停限制期
        - is_trading: 是否交易
        - list_date, market, limit_ratio: 辅助字段
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        super().__init__(use_gpu=use_gpu, start_date=start_date, end_date=end_date)
        self.output_path = DWD_OUTPUT_CONFIG.output_dir / DWD_OUTPUT_CONFIG.stock_status
        self.market_config = MARKET_CONFIG
    
    def _load_st_status(self) -> cudf.DataFrame:
        """加载ST状态数据"""
        logger.info("加载ST状态数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.st_status)
        
        if len(df) == 0:
            logger.warning("无法加载ST状态数据")
            return cudf.DataFrame(columns=['ts_code', 'st_start_date', 'st_end_date'])
        
        df = self.normalize_date_column(df, 'st_start_date')
        df = self.normalize_date_column(df, 'st_end_date')
        
        # 处理空的结束日期
        df['st_end_date'] = df['st_end_date'].fillna('2099-12-31')
        
        logger.info(f"加载ST状态数据完成，共 {len(df)} 条记录")
        return df[['ts_code', 'st_start_date', 'st_end_date']]
    
    def _load_suspend_info(self) -> cudf.DataFrame:
        """加载停牌信息"""
        logger.info("加载停牌信息...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.suspend_info)
        
        if len(df) == 0:
            logger.warning("无法加载停牌信息")
            return cudf.DataFrame(columns=['ts_code', 'trade_date'])
        
        df = self.normalize_date_column(df, 'trade_date')
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        df['is_suspended'] = 1
        
        logger.info(f"加载停牌信息完成，共 {len(df)} 条停牌记录")
        return df[['ts_code', 'trade_date', 'is_suspended']]
    
    def _load_price_data(self) -> cudf.DataFrame:
        """加载行情数据（用于判断涨跌停）"""
        logger.info("加载行情数据...")
        
        df = self.read_parquet_dir(
            DATA_SOURCE_PATHS.stock_daily_dir,
            columns=['ts_code', 'trade_date', 'close', 'pre_close', 'vol']
        )
        
        if len(df) == 0:
            raise ValueError("无法加载行情数据")
        
        df = self.normalize_date_column(df, 'trade_date')
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载行情数据完成，共 {len(df)} 行")
        return df
    
    def _determine_is_st(
        self,
        df: cudf.DataFrame,
        st_status: cudf.DataFrame,
    ) -> cudf.DataFrame:
        """确定每日ST状态（GPU批量处理）"""
        logger.info("计算ST状态...")
        
        df['is_st'] = 0
        
        if len(st_status) == 0:
            return df
        
        # 展开ST区间为日期序列
        # 这是一个挑战，因为cuDF不直接支持区间展开
        # 我们使用交叉连接+过滤的方式
        
        # 方法：将df与st_status按ts_code连接，然后过滤日期范围内的
        st_expanded = df[['ts_code', 'trade_date']].merge(
            st_status,
            on='ts_code',
            how='inner'
        )
        
        # 过滤在ST期间内的记录
        st_days = st_expanded[
            (st_expanded['trade_date'] >= st_expanded['st_start_date']) &
            (st_expanded['trade_date'] <= st_expanded['st_end_date'])
        ][['ts_code', 'trade_date']].drop_duplicates()
        
        st_days['is_st_flag'] = 1
        
        # 合并回主表
        df = df.merge(st_days, on=['ts_code', 'trade_date'], how='left')
        df['is_st'] = df['is_st_flag'].fillna(0).astype('int32')
        df = df.drop(columns=['is_st_flag'])
        
        st_count = int(df['is_st'].sum())
        logger.info(f"ST状态计算完成，共 {st_count} 条ST记录")
        
        return df
    
    def _determine_limit_ratio(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """确定每只股票每日的涨跌停比例"""
        logger.info("计算涨跌停比例...")
        
        # 获取股票列表
        stock_list = self.get_stock_list()
        
        # 合并市场和上市日期信息
        df = df.merge(
            stock_list[['ts_code', 'market', 'list_date']],
            on='ts_code',
            how='left'
        )
        
        # 计算上市天数
        trade_date_dt = cudf.to_datetime(df['trade_date'])
        list_date_dt = cudf.to_datetime(df['list_date'])
        df['days_since_listing'] = (trade_date_dt - list_date_dt).dt.days
        
        # 判断是否新股
        df['is_new'] = (df['days_since_listing'] < self.market_config.NEW_STOCK_DAYS).astype('int32')
        
        # 判断是否处于新股无涨跌停限制期
        # 根据市场板块获取无限制天数（默认5天）
        no_limit_days_map = self.market_config.NEW_STOCK_NO_LIMIT_DAYS
        default_no_limit = 5
        
        # 为每个市场设置无限制天数
        df['no_limit_days'] = default_no_limit
        for market, days in no_limit_days_map.items():
            df['no_limit_days'] = df['no_limit_days'].where(df['market'] != market, days)
        
        df['is_new_no_limit'] = (df['days_since_listing'] < df['no_limit_days']).astype('int32')
        
        # 计算涨跌停比例
        # 默认10%
        df['limit_ratio'] = 0.10
        
        # 根据市场设置基础比例
        limit_ratios = self.market_config.LIMIT_RATIOS
        for market, ratio in limit_ratios.items():
            df['limit_ratio'] = df['limit_ratio'].where(df['market'] != market, ratio)
        
        # ST股票特殊处理
        st_ratios = self.market_config.ST_LIMIT_RATIOS
        for market, ratio in st_ratios.items():
            mask = (df['market'] == market) & (df['is_st'] == 1)
            df['limit_ratio'] = df['limit_ratio'].where(~mask, ratio)
        
        # 新股无限制期设为100%
        df['limit_ratio'] = df['limit_ratio'].where(df['is_new_no_limit'] == 0, 1.0)
        
        df = df.drop(columns=['no_limit_days'])
        
        logger.info("涨跌停比例计算完成")
        return df
    
    def _determine_limit_status(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """判断是否涨停/跌停"""
        logger.info("判断涨跌停状态...")
        
        # 计算涨停价和跌停价
        limit_up_price = (df['pre_close'] * (1 + df['limit_ratio'])).round(2)
        limit_down_price = (df['pre_close'] * (1 - df['limit_ratio'])).round(2)
        
        tolerance = 0.001
        
        # 涨停判断
        df['is_limit_up'] = (
            (df['close'] >= limit_up_price * (1 - tolerance)) &
            (df['limit_ratio'] < 1.0)
        ).astype('int32')
        
        # 跌停判断
        df['is_limit_down'] = (
            (df['close'] <= limit_down_price * (1 + tolerance)) &
            (df['limit_ratio'] < 1.0)
        ).astype('int32')
        
        limit_up_count = int(df['is_limit_up'].sum())
        limit_down_count = int(df['is_limit_down'].sum())
        logger.info(f"涨跌停判断完成: 涨停 {limit_up_count} 次, 跌停 {limit_down_count} 次")
        
        return df
    
    def _determine_trading_status(
        self,
        df: cudf.DataFrame,
        suspend_info: cudf.DataFrame,
    ) -> cudf.DataFrame:
        """确定交易状态"""
        logger.info("确定交易状态...")
        
        if len(suspend_info) > 0:
            df = df.merge(
                suspend_info[['ts_code', 'trade_date', 'is_suspended']],
                on=['ts_code', 'trade_date'],
                how='left'
            )
            df['is_suspended'] = df['is_suspended'].fillna(0)
        else:
            df['is_suspended'] = 0
        
        # is_trading = 非停牌 且 成交量>0
        if 'vol' in df.columns:
            df['is_trading'] = ((df['is_suspended'] == 0) & (df['vol'] > 0)).astype('int32')
        else:
            df['is_trading'] = (df['is_suspended'] == 0).astype('int32')
        
        df = df.drop(columns=['is_suspended'])
        
        not_trading_count = int((df['is_trading'] == 0).sum())
        logger.info(f"交易状态确定完成: 非交易日 {not_trading_count} 天")
        
        return df
    
    def process(self) -> cudf.DataFrame:
        """执行数据处理 - 纯GPU操作"""
        # 1. 加载数据
        st_status = self._load_st_status()
        suspend_info = self._load_suspend_info()
        price_data = self._load_price_data()
        
        # 获取股票列表
        stock_codes = price_data['ts_code'].unique().to_arrow().to_pylist()
        logger.info(f"数据中包含 {len(stock_codes)} 只股票")
        
        # 2. 构建骨架表并合并行情数据
        skeleton = self.build_skeleton_table(stock_codes=stock_codes)
        
        df = skeleton.merge(
            price_data[['ts_code', 'trade_date', 'close', 'pre_close', 'vol']],
            on=['ts_code', 'trade_date'],
            how='left'
        )
        
        # 填充缺失的行情数据
        df = df.sort_values(['ts_code', 'trade_date'])
        df['close'] = df.groupby('ts_code')['close'].ffill()
        df['pre_close'] = df.groupby('ts_code')['pre_close'].ffill()
        df['vol'] = df['vol'].fillna(0)
        
        # 删除仍有缺失值的行
        df = df.dropna(subset=['close', 'pre_close'])
        
        del skeleton, price_data
        
        # 3. 计算ST状态
        df = self._determine_is_st(df, st_status)
        del st_status
        
        # 4. 计算涨跌停比例
        df = self._determine_limit_ratio(df)
        
        # 5. 判断涨跌停状态
        df = self._determine_limit_status(df)
        
        # 6. 确定交易状态
        df = self._determine_trading_status(df, suspend_info)
        del suspend_info
        
        # 7. 选择输出列
        output_columns = [
            'trade_date', 'ts_code',
            'is_st', 'is_limit_up', 'is_limit_down',
            'is_new', 'is_new_no_limit', 'is_trading',
            'list_date', 'market', 'limit_ratio',
        ]
        
        existing_columns = [col for col in output_columns if col in df.columns]
        df = df[existing_columns]
        df = df.sort_values(['trade_date', 'ts_code'])
        
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
    
    processor = StatusProcessor(use_gpu=True)
    df = processor.run()
    
    print(f"\n输出数据样例：")
    print(df.head(10).to_arrow().to_pandas())
    
    # 统计信息
    print(f"\n状态统计：")
    print(f"ST股票日: {int(df['is_st'].sum())}")
    print(f"涨停次数: {int(df['is_limit_up'].sum())}")
    print(f"跌停次数: {int(df['is_limit_down'].sum())}")
    print(f"新股交易日: {int(df['is_new'].sum())}")
    print(f"非交易日: {int((df['is_trading'] == 0).sum())}")
    
    return df


if __name__ == "__main__":
    main()
