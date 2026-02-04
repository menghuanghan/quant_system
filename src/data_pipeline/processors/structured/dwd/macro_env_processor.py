"""
MacroEnvProcessor - 宏观环境宽表处理器（纯cuDF GPU版本）

生成 dwd_macro_env 宽表，为模型增加"环境感知"能力（Market Regime）

数据源：
    - cn_gdp: GDP（季度）
    - cn_cpi: CPI（月度）
    - cn_pmi: PMI（月度）
    - cn_m2: M2货币供应量（月度）
    - lpr: LPR利率（不定期）
    - shibor: SHIBOR利率（日度）
    - market_congestion: 市场拥挤度（日度）
    - stock_bond_spread: 股债利差（日度）

处理逻辑：
    1. 这是一张 Time-Series 表，没有 ts_code 列，主键只有 trade_date
    2. 日历对齐：以交易日历为骨架
    3. PIT 对齐（关键）：
       - GDP: Q1->4月末, Q2->7月末, Q3->10月末, Q4->次年1月末
       - CPI: 映射到次月15日（实际9-12日公布，保守处理）
       - PMI: 映射到次月3日（实际1日公布，保守处理）
       - M2: 映射到次月15日（实际10-15日公布，保守处理）
    4. 低频填充：宏观数据多为月频/季频，使用 ffill 填充到日频
    5. 不在 DWD 层将宏观数据 Merge 到每只股票（避免数据冗余）

输出：独立的宏观环境小表，供后续特征工程层广播使用
"""

import logging
from typing import Optional

import cudf
import cupy as cp

from .base import BaseProcessor
from .config import (
    DATA_SOURCE_PATHS,
    DWD_OUTPUT_CONFIG,
    PROCESSING_CONFIG,
)

logger = logging.getLogger(__name__)


class MacroEnvProcessor(BaseProcessor):
    """
    宏观环境宽表处理器 - 纯GPU版本
    
    输出字段（主键：trade_date）：
        # GDP相关（季度）
        - gdp_yoy: GDP同比增速（不保留绝对值，避免YTD累计值的锯齿陷阱）
        # CPI相关（月度）
        - cpi_yoy: CPI同比
        - cpi_mom: CPI环比
        # PMI相关（月度）
        - pmi: 制造业PMI
        - pmi_prod: PMI生产指数
        - pmi_new_order: PMI新订单指数
        # 货币供应（月度）
        - m2: M2货币供应量
        - m2_yoy: M2同比增速
        # 利率（日度/不定期）
        - lpr_1y: 1年期LPR
        - lpr_5y: 5年期LPR
        - shibor_on: SHIBOR隔夜
        - shibor_1w: SHIBOR一周
        - shibor_1m: SHIBOR一月
        - shibor_3m: SHIBOR三月
        # 市场风险指标（日度）
        - market_congestion: 市场拥挤度
        - stock_bond_spread: 股债利差
        # 衍生指标
        - macro_regime: 宏观环境状态（扩张/收缩）
        - rate_trend: 利率趋势（上升/下降/平稳）
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        super().__init__(use_gpu=use_gpu, start_date=start_date, end_date=end_date)
        self.output_path = DWD_OUTPUT_CONFIG.output_dir / DWD_OUTPUT_CONFIG.macro_env
    
    def _build_date_skeleton(self) -> cudf.DataFrame:
        """构建交易日骨架（不含股票维度）"""
        trade_dates = self.get_trade_dates()
        return cudf.DataFrame({'trade_date': trade_dates})
    
    def _load_gdp(self) -> cudf.DataFrame:
        """加载GDP数据"""
        logger.info("加载GDP数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.cn_gdp)
        
        if len(df) == 0:
            logger.warning("无法加载GDP数据")
            return cudf.DataFrame()
        
        # quarter格式: 2024Q1, 2024Q2等
        # 需要转换为trade_date（取季末日期作为数据发布的大致日期）
        df['quarter_str'] = df['quarter'].astype(str)
        df['year'] = df['quarter_str'].str.slice(0, 4).astype('int32')
        df['q'] = df['quarter_str'].str.slice(5, 6).astype('int32')
        
        # 季末日期映射（实际发布日期通常滞后1-2个月，这里使用发布月末作为近似）
        # Q1 -> 4月末, Q2 -> 7月末, Q3 -> 10月末, Q4 -> 次年1月末
        def quarter_to_date(row):
            year = int(row['year'])
            q = int(row['q'])
            if q == 1:
                return f"{year}-04-30"
            elif q == 2:
                return f"{year}-07-31"
            elif q == 3:
                return f"{year}-10-31"
            else:  # Q4
                return f"{year+1}-01-31"
        
        # cuDF不直接支持apply，使用条件赋值
        df['trade_date'] = ''
        df.loc[df['q'] == 1, 'trade_date'] = df.loc[df['q'] == 1, 'year'].astype(str) + '-04-30'
        df.loc[df['q'] == 2, 'trade_date'] = df.loc[df['q'] == 2, 'year'].astype(str) + '-07-31'
        df.loc[df['q'] == 3, 'trade_date'] = df.loc[df['q'] == 3, 'year'].astype(str) + '-10-31'
        # Q4特殊处理
        mask_q4 = df['q'] == 4
        if mask_q4.any():
            df.loc[mask_q4, 'trade_date'] = (df.loc[mask_q4, 'year'] + 1).astype(str) + '-01-31'
        
        # 选择需要的字段（只保留同比增速，不保留绝对值避免YTD累计值锯齿陷阱）
        cols = ['trade_date', 'gdp_yoy']
        df = df[[c for c in cols if c in df.columns]]
        
        logger.info(f"加载GDP数据完成，共 {len(df)} 行")
        return df
    
    def _load_cpi(self) -> cudf.DataFrame:
        """加载CPI数据"""
        logger.info("加载CPI数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.cn_cpi)
        
        if len(df) == 0:
            logger.warning("无法加载CPI数据")
            return cudf.DataFrame()
        
        # month格式: 202401等
        df['month_str'] = df['month'].astype(str)
        
        # 转换为月末日期（实际发布日期通常在次月9日左右，这里用月末作为近似）
        df['year'] = df['month_str'].str.slice(0, 4)
        df['mon'] = df['month_str'].str.slice(4, 6)
        
        # PIT修复：CPI实际在次月9-12日公布，保守使用次月15日作为可用日期
        # 例如：202401的CPI在2024-02-09公布，应映射到2024-02-15
        df['year_int'] = df['year'].astype('int32')
        df['mon_int'] = df['mon'].astype('int32')
        
        # 计算次月：月份+1，超过12则年份+1
        df['pub_year'] = df['year_int']
        df['pub_mon'] = df['mon_int'] + 1
        
        # 处理跨年情况
        mask_overflow = df['pub_mon'] > 12
        df.loc[mask_overflow, 'pub_year'] = df.loc[mask_overflow, 'year_int'] + 1
        df.loc[mask_overflow, 'pub_mon'] = 1
        
        # 构建公布日期（次月15日，PIT合规）
        df['trade_date'] = (
            df['pub_year'].astype(str) + '-' + 
            df['pub_mon'].astype(str).str.zfill(2) + '-15'
        )
        
        # 清理临时列
        df = df.drop(columns=['year_int', 'mon_int', 'pub_year', 'pub_mon'], errors='ignore')
        
        # 选择需要的字段（使用全国同比）
        df = df.rename(columns={'nt_yoy': 'cpi_yoy', 'nt_mom': 'cpi_mom'})
        cols = ['trade_date', 'cpi_yoy', 'cpi_mom']
        df = df[[c for c in cols if c in df.columns]]
        
        logger.info(f"加载CPI数据完成，共 {len(df)} 行")
        return df
    
    def _load_pmi(self) -> cudf.DataFrame:
        """加载PMI数据"""
        logger.info("加载PMI数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.cn_pmi)
        
        if len(df) == 0:
            logger.warning("无法加载PMI数据")
            return cudf.DataFrame()
        
        # month格式: 202401等
        df['month_str'] = df['month'].astype(str)
        df['year'] = df['month_str'].str.slice(0, 4)
        df['mon'] = df['month_str'].str.slice(4, 6)
        
        # PIT修复：PMI实际在次月1日公布，使用次月3日作为保守可用日期
        df['year_int'] = df['year'].astype('int32')
        df['mon_int'] = df['mon'].astype('int32')
        df['pub_year'] = df['year_int']
        df['pub_mon'] = df['mon_int'] + 1
        mask_overflow = df['pub_mon'] > 12
        df.loc[mask_overflow, 'pub_year'] = df.loc[mask_overflow, 'year_int'] + 1
        df.loc[mask_overflow, 'pub_mon'] = 1
        df['trade_date'] = (
            df['pub_year'].astype(str) + '-' + 
            df['pub_mon'].astype(str).str.zfill(2) + '-03'
        )
        df = df.drop(columns=['year_int', 'mon_int', 'pub_year', 'pub_mon'], errors='ignore')
        
        # 选择需要的字段
        cols = ['trade_date', 'pmi', 'pmi_prod', 'pmi_new_order']
        df = df[[c for c in cols if c in df.columns]]
        
        logger.info(f"加载PMI数据完成，共 {len(df)} 行")
        return df
    
    def _load_m2(self) -> cudf.DataFrame:
        """加载M2数据"""
        logger.info("加载M2数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.cn_m2)
        
        if len(df) == 0:
            logger.warning("无法加载M2数据")
            return cudf.DataFrame()
        
        # month格式: 202401等
        df['month_str'] = df['month'].astype(str)
        df['year'] = df['month_str'].str.slice(0, 4)
        df['mon'] = df['month_str'].str.slice(4, 6)
        
        # PIT修复：M2实际在次月10-15日公布，使用次月15日作为保守可用日期
        df['year_int'] = df['year'].astype('int32')
        df['mon_int'] = df['mon'].astype('int32')
        df['pub_year'] = df['year_int']
        df['pub_mon'] = df['mon_int'] + 1
        mask_overflow = df['pub_mon'] > 12
        df.loc[mask_overflow, 'pub_year'] = df.loc[mask_overflow, 'year_int'] + 1
        df.loc[mask_overflow, 'pub_mon'] = 1
        df['trade_date'] = (
            df['pub_year'].astype(str) + '-' + 
            df['pub_mon'].astype(str).str.zfill(2) + '-15'
        )
        df = df.drop(columns=['year_int', 'mon_int', 'pub_year', 'pub_mon'], errors='ignore')
        
        # 选择需要的字段
        cols = ['trade_date', 'm2', 'm2_yoy']
        df = df[[c for c in cols if c in df.columns]]
        
        logger.info(f"加载M2数据完成，共 {len(df)} 行")
        return df
    
    def _load_lpr(self) -> cudf.DataFrame:
        """加载LPR数据"""
        logger.info("加载LPR数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.lpr)
        
        if len(df) == 0:
            logger.warning("无法加载LPR数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'date')
        df = df.rename(columns={'date': 'trade_date'})
        
        # 选择需要的字段
        cols = ['trade_date', 'lpr_1y', 'lpr_5y']
        df = df[[c for c in cols if c in df.columns]]
        
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载LPR数据完成，共 {len(df)} 行")
        return df
    
    def _load_shibor(self) -> cudf.DataFrame:
        """加载SHIBOR数据"""
        logger.info("加载SHIBOR数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.shibor)
        
        if len(df) == 0:
            logger.warning("无法加载SHIBOR数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'date')
        df = df.rename(columns={'date': 'trade_date'})
        
        # 重命名字段
        rename_map = {
            'on': 'shibor_on',
            '1w': 'shibor_1w',
            '1m': 'shibor_1m',
            '3m': 'shibor_3m',
            '6m': 'shibor_6m',
            '1y': 'shibor_1y',
        }
        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        # 选择需要的字段
        cols = ['trade_date', 'shibor_on', 'shibor_1w', 'shibor_1m', 'shibor_3m', 'shibor_6m', 'shibor_1y']
        df = df[[c for c in cols if c in df.columns]]
        
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载SHIBOR数据完成，共 {len(df)} 行")
        return df
    
    def _load_market_congestion(self) -> cudf.DataFrame:
        """加载市场拥挤度数据"""
        logger.info("加载市场拥挤度数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.market_congestion)
        
        if len(df) == 0:
            logger.warning("无法加载市场拥挤度数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'date')
        df = df.rename(columns={'date': 'trade_date', 'congestion': 'market_congestion'})
        
        cols = ['trade_date', 'market_congestion']
        df = df[[c for c in cols if c in df.columns]]
        
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载市场拥挤度数据完成，共 {len(df)} 行")
        return df
    
    def _load_stock_bond_spread(self) -> cudf.DataFrame:
        """加载股债利差数据"""
        logger.info("加载股债利差数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.stock_bond_spread)
        
        if len(df) == 0:
            logger.warning("无法加载股债利差数据")
            return cudf.DataFrame()
        
        df = self.normalize_date_column(df, 'date')
        df = df.rename(columns={'date': 'trade_date', 'spread': 'stock_bond_spread'})
        
        cols = ['trade_date', 'stock_bond_spread']
        df = df[[c for c in cols if c in df.columns]]
        
        df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
        
        logger.info(f"加载股债利差数据完成，共 {len(df)} 行")
        return df
    
    def _calculate_derived_features(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """计算衍生特征"""
        logger.info("计算宏观环境衍生特征...")
        
        df = df.sort_values('trade_date')
        
        # 1. 宏观环境状态（基于PMI荣枯线）
        if 'pmi' in df.columns:
            # PMI > 50 扩张，< 50 收缩
            df['pmi_regime'] = 0  # 收缩
            df.loc[df['pmi'] > 50, 'pmi_regime'] = 1  # 扩张
            df.loc[df['pmi'] > 52, 'pmi_regime'] = 2  # 强扩张
        
        # 2. 利率趋势（基于LPR变化）
        if 'lpr_1y' in df.columns:
            df['lpr_1y_prev'] = df['lpr_1y'].shift(1)
            df['lpr_trend'] = 0  # 平稳
            df.loc[df['lpr_1y'] > df['lpr_1y_prev'], 'lpr_trend'] = 1   # 上升
            df.loc[df['lpr_1y'] < df['lpr_1y_prev'], 'lpr_trend'] = -1  # 下降
            df = df.drop(columns=['lpr_1y_prev'])
        
        # 3. 货币环境（基于M2增速）
        if 'm2_yoy' in df.columns:
            # M2同比增速 > 10% 宽松，< 8% 紧缩
            df['money_regime'] = 0  # 中性
            df.loc[df['m2_yoy'] > 10, 'money_regime'] = 1   # 宽松
            df.loc[df['m2_yoy'] < 8, 'money_regime'] = -1   # 紧缩
        
        # 4. 市场风险偏好（基于股债利差）
        if 'stock_bond_spread' in df.columns:
            # 计算历史分位数（简化：与60日均值比较）
            df['spread_ma60'] = df['stock_bond_spread'].rolling(window=60, min_periods=1).mean()
            df['risk_appetite'] = 0  # 中性
            df.loc[df['stock_bond_spread'] > df['spread_ma60'], 'risk_appetite'] = 1   # 风险偏好上升
            df.loc[df['stock_bond_spread'] < df['spread_ma60'], 'risk_appetite'] = -1  # 风险偏好下降
            df = df.drop(columns=['spread_ma60'])
        
        # 5. 综合市场状态（regime）
        regime_cols = ['pmi_regime', 'money_regime', 'risk_appetite']
        regime_cols_present = [c for c in regime_cols if c in df.columns]
        if regime_cols_present:
            df['macro_score'] = 0
            for col in regime_cols_present:
                df['macro_score'] = df['macro_score'] + df[col].fillna(0)
            # 综合状态
            df['macro_regime'] = 0  # 中性
            df.loc[df['macro_score'] >= 2, 'macro_regime'] = 1   # 偏多
            df.loc[df['macro_score'] <= -2, 'macro_regime'] = -1  # 偏空
        
        logger.info("宏观环境衍生特征计算完成")
        return df
    
    def process(self) -> cudf.DataFrame:
        """处理并生成宏观环境宽表"""
        logger.info("开始处理宏观环境宽表...")
        
        # 1. 构建交易日骨架
        skeleton = self._build_date_skeleton()
        
        # 2. 加载各数据源
        gdp = self._load_gdp()
        cpi = self._load_cpi()
        pmi = self._load_pmi()
        m2 = self._load_m2()
        lpr = self._load_lpr()
        shibor = self._load_shibor()
        congestion = self._load_market_congestion()
        spread = self._load_stock_bond_spread()
        
        # 3. 合并所有数据到骨架表
        logger.info("合并宏观数据到交易日骨架...")
        
        result = skeleton.copy()
        
        # 低频数据合并后需要ffill
        low_freq_data = [
            (gdp, ['gdp_yoy']),  # 只保留同比增速，避免YTD累计值锯齿陷阱
            (cpi, ['cpi_yoy', 'cpi_mom']),
            (pmi, ['pmi', 'pmi_prod', 'pmi_new_order']),
            (m2, ['m2', 'm2_yoy']),
            (lpr, ['lpr_1y', 'lpr_5y']),
        ]
        
        for df, cols in low_freq_data:
            if len(df) > 0:
                df_cols = ['trade_date'] + [c for c in cols if c in df.columns]
                result = result.merge(df[df_cols], on='trade_date', how='left')
        
        # 高频数据直接合并
        high_freq_data = [
            (shibor, ['shibor_on', 'shibor_1w', 'shibor_1m', 'shibor_3m', 'shibor_6m', 'shibor_1y']),
            (congestion, ['market_congestion']),
            (spread, ['stock_bond_spread']),
        ]
        
        for df, cols in high_freq_data:
            if len(df) > 0:
                df_cols = ['trade_date'] + [c for c in cols if c in df.columns]
                result = result.merge(df[df_cols], on='trade_date', how='left')
        
        # 4. 前向填充低频数据
        logger.info("前向填充低频数据...")
        
        result = result.sort_values('trade_date')
        
        low_freq_cols = [
            'gdp', 'gdp_yoy',
            'cpi_yoy', 'cpi_mom',
            'pmi', 'pmi_prod', 'pmi_new_order',
            'm2', 'm2_yoy',
            'lpr_1y', 'lpr_5y',
        ]
        
        for col in low_freq_cols:
            if col in result.columns:
                result[col] = result[col].ffill()
        
        # SHIBOR也需要前向填充（非交易日无数据）
        shibor_cols = ['shibor_on', 'shibor_1w', 'shibor_1m', 'shibor_3m', 'shibor_6m', 'shibor_1y']
        for col in shibor_cols:
            if col in result.columns:
                result[col] = result[col].ffill()
        
        # 5. 计算衍生特征
        result = self._calculate_derived_features(result)
        
        # 6. 排序
        result = result.sort_values('trade_date').reset_index(drop=True)
        
        logger.info(f"宏观环境宽表处理完成，共 {len(result)} 行")
        return result
    
    def save(self, df: cudf.DataFrame):
        """保存处理结果"""
        self.save_parquet(df, self.output_path)
        logger.info(f"宏观环境宽表已保存到 {self.output_path}")
