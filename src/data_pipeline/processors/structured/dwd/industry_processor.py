"""
IndustryProcessor - 行业分类宽表处理器（纯cuDF GPU版本）

生成 dwd_stock_industry 宽表，用于行业与风格中性化

数据源：
    - stock_list_a: A股股票列表（包含 industry 字段，100%覆盖）
    - sw_index_member: 申万行业成分股（可选，区间数据：in_date 到 out_date）

处理逻辑：
    1. 优先使用 stock_list_a 的 industry 字段作为基础行业分类
    2. 如果 sw_index_member 可用，则补充申万一级行业分类
    3. 如果 sw_index_member 不可用，则根据 industry 字段推断申万一级行业
    4. industry_changed 标记基于行业代码变化计算
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
from .industry_mapping import get_sw_l1_from_industry, INDUSTRY_TO_SW_L1_MAP, SW_L1_NAME_TO_CODE

logger = logging.getLogger(__name__)


class IndustryProcessor(BaseProcessor):
    """
    行业分类宽表处理器 - 纯GPU版本
    
    输出字段：
        - trade_date, ts_code: 主键
        # 行业分类（源自 stock_list_a）
        - industry: 行业名称（Tushare 分类，100%覆盖）
        - industry_idx: 行业编码（用于模型训练，-1 表示未分类）
        # 申万一级行业分类（可选，如果 sw_index_member 可用）
        - sw_l1_code: 申万一级行业代码
        - sw_l1_name: 申万一级行业名称
        # 行业变更标记
        - industry_changed: 是否发生行业变更
        
    数据质量说明：
        - industry 字段来自 stock_list_a，覆盖率 100%
        - sw_l1_* 字段来自 sw_index_member，可能部分缺失
        - industry_idx 使用 -1 标记未分类，避免与真实行业冲突
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        super().__init__(use_gpu=use_gpu, start_date=start_date, end_date=end_date)
        self.output_path = DWD_OUTPUT_CONFIG.output_dir / DWD_OUTPUT_CONFIG.stock_industry
    
    def _load_stock_industry(self) -> cudf.DataFrame:
        """
        从 stock_list_a 加载行业数据
        
        Returns:
            包含 ts_code, industry 的 DataFrame
        """
        logger.info("加载 stock_list_a 行业数据...")
        
        # 读取元数据
        import pandas as pd
        stock_list_path = DATA_SOURCE_PATHS.stock_list_a
        
        if not stock_list_path.exists():
            logger.warning(f"stock_list_a 文件不存在: {stock_list_path}")
            return cudf.DataFrame()
        
        pdf = pd.read_parquet(stock_list_path)
        
        # 只保留需要的字段
        cols = ['ts_code', 'industry']
        pdf = pdf[cols].drop_duplicates()
        
        # 转为 cuDF
        df = cudf.from_pandas(pdf)
        
        # 处理缺失值
        df['industry'] = df['industry'].fillna('未分类')
        
        logger.info(f"加载 stock_list_a 行业数据完成，共 {len(df)} 只股票")
        return df
    
    def _load_sw_index_member(self) -> cudf.DataFrame:
        """
        加载申万行业成分股数据（可选）
        
        如果数据不存在，返回空 DataFrame
        """
        logger.info("尝试加载申万行业成分股数据...")
        
        try:
            df = self.read_parquet(DATA_SOURCE_PATHS.sw_index_member)
            
            if len(df) == 0:
                logger.warning("申万行业成分股数据为空，将仅使用 stock_list_a 行业数据")
                return cudf.DataFrame()
            
            df = self.normalize_date_column(df, 'in_date')
            
            # 处理 out_date（可能为null，表示当前仍在该行业）
            if 'out_date' in df.columns:
                df['out_date'] = df['out_date'].astype(str).fillna('2099-12-31')
                df = self.normalize_date_column(df, 'out_date')
            else:
                df['out_date'] = '2099-12-31'
            
            logger.info(f"加载申万行业成分股数据完成，共 {len(df)} 行")
            return df
        except Exception as e:
            logger.warning(f"加载申万行业成分股数据失败: {e}，将仅使用 stock_list_a 行业数据")
            return cudf.DataFrame()
    
    def _load_sw_index_classify(self) -> cudf.DataFrame:
        """加载申万行业分类数据"""
        logger.info("加载申万行业分类数据...")
        
        df = self.read_parquet(DATA_SOURCE_PATHS.sw_index_classify)
        
        if len(df) == 0:
            logger.warning("无法加载申万行业分类数据")
            return cudf.DataFrame()
        
        logger.info(f"加载申万行业分类数据完成，共 {len(df)} 行")
        return df
    
    def _expand_interval_to_daily(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        将区间数据展开到日频
        
        由于数据量可能很大，我们使用交叉连接+过滤的方式：
        1. 骨架表与行业成分表按ts_code连接
        2. 过滤在 [in_date, out_date) 区间内的记录
        """
        logger.info("将行业区间数据展开到日频...")
        
        # 获取交易日列表
        trade_dates = self.get_trade_dates()
        trade_dates_df = cudf.DataFrame({'trade_date': trade_dates})
        
        # 获取需要处理的股票列表
        stocks = df['ts_code'].unique()
        
        # 构建骨架（只包含有行业数据的股票）
        skeleton = cudf.DataFrame({'ts_code': stocks.to_arrow().to_pylist(), '_key': 1})
        trade_dates_df['_key'] = 1
        full_skeleton = trade_dates_df.merge(skeleton, on='_key').drop(columns=['_key'])
        
        # 与行业成分表连接
        result = full_skeleton.merge(df, on='ts_code', how='left')
        
        # 过滤在有效区间内的记录: in_date <= trade_date < out_date
        result = result[
            (result['trade_date'] >= result['in_date']) &
            (result['trade_date'] < result['out_date'])
        ]
        
        # 处理一只股票同一天可能有多条记录的情况（行业变更当天）
        # 取最新的记录（in_date最大的）
        result = result.sort_values(['ts_code', 'trade_date', 'in_date'], ascending=[True, True, False])
        result = result.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
        
        logger.info(f"行业区间数据展开完成，共 {len(result)} 行")
        return result
    
    def _add_industry_change_flag(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """添加行业变更标记（基于 industry 字段）"""
        logger.info("计算行业变更标记...")
        
        if 'industry' not in df.columns:
            df['industry_changed'] = 0
            return df
        
        df = df.sort_values(['ts_code', 'trade_date'])
        
        # 上一个交易日的行业
        df['prev_industry'] = df.groupby('ts_code')['industry'].shift(1)
        
        # 行业发生变化（排除首次出现的情况，排除"未分类"的变化）
        df['industry_changed'] = (
            (df['industry'] != df['prev_industry']) & 
            df['prev_industry'].notna() &
            (df['industry'] != '未分类') &
            (df['prev_industry'] != '未分类')
        ).astype('int32')
        df = df.drop(columns=['prev_industry'], errors='ignore')
        
        changed_count = int(df['industry_changed'].sum())
        logger.info(f"行业变更标记计算完成，共 {changed_count} 次变更")
        
        return df
    
    def process(self) -> cudf.DataFrame:
        """处理并生成行业分类宽表"""
        logger.info("开始处理行业分类宽表...")
        
        # 1. 构建完整骨架表
        skeleton = self.build_skeleton_table()
        logger.info(f"骨架表: {len(skeleton)} 行, {skeleton['ts_code'].nunique()} 只股票")
        
        # 2. 加载 stock_list_a 的行业数据（100%覆盖）
        stock_industry = self._load_stock_industry()
        
        if len(stock_industry) == 0:
            raise ValueError("无法加载行业数据，stock_list_a 不可用")
        
        # 3. 与骨架表合并（左连接）
        logger.info("合并行业数据到骨架表...")
        result = skeleton.merge(stock_industry, on='ts_code', how='left')
        
        # 4. 处理未匹配的股票（理论上不应有，因为骨架表来自 stock_list）
        unmatched = result['industry'].isna().sum()
        if unmatched > 0:
            logger.warning(f"有 {unmatched} 条记录未匹配到行业，标记为'未分类'")
            result['industry'] = result['industry'].fillna('未分类')
        
        # 5. 尝试加载申万一级行业数据补充
        sw_member = self._load_sw_index_member()
        
        if len(sw_member) > 0:
            logger.info("合并申万一级行业数据...")
            # 重命名字段
            rename_map = {
                'l1_code': 'sw_l1_code', 'l1_name': 'sw_l1_name',
            }
            for old_name, new_name in rename_map.items():
                if old_name in sw_member.columns:
                    sw_member = sw_member.rename(columns={old_name: new_name})
            
            # 展开区间到日频
            industry_daily = self._expand_interval_to_daily(sw_member)
            
            # 选择需要的字段
            sw_cols = ['ts_code', 'trade_date', 'sw_l1_code', 'sw_l1_name']
            sw_cols = [c for c in sw_cols if c in industry_daily.columns]
            industry_daily = industry_daily[sw_cols]
            
            # 合并
            result = result.merge(industry_daily, on=['ts_code', 'trade_date'], how='left')
        else:
            logger.info("申万一级行业数据不可用，仅使用基础行业分类")
            # 添加空的申万字段
            for col in ['sw_l1_code', 'sw_l1_name']:
                result[col] = None
        
        # 6. 使用 industry 字段推断/填充 sw_l1_name
        logger.info("使用 industry 字段推断申万一级行业...")
        
        # 创建映射表（从 industry_mapping.py 导入的 INDUSTRY_TO_SW_L1_MAP）
        industry_sw_l1_map = cudf.DataFrame({
            'industry': list(INDUSTRY_TO_SW_L1_MAP.keys()),
            '_inferred_sw_l1': list(INDUSTRY_TO_SW_L1_MAP.values())
        })
        
        # 合并映射
        result = result.merge(industry_sw_l1_map, on='industry', how='left')
        
        # 如果 sw_l1_name 为空或"未分类"，则使用推断值
        if 'sw_l1_name' in result.columns:
            # 找出需要填充的行：sw_l1_name 为 null 或 "未分类"
            needs_fill = result['sw_l1_name'].isna() | (result['sw_l1_name'] == '未分类')
            has_inferred = result['_inferred_sw_l1'].notna()
            
            # 使用 cuDF 的 where 方法进行条件填充
            result['sw_l1_name'] = result['sw_l1_name'].where(
                ~(needs_fill & has_inferred),
                result['_inferred_sw_l1']
            )
        else:
            # 如果没有 sw_l1_name 列，直接使用推断值
            result['sw_l1_name'] = result['_inferred_sw_l1']
        
        # 清理临时列
        result = result.drop(columns=['_inferred_sw_l1'], errors='ignore')
        
        # 6.5 使用 sw_l1_name 反查 sw_l1_code（填充映射推断未覆盖的部分）
        logger.info("使用 sw_l1_name 反查 sw_l1_code...")
        if 'sw_l1_code' in result.columns and 'sw_l1_name' in result.columns:
            name_to_code_df = cudf.DataFrame({
                'sw_l1_name': list(SW_L1_NAME_TO_CODE.keys()),
                '_inferred_sw_l1_code': list(SW_L1_NAME_TO_CODE.values())
            })
            result = result.merge(name_to_code_df, on='sw_l1_name', how='left')
            
            # 仅填充 sw_l1_code 为空的行
            code_needs_fill = result['sw_l1_code'].isna()
            has_inferred_code = result['_inferred_sw_l1_code'].notna()
            result['sw_l1_code'] = result['sw_l1_code'].where(
                ~(code_needs_fill & has_inferred_code),
                result['_inferred_sw_l1_code']
            )
            n_filled = int((code_needs_fill & has_inferred_code).sum())
            logger.info(f"反查填充了 {n_filled} 条 sw_l1_code 记录")
            result = result.drop(columns=['_inferred_sw_l1_code'], errors='ignore')
        
        # 最终填充剩余缺失值
        sw_name_cols = ['sw_l1_name']
        for col in sw_name_cols:
            if col in result.columns:
                result[col] = result[col].fillna('未分类')
        
        # 统计推断效果
        classified = (result['sw_l1_name'] != '未分类').sum()
        logger.info(f"申万一级行业覆盖率: {int(classified)}/{len(result)} = {int(classified)/len(result)*100:.2f}%")
        
        # 7. 计算行业变更标记（基于 industry 字段）
        result = self._add_industry_change_flag(result)
        
        # 8. 生成行业编码（用于模型训练）
        logger.info("生成行业编码...")
        
        # 对 industry 进行编码（-1 表示未分类，避免与真实行业冲突）
        if 'industry' in result.columns:
            unique_industries = result['industry'].unique().to_arrow().to_pylist()
            # 排序时把"未分类"放最后，编码为 -1
            sorted_industries = sorted([i for i in unique_industries if i != '未分类'])
            industry_map = {name: idx for idx, name in enumerate(sorted_industries)}
            industry_map['未分类'] = -1  # 未分类使用 -1
            
            industry_map_df = cudf.DataFrame({
                'industry': list(industry_map.keys()),
                'industry_idx': list(industry_map.values())
            })
            result = result.merge(industry_map_df, on='industry', how='left')
            result['industry_idx'] = result['industry_idx'].fillna(-1).astype('int32')
        
        # 申万一级行业编码
        if 'sw_l1_name' in result.columns:
            unique_l1 = result['sw_l1_name'].unique().to_arrow().to_pylist()
            sorted_l1 = sorted([i for i in unique_l1 if i != '未分类'])
            l1_map = {name: idx for idx, name in enumerate(sorted_l1)}
            l1_map['未分类'] = -1
            
            l1_map_df = cudf.DataFrame({
                'sw_l1_name': list(l1_map.keys()),
                'sw_l1_idx': list(l1_map.values())
            })
            result = result.merge(l1_map_df, on='sw_l1_name', how='left')
            result['sw_l1_idx'] = result['sw_l1_idx'].fillna(-1).astype('int32')
        
        # 9. 排序
        result = result.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        
        # 10. float64 → float32（节省内存）
        result = self.convert_float64_to_float32(result)
        
        # 11. 输出统计
        classified_count = (result['industry'] != '未分类').sum()
        total_count = len(result)
        logger.info(f"行业分类宽表处理完成，共 {total_count} 行")
        logger.info(f"行业覆盖率: {classified_count}/{total_count} = {classified_count/total_count*100:.2f}%")
        
        return result
    
    def save(self, df: cudf.DataFrame):
        """保存处理结果"""
        self.save_parquet(df, self.output_path)
        logger.info(f"行业分类宽表已保存到 {self.output_path}")
