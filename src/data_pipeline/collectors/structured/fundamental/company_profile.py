"""
公司静态画像（Company Profile）采集模块

数据类型包括：
- 上市公司基本信息
- 行业归属（申万/中信/巨潮）
- 主营业务介绍
- 管理层信息
"""

import logging
from typing import Optional, List, Literal
from datetime import datetime

import pandas as pd

from ..base import (
    BaseCollector,
    DataSource,
    DataSourceManager,
    retry_on_failure,
    StandardFields,
    CollectorRegistry
)

logger = logging.getLogger(__name__)


@CollectorRegistry.register("company_info")
class CompanyInfoCollector(BaseCollector):
    """
    上市公司基本信息采集器
    
    采集公司注册信息、法人代表、注册资本等
    主数据源：Tushare (stock_company)
    备用数据源：AkShare, BaoStock
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'exchange',         # 交易所
        'chairman',         # 法人代表
        'manager',          # 总经理
        'secretary',        # 董秘
        'reg_capital',      # 注册资本（万元）
        'setup_date',       # 注册日期
        'province',         # 所在省份
        'city',             # 所在城市
        'introduction',     # 公司介绍
        'website',          # 公司主页
        'email',            # 电子邮件
        'office',           # 办公室
        'ann_date',         # 公告日期
        'employees',        # 员工人数
        'main_business',    # 主要业务及产品
        'business_scope',   # 经营范围
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        exchange: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集上市公司基本信息
        
        Args:
            ts_code: 证券代码
            exchange: 交易所（SSE/SZSE/BSE）
        
        Returns:
            DataFrame: 标准化的公司基本信息数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, exchange)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条公司信息")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取公司信息失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(ts_code)
            if not df.empty:
                logger.info(f"从AkShare成功获取 {len(df)} 条公司信息")
                return df
        except Exception as e:
            logger.error(f"AkShare获取公司信息失败: {e}")
        
        logger.error("所有数据源均无法获取公司信息")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        exchange: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取公司信息"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if exchange:
            params['exchange'] = exchange
        
        df = pro.stock_company(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化日期格式
        df = self._convert_date_format(df, ['setup_date', 'ann_date'])
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, ts_code: Optional[str]) -> pd.DataFrame:
        """从AkShare获取公司信息"""
        import akshare as ak
        
        if not ts_code:
            logger.warning("AkShare需要指定ts_code")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        symbol = ts_code.split('.')[0]
        
        try:
            df = ak.stock_individual_info_em(symbol=symbol)
        except Exception as e:
            logger.warning(f"AkShare获取公司信息失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 转换为字典格式
        info_dict = dict(zip(df['item'], df['value']))
        
        result = pd.DataFrame([{
            'ts_code': ts_code,
            'exchange': ts_code.split('.')[1] if '.' in ts_code else None,
            'chairman': info_dict.get('法定代表人'),
            'reg_capital': info_dict.get('注册资本'),
            'setup_date': info_dict.get('成立日期'),
            'province': None,
            'city': info_dict.get('注册地址', '').split('省')[0] if info_dict.get('注册地址') else None,
            'website': info_dict.get('公司网站'),
            'email': info_dict.get('电子邮箱'),
            'employees': info_dict.get('员工人数'),
            'main_business': info_dict.get('经营范围'),
        }])
        
        for col in self.OUTPUT_FIELDS:
            if col not in result.columns:
                result[col] = None
        
        return result[self.OUTPUT_FIELDS]


@CollectorRegistry.register("industry_class")
class IndustryClassCollector(BaseCollector):
    """
    行业分类采集器
    
    采集申万、中信、巨潮等行业分类
    主数据源：Tushare (stock_basic + index_classify)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'name',             # 证券名称
        'industry',         # 行业（通用）
        'sw_l1',            # 申万一级行业
        'sw_l2',            # 申万二级行业
        'sw_l3',            # 申万三级行业
        'zx_l1',            # 中信一级行业
        'jc_l1',            # 巨潮一级行业
        'jc_l2',            # 巨潮二级行业
        'update_date',      # 更新日期
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        src: str = 'SW2021',
        **kwargs
    ) -> pd.DataFrame:
        """
        采集行业分类数据
        
        Args:
            ts_code: 证券代码
            src: 分类来源（SW2021=申万2021版，默认）
        
        Returns:
            DataFrame: 标准化的行业分类数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, src, **kwargs)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条行业分类数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取行业分类失败: {e}")
        
        # 降级到AkShare
        try:
            df = self._collect_from_akshare(**kwargs)
            if not df.empty:
                if ts_code:
                    df = df[df['ts_code'] == ts_code]
                logger.info(f"从AkShare成功获取 {len(df)} 条行业分类数据")
                return df
        except Exception as e:
            logger.error(f"AkShare获取行业分类失败: {e}")
        
        logger.error("所有数据源均无法获取行业分类数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        src: str,
        **kwargs
    ) -> pd.DataFrame:
        """从Tushare获取行业分类"""
        pro = self.tushare_api
        
        # 获取股票基本信息中的行业字段
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        
        df = pro.stock_basic(**params, fields='ts_code,name,industry')
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 尝试获取申万行业分类
        try:
            # 获取申万一级行业成分股
            sw_members = pro.index_member(index_code='', is_new='Y')
            if not sw_members.empty:
                # 合并行业分类信息
                df = df.merge(
                    sw_members[['con_code', 'index_name']].rename(
                        columns={'con_code': 'ts_code', 'index_name': 'sw_l1'}
                    ),
                    on='ts_code',
                    how='left'
                )
        except Exception as e:
            logger.debug(f"获取申万行业分类失败: {e}")
        
        # 设置更新日期：优先使用传入的 end_date，否则使用当前
        eff_date = kwargs.get('end_date') or datetime.now().strftime('%Y%m%d')
        if len(eff_date.replace('-', '')) == 8:
            eff_date = f"{eff_date[:4]}-{eff_date[4:6]}-{eff_date[6:8]}"
        df['update_date'] = eff_date
        
        # 确保包含所有字段
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, **kwargs) -> pd.DataFrame:
        """从AkShare获取行业分类（巨潮资讯）"""
        import akshare as ak
        
        # 获取上下文日期
        end_date = kwargs.get('end_date') or datetime.now().strftime('%Y%m%d')
        update_date = end_date
        if len(update_date.replace('-', '')) == 8:
            update_date = f"{update_date[:4]}-{update_date[4:6]}-{update_date[6:8]}"
        
        try:
            # 使用巨潮资讯的行业分类数据
            # 先获取所有股票的行业分类
            df_stocks = ak.stock_zh_a_spot_em()  # 获取所有A股列表
            
            if df_stocks.empty:
                logger.warning("无法获取A股列表")
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            # 为每只股票获取行业分类
            results = []
            for _, row in df_stocks.head(100).iterrows():  # 限制数量以避免过多请求
                symbol = row['代码']
                name = row['名称']
                
                try:
                    # 构建ts_code
                    if symbol.startswith(('6', '5')):
                        ts_code = f"{symbol}.SH"
                    else:
                        ts_code = f"{symbol}.SZ"
                    
                    # 尝试获取该股票的行业变动情况（包含行业分类）
                    industry_df = ak.stock_industry_change_cninfo(
                        symbol=symbol,
                        start_date="20200101",
                        end_date=end_date.replace('-', '')
                    )
                    
                    if not industry_df.empty:
                        # 取最新的行业分类
                        latest = industry_df.iloc[0]
                        results.append({
                            'ts_code': ts_code,
                            'name': name,
                            'industry': latest.get('行业大类'),
                            'jc_l1': latest.get('行业大类'),
                            'jc_l2': latest.get('行业中类'),
                            'update_date': update_date
                        })
                except Exception as e:
                    logger.debug(f"获取{symbol}行业分类失败: {e}")
                    continue
            
            if not results:
                logger.warning("未能获取任何行业分类数据")
                return pd.DataFrame(columns=self.OUTPUT_FIELDS)
            
            df = pd.DataFrame(results)
            
            # 确保包含所有字段
            for col in self.OUTPUT_FIELDS:
                if col not in df.columns:
                    df[col] = None
            
            return df[self.OUTPUT_FIELDS]
            
        except Exception as e:
            logger.warning(f"AkShare获取行业分类失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)


@CollectorRegistry.register("main_business")
class MainBusinessCollector(BaseCollector):
    """
    主营业务采集器
    
    采集公司主营业务构成
    主数据源：Tushare (fina_mainbz)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'end_date',         # 报告期
        'bz_item',          # 主营业务项目
        'bz_sales',         # 主营业务收入（元）
        'bz_profit',        # 主营业务利润（元）
        'bz_cost',          # 主营业务成本（元）
        'curr_type',        # 货币类型
        'update_flag',      # 更新标识
    ]
    
    def collect(
        self,
        ts_code: str,
        period: Optional[str] = None,
        type: str = 'P',
        **kwargs
    ) -> pd.DataFrame:
        """
        采集主营业务数据
        
        Args:
            ts_code: 证券代码（必填）
            period: 报告期（YYYYMMDD）
            type: 类型（P=产品，D=地区）
        
        Returns:
            DataFrame: 标准化的主营业务数据
        """
        if not ts_code:
            logger.error("采集主营业务需要指定ts_code")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, period, type)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条主营业务数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取主营业务失败: {e}")
        
        logger.error("无法获取主营业务数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: str,
        period: Optional[str],
        type: str,
        **kwargs
    ) -> pd.DataFrame:
        """从Tushare获取主营业务"""
        import time
        
        pro = self.tushare_api
        
        params = {'ts_code': ts_code, 'type': type}
        if period:
            params['period'] = period
        
        df = pro.fina_mainbz(**params)
        
        # 频率控制：避免触发 Tushare 限速（60次/分钟）
        time.sleep(1.0)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['end_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("management")
class ManagementCollector(BaseCollector):
    """
    管理层信息采集器
    
    采集公司高管信息
    主数据源：Tushare (stk_managers)
    """
    
    OUTPUT_FIELDS = [
        'ts_code',          # 证券代码
        'ann_date',         # 公告日期
        'name',             # 姓名
        'gender',           # 性别
        'lev',              # 岗位类别
        'title',            # 岗位
        'edu',              # 学历
        'national',         # 国籍
        'birthday',         # 出生年月
        'begin_date',       # 上任日期
        'end_date',         # 离任日期
        'resume',           # 简历
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集管理层信息
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 标准化的管理层信息数据
        """
        # 使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, start_date, end_date)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条管理层信息")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取管理层信息失败: {e}")
        
        # 降级到AkShare
        try:
            if ts_code:
                df = self._collect_from_akshare(ts_code)
                if not df.empty:
                    logger.info(f"从AkShare成功获取 {len(df)} 条管理层信息")
                    return df
        except Exception as e:
            logger.error(f"AkShare获取管理层信息失败: {e}")
        
        logger.error("无法获取管理层信息")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取管理层信息"""
        pro = self.tushare_api
        
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if ann_date:
            params['ann_date'] = ann_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        df = pro.stk_managers(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['ann_date', 'begin_date', 'end_date', 'birthday'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_akshare(self, ts_code: str) -> pd.DataFrame:
        """从AkShare获取管理层信息"""
        import akshare as ak
        
        symbol = ts_code.split('.')[0]
        
        try:
            df = ak.stock_gg_em(symbol=symbol)
        except Exception as e:
            logger.warning(f"AkShare获取高管信息失败: {e}")
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        # 标准化字段
        column_mapping = {
            '姓名': 'name',
            '职务': 'title',
            '性别': 'gender',
            '学历': 'edu',
            '出生日期': 'birthday',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_company_info(
    ts_code: Optional[str] = None,
    exchange: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    获取上市公司基本信息
    
    Args:
        ts_code: 证券代码
        exchange: 交易所
    
    Returns:
        DataFrame: 公司基本信息
    
    Example:
        >>> df = get_company_info(ts_code='000001.SZ')
    """
    collector = CompanyInfoCollector()
    return collector.collect(ts_code=ts_code, exchange=exchange, **kwargs)


def get_industry_class(
    ts_code: Optional[str] = None,
    src: str = 'SW2021',
    **kwargs
) -> pd.DataFrame:
    """
    获取行业分类数据
    
    Args:
        ts_code: 证券代码
        src: 分类来源
    
    Returns:
        DataFrame: 行业分类数据
    
    Example:
        >>> df = get_industry_class(ts_code='000001.SZ')
    """
    collector = IndustryClassCollector()
    return collector.collect(ts_code=ts_code, src=src, **kwargs)


def get_main_business(
    ts_code: str,
    period: Optional[str] = None,
    type: str = 'P',
    **kwargs
) -> pd.DataFrame:
    """
    获取主营业务数据
    
    Args:
        ts_code: 证券代码（必填）
        period: 报告期
        type: P=产品，D=地区
    
    Returns:
        DataFrame: 主营业务数据
    
    Example:
        >>> df = get_main_business(ts_code='000001.SZ')
    """
    collector = MainBusinessCollector()
    return collector.collect(ts_code=ts_code, period=period, type=type, **kwargs)


def get_management(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取管理层信息
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 管理层信息
    
    Example:
        >>> df = get_management(ts_code='000001.SZ')
    """
    collector = ManagementCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            start_date=start_date, end_date=end_date)
