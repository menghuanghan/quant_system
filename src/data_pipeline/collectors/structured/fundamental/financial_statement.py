"""
财务报表体系（Financial Statements）采集模块

数据类型包括：
- 资产负债表
- 利润表
- 现金流量表
- 财务指标（ROE/毛利率/杜邦）
"""

import logging
from typing import Optional, Literal
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

# 报告类型
ReportType = Literal['1', '2', '3', '4', '11']  # 1=合并报表，2=单季度，3=调整单季，4=调整合并，11=调整前合并


@CollectorRegistry.register("balance_sheet")
class BalanceSheetCollector(BaseCollector):
    """
    资产负债表采集器
    
    采集公司资产负债表数据
    主数据源：Tushare (balancesheet)
    备用数据源：AkShare, BaoStock
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'ann_date',             # 公告日期
        'end_date',             # 报告期
        'report_type',          # 报表类型
        'comp_type',            # 公司类型
        # 流动资产
        'total_cur_assets',     # 流动资产合计
        'money_cap',            # 货币资金
        'trad_asset',           # 交易性金融资产
        'accounts_receiv',      # 应收账款
        'oth_receiv',           # 其他应收款
        'prepayment',           # 预付款项
        'inventories',          # 存货
        # 非流动资产
        'total_nca',            # 非流动资产合计
        'lt_eqt_invest',        # 长期股权投资
        'invest_real_estate',   # 投资性房地产
        'fix_assets',           # 固定资产
        'cip',                  # 在建工程
        'intang_assets',        # 无形资产
        'goodwill',             # 商誉
        'lt_amor_exp',          # 长期待摊费用
        'defer_tax_assets',     # 递延所得税资产
        # 资产合计
        'total_assets',         # 资产总计
        # 流动负债
        'total_cur_liab',       # 流动负债合计
        'st_borr',              # 短期借款
        'acct_payable',         # 应付账款
        'taxes_payable',        # 应交税费
        # 非流动负债
        'total_ncl',            # 非流动负债合计
        'lt_borr',              # 长期借款
        'bond_payable',         # 应付债券
        # 负债合计
        'total_liab',           # 负债合计
        # 所有者权益
        'total_hldr_eqy_exc_min_int',  # 股东权益合计（不含少数股东权益）
        'total_hldr_eqy_inc_min_int',  # 股东权益合计（含少数股东权益）
        'cap_stk',              # 实收资本（或股本）
        'cap_rese',             # 资本公积
        'surplus_rese',         # 盈余公积
        'undist_profit',        # 未分配利润
        'minority_int',         # 少数股东权益
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        report_type: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集资产负债表数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            start_date: 公告开始日期
            end_date: 公告结束日期
            period: 报告期（YYYYMMDD，如20231231）
            report_type: 报告类型（1=合并报表）
        
        Returns:
            DataFrame: 标准化的资产负债表数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, start_date, end_date, period, report_type)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条资产负债表数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取资产负债表失败: {e}")
        
        # 降级到BaoStock
        try:
            if ts_code:
                df = self._collect_from_baostock(ts_code)
                if not df.empty:
                    logger.info(f"从BaoStock成功获取 {len(df)} 条资产负债表数据")
                    return df
        except Exception as e:
            logger.error(f"BaoStock获取资产负债表失败: {e}")
        
        logger.error("所有数据源均无法获取资产负债表数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        period: Optional[str],
        report_type: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取资产负债表"""
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
        if period:
            params['period'] = period
        if report_type:
            params['report_type'] = report_type
        
        df = pro.balancesheet(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['ann_date', 'f_ann_date', 'end_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_baostock(self, ts_code: str) -> pd.DataFrame:
        """从BaoStock获取资产负债表"""
        import baostock as bs
        
        if not self.source_manager.ensure_baostock_login():
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        symbol = ts_code.split('.')[0]
        exchange = ts_code.split('.')[1].lower()
        bs_code = f"{exchange}.{symbol}"
        
        rs = bs.query_balance_data(code=bs_code, year=datetime.now().year, quarter=4)
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 标准化字段映射
        column_mapping = {
            'pubDate': 'ann_date',
            'statDate': 'end_date',
            'totalAssets': 'total_assets',
            'totalLiab': 'total_liab',
            'totalShare': 'cap_stk',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("income_statement")
class IncomeStatementCollector(BaseCollector):
    """
    利润表采集器
    
    采集公司利润表数据
    主数据源：Tushare (income)
    备用数据源：BaoStock
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'ann_date',             # 公告日期
        'end_date',             # 报告期
        'report_type',          # 报表类型
        'comp_type',            # 公司类型
        # 营业收入
        'total_revenue',        # 营业总收入
        'revenue',              # 营业收入
        'int_income',           # 利息收入
        'prem_earned',          # 已赚保费
        # 营业成本
        'total_cogs',           # 营业总成本
        'oper_cost',            # 营业成本
        'int_exp',              # 利息支出
        'biz_tax_surchg',       # 营业税金及附加
        'sell_exp',             # 销售费用
        'admin_exp',            # 管理费用
        'fin_exp',              # 财务费用
        'rd_exp',               # 研发费用
        'assets_impair_loss',   # 资产减值损失
        # 营业利润
        'operate_profit',       # 营业利润
        'non_oper_income',      # 营业外收入
        'non_oper_exp',         # 营业外支出
        # 利润总额
        'total_profit',         # 利润总额
        'income_tax',           # 所得税费用
        # 净利润
        'n_income',             # 净利润（含少数股东损益）
        'n_income_attr_p',      # 净利润（归属母公司）
        'minority_gain',        # 少数股东损益
        # 每股指标
        'basic_eps',            # 基本每股收益
        'diluted_eps',          # 稀释每股收益
        'update_flag',          # 更新标识
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        report_type: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集利润表数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            start_date: 公告开始日期
            end_date: 公告结束日期
            period: 报告期
            report_type: 报告类型
        
        Returns:
            DataFrame: 标准化的利润表数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, start_date, end_date, period, report_type)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条利润表数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取利润表失败: {e}")
        
        # 降级到BaoStock
        try:
            if ts_code:
                df = self._collect_from_baostock(ts_code)
                if not df.empty:
                    logger.info(f"从BaoStock成功获取 {len(df)} 条利润表数据")
                    return df
        except Exception as e:
            logger.error(f"BaoStock获取利润表失败: {e}")
        
        logger.error("所有数据源均无法获取利润表数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        period: Optional[str],
        report_type: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取利润表"""
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
        if period:
            params['period'] = period
        if report_type:
            params['report_type'] = report_type
        
        df = pro.income(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['ann_date', 'f_ann_date', 'end_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_baostock(self, ts_code: str) -> pd.DataFrame:
        """从BaoStock获取利润表"""
        import baostock as bs
        
        if not self.source_manager.ensure_baostock_login():
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        symbol = ts_code.split('.')[0]
        exchange = ts_code.split('.')[1].lower()
        bs_code = f"{exchange}.{symbol}"
        
        rs = bs.query_profit_data(code=bs_code, year=datetime.now().year, quarter=4)
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        column_mapping = {
            'pubDate': 'ann_date',
            'statDate': 'end_date',
            'roeAvg': 'roe',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("cash_flow")
class CashFlowCollector(BaseCollector):
    """
    现金流量表采集器
    
    采集公司现金流量表数据
    主数据源：Tushare (cashflow)
    备用数据源：BaoStock
    """
    
    OUTPUT_FIELDS = [
        'ts_code',                  # 证券代码
        'ann_date',                 # 公告日期
        'end_date',                 # 报告期
        'report_type',              # 报表类型
        'comp_type',                # 公司类型
        # 经营活动现金流
        'n_cashflow_act',           # 经营活动产生的现金流量净额
        'c_fr_sale_sg',             # 销售商品、提供劳务收到的现金
        'c_pay_for_goods',          # 购买商品、接受劳务支付的现金
        'c_pay_to_for_empl',        # 支付给职工的现金
        'c_pay_for_tax',            # 支付的各项税费
        # 投资活动现金流
        'n_cashflow_inv_act',       # 投资活动产生的现金流量净额
        'c_fr_disp_fix_ast',        # 处置固定资产收回的现金
        'c_pay_acq_fix_ast',        # 购建固定资产支付的现金
        'c_pay_acq_stock',          # 投资支付的现金
        # 筹资活动现金流
        'n_cash_flows_fnc_act',     # 筹资活动产生的现金流量净额
        'c_fr_short_loan',          # 取得借款收到的现金
        'c_fr_issue_share',         # 吸收投资收到的现金
        'c_pay_repmt_debt',         # 偿还债务支付的现金
        'c_pay_div_profit',         # 分配股利、利润支付的现金
        # 现金净增加额
        'n_incr_cash_cash_equ',     # 现金及现金等价物净增加额
        'c_cash_equ_beg_period',    # 期初现金及现金等价物余额
        'c_cash_equ_end_period',    # 期末现金及现金等价物余额
        'update_flag',              # 更新标识
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        report_type: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集现金流量表数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            start_date: 公告开始日期
            end_date: 公告结束日期
            period: 报告期
            report_type: 报告类型
        
        Returns:
            DataFrame: 标准化的现金流量表数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, start_date, end_date, period, report_type)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条现金流量表数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取现金流量表失败: {e}")
        
        # 降级到BaoStock
        try:
            if ts_code:
                df = self._collect_from_baostock(ts_code)
                if not df.empty:
                    logger.info(f"从BaoStock成功获取 {len(df)} 条现金流量表数据")
                    return df
        except Exception as e:
            logger.error(f"BaoStock获取现金流量表失败: {e}")
        
        logger.error("所有数据源均无法获取现金流量表数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        period: Optional[str],
        report_type: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取现金流量表"""
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
        if period:
            params['period'] = period
        if report_type:
            params['report_type'] = report_type
        
        df = pro.cashflow(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['ann_date', 'f_ann_date', 'end_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_baostock(self, ts_code: str) -> pd.DataFrame:
        """从BaoStock获取现金流量表"""
        import baostock as bs
        
        if not self.source_manager.ensure_baostock_login():
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        symbol = ts_code.split('.')[0]
        exchange = ts_code.split('.')[1].lower()
        bs_code = f"{exchange}.{symbol}"
        
        rs = bs.query_cash_flow_data(code=bs_code, year=datetime.now().year, quarter=4)
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        column_mapping = {
            'pubDate': 'ann_date',
            'statDate': 'end_date',
            'CAToAsset': 'ca_to_asset',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


@CollectorRegistry.register("financial_indicator")
class FinancialIndicatorCollector(BaseCollector):
    """
    财务指标采集器
    
    采集公司财务指标数据（ROE/毛利率/杜邦分析等）
    主数据源：Tushare (fina_indicator)
    备用数据源：BaoStock
    """
    
    OUTPUT_FIELDS = [
        'ts_code',              # 证券代码
        'ann_date',             # 公告日期
        'end_date',             # 报告期
        # 每股指标
        'eps',                  # 基本每股收益
        'dt_eps',               # 稀释每股收益
        'bps',                  # 每股净资产
        'cfps',                 # 每股经营活动现金流
        'undist_profit_ps',     # 每股未分配利润
        'capital_rese_ps',      # 每股资本公积
        # 盈利能力
        'roe',                  # 净资产收益率
        'roe_waa',              # 加权平均净资产收益率
        'roe_dt',               # 净资产收益率-摊薄
        'roa',                  # 总资产报酬率
        'gross_margin',         # 毛利率
        'netprofit_margin',     # 销售净利率
        'op_income',            # 营业利润率
        # 偿债能力
        'current_ratio',        # 流动比率
        'quick_ratio',          # 速动比率
        'cash_ratio',           # 现金比率
        'ar_turn',              # 应收账款周转率
        'assets_turn',          # 总资产周转率
        # 杜邦分析
        'netprofit_yoy',        # 净利润同比增长率
        'or_yoy',               # 营业利润同比增长率
        'dt_eps_yoy',           # 每股收益同比增长率
        'bps_yoy',              # 每股净资产同比增长率
        # 资本结构
        'debt_to_assets',       # 资产负债率
        'ebit_of_gr',           # 息税前利润/营业收入
    ]
    
    def collect(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集财务指标数据
        
        Args:
            ts_code: 证券代码
            ann_date: 公告日期
            start_date: 公告开始日期
            end_date: 公告结束日期
            period: 报告期
        
        Returns:
            DataFrame: 标准化的财务指标数据
        """
        # 优先使用Tushare
        try:
            df = self._collect_from_tushare(ts_code, ann_date, start_date, end_date, period)
            if not df.empty:
                logger.info(f"从Tushare成功获取 {len(df)} 条财务指标数据")
                return df
        except Exception as e:
            logger.warning(f"Tushare获取财务指标失败: {e}")
        
        # 降级到BaoStock
        try:
            if ts_code:
                df = self._collect_from_baostock(ts_code)
                if not df.empty:
                    logger.info(f"从BaoStock成功获取 {len(df)} 条财务指标数据")
                    return df
        except Exception as e:
            logger.error(f"BaoStock获取财务指标失败: {e}")
        
        logger.error("所有数据源均无法获取财务指标数据")
        return pd.DataFrame(columns=self.OUTPUT_FIELDS)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_from_tushare(
        self,
        ts_code: Optional[str],
        ann_date: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        period: Optional[str]
    ) -> pd.DataFrame:
        """从Tushare获取财务指标"""
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
        if period:
            params['period'] = period
        
        df = pro.fina_indicator(**params)
        
        if df.empty:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = self._convert_date_format(df, ['ann_date', 'end_date'])
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]
    
    def _collect_from_baostock(self, ts_code: str) -> pd.DataFrame:
        """从BaoStock获取财务指标"""
        import baostock as bs
        
        if not self.source_manager.ensure_baostock_login():
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        symbol = ts_code.split('.')[0]
        exchange = ts_code.split('.')[1].lower()
        bs_code = f"{exchange}.{symbol}"
        
        # 获取盈利能力
        rs = bs.query_profit_data(code=bs_code, year=datetime.now().year, quarter=4)
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame(columns=self.OUTPUT_FIELDS)
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        column_mapping = {
            'pubDate': 'ann_date',
            'statDate': 'end_date',
            'roeAvg': 'roe',
            'npMargin': 'netprofit_margin',
            'gpMargin': 'gross_margin',
        }
        df = self._standardize_columns(df, column_mapping)
        df['ts_code'] = ts_code
        
        for col in self.OUTPUT_FIELDS:
            if col not in df.columns:
                df[col] = None
        
        return df[self.OUTPUT_FIELDS]


# ============= 便捷函数接口 =============

def get_balance_sheet(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    report_type: Optional[str] = None
) -> pd.DataFrame:
    """
    获取资产负债表数据
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        start_date: 开始日期
        end_date: 结束日期
        period: 报告期
        report_type: 报表类型
    
    Returns:
        DataFrame: 资产负债表数据
    
    Example:
        >>> df = get_balance_sheet(ts_code='000001.SZ', period='20231231')
    """
    collector = BalanceSheetCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            start_date=start_date, end_date=end_date,
                            period=period, report_type=report_type)


def get_income_statement(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    report_type: Optional[str] = None
) -> pd.DataFrame:
    """
    获取利润表数据
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        start_date: 开始日期
        end_date: 结束日期
        period: 报告期
        report_type: 报表类型
    
    Returns:
        DataFrame: 利润表数据
    
    Example:
        >>> df = get_income_statement(ts_code='000001.SZ', period='20231231')
    """
    collector = IncomeStatementCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            start_date=start_date, end_date=end_date,
                            period=period, report_type=report_type)


def get_cash_flow(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    report_type: Optional[str] = None
) -> pd.DataFrame:
    """
    获取现金流量表数据
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        start_date: 开始日期
        end_date: 结束日期
        period: 报告期
        report_type: 报表类型
    
    Returns:
        DataFrame: 现金流量表数据
    
    Example:
        >>> df = get_cash_flow(ts_code='000001.SZ', period='20231231')
    """
    collector = CashFlowCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            start_date=start_date, end_date=end_date,
                            period=period, report_type=report_type)


def get_financial_indicator(
    ts_code: Optional[str] = None,
    ann_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None
) -> pd.DataFrame:
    """
    获取财务指标数据
    
    Args:
        ts_code: 证券代码
        ann_date: 公告日期
        start_date: 开始日期
        end_date: 结束日期
        period: 报告期
    
    Returns:
        DataFrame: 财务指标数据
    
    Example:
        >>> df = get_financial_indicator(ts_code='000001.SZ', period='20231231')
    """
    collector = FinancialIndicatorCollector()
    return collector.collect(ts_code=ts_code, ann_date=ann_date,
                            start_date=start_date, end_date=end_date,
                            period=period)
