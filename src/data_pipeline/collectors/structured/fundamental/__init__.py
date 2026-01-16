"""
公司基本面数据域（Fundamental Data）采集模块

本模块提供公司基本面数据的采集功能，涵盖以下数据类型：

1. 公司静态画像（Company Profile）：
   - 上市公司基本信息
   - 行业归属（申万/中信/巨潮）
   - 主营业务介绍
   - 管理层信息

2. 财务报表体系（Financial Statements）：
   - 资产负债表
   - 利润表
   - 现金流量表
   - 财务指标（ROE/毛利率/杜邦）

3. 股权与资本结构（Ownership & Capital）：
   - 股本结构
   - 前十大股东/流通股东
   - 股权质押
   - 限售解禁
   - 股票回购
   - 分红送股

数据源优先级：Tushare > AkShare > BaoStock
"""

# 基类和工具（从父目录导入）
from ..base import (
    BaseCollector,
    DataSource,
    DataSourceManager,
    DataSourcePriority,
    CollectorRegistry,
    StandardFields,
    retry_on_failure,
    fallback_on_error,
)

# 公司静态画像采集器
from .company_profile import (
    # 采集器类
    CompanyInfoCollector,
    IndustryClassCollector,
    MainBusinessCollector,
    ManagementCollector,
    # 便捷函数
    get_company_info,
    get_industry_class,
    get_main_business,
    get_management,
)

# 财务报表采集器
from .financial_statement import (
    # 采集器类
    BalanceSheetCollector,
    IncomeStatementCollector,
    CashFlowCollector,
    FinancialIndicatorCollector,
    # 便捷函数
    get_balance_sheet,
    get_income_statement,
    get_cash_flow,
    get_financial_indicator,
)

# 股权与资本结构采集器
from .ownership_capital import (
    # 采集器类
    ShareStructureCollector,
    Top10HoldersCollector,
    PledgeCollector,
    ShareFloatCollector,
    RepurchaseCollector,
    DividendCollector,
    # 便捷函数
    get_share_structure,
    get_top10_holders,
    get_pledge,
    get_share_float,
    get_repurchase,
    get_dividend,
)


__all__ = [
    # 基类和工具
    'BaseCollector',
    'DataSource',
    'DataSourceManager',
    'DataSourcePriority',
    'CollectorRegistry',
    'StandardFields',
    'retry_on_failure',
    'fallback_on_error',
    
    # 公司静态画像
    'CompanyInfoCollector',
    'IndustryClassCollector',
    'MainBusinessCollector',
    'ManagementCollector',
    'get_company_info',
    'get_industry_class',
    'get_main_business',
    'get_management',
    
    # 财务报表
    'BalanceSheetCollector',
    'IncomeStatementCollector',
    'CashFlowCollector',
    'FinancialIndicatorCollector',
    'get_balance_sheet',
    'get_income_statement',
    'get_cash_flow',
    'get_financial_indicator',
    
    # 股权与资本结构
    'ShareStructureCollector',
    'Top10HoldersCollector',
    'PledgeCollector',
    'ShareFloatCollector',
    'RepurchaseCollector',
    'DividendCollector',
    'get_share_structure',
    'get_top10_holders',
    'get_pledge',
    'get_share_float',
    'get_repurchase',
    'get_dividend',
]


# 版本信息
__version__ = '1.0.0'
__author__ = 'Quant Team'
