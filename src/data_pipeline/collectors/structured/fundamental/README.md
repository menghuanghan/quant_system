# 公司基本面数据域（Fundamental Data）采集模块

## 模块概述

本模块提供公司基本面数据的采集功能，涵盖公司静态信息、财务报表和股权结构数据。

## 数据类型

### 1. 公司静态画像（Company Profile）

| 采集器 | 函数 | 说明 |
|--------|------|------|
| CompanyInfoCollector | `get_company_info()` | 上市公司基本信息 |
| IndustryClassCollector | `get_industry_class()` | 行业分类（申万/中信） |
| MainBusinessCollector | `get_main_business()` | 主营业务构成 |
| ManagementCollector | `get_management()` | 管理层信息 |

### 2. 财务报表体系（Financial Statements）

| 采集器 | 函数 | 说明 |
|--------|------|------|
| BalanceSheetCollector | `get_balance_sheet()` | 资产负债表 |
| IncomeStatementCollector | `get_income_statement()` | 利润表 |
| CashFlowCollector | `get_cash_flow()` | 现金流量表 |
| FinancialIndicatorCollector | `get_financial_indicator()` | 财务指标 |

### 3. 股权与资本结构（Ownership & Capital）

| 采集器 | 函数 | 说明 |
|--------|------|------|
| ShareStructureCollector | `get_share_structure()` | 股本结构 |
| Top10HoldersCollector | `get_top10_holders()` | 前十大股东 |
| PledgeCollector | `get_pledge()` | 股权质押 |
| ShareFloatCollector | `get_share_float()` | 限售解禁 |
| RepurchaseCollector | `get_repurchase()` | 股票回购 |
| DividendCollector | `get_dividend()` | 分红送股 |

## 快速开始

### 导入模块

```python
from src.data_pipeline.collectors.structured.fundamental import (
    # 公司画像
    get_company_info,
    get_industry_class,
    get_main_business,
    get_management,
    
    # 财务报表
    get_balance_sheet,
    get_income_statement,
    get_cash_flow,
    get_financial_indicator,
    
    # 股权结构
    get_share_structure,
    get_top10_holders,
    get_pledge,
    get_share_float,
    get_repurchase,
    get_dividend,
)
```

### 使用示例

#### 1. 获取公司信息

```python
# 获取公司基本信息
df = get_company_info(ts_code='000001.SZ')

# 获取行业分类
df = get_industry_class(ts_code='000001.SZ')

# 获取管理层信息
df = get_management(ts_code='000001.SZ')
```

#### 2. 获取财务报表

```python
# 资产负债表
df = get_balance_sheet(ts_code='000001.SZ', period='20231231')

# 利润表
df = get_income_statement(ts_code='000001.SZ', period='20231231')

# 现金流量表
df = get_cash_flow(ts_code='000001.SZ', period='20231231')

# 财务指标
df = get_financial_indicator(ts_code='000001.SZ', period='20231231')
```

#### 3. 获取股权信息

```python
# 前十大股东
df = get_top10_holders(ts_code='000001.SZ')

# 分红送股
df = get_dividend(ts_code='000001.SZ')

# 股权质押
df = get_pledge(ts_code='000001.SZ')
```

## API参考

### 公司信息

#### `get_company_info()`
| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 证券代码 |
| chairman | str | 法人代表 |
| manager | str | 总经理 |
| secretary | str | 董秘 |
| reg_capital | float | 注册资本 |
| setup_date | str | 注册日期 |
| province | str | 所在省份 |
| employees | int | 员工人数 |
| main_business | str | 主要业务 |

### 财务报表

#### `get_balance_sheet()` 主要字段
| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 证券代码 |
| end_date | str | 报告期 |
| total_assets | float | 资产总计 |
| total_liab | float | 负债合计 |
| total_cur_assets | float | 流动资产合计 |
| total_nca | float | 非流动资产合计 |
| money_cap | float | 货币资金 |
| accounts_receiv | float | 应收账款 |
| inventories | float | 存货 |
| fix_assets | float | 固定资产 |
| undist_profit | float | 未分配利润 |

#### `get_income_statement()` 主要字段
| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 证券代码 |
| end_date | str | 报告期 |
| total_revenue | float | 营业总收入 |
| revenue | float | 营业收入 |
| total_cogs | float | 营业总成本 |
| oper_cost | float | 营业成本 |
| sell_exp | float | 销售费用 |
| admin_exp | float | 管理费用 |
| fin_exp | float | 财务费用 |
| operate_profit | float | 营业利润 |
| total_profit | float | 利润总额 |
| n_income | float | 净利润 |
| basic_eps | float | 基本每股收益 |

#### `get_financial_indicator()` 主要字段
| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 证券代码 |
| end_date | str | 报告期 |
| eps | float | 基本每股收益 |
| bps | float | 每股净资产 |
| roe | float | 净资产收益率 |
| roa | float | 总资产报酬率 |
| gross_margin | float | 毛利率 |
| netprofit_margin | float | 销售净利率 |
| current_ratio | float | 流动比率 |
| quick_ratio | float | 速动比率 |
| debt_to_assets | float | 资产负债率 |
| netprofit_yoy | float | 净利润同比增长率 |
| revenue_yoy | float | 营业收入同比增长率 |

### 股权结构

#### `get_top10_holders()`
| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 证券代码 |
| end_date | str | 报告期 |
| holder_name | str | 股东名称 |
| hold_amount | float | 持股数量（股） |
| hold_ratio | float | 持股比例（%） |
| holder_type | str | 股东类型 |

#### `get_dividend()`
| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 证券代码 |
| end_date | str | 分红年度 |
| ann_date | str | 公告日期 |
| stk_div | float | 每股送转 |
| cash_div | float | 每股分红（元） |
| record_date | str | 股权登记日 |
| ex_date | str | 除权除息日 |

## 文件结构

```
fundamental/
├── __init__.py              # 模块导出
├── README.md                # 本文档
├── company_profile.py       # 公司静态画像采集器
├── financial_statement.py   # 财务报表采集器
└── ownership_capital.py     # 股权与资本结构采集器
```

## 数据源权限要求

| 接口 | Tushare积分要求 | 备用方案 |
|------|----------------|----------|
| stock_company | 120+ | AkShare |
| stk_managers | 2000+ | AkShare |
| fina_mainbz | 2000+ | - |
| balancesheet | 120+ | BaoStock |
| income | 120+ | BaoStock |
| cashflow | 120+ | BaoStock |
| fina_indicator | 120+ | BaoStock |
| top10_holders | 120+ | AkShare |
| dividend | 120+ | AkShare |
| pledge_stat | 120+ | - |
| share_float | 120+ | AkShare |
| repurchase | 120+ | - |

## 注意事项

1. **报告期格式**：财务数据的period参数使用YYYYMMDD格式（如20231231）
2. **报表类型**：默认获取合并报表，可通过report_type参数调整
3. **数据频率**：财务数据按季度发布，年报为12月31日
4. **数据延迟**：财务数据在公告后才能获取
5. **单位说明**：
   - 金额类字段通常为元
   - 股数类字段通常为股或万股
   - 比率类字段通常为百分比

## 更新日志

- v1.0.0 (2026-01-15): 初始版本
  - 支持公司静态信息采集
  - 支持三大财务报表采集
  - 支持财务指标采集
  - 支持股权结构信息采集
