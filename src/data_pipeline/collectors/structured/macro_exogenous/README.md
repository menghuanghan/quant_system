# 宏观与外生变量数据域（Macro & Exogenous）

## 概述

本模块提供宏观经济和外生变量相关数据的采集功能，支持多数据源自动切换（Tushare → AkShare）。

## 数据类型

### 1. 国内宏观 (china_macro.py)

| 采集器 | 数据类型 | 主要字段 | Tushare积分 |
|--------|----------|----------|-------------|
| `ChinaGDPCollector` | 中国GDP | quarter, gdp, gdp_yoy, pi, si, ti | 120 |
| `ChinaCPICollector` | 中国CPI | month, nt_yoy, nt_mom, town_yoy | 120 |
| `ChinaPPICollector` | 中国PPI | month, ppi_yoy, ppi_mom, ppi_accu | 120 |
| `ChinaPMICollector` | 中国PMI | month, pmi, pmi_pro, pmi_order | 120 |
| `ChinaMoneySupplyCollector` | 货币供应量 | month, m0, m1, m2, m2_yoy | 800 |
| `ShiborCollector` | Shibor利率 | date, on, 1w, 1m, 3m, 6m, 1y | 500 |
| `LPRCollector` | LPR利率 | date, lpr_1y, lpr_5y | 500 |
| `SocialFinanceCollector` | 社会融资规模 | month, total, rmb_loan, corp_bond | 2000 |

### 2. 国际宏观 (global_macro.py)

| 采集器 | 数据类型 | 主要字段 | Tushare积分 |
|--------|----------|----------|-------------|
| `USTreasuryCollector` | 美国国债收益率 | date, m3, y1, y2, y5, y10, y30 | 2000 |
| `EcoCalendarCollector` | 全球经济日历 | date, country, event, actual, previous | 5000 |

### 3. 行业经济 (industry_economy.py)

| 采集器 | 数据类型 | 主要字段 | Tushare积分 |
|--------|----------|----------|-------------|
| `BoxOfficeCollector` | 电影票房 | date, movie, box_office, avg_price | 1000 |
| `CarSalesCollector` | 汽车销量 | month, brand, model, sales_vol | 2000 |

## 便捷函数

```python
from src.data_pipeline.collectors.structured.macro_exogenous import (
    # 国内宏观
    get_cn_gdp,           # 获取中国GDP
    get_cn_cpi,           # 获取中国CPI
    get_cn_ppi,           # 获取中国PPI
    get_cn_pmi,           # 获取中国PMI
    get_cn_m2,            # 获取货币供应量
    get_shibor,           # 获取Shibor利率
    get_lpr,              # 获取LPR利率
    get_sf,               # 获取社会融资规模
    
    # 国际宏观
    get_us_treasury,      # 获取美国国债收益率
    get_eco_calendar,     # 获取经济日历
    
    # 行业经济
    get_box_office,       # 获取电影票房
    get_car_sales,        # 获取汽车销量
)
```

## 使用示例

```python
# 获取中国GDP数据
df = get_cn_gdp(start_q='2024Q1', end_q='2025Q4')

# 获取CPI月度数据
df = get_cn_cpi(start_m='202401', end_m='202512')

# 获取Shibor利率
df = get_shibor(start_date='20250101', end_date='20251231')

# 获取LPR利率
df = get_lpr()

# 获取美国国债收益率
df = get_us_treasury(start_date='20250101')

# 获取经济日历
df = get_eco_calendar(date='20250115')

# 获取电影票房
df = get_box_office(date='20250115')

# 获取汽车销量
df = get_car_sales(month='202501')
```

## 数据源支持

| 数据类型 | Tushare | AkShare |
|----------|---------|---------|
| GDP | ✓ | ✓ |
| CPI | ✓ | ✓ |
| PPI | ✓ | ✓ |
| PMI | ✓ | ✓ |
| 货币供应量 | ✓ | ✓ |
| Shibor | ✓ | ✓ |
| LPR | ✓ | ✓ |
| 社融 | ✓ | ✓ |
| 美国国债 | ✓ | ✓ (部分) |
| 经济日历 | ✓ | ✓ |
| 电影票房 | ✓ | ✓ |
| 汽车销量 | ✓ | ✓ |

## 注意事项

1. 部分接口需要较高积分（如社融需2000、经济日历需5000），积分不足时自动降级到 AkShare
2. AkShare 的宏观数据更新可能有延迟
3. 经济日历数据覆盖全球主要经济事件
4. 行业经济数据（票房、汽车销量）可用于行业研究和另类数据分析
