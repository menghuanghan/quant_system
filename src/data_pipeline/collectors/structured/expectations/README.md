# 预期与预测分析数据域（Expectations & Forecasts）

## 概述

本模块提供预期与预测分析相关的数据采集功能，包括盈利预测、机构评级、研究员指数等数据。

## 数据类型

### 1. 盈利预测 (earnings_forecast.py)

| 数据类型 | 采集器类 | 主数据源 | 备用数据源 | 积分要求 |
|---------|---------|---------|-----------|---------|
| 业绩预告 | `EarningsForecastCollector` | Tushare (forecast) | AkShare (stock_yjyg_em) | 800 |
| 券商盈利预测 | `BrokerForecastCollector` | Tushare (report_rc) | AkShare (stock_rank_forecast_cninfo) | 5000 |
| 一致预期 | `ConsensusForecastCollector` | AkShare (stock_analyst_rank_em) | 聚合计算 | - |

### 2. 机构评级 (institutional_rating.py)

| 数据类型 | 采集器类 | 主数据源 | 备用数据源 | 积分要求 |
|---------|---------|---------|-----------|---------|
| 机构评级 | `InstitutionalRatingCollector` | AkShare (stock_rank_forecast_cninfo) | Tushare (report_rc) | - |
| 评级汇总 | `RatingSummaryCollector` | AkShare (stock_rank_cg_em) | - | - |
| 机构调研 | `InstitutionalSurveyCollector` | Tushare (stk_surv) | AkShare (stock_jgdy_tj_em) | 2000 |

### 3. 研究员指数 (analyst_index.py)

| 数据类型 | 采集器类 | 主数据源 | 备用数据源 | 积分要求 |
|---------|---------|---------|-----------|---------|
| 分析师排行 | `AnalystRankCollector` | AkShare (stock_analyst_rank_em) | - | - |
| 分析师详情 | `AnalystDetailCollector` | AkShare (stock_analyst_detail_em) | - | - |
| 券商金股 | `BrokerGoldStockCollector` | AkShare (stock_rank_xstp_ths) | - | - |
| 预测修正 | `ForecastRevisionCollector` | AkShare (stock_rank_forecast_cninfo) | - | - |

## 使用示例

```python
from src.data_pipeline.collectors.structured.expectations import (
    get_earnings_forecast,
    get_broker_forecast,
    get_inst_rating,
    get_analyst_rank,
    get_broker_gold_stock,
)

# 获取业绩预告
df_forecast = get_earnings_forecast(period='20231231')

# 获取券商盈利预测
df_broker = get_broker_forecast(ts_code='000001.SZ')

# 获取机构评级
df_rating = get_inst_rating(rating_date='20240115')

# 获取分析师排行
df_analyst = get_analyst_rank()

# 获取券商金股
df_gold = get_broker_gold_stock(month='202401')
```

## 输出字段说明

### 业绩预告 (EarningsForecastCollector)
- `ts_code`: 证券代码
- `ann_date`: 公告日期
- `end_date`: 报告期
- `type`: 预告类型（预增/预减/扭亏/首亏/续亏/续盈/略增/略减）
- `p_change_min/max`: 净利润变动幅度（%）
- `net_profit_min/max`: 预告净利润（万元）
- `summary`: 业绩预告摘要
- `change_reason`: 变动原因

### 机构评级 (InstitutionalRatingCollector)
- `ts_code`: 证券代码
- `rating_date`: 评级日期
- `org_name`: 评级机构
- `analyst_name`: 分析师
- `rating`: 投资评级
- `rating_change`: 评级变化
- `target_price_low/high`: 目标价区间
- `target_price`: 目标价中值

### 分析师排行 (AnalystRankCollector)
- `analyst_name`: 分析师姓名
- `org_name`: 所属机构
- `industry`: 研究行业
- `stock_count`: 关注股票数
- `avg_return`: 平均收益率（%）
- `success_rate`: 成功率（%）
- `rank`: 综合排名

## 注意事项

1. **积分限制**：部分Tushare接口需要较高积分（如report_rc需5000积分），系统会自动降级到AkShare
2. **数据延迟**：券商研报数据通常有1-2天的发布延迟
3. **历史数据**：业绩预告等数据可追溯多年，但近期数据更完整
4. **数据源切换**：系统会自动在数据源间切换，确保数据可用性
