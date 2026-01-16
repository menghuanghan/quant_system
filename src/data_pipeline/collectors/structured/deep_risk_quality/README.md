# 深度风险与质量因子域（Deep Risk & Quality Factors）采集模块

## 模块概述

本模块负责采集深度风险与质量因子相关数据，包括估值扩散与拥挤度、资产质量异常以及ESG评价数据。这些数据用于构建风险评估模型和质量因子筛选策略。

## 数据域分类

### 1. 估值扩散与拥挤度
衡量市场整体估值水平和市场情绪指标：

- **A股等权重与中位数市盈率/市净率**: 全市场估值分布特征
- **大盘拥挤度**: 市场过热程度指标
- **股债利差**: 股票相对债券的吸引力
- **巴菲特指标**: 总市值/GDP，衡量股市整体估值水平

### 2. 资产质量异常
识别公司资产质量风险：

- **个股商誉明细**: 上市公司商誉规模及占比
- **商誉减值预期明细**: 商誉减值风险预警
- **破净股统计**: 市净率<1的股票统计

### 3. ESG评价
多维度ESG评级数据：

- **MSCI ESG评级**: 国际权威ESG评级
- **华证指数 ESG评级**: 华证ESG评价体系
- **路孚特 ESG评级**: Refinitiv ESG评分
- **秩鼎 ESG评级**: 秩鼎ESG评价

## 数据源说明

| 数据类型 | 主数据源 | 备用数据源 | 积分要求 |
|---------|---------|-----------|---------|
| A股市盈率/市净率 | AkShare | Tushare | 2000+ |
| 大盘拥挤度 | AkShare | 无 | - |
| 股债利差 | AkShare | 无 | - |
| 巴菲特指标 | AkShare | 无 | - |
| 个股商誉明细 | AkShare | Tushare | 2000+ |
| 商誉减值预期 | AkShare | 无 | - |
| 破净股统计 | AkShare | Tushare | 2000+ |
| MSCI ESG评级 | AkShare | 无 | - |
| 华证ESG评级 | AkShare | 无 | - |
| 路孚特ESG评级 | AkShare | 无 | - |
| 秩鼎ESG评级 | AkShare | 无 | - |

## 使用示例

### 1. 估值扩散与拥挤度数据采集

```python
from src.data_pipeline.collectors.structured.deep_risk_quality import (
    get_a_pe_pb_ew_median,
    get_market_congestion,
    get_stock_bond_spread,
    get_buffett_indicator
)

# 获取A股市盈率/市净率
pe_pb_data = get_a_pe_pb_ew_median(
    start_date='20240101',
    end_date='20241231'
)
print(f"市盈率/市净率数据: {len(pe_pb_data)} 条")

# 获取大盘拥挤度
congestion_data = get_market_congestion(
    start_date='20240101',
    end_date='20241231'
)
print(f"大盘拥挤度数据: {len(congestion_data)} 条")

# 获取股债利差
spread_data = get_stock_bond_spread(
    start_date='20240101',
    end_date='20241231'
)
print(f"股债利差数据: {len(spread_data)} 条")

# 获取巴菲特指标
buffett_data = get_buffett_indicator(
    start_date='20240101',
    end_date='20241231'
)
print(f"巴菲特指标数据: {len(buffett_data)} 条")
```

### 2. 资产质量异常数据采集

```python
from src.data_pipeline.collectors.structured.deep_risk_quality import (
    get_stock_goodwill,
    get_goodwill_impairment,
    get_break_net_stock
)

# 获取个股商誉明细（全市场）
goodwill_data = get_stock_goodwill()
print(f"商誉明细数据: {len(goodwill_data)} 条")

# 获取单只股票商誉明细
goodwill_single = get_stock_goodwill(
    ts_code='600000.SH',
    report_date='20241231'
)

# 获取商誉减值预期
impairment_data = get_goodwill_impairment()
print(f"商誉减值数据: {len(impairment_data)} 条")

# 获取破净股统计
break_net_data = get_break_net_stock(
    start_date='20240101',
    end_date='20241231'
)
print(f"破净股统计数据: {len(break_net_data)} 条")
```

### 3. ESG评级数据采集

```python
from src.data_pipeline.collectors.structured.deep_risk_quality import (
    get_esg_msci,
    get_esg_hz,
    get_esg_refinitiv,
    get_esg_zhiding
)

# 获取MSCI ESG评级（全市场）
msci_esg = get_esg_msci()
print(f"MSCI ESG评级: {len(msci_esg)} 条")

# 获取单只股票的各家ESG评级
ts_code = '600000.SH'
hz_esg = get_esg_hz(ts_code=ts_code)
refinitiv_esg = get_esg_refinitiv(ts_code=ts_code)
zhiding_esg = get_esg_zhiding(ts_code=ts_code)

print(f"华证ESG: {len(hz_esg)} 条")
print(f"路孚特ESG: {len(refinitiv_esg)} 条")
print(f"秩鼎ESG: {len(zhiding_esg)} 条")
```

### 4. 批量采集示例

```python
import pandas as pd
from src.data_pipeline.collectors.structured.deep_risk_quality import (
    APEPBEWMedianCollector,
    StockGoodwillCollector,
    MSCIESGCollector
)

# 创建采集器实例
valuation_collector = APEPBEWMedianCollector()
goodwill_collector = StockGoodwillCollector()
esg_collector = MSCIESGCollector()

# 批量采集
valuation_data = valuation_collector.collect(
    start_date='20240101',
    end_date='20241231'
)

goodwill_data = goodwill_collector.collect()
esg_data = esg_collector.collect()

# 保存数据
valuation_data.to_csv('data/raw/deep_risk_quality/valuation_data.csv', index=False)
goodwill_data.to_csv('data/raw/deep_risk_quality/goodwill_data.csv', index=False)
esg_data.to_csv('data/raw/deep_risk_quality/esg_data.csv', index=False)
```

## 数据字段说明

### A股市盈率/市净率

| 字段 | 类型 | 说明 |
|-----|------|-----|
| date | str | 日期 (YYYYMMDD) |
| pe_ew | float | 等权重市盈率 |
| pe_median | float | 中位数市盈率 |
| pb_ew | float | 等权重市净率 |
| pb_median | float | 中位数市净率 |

### 大盘拥挤度

| 字段 | 类型 | 说明 |
|-----|------|-----|
| date | str | 日期 (YYYYMMDD) |
| congestion | float | 拥挤度 |
| ma5 | float | 5日均线 |
| ma10 | float | 10日均线 |
| ma20 | float | 20日均线 |

### 股债利差

| 字段 | 类型 | 说明 |
|-----|------|-----|
| date | str | 日期 (YYYYMMDD) |
| spread | float | 股债利差 |
| stock_yield | float | 股票收益率 |
| bond_yield | float | 债券收益率 |

### 巴菲特指标

| 字段 | 类型 | 说明 |
|-----|------|-----|
| date | str | 日期 (YYYYMMDD) |
| buffett_index | float | 巴菲特指标 |
| total_market_cap | float | 总市值（亿元） |
| gdp | float | GDP（亿元） |

### 商誉明细

| 字段 | 类型 | 说明 |
|-----|------|-----|
| ts_code | str | 股票代码 |
| stock_name | str | 股票名称 |
| report_date | str | 报告期 |
| goodwill | float | 商誉（亿元） |
| net_assets | float | 净资产（亿元） |
| goodwill_ratio | float | 商誉占净资产比例(%) |
| total_assets | float | 总资产（亿元） |

### 商誉减值预期

| 字段 | 类型 | 说明 |
|-----|------|-----|
| ts_code | str | 股票代码 |
| stock_name | str | 股票名称 |
| report_date | str | 报告期 |
| goodwill | float | 商誉（亿元） |
| expected_impairment | float | 预期减值（亿元） |
| impairment_ratio | float | 减值占商誉比例(%) |

### 破净股统计

| 字段 | 类型 | 说明 |
|-----|------|-----|
| date | str | 统计日期 |
| total_count | int | 破净股总数 |
| break_net_count_sh | int | 上海破净股数量 |
| break_net_count_sz | int | 深圳破净股数量 |
| break_net_ratio | float | 破净股占比(%) |

### ESG评级

| 字段 | 类型 | 说明 |
|-----|------|-----|
| ts_code | str | 股票代码 |
| stock_name | str | 股票名称 |
| rating_date | str | 评级日期 |
| esg_rating | str | ESG评级 |
| esg_score | float | ESG得分 |
| environment_score | float | 环境得分 |
| social_score | float | 社会得分 |
| governance_score | float | 治理得分 |

## 注意事项

1. **日期格式**: 所有日期参数统一使用 `YYYYMMDD` 格式（如 `'20240115'`）
2. **股票代码**: 使用Tushare标准格式（如 `'600000.SH'`、`'000001.SZ'`）
3. **数据源切换**: 系统会自动尝试多个数据源，优先级为 Tushare → AkShare → BaoStock
4. **Tushare积分**: 部分接口需要2000+积分，请确保账户积分充足
5. **ESG数据覆盖**: ESG评级数据并非所有股票都有，仅覆盖主流机构评级的股票
6. **商誉数据**: 商誉数据通常按季度/年度更新，不是每个交易日都有新数据
7. **破净股统计**: Tushare备用源需要按日循环查询，数据量大时可能较慢

## 模块结构

```
deep_risk_quality/
├── __init__.py              # 模块初始化
├── README.md                # 本文档
├── valuation_dispersion.py # 估值扩散与拥挤度采集器
├── asset_quality.py         # 资产质量异常采集器
└── esg_ratings.py           # ESG评级采集器
```

## 更新日志

- **2026-01-16**: 初始版本发布
  - 实现4个估值扩散与拥挤度采集器
  - 实现3个资产质量异常采集器
  - 实现4个ESG评级采集器
  - 支持多数据源自动切换
  - 完善错误处理和日志记录
