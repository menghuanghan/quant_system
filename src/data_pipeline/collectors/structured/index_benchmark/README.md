# 指数与基准数据域（Index & Benchmark）

## 概述

本模块提供指数与基准相关数据的采集功能，支持多数据源自动切换（Tushare → AkShare → BaoStock）。

## 数据类型

### 1. 股票指数 (stock_index.py)

| 采集器 | 数据类型 | 主要字段 | Tushare积分 |
|--------|----------|----------|-------------|
| `IndexBasicCollector` | 指数基本信息 | ts_code, name, market, publisher, base_date | 120 |
| `IndexDailyCollector` | 指数日线行情 | ts_code, trade_date, open, high, low, close, vol | 120 |
| `IndexWeightCollector` | 指数成分权重 | index_code, con_code, trade_date, weight | 400 |
| `IndexMemberCollector` | 指数成分股 | index_code, con_code, con_name, in_date, is_new | 5000 |

### 2. 全球指数 (global_index.py)

| 采集器 | 数据类型 | 主要字段 | Tushare积分 |
|--------|----------|----------|-------------|
| `GlobalIndexCollector` | 全球指数行情 | ts_code, trade_date, open, close, pct_chg | 2000 |

## 便捷函数

```python
from src.data_pipeline.collectors.structured.index_benchmark import (
    get_index_basic,      # 获取指数基本信息
    get_index_daily,      # 获取指数日线行情
    get_index_weight,     # 获取指数成分权重
    get_index_member,     # 获取指数成分股
    get_index_global,     # 获取全球指数行情
)
```

## 使用示例

```python
# 获取上证综指基本信息
df = get_index_basic(ts_code='000001.SH')

# 获取沪深300全年日线行情
df = get_index_daily(ts_code='000300.SH', start_date='20250101', end_date='20251231')

# 获取沪深300最新成分权重
df = get_index_weight(index_code='000300.SH')

# 获取标普500历史行情
df = get_index_global(ts_code='SPX.GI', start_date='20250101')
```

## 数据源支持

| 数据类型 | Tushare | AkShare | BaoStock |
|----------|---------|---------|----------|
| 指数基本信息 | ✓ | ✓ | - |
| 指数日线行情 | ✓ | ✓ | ✓ |
| 指数成分权重 | ✓ | ✓ (部分) | - |
| 指数成分股 | ✓ | ✓ (部分) | - |
| 全球指数 | ✓ | ✓ | - |

## 注意事项

1. `IndexMemberCollector` 需要 5000 积分，2120 积分用户将自动降级到 AkShare
2. AkShare 的指数权重/成分股数据仅支持部分主流指数（沪深300、上证50、中证500等）
3. 全球指数代码格式：`SPX.GI`（标普500）、`IXIC.GI`（纳斯达克）、`DJI.GI`（道琼斯）
