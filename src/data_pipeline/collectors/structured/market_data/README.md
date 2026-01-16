# 市场行情数据域（Market Data）采集模块

## 模块概述

本模块提供市场行情数据的采集功能，是量化交易系统数据管道的核心组成部分。

### 数据类型

1. **K线与价格序列（Price & OHLCV）**
   - 股票日/周/月K线（支持前复权/后复权/不复权）
   - 指数日K线
   - ETF日K线

2. **实时与准实时行情（Realtime Market）**
   - 实时行情报价
   - 龙虎榜数据

3. **技术指标与衍生行情特征**
   - 每日基本指标（市盈率/市净率/换手率等）
   - 技术指标（MA/RSI/MACD/BOLL/KDJ）
   - 官方技术因子

## 数据源优先级

| 优先级 | 数据源 | 说明 |
|--------|--------|------|
| 1 | Tushare Pro | 主数据源，数据最全面 |
| 2 | AkShare | 备用数据源，免费 |
| 3 | BaoStock | 备用数据源，基础数据 |

## 快速开始

### 导入模块

```python
from src.data_pipeline.collectors.structured.market_data import (
    # K线数据
    get_stock_daily,
    get_stock_weekly,
    get_stock_monthly,
    get_index_daily,
    get_etf_daily,
    
    # 实时行情
    get_realtime_quote,
    get_top_list,
    
    # 技术指标
    get_daily_basic,
    get_technical_indicator,
    get_stk_factor,
)
```

### 使用示例

#### 1. 获取股票日K线

```python
# 获取平安银行前复权日K线
df = get_stock_daily(
    ts_code='000001.SZ',
    start_date='20240101',
    end_date='20241231',
    adj='qfq'  # 前复权
)

# 获取某日全市场日K线
df = get_stock_daily(trade_date='20240115')

# 不复权数据
df = get_stock_daily(ts_code='000001.SZ', adj=None)
```

#### 2. 获取指数日K线

```python
# 上证指数
df = get_index_daily(ts_code='000001.SH', start_date='20240101')

# 沪深300
df = get_index_daily(ts_code='000300.SH', start_date='20240101')
```

#### 3. 获取实时行情

```python
# 全市场实时行情
df = get_realtime_quote()

# 指定股票实时行情
df = get_realtime_quote(ts_codes=['000001.SZ', '600000.SH'])
```

#### 4. 获取技术指标

```python
# 计算技术指标（MA/RSI/MACD/BOLL/KDJ）
df = get_technical_indicator(
    ts_code='000001.SZ',
    start_date='20240101'
)

# 获取每日基本指标（PE/PB/换手率等）
df = get_daily_basic(ts_code='000001.SZ', trade_date='20240115')
```

## API参考

### K线数据

#### `get_stock_daily()`
获取股票日K线数据

**参数：**
- `ts_code` (str, optional): 证券代码
- `trade_date` (str, optional): 交易日期（获取全市场数据）
- `start_date` (str, optional): 开始日期（YYYYMMDD）
- `end_date` (str, optional): 结束日期（YYYYMMDD）
- `adj` (str, optional): 复权类型，'qfq'=前复权，'hfq'=后复权，None=不复权

**返回：**
| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 证券代码 |
| trade_date | str | 交易日期 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| pre_close | float | 昨收价 |
| change | float | 涨跌额 |
| pct_chg | float | 涨跌幅（%） |
| vol | float | 成交量（手） |
| amount | float | 成交额（千元） |

#### `get_stock_weekly()` / `get_stock_monthly()`
获取股票周/月K线数据，参数和返回值同 `get_stock_daily()`

#### `get_index_daily()`
获取指数日K线数据

**参数：**
- `ts_code` (str, optional): 指数代码（如 000001.SH）
- `trade_date` (str, optional): 交易日期
- `start_date` (str, optional): 开始日期
- `end_date` (str, optional): 结束日期

#### `get_etf_daily()`
获取ETF日K线数据

**参数：**
- `ts_code` (str, optional): ETF代码（如 510050.SH）
- `trade_date` (str, optional): 交易日期
- `start_date` (str, optional): 开始日期
- `end_date` (str, optional): 结束日期

### 实时行情

#### `get_realtime_quote()`
获取实时行情数据

**参数：**
- `ts_codes` (list, optional): 证券代码列表
- `market` (str, optional): 市场类型

**返回：**
| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 证券代码 |
| name | str | 证券名称 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 最新价 |
| pre_close | float | 昨收价 |
| change | float | 涨跌额 |
| pct_chg | float | 涨跌幅 |
| vol | float | 成交量 |
| amount | float | 成交额 |
| turnover_rate | float | 换手率 |
| pe_ratio | float | 市盈率 |
| pb_ratio | float | 市净率 |
| total_mv | float | 总市值 |
| circ_mv | float | 流通市值 |
| update_time | str | 更新时间 |

#### `get_top_list()`
获取龙虎榜数据

**参数：**
- `trade_date` (str, optional): 交易日期
- `ts_code` (str, optional): 证券代码

### 技术指标

#### `get_daily_basic()`
获取每日基本指标

**参数：**
- `ts_code` (str, optional): 证券代码
- `trade_date` (str, optional): 交易日期
- `start_date` (str, optional): 开始日期
- `end_date` (str, optional): 结束日期

**返回：**
| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 证券代码 |
| trade_date | str | 交易日期 |
| close | float | 收盘价 |
| turnover_rate | float | 换手率（%） |
| turnover_rate_f | float | 换手率（自由流通股） |
| volume_ratio | float | 量比 |
| pe | float | 市盈率 |
| pe_ttm | float | 市盈率TTM |
| pb | float | 市净率 |
| ps | float | 市销率 |
| ps_ttm | float | 市销率TTM |
| dv_ratio | float | 股息率（%） |
| dv_ttm | float | 股息率TTM |
| total_share | float | 总股本（万股） |
| float_share | float | 流通股本 |
| free_share | float | 自由流通股本 |
| total_mv | float | 总市值（万元） |
| circ_mv | float | 流通市值 |

#### `get_technical_indicator()`
计算技术指标

**参数：**
- `ts_code` (str, required): 证券代码（必填）
- `start_date` (str, optional): 开始日期
- `end_date` (str, optional): 结束日期
- `adj` (str, optional): 复权类型

**返回：**
| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 证券代码 |
| trade_date | str | 交易日期 |
| close | float | 收盘价 |
| ma5/10/20/60/120/250 | float | 均线 |
| ema12/26 | float | 指数均线 |
| macd | float | MACD值 |
| macd_dif | float | DIF |
| macd_dea | float | DEA |
| rsi6/12/24 | float | RSI指标 |
| boll_upper/mid/lower | float | 布林带 |
| kdj_k/d/j | float | KDJ指标 |

## 文件结构

```
market_data/
├── __init__.py          # 模块导出
├── README.md            # 本文档
├── price_kline.py       # K线与价格序列采集器
├── realtime.py          # 实时行情采集器
└── technical.py         # 技术指标采集器
```

## 数据源权限要求

| 接口 | Tushare积分要求 | 备用方案 |
|------|----------------|----------|
| daily（日K线） | 120+ | AkShare, BaoStock |
| weekly（周K线） | 120+ | AkShare, BaoStock |
| monthly（月K线） | 120+ | AkShare, BaoStock |
| index_daily（指数日K） | 2000+ | AkShare |
| fund_daily（ETF日K） | 2000+ | AkShare |
| daily_basic（基本指标） | 120+ | AkShare（部分） |
| stk_factor（技术因子） | 5000+ | 自计算替代 |
| top_list（龙虎榜） | 300+ | AkShare |

## 注意事项

1. **复权处理**：默认使用前复权（qfq），可通过adj参数调整
2. **日期格式**：统一使用YYYYMMDD格式输入，输出为YYYY-MM-DD格式
3. **单位说明**：
   - 成交量(vol)：手
   - 成交额(amount)：千元
   - 市值(mv)：万元
4. **数据源降级**：当主数据源失败时会自动降级到备用数据源
5. **技术指标**：`get_technical_indicator`基于K线数据实时计算，无需额外积分

## 更新日志

- v1.0.0 (2026-01-15): 初始版本
  - 支持股票/指数/ETF日K线
  - 支持周K线、月K线
  - 支持实时行情、龙虎榜
  - 支持技术指标计算
