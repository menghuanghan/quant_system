# 结构化数据采集模块 - 基础元数据域

## 概述

本模块实现了量化交易系统中**基础元数据域（Metadata Domain）**的原始数据采集功能，是整个数据管道的基础组件。

### 数据源优先级

- **主数据源**：Tushare Pro（需要2000+积分）
- **备用数据源**：AkShare、BaoStock

当主数据源不可用时，系统会自动切换到备用数据源。

## 数据类型

### 1. 证券与标的基础信息（Security Master）

| 数据类型 | 采集函数 | 说明 |
|---------|---------|------|
| A股股票列表 | `get_stock_list_a()` | 主板/中小板/创业板/科创板/北交所 |
| 港股股票列表 | `get_stock_list_hk()` | 港交所上市股票 |
| 美股股票列表 | `get_stock_list_us()` | NYSE/NASDAQ/AMEX |
| 股票曾用名 | `get_name_change()` | 历史名称变更记录 |
| ST标识状态 | `get_st_status()` | ST/*ST/风险警示标识 |
| A+H股票 | `get_ah_stock()` | 同时在A股和H股上市的股票 |

### 2. 交易日历与制度信息（Trading Calendar）

| 数据类型 | 采集函数 | 说明 |
|---------|---------|------|
| 交易日历 | `get_trade_calendar()` | 股票/期货各交易所日历 |
| 停复牌信息 | `get_suspend_info()` | 每日停复牌数据 |
| 涨跌停规则 | `get_price_limit_rule()` | 各板块涨跌停配置 |
| 集合竞价时间 | `get_auction_time()` | 各市场交易时间配置 |

## 快速开始

### 环境配置

确保在项目根目录的 `.env` 文件中配置了 Tushare Token：

```
TUSHARE_TOKEN=your_tushare_token_here
```

### 基本用法

```python
from src.data_pipeline.collectors.structured import (
    get_stock_list_a,
    get_trade_calendar,
    get_suspend_info,
)

# 获取A股上市股票列表
df = get_stock_list_a(list_status='L')
print(f"共 {len(df)} 只上市A股")

# 获取交易日历
calendar = get_trade_calendar(
    exchange='SSE',
    start_date='20250101',
    end_date='20251231'
)

# 获取停牌股票
suspend = get_suspend_info(trade_date='20250115', suspend_type='S')
```

## API 详细说明

### get_stock_list_a()

获取A股股票列表。

**参数：**
- `market` (str, 可选): 市场类型 - main(主板)/sme(中小板)/gem(创业板)/star(科创板)/bse(北交所)
- `exchange` (str, 可选): 交易所代码 - SSE(上交所)/SZSE(深交所)/BSE(北交所)
- `list_status` (str, 默认'L'): 上市状态 - L(上市)/D(退市)/P(暂停上市)
- `is_hs` (str, 可选): 是否沪深港通标的 - N(否)/H(沪股通)/S(深股通)

**输出字段：**
| 字段 | 类型 | 说明 |
|-----|------|------|
| ts_code | str | 证券代码（含交易所后缀） |
| symbol | str | 证券代码（纯数字） |
| name | str | 证券简称 |
| area | str | 地区 |
| industry | str | 所属行业 |
| fullname | str | 公司全称 |
| enname | str | 英文名称 |
| cnspell | str | 拼音缩写 |
| market | str | 市场类型 |
| exchange | str | 交易所代码 |
| curr_type | str | 交易货币 |
| list_status | str | 上市状态 |
| list_date | str | 上市日期 |
| delist_date | str | 退市日期 |
| is_hs | str | 是否沪深港通标的 |
| act_name | str | 实际控制人 |
| act_ent_type | str | 实控人企业性质 |

**示例：**
```python
# 获取所有上市A股
df = get_stock_list_a()

# 获取科创板股票
df = get_stock_list_a(market='star')

# 获取退市股票
df = get_stock_list_a(list_status='D')

# 获取沪股通标的
df = get_stock_list_a(is_hs='H')
```

### get_trade_calendar()

获取交易日历。

**参数：**
- `exchange` (str, 默认'SSE'): 交易所代码 - SSE/SZSE/BSE/CFFEX/SHFE/CZCE/DCE/INE
- `start_date` (str, 可选): 开始日期（YYYYMMDD格式）
- `end_date` (str, 可选): 结束日期（YYYYMMDD格式）
- `is_open` (int, 可选): 筛选交易日（1=交易日，0=非交易日）

**输出字段：**
| 字段 | 类型 | 说明 |
|-----|------|------|
| exchange | str | 交易所代码 |
| cal_date | str | 日历日期 |
| is_open | int | 是否交易日 |
| pretrade_date | str | 上一个交易日 |

**示例：**
```python
# 获取2025年交易日历
df = get_trade_calendar(start_date='20250101', end_date='20251231')

# 只获取交易日
df = get_trade_calendar(is_open=1)

# 获取期货交易日历
df = get_trade_calendar(exchange='SHFE')
```

### get_suspend_info()

获取停复牌信息。

**参数：**
- `ts_code` (str, 可选): 证券代码
- `trade_date` (str, 可选): 交易日期（YYYYMMDD格式）
- `suspend_type` (str, 可选): 停复牌类型 - S(停牌)/R(复牌)
- `start_date` (str, 可选): 开始日期
- `end_date` (str, 可选): 结束日期

**输出字段：**
| 字段 | 类型 | 说明 |
|-----|------|------|
| ts_code | str | 证券代码 |
| trade_date | str | 交易日期 |
| suspend_type | str | 停复牌类型 |
| suspend_timing | str | 停牌时间段 |
| suspend_reason | str | 停牌原因 |
| ann_date | str | 公告日期 |

**示例：**
```python
# 获取某日停牌股票
df = get_suspend_info(trade_date='20250115', suspend_type='S')

# 获取某股票的停复牌历史
df = get_suspend_info(ts_code='000001.SZ')
```

### 便捷函数

```python
from src.data_pipeline.collectors.structured import (
    get_trade_dates,
    is_trade_date,
)

# 获取交易日列表
dates = get_trade_dates(start_date='20250101', end_date='20250131')
# 返回: ['2025-01-02', '2025-01-03', '2025-01-06', ...]

# 判断是否交易日
result = is_trade_date('20250115')  # True
result = is_trade_date('20250118')  # False (周六)
```

## 采集器类

如果需要更灵活的控制，可以直接使用采集器类：

```python
from src.data_pipeline.collectors.structured import (
    StockListACollector,
    TradeCalendarCollector,
)

# 创建采集器实例
collector = StockListACollector()

# 调用采集方法
df = collector.collect(market='gem', list_status='L')
```

## 数据源自动切换

当主数据源（Tushare）不可用时，系统会自动切换到备用数据源：

```python
# 数据源优先级: Tushare -> AkShare -> BaoStock
# 切换是在函数内部自动完成的，对调用者透明
df = get_stock_list_a()  # 自动选择可用的数据源
```

## 注意事项

1. **积分要求**：Tushare Pro 部分接口需要一定积分才能访问，当前配置需要 2000+ 积分
2. **API 限流**：各数据源有调用频率限制，模块内置了重试机制
3. **日期格式**：输入参数使用 YYYYMMDD 格式，输出统一为 YYYY-MM-DD 格式
4. **数据缓存**：建议将基础数据（如股票列表）缓存到本地，避免频繁调用

## 扩展开发

如需添加新的采集器，可以继承 `BaseCollector` 基类：

```python
from src.data_pipeline.collectors.structured.base import (
    BaseCollector,
    CollectorRegistry,
    retry_on_failure,
)

@CollectorRegistry.register("my_collector")
class MyCollector(BaseCollector):
    OUTPUT_FIELDS = ['field1', 'field2']
    
    def collect(self, **kwargs):
        # 实现采集逻辑
        pass
```

## 文件结构

```
src/data_pipeline/collectors/structured/
├── __init__.py           # 模块导出
├── base.py               # 基类和工具函数
├── security_master.py    # 证券基础信息采集
└── trading_calendar.py   # 交易日历采集
```
