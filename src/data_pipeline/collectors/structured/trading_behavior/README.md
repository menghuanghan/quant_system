# 资金与交易行为数据域（Trading Behavior）采集模块

## 模块概述

本模块提供资金与交易行为数据的采集功能，涵盖资金流向、融资融券和特殊交易行为数据。

## 数据类型

### 1. 资金流向（Capital Flow）

| 采集器 | 函数 | 说明 |
|--------|------|------|
| MoneyFlowCollector | `get_money_flow()` | 个股资金流向 |
| MoneyFlowIndustryCollector | `get_money_flow_industry()` | 行业/板块资金流向 |
| MoneyFlowMarketCollector | `get_money_flow_market()` | 大盘资金流向 |
| HSGTFlowCollector | `get_hsgt_flow()` | 沪深港通资金 |

### 2. 融资融券与杠杆行为

| 采集器 | 函数 | 说明 |
|--------|------|------|
| MarginSummaryCollector | `get_margin_summary()` | 融资融券汇总 |
| MarginDetailCollector | `get_margin_detail()` | 融资融券明细 |
| MarginTargetCollector | `get_margin_target()` | 两融标的 |
| SLBCollector | `get_slb()` | 转融通 |

### 3. 特殊交易行为

| 采集器 | 函数 | 说明 |
|--------|------|------|
| TopListCollector | `get_top_list()` | 龙虎榜 |
| TopInstCollector | `get_top_inst()` | 龙虎榜营业部明细 |
| BlockTradeCollector | `get_block_trade()` | 大宗交易 |

## 快速开始

### 导入模块

```python
from src.data_pipeline.collectors.structured.trading_behavior import (
    # 资金流向
    get_money_flow,
    get_money_flow_industry,
    get_money_flow_market,
    get_hsgt_flow,
    
    # 融资融券
    get_margin_summary,
    get_margin_detail,
    get_margin_target,
    get_slb,
    
    # 特殊交易
    get_top_list,
    get_top_inst,
    get_block_trade,
)
```

### 使用示例

#### 1. 获取个股资金流向

```python
# 获取平安银行资金流向
df = get_money_flow(ts_code='000001.SZ', start_date='20240101')

# 获取某日全市场资金流向
df = get_money_flow(trade_date='20240115')
```

#### 2. 获取沪深港通资金

```python
# 获取近30日北向资金
df = get_hsgt_flow(start_date='20240101', end_date='20240131')
```

#### 3. 获取融资融券数据

```python
# 融资融券汇总
df = get_margin_summary(trade_date='20240115')

# 个股融资融券明细
df = get_margin_detail(ts_code='000001.SZ')
```

#### 4. 获取龙虎榜

```python
# 龙虎榜上榜股票
df = get_top_list(trade_date='20240115')

# 龙虎榜营业部明细
df = get_top_inst(trade_date='20240115')
```

## API参考

### 资金流向

#### `get_money_flow()` 主要字段
| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 证券代码 |
| trade_date | str | 交易日期 |
| buy_sm_vol | float | 小单买入量（手） |
| buy_sm_amount | float | 小单买入金额（万元） |
| buy_lg_vol | float | 大单买入量 |
| buy_lg_amount | float | 大单买入金额 |
| buy_elg_vol | float | 特大单买入量 |
| buy_elg_amount | float | 特大单买入金额 |
| net_mf_vol | float | 净流入量 |
| net_mf_amount | float | 净流入金额 |

#### `get_hsgt_flow()` 主要字段
| 字段 | 类型 | 说明 |
|------|------|------|
| trade_date | str | 交易日期 |
| hgt | float | 沪股通净流入 |
| sgt | float | 深股通净流入 |
| north_money | float | 北向资金（沪+深） |
| south_money | float | 南向资金 |

### 融资融券

#### `get_margin_summary()` 主要字段
| 字段 | 类型 | 说明 |
|------|------|------|
| trade_date | str | 交易日期 |
| exchange_id | str | 交易所（SSE/SZSE） |
| rzye | float | 融资余额（元） |
| rzmre | float | 融资买入额 |
| rqye | float | 融券余额 |
| rzrqye | float | 融资融券余额 |

### 特殊交易

#### `get_top_list()` 主要字段
| 字段 | 类型 | 说明 |
|------|------|------|
| trade_date | str | 交易日期 |
| ts_code | str | 证券代码 |
| name | str | 证券名称 |
| pct_change | float | 涨跌幅（%） |
| l_buy | float | 龙虎榜买入额（万元） |
| l_sell | float | 龙虎榜卖出额 |
| net_amount | float | 龙虎榜净买入额 |
| reason | str | 上榜理由 |

## 文件结构

```
trading_behavior/
├── __init__.py              # 模块导出
├── README.md                # 本文档
├── capital_flow.py          # 资金流向采集器
├── margin_trading.py        # 融资融券采集器
└── special_trading.py       # 特殊交易行为采集器
```

## 数据源权限要求

| 接口 | Tushare积分要求 | 备用方案 |
|------|----------------|----------|
| moneyflow | 2000+ | AkShare |
| moneyflow_hsgt | 120+ | AkShare |
| margin | 400+ | AkShare |
| margin_detail | 2000+ | AkShare |
| top_list | 300+ | AkShare |
| top_inst | 300+ | AkShare |
| block_trade | 600+ | AkShare |

## 注意事项

1. **资金流向数据**：个股资金流向需要较高积分，建议使用AkShare作为备用
2. **融资融券**：明细数据更新频率为日度
3. **龙虎榜**：仅包含当日上榜股票，非每日有数据
4. **大宗交易**：成交后T+1日披露
5. **沪深港通**：北向资金包含沪股通+深股通

## 更新日志

- v1.0.0 (2026-01-15): 初始版本
  - 支持个股/行业/大盘资金流向
  - 支持沪深港通资金流向
  - 支持融资融券汇总和明细
  - 支持龙虎榜和大宗交易
