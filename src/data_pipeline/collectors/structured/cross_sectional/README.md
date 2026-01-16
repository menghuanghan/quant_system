# Cross-sectional Structure 数据域（板块/行业/主题数据）

## 概述

本数据域负责采集A股市场的板块、行业和主题数据，包括：

1. **行业体系** (`industry.py`)
   - 申万行业分类（一级/二级/三级）
   - 申万行业成分股
   - 申万行业指数日行情

2. **概念与主题板块** (`concept.py`)
   - 同花顺概念板块列表及成分
   - 东方财富概念板块列表及成分
   - 开盘啦题材库及题材成分股

3. **板块行情与强弱** (`performance.py`)
   - 板块涨跌幅排行
   - 行业/概念板块实时行情
   - 板块历史日线行情
   - 板块热度排行
   - 涨停板池

## 数据源优先级

| 数据类型 | Tushare | AkShare | 备注 |
|---------|---------|---------|------|
| 申万行业分类 | index_classify | sw_index_first_info | Tushare需2000积分 |
| 申万行业成分 | index_member_all | sw_index_cons | Tushare需5000积分 |
| 申万行业行情 | sw_daily | sw_index_daily_indicator | Tushare需2000积分 |
| 同花顺概念列表 | ths_index | stock_board_concept_name_ths | Tushare需6000积分 |
| 同花顺概念成分 | ths_member | stock_board_concept_cons_ths | Tushare需6000积分 |
| 东方财富概念 | dc_index | stock_board_concept_name_em | Tushare需6000积分 |
| 板块行情 | - | stock_board_*_name_em | 仅AkShare |
| 涨停板池 | - | stock_zt_pool_em | 仅AkShare |

> **注意**: 用户当前Tushare积分为2120分，部分需要6000积分的接口会自动降级到AkShare

## 采集器详情

### 1. 行业体系采集器

#### SWIndexClassifyCollector - 申万行业分类

```python
from src.data_pipeline.collectors.structured.cross_sectional import get_sw_index_classify

# 获取一级行业分类
df = get_sw_index_classify(level='L1')

# 获取所有级别行业分类
df = get_sw_index_classify()  # 默认返回所有
```

**输出字段**:
| 字段名 | 类型 | 说明 |
|-------|-----|------|
| index_code | str | 申万指数代码 |
| index_name | str | 指数名称 |
| level | str | 行业级别(L1/L2/L3) |
| industry_name | str | 行业名称 |
| parent_code | str | 上级行业代码 |
| src | str | 数据来源 |
| is_pub | str | 是否发布指数 |
| pub_date | str | 发布日期 |

#### SWIndexMemberCollector - 申万行业成分

```python
from src.data_pipeline.collectors.structured.cross_sectional import get_sw_index_member

# 获取特定行业成分股
df = get_sw_index_member(index_code='801010.SI')  # 农林牧渔

# 获取全部成分关系
df = get_sw_index_member()
```

**输出字段**:
| 字段名 | 类型 | 说明 |
|-------|-----|------|
| index_code | str | 申万指数代码 |
| ts_code | str | 股票代码 |
| con_name | str | 股票名称 |
| in_date | str | 纳入日期 |
| out_date | str | 剔除日期 |
| is_new | str | 是否最新成分 |

#### SWDailyCollector - 申万行业指数日行情

```python
from src.data_pipeline.collectors.structured.cross_sectional import get_sw_daily

# 获取特定行业指数行情
df = get_sw_daily(index_code='801010.SI', start_date='20250101', end_date='20250115')
```

**输出字段**:
| 字段名 | 类型 | 说明 |
|-------|-----|------|
| index_code | str | 指数代码 |
| trade_date | str | 交易日期 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| vol | float | 成交量 |
| amount | float | 成交额 |
| pct_change | float | 涨跌幅(%) |
| pe | float | 市盈率 |
| pb | float | 市净率 |
| float_mv | float | 流通市值 |
| total_mv | float | 总市值 |

### 2. 概念与主题板块采集器

#### THSIndexCollector - 同花顺概念板块列表

```python
from src.data_pipeline.collectors.structured.cross_sectional import get_ths_index

# 获取同花顺概念板块列表
df = get_ths_index(index_type='N')  # N-概念, I-行业

# 获取所有板块
df = get_ths_index()
```

**输出字段**:
| 字段名 | 类型 | 说明 |
|-------|-----|------|
| ts_code | str | 板块代码 |
| name | str | 板块名称 |
| count | int | 成分股数量 |
| exchange | str | 交易所 |
| list_date | str | 上市日期 |
| index_type | str | 类型(N/I) |

#### THSMemberCollector - 同花顺概念成分

```python
from src.data_pipeline.collectors.structured.cross_sectional import get_ths_member

# 获取特定概念板块成分股
df = get_ths_member(ts_code='885938.TI')  # 人工智能
```

#### DCIndexCollector - 东方财富概念板块

```python
from src.data_pipeline.collectors.structured.cross_sectional import get_dc_index, get_dc_member

# 获取东方财富概念板块列表
df = get_dc_index()

# 获取特定板块成分股
df = get_dc_member(symbol='人工智能')
```

#### KPLConceptCollector - 开盘啦题材库

```python
from src.data_pipeline.collectors.structured.cross_sectional import get_kpl_concept, get_kpl_concept_cons

# 获取开盘啦题材库
df = get_kpl_concept(trade_date='20250115')

# 获取题材成分股
df = get_kpl_concept_cons(ts_code='xxx.KP')
```

### 3. 板块行情与强弱采集器

#### SectorPerformanceCollector - 板块涨跌幅排行

```python
from src.data_pipeline.collectors.structured.cross_sectional import get_sector_performance

# 获取行业板块涨跌幅
df = get_sector_performance(sector_type='industry')

# 获取概念板块涨跌幅
df = get_sector_performance(sector_type='concept')
```

**输出字段**:
| 字段名 | 类型 | 说明 |
|-------|-----|------|
| sector_code | str | 板块代码 |
| sector_name | str | 板块名称 |
| sector_type | str | 板块类型 |
| trade_date | str | 交易日期 |
| pct_change | float | 涨跌幅(%) |
| up_num | int | 上涨家数 |
| down_num | int | 下跌家数 |
| leading_stock | str | 领涨股 |
| leading_pct | float | 领涨股涨幅 |

#### SectorHistCollector - 板块历史行情

```python
from src.data_pipeline.collectors.structured.cross_sectional import get_sector_hist

# 获取概念板块历史行情
df = get_sector_hist('人工智能', sector_type='concept', start_date='20250101')

# 获取行业板块历史行情
df = get_sector_hist('银行', sector_type='industry')
```

#### SectorRankCollector - 板块热度排行

```python
from src.data_pipeline.collectors.structured.cross_sectional import get_sector_rank

# 按涨跌幅排行
df = get_sector_rank(sector_type='concept', rank_by='pct_change', top_n=20)

# 按换手率排行
df = get_sector_rank(sector_type='industry', rank_by='turnover_rate', top_n=30)
```

#### LimitUpPoolCollector - 涨停板池

```python
from src.data_pipeline.collectors.structured.cross_sectional import get_limit_up_pool

# 获取当日涨停板池
df = get_limit_up_pool()
```

**输出字段**:
| 字段名 | 类型 | 说明 |
|-------|-----|------|
| ts_code | str | 股票代码 |
| name | str | 股票名称 |
| trade_date | str | 交易日期 |
| close | float | 收盘价 |
| pct_change | float | 涨跌幅 |
| limit_up_time | str | 首次涨停时间 |
| open_times | int | 打开次数 |
| industry | str | 所属行业 |

## 采集脚本示例

创建 `scripts/collect_cross_sectional.py`:

```python
"""采集板块/行业/主题数据"""
import os
import pandas as pd
from datetime import datetime

from src.data_pipeline.collectors.structured.cross_sectional import (
    get_sw_index_classify,
    get_sw_index_member,
    get_ths_index,
    get_dc_index,
    get_sector_performance,
    get_sector_rank,
    get_limit_up_pool,
)

# 输出目录
OUTPUT_DIR = 'data/raw/cross_sectional'

def main():
    results = []
    
    # 1. 申万行业分类
    print("采集申万行业分类...")
    df = get_sw_index_classify()
    if not df.empty:
        df.to_csv(f'{OUTPUT_DIR}/industry/sw_index_classify.csv', index=False)
        results.append({'type': 'sw_index_classify', 'count': len(df)})
    
    # 2. 同花顺概念列表
    print("采集同花顺概念列表...")
    df = get_ths_index()
    if not df.empty:
        df.to_csv(f'{OUTPUT_DIR}/concept/ths_concept_index.csv', index=False)
        results.append({'type': 'ths_concept_index', 'count': len(df)})
    
    # 3. 板块涨跌幅排行
    print("采集板块涨跌幅...")
    df = get_sector_performance(sector_type='industry')
    if not df.empty:
        df.to_csv(f'{OUTPUT_DIR}/performance/industry_performance.csv', index=False)
        results.append({'type': 'industry_performance', 'count': len(df)})
    
    # 4. 涨停板池
    print("采集涨停板池...")
    df = get_limit_up_pool()
    if not df.empty:
        df.to_csv(f'{OUTPUT_DIR}/performance/limit_up_pool.csv', index=False)
        results.append({'type': 'limit_up_pool', 'count': len(df)})
    
    # 汇总
    summary = pd.DataFrame(results)
    summary.to_csv(f'{OUTPUT_DIR}/collection_summary.csv', index=False)
    print(f"采集完成，共 {len(results)} 类数据")

if __name__ == '__main__':
    os.makedirs(f'{OUTPUT_DIR}/industry', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/concept', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/performance', exist_ok=True)
    main()
```

## 目录结构

```
cross_sectional/
├── __init__.py          # 模块导出
├── README.md            # 本文档
├── industry.py          # 行业体系采集器
├── concept.py           # 概念板块采集器
└── performance.py       # 板块行情采集器
```

## 注意事项

1. **积分限制**: 部分Tushare接口需要高积分(6000+)，如积分不足会自动降级到AkShare
2. **日期格式**: 输入参数使用`YYYYMMDD`格式，输出统一为`YYYY-MM-DD`格式
3. **实时数据**: 部分接口只支持获取当日实时数据，不支持历史回溯
4. **频率限制**: AkShare部分接口有请求频率限制，大批量采集时注意添加延时
5. **数据更新**: 板块行情数据建议在交易时段采集以获取最新数据

## 更新日志

- 2025-01-15: 初始版本，实现行业/概念/板块行情三大类采集器
