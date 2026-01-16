# 衍生品与多资产数据域（Derivatives & Multi-Asset）

本模块提供衍生品与多资产数据的采集功能，涵盖ETF基金、期货期权、债券可转债等多种资产类型。

## 数据类型概览

| 分类 | 采集器 | 数据源 | Tushare积分 | 描述 |
|------|--------|--------|-------------|------|
| **ETF与基金** |
| 基金基本信息 | `FundBasicCollector` | Tushare/AkShare | 2000 | ETF、LOF等基金列表 |
| 基金日线行情 | `FundDailyCollector` | Tushare/AkShare | 5000 | ETF日线价格与成交量 |
| 基金净值 | `FundNavCollector` | Tushare/AkShare | 2000 | 基金单位净值与累计净值 |
| 基金持仓 | `FundPortfolioCollector` | Tushare/AkShare | 5000 | 基金重仓股持仓明细 |
| 基金规模 | `FundShareCollector` | Tushare/AkShare | 2000 | 基金份额与规模变化 |
| **期货与期权** |
| 期货基本信息 | `FuturesBasicCollector` | Tushare/AkShare | 2000 | 期货合约列表与规格 |
| 期货日线行情 | `FuturesDailyCollector` | Tushare/AkShare | 2000 | 期货日线价格与持仓 |
| 期货持仓排名 | `FuturesHoldingCollector` | Tushare/AkShare | 2000 | 交易所会员持仓排名 |
| 期货仓单 | `FuturesWarehouseCollector` | Tushare/AkShare | 2000 | 交易所仓单日报 |
| 期权基本信息 | `OptionsBasicCollector` | Tushare/AkShare | 5000 | 期权合约列表 |
| 期权日线行情 | `OptionsDailyCollector` | Tushare/AkShare | 2000 | 期权日线价格与Greeks |
| **债券与可转债** |
| 国债收益率曲线 | `YieldCurveCollector` | Tushare/AkShare | 特殊权限 | 中债国债收益率曲线 |
| 可转债基本信息 | `CBBasicCollector` | Tushare/AkShare | 2000 | 可转债列表与转股信息 |
| 可转债行情 | `CBDailyCollector` | Tushare/AkShare | 2000 | 可转债日线与溢价率 |
| 债券回购 | `RepoDailyCollector` | Tushare/AkShare | 2000 | 国债逆回购行情 |
| 可转债溢价率 | `CBPremiumCollector` | AkShare | - | 转股溢价率与纯债溢价率 |

## 快速开始

### 安装依赖

```bash
pip install tushare akshare pandas
```

### 基础用法

```python
from src.data_pipeline.collectors.structured.derivatives import (
    # ETF与基金
    get_fund_basic, get_fund_daily, get_fund_nav,
    # 期货与期权
    get_fut_basic, get_fut_daily, get_opt_daily,
    # 债券与可转债
    get_cb_basic, get_cb_daily, get_cb_premium,
)

# 1. 获取ETF列表
etf_list = get_fund_basic(market='E')  # E=场内ETF
print(f"共获取 {len(etf_list)} 只ETF")

# 2. 获取ETF日线行情
etf_daily = get_fund_daily(ts_code='510050.SH', start_date='20250101')

# 3. 获取期货合约
fut_list = get_fut_basic(exchange='DCE')  # 大连商品交易所

# 4. 获取期货行情
fut_daily = get_fut_daily(ts_code='I2501.DCE', start_date='20250101')

# 5. 获取可转债列表
cb_list = get_cb_basic()

# 6. 获取可转债溢价率
cb_premium = get_cb_premium()
```

## 详细API文档

### ETF与基金

#### get_fund_basic() - 基金基本信息

```python
def get_fund_basic(
    ts_code: Optional[str] = None,  # 基金代码
    market: Optional[str] = None,   # 市场: E=场内, O=场外
    status: Optional[str] = None,   # 状态: D=退市, I=发行, L=上市中
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | TS代码 |
| name | str | 基金名称 |
| management | str | 管理人 |
| custodian | str | 托管人 |
| fund_type | str | 投资类型 |
| found_date | str | 成立日期 |
| due_date | str | 到期日期 |
| list_date | str | 上市时间 |
| issue_date | str | 发行日期 |
| delist_date | str | 退市日期 |
| issue_amount | float | 发行份额(亿) |
| m_fee | float | 管理费 |
| c_fee | float | 托管费 |
| duration_year | float | 存续期 |
| p_value | float | 面值 |
| min_amount | float | 起点金额(万元) |
| exp_return | float | 预期收益率 |
| benchmark | str | 业绩比较基准 |
| status | str | 存续状态 |
| invest_type | str | 投资风格 |
| type | str | 基金类型 |
| trustee | str | 受托人 |
| purc_startdate | str | 申购起始日 |
| redm_startdate | str | 赎回起始日 |
| market | str | 上市市场 |

#### get_fund_daily() - 基金日线行情

```python
def get_fund_daily(
    ts_code: Optional[str] = None,    # 基金代码
    trade_date: Optional[str] = None, # 交易日期（YYYYMMDD）
    start_date: Optional[str] = None, # 开始日期
    end_date: Optional[str] = None,   # 结束日期
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | TS代码 |
| trade_date | str | 交易日期 |
| pre_close | float | 昨收盘价 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| change | float | 涨跌额 |
| pct_chg | float | 涨跌幅(%) |
| vol | float | 成交量(手) |
| amount | float | 成交金额(万元) |

#### get_fund_nav() - 基金净值

```python
def get_fund_nav(
    ts_code: Optional[str] = None,    # 基金代码
    nav_date: Optional[str] = None,   # 净值日期
    start_date: Optional[str] = None, # 开始日期
    end_date: Optional[str] = None,   # 结束日期
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | TS代码 |
| ann_date | str | 公告日期 |
| nav_date | str | 净值日期 |
| unit_nav | float | 单位净值 |
| accum_nav | float | 累计净值 |
| accum_div | float | 累计分红 |
| net_asset | float | 资产净值 |
| total_netasset | float | 合计资产净值 |
| adj_nav | float | 复权净值 |

#### get_fund_portfolio() - 基金持仓

```python
def get_fund_portfolio(
    ts_code: Optional[str] = None,   # 基金代码
    ann_date: Optional[str] = None,  # 公告日期
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | 基金代码 |
| ann_date | str | 公告日期 |
| end_date | str | 截止日期 |
| symbol | str | 股票代码 |
| mkv | float | 市值(元) |
| amount | float | 持有股票数量(股) |
| stk_mkv_ratio | float | 市值占总资产比例 |
| stk_float_ratio | float | 占流通股本比例 |

#### get_fund_share() - 基金规模

```python
def get_fund_share(
    ts_code: Optional[str] = None,    # 基金代码
    trade_date: Optional[str] = None, # 交易日期
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | TS代码 |
| trade_date | str | 交易日期 |
| fd_share | float | 基金份额(万份) |

### 期货与期权

#### get_fut_basic() - 期货基本信息

```python
def get_fut_basic(
    exchange: Optional[str] = None,  # 交易所: CFFEX/DCE/CZCE/SHFE/INE/GFEX
    fut_type: Optional[str] = None,  # 合约类型: 1=普通合约, 2=主力, 3=连续
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | 合约代码 |
| symbol | str | 交易标识 |
| exchange | str | 交易所 |
| name | str | 中文简称 |
| fut_code | str | 合约产品代码 |
| multiplier | float | 合约乘数 |
| trade_unit | str | 交易计量单位 |
| per_unit | float | 交易单位(数量) |
| quote_unit | str | 报价单位 |
| quote_unit_desc | str | 最小报价单位说明 |
| d_mode_desc | str | 交割方式说明 |
| list_date | str | 上市日期 |
| delist_date | str | 最后交易日期 |
| d_month | str | 交割月份 |
| last_ddate | str | 最后交割日 |
| trade_time_desc | str | 交易时间说明 |

#### get_fut_daily() - 期货日线行情

```python
def get_fut_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    exchange: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | TS合约代码 |
| trade_date | str | 交易日期 |
| pre_close | float | 昨收盘价 |
| pre_settle | float | 昨结算价 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| settle | float | 结算价 |
| change1 | float | 涨跌1(收盘-昨结算) |
| change2 | float | 涨跌2(收盘-昨收盘) |
| vol | float | 成交量(手) |
| amount | float | 成交金额(万元) |
| oi | float | 持仓量(手) |
| oi_chg | float | 持仓变化 |
| delv_settle | float | 交割结算价 |

#### get_fut_holding() - 期货持仓排名

```python
def get_fut_holding(
    trade_date: str,                   # 交易日期（必填）
    symbol: Optional[str] = None,      # 品种代码
    exchange: Optional[str] = None,    # 交易所
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| trade_date | str | 交易日期 |
| symbol | str | 品种代码 |
| broker | str | 期货公司 |
| vol | float | 成交量 |
| vol_chg | float | 成交变化 |
| long_hld | float | 持买仓量 |
| long_chg | float | 持买仓变化 |
| short_hld | float | 持卖仓量 |
| short_chg | float | 持卖仓变化 |
| exchange | str | 交易所 |

#### get_fut_wsr() - 期货仓单

```python
def get_fut_wsr(
    trade_date: str,                   # 交易日期（必填）
    symbol: Optional[str] = None,      # 品种代码
    exchange: Optional[str] = None,    # 交易所
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| trade_date | str | 交易日期 |
| symbol | str | 产品代码 |
| warehouse | str | 仓库简称 |
| wh_code | str | 仓库编号 |
| vol | float | 库存 |
| vol_chg | float | 增减 |
| area | str | 地区 |
| year | str | 年度 |
| grade | str | 等级 |
| brand | str | 品牌 |
| exchange | str | 交易所 |

#### get_opt_basic() - 期权基本信息

```python
def get_opt_basic(
    exchange: Optional[str] = None,  # 交易所: SSE/SZSE/CFFEX/DCE/CZCE/SHFE
    opt_code: Optional[str] = None,  # 标准合约代码
    call_put: Optional[str] = None,  # 期权类型: C=认购, P=认沽
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | TS代码 |
| exchange | str | 交易所 |
| name | str | 合约名称 |
| per_unit | float | 合约单位 |
| opt_code | str | 标准合约代码 |
| opt_type | str | 合约类型 |
| call_put | str | 期权类型 |
| exercise_type | str | 行权方式 |
| exercise_price | float | 行权价格 |
| s_month | str | 结算月 |
| maturity_date | str | 到期日 |
| list_price | float | 挂牌基准价 |
| list_date | str | 开始交易日期 |
| delist_date | str | 最后交易日期 |
| last_edate | str | 最后行权日期 |
| last_ddate | str | 最后交割日期 |
| quote_unit | str | 报价单位 |
| min_price_chg | str | 最小价格波幅 |

#### get_opt_daily() - 期权日线行情

```python
def get_opt_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    exchange: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | TS代码 |
| trade_date | str | 交易日期 |
| exchange | str | 交易所 |
| pre_settle | float | 昨结算价 |
| pre_close | float | 昨收盘价 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| settle | float | 结算价 |
| vol | float | 成交量(手) |
| amount | float | 成交金额(万元) |
| oi | float | 持仓量(手) |

### 债券与可转债

#### get_yield_curve() - 国债收益率曲线

```python
def get_yield_curve(
    ts_code: str = '1001.CB',          # 收益率曲线编码
    curve_type: str = '0',             # 曲线类型: 0=到期, 1=即期
    trade_date: Optional[str] = None,  # 交易日期
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| trade_date | str | 交易日期 |
| ts_code | str | 曲线编码 |
| curve_name | str | 曲线名称 |
| curve_type | str | 曲线类型 |
| curve_term | float | 期限(年) |
| yield | float | 收益率(%) |

#### get_cb_basic() - 可转债基本信息

```python
def get_cb_basic(
    ts_code: Optional[str] = None,    # 转债代码
    list_date: Optional[str] = None,  # 上市日期
    exchange: Optional[str] = None,   # 上市地点
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | 转债代码 |
| bond_full_name | str | 转债名称 |
| bond_short_name | str | 转债简称 |
| stk_code | str | 正股代码 |
| stk_short_name | str | 正股简称 |
| maturity | float | 发行期限(年) |
| issue_size | float | 发行总额(元) |
| list_date | str | 上市日期 |
| conv_price | float | 最新转股价 |
| newest_rating | str | 最新信用等级 |

#### get_cb_daily() - 可转债行情

```python
def get_cb_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | 转债代码 |
| trade_date | str | 交易日期 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| pct_chg | float | 涨跌幅(%) |
| vol | float | 成交量(手) |
| amount | float | 成交金额(万元) |
| bond_value | float | 纯债价值 |
| cb_value | float | 转股价值 |
| cb_over_rate | float | 转股溢价率(%) |

#### get_repo_daily() - 债券回购行情

```python
def get_repo_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | TS代码 |
| trade_date | str | 交易日期 |
| repo_maturity | str | 期限品种 |
| open | float | 开盘价(%) |
| high | float | 最高价(%) |
| low | float | 最低价(%) |
| close | float | 收盘价(%) |
| weight | float | 加权价(%) |
| amount | float | 成交金额(万元) |

#### get_cb_premium() - 可转债溢价率

```python
def get_cb_premium(
    ts_code: Optional[str] = None,  # 转债代码
) -> pd.DataFrame
```

返回字段：
| 字段 | 类型 | 描述 |
|------|------|------|
| ts_code | str | 转债代码 |
| bond_short_name | str | 转债简称 |
| stk_code | str | 正股代码 |
| close | float | 转债价格 |
| stk_close | float | 正股价格 |
| conv_price | float | 转股价 |
| cb_value | float | 转股价值 |
| cb_over_rate | float | 转股溢价率(%) |
| bond_value | float | 纯债价值 |
| double_low | float | 双低值 |
| ytm | float | 到期收益率(%) |

## 使用示例

### 示例1：构建ETF投资组合

```python
from src.data_pipeline.collectors.structured.derivatives import (
    get_fund_basic, get_fund_daily, get_fund_nav
)
import pandas as pd

# 1. 获取场内ETF列表
etf_list = get_fund_basic(market='E', status='L')
print(f"当前上市ETF数量: {len(etf_list)}")

# 2. 筛选宽基指数ETF
broad_etfs = ['510050.SH', '510300.SH', '510500.SH', '159919.SZ']

# 3. 获取行情数据
for code in broad_etfs:
    df = get_fund_daily(ts_code=code, start_date='20250101')
    print(f"{code}: {len(df)} 条行情记录")
```

### 示例2：期货价差分析

```python
from src.data_pipeline.collectors.structured.derivatives import (
    get_fut_basic, get_fut_daily
)

# 1. 获取铁矿石合约
i_contracts = get_fut_basic(exchange='DCE')
i_contracts = i_contracts[i_contracts['fut_code'] == 'I']

# 2. 获取主力合约行情
main_contract = 'I2505.DCE'  # 假设主力为05合约
df_main = get_fut_daily(ts_code=main_contract, start_date='20250101')

# 3. 计算价差
# ...
```

### 示例3：可转债低溢价策略

```python
from src.data_pipeline.collectors.structured.derivatives import get_cb_premium

# 获取全市场可转债溢价率
cb_premium = get_cb_premium()

# 筛选低溢价转债
low_premium = cb_premium[
    (cb_premium['cb_over_rate'] < 10) &  # 溢价率<10%
    (cb_premium['close'] < 130)          # 价格<130
].sort_values('double_low')              # 按双低排序

print(f"低溢价转债数量: {len(low_premium)}")
print(low_premium[['ts_code', 'bond_short_name', 'close', 'cb_over_rate', 'double_low']].head(10))
```

## 数据源说明

### Tushare

- 需要2000+积分才能使用大部分接口
- 部分接口（fund_daily, fund_portfolio, opt_basic）需要5000+积分
- 国债收益率曲线(yc_cb)需要特殊权限
- 配置Token: 在 `.env` 文件中设置 `TUSHARE_TOKEN`

### AkShare

- 免费开源数据源，无积分限制
- 数据质量和及时性可能略逊于Tushare
- 作为Tushare不可用时的备用数据源

## 注意事项

1. **积分要求**: 部分Tushare API需要高积分，系统会自动降级到AkShare
2. **数据延迟**: AkShare部分数据可能有延迟，建议优先使用Tushare
3. **日期格式**: 所有日期参数使用 `YYYYMMDD` 格式
4. **代码格式**: 
   - 基金代码: `510050.SH`, `159919.SZ`
   - 期货代码: `I2505.DCE`, `CU2503.SHFE`
   - 可转债代码: `110030.SH`, `128036.SZ`
5. **交易所代码**:
   - 期货: CFFEX(中金所), DCE(大商所), CZCE(郑商所), SHFE(上期所), INE(能源中心), GFEX(广期所)
   - 期权: SSE(上交所), SZSE(深交所), CFFEX, DCE, CZCE, SHFE
