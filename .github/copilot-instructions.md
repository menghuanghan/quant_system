# Copilot Instructions - 量化交易系统

## 系统架构概览

这是一个模块化的A股量化交易系统，采用**数据域驱动架构（Data Domain Architecture）**组织代码，包含：

- **数据管道层** (`src/data_pipeline/collectors/structured/`): 按8个数据域采集原始数据
- **特征工程层** (`src/feature_engineering/`): 特征计算、融合和注册机制
- **模型层** (`src/models/`): ML/DL/RL模型实现
- **回测引擎** (`src/backtesting/`): 策略回测与性能分析
- **执行层** (`src/execution/`): 订单管理与成本模型
- **风险控制** (`src/risk_management/`): 限制、监控与风控逻辑
- **投资组合** (`src/portfolio/`): 组合优化、配置与再平衡

## 数据域架构（核心设计）

所有数据采集器按业务域组织在 `src/data_pipeline/collectors/structured/` 下：

1. **metadata**: 基础元数据（股票列表、交易日历、停复牌）
2. **market_data**: 市场行情（K线、实时报价、技术指标）
3. **fundamental**: 公司基本面（财报、股权结构、管理层）
4. **trading_behavior**: 资金与交易行为（资金流向、融资融券、龙虎榜）
5. **cross_sectional**: 板块/行业/主题数据
6. **derivatives**: 衍生品与多资产数据
7. **index_benchmark**: 指数与基准数据
8. **macro_exogenous**: 宏观与外生变量

**原则**: 每个数据域独立，通过标准字段（如 `ts_code`, `trade_date`）关联。

## 采集器模式（关键约定）

### 基类继承与注册
所有采集器继承 `BaseCollector` 并通过装饰器注册：

```python
from src.data_pipeline.collectors.structured.base import BaseCollector, CollectorRegistry

@CollectorRegistry.register("stock_daily")
class StockDailyCollector(BaseCollector):
    OUTPUT_FIELDS = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
    
    def collect(self, ts_code: str, start_date: Optional[str] = None, **kwargs) -> pd.DataFrame:
        # 实现采集逻辑
        pass
```

### 多数据源降级策略
系统默认优先级：**Tushare → AkShare → BaoStock**

- 使用 `@retry_on_failure(max_retries=3)` 实现重试逻辑
- Tushare需2000+积分，失败时自动降级
- 数据源管理通过单例 `DataSourceManager` 统一处理

### 便捷函数接口
每个采集器配套一个公共函数（放在模块 `__init__.py`）：

```python
def get_stock_daily(ts_code: str, start_date: Optional[str] = None, ...) -> pd.DataFrame:
    collector = StockDailyCollector()
    return collector.collect(ts_code=ts_code, start_date=start_date, ...)
```

**示例**: 参考 [src/data_pipeline/collectors/structured/market_data/price_kline.py](src/data_pipeline/collectors/structured/market_data/price_kline.py) 的 `StockDailyCollector` 和 `get_stock_daily()`。

## 开发工作流

### 环境配置
1. 在 `.env` 设置 `TUSHARE_TOKEN` 和数据库配置
2. Docker服务：TimescaleDB (5432)、MongoDB (27600)、Redis (6379)
3. 启动容器：`docker-compose -f docker/docker-compose.yml up -d`

### 数据采集脚本
使用 `scripts/` 目录的样例脚本测试采集器：
- `collect_metadata.py`: 采集股票列表、交易日历等元数据
- `collect_market_data.py`: 采集K线、实时行情
- `collect_fundamental.py`: 采集财报、管理层信息
- `collect_trading_behavior.py`: 采集资金流向、融资融券

**执行**: `python scripts/collect_metadata.py`（会自动创建 `data/raw/{domain}` 目录并保存CSV）

### 测试规范
测试用例在 `tests/` 目录，使用简单的函数式测试（非pytest框架）：
- 示例: [tests/test_metadata_collectors.py](tests/test_metadata_collectors.py)
- 直接运行: `python tests/test_metadata_collectors.py`

## 关键编码模式

### 日期格式统一
- 外部接口参数：`YYYYMMDD` 字符串（如 `'20250115'`）
- 内部处理：使用 `pd.to_datetime()` 转换
- BaoStock需要 `YYYY-MM-DD` 格式，用 `_convert_date_format()` 工具方法处理

### 字段标准化
使用 `_standardize_columns()` 映射不同数据源的列名：

```python
column_mapping = {
    'date': 'trade_date',      # BaoStock -> 标准字段
    'code': 'bs_code',
    'preclose': 'pre_close',
}
df = self._standardize_columns(df, column_mapping)
```

### 缺失字段填充
确保返回DataFrame包含所有 `OUTPUT_FIELDS`：

```python
for col in self.OUTPUT_FIELDS:
    if col not in df.columns:
        df[col] = None
return df[self.OUTPUT_FIELDS]
```

## 项目特定注意事项

1. **不要直接操作原始数据文件** (`data/raw/`): 这些是采集脚本的输出，应通过采集器重新生成
2. **BaoStock代码格式**: 使用 `{exchange}.{symbol}` 格式（如 `sh.600000`），需从 `ts_code` 转换
3. **Docker环境变量**: MongoDB端口映射为27600（非默认27017），避免与本地实例冲突
4. **数据域README**: 每个数据域有独立的 `README.md` 文档采集器用法（参考 [src/data_pipeline/collectors/structured/metadata/README.md](src/data_pipeline/collectors/structured/metadata/README.md)）
5. **空目录占位**: 未实现模块用空 `__init__.py` 占位（如 `feature_engineering/registry.py`）

## 常见任务快速参考

### 添加新采集器
1. 在对应数据域目录创建采集器类（继承 `BaseCollector`）
2. 定义 `OUTPUT_FIELDS` 和 `collect()` 方法
3. 实现 `_collect_from_tushare/akshare/baostock()` 方法
4. 添加便捷函数到模块 `__init__.py`
5. 更新数据域 `README.md` 文档

### 调试采集失败
1. 检查 `.env` 中的 `TUSHARE_TOKEN` 是否有效
2. 查看日志输出的数据源降级信息
3. 对于BaoStock，确认已调用 `ensure_baostock_login()`
4. 验证日期范围参数格式（`YYYYMMDD`）

### 查看数据覆盖情况
检查 `data/raw/{domain}/collection_summary.csv` 文件（由采集脚本生成的汇总记录）

---

**文档维护**: 修改架构时同步更新此文件和各数据域README。如有疑问参考现有实现作为模板。
