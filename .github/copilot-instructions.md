# Copilot Instructions - 量化交易系统

## 系统架构概览

这是一个模块化的A股量化交易系统，采用**数据域驱动架构（Data Domain Architecture）**组织代码，包含：

- **数据管道层** (`src/data_pipeline/collectors/structured/`): 按8个数据域采集原始数据
# Copilot Instructions — 量化交易系统（精简版）

**目的**：让 AI 编码代理快速上手仓库，定位关键约定、开发命令与示例文件。

**一目了然的模块**
- 数据采集：src/data_pipeline/collectors/structured/（按“数据域”组织，见 metadata/ market_data/ fundamental/ trading_behavior/ 等）
- 特征工程：src/feature_engineering/（`registry.py`、fusion 子模块）
- 模型：src/models/（ML/DL/RL 分层）
- 回测：src/backtesting/（引擎 + analysis + visualizer）
- 执行/风控/组合：src/execution/, src/risk_management/, src/portfolio/

**关键约定（必须遵守）**
- 采集器继承 `BaseCollector` 并通过 `CollectorRegistry.register()` 注册（参见 [src/data_pipeline/collectors/structured/base.py](src/data_pipeline/collectors/structured/base.py)）。
- 每个采集器定义 `OUTPUT_FIELDS` 并保证返回包含这些列的 `pd.DataFrame`。
- 日期参数外部使用 `YYYYMMDD`（内部用 `pd.to_datetime()`），BaoStock 例外需 `YYYY-MM-DD`（见 `_convert_date_format()`）。
- 数据源优先级：Tushare → AkShare → BaoStock；使用 `@retry_on_failure` + `DataSourceManager` 进行降级和重试处理。

**常用开发命令（Windows 开发者示例）**
- 启用虚拟环境（PowerShell）:
```powershell
& .venv\Scripts\Activate.ps1
```
- 启动依赖服务（TimescaleDB、Mongo、Redis）:
```bash
docker-compose -f docker/docker-compose.yml up -d
```
- 运行示例采集脚本:
```bash
python scripts/collect_metadata.py
```
- 运行 tests 目录的简单测试:
```bash
python tests/test_metadata_collectors.py
```

**重要文件示例（定位问题与扩展时请参考）**
- 采集器基类：src/data_pipeline/collectors/structured/base.py
- 示例采集器：src/data_pipeline/collectors/structured/market_data/price_kline.py
- 脚本入口：scripts/run_full_collection.py 和 scripts/collect_*.py
- 特征注册：src/feature_engineering/registry.py

**集成与运行时注意事项**
- 在仓库根目录放置 `.env`（包含 `TUSHARE_TOKEN`、DB 连接等），缺少令牌时会触发数据源降级。
- 切勿直接修改 `data/raw/`：这些文件由采集器生成，所有变更应通过采集器代码实现。
- 本地 Mongo 映射到容器端口 `27600`（和默认 `27017` 不同），注意避免端口冲突。

**快速任务指南（添加新采集器）**
1. 在对应数据域目录新增类，继承 `BaseCollector`。
2. 指定 `OUTPUT_FIELDS`，实现 `collect()`。
3. 实现或调用 `_collect_from_tushare/_collect_from_akshare/_collect_from_baostock()`。
4. 在模块 `__init__.py` 提供便捷函数（如 `get_stock_daily()`）。
5. 更新该数据域的 `README.md`（目录下已有模板）。

如需扩展：优先阅读上述示例文件以复用已有模式；遇到数据源或格式差异时，搜索 `_standardize_columns`、`_convert_date_format` 相关实现。

请求反馈：如果需要，我可以把这份精简版合并回更详尽的原稿或把某个数据域展开成单独的 AI 指南。
