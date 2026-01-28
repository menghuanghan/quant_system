# Copilot Instructions — 量化交易系统（AI 编码指南）

目标：让 AI 编码代理能快速上手本仓库的关键模式、运行方式与扩展点。

核心概览
- 数据域驱动：采集器在 `src/data_pipeline/collectors/structured/*`（如 `market_data/`,`fundamental/` 等），非结构化在 `src/data_pipeline/collectors/unstructured`。
- 统一调度：主入口 `scripts/run_full_collection.py`（基于 `FullCollectionScheduler`，调度实现见 `src/data_pipeline/scheduler/structured/full/collection/scheduler.py`）。
- 输出与断点：所有原始产物写入 `data/raw/structured/{domain}/`（parquet，压缩 snappy）；断点/元信息在 `data/checkpoints` 与 `data/state`。

关键约定（必须遵守）
- 采集器类继承 `BaseCollector`（`src/data_pipeline/collectors/structured/base.py`），并通过 `@CollectorRegistry.register("domain/name")` 注册。
- 每个采集器应暴露可被调度器调用的函数（例如 `get_stock_daily`），并声明 `OUTPUT_FIELDS` 或通过 `_ensure_columns()` 填充缺失列。
- 输入/输出日期统一：外部参数使用 `YYYYMMDD`；内部或调用 Baostock 时可用 `_convert_date_format()` 转为 `YYYY-MM-DD`。
- 数据源顺序：默认优先级 `TUSHARE -> AKSHARE -> BAOSTOCK`（`DataSourcePriority.DEFAULT_ORDER`）；使用 `retry_on_failure` 与 `fallback_on_error` 装饰器实现稳健调用。

运行与调度要点
- 启动依赖服务：
  docker-compose -f docker/docker-compose.yml up -d
- 列任务/运行示例：
  python scripts/run_full_collection.py --list-domains
  python scripts/run_full_collection.py --list-tasks
  python scripts/run_full_collection.py --start-date 20210101 --end-date 20251231 --domains market_data --skip-existing
- 并发与限速：调度器默认序列执行（`max_workers` 建议 1）以避免 API 限流；任务内部有 sleep（约 0.3s）与重试逻辑。

调试与数据格式细节
- 采集器返回的 DataFrame 若包含日期字段，会被调度器按 `start_date/end_date` 过滤（通常 `trade_date` 或任务 `date_field`）。
- 输出文件模板可包含 `{ts_code}` 或其它占位符，调度器会替换并按实体拆分保存（见 scheduler 的 `split_by` 逻辑）。
- 采集报告保存在 `data/raw/structured/collection_report.parquet` 与 CSV，日志写入 `logs/full_collection.log`。

常见修改场景（快速上手）
- 添加采集器：在对应域目录新增文件，继承 `BaseCollector` 或导出函数并用 `CollectorRegistry.register` 注册；更新对应 README：`src/data_pipeline/collectors/structured/<domain>/README.md`。
- 增加任务：编辑 `src/data_pipeline/scheduler/structured/full/collection/config.py`（修改 `TASKS_BY_DOMAIN` / `CollectionTask`），确保 `collector_func` 名称与采集器一致。

重要文件参考
- 采集器基类：src/data_pipeline/collectors/structured/base.py
- 调度器实现：src/data_pipeline/scheduler/structured/full/collection/scheduler.py
- 任务配置：src/data_pipeline/scheduler/structured/full/collection/config.py
- 入口脚本：scripts/run_full_collection.py
- 各域说明：src/data_pipeline/collectors/structured/*/README.md

反馈
请指出需补充的具体示例（采集器模板、config 片段、或某域详细流程），我会继续迭代该文件。
