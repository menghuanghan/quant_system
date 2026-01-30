# 摘要生成器 (Summarizer)

基于大语言模型(LLM)的生成式摘要工具，将 `content_text` 转换为 50-100 字的 `content` 摘要。

## 特点

- **生成式摘要**: 使用 Qwen2.5-7B-Instruct 本地模型，能理解"利好/利空"的逻辑
- **金融专业Prompt**: 针对公告、研报、政策、新闻等不同类型设计专业Prompt
- **文本预处理**: 自动清洗冗余信息（免责声明、页眉页脚、联系方式等）
- **批量处理**: 支持批量摘要生成和 DataFrame 操作
- **GPU加速**: 通过 Ollama 自动管理 GPU 推理

## 依赖

- Ollama (本地LLM服务)
- Qwen2.5-7B-Instruct 模型

## 安装

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载模型
ollama pull qwen2.5:7b-instruct

# 安装 Python 依赖
pip install ollama
```

## 使用示例

### 基本使用

```python
from src.data_pipeline.processors.unstructured.summarizer import (
    Summarizer,
    DataType
)

# 创建摘要生成器
summarizer = Summarizer()

# 生成公告摘要
result = summarizer.summarize(
    content_text="公司2024年净利润同比增长50%，主要受益于新产品放量...",
    data_type=DataType.ANNOUNCEMENT
)

if result.success:
    print(result.content)  # 50-100字摘要
    print(f"压缩比: {result.compression_ratio:.1f}x")
```

### 批量处理

```python
texts = [
    "公告1内容...",
    "研报内容...",
    "政策内容..."
]

data_types = [
    DataType.ANNOUNCEMENT,
    DataType.REPORT,
    DataType.POLICY
]

batch_result = summarizer.summarize_batch(texts, data_types)

print(f"成功率: {batch_result.success_rate:.2%}")
print(f"吞吐量: {batch_result.throughput:.2f}/秒")
```

### DataFrame 处理

```python
import pandas as pd

df = pd.read_parquet("data.parquet")

# 自动为整个DataFrame生成摘要
df_result = summarizer.summarize_dataframe(
    df,
    text_column='content_text',
    output_column='content',
    type_column='data_type'  # 可选，用于选择合适的Prompt
)

# 结果包含: content, content_success, content_error 列
```

### 与 Content Extractor 配合使用

```python
from src.data_pipeline.processors.unstructured import (
    ContentExtractor,
    Summarizer,
    DataType
)

# 1. 提取PDF/HTML内容
extractor = ContentExtractor()
extract_result = extractor.extract(
    "http://static.cninfo.com.cn/xxx.PDF",
    domain='announcements'
)

# 2. 生成摘要
summarizer = Summarizer()
if extract_result.success:
    summary_result = summarizer.summarize(
        extract_result.content_text,
        DataType.ANNOUNCEMENT
    )
    print(summary_result.content)
```

## 数据类型 (DataType)

支持的数据类型及对应的专业Prompt:

| 类型 | 说明 | 重点关注 |
|------|------|---------|
| `ANNOUNCEMENT` | 通用公告 | 业绩数据、重大事项、风险提示 |
| `ANNOUNCEMENT_FINANCIAL` | 财务公告 | 营收、净利润、同比变化、业绩原因 |
| `ANNOUNCEMENT_MAJOR` | 重大事项 | 并购细节、金额、审批进度 |
| `REPORT` | 研报 | 投资评级、目标价、核心逻辑 |
| `REPORT_COMPANY` | 个股研报 | 评级、估值、EPS预测 |
| `REPORT_INDUSTRY` | 行业研报 | 景气度、驱动因素、重点标的 |
| `POLICY` | 政策 | 核心措施、受益行业、实施时间 |
| `POLICY_INDUSTRY` | 产业政策 | 支持行业、补贴措施、受益领域 |
| `NEWS` | 新闻 | 核心事件、涉及行业、市场影响 |
| `NEWS_MARKET` | 市场新闻 | 指数、板块、原因分析 |
| `EVENT` | 事件 | 事件类型、关键时间点 |

## 配置选项

```python
from src.data_pipeline.processors.unstructured.summarizer import (
    Summarizer,
    SummarizerConfig
)

config = SummarizerConfig(
    # LLM配置
    model_name="qwen2.5:7b-instruct",
    ollama_host="http://localhost:11434",
    
    # 生成参数
    temperature=0.3,        # 较低温度确保输出稳定
    max_tokens=256,
    
    # 摘要长度控制
    min_summary_chars=50,
    max_summary_chars=150,
    target_summary_chars=100,
    
    # 输入预处理
    max_input_chars=8000,   # 超过则智能截断
    
    # 重试配置
    max_retries=3,
    timeout=60.0,
)

summarizer = Summarizer(config=config)
```

## 性能指标

在 RTX 5070 (12GB) 上测试:

| 指标 | 数值 |
|------|------|
| 首次加载(预热) | ~40秒 |
| 单条摘要 | ~700-900ms |
| 批量吞吐量 | ~1.8条/秒 |
| 输入tokens/条 | ~200 |
| 输出tokens/条 | ~50 |
| 压缩比 | 2-5x |

## 模块结构

```
summarizer/
├── __init__.py          # 模块导出
├── base.py              # 基类和数据结构 (DataType, SummaryResult, Config)
├── llm_client.py        # Ollama客户端封装 (LLMClient, AsyncLLMClient)
├── prompts.py           # Prompt模板系统 (PromptTemplates)
├── text_preprocessor.py # 文本预处理 (TextPreprocessor)
├── summarizer.py        # 主接口 (Summarizer, BatchSummaryResult)
└── README.md            # 本文档
```

## API 参考

### Summarizer

```python
class Summarizer:
    def __init__(self, config: SummarizerConfig = None, llm_client: LLMClient = None)
    
    def summarize(
        self,
        content_text: str,
        data_type: DataType = DataType.GENERIC,
        title: str = None,
        preprocess: bool = True
    ) -> SummaryResult
    
    def summarize_batch(
        self,
        texts: List[str],
        data_types: List[DataType] = None,
        titles: List[str] = None,
        max_workers: int = 1,
        progress_callback: Callable = None
    ) -> BatchSummaryResult
    
    def summarize_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'content_text',
        output_column: str = 'content',
        type_column: str = None,
        domain_column: str = None
    ) -> pd.DataFrame
    
    def warm_up(self) -> float  # 预热模型
    def is_available(self) -> bool  # 检查服务可用性
```

### SummaryResult

```python
@dataclass
class SummaryResult:
    success: bool
    content: Optional[str]           # 生成的摘要
    original_char_count: int
    summary_char_count: int
    compression_ratio: float
    process_time_ms: float
    input_tokens: int
    output_tokens: int
    error_message: Optional[str]
```

### BatchSummaryResult

```python
@dataclass
class BatchSummaryResult:
    total: int
    success_count: int
    failed_count: int
    results: List[SummaryResult]
    elapsed_time_seconds: float
    success_rate: float              # 属性
    throughput: float                # 属性 (条/秒)
    total_input_tokens: int
    total_output_tokens: int
    avg_compression_ratio: float
```

## 异步使用

```python
import asyncio
from src.data_pipeline.processors.unstructured.summarizer.summarizer import (
    AsyncSummarizer
)

async def main():
    summarizer = AsyncSummarizer()
    
    # 并发生成多个摘要
    results = await summarizer.summarize_batch(
        texts=["文本1", "文本2", "文本3"],
        max_concurrent=4
    )
    
    print(f"成功率: {results.success_rate:.2%}")

asyncio.run(main())
```

## 故障排除

### Ollama服务未启动

```bash
# 检查服务状态
systemctl status ollama

# 启动服务
sudo systemctl start ollama

# 检查是否可用
curl http://localhost:11434/api/tags
```

### 模型未下载

```bash
# 列出已下载的模型
ollama list

# 下载模型
ollama pull qwen2.5:7b-instruct
```

### GPU未被使用

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Ollama GPU支持
ollama run qwen2.5:7b-instruct "hello"
# 运行时观察 nvidia-smi 中的GPU利用率
```

## 版本历史

- **1.0.0**: 初始版本
  - 支持 Qwen2.5-7B-Instruct 模型
  - 17种数据类型的专业Prompt
  - 文本预处理（去冗余、智能截断）
  - 批量处理和DataFrame支持
  - 同步/异步API
