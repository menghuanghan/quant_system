# Scorer - 量化打分器

将摘要/标题文本转换为情感分数(-100到100)，用于量化分析金融文本的利好/利空程度。

## 功能特点

- **双策略打分**：规则匹配（毫秒级）+ LLM语义理解
- **自动策略选择**：根据数据类型自动选择最佳打分方式
- **混合打分**：规则优先，LLM兜底
- **批量处理**：支持DataFrame行级打分
- **GPU加速**：支持cuDF加速批量规则匹配

## 架构

```
scorer/
├── __init__.py       # 模块导出
├── base.py           # 基类、数据结构、枚举
├── rule_scorer.py    # 规则打分器（40+条规则）
├── llm_scorer.py     # LLM打分器（Ollama/Qwen2.5-7B）
├── scorer.py         # 主接口（统一API）
└── README.md         # 本文档
```

## 快速开始

### 安装依赖

```bash
# 基础依赖
pip install pandas requests

# GPU加速（可选）
pip install cudf-cu12
```

### 基本用法

```python
from src.data_pipeline.processors.unstructured.scorer import (
    Scorer, score, ScoringMethod, ScorerConfig
)

# 方式1：便捷函数
result = score("公司净利润同比增长80%")
print(result.score)   # 50
print(result.level)   # ScoreLevel.BULLISH
print(result.reason)  # "业绩增长"

# 方式2：类实例（推荐）
scorer = Scorer()
result = scorer.score("关于公司被立案调查的公告")
print(result.score)   # -90
print(result.level)   # ScoreLevel.EXTREMELY_BEARISH
```

### 指定打分方法

```python
from src.data_pipeline.processors.unstructured.scorer import (
    Scorer, ScorerConfig, ScoringMethod
)

# 规则模式（最快，适合格式化文本）
config = ScorerConfig(default_method=ScoringMethod.RULE)
scorer = Scorer(config=config)
result = scorer.score("2024年度业绩预增公告")

# LLM模式（更智能，适合复杂文本）
config = ScorerConfig(default_method=ScoringMethod.LLM)
scorer = Scorer(config=config)
result = scorer.score("公司发布新战略布局，未来增长可期")

# 混合模式（规则优先，LLM兜底）
config = ScorerConfig(default_method=ScoringMethod.HYBRID)
scorer = Scorer(config=config)
result = scorer.score("公司公告内容...")

# 自动模式（根据数据类型选择）
config = ScorerConfig(default_method=ScoringMethod.AUTO)
scorer = Scorer(config=config)
result = scorer.score(
    content="关于公司中标重大项目的公告",
    data_type="news/exchange"  # 交易所公告 → 规则打分
)
```

### 批量打分

```python
from src.data_pipeline.processors.unstructured.scorer import Scorer

scorer = Scorer()

# 列表批量
contents = [
    "关于公司被立案调查的公告",
    "2024年度业绩预增公告",
    "关于公司中标重大项目的公告",
]
results = scorer.score_batch(contents)

for r in results.results:
    print(f"{r.score:+4d} | {r.level.name} | {r.content[:30]}")
```

### DataFrame 打分

```python
import pandas as pd
from src.data_pipeline.processors.unstructured.scorer import Scorer

# 准备数据
df = pd.DataFrame({
    'title': [
        '关于公司被立案调查的公告',
        '2024年度业绩预增公告',
        '关于召开股东大会的通知',
    ],
    'data_type': ['news/exchange', 'news/exchange', 'news/exchange']
})

# 打分
scorer = Scorer()
df_scored = scorer.score_dataframe(
    df,
    content_column='title',
    score_column='score',
    level_column='score_level',
    reason_column='score_reason',
)

print(df_scored[['title', 'score', 'score_level']])
#                         title  score        score_level
# 0       关于公司被立案调查的公告    -90  extremely_bearish
# 1            2024年度业绩预增公告     50            bullish
# 2       关于召开股东大会的通知      0            neutral
```

## 分数等级

| 分数范围 | 等级 | 含义 |
|---------|------|------|
| 75 ~ 100 | EXTREMELY_BULLISH | 极度利好 |
| 25 ~ 75 | BULLISH | 利好 |
| 5 ~ 25 | SLIGHTLY_BULLISH | 轻微利好 |
| -5 ~ 5 | NEUTRAL | 中性 |
| -25 ~ -5 | SLIGHTLY_BEARISH | 轻微利空 |
| -75 ~ -25 | BEARISH | 利空 |
| -100 ~ -75 | EXTREMELY_BEARISH | 极度利空 |

## 规则打分器

内置 40+ 条金融领域规则，覆盖：

### 极度利空规则（-100 到 -75）
- 立案调查、被侦查
- 退市、终止上市、摘牌
- ST/*ST/退市风险警示
- 财务造假、虚假陈述
- 暂停上市、停止交易

### 利空规则（-75 到 -25）
- 监管函、关注函、问询函
- 警示函、责令改正
- 行政处罚、罚款
- 诉讼、仲裁
- 业绩预亏、业绩下滑
- 股东减持
- 违规行为

### 轻微利空规则（-25 到 -5）
- 终止重组/并购
- 审计意见问题
- 会计差错更正
- 高管辞职

### 中性规则（-5 到 5）
- 董事会决议、股东大会
- 制度修订
- 定期报告（季报/半年报/年报）

### 轻微利好规则（5 到 25）
- 经营稳定
- 股权激励
- 政府补助

### 利好规则（25 到 75）
- 业绩预增
- 股东增持、回购
- 获得资质认证
- 合同订单
- 分红送股
- 研发创新
- 战略合作

### 极度利好规则（75 到 100）
- 业绩大幅预增（50%+）
- 重大合同中标
- 摘帽
- 重大重组完成
- 重大事项获批

## LLM 打分器

使用 Ollama + Qwen2.5-7B-Instruct 进行语义理解打分。

### 配置

```python
from src.data_pipeline.processors.unstructured.scorer import ScorerConfig

config = ScorerConfig(
    model_name="qwen2.5:7b-instruct",    # 模型名称
    ollama_host="http://localhost:11434", # Ollama地址
    temperature=0.1,                      # 温度（低=稳定）
    max_tokens=256,                       # 最大生成长度
    timeout=30.0,                         # 超时时间
    max_retries=3,                        # 重试次数
)
```

### 输出格式

LLM 输出 JSON 格式：
```json
{
    "score": 75,
    "reason": "业绩大幅增长，利好明显"
}
```

## 自动策略选择

`AUTO` 模式会根据数据类型自动选择打分方式：

| 数据类型 | 打分方式 | 原因 |
|---------|---------|------|
| news/exchange | RULE | 交易所公告格式规范 |
| announcements/* | RULE | 公告类格式化程度高 |
| policy/* | LLM | 政策需要语义理解 |
| reports/* | LLM | 研报需要深度分析 |
| news/* (其他) | LLM | 新闻内容多样 |

## API 参考

### ScoreResult

打分结果数据类：

```python
@dataclass
class ScoreResult:
    success: bool           # 是否成功
    content: str            # 输入内容
    score: int              # 分数 (-100 到 100)
    normalized_score: float # 归一化分数 (-1.0 到 1.0)
    level: ScoreLevel       # 分数等级
    reason: str             # 打分理由
    method: ScoringMethod   # 使用的打分方法
    matched_rule: str       # 匹配的规则（规则打分时）
    confidence: float       # 置信度
    data_type: str          # 数据类型
    elapsed_time: float     # 耗时
    timestamp: datetime     # 时间戳
    
    # 属性
    is_bullish: bool        # 是否利好 (score > 5)
    is_bearish: bool        # 是否利空 (score < -5)
    is_neutral: bool        # 是否中性
```

### BatchScoreResult

批量打分结果：

```python
@dataclass  
class BatchScoreResult:
    total: int              # 总数
    success_count: int      # 成功数
    failed_count: int       # 失败数
    results: List[ScoreResult]
    elapsed_time_seconds: float
    
    # 统计
    bullish_count: int      # 利好数量
    bearish_count: int      # 利空数量
    neutral_count: int      # 中性数量
    avg_score: float        # 平均分数
    rule_scored: int        # 规则打分数量
    llm_scored: int         # LLM打分数量
    
    # 方法
    success_rate: float     # 成功率
    throughput: float       # 吞吐量
    to_dataframe()          # 转DataFrame
```

## 性能

| 方法 | 单条耗时 | 批量吞吐 | 适用场景 |
|-----|---------|---------|---------|
| 规则打分 | < 1ms | 10000+/s | 格式化文本 |
| LLM打分 | 0.3-1s | 1-3/s | 复杂文本 |
| 混合打分 | 1ms-1s | 视命中率 | 通用 |

## 示例：完整流程

```python
import pandas as pd
from src.data_pipeline.processors.unstructured.scorer import (
    Scorer, ScorerConfig, ScoringMethod
)

# 1. 配置
config = ScorerConfig(
    default_method=ScoringMethod.AUTO,
    model_name="qwen2.5:7b-instruct",
)

# 2. 初始化
scorer = Scorer(config=config)

# 3. 读取数据
df = pd.read_parquet("data/raw/unstructured/announcements/data.parquet")

# 4. 打分
df_scored = scorer.score_dataframe(
    df,
    content_column='title',
    data_type_column='data_type',
    score_column='sentiment_score',
    level_column='sentiment_level',
)

# 5. 筛选利好/利空
bullish = df_scored[df_scored['sentiment_score'] > 25]
bearish = df_scored[df_scored['sentiment_score'] < -25]

print(f"利好公告: {len(bullish)}")
print(f"利空公告: {len(bearish)}")

# 6. 保存
df_scored.to_parquet("data/processed/announcements_scored.parquet")
```

## 扩展规则

添加自定义规则：

```python
from src.data_pipeline.processors.unstructured.scorer.rule_scorer import (
    RuleScorer, ScoringRule
)

# 创建自定义规则
custom_rules = [
    ScoringRule(
        pattern=r"新能源|碳中和",
        score=30,
        category="热点题材",
        description="新能源相关",
        priority=50
    ),
]

# 扩展规则打分器
scorer = RuleScorer()
scorer.add_rules(custom_rules)

# 测试
result = scorer.score("公司新能源项目获批")
print(result.score)  # 30
```

## 依赖

- Python 3.10+
- pandas >= 1.5.0
- requests >= 2.28.0
- Ollama (LLM打分器需要)
- cudf (GPU加速，可选)

## 测试

```bash
# 运行测试
python tests/test_scorer.py
```
