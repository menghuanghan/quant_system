"""
基于LLM的打分器

使用大语言模型理解文本语义，输出情感分数和理由。
支持JSON格式输出，确保结构化结果。

优点：
- 语义理解深入
- 能处理复杂/模糊的表述
- 提供打分理由
"""

import re
import json
import logging
import time
from typing import Optional, List, Dict, Any

import ollama
from ollama import Client
OLLAMA_AVAILABLE = True

from .base import (
    ScoreResult,
    ScorerConfig,
    ScoringMethod,
    BaseScorer,
    SCORE_GUIDELINES,
)

logger = logging.getLogger(__name__)


class LLMScorer(BaseScorer):
    """
    基于LLM的打分器
    
    使用大语言模型（如Qwen2.5-7B）理解文本语义，
    输出结构化的JSON格式结果，包含分数和理由。
    
    使用示例：
    ```python
    scorer = LLMScorer()
    
    result = scorer.score(
        content="公司净利润同比增长80%，超预期",
        data_type="announcement"
    )
    print(result.score)   # 75
    print(result.reason)  # "业绩大幅超预期，利好股价"
    ```
    """
    
    # 系统提示
    SYSTEM_PROMPT = f"""你是一个专业的金融量化分析师，专门分析文本的市场情感倾向。

你的任务是：
1. 阅读输入的金融文本（公告摘要、研报观点、新闻标题等）
2. 判断其对股价的影响倾向（利好/利空/中性）
3. 给出-100到100的分数
4. 简要说明打分理由

{SCORE_GUIDELINES}

【重要】你必须以JSON格式输出，格式如下：
{{
    "score": <分数，整数，-100到100>,
    "reason": "<简要理由，不超过50字>"
}}

只输出JSON，不要输出其他内容。"""

    # 用户提示模板
    USER_PROMPT_TEMPLATE = """请分析以下{data_type_cn}文本的市场情感倾向，并给出分数和理由。

文本内容：
{content}

请以JSON格式输出（只输出JSON）："""

    # 数据类型中文映射
    DATA_TYPE_CN = {
        'announcement': '公告',
        'report': '研报',
        'policy': '政策',
        'news': '新闻',
        'event': '事件',
        '': '金融',
    }
    
    def __init__(self, config: Optional[ScorerConfig] = None):
        """初始化LLM打分器"""
        super().__init__(config)
        
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama库未安装。请运行: pip install ollama")
        
        self.client = Client(host=self.config.ollama_host)
        self._check_model()
        
        self.logger.info(
            f"LLM打分器初始化完成 - 模型: {self.config.model_name}"
        )
    
    def _check_model(self) -> bool:
        """检查模型是否可用"""
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            
            if self.config.model_name not in model_names:
                base_name = self.config.model_name.split(':')[0]
                matching = [m for m in model_names if m.startswith(base_name)]
                if matching:
                    self.config.model_name = matching[0]
                    return True
                
                self.logger.warning(f"模型 {self.config.model_name} 未找到")
                return False
            return True
        except Exception as e:
            self.logger.error(f"检查模型失败: {e}")
            return False
    
    def score(
        self,
        content: str,
        data_type: str = "",
        **kwargs
    ) -> ScoreResult:
        """
        使用LLM对文本进行打分
        
        Args:
            content: 输入文本（摘要或标题）
            data_type: 数据类型
            **kwargs: 额外参数
            
        Returns:
            ScoreResult: 打分结果
        """
        start_time = time.time()
        
        if not content or not content.strip():
            return ScoreResult.failure(
                content="",
                error_message="输入文本为空",
                error_code="EMPTY_INPUT",
                data_type=data_type
            )
        
        # 限制输入长度
        if len(content) > 1000:
            content = content[:1000] + "..."
        
        # 获取数据类型中文
        data_type_base = data_type.split('/')[-1].split('_')[0] if data_type else ""
        data_type_cn = self.DATA_TYPE_CN.get(data_type_base, "金融")
        
        # 构建消息
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            data_type_cn=data_type_cn,
            content=content
        )
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # 调用LLM
        try:
            response = self._call_llm_with_retry(messages)
            
            # 解析JSON响应
            score_value, reason = self._parse_response(response)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return ScoreResult(
                success=True,
                score=score_value,
                reason=reason,
                content=content,
                data_type=data_type,
                method=ScoringMethod.LLM,
                model_name=self.config.model_name,
                process_time_ms=elapsed_ms,
            )
            
        except Exception as e:
            self.logger.error(f"LLM打分失败: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return ScoreResult.failure(
                content=content,
                error_message=str(e),
                error_code="LLM_ERROR",
                data_type=data_type
            )
    
    def _call_llm_with_retry(self, messages: List[Dict[str, str]]) -> str:
        """带重试的LLM调用"""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat(
                    model=self.config.model_name,
                    messages=messages,
                    options={
                        'temperature': self.config.temperature,
                        'num_predict': self.config.max_tokens,
                    }
                )
                return response.message.content
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"LLM调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"LLM调用失败: {last_error}")
    
    def _parse_response(self, response: str) -> tuple:
        """
        解析LLM响应，提取分数和理由
        
        支持多种格式：
        1. 标准JSON: {"score": 50, "reason": "..."}
        2. 带markdown代码块的JSON
        3. 非标准格式的回退解析
        """
        response = response.strip()
        
        # 尝试提取JSON块
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                score = int(data.get('score', 0))
                reason = str(data.get('reason', ''))
                # 确保分数在范围内
                score = max(-100, min(100, score))
                return score, reason
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        
        # 尝试从代码块中提取
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_block_match:
            try:
                data = json.loads(code_block_match.group(1))
                score = int(data.get('score', 0))
                reason = str(data.get('reason', ''))
                score = max(-100, min(100, score))
                return score, reason
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        
        # 回退：尝试从文本中提取分数
        score_match = re.search(r'["\']?score["\']?\s*[:：]\s*(-?\d+)', response)
        reason_match = re.search(r'["\']?reason["\']?\s*[:：]\s*["\']?([^"\'}\n]+)', response)
        
        if score_match:
            score = int(score_match.group(1))
            score = max(-100, min(100, score))
            reason = reason_match.group(1).strip() if reason_match else "无具体理由"
            return score, reason
        
        # 最后尝试：查找任何数字作为分数
        number_match = re.search(r'(-?\d+)\s*分?', response)
        if number_match:
            score = int(number_match.group(1))
            if -100 <= score <= 100:
                return score, "LLM输出格式异常，仅提取到分数"
        
        # 完全无法解析，返回中性
        self.logger.warning(f"无法解析LLM响应: {response[:200]}")
        return 0, "无法解析LLM响应，默认中性"
    
    def score_batch(
        self,
        contents: List[str],
        data_types: Optional[List[str]] = None,
        **kwargs
    ) -> List[ScoreResult]:
        """
        批量LLM打分
        
        注意：LLM打分是串行的，速度较慢。
        如果需要高速批量打分，请使用RuleScorer。
        
        Args:
            contents: 文本列表
            data_types: 数据类型列表
            **kwargs: 额外参数
            
        Returns:
            List[ScoreResult]: 打分结果列表
        """
        if not contents:
            return []
        
        if data_types is None:
            data_types = [""] * len(contents)
        elif len(data_types) != len(contents):
            data_types = data_types + [""] * (len(contents) - len(data_types))
        
        results = []
        for i, (content, dtype) in enumerate(zip(contents, data_types)):
            result = self.score(content, dtype, **kwargs)
            results.append(result)
            
            # 打印进度
            if (i + 1) % 10 == 0:
                self.logger.info(f"LLM打分进度: {i + 1}/{len(contents)}")
        
        return results
    
    def is_available(self) -> bool:
        """检查LLM服务是否可用"""
        try:
            self.client.list()
            return True
        except Exception:
            return False


class FastLLMScorer(LLMScorer):
    """
    快速LLM打分器
    
    使用更简短的提示词和更低的token限制，
    牺牲一些准确性换取速度。
    """
    
    SYSTEM_PROMPT = """你是金融分析师。判断文本对股价的影响，输出JSON格式：
{"score": <-100到100的整数>, "reason": "<理由，10字内>"}
分数标准：100=极度利好，50=利好，0=中性，-50=利空，-100=极度利空。
只输出JSON。"""

    USER_PROMPT_TEMPLATE = """分析：{content}
JSON输出："""

    def __init__(self, config: Optional[ScorerConfig] = None):
        if config is None:
            config = ScorerConfig()
        # 减少token限制以加速
        config.max_tokens = 100
        config.temperature = 0.05
        super().__init__(config)
