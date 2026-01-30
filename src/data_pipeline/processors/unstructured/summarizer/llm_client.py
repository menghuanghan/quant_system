"""
LLM客户端模块

封装Ollama API调用，提供：
- 同步/异步接口
- 自动重试机制
- GPU推理（由Ollama自动管理）
- 流式输出支持
"""

import logging
import time
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass

try:
    import ollama
    from ollama import Client, AsyncClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from .base import SummarizerConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM响应数据结构"""
    content: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_duration_ns: Optional[int] = None   # Ollama返回的总耗时（纳秒）
    load_duration_ns: Optional[int] = None    # 模型加载耗时
    eval_duration_ns: Optional[int] = None    # 推理耗时
    
    @property
    def eval_duration_ms(self) -> float:
        """推理耗时（毫秒）"""
        if self.eval_duration_ns:
            return self.eval_duration_ns / 1_000_000
        return 0.0
    
    @property
    def tokens_per_second(self) -> float:
        """推理速度（tokens/秒）"""
        if self.output_tokens and self.eval_duration_ns and self.eval_duration_ns > 0:
            return self.output_tokens / (self.eval_duration_ns / 1_000_000_000)
        return 0.0


class LLMClient:
    """
    LLM客户端 - Ollama封装
    
    提供与本地部署的LLM（如Qwen2.5-7B-Instruct）交互的接口。
    
    使用示例：
    ```python
    client = LLMClient()
    
    # 简单生成
    response = client.generate("请总结这段文字：...")
    print(response.content)
    
    # 带系统提示
    response = client.chat(
        messages=[
            {"role": "system", "content": "你是一个金融分析师"},
            {"role": "user", "content": "请分析这份公告..."}
        ]
    )
    ```
    """
    
    def __init__(self, config: Optional[SummarizerConfig] = None):
        """
        初始化LLM客户端
        
        Args:
            config: 配置对象
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama库未安装。请运行: pip install ollama"
            )
        
        self.config = config or SummarizerConfig()
        self.client = Client(host=self.config.ollama_host)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 检查模型是否可用
        self._check_model()
    
    def _check_model(self) -> bool:
        """检查模型是否已下载"""
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            
            if self.config.model_name not in model_names:
                # 尝试匹配不带tag的名称
                base_name = self.config.model_name.split(':')[0]
                matching = [m for m in model_names if m.startswith(base_name)]
                
                if matching:
                    self.logger.info(f"使用匹配的模型: {matching[0]}")
                    self.config.model_name = matching[0]
                    return True
                
                self.logger.warning(
                    f"模型 {self.config.model_name} 未找到。"
                    f"可用模型: {model_names}"
                )
                return False
            return True
        except Exception as e:
            self.logger.error(f"检查模型失败: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        生成文本
        
        Args:
            prompt: 用户提示
            system: 系统提示
            **kwargs: 额外参数（覆盖默认配置）
            
        Returns:
            LLMResponse: 生成结果
        """
        options = self.config.to_ollama_options()
        options.update(kwargs)
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        return self._call_with_retry(messages, options)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """
        对话式生成
        
        Args:
            messages: 消息列表，格式为[{"role": "user/system/assistant", "content": "..."}]
            **kwargs: 额外参数
            
        Returns:
            LLMResponse: 生成结果
        """
        options = self.config.to_ollama_options()
        options.update(kwargs)
        
        return self._call_with_retry(messages, options)
    
    def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        options: Dict[str, Any]
    ) -> LLMResponse:
        """带重试的API调用"""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat(
                    model=self.config.model_name,
                    messages=messages,
                    options=options,
                )
                
                # 解析响应
                return LLMResponse(
                    content=response.message.content,
                    model=response.model,
                    input_tokens=response.prompt_eval_count,
                    output_tokens=response.eval_count,
                    total_duration_ns=response.total_duration,
                    load_duration_ns=response.load_duration,
                    eval_duration_ns=response.eval_duration,
                )
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"LLM调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}"
                )
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        
        # 所有重试都失败
        raise RuntimeError(f"LLM调用失败，已重试{self.config.max_retries}次: {last_error}")
    
    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式生成文本
        
        Args:
            prompt: 用户提示
            system: 系统提示
            **kwargs: 额外参数
            
        Yields:
            str: 生成的文本片段
        """
        options = self.config.to_ollama_options()
        options.update(kwargs)
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            stream = self.client.chat(
                model=self.config.model_name,
                messages=messages,
                options=options,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.message.content:
                    yield chunk.message.content
                    
        except Exception as e:
            self.logger.error(f"流式生成失败: {e}")
            raise
    
    def is_available(self) -> bool:
        """检查服务是否可用"""
        try:
            self.client.list()
            return True
        except Exception:
            return False
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """获取当前模型信息"""
        try:
            info = self.client.show(self.config.model_name)
            # 兼容不同版本的ollama响应结构
            model_name = getattr(info, 'model', None) or getattr(info, 'modelfile', None) or self.config.model_name
            details = getattr(info, 'details', None) or {}
            
            if hasattr(details, 'family'):
                # 新版本ollama
                return {
                    'model': model_name,
                    'family': details.family if details else None,
                    'parameter_size': details.parameter_size if details else None,
                    'quantization': details.quantization_level if details else None,
                }
            else:
                # 旧版本或字典格式
                return {
                    'model': model_name,
                    'family': details.get('family') if isinstance(details, dict) else None,
                    'parameter_size': details.get('parameter_size') if isinstance(details, dict) else None,
                    'quantization': details.get('quantization_level') if isinstance(details, dict) else None,
                }
        except Exception as e:
            self.logger.error(f"获取模型信息失败: {e}")
            return None
    
    def warm_up(self) -> float:
        """
        预热模型（首次加载可能较慢）
        
        Returns:
            float: 预热耗时（毫秒）
        """
        start = time.time()
        try:
            self.generate("你好", temperature=0)
            elapsed = (time.time() - start) * 1000
            self.logger.info(f"模型预热完成，耗时: {elapsed:.0f}ms")
            return elapsed
        except Exception as e:
            self.logger.error(f"模型预热失败: {e}")
            raise


class AsyncLLMClient:
    """
    异步LLM客户端
    
    用于批量处理时提高并发效率。
    
    使用示例：
    ```python
    import asyncio
    
    async def main():
        client = AsyncLLMClient()
        
        tasks = [
            client.generate("总结文本1"),
            client.generate("总结文本2"),
        ]
        results = await asyncio.gather(*tasks)
    
    asyncio.run(main())
    ```
    """
    
    def __init__(self, config: Optional[SummarizerConfig] = None):
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama库未安装")
        
        self.config = config or SummarizerConfig()
        self.client = AsyncClient(host=self.config.ollama_host)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """异步生成文本"""
        options = self.config.to_ollama_options()
        options.update(kwargs)
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        return await self._call_with_retry(messages, options)
    
    async def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        options: Dict[str, Any]
    ) -> LLMResponse:
        """带重试的异步API调用"""
        import asyncio
        
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat(
                    model=self.config.model_name,
                    messages=messages,
                    options=options,
                )
                
                return LLMResponse(
                    content=response.message.content,
                    model=response.model,
                    input_tokens=response.prompt_eval_count,
                    output_tokens=response.eval_count,
                    total_duration_ns=response.total_duration,
                    load_duration_ns=response.load_duration,
                    eval_duration_ns=response.eval_duration,
                )
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"异步LLM调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}"
                )
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"异步LLM调用失败: {last_error}")
