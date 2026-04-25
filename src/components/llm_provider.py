from typing import Dict, List, Optional, Union, Any
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum, auto
import json
import time
import re
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from data_processing.utils import exact_match_score, f1_score, get_bert_score
"""
Based on the systheis q-a pair, estimate the cost, latency, and other performance metrics by different testing models.
"""
class ProviderType(Enum):
    """支持的LLM提供商类型"""
    OPENAI = auto()
    ALIPAY = auto()

@dataclass
class GenerationConfig:
    """生成参数配置"""
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    extra_body: Optional[Dict[str, Any]] = None

@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    usage: Dict[str, int]
    model: str
    provider: str
    finish_reason: Optional[str] = None
    parsed_output: Optional[Any] = None
    reasoning_content: Optional[str] = None

class LLMProvider:
    """统一的LLM调用接口"""
    
    def __init__(self,
                 provider: Union[str, ProviderType],
                 model: str,
                 config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        """
        初始化LLM提供商
        
        Args:
            provider: 提供商名称或类型
            model: 模型名称
            config: 配置信息(包含API密钥等)
            logger: 可选的日志记录器
        """
        if isinstance(provider, str):
            try:
                self.provider = ProviderType[provider.upper()]
            except KeyError:
                raise ValueError(f"Unsupported provider: {provider}")
        else:
            self.provider = provider
            
        self.model = model
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化客户端
        self.client = self._init_client()
        
    def _init_client(self) -> Any:
        """初始化对应提供商的客户端"""
        try:
            if self.provider == ProviderType.OPENAI:
                return OpenAI(
                    base_url=self.config.get('base_url'),
                    api_key=self.config['api_key']
                )
            elif self.provider == ProviderType.ALIPAY:
                # 蚂蚁通义千问API客户端初始化
                return OpenAI(
                    base_url=self.config['base_url'],
                    api_key=self.config['api_key']
                )
        except Exception as e:
            self.logger.error(f"Failed to initialize client for {self.provider}: {str(e)}")
            raise
            
    async def generate(self,
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      config: Optional[Union[Dict[str, Any], GenerationConfig]] = None,
                      max_retries: int = 3) -> GenerationResult:
        """
        生成文本
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            config: 生成参数配置
            max_retries: 最大重试次数
            
        Returns:
            生成结果
        """
        
        for attempt in range(max_retries):
            try:
                if self.provider == ProviderType.OPENAI:
                    return await self._generate_openai(prompt, system_prompt, config)

                elif self.provider == ProviderType.ALIPAY:
                    if isinstance(config, dict):
                        gen_config = GenerationConfig(**config)
                    else:
                        gen_config = config
                    return await self._generate_alipay(prompt, system_prompt, gen_config)

                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1
                    self.logger.info(f"Waiting {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
        
        # 如果所有重试都失败了，抛出一个默认的异常
        raise Exception("All retry attempts failed")
                    
    async def _generate_openai(self,
                             prompt: str,
                             system_prompt: Optional[str],
                             config: Dict[str, Any],
                           ) -> GenerationResult:
        """OpenAI API生成实现"""
        
        # 准备消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # 使用chat.completions API
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                **config
            )
            # print(self.model)
            response_text = completion.choices[0].message.content
            # print(f"Response text: {completion.choices[0]}")
            return GenerationResult(
                text=response_text,
                usage={
                    'prompt_tokens': completion.usage.prompt_tokens,
                    'completion_tokens': completion.usage.completion_tokens,
                    'total_tokens': completion.usage.total_tokens
                },
                model=self.model,
                provider=self.provider.name,
                finish_reason=completion.choices[0].finish_reason
            )
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise
        
    async def _generate_alipay(self,
                             prompt: str,
                             system_prompt: Optional[str],
                             config: GenerationConfig, verbose: bool = False) -> GenerationResult:
        """蚂蚁通义千问API生成实现"""
        # 准备消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 检查是否需要流式输出（思考模式必须使用流式输出）
        extra_body = config.extra_body
        is_thinking = extra_body and extra_body.get('enable_thinking', False)
        stream_enabled = config.stream

        # 准备API调用参数
        if is_thinking:
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "stop": config.stop,
                "stream": True,  # 思考模式必须启用流式输出
                "extra_body": config.extra_body,
                "stream_options": {"include_usage": True}
            }
        else:
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "stop": config.stop,
                "stream": stream_enabled,
                "extra_body": config.extra_body,
            }
              
        try:
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                **api_params,
            )
            
            if stream_enabled or is_thinking:
                # 处理流式输出
                full_text = ""
                reasoning_text = ""
                final_usage = None
                is_answering = False
                
                if is_thinking and verbose:
                    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
                for chunk in completion:
                    # 如果chunk.choices为空，则获取usage信息
                    if not chunk.choices:
                        if hasattr(chunk, 'usage'):
                            final_usage = chunk.usage.model_dump()
                        continue

                    delta = chunk.choices[0].delta
                    
                    # 处理思考过程
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                        if verbose:
                            print(delta.reasoning_content, end='', flush=True)
                        reasoning_text += delta.reasoning_content
                    else:
                        # 处理回答内容
                        if hasattr(delta, 'content'):
                            if delta.content and not is_answering and is_thinking:
                                if verbose:
                                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                                is_answering = True
                            if delta.content:
                                if verbose: 
                                    print(delta.content, end='', flush=True)
                                full_text += delta.content
                return GenerationResult(
                    text=full_text,
                    usage=final_usage or {},
                    model=self.model,
                    provider=self.provider.name,
                    finish_reason= None,
                    reasoning_content=reasoning_text if reasoning_text else None
                )
            else:
                # 非流式输出处理
                return GenerationResult(
                    text=completion.choices[0].message.content,
                    usage={
                        'prompt_tokens': completion.usage.prompt_tokens,
                        'completion_tokens': completion.usage.completion_tokens,
                        'total_tokens': completion.usage.total_tokens
                    },
                    model=self.model,
                    provider=self.provider.name,
                    finish_reason=completion.choices[0].finish_reason
                )
                
        except Exception as e:
            self.logger.error(f"Error in _generate_alipay: {str(e)}")
            raise

    def eval(self, prediction: str, ground_truth: str, metric: str = "em") -> float:
        """
        评估模型预测结果
        
        Args:
            prediction: 模型预测的文本
            ground_truth: 标准答案文本
            metric: 评估指标，支持 'em'(精确匹配), 'f1_score'(F1分数), 
                   'bert_score'(BERT语义相似度), 'em_mc'(多选题精确匹配)
                   
        Returns:
            float: 评估分数 (0-1之间)
        """
        try:
            # 精确匹配评估
            if metric == 'em':
                result = exact_match_score(prediction, ground_truth)
                return float(result)
            
            # 多选题精确匹配
            elif metric == 'em_mc':
                result = exact_match_score(prediction, ground_truth, normal_method="mc")
                return float(result)
            
            # BERT语义相似度分数
            elif metric == 'bert_score':
                result = get_bert_score([prediction], [ground_truth])
                return result
            
            # F1分数评估
            elif metric == 'f1_score':
                # 从预测和ground truth中提取<answer>标签中的内容
                pred_match = re.search(r'<answer>(.*?)</answer>', prediction)
                prediction = pred_match.group(1).strip() if pred_match else prediction                
                # 计算F1分数
                f1, _, _ = f1_score(prediction, ground_truth)
                return f1
            
            # GSM8K数学问题评估
            elif metric == 'GSM8K':
                # 从ground truth中提取最终答案
                ground_truth = ground_truth.split("####")[-1].strip()
                
                # 查找预测中的答案（<X>格式）
                match = re.search(r'\<(\d+)\>', prediction)
                if match and match.group(1) == ground_truth:
                    return 1.0
                return 0.0
            
            else:
                print(f"不支持的评估指标: {metric}")
                return 0.0
                
        except Exception as e:
            print(f"评估过程出错: {str(e)}")
            return 0.0

    def calculate_cost(self, input_tokens: int, output_tokens: int, input_price: float, output_price: float) -> float:
        """
        计算API调用成本
        
        Args:
            input_tokens: 输入token数量
            output_tokens: 输出token数量
            
        Returns:
            float: 调用成本（美元）
        """
    
        # 计算成本 (转换为每k tokens)
        input_cost = (input_tokens / 1000.0) * input_price
        output_cost = (output_tokens / 1000.0) * output_price
        
        total_cost = input_cost + output_cost
        return total_cost
            

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # 初始化配置
        config = {
            "api_key": None,
            "base_url": "https://api.example.com/v1"  # 对于非OpenAI的提供商
        }

        # 创建provider实例
        provider = LLMProvider(
            provider="OPENAI",  # 或 ProviderType.OPENAI
            model="gpt-3.5-turbo",
            config=config
        )

        # 生成配置
        gen_config = {
            "temperature": 0.7,
            "max_tokens": 1000
        }

        try:
            # 生成文本
            result = await provider.generate(
                prompt="What is the capital of France?",
                system_prompt="You are a helpful assistant.",
                config=gen_config
            )

            print(f"Response: {result.text}")
            print(f"Token usage: {result.usage}")
        except Exception as e:
            print(f"Error: {str(e)}")

    # 运行异步主函数
    asyncio.run(main())