from typing import Dict, List, Optional, Union, Any, Tuple
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum, auto
import json
import time
from openai import OpenAI
from openai.types.chat import ChatCompletion
import os
import yaml
import asyncio
import copy
import re
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import string
from collections import Counter

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        # Fallback to punkt if punkt_tab is not available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
from .utils import (
    normalize_answer,
    exact_match_score,
    f1_score,
    get_bert_score,
    model_prompting
)
from .mbpp_utils import pass_at_1, extract_code_blocks

def extract_answer_letter(text):
    """
    尝试从文本中提取所有可能是答案的单个大写或小写字母。
    忽略括号、多余的文字等。
    """
    # 匹配形如 (D)、D、Dor(D)、A or B or (C) 等
    matches = re.findall(r'\(?([A-Z])\)?', text, re.IGNORECASE)
    return [m.upper() for m in matches]  # 标准化为大写


def mc_match_score(prediction, ground_truth):
    """
    比较 prediction 和 ground_truth 是否匹配。
    prediction: 多样的字符串 (可能含多个选项、带括号或不带括号)
    ground_truth: 形如 (D)
    返回 float: 1.0 表示匹配成功，0.0 表示不匹配
    """
    # 提取 ground_truth 的字母
    gt_match = re.match(r'\(([A-Z])\)', ground_truth.strip(), re.IGNORECASE)
    if not gt_match:
        return 0.0
    gt_letter = gt_match.group(1).upper()

    # 提取 prediction 中所有可能的选项字母
    pred_letters = extract_answer_letter(prediction)

    # 判断是否包含正确选项
    return 1.0 if gt_letter in pred_letters else 0.0
    
class ProviderType(Enum):
    """支持的LLM提供商类型"""
    OPENAI = auto()
    ALIPAY = auto()
    HUANQIU = auto()  # 添加环球供应商
    ZHIZENGZENG = auto()  # 智增增供应商
    OPENROUTER = auto()  # 环球供应商

@dataclass
class GenerationConfig:
    """生成参数配置"""
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = False

@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    usage: Dict[str, int]
    model: str
    provider: str
    finish_reason: Optional[str] = None
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
        
        # 初始化客户端和配置
        self.client, self.model_config, self.extra_body = self._init_client()
        
    def _init_client(self) -> Tuple[Any, Dict, Dict]:
        """
        初始化对应提供商的客户端和配置
        Returns:
            Tuple[client, model_config, extra_body]
        """

        if self.provider == ProviderType.OPENAI:
            client = OpenAI(
                base_url=self.config.get('base_url'),
                api_key=self.config['api_key']
            )
            return client, {}, {}
            
        elif self.provider == ProviderType.ALIPAY:
            # 处理thinking模式
            is_thinking_mode = "thinking" in self.model
            org_model_name = self.model
            self.model = self.model.replace("-thinking", "")
            
            model_config = copy.deepcopy(self.config['providers']['alipay']['models']['available'][org_model_name])
            
            
            # 设置thinking模式的参数
            if is_thinking_mode:
                extra_body = {
                    "enable_thinking": model_config['parameters'].get('enable_thinking', False),
                    "thinking_budget": model_config['parameters'].get('thinking_budget', 30000)
                }
                # thinking模式必须启用stream
                model_config['parameters']['stream'] = True
            else:
                extra_body = {"enable_thinking": False}
            
            # 初始化客户端
            client = OpenAI(
                base_url=self.config['providers']['alipay']['api']['base_url'],
                api_key=self.config['providers']['alipay']['api']['key']
            )
            
            return client, model_config, extra_body
            
        elif self.provider == ProviderType.HUANQIU:
            # 环球API客户端初始化 - 使用OpenAI兼容接口
            client = OpenAI(
                base_url=self.config['base_url'],
                api_key=self.config['api_key']
            )
            return client, {}, {}
            
        elif self.provider == ProviderType.ZHIZENGZENG:
            # 智增增API客户端初始化 - 使用OpenAI兼容接口
            client = OpenAI(
                base_url=self.config['base_url'],
                api_key=self.config['api_key']
            )
            return client, {}, {}
        elif self.provider == ProviderType.OPENROUTER:
            client = OpenAI(
                base_url=self.config['openrouter']['api']['base_url'],
                api_key=self.config['openrouter']['api']['key']
            )
            model_config = copy.deepcopy(self.config['openrouter']['models']['available'][self.model])
            model_config['parameters']['stream'] = False
            return client, model_config, {}

     
            
    def get_generation_config(self) -> GenerationConfig:
        """获取生成配置"""
        params = self.model_config.get('parameters', {})
        return GenerationConfig(
            temperature=params.get('temperature', 0.7),
            max_tokens=params.get('max_output_length', params.get('max_tokens', 2000)),
            stream=params.get('stream', False)
        )
        
    def get_extra_body(self) -> Dict[str, Any]:
        """获取额外参数"""
        return self.extra_body.copy()
        
    async def generate(self,
                      messages: List[Dict[str, str]],
                      config: Optional[GenerationConfig] = None,
                      max_retries: int = 3,
                      extra_body: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """
        生成文本
        
        Args:
            messages: 消息列表
            config: 生成参数配置(可选,默认使用模型配置)
            max_retries: 最大重试次数
            extra_body: 额外参数(可选,默认使用模型配置)
            
        Returns:
            生成结果
        """
        # 使用传入的配置或默认配置
        config = config or self.get_generation_config()
        extra_body = extra_body or self.get_extra_body()
        
        for attempt in range(max_retries):
            try:
                if self.provider == ProviderType.OPENAI:
                    return await self._generate_openai(messages, config)
                elif self.provider == ProviderType.ALIPAY:
                    return await self._generate_alipay(messages, config, extra_body)
                elif self.provider == ProviderType.HUANQIU:
                    return await self._generate_huanqiu(messages, config)
                elif self.provider == ProviderType.ZHIZENGZENG:
                    return await self._generate_zhizengzeng(messages, config)
                elif self.provider == ProviderType.OPENROUTER:
                    return await self._generate_openai(messages, config)
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1
                    self.logger.info(f"Waiting {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
        
        raise Exception("No provider matched")
        
    async def _generate_openai(self,
                             prompt: str,
                             config: GenerationConfig,       system_prompt: Optional[str]=None) -> GenerationResult:
        """OpenAI API生成实现"""
        # messages = []
        # if system_prompt:
        #     messages.append({"role": "system", "content": system_prompt})
        # messages.append( prompt)
        # print(messages)
        completion = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=prompt,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop
        )
        
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
        
    async def _generate_alipay(self,
                             messages: List[Dict[str, str]],
                             config: GenerationConfig, extra_body: Optional[Dict[str, Any]] = None, verbose: bool = False) -> GenerationResult:
        """蚂蚁通义千问API生成实现"""
        # 检查是否需要流式输出（思考模式必须使用流式输出）
        is_thinking = extra_body and extra_body.get('enable_thinking', False)
        stream_enabled = config.stream
        print(f"流式输出: {stream_enabled}")

        # 准备API调用参数
        if is_thinking:
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "stop": config.stop,
                "stream": True,  # 思考模式必须启用流式输出
                "extra_body": extra_body,
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
                "extra_body": extra_body,
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
            elif metric == 'mmluredux':
                pred_match = re.search(r'<answer>(.*?)</answer>', prediction)
                prediction = pred_match.group(1).strip() if pred_match else prediction 
                result = mc_match_score(prediction, ground_truth)
                return float(result)
                 
            # GSM8K数学问题评估
            elif metric == 'GSM8K':
                # 从ground truth中提取最终答案
                ground_truth = ground_truth.split("####")[-1].strip()
                
                # 查找预测中的答案（<X>格式）
                match = re.search(r'\<(\d+)\>', prediction)
                if match and match.group(1) == ground_truth:
                    return 1.0
                return 0.0
            elif metric == 'bleu':
                # Tokenize prediction and ground truth
                pred_tokens = word_tokenize(prediction.lower())
                ref_tokens = word_tokenize(ground_truth.lower())
                
                # Use smoothing function to handle cases where n-gram matches are zero
                smoothing = SmoothingFunction().method1
                result = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
                return float(result)
            elif metric == 'pass@1':
                # Extract code from prediction using multiple methods
                code = None
                
                # Method 1: Try to extract from <code> tags first
                code_match = re.search(r'<code>(.*?)</code>', prediction, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                
                # Method 2: If no <code> tags found, try markdown code blocks
                if not code:
                    code = extract_code_blocks(prediction)
                
                # Method 3: If no code block found, use the prediction as-is
                if not code:
                    code = prediction.strip()
                
                # Process ground_truth (test cases)
                # ground_truth can be a single test case or multiple test cases separated by newlines
                test_cases = ground_truth.strip()
                
                try:
                    # Use pass_at_1 function from mbpp_utils
                    # For single sample, pass@1 is 1.0 if code passes all tests, 0.0 otherwise
                    # pass_at_1 expects: references (str or list[str]), predictions (str or list[list[str]])
                    # Convert single code string to list format expected by pass_at_1
                    result = pass_at_1(references=test_cases, predictions=[code])
                    return float(result)
                except Exception as e:
                    print(f"Error evaluating pass@1: {str(e)}")
                    return 0.0
            else:
                print(f"不支持的评估指标: {metric}")
                return 0.0
                
        except Exception as e:
            print(f"评估过程出错: {str(e)}")
            return 0.0

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        计算API调用成本
        
        Args:
            input_tokens: 输入token数量
            output_tokens: 输出token数量
            
        Returns:
            float: 调用成本（美元）
        """
        try:
            # 从配置中获取价格信息
            model_pricing = self.config['providers'][self.provider.name.lower()]['models']['available'][self.model]['pricing']
            input_price = model_pricing['input_price']  # 每1k tokens的输入价格
            output_price = model_pricing['output_price']  # 每1k tokens的输出价格
            
            # 计算成本 (转换为每k tokens)
            input_cost = (input_tokens / 1000.0) * input_price
            output_cost = (output_tokens / 1000.0) * output_price
            
            total_cost = input_cost + output_cost
            return total_cost
            
        except KeyError as e:
            print(f"无法获取模型价格信息: {str(e)}")
            return 0.0
        except Exception as e:
            print(f"计算成本时出错: {str(e)}")
            return 0.0

if __name__ == "__main__":
    import logging
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    async def test_model(model_name: str, messages: List[Dict[str, str]], provider_config: Dict):
        """测试单个模型"""
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"测试模型: {model_name}")
            
            # 创建provider实例
            provider = LLMProvider(
                provider="ALIPAY",
                model=model_name,
                config=provider_config,
                logger=logger
            )
            
            # 获取配置信息
            gen_config = provider.get_generation_config()
            extra_body = provider.get_extra_body()
            
            # 打印配置信息
            logger.info("\n配置信息:")
            logger.info(f"- Temperature: {gen_config.temperature}")
            logger.info(f"- Max Tokens: {gen_config.max_tokens}")
            logger.info(f"- Stream: {gen_config.stream}")
            logger.info(f"- Thinking Mode: {extra_body.get('enable_thinking', False)}")
            if extra_body.get('enable_thinking'):
                logger.info(f"- Thinking Budget: {extra_body.get('thinking_budget')}")
            
            # 生成响应
            logger.info("\n生成响应:")
            result = await provider.generate(
                messages=messages,
                config=gen_config,
                extra_body=extra_body
            )
            
            # 打印结果
            logger.info(f"\n响应内容: {result.text}")
            logger.info(f"\nToken统计:")
            logger.info(f"- 输入tokens: {result.usage.get('prompt_tokens', 0)}")
            logger.info(f"- 输出tokens: {result.usage.get('completion_tokens', 0)}")
            logger.info(f"- 总tokens: {result.usage.get('total_tokens', 0)}")
            logger.info(f"完成原因: {result.finish_reason}")
            
            if result.reasoning_content:
                logger.info(f"\n推理过程:\n{result.reasoning_content}")
            
            logger.info(f"{'='*50}\n")
            
        except Exception as e:
            logger.error(f"测试失败: {str(e)}")
    
    async def main():
        # 读取配置文件
        with open('configs/models.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        
        # 准备测试消息
        test_cases = [
            {
                "name": "简单问答",
                "messages": [
                    {"role": "system", "content": "你是一个专业的AI助手，请用简洁专业的方式回答问题。"},
                    {"role": "user", "content": "请解释一下什么是量子纠缠。"}
                ]
            },
        ]
        
        # 测试不同模型
        models_to_test = [
            "qwen3-32b",           # 标准模式
            "qwen3-32b-thinking",  # 思考模式
        ]
        
        # 运行测试
        for test_case in test_cases:
            logger.info(f"\n开始测试用例: {test_case['name']}")
            for model in models_to_test:
                await test_model(model, test_case['messages'], config)
    
    # 运行测试
    asyncio.run(main())

    