"""
Metric Estimator for evaluating different Qwen models on various task types
"""
from typing import List, Dict, Tuple, Optional, Any
import asyncio
import json
import time
import pandas as pd
import yaml
import os
from dataclasses import dataclass
from src.components.llm_provider import LLMProvider, ProviderType, GenerationConfig, GenerationResult
import re


class MetricEstimator:
    """指标评估器"""
    
    def __init__(self, config_path: str = "configs/models.yaml"):
        """
        初始化评估器
        
        Args:
            config_path: 模型配置文件路径
        """
        self.config_path = config_path
        self.models_config = self._load_config()
        self.judge_provider = self._create_judge_provider()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载模型配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            exit()
    
    def _load_qa_data(self, data_path: str) -> Dict[str, Any]:
        """加载问答数据"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 假设数据结构是字典，key为问题类型索引，value为问题列表
            if isinstance(data, dict):
                return data
            else:
                print(f"Unexpected data format. Expected dict, got {type(data)}")
                return {}
                
        except Exception as e:
            print(f"Error loading QA data: {str(e)}")
            return {}
    
    def _create_provider(self, model_name: str) -> LLMProvider:
        """创建LLM提供商实例"""
        return LLMProvider(
            provider=ProviderType.ALIPAY,
            model=model_name,
            config={"base_url": self.models_config['providers']['alipay']['api']['base_url'],
             "api_key": self.models_config['providers']['alipay']['api']['key']}
        )
        # return LLMProvider(
        #     provider=ProviderType.OPENAI,
        #     model=model_name,
        #     config= {
        #         "base_url": self.models_config['judge_provider']['base_url'],
        #         "api_key": self.models_config['judge_provider']['api_key'],
        #     }
        # )
    async def _get_llm_response(self, provider, question, gen_config, max_retries=1):
        """
        获取LLM响应并记录统计信息
        
        Args:
            provider: LLM提供商实例
            question: 问题文本
            
        Returns:
            response: 模型响应
            stats: 统计信息
        """
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                start_time = time.time()
                
                # 创建生成配置                
                # 获取响应
                result = await provider.generate(
                    prompt=question,
                    config=gen_config
                )
                
                # 计算统计信息
                stats = {
                    'input_tokens': result.usage.get('prompt_tokens', 0),
                    'output_tokens': result.usage.get('completion_tokens', 0),
                    'response_time': time.time() - start_time,
                    'reasoning_content': result.reasoning_content
                }
                
                return result.text, stats
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    print(f"Attempt {retry_count}/{max_retries} failed: {str(e)}")
                    print(f"Waiting {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"All {max_retries} attempts failed: {str(e)}")
                    # 返回默认值
                    default_stats = {
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'response_time': 0.0,
                        'reasoning_content': "",
                        'error_message': str(e)
                    }
                    return "", default_stats
        
        # 确保所有路径都有返回值
        default_stats = {
            'input_tokens': 0,
            'output_tokens': 0,
            'response_time': 0.0,
            'reasoning_content': ""
        }
        return "", default_stats
    
    def _create_judge_provider(self) -> LLMProvider:
        """创建judge模型提供商"""
        return LLMProvider(
            provider=ProviderType.OPENAI,
            model="gpt-4.1-nano-2025-04-14",
            config= {
                "base_url": self.models_config['judge_provider']['base_url'],
                "api_key": self.models_config['judge_provider']['api_key'],
            }
        )
    
    async def _evaluate_with_llm_judge(self, question: str, reference_answer: str, response: str) -> float:
        """
        使用LLM作为judge评估回答质量
        
        Args:
            question: 问题
            response: 模型回答
            task_type: 任务类型
            
        Returns:
            effectiveness_score: 效果分数 (0-1)
        """

        # 构建评估提示词
        evaluation_prompt = f"""
You are an expert evaluator. Your task is to score the quality and correctness of a model-generated answer to a given question, using a reference answer as the gold standard.

You will be given:
- QUESTION
- REFERENCE ANSWER (correct)
- MODEL ANSWER (to evaluate)

Your goal is to assign a score between **0.0 and 1.0**, where:
- 1.0 = Fully correct and semantically equivalent to the reference.
- 0.5 = Partially correct or incomplete
- 0.0 = Completely incorrect, irrelevant, or nonsensical

Respond with **only the numeric score**, nothing else.

---

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

MODEL ANSWER:
{response}.
Your score:
"""
        gen_config = {
            "temperature": 0.0,
            "max_tokens": 20
        }
        # 获取judge评估
        judge_response, judge_stats = await self._get_llm_response(self.judge_provider, evaluation_prompt, gen_config)
        print(f"Judge response: {judge_response}")
        score_match = re.search(r'\b([0-1](?:\.\d{1,2})?)\b', judge_response.strip())
        if score_match:
            score = float(score_match.group())
            return max(0.0, min(1.0, score)), judge_stats  # 确保分数在0-1之间
        else:
            if "error_message" not in judge_stats.keys():
                judge_stats['error_message'] = f"fail to parse from {judge_response}"
            print(f"无法解析judge分数: {judge_stats['error_message']}")
            return None, judge_stats 
    
    async def evaluate_model(self, model_name: str, qa_data: Dict[str, Any], output_dir: str = "kg_data", file_format: str = "{}.json"):
        """
        评估单个模型在所有问题上的表现
        
        Args:
            model_name: 模型名称
            qa_data: 问答数据字典，key为问题类型索引，value为问题列表
            
        Returns:
            results: 任务结果列表
        """
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*50}")
        
        provider = self._create_provider(model_name)
        model_pricing = self.models_config['providers']['alipay']['models']['available'][model_name]['pricing']
        input_price = model_pricing['input_price']  # 每1M tokens的输入价格
        output_price = model_pricing['output_price']  # 每1M tokens的输出价格
        all_results = {}
        gen_config = {
            "temperature": 0.0,
            "max_tokens": 2048,
            "stream": False,
            "extra_body": {"enable_thinking": False},
        }
        model_output_file = os.path.join(output_dir, file_format.format(model_name))
        # Check if model output file exists
        if os.path.exists(model_output_file):
            all_results = json.load(open(model_output_file, 'r'))
        else:
            all_results = {}
        for task_type_idx in list(qa_data.keys())[320:]:
            # 如果该任务类型已完成，跳过
            # print(f"task_type_idx: {task_type_idx}")
            # print(f"all_results: {len(all_results[task_type_idx])}")
            # print(f"qa_data: {qa_data[task_type_idx]}")
            # print(len(all_results[task_type_idx]))
            # print(len(qa_data[task_type_idx]))
            qa_data[task_type_idx] = qa_data[task_type_idx][:80]
            if task_type_idx in all_results.keys() and len(all_results[task_type_idx]) >= len(qa_data[task_type_idx]):
                print(f"Skipping completed task type {task_type_idx}")
                continue
            if task_type_idx in all_results.keys() and len(all_results[task_type_idx]) < len(qa_data[task_type_idx]):
                completed_questions = len(all_results[task_type_idx])
                print(f"Continuing from completed questions {completed_questions}")
            else:
                all_results[task_type_idx] = []
                completed_questions = 0
            # exit(0)
            questions = qa_data[task_type_idx]
            # print(f"\nProcessing task type {task_type_idx}: {len(questions)} questions")
            
            for i, qa_pair in enumerate(questions[completed_questions:]):
                print(f"Processing question {i+1}/{len(questions) -  completed_questions:} in {task_type_idx}-th task type")
                
                # 获取模型响应
                response, stats = await self._get_llm_response(provider, "Please answer the following question: " + qa_pair[0] + "\n\nAnswer:", gen_config)
                
                # 计算成本
                cost = provider.calculate_cost(stats['input_tokens'], stats['output_tokens'], input_price, output_price)
                
                # 使用LLM作为judge评估效果
                effectiveness, judge_stats = await self._evaluate_with_llm_judge(
                    qa_pair[0], qa_pair[1], response)
                
                if effectiveness is None:
                    all_results[task_type_idx].append({
                    "response": response,
                    "cost": cost,
                    "effectiveness": effectiveness,
                    "latency": stats['response_time'],
                    "input_tokens": stats['input_tokens'],
                    "output_tokens": stats['output_tokens'],
                    "error_message": stats['error_message'] if 'error_message' in stats.keys() else "judge failed"
                })
                else:
                    all_results[task_type_idx].append({
                        "response": response,
                        "cost": cost,
                        "effectiveness": effectiveness,
                        "latency": stats['response_time'],
                        "input_tokens": stats['input_tokens'],
                        "output_tokens": stats['output_tokens'],
                    })
                
                print(f"Response time: {stats['response_time']:.2f}s")
                print(f"Cost: ${cost:.6f}")
                if effectiveness is not None:
                    print(f"Effectiveness: {effectiveness:.4f}")
                else:
                    print(f"Effectiveness: None")

                # 保存该模型的所有结果
            with open(model_output_file, 'w') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"Saved all results for model {model_name} to {model_output_file}")
            
        return all_results
    
    def _check_completed_task_types(self, model_output_file: str) -> set:
        """检查已完成的任务类型"""
        completed_types = set()
        model_output_dir = os.path.dirname(model_output_file)
        
        if os.path.exists(model_output_dir):
            for filename in os.listdir(model_output_dir):
                if filename.startswith("task_type_") and filename.endswith("_results.csv"):
                    # 提取任务类型索引
                    task_type = filename.replace("task_type_", "").replace("_results.csv", "")
                    completed_types.add(task_type)
        
        return completed_types
    
    def calculate_model_performance(self, results: Dict[str, Any], model_name: str):
        """
        计算模型性能统计
        
        Args:
            results: 任务结果列表
            
        Returns:
            performance: 模型性能统计
        """
        if not results:
            return None
        
        # 基础统计
        total_questions = sum(len(results[task_type]) for task_type in results.keys())
        total_cost = sum(sum(r['cost'] for r in results[task_type]) for task_type in results.keys())
        
        # 计算有效的effectiveness数量和总和
        total_valid_effectiveness = 0
        sum_effectiveness = 0
        for task_type in results.keys():
            for r in results[task_type]:
                if r['effectiveness'] is not None:
                    total_valid_effectiveness += 1
                    sum_effectiveness += r['effectiveness']
        
        # 平均指标
        avg_response_time = sum(sum(r['latency'] for r in results[task_type]) for task_type in results.keys()) / total_questions
        avg_cost = total_cost / total_questions
        # 只对有效的effectiveness计算平均值
        avg_effectiveness = sum_effectiveness / total_valid_effectiveness if total_valid_effectiveness > 0 else None
        
        stat = {
            "model_name": model_name,
            "total_questions": total_questions,
            "valid_effectiveness_count": total_valid_effectiveness,
            "avg_response_time": avg_response_time,
            "avg_cost": avg_cost,
            "avg_effectiveness": avg_effectiveness,
            "total_cost": total_cost,
        }
        return stat

    def save_performance_summary(self, performances: List[Dict[str, Any]], output_path: str):
        """保存性能摘要到CSV文件"""
        summary_data = []
        for perf in performances:
            summary_data.append({
                'model_name': perf['model_name'],
                'total_questions': perf['total_questions'],
                'avg_response_time': perf['avg_response_time'],
                'avg_cost': perf['avg_cost'],
                'avg_effectiveness': perf['avg_effectiveness'],
                'total_cost': perf['total_cost'],
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        print(f"Performance summary saved to: {output_path}")
    
    async def run_evaluation(self, 
                           data_path: str,
                           models: List[str],
                           output_dir: str = "kg_data"):
        """
        运行完整的评估流程
        
        Args:
            data_path: 问答数据文件路径
            models: 要评估的模型列表
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        print("Loading QA data...")
        qa_data = self._load_qa_data(data_path)
        print(f"Loaded {len(qa_data)} task types")
        
        # 计算总问题数
        total_questions = sum(len(questions) for questions in qa_data.values())
        print(f"Total questions: {total_questions}")
        
        if not qa_data:
            print("No QA data loaded. Exiting.")
            return

        all_performances = []
        
        # 评估每个模型
        for model_name in models:
            results = await self.evaluate_model(model_name, qa_data, output_dir, file_format="{}.json")                
            # 计算性能统计
            performance = self.calculate_model_performance(results, model_name)
            if performance is not None:
                all_performances.append(performance)

                           
        for performance in all_performances:
            print(f"\n{'-'*30} Performance Summary {'-'*30}")
            print(f"Model: {performance['model_name']}")
            print(f"Total Questions: {performance['total_questions']}")
            print(f"Valid Effectiveness Count: {performance['valid_effectiveness_count']}")
            print(f"Average Response Time: {performance['avg_response_time']:.2f}s")
            print(f"Average Cost: ${performance['avg_cost']:.6f}")
            if performance['avg_effectiveness'] is not None:
                print(f"Average Effectiveness: {performance['avg_effectiveness']:.4f}")
            else:
                print(f"Average Effectiveness: None")
        
        # 保存性能摘要
        summary_path = os.path.join(output_dir, "performance_summary.csv")
        self.save_performance_summary(all_performances, summary_path)
        
        print(f"\n{'='*50}")
        print("Evaluation completed!")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*50}")

async def main():
    """主函数"""
    # 配置
    dir_path = "/hdd2/lh/agenticrouter_data/kg_data_supp"
    data_path = os.path.join(dir_path, "generated_qa_difficulty_nodes.json")
    models_to_test = [
        # "qwen3-0.6b",
        # "qwen3-1.7b",
        # "qwen3-14b",
        # "qwen3-8b", 
        "qwen3-32b",
        # "qwen3-235b-a22b"
    ]

    # models_to_test = [
        # "qwen3-0.6b",
        # "qwen3-1.7b",
        # "qwen3-14b",
        # "qwen3-8b", 
        # "qwen3-32b",
    #     "qwen3-235b-a22b"
    # ]

    # models_to_test = [
    # "gemini-2.5-flash-lite", # 这两个还没跑呢
    # "gemini-2.5-flash",
    # # "gemini-2.0-flash",
    # # "gemini-2.0-flash-lite-preview-02-05",
    # # "doubao-seed-1.6-flash",
    # # "ERNIE-Speed-128K"
    # ]
    # models_to_test = [ "gemini-2.5-flash"]
    # "gemini-2.5-flash-lite", # 这两个还没跑呢
    # "gemini-2.5-flash",
    # # "gemini-2.0-flash",
    # # "gemini-2.0-flash-lite-preview-02-05",
    # # "doubao-seed-1.6-flash",
    # # "ERNIE-Speed-128K"
    # ]
    output_dir = dir_path
    # 创建评估器
    estimator = MetricEstimator()
    
    # 运行评估
    await estimator.run_evaluation(
        data_path=data_path,
        models=models_to_test,
        output_dir=output_dir
    )

if __name__ == "__main__":
    asyncio.run(main())

