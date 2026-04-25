"""
为不同的llm模型构建路由数据
"""
from typing import List, Dict, Tuple, Optional, Any
from data_processing.llm_engine import LLMProvider, GenerationConfig, ProviderType
import pandas as pd
from transformers import AutoTokenizer
import yaml
import asyncio
import time
import os

class DataBuilder:
    def __init__(self, 
                 dataset_name: str, 
                 data_dir: str, 
                 config: Dict, 
                 model_name: str):
        """
        初始化数据构建类
        Args:
            dataset_name: 数据集名称
            data_dir: 数据目录路径
            config: 配置信息
            model_name: 模型名称
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.config = config
        self.model_name = model_name
        
        # 加载数据集的不同分割
        self.splits = {}
        for split in ['test', 'val', 'train']:
            file_path = os.path.join(data_dir, f'{split}.csv')
            if os.path.exists(file_path):
                self.splits[split] = pd.read_csv(file_path)
                print(f"Loaded {split} split: {len(self.splits[split])} samples")
        
        # 从yaml配置文件加载模型配置
        with open('configs/models.yaml', 'r', encoding='utf-8') as f:
            self.models_config = yaml.safe_load(f)
            
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # 初始化provider
        self.provider = self.create_provider()

    def create_provider(self) -> LLMProvider:
        """创建Provider实例"""
        # return LLMProvider(
        #     provider="ALIPAY",
        #     model=self.model_name,
        #     config=self.models_config
        # )
        return LLMProvider(
            provider=ProviderType.OPENROUTER,
            model=self.model_name,
            config=self.models_config
        )

    async def get_llm_response(self, query: str) -> Tuple[str, Dict]:
        """
        获取LLM响应并记录相关统计信息
        Returns:
            response: 模型响应
            stats: 包含token统计和时间信息的字典
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                start_time = time.time()
                
                # 获取生成配置
                gen_config = self.provider.get_generation_config()
                extra_body = self.provider.get_extra_body()
                
                # 计算输入tokens
                input_tokens = len(self.tokenizer.encode(query))
                
                # 准备消息
                messages = [
                    # {"role": "system", "content": "Please answe the question following the instruction."},
                    {"role": "user", "content": query}
                ]
                # print(messages)
                # 获取响应
                result = await self.provider.generate(
                    messages=messages,
                    config=gen_config,
                    extra_body=extra_body
                )
                
                # 计算输出tokens和统计信息
                stats = {
                    'input_tokens': result.usage.get('prompt_tokens', input_tokens),
                    'output_tokens': result.usage.get('completion_tokens', len(self.tokenizer.encode(result.text))),
                    'response_time': time.time() - start_time,
                    'reasoning_content': result.reasoning_content
                }
                
                return result.text, stats
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # 指数退避
                    print(f"尝试 {retry_count}/{max_retries} 失败: {str(e)}")
                    print(f"等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"在 {max_retries} 次尝试后仍然失败: {str(e)}")
                    # 返回默认值
                    default_stats = {
                        'input_tokens': len(self.tokenizer.encode(query)),
                        'output_tokens': 0,
                        'response_time': None,
                        'reasoning_content': ""
                    }
                    return "", default_stats

    async def process_split(self, split_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """处理单个数据集分割"""
        results = []
        
        for idx, row in df.iterrows():
            query = str(row['query'])  # 确保query是字符串类型
            ground_truth = row.get('ground_truth')
            metric = row.get('metric')
            # 处理长文本
            if self.dataset_name == "multi_news":
                tokens = self.tokenizer.tokenize(query)
                extracted_text = tokens[:3000]
                query = self.tokenizer.convert_tokens_to_string(extracted_text)
            
            print(f"\nProcessing {split_name} split, row {idx}")
            
            # 获取模型响应和统计信息
            response, stats = await self.get_llm_response(query)
            print(stats)
            # 创建结果行
            result_row = {
                **row.to_dict(),  # 保留原始数据
                'model_name': self.model_name,
                'response': response,
                'split': split_name,
                'input_tokens': stats['input_tokens'],
                'output_tokens': stats['output_tokens'],
                'response_time': stats['response_time'],
                'reasoning_content': stats.get('reasoning_content')
            }

            # 计算效果评估
            # if ground_truth is not None:
            result_row['effect'] = self.provider.eval(response, ground_truth, metric=metric)
            print(f"Response: {response}")
            print(f"Ground truth: {ground_truth}")
            print(f"Effectiveness ({metric}): {result_row['effect']:.4f}")
            
            results.append(result_row)
        
        return pd.DataFrame(results)

    async def process_all_splits(self, use_saved_flag=False):
        """处理所有数据集分割"""
        all_results = []  # 存储所有分割的结果
        
        for split_name, df in self.splits.items():
            print(f"\nProcessing {split_name} split...")
            
            output_file = os.path.join(self.data_dir, f'{self.model_name}_{split_name}.csv')
            
            # 如果use_saved_flag为True且文件存在，则直接加载已有文件
            if use_saved_flag and os.path.exists(output_file):
                print(f"Loading existing results from {output_file}")
                results_df = pd.read_csv(output_file)
            else:
                results_df = await self.process_split(split_name, df)
                results_df.to_csv(output_file, index=False)
                print(f"Saved results to {output_file}")
            
            all_results.append(results_df)
        
        # 合并所有结果
        combined_df = pd.concat(all_results, axis=0, ignore_index=True)
        combined_output = os.path.join(self.data_dir, f'{self.model_name}.csv')
        combined_df.to_csv(combined_output, index=False)
        print(f"\nSaved combined results to {combined_output}")
        
        # 删除临时文件
        for split_name in self.splits.keys():
            temp_file = os.path.join(self.data_dir, f'{self.model_name}_{split_name}.csv')
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Removed temporary file: {temp_file}")

if __name__ == "__main__":
    async def main():
        # 加载配置
        with open("configs/models.yaml", 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # # 所有需要测试的模型
        models_to_test = [
            # "qwen3-14b",
            # "qwen3-8b",
            # "qwen3-1.7b",
            # "qwen3-0.6b",
            # "qwen3-32b",
            # "qwen3-235b-a22b"
            # "gpt-5",
            # "gpt-5-mini",
            "gpt-5-nano",
        ]        
        # models_to_test = [
        #     # "gemini-2.5-flash-lite",
        #     # "gemini-2.5-flash",
        #     # "gemini-2.0-flash",
        #     # "gemini-2.0-flash-lite-preview-02-05",
        #     # "doubao-seed-1.6-flash",
        #     # "ERNIE-Speed-128K"
        # ]
                    # "qwen3-235b-a22b",
            # "qwen3-30b-a3b",
        
        # 数据集设置
        dataset_names = [
            "SQUAD",
            "alpaca_data",
            "GSM8K",
            "multi_news",
            # "mbpp",
            # "wmt",
            # "legalbench",
            # "medmcqa",
            # "mmlu_redux"
        ]
        
        data_dirs = [
            "/hdd2/lh/agenticrouter_data/data/SQUAD",
            "/hdd2/lh/agenticrouter_data/data/alpaca_data",
            "/hdd2/lh/agenticrouter_data/data/GSM8K",
            "/hdd2/lh/agenticrouter_data/data/multi_news",
            # "/hdd2/lh/agenticrouter_data/data/mbpp",
            # "/hdd2/lh/agenticrouter_data/data/wmt",
            # "/hdd2/lh/agenticrouter_data/data/legalbench",
            # "/hdd2/lh/agenticrouter_data/data/medmcqa",
        ]
        
        # 是否使用已保存的结果
        use_saved_flag = True
        
        # 遍历每个数据集
        for dataset_name, data_dir in zip(dataset_names, data_dirs):
            print(f"\n{'#'*50}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'#'*50}\n")
            
            # 依次测试每个模型
            for model_name in models_to_test:
                print(f"\n{'='*50}")
                print(f"Testing model: {model_name}")
                print(f"{'='*50}\n")
                
                
                # 创建数据构建器实例
                builder = DataBuilder(
                    dataset_name=dataset_name, 
                    data_dir=data_dir, 
                    config=config,
                    model_name=model_name
                )
                # 处理数据并等待完成
                await builder.process_all_splits(use_saved_flag=use_saved_flag)
            
    
    # 使用asyncio.run来运行主函数
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # 如果已经在事件循环中，使用get_event_loop
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())