"""
This file is to load the data of training, validation and test for different LLMs rourted by our router.
"""
import pandas as pd
import yaml
from typing import Dict, List, Any, Tuple, Union, Optional
from pathlib import Path
import os
import numpy as np
import json
from collections import defaultdict
import random
import math

import torch

# def set_seed(seed: int = 42):
#     """
#     Set random seed for reproducibility.

#     Args:
#         seed (int): The seed value to use.
#     """
#     random.seed(seed)                      # Python 内置随机模块
#     np.random.seed(seed)                   # NumPy 随机模块
#     torch.manual_seed(seed)                # CPU 上的 PyTorch 随机性
#     torch.cuda.manual_seed(seed)           # 当前 GPU 上的随机性
#     torch.cuda.manual_seed_all(seed)       # 所有 GPU 的随机性（如果使用多 GPU）

#     torch.backends.cudnn.deterministic = True   # 让 cudnn 使用确定性算法
#     torch.backends.cudnn.benchmark = False      # 禁用自动优化算法选择（可能影响性能但确保复现）

#     print(f"[INFO] Random seed set to {seed}")

DOMAIN_FORMAT = "Pleas give me  questions of {} domain that {}"

SUBCAT_FORMAT = "Pleas give me questions of subcatogry task {} of {} domian that {}"

DIFF_FORMAT = "Pleas give me questions of subcatogry task {} at {} difficlty level that {}"

# DIFF_FORMAT = "Pleas give me questions of subcatogry task {} at {} difficlty level"

def calculate_token_cost(input_tokens, output_tokens, input_price, output_price):
    """计算基于token数量和模型定价的实际成本"""
    input_cost = (input_tokens / 1000000) * input_price  # 转换为百万token单位
    output_cost = (output_tokens / 1000000) * output_price
    return input_cost + output_cost
    
class DatasetGen:
    """数据集加载器"""
    
    def __init__(self, data_dir: str = "data", task_names: List[str] = None,
     models_config_path: str = "configs/models.yaml", file_format: str = "{}.csv", model_names: List[str] = None,
      query_task_type_path: str = "/hdd2/lh/agenticrouter_data/query_task_type_results/classification_results.json",
      support_multiple_labels: bool = False):
        self.data_dir = Path(data_dir)
        self.task_names = task_names 
        self.file_format = file_format
        self.model_names = model_names 
        self.models_config = self._load_models_config(models_config_path)
        self.model_prices = self._load_model_prices()
        self.query_task_type_path = query_task_type_path
        self.support_multiple_labels = support_multiple_labels
        train_diff_id, val_diff_id, test_diff_id = self._load_model_task_type_data(
            self.query_task_type_path, support_multiple=support_multiple_labels
        )
        self.split_to_diff_ids = {
            'train': train_diff_id,
            'val': val_diff_id,
            'test': test_diff_id
        }
        
    def _load_models_config(self, config_path: str) -> Dict:
        """Load models configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    
    def _load_model_prices(self) -> pd.DataFrame:
        """Load model pricing information."""
        prices_data = {}
        for provider, provider_config in self.models_config.get('providers', {}).items():
            if not provider_config.get('enabled', False):
                continue
            for model_name, model_config in provider_config['models'].get('available', {}).items():
                # print(model_config)
                pricing = model_config.get('pricing', {})
                prices_data[model_name] = {
                    'input_price': pricing.get('input_price', 0),
                    'output_price': pricing.get('output_price', 0)
                }
        
        return prices_data

    def _load_model_task_type_data(self, query_task_type_path: str, support_multiple: bool = False) -> Dict:
        """
        Load model task type data from JSON file.
        
        Args:
            query_task_type_path: Path to the JSON file containing task type classifications
            support_multiple: If True, supports multi-label classification (one query can have multiple task types)
                            If False, supports single-label classification (one query has one task type)
        
        Returns:
            Tuple of (train_diff_id, val_diff_id, test_diff_id)
            - If support_multiple=False: Lists of difficulty IDs (one per query)
            - If support_multiple=True: Lists of lists of difficulty IDs (multiple per query)
        """
        with open(query_task_type_path, 'r', encoding='utf-8') as f:
            query_task_type_data = json.load(f)
        
        def extract_difficulty_ids(data_dict, is_multiple):
            """Extract difficulty IDs from data dictionary"""
            result = []
            for q_id, value in data_dict.items():
                if is_multiple:
                    # Multi-label format: value is a list of dicts
                    if isinstance(value, list):
                        # Extract difficulty IDs from each dict in the list
                        diff_ids = [item["difficulty"] for item in value if "difficulty" in item]
                        result.append(diff_ids)
                    else:
                        # Fallback: treat as single label
                        result.append([value["difficulty"]] if isinstance(value, dict) and "difficulty" in value else [])
                else:
                    # Single-label format: value is a dict with "difficulty" key
                    if isinstance(value, dict) and "difficulty" in value:
                        result.append(value["difficulty"])
                    elif isinstance(value, list) and len(value) > 0:
                        # If it's a list, take the first one (backward compatibility)
                        result.append(value[0]["difficulty"] if isinstance(value[0], dict) and "difficulty" in value[0] else "")
                    else:
                        result.append("")
            return result
        
        # Detect format automatically if support_multiple is None
        # Check first non-empty value to determine format
        sample_value = None
        for split in ["train", "val", "test"]:
            if split in query_task_type_data and query_task_type_data[split]:
                sample_value = list(query_task_type_data[split].values())[0]
                break
        
        is_multiple_format = False
        if sample_value is not None:
            if isinstance(sample_value, list):
                is_multiple_format = True
            elif isinstance(sample_value, dict) and "difficulty" in sample_value:
                is_multiple_format = False
        
        # Use support_multiple flag if explicitly set, otherwise auto-detect
        use_multiple = support_multiple if support_multiple else is_multiple_format
        
        train_diff_id = extract_difficulty_ids(query_task_type_data["train"], use_multiple)
        val_diff_id = extract_difficulty_ids(query_task_type_data["val"], use_multiple)
        test_diff_id = extract_difficulty_ids(query_task_type_data["test"], use_multiple)
        
        # Store format info for later use
        self.is_multiple_label = use_multiple
        
        return train_diff_id, val_diff_id, test_diff_id

    def sample_data(self, data_df: pd.DataFrame, shot_num: Optional[int] = 5, split: str = "train") -> pd.DataFrame:
        """
        为每个dataset_name采样指定数量的数据点，并添加difficulty_id列
        
        Args:
            data_df: 输入的DataFrame，包含'query'和'task_description'列
            shot_num: 每个dataset需要采样的数据点数量，如果为None则返回所有有效数据点
            split: 数据分割类型 ('train', 'val', 'test')
            
        Returns:
            采样后的DataFrame，包含新增的'difficulty_id'列
            - 如果support_multiple_labels=False: difficulty_id是单个字符串
            - 如果support_multiple_labels=True: difficulty_id是列表（多标签）
        """
        # 创建DataFrame的副本
        df = data_df.copy()
        
        # 初始化difficulty_id列
        if self.is_multiple_label:
            df['difficulty_id'] = None  # Will be list
        else:
            df['difficulty_id'] = ''
        
        # 获取所有唯一的dataset names
        dataset_names = df['task_description'].unique()
        # 存储采样后的数据
        sampled_dfs = []
        for dataset_name in dataset_names:
            # 获取当前dataset的数据
            dataset_mask = df['task_description'] == dataset_name
            dataset_df = df[dataset_mask].copy()
            # 根据索引分配difficulty_id
            split_indices = self.task_indices[split][dataset_name]
            for idx in split_indices:
                if idx < len(self.split_to_diff_ids[split]):
                    diff_id_value = self.split_to_diff_ids[split][idx]
                    if self.is_multiple_label:
                        # Multi-label: diff_id_value is a list
                        dataset_df.loc[idx, 'difficulty_id'] = diff_id_value if isinstance(diff_id_value, list) else [diff_id_value]
                    else:
                        # Single-label: diff_id_value is a string
                        dataset_df.loc[idx, 'difficulty_id'] = diff_id_value if isinstance(diff_id_value, str) else str(diff_id_value)
       
            # 过滤掉difficulty_id为空的行
            if self.is_multiple_label:
                valid_data = dataset_df[dataset_df['difficulty_id'].notna() & (dataset_df['difficulty_id'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False))]
            else:
                valid_data = dataset_df[dataset_df['difficulty_id'] != '']
            
            if len(valid_data) == 0:
                print(f"No valid data for {dataset_name} in {split}")
                exit(0)
                
            if shot_num is not None:
                # 如果需要采样特定数量
                if len(valid_data) >= shot_num:
                    # 随机采样指定数量的数据
                    sampled_data = valid_data.sample(n=shot_num, random_state=42)
                else:
                    # 如果数据量不足，使用所有有效数据
                    sampled_data = valid_data
            else:
                # 如果shot_num为None，使用所有有效数据
                sampled_data = valid_data
            
            sampled_dfs.append(sampled_data)
        
        # 合并所有采样的数据
        if sampled_dfs:
            result_df = pd.concat(sampled_dfs, ignore_index=True)
        else:
            # 如果没有有效数据，返回空DataFrame但保持原有的列结构
            result_df = pd.DataFrame(columns=df.columns)
            
        return result_df

    def load_data(self) -> tuple:
        """Load and process data for all datasets and models, returning train, validation, and test DataFrames."""
        train_data = {'query': [], 'answer': [], 'effect': [],  'cost': [], 'latency': [], 'task_description': []}
        val_data = {'query': [], 'answer': [], 'effect': [], 'cost': [], 'latency': [], 'task_description': []}
        test_data = {'query': [], 'answer': [], 'effect': [], 'cost': [], 'latency': [], 'task_description': []}
        self.task_indices = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list)
        }
        for dataset_name in self.task_names:
            dataset_dir = os.path.join(self.data_dir, dataset_name)
            if not os.path.exists(dataset_dir):
                print(f"Dataset directory {dataset_dir} does not exist.")
                continue
            print(f"Processing dataset: {dataset_name}")
            data_diff_models = []
            price_diff_models = []
            for model_name in self.model_names:
                pricing = self.model_prices[model_name]
                data_path = os.path.join(dataset_dir, self.file_format.format(model_name))
                data = pd.read_csv(data_path)
                data_diff_models.append(data)
                price_diff_models.append(pricing)
                # Check if all dataframes have the same length
                if data_diff_models:
                    first_len = len(data_diff_models[0])
                    for i, df in enumerate(data_diff_models[1:], 1):
                        if len(df) != first_len:
                            print(f"Warning: Model {self.model_names[i]} has {len(df)} rows, "
                                  f"expected {first_len} rows like {self.model_names[0]}")
                            exit()
            for i in range(len(data_diff_models[0])):
                split = data_diff_models[0]['split'].iloc[i] 
                query = data_diff_models[0]['query'].iloc[i] 
                effect = []
                cost = []
                latency = []
                answer = []
                for j in range(len(data_diff_models)):
                    flag = False
                    answer.append(data_diff_models[j]['response'].iloc[i])
                    effect.append(data_diff_models[j]['effect'].iloc[i])
                    # else:
                    # GSM8K使用effect，其他任务使用llm_judge_effect
                    # if dataset_name == 'GSM8K':
                    #     effect.append(data_diff_models[j]['effect'].iloc[i])
                    # else:
                    #     if pd.notna(data_diff_models[j]['llm_judge_effect'].iloc[i]):
                    #         effect.append(data_diff_models[j]['llm_judge_effect'].iloc[i])
                    #     else:
                    #         effect.append(0)
                    #         flag = True
                    # if pd.notna(data_diff_models[j]['llm_judge_effect'].iloc[i]):
                    #     effect.append(data_diff_models[j]['llm_judge_effect'].iloc[i])
                    # else:
                    #     effect.append(0)
                    #     flag = True

                    input_tokens = data_diff_models[j]['input_tokens'].iloc[i] 
                    output_tokens = data_diff_models[j]['output_tokens'].iloc[i] 
                    cost.append(calculate_token_cost(input_tokens, output_tokens, 
                    price_diff_models[j]['input_price'], price_diff_models[j]['output_price']))
                    latency_val = data_diff_models[j]['response_time'].iloc[i]
                    # Handle NaN latency values
                    if pd.isna(latency_val):
                        latency_val = self._replace_nan_latency(data_diff_models, j, i)
                        # print(f"Replaced NaN latency for model {self.model_names[j]} at index {i} with {latency_val:.4f}")
                    latency.append(latency_val)
                
                if flag:
                #     # print(f"Flag is True for {query}")
                    continue
                if split == 'train':
                    train_data['query'].append(query)
                    train_data['effect'].append(effect)
                    train_data['cost'].append(cost)
                    train_data['latency'].append(latency)
                    train_data['task_description'].append(dataset_name)
                    train_data['answer'].append(answer)
                    self.task_indices['train'][dataset_name].append(len(train_data['query'])-1)
                elif split == 'val':
                    val_data['query'].append(query)
                    val_data['effect'].append(effect)
                    val_data['cost'].append(cost)
                    val_data['latency'].append(latency)
                    val_data['task_description'].append(dataset_name)
                    val_data['answer'].append(answer)
                    self.task_indices['val'][dataset_name].append(len(val_data['query'])-1)
                else:  # default to test if split is not specified or is 'test'
                    test_data['query'].append(query)
                    test_data['effect'].append(effect)
                    test_data['cost'].append(cost)
                    test_data['latency'].append(latency)
                    test_data['task_description'].append(dataset_name)
                    test_data['answer'].append(answer)
                    self.task_indices['test'][dataset_name].append(len(test_data['query'])-1)
        train_data_df = pd.DataFrame(train_data)
        val_data_df = pd.DataFrame(val_data)
        test_data_df = pd.DataFrame(test_data)

        return train_data_df, val_data_df, test_data_df

    def load_data_with_dual_effect(self) -> tuple:
        """
        Load data with both traditional effect and llm_judge_effect for analysis.
        Returns (train_df, val_df, test_df) where each has 'effect', 'llm_judge_effect' (if available), 'cost', 'latency'.
        """
        train_data = {'query': [], 'answer': [], 'effect': [], 'llm_judge_effect': [], 'cost': [], 'latency': [], 'task_description': []}
        val_data = {'query': [], 'answer': [], 'effect': [], 'llm_judge_effect': [], 'cost': [], 'latency': [], 'task_description': []}
        test_data = {'query': [], 'answer': [], 'effect': [], 'llm_judge_effect': [], 'cost': [], 'latency': [], 'task_description': []}
        self.task_indices = {'train': defaultdict(list), 'val': defaultdict(list), 'test': defaultdict(list)}
        
        for dataset_name in self.task_names:
            dataset_dir = os.path.join(self.data_dir, dataset_name)
            if not os.path.exists(dataset_dir):
                continue
            data_diff_models = []
            price_diff_models = []
            for model_name in self.model_names:
                data_path = os.path.join(dataset_dir, self.file_format.format(model_name))
                if not os.path.exists(data_path):
                    continue
                data = pd.read_csv(data_path)
                data_diff_models.append(data)
                pricing = self.model_prices.get(model_name, {'input_price': 0, 'output_price': 0})
                price_diff_models.append(pricing)
            
            if not data_diff_models:
                continue
                
            has_llm_judge = 'llm_judge_effect' in data_diff_models[0].columns
            for i in range(len(data_diff_models[0])):
                split = data_diff_models[0]['split'].iloc[i]
                query = data_diff_models[0]['query'].iloc[i]
                effect, llm_effect, cost, latency, answer = [], [], [], [], []
                for j in range(len(data_diff_models)):
                    answer.append(data_diff_models[j]['response'].iloc[i])
                    eff_val = data_diff_models[j]['effect'].iloc[i]
                    effect.append(eff_val if pd.notna(eff_val) else 0)
                    if has_llm_judge:
                        lj_val = data_diff_models[j]['llm_judge_effect'].iloc[i]
                        llm_effect.append(lj_val if pd.notna(lj_val) else 0)
                    else:
                        llm_effect.append(eff_val if pd.notna(eff_val) else 0)
                    input_tokens = data_diff_models[j]['input_tokens'].iloc[i]
                    output_tokens = data_diff_models[j]['output_tokens'].iloc[i]
                    cost.append(calculate_token_cost(input_tokens, output_tokens, 
                        price_diff_models[j]['input_price'], price_diff_models[j]['output_price']))
                    latency_val = data_diff_models[j]['response_time'].iloc[i]
                    latency.append(latency_val if pd.notna(latency_val) else 0)
                
                row_data = {'query': query, 'answer': answer, 'effect': effect, 'llm_judge_effect': llm_effect,
                           'cost': cost, 'latency': latency, 'task_description': dataset_name}
                if split == 'train':
                    for k, v in row_data.items():
                        train_data[k].append(v)
                    self.task_indices['train'][dataset_name].append(len(train_data['query']) - 1)
                elif split == 'val':
                    for k, v in row_data.items():
                        val_data[k].append(v)
                    self.task_indices['val'][dataset_name].append(len(val_data['query']) - 1)
                else:
                    for k, v in row_data.items():
                        test_data[k].append(v)
                    self.task_indices['test'][dataset_name].append(len(test_data['query']) - 1)
        
        return pd.DataFrame(train_data), pd.DataFrame(val_data), pd.DataFrame(test_data)

    def get_max_min_values_per_task(self, train_data_df, val_data_df, test_data_df):
        """
        Get the max and min values of the metrics for each task and normalize costs
        
        Args:
            train_data_df: DataFrame containing training data
            val_data_df: DataFrame containing validation data
            test_data_df: DataFrame containing test data
            
        Returns:
            Tuple[Dict, Dict]: Maximum and minimum values for each metric
        """
        # 合并所有数据集
        
        # 初始化每个任务的最大最小值字典
        self.task_max_min = {}
        
        # 对每个任务分别计算最大最小值
        for task_name in self.task_names:
            # 获取该任务在所有split中的索引
            train_indices = self.task_indices['train'][task_name]
            val_indices = self.task_indices['val'][task_name]
            test_indices = self.task_indices['test'][task_name]
            # 获取该任务的所有数据
            task_data_train = train_data_df.iloc[train_indices]
            task_data_val = val_data_df.iloc[val_indices] 
            task_data_test = test_data_df.iloc[test_indices] 
            # 合并该任务的所有数据
            task_data = pd.concat([task_data_train, task_data_val, task_data_test])
            
            # 计算该任务的cost最大最小值
            task_costs = np.array([cost_list for cost_list in task_data['cost'] ])
            task_latencies = np.array([latency_list for latency_list in task_data['latency']])
            task_effects = np.array([effect_list for effect_list in task_data['effect']])
   
            
            self.task_max_min[task_name] = {
                'cost_max': np.max(task_costs),
                'cost_min': np.min(task_costs),
                'latency_max': np.max(task_latencies),
                'latency_min': np.min(task_latencies),
                'effect_max': np.max(task_effects),
                'effect_min': np.min(task_effects)
            }
            # print(task_max_min[task_name])
            
            # 对该任务的每个split进行归一化
            def normalize_costs(df, indices):
                    df.loc[indices, 'normalized_cost'] = df.loc[indices, 'cost'].apply(
                        lambda x: [(c - self.task_max_min[task_name]['cost_min']) / 
                                 (self.task_max_min[task_name]['cost_max'] - self.task_max_min[task_name]['cost_min'])
                                 for c in x]
                    )
            normalize_costs(train_data_df, train_indices)
            normalize_costs(val_data_df, val_indices)
            normalize_costs(test_data_df, test_indices)

            def normalized_latencies(df, indices):
                    df.loc[indices, 'normalized_latency'] = df.loc[indices, 'latency'].apply(
                        lambda x: [(l - self.task_max_min[task_name]['latency_min']) / 
                                 (self.task_max_min[task_name]['latency_max'] - self.task_max_min[task_name]['latency_min'])
                                 for l in x]
                    )
            normalized_latencies(train_data_df, train_indices)
            normalized_latencies(val_data_df, val_indices)
            normalized_latencies(test_data_df, test_indices)
        
        return self.task_max_min

    def get_max_min_values(self, train_data_df, val_data_df, test_data_df):
        """
        Get the max and min values of the metrics (effect, cost, latency) across all splits
        
        Args:
            train_data_df: DataFrame containing training data
            val_data_df: DataFrame containing validation data
            test_data_df: DataFrame containing test data
            
        Returns:
            Tuple[Dict, Dict]: Maximum and minimum values for each metric
        """
        # 合并所有数据集
        all_data = pd.concat([train_data_df, val_data_df, test_data_df])
        
        # 处理cost（每个元素是list）
        all_costs = np.array([c for cost_list in all_data['cost'] for c in cost_list])
        all_latencies = np.array([l for latency_list in all_data['latency'] for l in latency_list])
        
        max_values = {'effect': 1, 'cost': np.max(all_costs), 'latency': np.max(all_latencies)}
        min_values = {'effect': 0, 'cost': np.min(all_costs), 'latency': np.min(all_latencies)}
        
        return min_values, max_values
    def _replace_nan_latency(self, data_diff_models, model_idx, row_idx):
        """
        Get replacement latency for NaN values using nearby samples' average * 3
        """
        # Collect nearby non-NaN latency values
        nearby_latencies = []
        
  
        # If not enough samples, check nearby queries for the same LLM
        if len(nearby_latencies) < 3:
            start_idx = max(0, row_idx - 2)
            end_idx = min(len(data_diff_models[model_idx]), row_idx + 3)
            for i in range(start_idx, end_idx):
                if i != row_idx:  # Skip current query
                    lat = data_diff_models[model_idx]['response_time'].iloc[i]
                    if not np.isnan(lat):
                        nearby_latencies.append(lat)
        
        # If still not enough, use all non-NaN values from the same LLM
        if len(nearby_latencies) < 3:
            same_llm_latencies = data_diff_models[model_idx]['response_time'].iloc[row_idx]
            if not np.isnan(same_llm_latencies):
                nearby_latencies.append(same_llm_latencies)
        
        # Calculate replacement value (average * 3)
        if len(nearby_latencies) > 0:
            avg_latency = np.mean(nearby_latencies)
            replacement = avg_latency * 3.0
        else:
            # Fallback: use overall average * 3
            all_latencies = data_diff_models[model_idx]['response_time'].iloc[row_idx]
            non_nan_latencies = all_latencies[~np.isnan(all_latencies)]
            if len(non_nan_latencies) > 0:
                replacement = np.mean(non_nan_latencies) * 3.0
            else:
                replacement = 10.0  # Ultimate fallback
        
        return replacement



def load_kg(path: str) -> Dict[str, Any]:
        """Load knowledge graph from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)



def merge_qa_data_with_exemplars(
    train_qa_data: Dict[str, List[Dict]], 
    train_qa_data_real: Dict[str, List[Dict]], 
    exemplar_size: int = 5,
    random_seed: int = 42
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    合并 train_qa_data 和 train_qa_data_real，并为每个任务保存若干个 qa 对作为 exemplar
    
    Args:
        train_qa_data: 第一个训练 QA 数据字典 {task_name: [qa_pairs]}
        train_qa_data_real: 第二个训练 QA 数据字典 {task_name: [qa_pairs]}  
        exemplar_size: 每个任务保留的 exemplar 数量
        random_seed: 随机种子，用于可重现的结果
        
    Returns:
        Tuple[merged_data, exemplar_data]:
            - merged_data: 合并后的完整数据 {task_name: [qa_pairs]}
            - exemplar_data: 每个任务的 exemplar 数据 {task_name: [qa_pairs]}
    """

    
    # 设置随机种子以确保可重现性
    random.seed(random_seed)
    
    merged_data = defaultdict(list)
    exemplar_data = {}
    
    # 获取所有任务名称
    all_tasks = train_qa_data.keys()
    
    print(f"开始合并数据，共发现 {len(all_tasks)} 个任务")
    
    for task_name in all_tasks:
        # 收集该任务的所有 qa 对 
        task_qa_pairs = []
        
        # 从 train_qa_data 添加数据
                    
        # 从 train_qa_data_real 添加数据
        if task_name in train_qa_data_real:
            task_qa_pairs.extend(train_qa_data_real[task_name])
            # print(f"  {task_name}: 从 train_qa_data_real 添加 {len(train_qa_data_real[task_name])} 条")
        if len(task_qa_pairs) < exemplar_size:

            if task_name in train_qa_data:
                task_qa_pairs.extend(train_qa_data[task_name])
            # print(f"  {task_name}: 从 train_qa_data 添加 {len(train_qa_data[task_name])} 条")

        # 随机打乱数据
        random.shuffle(task_qa_pairs)
        
        # 保存合并后的数据
        merged_data[task_name] = task_qa_pairs
        
        # 选择 exemplar 数据
        exemplar_count = min(exemplar_size, len(task_qa_pairs))
        exemplar_data[task_name] = task_qa_pairs[:exemplar_count]
            
            # print(f"  {task_name}: 去重后保留 {len(unique_qa_pairs)} 条，exemplar {exemplar_count} 条")
    
    print(f"\n合并完成！总共 {len(merged_data)} 个任务")
    
    return  exemplar_data



def split_qa_data(qa_data: Dict[str, Any], split_ratio: float = 0.8) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split QA data into training and validation sets based on the given ratio.
    
    Args:
        qa_data: Dictionary containing QA examples for different keys
        split_ratio: Ratio for splitting data (e.g., 0.8 means 80% training, 20% validation)
        
    Returns:
        Tuple of (training_data, validation_data)
    """
    train_data = {}
    val_data = {}
    
    for key, examples in qa_data.items():
        if not isinstance(examples, list):
            continue
            
        # Shuffle the examples
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        # Calculate split point
        split_idx = math.floor(len(shuffled) * split_ratio)
        
        # Split the data
        train_data[key] = shuffled[:split_idx]
        val_data[key] = shuffled[split_idx:]
    
    return train_data, val_data


def load_few_shot_results(results_path: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """加载few_shot_t_labeler.py产生的分类结果"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_classification_to_qa_format(
    classification_results: Dict[str, Dict[str, str]], 
    original_data: pd.DataFrame,
    kg_data: Dict[str, Any]
) -> Dict[str, List[Tuple[str, str]]]:
    """
    将分类结果转换为QA格式，用于构建example-based表征
    
    Args:
        classification_results: few_shot_t_labeler的分类结果
        original_data: 原始数据
        kg_data: 知识图谱数据
        
    Returns:
        Dict[difficulty_id, List[Tuple[query, answer]]]
    """
    qa_data = defaultdict(list)
    assert len(classification_results) == len(original_data)
    for idx_str, classification in classification_results.items():
        idx = int(idx_str)
            
        query = original_data.iloc[idx]['query']
        # 假设original_data中有answer列，如果没有可以用空字符串
        answer = original_data.iloc[idx].get('answer', '')
        
        difficulty_id = classification.get('difficulty', '')
        if difficulty_id:
            qa_data[difficulty_id].append((query, answer))
    
    return dict(qa_data)




def normalize_metric(x: Union[list, torch.Tensor], min_val: float, max_val: float, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    x = torch.clamp(x, min_val, max_val)
    normalized = (x - min_val) / (max_val - min_val)
    if mask is not None:
        normalized = normalized * mask
    return normalized


def process_qa_pairs_and_metrics(task_name: str, task_sample_indices: Dict[str, List[int]], 
                                task_sample_val_indices: Dict[str, List[int]], qa_pairs: List[Dict], 
                                qa_data: Dict[str, List]) -> Dict[str, Any]:
    """
    Process QA pairs and extract metrics for both training and validation sets.
    
    Args:
        task_name: Name of the current task
        task_sample_indices: Dictionary mapping task names to training sample indices
        task_sample_val_indices: Dictionary mapping task names to validation sample indices
        qa_pairs: List of QA pairs for the current task
        qa_data: Dictionary containing all QA data
        
    Returns:
        Dictionary containing processed QA pairs and metrics for both training and validation sets
    """
    # Initialize lists for storing processed data
    qa_pairs_org = []
    qa_pairs_val = []
    cost = []
    effectiveness = []
    latency = []
    mask_org = []
    mask_val = []
    cost_val = []
    effectiveness_val = []
    latency_val = []
    
    # Process training data
    for qa_index in task_sample_indices[task_name]:
        # print(qa_index)
        # print(qa_index)
        # print(type(qa_index))
        # print(len(qa_pairs))
        qa = qa_pairs[qa_index]
        # Handle None values in metrics
        if qa["effectiveness"] is None:
            qa["effectiveness"] = 0
        if qa["cost"] is None or qa["latency"] is None:
            qa["cost"] = 0
            qa["latency"] = 0
            mask_org.append(0)  # 0 is mask, 1 is not mask
        else:
            mask_org.append(1)
            
        # Add query from qa_data
        qa["query"] = qa_data[task_name][qa_index][0]
        qa_pairs_org.append(qa["query"])
        
        # Extract metrics
        cost.append(qa["cost"]/1000)
        effectiveness.append(qa["effectiveness"])
        latency.append(qa["latency"])
        
    # Process validation data
    for qa_index in task_sample_val_indices[task_name]:
        qa = qa_pairs[qa_index]
        # Handle None values in metrics
        if qa["effectiveness"] is None:
            qa["effectiveness"] = 0
        if qa["cost"] is None or qa["latency"] is None:
            qa["cost"] = 0
            qa["latency"] = 0
            mask_val.append(0)  # 0 is mask, 1 is not mask
        else:
            mask_val.append(1)
            
        # Add query from qa_data
        qa["query"] = qa_data[task_name][qa_index][0]
        qa_pairs_val.append(qa["query"])
        
        # Extract metrics
        cost_val.append(qa["cost"]/1000)
        effectiveness_val.append(qa["effectiveness"])
        latency_val.append(qa["latency"])
        
    return {
        "qa_pairs_org": qa_pairs_org,
        "qa_pairs_val": qa_pairs_val,
        "cost": torch.tensor(cost),
        "effectiveness": torch.tensor(effectiveness, dtype=torch.float32),
        "latency": torch.tensor(latency),  
        "cost_val": torch.tensor(cost_val),
        "effectiveness_val": torch.tensor(effectiveness_val, dtype=torch.float32),
        "latency_val": torch.tensor(latency_val),
        "mask_org": torch.tensor(mask_org),
        "mask_val": torch.tensor(mask_val)
    }



def _load_model_prices_from_config(config_path: str = "configs/models.yaml") -> Dict[str, Dict[str, float]]:
    """Load model pricing (input_price, output_price per million tokens) from YAML config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    prices_data = {}
    for provider, provider_config in config.get('providers', {}).items():
        if not provider_config.get('enabled', False):
            continue
        for model_name, model_config in provider_config['models'].get('available', {}).items():
            pricing = model_config.get('pricing', {})
            prices_data[model_name] = {
                'input_price': pricing.get('input_price', 0),
                'output_price': pricing.get('output_price', 0)
            }
    return prices_data


def load_kg_qa_data(kg_qa_path: str, result_format: str = "{model}.json", train_shot_num: int = 5, val_shot_num: int = 5, model_names: List[str] = None,
                model_result_dir: str = None, task_hierarchy: Dict[str, Any] = None, normalize_mode: str = "none",
                use_token_based_cost: bool = False, models_config_path: str = "configs/models.yaml") -> Dict[str, Any]:
    """
    Load QA data and model performance metrics from JSON files
    
    Args:
        kg_qa_path: Path to QA data file
        result_format: Format string for model result files
        train_shot_num: Number of examples per task for training
        val_shot_num: Number of examples per task for validation
        model_names: List of model names to process
        model_result_dir: Directory containing model result files
        task_hierarchy: Task hierarchy information
        normalize_mode: Normalization mode ("none", "domain", "subtask", "difficulty")
        
    Returns:
        Dictionary containing formatted training and validation data
    """
    va_len = 40
    # Load base QA data
    with open(kg_qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    qa_data = {k: v[:va_len] for k, v in qa_data.items()}
                        
    # Load model performance data using consistent indices
    para_qa_data = {}
    
    # Build task relationship maps
    difficulty_to_subcat = {}
    difficulty_to_domain = {}
    subcat_to_domain = {}
    
    if task_hierarchy:
        for diff_node in task_hierarchy["difficulty_levels"]:
            diff_id = str(diff_node["id"])
            subcat_name = diff_node["subcategory"]
            domain_name = diff_node["domain"]
            
            # Find subcategory ID
            for subcat in task_hierarchy["subcategories"]:
                if subcat["name"] == subcat_name:
                    difficulty_to_subcat[diff_id] = str(subcat["id"])
                    subcat_to_domain[str(subcat["id"])] = str(subcat["parent_id"])
                    break
                    
            # Find domain ID
            for domain in task_hierarchy["domains"]:
                if domain["name"] == domain_name:
                    difficulty_to_domain[diff_id] = str(domain["id"])
                    break

    # First, collect all data for each task
    all_model_data = {}
    for model in model_names:
        result_path = os.path.join(model_result_dir, result_format.format(model=model))
        
        if not os.path.exists(result_path):
            print(f"Model result file not found: {result_path}")
            exit(0)
            
        with open(result_path, "r", encoding="utf-8") as f:
            all_model_data[model] = json.load(f)

    model_prices = _load_model_prices_from_config(models_config_path) if use_token_based_cost else {}
            
    # 收集所有指标用于归一化
    all_metrics = {
        "domain": defaultdict(lambda: {"cost": [], "latency": []}),
        "subtask": defaultdict(lambda: {"cost": [], "latency": []}),
        "difficulty": defaultdict(lambda: {"cost": [], "latency": []})
    }
    
    # 第一遍：收集所有有效的指标值
    all_task_names = qa_data.keys()
    for task_name in all_task_names:
        query_num = len(qa_data[task_name])
        for model in model_names:
            qa_pairs = all_model_data[model][task_name]
            pricing = model_prices.get(model, {}) if use_token_based_cost else {}
            for qa in qa_pairs:
                if qa["cost"] is not None and qa["latency"] is not None:
                    if use_token_based_cost and pricing and "input_tokens" in qa and "output_tokens" in qa:
                        cost_val = calculate_token_cost(
                            qa["input_tokens"], qa["output_tokens"],
                            pricing["input_price"], pricing["output_price"]
                        )
                    else:
                        cost_val = qa["cost"] / 1000
                    latency_val = qa["latency"]
                    
                    if normalize_mode != "none":
                        # 存储到相应层级的指标集合中
                        if normalize_mode in ["domain", "all"] and task_name in difficulty_to_domain:
                            domain_id = difficulty_to_domain[task_name]
                            all_metrics["domain"][domain_id]["cost"].append(cost_val)
                            all_metrics["domain"][domain_id]["latency"].append(latency_val)
                            
                        if normalize_mode in ["subtask", "all"] and task_name in difficulty_to_subcat:
                            subcat_id = difficulty_to_subcat[task_name]
                            all_metrics["subtask"][subcat_id]["cost"].append(cost_val)
                            all_metrics["subtask"][subcat_id]["latency"].append(latency_val)
                            
                        if normalize_mode in ["difficulty", "all"]:
                            all_metrics["difficulty"][task_name]["cost"].append(cost_val)
                            all_metrics["difficulty"][task_name]["latency"].append(latency_val)
    
    # 计算每个层级的最大最小值
    normalization_params = {
        "domain": {},
        "subtask": {},
        "difficulty": {}
    }
    
    if normalize_mode != "none":
        for level in ["domain", "subtask", "difficulty"]:
            for id_, metrics in all_metrics[level].items():
                normalization_params[level][id_] = {
                    "cost_min": min(metrics["cost"]),
                    "cost_max": max(metrics["cost"]),
                    "latency_min": min(metrics["latency"]),
                    "latency_max": max(metrics["latency"])
                }
    
    # 第二遍：处理数据并应用归一化
    train_data = {
        'queries': [],
        'difficulty_ids': [],
        'domain_ids': [],
        'subtask_ids': [],
        'cost': [],
        'effect': [],
        'latency': []
    }
    
    val_data = {
        'queries': [],
        'difficulty_ids': [],
        'domain_ids': [],
        'subtask_ids': [],
        'cost': [],
        'effect': [],
        'latency': []
    }
    print(f"Loading data with normalize mode: {normalize_mode}")
    for task_name in all_task_names:
        query_num = len(qa_data[task_name])
        # print(query_num)
        queryset_task = [qa[0] for qa in qa_data[task_name]]
        difficulty_ids_task = [task_name for _ in qa_data[task_name]]
        
        # 收集每个模型的数据
        all_costs = []
        all_effectivenesses = []
        all_latencies = []
        all_masks = []
        
        for model in model_names:
            qa_pairs = all_model_data[model][task_name][:va_len]
            # qa_pairs = all_model_data[model][task_name]
            # print(len(qa_pairs))
            # print(query_num)
            assert len(qa_pairs) == query_num
            mask_org = []
            cost = []
            effectiveness = []
            latency = []
            pricing = model_prices.get(model, {}) if use_token_based_cost else {}
            
            for i, qa in enumerate(qa_pairs):
                # 处理无效值
                if qa["effectiveness"] is None:
                    qa["effectiveness"] = 0
                if qa["cost"] is None or qa["latency"] is None:
                    qa["cost"] = 0
                    qa["latency"] = 0
                    mask_org.append(0)
                else:
                    mask_org.append(1)
                
                if use_token_based_cost and pricing and "input_tokens" in qa and "output_tokens" in qa:
                    cost_val = calculate_token_cost(
                        qa["input_tokens"], qa["output_tokens"],
                        pricing["input_price"], pricing["output_price"]
                    )
                else:
                    cost_val = qa["cost"] / 1000
                latency_val = qa["latency"]
                
                # 应用归一化
                if normalize_mode != "none" and mask_org[-1] == 1:
                    if normalize_mode == "domain" and task_name in difficulty_to_domain:
                        domain_id = difficulty_to_domain[task_name]
                        params = normalization_params["domain"][domain_id]
                        cost_val = (cost_val - params["cost_min"]) / (params["cost_max"] - params["cost_min"])
                        latency_val = (latency_val - params["latency_min"]) / (params["latency_max"] - params["latency_min"])
                    elif normalize_mode == "subtask" and task_name in difficulty_to_subcat:
                        subcat_id = difficulty_to_subcat[task_name]
                        params = normalization_params["subtask"][subcat_id]
                        cost_val = (cost_val - params["cost_min"]) / (params["cost_max"] - params["cost_min"])
                        latency_val = (latency_val - params["latency_min"]) / (params["latency_max"] - params["latency_min"])
                    elif normalize_mode == "difficulty":
                        params = normalization_params["difficulty"][task_name]
                        cost_val = (cost_val - params["cost_min"]) / (params["cost_max"] - params["cost_min"])
                        latency_val = (latency_val - params["latency_min"]) / (params["latency_max"] - params["latency_min"])
                
                cost.append(cost_val)
                effectiveness.append(qa["effectiveness"])
                latency.append(latency_val)
                
            all_costs.append(torch.tensor(cost, dtype=torch.float32))
            all_effectivenesses.append(torch.tensor(effectiveness, dtype=torch.float32  ))
            all_latencies.append(torch.tensor(latency, dtype=torch.float32))
            all_masks.append(torch.tensor(mask_org, dtype=torch.float32))
            
        all_costs = torch.stack(all_costs, dim=0)  # shape: (n_models, n_examples)
        all_effectivenesses = torch.stack(all_effectivenesses, dim=0)
        all_latencies = torch.stack(all_latencies, dim=0)
        all_masks = torch.stack(all_masks, dim=0)
        
        # 移除mask中有0的样本
        valid_indices = torch.all(all_masks == 1, dim=0)
        if torch.any(valid_indices):
            all_costs = all_costs[:, valid_indices]
            all_effectivenesses = all_effectivenesses[:, valid_indices]
            all_latencies = all_latencies[:, valid_indices]
            all_masks = all_masks[:, valid_indices]
            valid_queries = [q for i, q in enumerate(queryset_task) if valid_indices[i]]
            valid_difficulty_ids = [d for i, d in enumerate(difficulty_ids_task) if valid_indices[i]]
            
            # 随机划分训练集和验证集
            n_valid = len(valid_queries)
            n_train = min(train_shot_num, n_valid)
            n_val = min(val_shot_num, n_valid - n_train)
            
            if n_train > 0:
                indices = torch.randperm(n_valid)
                train_indices = indices[:n_train]
                val_indices = indices[n_train:n_train + n_val]
                
                # 添加到训练集
                for idx in train_indices:
                    difficulty_id = valid_difficulty_ids[idx]
                    domain_id = difficulty_to_domain[difficulty_id]
                    subtask_id = difficulty_to_subcat[difficulty_id]
                    
                    train_data['queries'].append(valid_queries[idx])
                    train_data['difficulty_ids'].append(difficulty_id)
                    train_data['domain_ids'].append(domain_id)
                    train_data['subtask_ids'].append(subtask_id)
                    train_data['cost'].append(all_costs[:, idx].tolist())
                    train_data['effect'].append(all_effectivenesses[:, idx].tolist())
                    train_data['latency'].append(all_latencies[:, idx].tolist())
                
                # 添加到验证集
                for idx in val_indices:
                    difficulty_id = valid_difficulty_ids[idx]
                    domain_id = difficulty_to_domain[difficulty_id]
                    subtask_id = difficulty_to_subcat[difficulty_id]
                    
                    val_data['queries'].append(valid_queries[idx])
                    val_data['difficulty_ids'].append(difficulty_id)
                    val_data['domain_ids'].append(domain_id)
                    val_data['subtask_ids'].append(subtask_id)
                    val_data['cost'].append(all_costs[:, idx].tolist())
                    val_data['effect'].append(all_effectivenesses[:, idx].tolist())
                    val_data['latency'].append(all_latencies[:, idx].tolist())
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'normalization_params': normalization_params if normalize_mode != "none" else None
    }
    # 


            # train_formatted = {
            #     'queries': train_data['query'].tolist(),
            #     'difficulty_ids': train_data['difficulty_id'].tolist(),
            #     'cost': train_data['normalized_cost'].tolist(),
            #     'effect': train_data['effect'].tolist(),
            #     'latency': train_data['normalized_latency'].tolist()
            # }