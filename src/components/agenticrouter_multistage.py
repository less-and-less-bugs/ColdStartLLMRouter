"""
AgenticRouter - 智能LLM路由器
基于效用函数优化模型选择，支持多数据集测试和性能评估
"""
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from collections import defaultdict
import time

from src.components.estimator_model_multistage import EstimatorModel
from src.utils.logger import LLMLogger
from src.utils.data_loader import DatasetGen, load_kg_qa_data, normalize_metric


# MODEL_NAMES = [
#         "qwen3-0.6b",
#         "qwen3-1.7b", 
#         "qwen3-8b",
#         "qwen3-14b",
#         "qwen3-32b",
#         "qwen3-235b-a22b"
#     ]

# MODEL_NAMES = [
#     "gemini-2.5-flash-lite", # 这两个还没跑呢
#     "gemini-2.5-flash",
#     "gemini-2.0-flash",
#     "gemini-2.0-flash-lite-preview-02-05",
#     "doubao-seed-1.6-flash",
# ]

MODEL_NAMES = [
       "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
    "doubao-seed-1.6-flash",
]

METRICS = ["cost", "effectiveness", "latency"]

TASK_NAMES = ["alpaca_data", "GSM8K", "multi_news", "SQUAD"]
# TASK_NAMES = ["mbpp"]
# TASK_NAMES = ["wmt","mbpp","legalbench","medmcqa"]

@dataclass
class RoutingConstraints:
    """路由约束条件"""
    max_latency: Optional[float] = None      # L_max: 最大延迟限制
    min_performance: Optional[float] = None  # P_min: 最小性能要求  
    max_cost: Optional[float] = None         # C_max: 最大成本限制




class AgenticRouter:
    """
    智能Agent路由器 - 基于效用函数优化的LLM选择器
    """
    
    def __init__(self, 
                 kg_path: str = "kg_data/kg_data.json",
                 model_result_dir: str = "kg_data", 
                 data_dir: str = "data",
                 method: str = "desc_sim",
                 qa_data_path: str = "kg_data/generated_qa_difficulty_nodes.json",
                 task_names: List[str] = None,
                 model_names: List[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 metrics: List[str] = ["cost", "effectiveness", "latency"],
                 stage="cold start",
                 ptq_mode: str = "softmax",
                 pcqm_mode: str = "mean",
                 sim_saved_model_path: str = None,
                 prompt_tuning_save_dir: str = None,
                 query_task_type_path: str = None,
                 use_variational: bool = False,
                 use_synthetic_data: bool = False,
                 shot_num_train: int = 30,
                 shot_num_val: int = 10,
                 update_mode: str = "replace",
                 normalize_mode: str = "difficulty"): # "none", "domain", "subtask", "difficulty", "all"
        """
        初始化AgenticRouter
        
        Args:
            kg_path: 知识图谱路径
            model_result_dir: 模型结果目录
            data_dir: 数据目录
            method: 任务概率估计方法
            device: 计算设备
        """
        self.logger = LLMLogger()
        self.device = device
        self.method = method
        self.task_names = task_names
        self.model_names = model_names
        self.shot_num_train = shot_num_train
        self.shot_num_val = shot_num_val
        self.update_mode = update_mode
        self.normalize_mode = normalize_mode # "none", "domain", "subtask", "difficulty", "all"
        self.metrics = metrics
        self.stage = stage # "cold start", "few-shot", "continal setting"
        self.prompt_tuning_save_dir = prompt_tuning_save_dir
        self.use_variational = use_variational
        self.use_synthetic_data = use_synthetic_data
        self.qa_data_path = qa_data_path
        self.model_result_dir = model_result_dir
        self.kg_path = kg_path
        self.data_loader = DatasetGen(data_dir=data_dir, task_names=self.task_names, model_names=self.model_names, query_task_type_path=query_task_type_path)
        # 加载模型信息和价格
        self.train_data, self.val_data, self.test_data = self.data_loader.load_data()
        # print(len(self.train_data))
        # print(len(self.val_data))
        # print(len(self.test_data))
        # exit()
        # self.index_train_data, self.index_val_data, self.index_test_data = self.data_loader.get_task_split_indices(self.train_data, self.val_data, self.test_data)
        self.data_loader.get_max_min_values_per_task(self.train_data, self.val_data, self.test_data) # complete the normization for each task

    
        # self.logger.info(f"Train data: {len(self.train_data)}")
        # self.logger.info(f"Val data: {len(self.val_data)}")
        # self.logger.info(f"Test data: {len(self.test_data)}")

        self.estimator = EstimatorModel(
            kg_path=kg_path,
            model_result_dir=model_result_dir,
            qa_data_path=qa_data_path,
            method=method,
            device=device,
            metrics=metrics,
            model_names=self.model_names,
            ptq_mode=ptq_mode,
            pcqm_mode=pcqm_mode,
            max_values=None,
            min_values=None,
            sim_saved_model_path=sim_saved_model_path,
            use_variational=use_variational
        )
        
    def route_query(self, 
                   metric: torch.Tensor,
                   weights: torch.Tensor = torch.tensor([1/3, 1/3, 1/3]),
                   constraints: RoutingConstraints = None):
        """
        核心路由方法 - 为查询选择最优模型
        
        Args:
            metric: 输入查询在多个模型的指标，shape: (b, n_models, 3)，where b is the batch size.
            weights: 效用函数权重
            constraints: 约束条件.
            max_values: 最大值，shape: (3,)
            min_values: 最小值，shape: (3,)
            
        Returns:
            RoutingResult: 路由结果
        """     
        # 确保 metric 是三维张量
        if len(metric.shape) == 2:
            metric = metric.unsqueeze(0)
        
        batch_size = metric.shape[0]
        n_models = metric.shape[1]
        

        # 计算效用，使用矩阵运算
        utility = torch.sum(metric * weights.unsqueeze(0).unsqueeze(0), dim=2) # shape: (b, n_models)
        # print(utility.shape)
        # 检查约束条件
        if constraints is not None:
            satisfies_constraints = torch.ones(batch_size, n_models, dtype=torch.bool)
            pass # todo
        else:
            satisfies_idx = utility.argmax(dim=1)  # shape: (b, )            
            # 通过 satisfies_idx 获取选中模型的 utility 和 metric
            pred_utilities = utility[torch.arange(batch_size), satisfies_idx]  # shape: (b, )
            pred_metrics = metric[torch.arange(batch_size), satisfies_idx, :]  # shape: (b, 3)
            
        
        return satisfies_idx, pred_utilities, pred_metrics
        
    def sample_evaluate(self, 
                      data_points: Union[Dict, List[Dict]],
                      weights: torch.Tensor = torch.tensor([1/3, 1/3, 1/3]),
                      constraints: RoutingConstraints = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估单个样本或批量样本的性能
        
        Args:
            data_points: 单个数据点或数据点列表
            weights: 效用函数权重
            constraints: 约束条件
            
        Returns:
            Tuple containing satisfies_idx, real_utility, real_metric, pred_utility, pred_metric
        """
        if isinstance(data_points, dict):
            data_points = [data_points]
        
        batch_size = len(data_points)
        pctm = self.estimator.estimate_pctm("mean")
        pctm_tensor = torch.from_numpy(pctm).to(self.estimator.device)
        
        queries = [dp['query'] for dp in data_points]
        batch_metrics = []
        batch_real_stats = []
        
        for dp in data_points:
            cost = dp['cost']  # (n_models, )
            effectiveness = dp['effectiveness']  # (n_models, )
            latency = dp['latency']  # (n_models, )
            stat = torch.tensor([cost, effectiveness, latency])  # shape: (n_models, 3)
            batch_real_stats.append(stat)
            metrics = self.estimator.estimate_pcqm(dp['query'], topk=3, estimated_pctm=pctm_tensor)  # shape: (1, n_models, 3)
            batch_metrics.append(metrics.squeeze(0))
        
        batch_metrics_tensor = torch.stack(batch_metrics)  # shape: (batch_size, n_models, 3)
        batch_real_stats_tensor = torch.stack(batch_real_stats)  # shape: (batch_size, n_models, 3)
        
        satisfies_idx, pred_utilities, pred_metrics = self.route_query(
            metric=batch_metrics_tensor,
            weights=weights,
            constraints=constraints
        )  # shape: satisfies_idx(batch_size,), pred_utilities(batch_size,), pred_metrics(batch_size, 3)
        
        real_metric = batch_real_stats_tensor[torch.arange(batch_size), satisfies_idx, :]  # shape: (batch_size, 3)
        real_utility = torch.sum(real_metric * weights, dim=1)  # shape: (batch_size,)
        
        return satisfies_idx, real_utility, real_metric, pred_utilities, pred_metrics
                


    def update_embeddings_with_few_shot(self, 
                                    shot_num_train: int = 50, 
                                    shot_num_test: int = None,
                                    update_mode: str = "replace",  # "replace" or "stack"
                                    update_routing_params: bool = True,
                                    tuning_mode: str = "lora",  # "lora", "full", "prompt", "prefix", or "p-tuning"
                                    use_grad_for_train: bool = True,
                                    tuning_config: Optional[Dict[str, Any]] = None) -> None:
        """
        通过 few-shot tuning 更新 difficulty embeddings
        
        Args:
            shot_num_train: 训练集每个任务采样的样本数量
            shot_num_test: 测试集每个任务采样的样本数量
            update_mode: 更新模式，"replace"表示替换原有数据，"stack"表示堆叠新旧数据
            update_routing_params: 是否更新路由相关参数(mu, sigma等)
        """
        if self.use_synthetic_data:
            # 使用合成数据进行训练
            data_synthetic = load_kg_qa_data(kg_qa_path=self.qa_data_path, result_format="{model}.json", 
            train_shot_num=self.shot_num_train, val_shot_num=self.shot_num_val, model_names=self.model_names, model_result_dir= self.model_result_dir, task_hierarchy = self.estimator.task_hierarchy, normalize_mode = self.normalize_mode)
            
            train_formatted = data_synthetic['train_data']
            val_formatted = data_synthetic['val_data']
            test_formatted = data_synthetic['val_data']
            print(f"Using {len(train_formatted['queries'])} synthetic training examples and {len(val_formatted['queries'])} synthetic validation examples")
        else:
            # 使用真实数据进行训练
            train_data = self.data_loader.sample_data(self.train_data, shot_num=shot_num_train, split="train")
            val_data = self.data_loader.sample_data(self.val_data, shot_num=shot_num_test, split="val")

            test_data = self.data_loader.sample_data(self.test_data, shot_num=shot_num_test, split="test")
            
            difficulty_to_subcat = {}
            difficulty_to_domain = {}
            subcat_to_domain = {}
            task_hierarchy = self.estimator.task_hierarchy
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


                self.logger.info(f"Sampled {len(train_data)} training examples and {len(test_data)} test examples and {len(val_data)} val examples")
                
                # 准备训练数据
                train_formatted = {
                    'queries': train_data['query'].tolist(),
                    'difficulty_ids': train_data['difficulty_id'].tolist(),
                    'domain_ids': [difficulty_to_domain[diff_id] for diff_id in train_data['difficulty_id'].tolist()],
                    'subtask_ids': [difficulty_to_subcat[diff_id] for diff_id in train_data['difficulty_id'].tolist()],
                    'cost': train_data['normalized_cost'].tolist(),
                    'effect': train_data['effect'].tolist(),
                    'latency': train_data['normalized_latency'].tolist()
                }

                val_formatted = {
                    'queries': val_data['query'].tolist(),
                    'difficulty_ids': val_data['difficulty_id'].tolist(),
                    'domain_ids': [difficulty_to_domain[diff_id] for diff_id in val_data['difficulty_id'].tolist()],
                    'subtask_ids': [difficulty_to_subcat[diff_id] for diff_id in val_data['difficulty_id'].tolist()],
                    'cost': val_data['normalized_cost'].tolist(),
                    'effect': val_data['effect'].tolist(),
                    'latency': val_data['normalized_latency'].tolist()
                }
            
                # 准备测试数据
                test_formatted = {
                    'queries': test_data['query'].tolist(),
                    'difficulty_ids': test_data['difficulty_id'].tolist(),
                    'domain_ids': [difficulty_to_domain[diff_id] for diff_id in test_data['difficulty_id'].tolist()],
                    'subtask_ids': [difficulty_to_subcat[diff_id] for diff_id in test_data['difficulty_id'].tolist()],
                    'cost': test_data['normalized_cost'].tolist(),
                    'effect': test_data['effect'].tolist(),
                    'latency': test_data['normalized_latency'].tolist()
                }

        # 调用 estimator 的 few-shot tuning 方法
        updated_embeddings = self.estimator.few_shot_tuning_sentence_bert(
            train_data=train_formatted,
            test_data=test_formatted,
            val_data=val_formatted,
            save_dir=self.prompt_tuning_save_dir,
            tuning_mode=tuning_mode,
            use_grad_for_train=use_grad_for_train,
            tuning_config=tuning_config
        )
        
        if not self.use_variational:
            # Update difficulty embeddings
            self.estimator.difficulty_embeddings = {
                str(i): emb for i, emb in enumerate(updated_embeddings)
            }
            # 更新 task hierarchy representation
            if self.method == "desc_sim":
                self.estimator.task_hierarchy_representation["embeddings"] = updated_embeddings.to(self.device)

                
                
            # 更新 difficulty example embeddings
            for task_id in train_data['difficulty_id'].unique():
                task_queries = train_data[train_data['difficulty_id'] == task_id]['query'].tolist()
                if task_queries:
                    query_emb = self.estimator.sim_tool.encode_batch(task_queries).to(self.device)
                    if update_mode == "replace":
                        self.estimator.difficulty_example_embeddings[str(task_id)] = query_emb
                    else:  # stack mode
                        if str(task_id) in self.estimator.difficulty_example_embeddings:
                            old_emb = self.estimator.difficulty_example_embeddings[str(task_id)]
                            self.estimator.difficulty_example_embeddings[str(task_id)] = torch.cat([old_emb, query_emb], dim=0)
                        else:
                            self.estimator.difficulty_example_embeddings[str(task_id)] = query_emb

            # 更新路由参数
            if update_routing_params:
                # 更新 para_qa_data
                for task_id in train_data['difficulty_id'].unique():
                    task_data = train_data[train_data['difficulty_id'] == task_id]
                    if len(task_data) > 0:
                        task_id_str = str(task_id)
                        
                        # 准备新的指标数据
                        new_cost = torch.tensor(task_data['normalized_cost'].tolist()).T.to(self.device)
                        new_effectiveness = torch.tensor(task_data['effect'].tolist()).T.to(self.device)
                        new_latency = torch.tensor(task_data['latency'].tolist()).T.to(self.device)
                        new_mask = torch.ones_like(new_cost).to(self.device)
                        # print(new_cost.shape)
                        # print(new_effectiveness.shape)
                        # print(new_latency.shape)
                        # print(new_mask.shape)
                        
                        if update_mode == "replace":
                            self.estimator.para_qa_data[task_id_str] = {
                                'cost': new_cost,
                                'effectiveness': new_effectiveness,
                                'latency': new_latency,
                                'mask': new_mask,
                                'qa_pairs': task_data['query'].tolist()
                            }
                        else:  # stack mode
                            if task_id_str in self.estimator.para_qa_data:
                                old_data = self.estimator.para_qa_data[task_id_str]
                                self.estimator.para_qa_data[task_id_str] = {
                                    'cost': torch.cat([old_data['cost'], new_cost], dim=0),
                                    'effectiveness': torch.cat([old_data['effectiveness'], new_effectiveness], dim=0),
                                    'latency': torch.cat([old_data['latency'], new_latency], dim=0),
                                    'mask': torch.cat([old_data['mask'], new_mask], dim=0),
                                    'qa_pairs': old_data['qa_pairs'] + task_data['query'].tolist()
                                }
                            else:
                                self.estimator.para_qa_data[task_id_str] = {
                                    'cost': new_cost,
                                    'effectiveness': new_effectiveness,
                                    'latency': new_latency,
                                    'mask': new_mask,
                                    'qa_pairs': task_data['query'].tolist()
                                }
                
                # 重新初始化路由参数
                self.estimator._initialize_parameters()
            
        self.logger.info("Successfully updated difficulty embeddings and routing parameters through few-shot tuning")

    def evaluate_model(self, 
                      test_data: pd.DataFrame,
                      weights: torch.Tensor = torch.tensor([-1/3, 1/3, -1/3]),
                      constraints: RoutingConstraints = None,
                      max_samples: int = None, 
                      batch_size: int = 10,
                      enable_few_shot: bool = False,
                      few_shot_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            test_data: 测试数据
            weights: 效用函数权重
            constraints: 路由约束条件
            max_samples: 最大样本数
            batch_size: 批处理大小
            enable_few_shot: 是否启用 few-shot tuning
            few_shot_config: few-shot tuning 的配置，可包含以下参数:
                - shot_num_train: int = 50  # 训练样本数
                - shot_num_test: int = None  # 测试样本数
                - update_mode: str = "replace"  # "replace" 或 "stack"
                - update_routing_params: bool = True  # 是否更新路由参数
                - tuning_mode: str = "lora"  # "lora", "full", 或 "prompt"
                - use_base_model_for_eval: bool = False  # 是否使用基础模型进行评估
                - use_grad_for_train: bool = True  # 是否在训练中使用梯度
                - lora_config: Optional[Dict] = None  # LoRA 配置
        """
        """
        Evaluate the model on test data
        
        Args:
            test_data: DataFrame containing test data with columns 'query', 'cost', 'effectiveness', 'latency'
            weights: Tensor of weights for metrics (cost, effectiveness, latency)
            constraints: Routing constraints object
            max_samples: Maximum number of samples to evaluate. If None, evaluate all samples.
            batch_size: Size of each batch for processing
        
        Returns:
            Dictionary containing evaluation results
        """
        if max_samples is not None:
            test_data = test_data.head(max_samples)
        
        total_samples = len(test_data)
        print(f"Evaluating on {total_samples} samples with batch size {batch_size}")
        self.estimator._initialize_parameters()
        # 如果启用 few-shot tuning
        if enable_few_shot:
            # 设置默认配置
            default_config = {
                "shot_num_train": 50,
                "shot_num_test": None,
                "update_mode": "replace",
                "update_routing_params": True,
                "tuning_mode": "lora",
                "use_grad_for_train": True,
                "tuning_config": None
            }
            
            # 合并用户配置
            if few_shot_config is None:
                few_shot_config = {}
            config = {**default_config, **few_shot_config}
            
            # 执行 few-shot tuning
            self.logger.info("Starting few-shot tuning before evaluation...")
            self.update_embeddings_with_few_shot(
                shot_num_train=config["shot_num_train"],
                shot_num_test=config["shot_num_test"],
                update_mode=config["update_mode"],
                update_routing_params=config["update_routing_params"],
                tuning_mode=config["tuning_mode"],
                use_grad_for_train=config["use_grad_for_train"],
                tuning_config=config["tuning_config"]
            )
            self.logger.info("Variational Few-shot tuning completed")
   
        
        # 定义归一化函数


        all_satisfies_idx = []
        all_real_utility = []
        all_real_metric = []
        all_pred_utility = []
        all_pred_metric = []
        all_no_norm_cost = []
        for i in range(0, total_samples, batch_size):
            batch_data = test_data.iloc[i:i+batch_size]
            batch_queries = batch_data['query'].tolist()
            batch_real_stats = []
            batch_no_norm_cost = []
            for idx, data_point in batch_data.iterrows():
                cost = data_point['normalized_cost']  # shape: (n_models, )
                effectiveness = data_point['effect']  # shape: (n_models, )
                latency = data_point['latency']  # shape: (n_models, )
                stat = torch.tensor([cost, effectiveness, latency], device=self.estimator.device)  # shape: (n_metrics, n_models)
                batch_real_stats.append(stat)
                batch_no_norm_cost.append(torch.tensor(data_point['cost'])) # 
            
            batch_real_stats_tensor = torch.stack(batch_real_stats, dim=0)  # shape: (batch_size, n_metrics, n_models)
            batch_no_norm_cost_tensor = torch.stack(batch_no_norm_cost, dim=0).to(self.estimator.device)  # shape: (batch_size, n_models)
            # Estimate metrics for the entire batch
            batch_metrics = self.estimator.estimate_pcqm(
                query=batch_queries, 
            )  # shape: (batch_size, n_models, n_metrics)
            
            # Route queries to select the best model for each sample in the batch
            satisfies_idx, pred_utilities, pred_metrics = self.route_query(
                metric=batch_metrics,
                weights=weights,
                constraints=constraints,
            )  # shape: satisfies_idx(batch_size,), pred_utilities(batch_size,), pred_metrics(batch_size, n_metrics)
            
            batch_size_actual = batch_metrics.shape[0]
            
            # Select real metrics for the chosen models using satisfies_idx
            real_metric = batch_real_stats_tensor[torch.arange(batch_size_actual, device=self.estimator.device), :, satisfies_idx]  # shape: (batch_size, n_metrics)
            real_no_norm_cost = batch_no_norm_cost_tensor[torch.arange(batch_size_actual, device=self.estimator.device), satisfies_idx] # shape: (batch_size,)

            real_utility = torch.sum(real_metric * weights.to(self.estimator.device), dim=1)  # shape: (batch_size,)
            
            all_satisfies_idx.append(satisfies_idx)
            all_real_utility.append(real_utility)
            all_real_metric.append(real_metric)
            all_pred_utility.append(pred_utilities)
            all_pred_metric.append(pred_metrics)
            all_no_norm_cost.append(real_no_norm_cost)

        # Concatenate all batches
        all_satisfies_idx_tensor = torch.cat(all_satisfies_idx, dim=0)  # shape: (total_samples,)
        all_real_utility_tensor = torch.cat(all_real_utility, dim=0)  # shape: (total_samples,)
        all_real_metric_tensor = torch.cat(all_real_metric, dim=0)  # shape: (total_samples, n_metrics)
        all_pred_utility_tensor = torch.cat(all_pred_utility, dim=0)  # shape: (total_samples,)
        all_pred_metric_tensor = torch.cat(all_pred_metric, dim=0)  # shape: (total_samples, n_metrics)
        
        # Calculate average metrics across all samples
        avg_real_metric = torch.mean(all_real_metric_tensor, dim=0)  # shape: (n_metrics,)
        avg_pred_metric = torch.mean(all_pred_metric_tensor, dim=0)  # shape: (n_metrics,)
        avg_real_utility = torch.mean(all_real_utility_tensor, dim=0)  # shape: ()
        avg_pred_utility = torch.mean(all_pred_utility_tensor, dim=0)  # shape: ()
        sum_no_norm_cost = torch.sum(torch.cat(all_no_norm_cost, dim=0)) # shape: (n_models,)


        # Count how often each model is selected as the best
        model_selection_counts = torch.bincount(all_satisfies_idx_tensor, minlength=len(self.model_names))
        model_selection_ratios = model_selection_counts.float() / total_samples
        
        # Prepare results dictionary
        results = {
            "avg_real_metric": {
                "cost": avg_real_metric[0].item(),
                "effectiveness": avg_real_metric[1].item(),
                "latency": avg_real_metric[2].item(),
            },
            "avg_pred_metric": {
                "cost": avg_pred_metric[0].item(),
                "effectiveness": avg_pred_metric[1].item(),
                "latency": avg_pred_metric[2].item()
            },
            "total_samples": total_samples,
            "sum_no_norm_cost": sum_no_norm_cost.item(),
            "avg_real_utility": avg_real_utility.item(),
            "avg_pred_utility": avg_pred_utility.item(),
            "model_selection_ratios": dict(zip(self.model_names, model_selection_ratios.tolist())),
            # "max_values": self.max_values,
            # "min_values": self.min_values
        }
        
        return results
        
    def evaluate_model_variational(self, 
                      test_data: pd.DataFrame,
                      weights: torch.Tensor = torch.tensor([-1/3, 1/3, -1/3]),
                      constraints: RoutingConstraints = None,
                      max_samples: int = None, 
                      batch_size: int = 10,
                      enable_few_shot: bool = False,
                      few_shot_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            test_data: 测试数据
            weights: 效用函数权重
            constraints: 路由约束条件
            max_samples: 最大样本数
            batch_size: 批处理大小
            enable_few_shot: 是否启用 few-shot tuning
            few_shot_config: few-shot tuning 的配置，可包含以下参数:
                - shot_num_train: int = 50  # 训练样本数
                - shot_num_test: int = None  # 测试样本数
                - update_mode: str = "replace"  # "replace" 或 "stack"
                - update_routing_params: bool = True  # 是否更新路由参数
                - tuning_mode: str = "lora"  # "lora", "full", 或 "prompt"
                - use_base_model_for_eval: bool = False  # 是否使用基础模型进行评估
                - use_grad_for_train: bool = True  # 是否在训练中使用梯度
                - lora_config: Optional[Dict] = None  # LoRA 配置
        """
        """
        Evaluate the model on test data
        
        Args:
            test_data: DataFrame containing test data with columns 'query', 'cost', 'effectiveness', 'latency'
            weights: Tensor of weights for metrics (cost, effectiveness, latency)
            constraints: Routing constraints object
            max_samples: Maximum number of samples to evaluate. If None, evaluate all samples.
            batch_size: Size of each batch for processing
        
        Returns:
            Dictionary containing evaluation results
        """
        if max_samples is not None:
            test_data = test_data.head(max_samples)
        
        total_samples = len(test_data)
        print(f"Evaluating on {total_samples} samples with batch size {batch_size}")
        self.estimator._initialize_parameters()
        # 如果启用 few-shot tuning
        if enable_few_shot:
            # 设置默认配置
            default_config = {
                "shot_num_train": 50,
                "shot_num_test": None,
                "update_mode": "replace",
                "update_routing_params": True,
                "tuning_mode": "lora",
                "use_grad_for_train": True,
                "tuning_config": None
            }
            
            # 合并用户配置
            if few_shot_config is None:
                few_shot_config = {}
            config = {**default_config, **few_shot_config}
            
            # 执行 few-shot tuning
            self.logger.info("Starting few-shot tuning before evaluation...")
            self.update_embeddings_with_few_shot(
                shot_num_train=config["shot_num_train"],
                shot_num_test=config["shot_num_test"],
                update_mode=config["update_mode"],
                update_routing_params=config["update_routing_params"],
                tuning_mode=config["tuning_mode"],
                use_grad_for_train=config["use_grad_for_train"],
                tuning_config=config["tuning_config"]
            )
            self.logger.info("Few-shot tuning completed")
   
        
        # 定义归一化函数


        all_satisfies_idx = []
        all_real_utility = []
        all_real_metric = []
        all_pred_utility = []
        all_pred_metric = []
        all_no_norm_cost = []
        for i in range(0, total_samples, batch_size):
            batch_data = test_data.iloc[i:i+batch_size]
            batch_queries = batch_data['query'].tolist()
            batch_real_stats = []
            batch_no_norm_cost = []
            for idx, data_point in batch_data.iterrows():
                cost = data_point['normalized_cost']  # shape: (n_models, )
                effectiveness = data_point['effect']  # shape: (n_models, )
                latency = data_point['latency']  # shape: (n_models, )
                stat = torch.tensor([cost, effectiveness, latency], device=self.estimator.device)  # shape: (n_metrics, n_models)
                batch_real_stats.append(stat)
                batch_no_norm_cost.append(torch.tensor(data_point['cost'])) # 
            
            batch_real_stats_tensor = torch.stack(batch_real_stats, dim=0)  # shape: (batch_size, n_metrics, n_models)
            batch_no_norm_cost_tensor = torch.stack(batch_no_norm_cost, dim=0).to(self.estimator.device)  # shape: (batch_size, n_models)
            # Estimate metrics for the entire batch
            batch_metrics = self.estimator.estimate_pcqm_variational(
                query=batch_queries, 
            )  # shape: (batch_size, n_models, n_metrics)
            
            # Route queries to select the best model for each sample in the batch
            satisfies_idx, pred_utilities, pred_metrics = self.route_query(
                metric=batch_metrics,
                weights=weights,
                constraints=constraints,
            )  # shape: satisfies_idx(batch_size,), pred_utilities(batch_size,), pred_metrics(batch_size, n_metrics)
            
            batch_size_actual = batch_metrics.shape[0]
            
            # Select real metrics for the chosen models using satisfies_idx
            real_metric = batch_real_stats_tensor[torch.arange(batch_size_actual, device=self.estimator.device), :, satisfies_idx]  # shape: (batch_size, n_metrics)
            real_no_norm_cost = batch_no_norm_cost_tensor[torch.arange(batch_size_actual, device=self.estimator.device), satisfies_idx] # shape: (batch_size,)

            real_utility = torch.sum(real_metric * weights.to(self.estimator.device), dim=1)  # shape: (batch_size,)
            
            all_satisfies_idx.append(satisfies_idx)
            all_real_utility.append(real_utility)
            all_real_metric.append(real_metric)
            all_pred_utility.append(pred_utilities)
            all_pred_metric.append(pred_metrics)
            all_no_norm_cost.append(real_no_norm_cost)

        # Concatenate all batches
        all_satisfies_idx_tensor = torch.cat(all_satisfies_idx, dim=0)  # shape: (total_samples,)
        all_real_utility_tensor = torch.cat(all_real_utility, dim=0)  # shape: (total_samples,)
        all_real_metric_tensor = torch.cat(all_real_metric, dim=0)  # shape: (total_samples, n_metrics)
        all_pred_utility_tensor = torch.cat(all_pred_utility, dim=0)  # shape: (total_samples,)
        all_pred_metric_tensor = torch.cat(all_pred_metric, dim=0)  # shape: (total_samples, n_metrics)
        
        # Calculate average metrics across all samples
        avg_real_metric = torch.mean(all_real_metric_tensor, dim=0)  # shape: (n_metrics,)
        avg_pred_metric = torch.mean(all_pred_metric_tensor, dim=0)  # shape: (n_metrics,)
        avg_real_utility = torch.mean(all_real_utility_tensor, dim=0)  # shape: ()
        avg_pred_utility = torch.mean(all_pred_utility_tensor, dim=0)  # shape: ()
        sum_no_norm_cost = torch.sum(torch.cat(all_no_norm_cost, dim=0)) # shape: (n_models,)


        # Count how often each model is selected as the best
        model_selection_counts = torch.bincount(all_satisfies_idx_tensor, minlength=len(self.model_names))
        model_selection_ratios = model_selection_counts.float() / total_samples
        
        # Prepare results dictionary
        results = {
            "avg_real_metric": {
                "cost": avg_real_metric[0].item(),
                "effectiveness": avg_real_metric[1].item(),
                "latency": avg_real_metric[2].item(),
            },
            "avg_pred_metric": {
                "cost": avg_pred_metric[0].item(),
                "effectiveness": avg_pred_metric[1].item(),
                "latency": avg_pred_metric[2].item()
            },
            "total_samples": total_samples,
            "sum_no_norm_cost": sum_no_norm_cost.item(),
            "avg_real_utility": avg_real_utility.item(),
            "avg_pred_utility": avg_pred_utility.item(),
            "model_selection_ratios": dict(zip(self.model_names, model_selection_ratios.tolist())),
            # "max_values": self.max_values,
            # "min_values": self.min_values
        }
                
        # Calculate per-task metrics
        per_task_results = {}
        for task_name in self.task_names:
            task_indices = self.data_loader.task_indices["test"][task_name]
            if not task_indices:
                continue
                
            # Convert indices to tensor indices for filtering
            task_idx_tensor = torch.tensor([i for i in range(len(test_data)) if i in task_indices], 
                                         device=self.estimator.device)
            
            # Filter metrics for this task
            task_real_metric = all_real_metric_tensor[task_idx_tensor]
            task_pred_metric = all_pred_metric_tensor[task_idx_tensor]
            task_real_utility = all_real_utility_tensor[task_idx_tensor]
            task_pred_utility = all_pred_utility_tensor[task_idx_tensor]
            task_satisfies_idx = all_satisfies_idx_tensor[task_idx_tensor]
            task_no_norm_cost = torch.cat(all_no_norm_cost, dim=0)[task_idx_tensor]
            
            # Calculate task-specific averages
            task_avg_real_metric = torch.mean(task_real_metric, dim=0)
            task_avg_pred_metric = torch.mean(task_pred_metric, dim=0)
            task_avg_real_utility = torch.mean(task_real_utility)
            task_avg_pred_utility = torch.mean(task_pred_utility)
            task_sum_no_norm_cost = torch.sum(task_no_norm_cost)
            
            # Calculate task-specific model selection ratios
            task_model_counts = torch.bincount(task_satisfies_idx, minlength=len(self.model_names))
            task_model_ratios = task_model_counts.float() / len(task_indices)
            
            per_task_results[task_name] = {
                "avg_real_metric": {
                    "cost": task_avg_real_metric[0].item(),
                    "effectiveness": task_avg_real_metric[1].item(),
                    "latency": task_avg_real_metric[2].item(),
                },
                "avg_pred_metric": {
                    "cost": task_avg_pred_metric[0].item(),
                    "effectiveness": task_avg_pred_metric[1].item(),
                    "latency": task_avg_pred_metric[2].item()
                },
                "total_samples": len(task_indices),
                "sum_no_norm_cost": task_sum_no_norm_cost.item(),
                "avg_real_utility": task_avg_real_utility.item(),
                "avg_pred_utility": task_avg_pred_utility.item(),
                "model_selection_ratios": dict(zip(self.model_names, task_model_ratios.tolist())),
            }
            
        # Add per-task results to the main results dictionary
        results["per_task"] = per_task_results
        
        return results
        return results      

def main():
    """Main function for testing the AgenticRouter"""
    model_names = MODEL_NAMES
    task_names = TASK_NAMES
    method_list = ["desc_sim"]
    # method = "desc_sim"
    result_dir = "supp"
    os.makedirs(result_dir, exist_ok=True)
    # topk = 3
    weights_list  = [torch.tensor([-0.8, 0.2,0]), torch.tensor([-1/2, 1/2,0]), torch.tensor([-0.2, 0.8, 0])]
    pcqm_mode_list = ["mean", "weighted"]
    pcqm_mode = "weighted"
    
    # Few-shot tuning 配置
    few_shot_configs = [
        # LoRA configuration
        # {
        #     "enable_few_shot": True,
        #     "few_shot_config": {
        #         "shot_num_train": 500,
        #         "tuning_mode": "lora",
        #         "update_mode": "replace",
        #         "tuning_config": {
        #             "r": 8,
        #             "lora_alpha": 16
        #         }
        #     }
        # },
        # Variational Inference configuration with real data
        # {
        #     "enable_few_shot": True,
        #     "few_shot_config": {
        #         "shot_num_train":500,
        #         "tuning_mode": "variational",
        #         "update_mode": "replace",
        #         "tuning_config": {
        #             "hidden_dim": 256,
        #             "use_implicit_tasks": False,
        #             "temperature": 0.07,
        #             "kl_weight": 1,
        #             "cost_weight": 1,
        #             "effect_weight": 1,
        #             "metrics": ["cost", "effect"]
        #         }
        #     }
        # },
        # Variational Inference configuration with synthetic data
        {
            "enable_few_shot": True,
            "few_shot_config": {
                "shot_num_train":None,
                "tuning_mode": "variational",
                "update_mode": "replace",
                "tuning_config": {
                    "hidden_dim": 256,
                    "use_implicit_tasks": False,
                    "temperature": 0.07,
                    "kl_weight": 1,
                    "cost_weight": 1,
                    "effect_weight": 1,
                    "metrics": ["cost", "effect"],
                    "lr": 1e-4
                }
            }
        },

        # Prompt Tuning configuration
        # {
        #     "enable_few_shot": True,
        #     "few_shot_config": {
        #         "shot_num_train": 50,
        #         "tuning_mode": "prompt",
        #         "update_mode": "replace",
        #         "tuning_config": {
        #             "num_virtual_tokens": 8,
        #             "prompt_tuning_init_text": "Classify the difficulty level of this task:"
        #         }
        #     }
        # },
        # # Prefix Tuning configuration
        # {
        #     "enable_few_shot": True,
        #     "few_shot_config": {
        #         "shot_num_train": 50,
        #         "tuning_mode": "prefix",
        #         "update_mode": "replace",
        #         "tuning_config": {
        #             "num_virtual_tokens": 8,
        #             "prefix_projection": True
        #         }
        #     }
        # },
        # # P-Tuning configuration
        # {
        #     "enable_few_shot": True,
        #     "few_shot_config": {
        #         "shot_num_train": 50,
        #         "tuning_mode": "p-tuning",
        #         "update_mode": "replace",
        #         "tuning_config": {
        #             "num_virtual_tokens": 8,
        #             "encoder_hidden_size": 128
        #         }
        #     }
        # },
        # # Full parameter tuning
        # {
        #     "enable_few_shot": True,
        #     "few_shot_config": {
        #         "shot_num_train": 50,
        #         "tuning_mode": "full",
        #         "update_mode": "stack"
        #     }
        # },
        # # Baseline model without tuning
        # {
        #     "enable_few_shot": False
        # }
    ]
    
    for method in method_list:
        # for ptq_mode in ["softmax", "argmax"]:
        for ptq_mode in [ "argmax"]:
            for few_shot_setup in few_shot_configs:
                result_dict = {}
                for weights in weights_list:
                    # Check if we're using variational inference
                    use_variational = few_shot_setup.get("few_shot_config", {}).get("tuning_mode") == "variational" if isinstance(few_shot_setup, dict) else False
                    use_synthetic_data = True

                    # "/hdd2/lh/agenticrouter_data/finetune_results/full_finetune_domain
                    # /hdd2/lh/agenticrouter_data/query_task_type_results/classification_results.json
                    router = AgenticRouter(kg_path="/hdd2/lh/agenticrouter_data/kg_data/kg_data.json",
                    model_result_dir="/hdd2/lh/agenticrouter_data/kg_data",
                    data_dir="/hdd2/lh/agenticrouter_data/data", qa_data_path="/hdd2/lh/agenticrouter_data/kg_data/generated_qa_difficulty_nodes_40.json",
                    model_names=model_names, task_names=task_names, 
                    method=method, ptq_mode=ptq_mode, pcqm_mode=pcqm_mode, sim_saved_model_path=None,
                    query_task_type_path="/hdd2/lh/agenticrouter_data/query_task_type_results/classification_results.json",
                    use_variational=use_variational, use_synthetic_data=use_synthetic_data,
                    prompt_tuning_save_dir="/hdd2/lh/agenticrouter_data/finetune_results/variantional_reviwer1_q3", shot_num_train=3, shot_num_val=2)

                    # router = AgenticRouter(kg_path="/hdd2/lh/agenticrouter_data/kg_data_gemini/kg_data.json",
                    # model_result_dir="/hdd2/lh/agenticrouter_data/kg_data_gemini",
                    # data_dir="/hdd2/lh/agenticrouter_data/data", qa_data_path="/hdd2/lh/agenticrouter_data/kg_data_gemini/generated_qa_difficulty_nodes.json",
                    # model_names=model_names, task_names=task_names, 
                    # method=method, ptq_mode=ptq_mode, pcqm_mode=pcqm_mode, sim_saved_model_path=None,
                    # query_task_type_path="/hdd2/lh/agenticrouter_data/query_task_type_results_gemini/classification_results.json",
                    # use_variational=use_variational, use_synthetic_data=use_synthetic_data,
                    # prompt_tuning_save_dir="/hdd2/lh/agenticrouter_data/finetune_results/variantional_gemini", shot_num_train=30, shot_num_val=10)

                    # router = AgenticRouter(kg_path="/hdd2/lh/agenticrouter_data/kg_data_gemini/kg_data.json",
                    # model_result_dir="/hdd2/lh/agenticrouter_data/kg_data_gemini",
                    # data_dir="/hdd2/lh/agenticrouter_data/data", qa_data_path="/hdd2/lh/agenticrouter_data/kg_data_gemini/generated_qa_difficulty_nodes.json",
                    # model_names=model_names, task_names=task_names, 
                    # method=method, ptq_mode=ptq_mode, pcqm_mode=pcqm_mode, sim_saved_model_path=None,
                    # query_task_type_path="/hdd2/lh/agenticrouter_data/query_task_type_results_gemini_wmt/classification_results.json",
                    # use_variational=use_variational, use_synthetic_data=use_synthetic_data,
                    # prompt_tuning_save_dir="/hdd2/lh/agenticrouter_data/finetune_results/variantional_wmt_indomain", shot_num_train=30, shot_num_val=10)

                    # router = AgenticRouter(kg_path="/hdd2/lh/agenticrouter_data/kg_data_supp/kg_data.json",
                    # model_result_dir="/hdd2/lh/agenticrouter_data/kg_data_supp",
                    # data_dir="/hdd2/lh/agenticrouter_data/data", qa_data_path="/hdd2/lh/agenticrouter_data/kg_data_supp/generated_qa_difficulty_nodes.json",
                    # model_names=model_names, task_names=task_names, 
                    # method=method, ptq_mode=ptq_mode, pcqm_mode=pcqm_mode, sim_saved_model_path=None,
                    # query_task_type_path="/hdd2/lh/agenticrouter_data/query_task_type_results_supp/classification_results.json",
                    # use_variational=use_variational, use_synthetic_data=use_synthetic_data,
                    # prompt_tuning_save_dir="/hdd2/lh/agenticrouter_data/finetune_results/variantional_supp", shot_num_train=30, shot_num_val=10)


                    print("=== Testing AgenticRouter ===")
                    print(f"Models: {router.model_names}")
                    print(f"Method: {method}")

                    # 评估模型
                    print("\n--- Evaluating Model on Test Data ---")
                    if use_variational:
                        eval_result = router.evaluate_model_variational(
                            test_data=router.test_data,
                            weights=weights.to(router.estimator.device),
                            max_samples=None,  # 限制样本数以便快速测试
                            batch_size=10,
                            **few_shot_setup  # 添加 few-shot tuning 配置
                        )
                    else:
                        eval_result = router.evaluate_model(
                            test_data=router.test_data,
                                weights=weights.to(router.estimator.device),
                                max_samples=None,  # 限制样本数以便快速测试
                                batch_size=10,
                                **few_shot_setup  # 添加 few-shot tuning 配置
                            )
                

                    print(f"评估完成 - 处理样本数: {eval_result['total_samples']}")
                    print(f"平均真实效用: {eval_result['avg_real_utility']:.4f}")
                    print(f"平均预测效用: {eval_result['avg_pred_utility']:.4f}")
                    print(f"真实指标平均值: {eval_result['avg_real_metric']}")
                    print(f"预测指标平均值: {eval_result['avg_pred_metric']}")
                    print(f"模型分布: {eval_result['model_selection_ratios']}")

                    result_dict[f"weights_{weights.tolist()}"] = eval_result
                    print("\n=== Testing completed ===")
                    
                tuning_mode = few_shot_setup.get("few_shot_config", {}).get("tuning_mode", "base")
                with open(os.path.join(result_dir, f"mbpp_indomain.json"), "w") as f:
                    for key, value in result_dict.items():
                        json.dump({key: value}, f, indent=2)
                        f.write("\n")

if __name__ == "__main__":
    main()
