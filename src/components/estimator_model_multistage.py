"""
Estimator Model for LLM Router
Implements multiple estimation methods for p(t|q), p(c|t,M), p(r|t,M), p(l|t,M)
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from src.utils.similarity import SimilarityTool, SimilarityTool_hugg
from src.utils.logger import LLMLogger
from src.utils.query_type_graph import extract_task_hierarchy
from src.utils.data_loader import DOMAIN_FORMAT, SUBCAT_FORMAT, DIFF_FORMAT
import pandas as pd
import random
from src.utils.data_loader import normalize_metric
from collections import defaultdict
from src.utils.data_loader import process_qa_pairs_and_metrics



from src.components.variational_prompt_tuner import VariationalPromptTuner
from src.components.variational_prompt_tuner_multiple import VariationalPromptTunerMultiple


class EstimatorModel:
    """
    Estimator for task type probability p(t|q) and model performance metrics p(c|t,M), p(r|t,M), p(l|t,M)
    """
    
    def masked_std(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate standard deviation of values considering a mask tensor.
        
        Args:
            values: Input tensor of values
            mask: Tensor of same shape as values with 1s for valid values and 0s for masked values
            
        Returns:
            Standard deviation of unmasked values. Returns 0 if all values are masked.
        """
        # Ensure inputs are tensors
        # Count number of unmasked elements
        valid_count = mask.sum()
        
        # If all elements are masked or only one element, return 0
        if valid_count <= 1:
            return torch.tensor(0.0)
            
        # Calculate mean of unmasked values
        mean = self.masked_mean(values, mask)
        
        # Calculate squared differences from mean
        squared_diff = (values - mean) ** 2
        
        # Apply mask and calculate mean of squared differences
        masked_squared_diff = squared_diff * mask
        variance = masked_squared_diff.sum() / (valid_count - 1)  # Using n-1 for sample standard deviation
        
        return torch.sqrt(variance)

    def masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate mean of values considering a mask tensor.
        
        Args:
            values: Input tensor of values
            mask: Tensor of same shape as values with 1s for valid values and 0s for masked values
            
        Returns:
            Mean of unmasked values. Returns 0 if all values are masked.
        """
        # Ensure inputs are tensors
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float32)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
            
        # Multiply values by mask to zero out masked values
        masked_values = values * mask
        
        # Count number of unmasked elements
        valid_count = mask.sum()
        
        # If all elements are masked, return 0
        if valid_count == 0:
            return torch.tensor(0.0)
            
        # Sum masked values and divide by count of unmasked elements
        return masked_values.sum() / valid_count
    
    def __init__(self, 
                 kg_path: str = "kg_data/kg_data.json", 
                 model_result_dir: str = "kg_data", 
                 model_names: Optional[List[str]] = None,
                 qa_data_path: str = "kg_data/generated_qa_difficulty_nodes.json",
                 method: str = "desc_sim",
                 pctm_method: str = "mean",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 split_ratio: float = 0.5,
                 metrics: Optional[List[str]] = ["cost", "effectiveness", "latency"],
                 ptq_mode: str = "softmax",
                 min_values: Optional[Dict[str, float]] = None,
                 max_values: Optional[Dict[str, float]] = None, sim_saved_model_path: str = None, pcqm_mode: str = "mean", use_variational: bool = False, support_multiple_labels: bool = False):
        """
        Initialize EstimatorModel
        
        Args:
            kg_path: Path to knowledge graph JSON file
            model_result_dir: Directory containing model result files
            model_names: List of model names to consider
            qa_data_path: Path to QA data file
            device: Device to use for torch computations
        """
        self.kg_path = kg_path
        self.model_result_dir = model_result_dir
        self.qa_data_path = qa_data_path
        self.model_names = model_names 
        self.method = method
        self.pctm_method = pctm_method
        self.device = device
        self.split_ratio = split_ratio
        self.metrics = metrics
        self.ptq_mode = ptq_mode
        self.pcqm_mode = pcqm_mode
        self.use_variational = use_variational
        self.support_multiple_labels = support_multiple_labels


        self.sim_tool = SimilarityTool_hugg(model_path = sim_saved_model_path)
        self.logger = LLMLogger()
        self.min_values = min_values
        self.max_values = max_values
        
        # Initialize embedding storage dictionaries
        self.domain_embeddings = {}
        self.subcat_embeddings = {}
        self.difficulty_embeddings = {}
        self.domain_texts = []
        self.subcat_texts = []
        self.difficulty_texts = []
        self.domain_example_embeddings = defaultdict(list)
        self.subcat_example_embeddings = defaultdict(list)
        self.difficulty_example_embeddings = {}
        
        # Load data
        self.kg = self._load_kg(kg_path)
        # Extract task hierarchies
        self.task_hierarchy = extract_task_hierarchy(kg_path)
        # print(len(self.task_hierarchy["domains"]))
        # print(len(self.task_hierarchy["subcategories"]))
        # print(len(self.task_hierarchy["difficulty_levels"]))
        # exit(0)
        self.qa_data = self._load_qa_data(qa_data_path, split_ratio=self.split_ratio)
        self.para_qa_data = self.qa_data["model_metrics"]
        self._precompute_embeddings(self.task_hierarchy, self.para_qa_data)
        # Obtain the representation of the task hierarchy
        self.task_hierarchy_representation = self._load_task_prototype_representation(self.task_hierarchy, method)


    def _precompute_embeddings(self, task_hierarchy: Dict[str, Any], example_data: Dict[str, Any]):
        # 计算基于definition的embeddings
        for domain in self.task_hierarchy["domains"]:
            self.domain_embeddings[str(domain["id"])] = self.sim_tool.encode(DOMAIN_FORMAT.format(domain["name"], domain["definition"])).to(self.device)
            self.domain_texts.append(DOMAIN_FORMAT.format(domain["name"], domain["definition"]))
            
        for subcat in self.task_hierarchy["subcategories"]:
            self.subcat_embeddings[str(subcat["id"])] = self.sim_tool.encode(SUBCAT_FORMAT.format(subcat["name"], self.task_hierarchy["domains"][subcat["parent_id"]]["name"], subcat["definition"])).to(self.device)
            self.subcat_texts.append(SUBCAT_FORMAT.format(subcat["name"], self.task_hierarchy["domains"][subcat["parent_id"]]["name"], subcat["definition"]))
            
        for diff in self.task_hierarchy["difficulty_levels"]:
            self.difficulty_embeddings[str(diff["id"])] = self.sim_tool.encode(DIFF_FORMAT.format(self.task_hierarchy["subcategories"][diff["parent_id"]]["name"], diff["name"], diff["definition"])).to(self.device) 
            # self.difficulty_texts.append(DIFF_FORMAT.format(self.task_hierarchy["subcategories"][diff["parent_id"]]["name"], diff["name"], diff["definition"]))    
            self.difficulty_texts.append(DIFF_FORMAT.format(self.task_hierarchy["subcategories"][diff["parent_id"]]["name"], diff["name"], diff["definition"]))   
            
        # 计算基于example的embeddings
        for task_id, examples in example_data.items():
            task_info = None
            for level in task_hierarchy["difficulty_levels"]:
                if str(level["id"]) == task_id:
                    task_info = level
                    break
                    
            if task_info is None or not examples:
                continue
                
            subcat_id = str(task_info["parent_id"])
            domain_id = str(task_hierarchy["subcategories"][task_info["parent_id"]]["parent_id"])
            q_examples = [query for query in examples["qa_pairs"]]
            query_emb = self.sim_tool.encode_batch(q_examples).to(self.device)
            self.domain_example_embeddings[domain_id].append(query_emb)
            self.subcat_example_embeddings[subcat_id].append(query_emb)
            self.difficulty_example_embeddings[task_id] = query_emb
            
    def _load_task_prototype_representation(self, task_hierarchy: Dict[str, Any], method: str) -> Dict[str, Any]:
        """
        Load and encode task hierarchy representation
        
        Args:
            method: Method for task similarity calculation
            include_answer: Whether to include answers in example encoding for example_sim method
            
        Returns:
            Dictionary containing encoded embeddings and task names
        """
        difficulty_levels = task_hierarchy["difficulty_levels"]
        task_names = [str(i) for i, task in enumerate(difficulty_levels)]
        
        if method == "desc_sim":
            embeddings = torch.stack([self.difficulty_embeddings[task_name] for task_name in task_names], dim=0) # n_task, dim
        elif method == "example_sim":
            # Encode examples from qa_data
            all_examples_embeddings = []
            # First pass: find maximum number of examples
            for task_name in task_names:
                all_examples_embeddings.append(torch.mean(self.difficulty_example_embeddings[task_name], dim=0))
    
            embeddings = F.normalize(torch.stack(all_examples_embeddings, dim=0), dim=1) # n_task, dim
        
        elif method == "example_des_sim":
            # 将definition和example embeddings取平均
            des_embeddings = torch.stack([self.difficulty_embeddings[task_name] for task_name in task_names], dim=0) # n_task, dim
                        # Encode examples from qa_data
            all_examples_embeddings = []
            # First pass: find maximum number of examples
            for task_name in task_names:
                all_examples_embeddings.append(torch.mean(self.difficulty_example_embeddings[task_name], dim=0))
            example_embeddings = torch.stack(all_examples_embeddings, dim=0) # n_task, dim
            embeddings = F.normalize(torch.mean(torch.stack([des_embeddings, example_embeddings]), dim=0), dim=1)  # (2, n_task, dim)
            exit(0)
        else:
            raise ValueError(f"Invalid method: {method}")
            
        return {
            "embeddings": embeddings,
            "task_names": task_names
        }
        
    def _load_kg(self, path: str) -> Dict[str, Any]:
        """Load knowledge graph from JSON file"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load knowledge graph: {e}")
            return {}

    def _load_qa_data(self, path: str, result_format: str = "{model}.json", split_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Load QA data and model performance metrics from JSON files
        
        Args:
            path: Path to QA data file
            result_format: Format string for model result files
            sample_size: Number of QA pairs to sample for each task. If None, use all data
            
        Returns:
            Dictionary containing both QA data and model performance metrics
        """
        # Load base QA data
        with open(path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        qa_data = {k: v[:80] for k, v in qa_data.items()}
            
        task_sample_indices = {}
        task_sample_val_indices = {}
        for task_name in qa_data.keys():
            qa_pairs = qa_data[task_name]
            n_pairs = len(qa_pairs)
            indices = list(range(n_pairs))
            random.shuffle(indices)
            task_sample_indices[task_name] = indices[:int(n_pairs * split_ratio)]
            task_sample_val_indices[task_name] = indices[int(n_pairs * split_ratio):]
        # print(task_sample_indices)
                            
        # Load model performance data using consistent indices
        para_qa_data = {}
        para_qa_data_val = {}
        
        # First, collect all data for each task
        all_model_data = {}
        for model in self.model_names:
            result_path = os.path.join(self.model_result_dir, result_format.format(model=model))
            
            if not os.path.exists(result_path):
                self.logger.warning(f"Model result file not found: {result_path}")
                exit(0)
                
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    all_model_data[model] = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load model data for {model}: {e}")
                continue
        
        # Process data task by task
        all_task_names = set()
        for model_data in all_model_data.values():
            all_task_names.update(model_data.keys())
            
        for task_name in all_task_names:
            # Initialize arrays to store data from all models
            all_costs = []
            all_effectivenesses = []
            all_latencies = []
            all_masks = []
            all_qa_pairs = []
            
            all_costs_val = []
            all_effectivenesses_val = []
            all_latencies_val = []
            all_masks_val = []
            all_qa_pairs_val = []
            
            # Collect data from each model
            for model in self.model_names:
                if task_name not in all_model_data[model]:
                    continue
                    
                qa_pairs = all_model_data[model][task_name]
                
                # Process QA pairs and metrics
                qa_data_processed = process_qa_pairs_and_metrics(
                    task_name,
                    task_sample_indices,
                    task_sample_val_indices,
                    qa_pairs,
                    qa_data
                )
                
                # Training data
                all_costs.append(qa_data_processed["cost"])
                all_effectivenesses.append(qa_data_processed["effectiveness"])
                all_latencies.append(qa_data_processed["latency"])
                all_masks.append(qa_data_processed["mask_org"])
                all_qa_pairs.append(qa_data_processed["qa_pairs_org"])
                
                # Validation data
                all_costs_val.append(qa_data_processed["cost_val"])
                all_effectivenesses_val.append(qa_data_processed["effectiveness_val"])
                all_latencies_val.append(qa_data_processed["latency_val"])
                all_masks_val.append(qa_data_processed["mask_val"])
                all_qa_pairs_val.append(qa_data_processed["qa_pairs_val"])
            # Stack all model data for this task
            para_qa_data[task_name] = {
                "cost": torch.stack(all_costs, dim=0).to(self.device),  # shape: (n_models, n_examples)
                "effectiveness": torch.stack(all_effectivenesses, dim=0).to(self.device),
                "latency": torch.stack(all_latencies, dim=0).to(self.device),
                "mask": torch.stack(all_masks, dim=0).to(self.device), # example对不同模型的不同任务不起作用
                "qa_pairs": all_qa_pairs[0]  # List of lists of QA pairs
            }
            
            para_qa_data_val[task_name] = {
                "cost": torch.stack(all_costs_val, dim=0).to(self.device),
                "effectiveness": torch.stack(all_effectivenesses_val, dim=0).to(self.device),
                "latency": torch.stack(all_latencies_val, dim=0).to(self.device),
                "mask": torch.stack(all_masks_val, dim=0).to(self.device),
                "qa_pairs": all_qa_pairs_val[0]
            }
                
        return {
            "qa_data": qa_data,
            "model_metrics": para_qa_data,
            "sample_indices": task_sample_indices,
            "sample_indices_val": task_sample_val_indices,
            "para_qa_data_val": para_qa_data_val
        }
 
    def _initialize_parameters(self, tau=None, mu_c=None, mu_r=None, mu_l=None, sigma_c=None, sigma_r=None, sigma_l=None, beta=None, k_prototype=None):
        """
        Initialize parameters for router
        """
        self.tau = tau
        self.beta = beta
        self.mu_c = mu_c
        self.mu_r = mu_r
        self.mu_l = mu_l
        self.sigma_c = sigma_c
        self.sigma_r = sigma_r
        self.sigma_l = sigma_l
        self.k_prototype = k_prototype

        if self.tau is None:
            self.tau = 0.07
        if self.beta is None:
            self.beta = 0.2
        self.cost_max = defaultdict(list)
        self.cost_min = defaultdict(list)
        self.latency_max = defaultdict(list)
        self.latency_min = defaultdict(list)

        # Initialize tensors for masked computation
        for task_name in self.para_qa_data.keys():
            task_data = self.para_qa_data[task_name]
            
            # Get costs, latencies and masks for this task
            # shape: (n_models, n_examples)
            costs = task_data["cost"]
            latencies = task_data["latency"]
            masks = task_data["mask"]
            
            # Flatten for computing global min/max
            flat_costs = costs.reshape(-1)
            flat_latencies = latencies.reshape(-1)
            flat_masks = masks.reshape(-1)
            
            # Apply mask (multiply by mask to make masked values 0)
            masked_costs = flat_costs * flat_masks
            masked_latencies = flat_latencies * flat_masks
            
            # Replace 0s (masked values) with negative infinity for min and positive infinity for max
            masked_costs_for_max = torch.where(flat_masks == 1, masked_costs, float('-inf'))
            masked_costs_for_min = torch.where(flat_masks == 1, masked_costs, float('inf'))
            masked_latencies_for_max = torch.where(flat_masks == 1, masked_latencies, float('-inf'))
            masked_latencies_for_min = torch.where(flat_masks == 1, masked_latencies, float('inf'))
            
            # Compute max and min only for unmasked values
            self.cost_max[task_name] = masked_costs_for_max.max().item()
            self.cost_min[task_name] = masked_costs_for_min.min().item()
            self.latency_max[task_name] = masked_latencies_for_max.max().item()
            self.latency_min[task_name] = masked_latencies_for_min.min().item()
            
        if self.mu_c is None:
            self.mu_c = []
            for task_name in self.para_qa_data.keys():
                # Normalize costs and compute mean for each model
                costs = self.para_qa_data[task_name]["cost"]  # (n_models, n_examples)
                masks = self.para_qa_data[task_name]["mask"]  # (n_models, n_examples)
                # normalized_costs = normalize_metric(costs, self.cost_min[task_name], self.cost_max[task_name], masks)
                normalized_costs = costs
                model_means = torch.stack([self.masked_mean(normalized_costs[i], masks[i]) for i in range(len(self.model_names))])
                self.mu_c.append(model_means)
            self.mu_c = torch.stack(self.mu_c, dim=1).to(self.device, dtype=torch.float32)  # (n_models, n_tasks)
        
            
        if self.mu_r is None:
            self.mu_r = []
            for task_name in self.para_qa_data.keys():
                effectiveness = self.para_qa_data[task_name]["effectiveness"]  # (n_models, n_examples)
                masks = self.para_qa_data[task_name]["mask"]  # (n_models, n_examples)
                model_means = torch.stack([self.masked_mean(effectiveness[i], masks[i]) for i in range(len(self.model_names))])
                self.mu_r.append(model_means)
            self.mu_r = torch.stack(self.mu_r, dim=1).to(self.device, dtype=torch.float32)  # (n_models, n_tasks)
            
        if self.mu_l is None:
            self.mu_l = []
            for task_name in self.para_qa_data.keys():
                latencies = self.para_qa_data[task_name]["latency"]  # (n_models, n_examples)
                masks = self.para_qa_data[task_name]["mask"]  # (n_models, n_examples)
                normalized_latencies = normalize_metric(latencies, self.latency_min[task_name], self.latency_max[task_name], masks)
                model_means = torch.stack([self.masked_mean(normalized_latencies[i], masks[i]) for i in range(len(self.model_names))])
                self.mu_l.append(model_means)
            self.mu_l = torch.stack(self.mu_l, dim=1).to(self.device, dtype=torch.float32)  # (n_models, n_tasks)
            
        if self.sigma_c is None:
            self.sigma_c = []
            for task_name in self.para_qa_data.keys():
                costs = self.para_qa_data[task_name]["cost"]  # (n_models, n_examples)
                masks = self.para_qa_data[task_name]["mask"]  # (n_models, n_examples)
                normalized_costs = normalize_metric(costs, self.cost_min[task_name], self.cost_max[task_name], masks)
                model_stds = torch.stack([self.masked_std(normalized_costs[i], masks[i]) for i in range(len(self.model_names))]).to(self.device)
                self.sigma_c.append(model_stds)
            self.sigma_c = torch.stack(self.sigma_c, dim=1).to(self.device, dtype=torch.float32)  # (n_models, n_tasks)
            
        if self.sigma_r is None:
            self.sigma_r = []
            for task_name in self.para_qa_data.keys():
                effectiveness = self.para_qa_data[task_name]["effectiveness"]  # (n_models, n_examples)
                masks = self.para_qa_data[task_name]["mask"]  # (n_models, n_examples)
                model_stds = torch.stack([self.masked_std(effectiveness[i], masks[i]) for i in range(len(self.model_names))]).to(self.device)
                self.sigma_r.append(model_stds)
            self.sigma_r = torch.stack(self.sigma_r, dim=1).to(self.device, dtype=torch.float32)  # (n_models, n_tasks)
            
        if self.sigma_l is None:
            self.sigma_l = []
            for task_name in self.para_qa_data.keys():
                latencies = self.para_qa_data[task_name]["latency"]  # (n_models, n_examples)
                masks = self.para_qa_data[task_name]["mask"]  # (n_models, n_examples)
                normalized_latencies = normalize_metric(latencies, self.latency_min[task_name], self.latency_max[task_name], masks)
                model_stds = torch.stack([self.masked_std(normalized_latencies[i], masks[i]) for i in range(len(self.model_names))]).to(self.device)
                self.sigma_l.append(model_stds)
            self.sigma_l = torch.stack(self.sigma_l, dim=1).to(self.device, dtype=torch.float32)  # (n_models, n_tasks)
        if self.k_prototype is None:
            self.k_prototype = self.task_hierarchy_representation["embeddings"].to(self.device) # K, dim
        # print(self.mu_c.shape, self.mu_r.shape, self.mu_l.shape, self.sigma_c.shape, self.sigma_r.shape, self.sigma_l.shape)
        # exit(0)
    def estimate_ptq(self, query: Union[str, List[str]]) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Estimate p(t|q) using specified method
        
        Args:
            query: Input query or list of queries
            topk: Number of top tasks to consider
            **kwargs: Additional arguments for specific methods  
        Returns:
            Dictionary mapping task names to probabilities for a single query, or list of such dictionaries for batch input
        """
        # 如果输入是单个字符串，将其转换为列表
        if isinstance(query, str):
            query = [query]
        
        # 编码所有查询
        query_embedding = self.sim_tool.encode_batch(query, convert_to_numpy=False)  # (batch_size, dim)
        query_embedding = query_embedding.to(self.device)
        
        # 计算所有查询与任务的相似度
        similarities = torch.matmul(query_embedding, self.k_prototype.T)  # shape: (batch_size, K)
        
        if self.ptq_mode == "softmax":
            z = torch.softmax(similarities/self.tau, dim=1) # shape: (batch_size, K)
        elif self.ptq_mode == "argmax":
            # 取得每一行最大值的位置（索引）
            z = torch.argmax(similarities, dim=1, keepdim=True)  # shape: (B, 1)

            # 创建 one-hot mask
            z = torch.zeros_like(similarities).scatter_(1, z, 1)

        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        return z, query_embedding
     
    def estimate_pcqm(self, query: Union[str, List[str]]) -> torch.Tensor:
        """
        Estimate metrics (cost, latency, effectiveness) for all models based on query
        
        Args:
            query: Input query or list of queries
            topk: Number of top tasks to consider
            estimated_pctm: Pre-computed pctm tensor with shape (n_models, n_tasks, n_metrics).
                          If None, will compute using estimate_pctm.
            
        Returns:
            Tensor with shape (batch_size, n_models, n_metrics) containing estimated metrics for each model
        """
        if isinstance(query, str):
            query = [query]
        # Original implementation
        # Get task probabilities p(t|q)
        task_probs, query_embedding = self.estimate_ptq(query) # shape: (batch_size, K)
        results = []
        if self.pcqm_mode == "mean":
            pred_cost = torch.matmul(task_probs, self.mu_c.T) # shape: (batch_size, M)
            results.append(pred_cost)
            pred_effectiveness = torch.matmul(task_probs, self.mu_r.T) # shape: (batch_size, M)
            results.append(pred_effectiveness)
            pred_latency = torch.matmul(task_probs, self.mu_l.T) # shape: (batch_size, M)
            results.append(pred_latency)
            return torch.stack(results, dim=2) # shape: (batch_size, M, n_metrics)
        elif self.pcqm_mode == "weighted":
            index_list = torch.argmax(task_probs, dim=1)  # shape: (batch_size,)
            batch_results = []
            
            for i, index in enumerate(index_list):
                index = str(index.item())
                # Get task data
                task_data = self.para_qa_data[index]
                
                # Calculate similarity between query and examples
                sim_ex_query = torch.matmul(query_embedding[i].unsqueeze(0), 
                                          self.difficulty_example_embeddings[index].T)  # shape: (1, n_examples)
                sim_ex_query = sim_ex_query / self.tau
                
                # Process each model separately due to different masks
                model_metrics = []
                for model_idx, model_name in enumerate(self.model_names):
                    # Get model-specific mask and apply it
                    mask = task_data["mask"][model_idx]  # (n_examples,)
                    masked_sim = torch.where(mask == 1, sim_ex_query, 
                                          torch.tensor(float('-inf')).to(self.device))
                    sim_weights = torch.softmax(masked_sim, dim=1)  # (1, n_examples)
                    
                    # Get and normalize metrics for this model
                    costs = task_data["cost"][model_idx].to(dtype=torch.float32)  # (n_examples,)
                    effectiveness = task_data["effectiveness"][model_idx].to(dtype=torch.float32)  # (n_examples,)
                    latencies = task_data["latency"][model_idx].to(dtype=torch.float32)  # (n_examples,)
                    
                    # Normalize metrics using model-specific mask
                    normalized_costs = normalize_metric(
                        costs.unsqueeze(0),  # add batch dim
                        float(self.cost_min[index]), 
                        float(self.cost_max[index]),
                        mask.unsqueeze(0).to(dtype=torch.float32)  # add batch dim
                    ).squeeze(0)  # remove batch dim
                    
                    normalized_latencies = normalize_metric(
                        latencies.unsqueeze(0),
                        float(self.latency_min[index]),
                        float(self.latency_max[index]),
                        mask.unsqueeze(0).to(dtype=torch.float32)
                    ).squeeze(0)
                    
                    # Calculate weighted predictions for this model
                    pred_cost = torch.matmul(sim_weights, normalized_costs)  # (1,)
                    pred_effectiveness = torch.matmul(sim_weights, effectiveness)  # (1,)
                    pred_latency = torch.matmul(sim_weights, normalized_latencies)  # (1,)
                    
                    # Combine metrics for this model
                    model_metrics.append(torch.stack([
                        pred_cost.squeeze(),
                        pred_effectiveness.squeeze(),
                        pred_latency.squeeze()
                    ]))
                
                # Stack metrics for all models
                metrics = torch.stack(model_metrics, dim=0)  # (n_models, 3)
                batch_results.append(metrics)
            
            results = torch.stack(batch_results, dim=0)  # (batch_size, n_models, n_metrics)
            return results  

        
    def few_shot_tuning_sentence_bert(self, train_data: Dict[str, Any], val_data: Dict[str, Any], test_data: Dict[str, Any],
                                    save_dir: Optional[str] = None, tuning_mode = 'lora',
                                    use_grad_for_train=False,
                                    tuning_config=None) -> torch.Tensor:
        """
        Fine-tune sentence BERT model using prompt tuning for difficulty classification
        
        Args:
            train_data: Dictionary containing training data
                {
                    'queries': List of query strings,
                    'difficulty_ids': List of difficulty IDs
                }
            test_data: Dictionary containing test data (same format as train_data)
            prompt_sharing: Whether to use separate prompts for each difficulty ("separate")
                          or share prompts across difficulties ("shared")
            num_virtual_tokens: Number of virtual tokens to use for prompt tuning
            save_dir: Directory to save trained prompts
            
        Returns:
            Updated difficulty embeddings tensor
        """
        if tuning_mode == 'variational':
            return self.variational_inference_tuning(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                save_dir=save_dir,
                tuning_config=tuning_config,
                use_grad_for_train=use_grad_for_train,
            )
        raise NotImplementedError(
            "Only tuning_mode='variational' is supported; use variational_prompt_tuner. "
            f"Got tuning_mode={tuning_mode!r}."
        )
    
    def variational_inference_tuning(self, train_data: Dict[str, Any], val_data: Dict[str, Any], test_data: Dict[str, Any],
                                   save_dir: Optional[str] = None, tuning_config: Optional[Dict[str, Any]] = None,
                                   use_grad_for_train: bool = False) -> torch.Tensor:
        """
        Use variational inference to learn p(c|q,M) = Σ_t p(c|t,q,M) * p(t|q)
        """
        if self.support_multiple_labels:
            tuner = VariationalPromptTunerMultiple(
                device=self.device,
                tuning_mode='variational',
                use_grad_for_train=use_grad_for_train,
                num_models=len(self.model_names),
                variational_config=tuning_config,
            )
        else:
            tuner = VariationalPromptTuner(
                device=self.device,
                tuning_mode='variational',
                use_grad_for_train=use_grad_for_train,
                num_models=len(self.model_names),
                variational_config=tuning_config,
            )

        print(tuning_config["lr"])
        tuner.train_variational(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            difficult_prompt=self.difficulty_texts,
            model_names=self.model_names,
            num_epochs=20,
            learning_rate=tuning_config["lr"],
            batch_size=64,
            save_dir=save_dir,
        )

        self.variational_tuner = tuner
        return None

    def estimate_pcqm_variational(self, query: Union[str, List[str]]) -> torch.Tensor:
        """
        Estimate cost using variational inference: p(c|q,M) = Σ_t p(c|t,q,M) * p(t|q)
        
        Args:
            query: Input query or list of queries
            
        Returns:
            Tensor with shape (batch_size, n_models, 1) containing estimated costs for each model
        """
        if not hasattr(self, 'variational_tuner') or self.variational_tuner is None:
            raise ValueError("Variational tuner not initialized. Please run variational training first.")
        
        if isinstance(query, str):
            query = [query]
        
        batch_size = len(query)
        n_models = len(self.model_names)
        
        # Model-specific predictions: p_θ(metric|t,q) for each model - more efficient
        cost_predictions = self.variational_tuner.predict_metrics_all_models(query, "cost")  # (batch_size, n_models)
        effectiveness_predictions = self.variational_tuner.predict_metrics_all_models(query, "effect")  # (batch_size, n_models)
        
        # Stack metrics: cost, effectiveness, latency
        cost_tensor = torch.stack([
            cost_predictions,      # (batch_size, n_models)
            effectiveness_predictions,  # (batch_size, n_models)
            torch.zeros_like(cost_predictions).to(self.device)  # Placeholder for latency - use original method
        ], dim=2)  # (batch_size, n_models, 3)
        
        return cost_tensor
    
    def get_task_distribution_variational(self, query: Union[str, List[str]]) -> torch.Tensor:
        """
        Get task distribution q_φ(t|q) using variational inference
        
        Args:
            query: Input query or list of queries
            
        Returns:
            Tensor with shape (batch_size, num_tasks) containing task probabilities
        """
        if not hasattr(self, 'variational_tuner') or self.variational_tuner is None:
            raise ValueError("Variational tuner not initialized. Please run variational training first.")
        
        if isinstance(query, str):
            query = [query]
        
        return self.variational_tuner.get_task_distribution(query)
