"""
Fine-tuning trainer for sentence embedding models using InfoNCE loss
基于InfoNCE损失的句子嵌入模型微调训练器

支持：
1. 全参数微调和高效微调(LoRA)
2. 三种数据集级别训练(domain, subtask, difficulty)
3. HuggingFace Trainer集成
4. InfoNCE对比学习损失
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import accuracy_score
try:
    import wandb
except ImportError:
    wandb = None
from src.utils.data_loader import split_qa_data, convert_classification_to_qa_format

from src.utils.curriculum_finetune_generator import CurriculumDataGenerator, create_contrastive_data_collator
from src.utils.query_type_graph import extract_task_hierarchy
from src.utils.logger import LLMLogger
from src.utils.data_loader import DOMAIN_FORMAT, SUBCAT_FORMAT, DIFF_FORMAT, DatasetGen

# set_seed(42)

SAVE_DIR = "/hdd2/lh/agenticrouter_data/finetune_results"
# 设置日志
logger = LLMLogger(log_dir=SAVE_DIR + "/logs/finetune")


def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class FewShotEvaluationCallback(TrainerCallback):


    """在每个epoch结束时进行Few-shot评估的回调函数"""
    
    def __init__(
        self, 
        few_shot_data: Dict[str, Any],
        kg_data: Dict[str, Any],
        model_wrapper: 'SentenceTransformerModel',
        config: 'FinetuneConfig',
        output_dir: str
    ):
        """
        初始化Few-shot评估回调
        
        Args:
            few_shot_data: few-shot测试数据
            kg_data: 知识图谱数据
            model_wrapper: 模型包装器
            config: 训练配置
            output_dir: 输出目录
        """
        self.few_shot_data = few_shot_data
        self.model_wrapper = model_wrapper
        self.config = config
        self.output_dir = output_dir
        self.kg_data = kg_data
        self.epoch_results = []
        
        # 存储预计算的嵌入
        self.domain_embeddings = {}
        self.subcat_embeddings = {}
        self.difficulty_embeddings = {}

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 导入格式常量

        self.domain_format = DOMAIN_FORMAT
        self.subcat_format = SUBCAT_FORMAT
        self.diff_format = DIFF_FORMAT
    
    def _precompute_embeddings(self, model):
        """
        预计算不同粒度的embeddings，参考ana.ipynb中的_precompute_embeddings方法
        """
        device = next(model.parameters()).device
        
        with torch.no_grad():
            # 计算domain embeddings
            for domain in self.kg_data["domains"]:
                domain_text = self.domain_format.format(domain["name"], domain["definition"])
                domain_tokens = self.model_wrapper.tokenizer(
                    [domain_text],
                    padding=self.config.padding,
                    truncation=self.config.truncation,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                ).to(device)
                
                domain_embedding  = model.encode(
                    query_input_ids=domain_tokens['input_ids'],
                    query_attention_mask=domain_tokens['attention_mask']
                )
                # domain_embedding = mean_pooling(domain_outputs, domain_tokens['attention_mask'])
                self.domain_embeddings[str(domain["id"])] = domain_embedding
            
            # 计算subcategory embeddings
            for subcat in self.kg_data["subcategories"]:
                # 找到对应的domain名称
                domain_name = None
                for domain in self.kg_data["domains"]:
                    if domain["id"] == subcat["parent_id"]:
                        domain_name = domain["name"]
                        break
                
                if domain_name:
                    subcat_text = self.subcat_format.format(subcat["name"], domain_name, subcat["definition"])
                    subcat_tokens = self.model_wrapper.tokenizer(
                        [subcat_text],
                        padding=self.config.padding,
                        truncation=self.config.truncation,
                        max_length=self.config.max_length,
                        return_tensors='pt'
                    ).to(device)
                    
                    subcat_embedding = model.encode(
                        query_input_ids=subcat_tokens['input_ids'],
                        query_attention_mask=subcat_tokens['attention_mask']
                    )
                    self.subcat_embeddings[str(subcat["id"])] = subcat_embedding
            
            # 计算difficulty embeddings
            for diff in self.kg_data["difficulty_levels"]:
                # 找到对应的subcategory名称
                subcat_name = None
                for subcat in self.kg_data["subcategories"]:
                    if subcat["id"] == diff["parent_id"]:
                        subcat_name = subcat["name"]
                        break
                
                if subcat_name:
                    diff_text = self.diff_format.format(subcat_name, diff["name"], diff["definition"])
                    diff_tokens = self.model_wrapper.tokenizer(
                        [diff_text],
                        padding=self.config.padding,
                        truncation=self.config.truncation,
                        max_length=self.config.max_length,
                        return_tensors='pt'
                    ).to(device)
                    
                    diff_embedding = model.encode(
                        query_input_ids=diff_tokens['input_ids'],
                        query_attention_mask=diff_tokens['attention_mask']
                    )
                    self.difficulty_embeddings[str(diff["id"])] = diff_embedding
        
        logger.info(f"Precomputed embeddings: {len(self.domain_embeddings)} domains, "
                   f"{len(self.subcat_embeddings)} subcategories, {len(self.difficulty_embeddings)} difficulties")
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """在每个epoch结束时执行few-shot评估"""
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Running Few-shot Evaluation at Epoch {state.epoch}")
        logger.info(f"{'='*50}")
        
        # 设置模型为评估模式
        model = kwargs.get('model', self.model_wrapper)
        model.eval()
        
        # 预计算所有层级的embeddings
        logger.info("Precomputing embeddings for all levels...")
        self._precompute_embeddings(model)
        
        epoch_result = {
            'epoch': int(state.epoch),
            'global_step': state.global_step,
            'classification_accuracies': {}
        }
        
        # 执行分类准确率评估
        logger.info("Evaluating classification accuracy...")

        accuracies = self._evaluate_classification_accuracy(model)
        epoch_result['classification_accuracies'] = accuracies
        
        # 打印结果
        for level in ["domain", "subcategory", "difficulty"]:
            if level in accuracies:
                acc = accuracies[level].get(1, 0.0)  # top-1准确率
                logger.info(f"{level.capitalize()} Top-1 Accuracy: {acc:.4f}")
                    

        # 保存当前epoch的结果
        self.epoch_results.append(epoch_result)
        
        # 保存到文件
        results_path = os.path.join(self.output_dir, f"few_shot_results_epoch_{int(state.epoch)}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(epoch_result, f, indent=2, ensure_ascii=False)
        
        # 保存累积结果
        all_results_path = os.path.join(self.output_dir, "few_shot_results_all_epochs.json")
        with open(all_results_path, 'w', encoding='utf-8') as f:
            json.dump(self.epoch_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Few-shot evaluation results saved to {results_path}")
        
        # 记录到日志系统
        logger.log_interaction(
            prompt=f"Few-shot evaluation at epoch {int(state.epoch)}",
            response=f"Completed classification accuracy evaluation",
            system_prompt="Few-shot evaluation",
            metadata=epoch_result
        )
    
    def _evaluate_classification_accuracy(self, model) -> Dict[str, Dict[int, float]]:
        """
        评估分类准确率，参考ana.ipynb中的evaluate_classification_accuracy方法
        """
        device = next(model.parameters()).device
        k_values = [1, 3, 5]  # 计算top-k准确率
        max_k = max(k_values)
        
        # 初始化计数器
        correct_counts = {
            "domain": {k: 0 for k in k_values},
            "subcategory": {k: 0 for k in k_values}, 
            "difficulty": {k: 0 for k in k_values}
        }
        total_counts = {
            "domain": {k: 0 for k in k_values},
            "subcategory": {k: 0 for k in k_values},
            "difficulty": {k: 0 for k in k_values}
        }
        
        with torch.no_grad():
            # 遍历few-shot数据中的每个任务
            for task_id, examples in self.few_shot_data.items():
                # 获取真实的层级信息
                true_difficulty_id = str(task_id)
                true_subcat_id = str(self.kg_data["difficulty_levels"][int(task_id)]["parent_id"])
                true_domain_id = str(self.kg_data["subcategories"][int(true_subcat_id)]["parent_id"])
        
                # 对每个样本进行预测
                for example in examples:
                    query = example[0]
       
                    # 编码query
                    query_tokens = self.model_wrapper.tokenizer(
                        [query],
                        padding=self.config.padding,
                        truncation=self.config.truncation,
                        max_length=self.config.max_length,
                        return_tensors='pt'
                    )
                    # print(query_tokens['input_ids'].to(device).shape)
                    # print(query_tokens['attention_mask'].to(device).shape)
                    
                    query_embedding = model.encode(
                        query_input_ids=query_tokens['input_ids'].to(device),
                        query_attention_mask=query_tokens['attention_mask'].to(device)
                    )
   
                    
                    # 预测不同层级
                    topk_domains = self._predict_topk(query_embedding, self.domain_embeddings, max_k)
                    topk_subcats = self._predict_topk(query_embedding, self.subcat_embeddings, max_k)
                    topk_diffs = self._predict_topk(query_embedding, self.difficulty_embeddings, max_k)
                    
                    # 计算准确率
                    for k in k_values:
                        if true_domain_id in [domain[0] for domain in topk_domains[:k]]:
                            correct_counts["domain"][k] += 1
                        total_counts["domain"][k] += 1
                        
                        if true_subcat_id in [subcat[0] for subcat in topk_subcats[:k]]:
                            correct_counts["subcategory"][k] += 1
                        total_counts["subcategory"][k] += 1
                        
                        if true_difficulty_id in [diff[0] for diff in topk_diffs[:k]]:
                            correct_counts["difficulty"][k] += 1
                        total_counts["difficulty"][k] += 1
        
        # 计算每个层级的准确度
        accuracies = {}
        for level in ["domain", "subcategory", "difficulty"]:
            accuracies[level] = {
                k: correct_counts[level][k] / total_counts[level][k] if total_counts[level][k] > 0 else 0 
                for k in k_values
            }
        print("accuracies", accuracies)
        return accuracies
    
    def _predict_topk(self, query_embedding: torch.Tensor, embeddings_dict: Dict[str, torch.Tensor], k: int) -> List[Tuple[str, float]]:
        """
        预测top-k最相似的类别
        
        Args:
            query_embedding: 查询嵌入
            embeddings_dict: 类别嵌入字典 {id: embedding}
            k: 返回top-k个结果
            
        Returns:
            List[(category_id, similarity_score)]
        """

        similarities = []
        for category_id, category_embedding in embeddings_dict.items():
            similarity = torch.cosine_similarity(query_embedding, category_embedding, dim=1).item()
            similarities.append((category_id, similarity))
        
        # 按相似度降序排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


class InfoNCELoss(nn.Module):
    """InfoNCE对比学习损失函数"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, query_embeddings: torch.Tensor, 
                prompt_embeddings: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        Args:
            query_embeddings: 查询嵌入 (batch_size, hidden_size)
            prompt_embeddings: 提示嵌入 (batch_size, hidden_size)  
            labels: 对比学习标签 (batch_size, batch_size)
                   1表示在计算分母时需要考虑的负样本，0表示不考虑的样本
                   正样本是对角线元素（query与对应prompt的匹配）
                   
        Returns:
            InfoNCE损失值
        """
        # 归一化嵌入
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1) # shape (batch_size, hidden_size)
        prompt_embeddings = F.normalize(prompt_embeddings, p=2, dim=1) # shape (batch_size, hidden_size)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(query_embeddings, prompt_embeddings.T) / self.temperature # shape (batch_size, batch_size)
        
        # 正样本相似度（对角线元素）
        positive_similarities = torch.diag(similarity_matrix) # shape (batch_size,)
        
        # 创建掩码：对角线为True（正样本），labels=1的位置为True（考虑的负样本）
        # batch_size = query_embeddings.size(0)
        considered_mask = labels.bool()
        
        # 对不考虑的样本位置设置为很小的值（避免影响logsumexp）
        masked_similarity = similarity_matrix.clone()

        masked_similarity[~considered_mask] = -float('inf')

        
        # 计算分母：对每行进行logsumexp（正样本 + 考虑的负样本）
        denominators = torch.logsumexp(masked_similarity, dim=1) # shape (batch_size,)
        
        # InfoNCE损失：-log(exp(positive)/sum(exp(considered)))
        # = -(positive - log(sum(exp(considered))))
        loss = -(positive_similarities - denominators).mean()
        
        return loss, similarity_matrix.cpu()


class SentenceTransformerModel(nn.Module):
    """包装的句子变换器模型"""
    
    def __init__(self, model_name: str, use_lora: bool = False, lora_config: Optional[Dict] = None):
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 应用LoRA（如果启用）
        if use_lora:
            if lora_config is None:
                lora_config = {
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["query", "key", "value", "dense"],
                    "lora_dropout": 0.1,
                    "bias": "none"
                }
            
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                **lora_config
            )
            self.model = get_peft_model(self.model, peft_config)
            logger.info(f"Applied LoRA with config: {lora_config}")
        
        # InfoNCE损失函数
        self.loss_fn = InfoNCELoss()
        
    def forward(self, query_input_ids, query_attention_mask,
                prompt_input_ids, prompt_attention_mask, labels):
        """前向传播"""
        
        # 编码查询
        query_outputs = self.model(input_ids=query_input_ids, attention_mask=query_attention_mask)
        query_embeddings = mean_pooling(query_outputs, query_attention_mask)
        
        # 编码提示
        prompt_outputs = self.model(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask)
        prompt_embeddings = mean_pooling(prompt_outputs, prompt_attention_mask)
        

        loss, similarity_matrix = self.loss_fn(query_embeddings, prompt_embeddings, labels) # ,shape (batch_size, batch_size)
            
        return {
            'loss': loss,
            'logits': similarity_matrix,
            'labels': labels
        }

    def encode(self, query_input_ids, query_attention_mask):
        query_outputs = self.model(input_ids=query_input_ids, attention_mask=query_attention_mask)
        query_embeddings = mean_pooling(query_outputs, query_attention_mask)
        return F.normalize(query_embeddings, p=2, dim=1)

    
@dataclass
class FinetuneConfig:
    """微调配置"""
    
    # 模型配置
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_lora: bool = False
    lora_config: Optional[Dict] = None
    
    # 数据配置
    data_level: str = "domain"  # "domain", "subtask", "difficulty"
    batch_size: int = 64
    max_length: int = 512
    padding: bool = True
    truncation: bool = True
    
    # 训练配置
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    temperature: float = 0.1
    
    # 输出配置
    output_dir: str = SAVE_DIR 
    logging_steps: int = 10
    save_steps: int = 1  # 每个epoch保存一次
    eval_steps: int = 1  # 每个epoch评估一次
    
    # 评估配置
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # 早停配置
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001


class SentenceTransformerTrainer:
    """句子变换器微调训练器"""
    
    def __init__(self, config: FinetuneConfig, few_shot_data_path: Optional[str] = None, kg_data_path: Optional[str] = None):
        self.config = config
        self.model = None
        self.trainer = None
        self.few_shot_data = None
        self.kg_data = None
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 加载few-shot数据
        if os.path.exists(few_shot_data_path) and kg_data_path and os.path.exists(kg_data_path):
            self.few_shot_data = self._load_few_shot_data(few_shot_data_path, kg_data_path)
            logger.info(f"Loaded few-shot data from {few_shot_data_path}")
        else:
            logger.error(f"Few-shot data or kg data not found")
            exit()
        
        # 保存配置
        config_path = os.path.join(config.output_dir, "config.json")    
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(vars(config), f, indent=2, ensure_ascii=False)
            
        logger.info(f"Initialized trainer with config: {config}")

    
    def _load_few_shot_data(self, few_shot_data_path: str, kg_data_path: str, all_llm_data_path: str="/hdd2/lh/agenticrouter_data/data") -> Dict[str, Any]:
        """
        加载few-shot评估数据
        
        Args:
            few_shot_data_path: few-shot数据文件路径
            
        Returns:
            处理后的few-shot数据
        """
        
        data_loader = DatasetGen(data_dir=all_llm_data_path, task_names=["alpaca_data", "GSM8K", "multi_news", "SQUAD"], 
                    model_names=["qwen3-8b"])
        _, _, test_data = data_loader.load_data()
        try:
            self.kg_data = extract_task_hierarchy(kg_data_path)
            with open(few_shot_data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            raw_data = raw_data["test"]
            raw_data = convert_classification_to_qa_format(raw_data, test_data, self.kg_data)
            
            # 直接使用原始数据，假设格式为 {task_id: [(query, answer), ...]}
            logger.info(f"Loaded loading few-shot data and kg data")
            return raw_data
            
        except Exception as e:
            logger.error(f"Error loading few-shot data and kg data: {e}")
            exit()
    

    def load_data(self,  qa_data_path: str, split_ratio: float = 0.5) -> Tuple[Dataset, Dataset, CurriculumDataGenerator, CurriculumDataGenerator, Dict[str, Any]]:
        """
        加载和准备数据
        
        Args:
            qa_data_path: QA数据路径
            split_ratio: 训练/验证数据分割比例
            
        Returns:
            (train_dataset, eval_dataset, train_generator, eval_generator, kg_data)
        """
        logger.info(f"Loading data from {qa_data_path}")

        # 加载QA数据
        with open(qa_data_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        train_data, val_data = split_qa_data(qa_data, split_ratio)
        train_data_num = sum([len(examples) for examples in train_data.values()])
        val_data_num = sum([len(examples) for examples in val_data.values()])
        logger.info(f"Split data: {train_data_num} train examples, {val_data_num} val examples")
        
        # 创建数据生成器
        train_generator = CurriculumDataGenerator(self.kg_data, train_data)
        val_generator = CurriculumDataGenerator(self.kg_data, val_data)
        
        # 创建datasets
        train_datasets = train_generator.create_curriculum_datasets(target_task_gained=[self.config.data_level])
        val_datasets = val_generator.create_curriculum_datasets(target_task_gained=[self.config.data_level], shuffle=True)
        
        train_dataset = train_datasets[self.config.data_level]
        val_dataset = val_datasets[self.config.data_level]
        
        logger.info(f"Created Datasets for level: {self.config.data_level}")
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset, train_generator, val_generator
    
    def setup_model(self):
        """设置模型"""
        logger.info(f"Setting up model: {self.config.model_name}")
        
        self.model = SentenceTransformerModel(
            model_name=self.config.model_name,
            use_lora=self.config.use_lora,
            lora_config=self.config.lora_config
        )
        
        # 更新InfoNCE损失的温度参数
        self.model.loss_fn.temperature = self.config.temperature
        
        logger.info(f"Model setup complete. LoRA enabled: {self.config.use_lora}")
    
    def compute_metrics(self, eval_pred, compute_result=False):
        """
        计算评估指标
        
        Args:
            eval_pred: EvalPrediction对象，包含predictions和label_ids
            compute_result: 是否计算最终结果。当为True时，返回全局统计结果；
                          当为False时，累积batch级别的统计数据。
        
        Returns:
            评估指标字典
        """
        # 初始化或获取累积统计数据
        if not hasattr(self, '_eval_metrics'):
            self._eval_metrics = {
                'total_correct': 0,
                'total_samples': 0,
                'total_similarity': 0,
                'total_positives': 0
            }
        
        # 如果是最后一次调用，返回累积的结果
        if compute_result:
            metrics = self._compute_final_metrics()
            # 重置累积数据
            delattr(self, '_eval_metrics')
            return metrics
            
        # 获取当前batch的预测和标签
        logits = eval_pred.predictions[0]
        labels = eval_pred.predictions[1]
        # print(logits.shape)
        # print(labels.shape)
        # print(logits[0])
        # print(labels[0])
        logits = logits.cpu().numpy()
        labels = labels.cpu().numpy()
            
        
        # 打印形状以便调试
        # logger.info(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
        
        # 处理当前batch
        batch_size = logits.shape[0]
        
        # 创建对角线mask（真实的正样本）
        diagonal_mask = np.eye(batch_size, dtype=bool)
        
        # 创建掩码版本的logits，将label=0的位置设置为-inf
        masked_logits = logits.copy()
        masked_logits[labels == 0] = -float('inf')
        
        # 获取每个query的预测结果（只考虑label=1的部分）
        predicted_indices = np.argmax(masked_logits, axis=1)
        
        # 计算预测是否命中对角线（正确的匹配）
        correct_predictions = predicted_indices == np.arange(batch_size)
        
        # 累积统计
        self._eval_metrics['total_correct'] += correct_predictions.sum()
        self._eval_metrics['total_samples'] += batch_size
        self._eval_metrics['total_similarity'] += logits[diagonal_mask].sum()
        self._eval_metrics['total_positives'] += batch_size
        
        # 返回当前batch的指标
        return {
            "batch_accuracy": float(correct_predictions.sum() / batch_size),
            "batch_avg_similarity": float(logits[diagonal_mask].mean())
        }
    
    def _compute_final_metrics(self):
        """计算最终的评估指标"""
        total_correct = self._eval_metrics['total_correct']
        total_samples = self._eval_metrics['total_samples']
        total_similarity = self._eval_metrics['total_similarity']
        total_positives = self._eval_metrics['total_positives']
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_similarity = total_similarity / total_positives if total_positives > 0 else 0
        
        return {
            "accuracy": float(accuracy),
            "avg_similarity": float(avg_similarity),
            "correct_count": int(total_correct),
            "total_count": int(total_samples)
        }
    
    def setup_trainer(self, train_dataset, eval_dataset, train_generator: CurriculumDataGenerator, kg_data: Dict[str, Any]):
        """设置HuggingFace Trainer"""
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to="wandb" if wandb and wandb.run else None,
            batch_eval_metrics=True,
            run_name=f"finetune_{self.config.data_level}_{self.config.model_name.split('/')[-1]}",
            remove_unused_columns=False,  # 保持所有列
            dataloader_drop_last=True,
        )
        # 创建对比学习数据收集器
        data_collator = create_contrastive_data_collator(
            generator=train_generator,
            tokenizer=self.model.tokenizer,
            max_length=self.config.max_length,
            level=self.config.data_level
        )
        
        # 准备回调函数列表
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold
            )
        ]
        
        # 如果有few-shot数据，添加few-shot评估回调
        if self.few_shot_data:
            few_shot_callback = FewShotEvaluationCallback(
                few_shot_data=self.few_shot_data,
                kg_data=kg_data,
                model_wrapper=self.model,
                config=self.config,
                output_dir=os.path.join(self.config.output_dir, "few_shot_results")
            )
            callbacks.append(few_shot_callback)
            logger.info(f"Added few-shot evaluation callback for {len(self.few_shot_data)} tasks")
        
        # 创建Trainer
        # print("train_args", training_args.batch_size)
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        logger.info("Trainer setup complete")
    
    def train(self, qa_data_path: str, split_ratio: float = 0.75):
        """执行训练"""
        logger.info("Starting training...")
        
        # 加载数据
        train_dataset, eval_dataset, train_generator, eval_generator = self.load_data(qa_data_path, split_ratio=split_ratio)
        
        # 设置模型
        self.setup_model()
        
        # 设置训练器
        self.setup_trainer(train_dataset, eval_dataset, train_generator, self.kg_data)
        
        # 记录训练开始的配置
        logger.log_interaction(
            prompt=f"Starting fine-tuning with config: {vars(self.config)}",
            response="Training initiated",
            system_prompt="Fine-tuning sentence transformer model",
            metadata={
                "model_name": self.config.model_name,
                "data_level": self.config.data_level,
                "use_lora": self.config.use_lora,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs
            }
        )
        
        # 开始训练
        train_result = self.trainer.train()
        
        # 保存模型
        self.trainer.save_model()
        
        # 保存训练结果
        results_path = os.path.join(self.config.output_dir, "train_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(train_result.metrics, f, indent=2, ensure_ascii=False)
        
        # 记录训练完成
        logger.log_interaction(
            prompt=f"Training completed for {self.config.data_level} level",
            response=f"Final metrics: {train_result.metrics}",
            system_prompt="Fine-tuning completed",
            metadata={
                "final_train_loss": train_result.metrics.get('train_loss'),
                "train_runtime": train_result.metrics.get('train_runtime'),
                "train_samples_per_second": train_result.metrics.get('train_samples_per_second'),
                "output_dir": self.config.output_dir
            }
        )
        
        logger.info(f"Training completed. Results saved to {results_path}")
        
        return train_result
    
    def evaluate(self, eval_dataset: Dataset = None):
        """评估模型"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")
        
        eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        # 保存评估结果
        results_path = os.path.join(self.config.output_dir, "eval_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation completed. Results saved to {results_path}")
        
        return eval_result


def run_experiments(kg_data_path: str, qa_data_path: str, few_shot_results_path: str, base_output_dir: str = SAVE_DIR):
    """
    运行多个实验配置
    
    Args:
        kg_data_path: 知识图谱数据路径
        qa_data_path: QA数据路径
        few_shot_results_path: few-shot评估数据路径
        base_output_dir: 基础输出目录
    """
    
    # 实验配置
    experiments = [
        # 全参数微调 - 三个数据级别
        # {
        #     "name": "full_finetune_domain",
        #     "config": FinetuneConfig(
        #         data_level="difficulty",
        #         use_lora=False,
        #         learning_rate=2e-5,
        #         num_epochs=3,
        #         batch_size=128,
        #         output_dir=os.path.join(base_output_dir, "full_finetune_domain")
        #     )
        # },
        # {
        #     "name": "full_finetune_subtask", 
        #     "config": FinetuneConfig(
        #         data_level="subtask",
        #         use_lora=False,
        #         learning_rate=2e-5,
        #         num_epochs=3,
        #         batch_size=32,
        #         output_dir=os.path.join(base_output_dir, "full_finetune_subtask")
        #     )
        # },
        # {
        #     "name": "full_finetune_difficulty",
        #     "config": FinetuneConfig(
        #         data_level="difficulty", 
        #         use_lora=False,
        #         learning_rate=2e-5,
        #         num_epochs=3,
        #         batch_size=32,
        #         output_dir=os.path.join(base_output_dir, "full_finetune_difficulty")
        #     )
        # },
        
        # LoRA微调 - 三个数据级别
        {
            "name": "lora_finetune_domain",
            "config": FinetuneConfig(
                data_level="difficulty",
                use_lora=True,
                lora_config={
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["query", "key", "value", "dense"],
                    "lora_dropout": 0.1,
                    "bias": "none"
                },
                learning_rate=1e-4,  # LoRA通常用更高的学习率
                num_epochs=5,
                batch_size=128,
                output_dir=os.path.join(base_output_dir, "lora_finetune_domain")
            )
        },
        # {
        #     "name": "lora_finetune_subtask",
        #     "config": FinetuneConfig(
        #         data_level="subtask",
        #         use_lora=True,
        #         lora_config={
        #             "r": 16,
        #             "lora_alpha": 32,
        #             "target_modules": ["query", "key", "value", "dense"],
        #             "lora_dropout": 0.1,
        #             "bias": "none"
        #         },
        #         learning_rate=1e-4,
        #         num_epochs=5,
        #         batch_size=32,
        #         output_dir=os.path.join(base_output_dir, "lora_finetune_subtask")
        #     )
        # },
        # {
        #     "name": "lora_finetune_difficulty",
        #     "config": FinetuneConfig(
        #         data_level="difficulty",
        #         use_lora=True,
        #         lora_config={
        #             "r": 16,
        #             "lora_alpha": 32,
        #             "target_modules": ["query", "key", "value", "dense"],
        #             "lora_dropout": 0.1,
        #             "bias": "none"
        #         },
        #         learning_rate=1e-4,
        #         num_epochs=5,
        #         batch_size=32,
        #         output_dir=os.path.join(base_output_dir, "lora_finetune_difficulty")
        #     )
        # }
    ]
    
    results = {}
    
    for exp in experiments:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running experiment: {exp['name']}")
        logger.info(f"{'='*50}")
        
        
        # 记录实验开始
        logger.log_interaction(
            prompt=f"Starting experiment: {exp['name']}",
            response="Experiment initiated",
            system_prompt="Batch fine-tuning experiments",
            metadata={
                "experiment_name": exp['name'],
                "config": vars(exp['config'])
            }
        )
        
        # 初始化训练器
        trainer = SentenceTransformerTrainer(exp['config'], few_shot_data_path=few_shot_results_path, kg_data_path=kg_data_path)
        
        # 运行训练
        train_result = trainer.train( qa_data_path)
        
        # 保存结果
        results[exp['name']] = {
            "config": vars(exp['config']),
            "train_metrics": train_result.metrics,
            "status": "completed"
        }
        
        # 记录实验成功
        logger.log_interaction(
            prompt=f"Experiment {exp['name']} completed",
            response=f"Success with final loss: {train_result.metrics.get('train_loss', 'N/A')}",
            system_prompt="Experiment completed successfully",
            metadata={
                "experiment_name": exp['name'],
                "final_metrics": train_result.metrics,
                "status": "completed"
            }
        )
        
        logger.info(f"Experiment {exp['name']} completed successfully")
        

    
    # 保存所有实验结果
    results_path = os.path.join(base_output_dir, "all_experiments_results.json")
    os.makedirs(base_output_dir, exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nAll experiments completed. Results saved to {results_path}")
    
    # 打印结果摘要
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("="*60)
    
    for exp_name, result in results.items():
        status = result.get("status", "unknown")
        logger.info(f"\n{exp_name}: {status.upper()}")
        
        if status == "completed" and "train_metrics" in result:
            metrics = result["train_metrics"]
            logger.info(f"  Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
            logger.info(f"  Training time: {metrics.get('train_runtime', 'N/A'):.2f}s")
        elif status == "failed":
            logger.info(f"  Error: {result.get('error', 'Unknown error')}")
    
    return results


def main():
    """主函数 - 运行示例训练"""
    
    # 数据路径（根据ana.ipynb中的路径）
    kg_data_path = "/hdd2/lh/agenticrouter_data/kg_data/kg_data.json"
    qa_data_path = "/hdd2/lh/agenticrouter_data/kg_data/generated_qa_difficulty_nodes.json"
    few_shot_results_path = "/hdd2/lh/agenticrouter_data/query_task_type_results/classification_results.json"
    
    # 检查数据文件是否存在
    if not os.path.exists(kg_data_path):
        logger.error(f"KG data file not found: {kg_data_path}")
        return
    
    if not os.path.exists(qa_data_path):
        logger.error(f"QA data file not found: {qa_data_path}")
        return
    
    if not os.path.exists(few_shot_results_path):
        logger.error(f"Few shot results file not found: {few_shot_results_path}")
        return
    
    # 运行实验
    logger.info("Starting fine-tuning experiments...")
    
    results = run_experiments(
        kg_data_path=kg_data_path,
        qa_data_path=qa_data_path,
        few_shot_results_path=few_shot_results_path,
        base_output_dir=SAVE_DIR
    )
    
    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()
