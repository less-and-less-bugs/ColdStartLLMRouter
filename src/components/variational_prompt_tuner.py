"""
Variational Inference-based Few-shot Tuning for Cost-aware Task Classification
Implementing p(c|q,M) = Σ_t p(c|t,q,M) * p(t|q) using variational inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Literal
from transformers import AutoModel, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PromptTuningConfig,
    PromptTuningInit,
    PrefixTuningConfig,
    PromptEncoderConfig
)
from src.utils.similarity import mean_pooling
import os
from tqdm import tqdm
import numpy as np

class VariationalEncoder(nn.Module):
    """
    Variational Encoder: q_φ(t|q)
    计算查询和任务之间的相似度，输出查询在每个任务上的 logits
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 查询和任务的联合编码器
        self.joint_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # 拼接查询和任务嵌入
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # 输出一个相似度分数
        )
        
        # 注意力层，用于增强相似度计算
        self.query_attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        self.task_attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        

        
    def forward(self, query_embeddings: torch.Tensor, task_embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算查询在所有任务上的 logits
        
        Args:
            query_embeddings: [batch_size, embed_dim]
            task_embeddings: [n_tasks, embed_dim]
        Returns:
            logits: [batch_size, n_tasks] - 每个查询在每个任务上的 logits
        """
        batch_size = query_embeddings.size(0)
        n_tasks = task_embeddings.size(0)
        
        # 应用注意力机制到查询和任务嵌入
        encoded_query = self.query_attention(query_embeddings)  # [batch_size, embed_dim]
        encoded_tasks = self.task_attention(task_embeddings)    # [n_tasks, embed_dim]
        
        # 扩展维度以计算所有查询-任务对
        # [batch_size, 1, embed_dim] x [1, n_tasks, embed_dim]
        query_expanded = encoded_query.unsqueeze(1)            # [batch_size, 1, embed_dim]
        task_expanded = encoded_tasks.unsqueeze(0)             # [1, n_tasks, embed_dim]
        
        # 拼接查询和任务嵌入
        # 先扩展维度使其匹配
        query_tiled = query_expanded.expand(-1, n_tasks, -1)  # [batch_size, n_tasks, embed_dim]
        task_tiled = task_expanded.expand(batch_size, -1, -1) # [batch_size, n_tasks, embed_dim]
        
        # 拼接 [batch_size, n_tasks, embed_dim * 2]
        joint_embed = torch.cat([query_tiled, task_tiled], dim=-1)
        
        # 重塑为 2D 以通过联合编码器
        joint_embed_flat = joint_embed.reshape(-1, self.embed_dim * 2)  # [batch_size * n_tasks, embed_dim * 2]
        
        # 计算相似度
        similarities = self.joint_encoder(joint_embed_flat)  # [batch_size * n_tasks, 1]
        
        # 重塑回 [batch_size, n_tasks]
        logits = similarities.reshape(batch_size, n_tasks)
        
        return logits, encoded_query, encoded_tasks

class MetricDecoder(nn.Module):
    """
    Metric Decoder: p_θ(metric|t,q)
    Predicts a specific metric (cost, effect, latency) given task and query embeddings
    """
    def __init__(self, query_dim: int, task_dim: int, hidden_dim: int = 256, 
                 metric_type: str = "cost"):
        super().__init__()
        self.query_dim = query_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim
        self.metric_type = metric_type
        
        # Combined input dimension
        # input_dim = query_dim + task_dim
        input_dim = task_dim
        # MLP for metric prediction
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # All metrics are normalized to [0, 1]
        )
        
    def forward(self, task_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_embeddings: (batch_size, query_dim)
            task_embeddings: (batch_size, task_dim)
        Returns:
            metric_pred: (batch_size, 1) - predicted metric in [0, 1]
        """
        combined = task_embeddings
        metric_pred = self.decoder(combined)
        return metric_pred

class VariationalPromptTuner:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        tuning_mode: Literal["lora", "full", "prompt", "prefix", "p-tuning", "variational"] = "variational",
        tuning_config: Optional[Dict[str, Any]] = None,
        use_grad_for_train: bool = True,
        num_models: int = 6,  # Number of LLM models
        variational_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Variational Prompt Tuner with variational inference framework
        
        Args:
            device: Device to use for computation
            model_name: Name of the base sentence transformer model
            tuning_mode: Tuning mode - includes "variational" for our new method
            tuning_config: Configuration for the selected tuning mode
            num_models: Number of LLM models in the router
            variational_config: Configuration for variational inference components
        """
        self.device = device
        self.model_name = model_name
        self.tuning_mode = tuning_mode
        self.use_grad_for_train = use_grad_for_train
        self.num_models = num_models
        
        # Load base model and tokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get model dimensions
        self.model_dim = self.model.config.hidden_size
        
        # Variational inference configuration
        default_variational_config = {
            "hidden_dim": 256,
            "use_implicit_tasks": False,  # Whether to use implicit task embeddings
            "num_implicit_tasks": None,   # Number of implicit tasks (if used)
            "temperature": 1.0,           # Temperature for Gumbel-Softmax
            "kl_weight": 1.0,            # Weight for KL divergence term
            "cost_weight": 1.0,          # Weight for cost reconstruction term
            "effect_weight": 1.0,  # Weight for effectiveness reconstruction term
            "latency_weight": 1.0,       # Weight for latency reconstruction term
            "metrics": ["cost", "effect"]  # Which metrics to predict
        }
        self.variational_config = {**default_variational_config, **(variational_config or {})}
        
        # Initialize variational components (will be set up in setup_variational_components)
        self.variational_encoder = None
        self.metric_decoders = {}  # Decoders for different metrics and models
        self.task_embeddings = None
        
        # Configure base model based on tuning mode
        if tuning_mode == "variational":
            # For variational mode, we'll use the base model for encoding
            # and add our variational components on top
            self.model = self.model.to(device)
        else:
            # Apply existing PEFT configurations
            self._setup_peft_model(tuning_mode, tuning_config)
                
        # Initialize base model for non-tuned encoding
        self.base_model = AutoModel.from_pretrained(model_name).to(device)
        self.base_model.eval()
        
    def _setup_peft_model(self, tuning_mode: str, tuning_config: Optional[Dict[str, Any]]):
        """Setup PEFT model configuration (existing logic from original class)"""
        if tuning_mode == "lora":
            default_config = {
                "r": 8, "lora_alpha": 16, "target_modules": ["query", "value"],
                "lora_dropout": 0.05, "bias": "none", "task_type": TaskType.FEATURE_EXTRACTION
            }
            config_dict = {**default_config, **(tuning_config or {})}
            peft_config = LoraConfig(**config_dict)
            
        elif tuning_mode == "prompt":
            default_config = {
                "num_virtual_tokens": 8, "prompt_tuning_init": PromptTuningInit.TEXT,
                "prompt_tuning_init_text": "Classify the difficulty level of this task:",
                "tokenizer_name_or_path": self.model_name, "task_type": TaskType.FEATURE_EXTRACTION
            }
            config_dict = {**default_config, **(tuning_config or {})}
            peft_config = PromptTuningConfig(**config_dict)
            
        elif tuning_mode == "prefix":
            default_config = {
                "num_virtual_tokens": 8, "prefix_projection": True,
                "task_type": TaskType.FEATURE_EXTRACTION
            }
            config_dict = {**default_config, **(tuning_config or {})}
            peft_config = PrefixTuningConfig(**config_dict)
            
        elif tuning_mode == "p-tuning":
            default_config = {
                "num_virtual_tokens": 8, "encoder_hidden_size": 128,
                "task_type": TaskType.FEATURE_EXTRACTION
            }
            config_dict = {**default_config, **(tuning_config or {})}
            peft_config = PromptEncoderConfig(**config_dict)
            
        elif tuning_mode == "full":
            for param in self.model.parameters():
                param.requires_grad = True
            self.model = self.model.to(self.device)
            return
        
        # Apply PEFT configuration
        if tuning_mode != "full":
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, peft_config)
            self.model = self.model.to(self.device)
    
    def setup_variational_components(self, num_tasks: int, difficult_prompt: List[str],
                                   model_names: List[str]):
        """
        Setup variational inference components
        
        Args:
            num_tasks: Number of difficulty tasks
            task_embeddings: Pre-computed task embeddings (num_tasks, task_dim)
            model_names: List of model names
        """
        task_embeddings = self.encode_text(difficult_prompt, use_base_model=True, use_grad=False)

        # Setup task embeddings
        if self.variational_config["use_implicit_tasks"]:
            # Use learnable implicit task embeddings
            implicit_tasks = self.variational_config["num_implicit_tasks"] or num_tasks
            self.task_embeddings = nn.Parameter(
                torch.randn(implicit_tasks, self.model_dim, device=self.device)
            )
            self.num_tasks = implicit_tasks
        else:
            # Use provided task embeddings
            self.task_embeddings = task_embeddings.to(self.device)
            self.num_tasks = num_tasks
        
        # No model embeddings needed since we use separate decoders
        self.model_embeddings = None
        
        # Setup variational encoder q_φ(t|q)
        self.variational_encoder = VariationalEncoder(
            embed_dim=self.model_dim,
            hidden_dim=self.variational_config["hidden_dim"]
        ).to(self.device)
        
        # Setup metric decoders - one p_θ(metric|t,q) per model per metric
        self.metric_decoders = nn.ModuleDict()
        metrics_to_predict = self.variational_config["metrics"]
        
        for metric in metrics_to_predict:
            for i, model_name in enumerate(model_names):
                decoder_key = f"{metric}_model_{i}"
                self.metric_decoders[decoder_key] = MetricDecoder(
                    query_dim=self.model_dim,
                    task_dim=self.model_dim,
                    hidden_dim=self.variational_config["hidden_dim"],
                    metric_type=metric
                ).to(self.device)
    
    def gumbel_softmax(self, logits: torch.Tensor, temperature: float = 1.0, 
                      hard: bool = False) -> torch.Tensor:
        """
        Gumbel-Softmax sampling for differentiable discrete sampling
        
        Args:
            logits: (batch_size, num_categories)
            temperature: Temperature parameter
            hard: Whether to return hard (one-hot) or soft samples
        """
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        
        # Add noise to logits and apply softmax
        y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
        
        if hard:
            # Straight-through estimator
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = y_hard - y.detach() + y
        
        return y
    
    def encode_text(self, 
                    texts: Union[str, List[str]], 
                    batch_size: int = 32,
                    use_base_model: bool = False,
                    use_grad: bool = False) -> torch.Tensor:
        """
        Encode text(s) using mean pooling with flexible options and batch processing
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Select model
        model = self.base_model if use_base_model else self.model
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model output
            if use_grad:
                outputs = model(**encoded)
            else:
                with torch.no_grad():
                    outputs = model(**encoded)
            
            # Apply mean pooling
            batch_embeddings = mean_pooling(outputs, encoded['attention_mask'])
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings
    
    
    def forward_variational(self, query_embeddings: torch.Tensor, 
                           true_metrics: Optional[Dict[str, torch.Tensor]] = None,
                           temperature: Optional[float] = None,
                           true_task_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through variational model
        
        Args:
            query_embeddings: (batch_size, model_dim)
            true_metrics: Dict[str, (batch_size,)] - true metrics for training (cost, effectiveness, etc.)
            temperature: Temperature for Gumbel-Softmax
            true_task_ids: (batch_size,) - true task IDs (if available, for non-implicit tasks)
            
        Returns:
            Dictionary containing:
            - task_logits: (batch_size, num_tasks)
            - task_probs: (batch_size, num_tasks)
            - task_samples: (batch_size, num_tasks) - Gumbel-Softmax samples or one-hot
            - metric_predictions: Dict[str, (batch_size,)] - predictions for each metric
            - prior_probs: (batch_size, num_tasks) - uniform prior
        """
        batch_size = query_embeddings.size(0)
        
        # Get task distribution q_φ(t|q) and encoded embeddings
        task_logits, encoded_queries, encoded_tasks = self.variational_encoder(query_embeddings, self.task_embeddings)
        # print(encoded_queries.shape, encoded_tasks.shape)
        task_probs = F.softmax(task_logits/0.07, dim=-1)  # [batch_size, num_tasks]
        # print(torch.max(task_probs, dim=-1))
        
        # Predict metrics for each task embedding separately
        metrics_to_predict = self.variational_config["metrics"]
        metric_predictions = {}
        
        batch_size = query_embeddings.size(0)
        num_tasks = self.task_embeddings.size(0)
        
        # 扩展编码后的 query_embeddings 以匹配每个 task
        # [batch_size, 1, embed_dim] -> [batch_size, num_tasks, embed_dim]
        expanded_queries = encoded_queries.unsqueeze(1).expand(-1, num_tasks, -1)
        
        # 扩展编码后的 task_embeddings
        # [num_tasks, embed_dim] -> [batch_size, num_tasks, embed_dim]
        expanded_tasks = encoded_tasks.unsqueeze(0).expand(batch_size, -1, -1)
        
        for metric in metrics_to_predict:
            all_model_predictions = []
            
            for model_idx in range(self.num_models):
                decoder_key = f"{metric}_model_{model_idx}"
                decoder = self.metric_decoders[decoder_key]
                
                # 重塑维度以批量处理所有任务
                flat_queries = expanded_queries.reshape(-1, self.model_dim)  # [batch_size * num_tasks, embed_dim]
                flat_tasks = expanded_tasks.reshape(-1, self.model_dim)      # [batch_size * num_tasks, embed_dim]
                
                # 获取每个任务的预测
                task_preds = decoder(  flat_tasks)  # [batch_size * num_tasks, 1]
                task_preds = task_preds.reshape(batch_size, num_tasks)  # [batch_size, num_tasks]
                
                # 使用任务概率加权求和
                weighted_pred = torch.sum(task_preds * task_probs, dim=1, keepdim=True)  # [batch_size, 1]
                all_model_predictions.append(weighted_pred)
            
            # 合并所有模型的预测
            metric_predictions[metric] = torch.cat(all_model_predictions, dim=-1)  # [batch_size, n_models]

        prior_probs = torch.zeros_like(task_probs)  # [batch_size, num_tasks]
        if true_task_ids is not None:
            prior_probs.scatter_(1, true_task_ids.unsqueeze(1), 1.0)  # Set 1 at true task ID positions
        else:
            prior_probs = None

        return {
              "task_logits": task_logits,  # [batch_size, num_tasks]
              "task_probs": task_probs,    # [batch_size, num_tasks]
              "metric_predictions": metric_predictions,  # Dict[str, Tensor[batch_size, n_models]]
              "prior_probs": prior_probs,  # [batch_size, num_tasks]
          }
    
    def compute_elbo_loss(self, forward_output: Dict[str, torch.Tensor], 
                         true_metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss: E[log p(metrics|t,q,M)] - KL(q(t|q) || p(t|q))
        
        Args:
            forward_output: Output from forward_variational
            true_metrics: Dict[str, (batch_size,)] - true normalized metrics
            
        Returns:
            Dictionary containing loss components
        """
        # Reconstruction terms for each metric: E[log p(metric|t,q,M)]
        metric_predictions = forward_output["metric_predictions"]
 
        reconstruction_losses = {}
        mape_losses = {}  # 添加MAPE损失字典
        total_reconstruction_loss = 0
        total_mape = 0
        
        for metric in metric_predictions.keys():
            if metric in true_metrics:
                pred = metric_predictions[metric]
                true_val = true_metrics[metric]

                # MSE损失
                loss = F.mse_loss(pred, true_val)
                reconstruction_losses[metric] = loss
                
                # 计算绝对误差
                with torch.no_grad():
                    mape = torch.mean(torch.abs(true_val - pred))
                    mape_losses[metric] = mape
                    total_mape += mape
                
                # Apply metric-specific weights
                weight_key = f"{metric}_weight"
                weight = self.variational_config[weight_key]
                # print(weight_key, weight)
                total_reconstruction_loss += weight * loss
        
        # KL divergence term: KL(q(t|q) || p(t|q))
        q_probs = forward_output["task_probs"]
        p_probs = forward_output["prior_probs"]
        
        # KL divergence for categorical distributions
        kl_div = F.kl_div(
            torch.log(q_probs + 1e-8), 
            p_probs, 
            reduction='batchmean'
        )
        
        # ELBO = Reconstruction - KL
        # We minimize negative ELBO, so we maximize ELBO
        elbo = -total_reconstruction_loss - self.variational_config["kl_weight"] * kl_div
        total_loss = -elbo  # Minimize negative ELBO
        with torch.no_grad():
        # Calculate accuracy
            pred_tasks = torch.argmax(q_probs, dim=1)  # Get predicted task indices
            true_tasks = torch.argmax(p_probs, dim=1)  # Get true task indices
            accuracy = (pred_tasks == true_tasks).float().mean().item()
            
        result = {
            "total_loss": total_loss,
            "total_reconstruction_loss": total_reconstruction_loss,
            "total_mape": total_mape / len(metric_predictions.keys()),  # 平均MAPE
            "kl_divergence": kl_div,
            "elbo": elbo,
            "accuracy": accuracy
        }
        
        # Add individual reconstruction losses and MAPE
        for metric in metric_predictions.keys():
            if metric in true_metrics:
                result[f"{metric}_reconstruction_loss"] = reconstruction_losses[metric]
                result[f"{metric}_mape"] = mape_losses[metric]
            
        return result
    
    def prepare_variational_data_batch(self, data: Dict[str, Any], 
                                     batch_size: int = 32, shuffle: bool = True):
        """
        Prepare data batch for variational training
        
        Args:
            data: Dictionary containing:
                - queries: List[str] - 查询列表
                - cost: [n_examples, n_models] - 每个样本在每个模型上的成本
                - effectiveness: [n_examples, n_models] - 每个样本在每个模型上的效果
                - difficulty_ids: [n_examples] - 每个样本的任务ID
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            dataloader: DataLoader for training
            queries: List of query strings
            metric_names: List of metric names included in the data
        """
        # 准备基本数据
        query_indices = torch.arange(len(data['queries']))
        data['difficulty_ids'] = [int(diff_id) for diff_id in data['difficulty_ids']]
        task_ids = torch.tensor(data['difficulty_ids'], dtype=torch.long) # 
        
        # 准备指标数据
        metrics = {}
        for metric in self.variational_config["metrics"]:
            metrics[metric] = torch.tensor(data[metric], dtype=torch.float32)  # [n_examples, n_models] for each metric

        
        # 创建数据集
        dataset = torch.utils.data.TensorDataset(
            query_indices,  # [n_examples]
            task_ids,      # [n_examples]
            *[metrics[m] for m in self.variational_config["metrics"]]  # [n_examples, n_models] for each metric
        )
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        return dataloader, data['queries'], self.variational_config["metrics"]

    def load_model(self, save_dir: str):
        best_state = torch.load(os.path.join(save_dir, "variational_model.pt"))
        self.variational_encoder.load_state_dict(best_state['variational_encoder'])
        self.metric_decoders.load_state_dict(best_state['metric_decoders'])
        if isinstance(self.task_embeddings, nn.Parameter):
            self.task_embeddings.data = best_state['task_embeddings']
        else:
            self.task_embeddings = best_state['task_embeddings']

    def train_variational(self, 
                         train_data: Dict[str, Any], 
                         val_data: Dict[str, Any],
                         test_data: Dict[str, Any], 
                         difficult_prompt: List[str],
                         model_names: List[str],
                         num_epochs: int = 10, 
                         learning_rate: float = 1e-3, 
                         batch_size: int = 32,
                         save_dir: Optional[str] = None):
        """
        Train variational model using ELBO objective
        
        Args:
            train_data: Dictionary containing training data
                {
                    'queries': List of query strings,
                    'cost': List of cost arrays (normalized),
                    'difficulty_ids': List of difficulty IDs (for initialization)
                }
            val_data: Validation data (same format)
            test_data: Test data (same format)
            difficult_prompt: List of difficulty descriptions
            model_names: List of model names
        """
        # Setup variational components
        self.setup_variational_components(
            num_tasks=len(difficult_prompt),
           difficult_prompt=difficult_prompt,
            model_names=model_names,
        )
          # Load saved model if exists
        if os.path.exists(os.path.join(save_dir, "variational_model.pt")):
            self.load_model(save_dir)
            print("Loaded variational model from", save_dir)
            return self.task_embeddings

          # Prepare optimizers
        variational_params = list(self.variational_encoder.parameters()) + \
                           list(self.metric_decoders.parameters())
            
        if self.variational_config["use_implicit_tasks"]:
            variational_params.append(self.task_embeddings) # 修改成用已有的task embedding初始化，然后迭代。
            
        optimizer = torch.optim.AdamW(variational_params, lr=learning_rate)
        

        
        train_loader, train_queries, metric_names = self.prepare_variational_data_batch(
            train_data, batch_size, shuffle=True
        )
        
        # Training loop
        best_loss = float('inf')
        best_state = None
        patience =3
        patience_counter = 0
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            total_accuracy = 0
            epoch_loss_outputs = []  # 用于收集每个批次的损失输出
            
            # Set models to training mode
            self.variational_encoder.train()
            for decoder in self.metric_decoders.values():
                decoder.train()
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # 解析批次数据
                query_indices = batch[0].to(self.device)  # [batch_size]
                task_ids = batch[1].to(self.device)      # [batch_size]
                
                # 获取每个指标的值 [batch_size, n_models]
                batch_metrics = {}
                for i, metric_name in enumerate(metric_names):
                    batch_metrics[metric_name] = batch[i + 2].to(self.device)
                
                # 获取查询嵌入
                batch_queries = [train_queries[i] for i in query_indices.cpu()]
                query_embeddings = self.encode_text(batch_queries, use_base_model=True, use_grad=False)
                
                # 前向传播
                forward_output = self.forward_variational(
                    query_embeddings=query_embeddings,
                    true_metrics=batch_metrics,
                    true_task_ids=task_ids
                )
                
                # Compute ELBO loss
                loss_output = self.compute_elbo_loss(forward_output, batch_metrics)
                
                # Backward pass
                optimizer.zero_grad()
                loss_output["total_loss"].backward()
                optimizer.step()
                
                # 收集损失输出
                epoch_loss_outputs.append(loss_output)
                
                # Accumulate losses
                total_loss += loss_output["total_loss"].item()
                total_recon_loss += loss_output["total_reconstruction_loss"].item()
                total_kl_loss += loss_output["kl_divergence"].item()
                total_accuracy += loss_output["accuracy"]
            
            # Average losses
            avg_loss = total_loss / len(train_loader)
            avg_recon_loss = total_recon_loss / len(train_loader)
            avg_kl_loss = total_kl_loss / len(train_loader)
            avg_accuracy = total_accuracy / len(train_loader)
            
            # 计算每个指标的平均MAPE
            metric_mapes = {}
            for metric in metric_names:
                metric_mapes[metric] = sum(batch_loss_output[f"{metric}_mape"].item() 
                                        for batch_loss_output in epoch_loss_outputs) / len(train_loader)
            
            print(f"\nEpoch {epoch+1}:")
            print(f"  Training Metrics:")
            print(f"    Total Loss: {avg_loss:.4f}, Reconstruction Loss: {avg_recon_loss:.4f}, KL Divergence: {avg_kl_loss:.4f}")
            print(f"    Accuracy: {avg_accuracy:.4f}")
            print("    MAPE for each metric:")
            for metric, mape in metric_mapes.items():
                print(f"      {metric}: {mape:.2f}")
            
            # Validation
            val_metrics = self.evaluate_variational(val_data, model_names, batch_size)
            print(f"\n  Validation Metrics:")
            print(f"    Total Loss: {val_metrics['loss']:.4f}, Reconstruction Loss: {val_metrics['recon_loss']:.4f}")
            print(f"    KL Divergence: {val_metrics['kl_loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
            print("    MAPE for each metric:")
            for metric, mape in val_metrics['metric_mapes'].items():
                print(f"      {metric}: {mape:.2f}")
            
            # Early stopping
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                patience_counter = 0
                best_state = {
                    'variational_encoder': self.variational_encoder.state_dict(),
                    'metric_decoders': self.metric_decoders.state_dict(),
                    'model_embeddings': self.model_embeddings.data.clone() if self.model_embeddings is not None else None,
                    'task_embeddings': self.task_embeddings.data.clone() if isinstance(self.task_embeddings, nn.Parameter) else self.task_embeddings.clone()
                }
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        # Load best model
        if best_state:
            self.variational_encoder.load_state_dict(best_state['variational_encoder'])
            self.metric_decoders.load_state_dict(best_state['metric_decoders'])
            if isinstance(self.task_embeddings, nn.Parameter):
                self.task_embeddings.data = best_state['task_embeddings']
            else:
                self.task_embeddings = best_state['task_embeddings']
        
        # Save model
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(best_state, os.path.join(save_dir, "variational_model.pt"))
        
        # Return updated task embeddings for integration with router
        return self.task_embeddings
    
    def evaluate_variational(self, val_data: Dict[str, Any], model_names: List[str], 
                           batch_size: int = 32) -> Dict[str, Any]:
        """
        Evaluate variational model on validation data
        Returns:
            Dictionary containing various metrics including MAPE for each metric
        """
        self.variational_encoder.eval()
        for decoder in self.metric_decoders.values():
            decoder.eval()
        
        # 准备验证数据
        val_loader, val_queries, val_metric_names = self.prepare_variational_data_batch(
            val_data, batch_size, shuffle=False
        )
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_accuracy = 0
        metric_mapes = {metric: 0.0 for metric in val_metric_names}
        
        with torch.no_grad():
            for batch in val_loader:
                # 解析验证数据
                query_indices = batch[0].to(self.device)  # [batch_size]
                task_ids = batch[1].to(self.device)      # [batch_size]
                
                # 获取每个指标的值 [batch_size, n_models]
                batch_metrics = {}
                for i, metric_name in enumerate(val_metric_names):
                    batch_metrics[metric_name] = batch[i + 2].to(self.device)
                
                # 获取查询嵌入
                batch_queries = [val_queries[i] for i in query_indices.cpu()]
                query_embeddings = self.encode_text(batch_queries, use_base_model=True, use_grad=False)
                
                # 前向传播
                forward_output = self.forward_variational(
                    query_embeddings=query_embeddings,
                    true_metrics=batch_metrics,
                    true_task_ids=task_ids
                )
                
                loss_output = self.compute_elbo_loss(forward_output, batch_metrics)
                total_loss += loss_output["total_loss"].item()
                total_recon_loss += loss_output["total_reconstruction_loss"].item()
                total_kl_loss += loss_output["kl_divergence"].item()
                total_accuracy += loss_output["accuracy"]
                
                # 累加每个指标的MAPE
                for metric in val_metric_names:
                    metric_mapes[metric] += loss_output[f"{metric}_mape"].item()
        
        # 计算平均值
        num_batches = len(val_loader)
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'metric_mapes': {metric: mape / num_batches for metric, mape in metric_mapes.items()}
        }
    


    def predict_metrics_all_models(self, queries: List[str], metric: str = "cost") -> torch.Tensor:
        """
        Predict specified metric for given queries across ALL models efficiently
        
        Args:
            queries: List of query strings
            metric: Which metric to predict ("cost", "effectiveness", etc.)
            
        Returns:
            predicted_metrics: (batch_size, n_models) tensor of predicted metrics
        """
        if metric not in self.variational_config["metrics"]:
            raise ValueError(f"Metric '{metric}' not in configured metrics: {self.variational_config['metrics']}")
            
        self.variational_encoder.eval()
        for decoder in self.metric_decoders.values():
            decoder.eval()
        
        with torch.no_grad():
            query_embeddings = self.encode_text(queries, use_base_model=True, use_grad=False)
            
            # 直接使用 forward_variational
            forward_output = self.forward_variational(
                query_embeddings=query_embeddings,
                true_metrics=None,
                true_task_ids=None
            )
            
            predictions = forward_output["metric_predictions"][metric]
            # print(f"Predictions shape: {predictions.shape}")  # 应该是 [batch_size, n_models]
            return predictions

