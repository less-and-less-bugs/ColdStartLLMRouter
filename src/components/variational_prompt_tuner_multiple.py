"""
Variational Inference-based Few-shot Tuning for Multi-label Cost-aware Task Classification
Implementing p(c|q,M) = Σ_t p(c|t,q,M) * p(t|q) using variational inference with multi-label support
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

# Import base classes from single-label version
from src.components.variational_prompt_tuner import VariationalEncoder, MetricDecoder

class VariationalPromptTunerMultiple:
    """
    Multi-label version of VariationalPromptTuner
    Supports queries with multiple task type labels
    """
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        tuning_mode: Literal["variational"] = "variational",
        tuning_config: Optional[Dict[str, Any]] = None,
        use_grad_for_train: bool = True,
        num_models: int = 6,
        variational_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Variational Prompt Tuner with multi-label support
        
        Args:
            device: Device to use for computation
            model_name: Name of the base sentence transformer model
            tuning_mode: Tuning mode (only "variational" supported for multi-label)
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
            "use_implicit_tasks": False,
            "num_implicit_tasks": None,
            "temperature": 1.0,
            "kl_weight": 1.0,
            "cost_weight": 1.0,
            "effect_weight": 1.0,
            "latency_weight": 1.0,
            "metrics": ["cost", "effect"]
        }
        self.variational_config = {**default_variational_config, **(variational_config or {})}
        
        # Initialize variational components
        self.variational_encoder = None
        self.metric_decoders = {}
        self.task_embeddings = None
        
        # Configure base model
        self.model = self.model.to(device)
        
        # Initialize base model for non-tuned encoding
        self.base_model = AutoModel.from_pretrained(model_name).to(device)
        self.base_model.eval()
    
    def setup_variational_components(self, num_tasks: int, difficult_prompt: List[str],
                                   model_names: List[str]):
        """
        Setup variational inference components
        
        Args:
            num_tasks: Number of difficulty tasks
            difficult_prompt: List of difficulty descriptions
            model_names: List of model names
        """
        task_embeddings = self.encode_text(difficult_prompt, use_base_model=True, use_grad=False)

        # Setup task embeddings
        if self.variational_config["use_implicit_tasks"]:
            implicit_tasks = self.variational_config["num_implicit_tasks"] or num_tasks
            self.task_embeddings = nn.Parameter(
                torch.randn(implicit_tasks, self.model_dim, device=self.device)
            )
            self.num_tasks = implicit_tasks
        else:
            self.task_embeddings = task_embeddings.to(self.device)
            self.num_tasks = num_tasks
        
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
    
    def encode_text(self, 
                    texts: Union[str, List[str]], 
                    batch_size: int = 32,
                    use_base_model: bool = False,
                    use_grad: bool = False) -> torch.Tensor:
        """Encode text(s) using mean pooling with flexible options and batch processing"""
        if isinstance(texts, str):
            texts = [texts]
            
        model = self.base_model if use_base_model else self.model
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            if use_grad:
                outputs = model(**encoded)
            else:
                with torch.no_grad():
                    outputs = model(**encoded)
            
            batch_embeddings = mean_pooling(outputs, encoded['attention_mask'])
            all_embeddings.append(batch_embeddings)
        
        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings
    
    def forward_variational(self, query_embeddings: torch.Tensor, 
                           true_metrics: Optional[Dict[str, torch.Tensor]] = None,
                           temperature: Optional[float] = None,
                           true_task_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through variational model (same as single-label version)
        
        Args:
            query_embeddings: (batch_size, model_dim)
            true_metrics: Dict[str, (batch_size,)] - true metrics for training
            temperature: Temperature for Gumbel-Softmax
            true_task_ids: (batch_size, num_tasks) - multi-label binary tensor for true task IDs
            
        Returns:
            Dictionary containing task probabilities and metric predictions
        """
        batch_size = query_embeddings.size(0)
        
        # Get task distribution q_φ(t|q) and encoded embeddings
        task_logits, encoded_queries, encoded_tasks = self.variational_encoder(
            query_embeddings, self.task_embeddings
        )
        task_probs = F.softmax(task_logits, dim=-1)  # [batch_size, num_tasks]
        
        # Predict metrics for each task embedding separately
        metrics_to_predict = self.variational_config["metrics"]
        metric_predictions = {}
        
        batch_size = query_embeddings.size(0)
        num_tasks = self.task_embeddings.size(0)
        
        expanded_queries = encoded_queries.unsqueeze(1).expand(-1, num_tasks, -1)
        expanded_tasks = encoded_tasks.unsqueeze(0).expand(batch_size, -1, -1)
        
        for metric in metrics_to_predict:
            all_model_predictions = []
            
            for model_idx in range(self.num_models):
                decoder_key = f"{metric}_model_{model_idx}"
                decoder = self.metric_decoders[decoder_key]
                
                flat_queries = expanded_queries.reshape(-1, self.model_dim)
                flat_tasks = expanded_tasks.reshape(-1, self.model_dim)
                
                task_preds = decoder(flat_tasks)  # [batch_size * num_tasks, 1]
                task_preds = task_preds.reshape(batch_size, num_tasks)  # [batch_size, num_tasks]
                
                # Use task probability weighted sum
                weighted_pred = torch.sum(task_preds * task_probs, dim=1, keepdim=True)  # [batch_size, 1]
                all_model_predictions.append(weighted_pred)
            
            metric_predictions[metric] = torch.cat(all_model_predictions, dim=-1)  # [batch_size, n_models]

        return {
            "task_logits": task_logits,  # [batch_size, num_tasks]
            "task_probs": task_probs,    # [batch_size, num_tasks]
            "metric_predictions": metric_predictions,  # Dict[str, Tensor[batch_size, n_models]]
        }
    
    def compute_elbo_loss_multilabel(self, forward_output: Dict[str, torch.Tensor], 
                                   true_metrics: Dict[str, torch.Tensor],
                                   true_task_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss for multi-label classification
        
        Args:
            forward_output: Output from forward_variational
            true_metrics: Dict[str, (batch_size, n_models)] - true normalized metrics
            true_task_labels: (batch_size, num_tasks) - binary tensor indicating which tasks are active
            
        Returns:
            Dictionary containing loss components
        """
        # Reconstruction terms for each metric
        metric_predictions = forward_output["metric_predictions"]
        
        reconstruction_losses = {}
        mape_losses = {}
        total_reconstruction_loss = 0
        total_mape = 0
        
        for metric in metric_predictions.keys():
            if metric in true_metrics:
                pred = metric_predictions[metric]
                true_val = true_metrics[metric]
                
                # MSE loss
                loss = F.mse_loss(pred, true_val)
                reconstruction_losses[metric] = loss
                
                # Calculate MAPE
                with torch.no_grad():
                    mape = torch.mean(torch.abs(true_val - pred))
                    mape_losses[metric] = mape
                    total_mape += mape
                
                # Apply metric-specific weights
                weight_key = f"{metric}_weight"
                weight = self.variational_config[weight_key]
                total_reconstruction_loss += weight * loss
        
        # KL divergence term for multi-label: KL(q(t|q) || p(t|q))
        # For multi-label, we use independent Bernoulli distributions for each task
        q_probs = forward_output["task_probs"]  # [batch_size, num_tasks]
        p_probs = true_task_labels.float()  # [batch_size, num_tasks] - binary labels
        
        # Normalize p_probs to sum to 1 for each sample (or use uniform prior)
        # Option 1: Use true labels as prior (normalized)
        p_probs_normalized = p_probs / (p_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        # Option 2: Use uniform prior (alternative)
        # p_probs_normalized = torch.ones_like(q_probs) / self.num_tasks
        
        # KL divergence for each task independently, then sum
        # KL(q||p) = sum_t q_t * log(q_t / p_t)
        kl_div_per_task = q_probs * torch.log(q_probs / (p_probs_normalized + 1e-8) + 1e-8)
        kl_div = kl_div_per_task.sum(dim=1).mean()  # Average over batch
        
        # ELBO = Reconstruction - KL
        elbo = -total_reconstruction_loss - self.variational_config["kl_weight"] * kl_div
        total_loss = -elbo  # Minimize negative ELBO
        
        # Calculate multi-label accuracy (subset accuracy)
        # IMPORTANT: Since we use softmax, probabilities sum to 1.
        # For multi-label classification with softmax, we need to use top-k selection
        # instead of threshold-based selection (which would fail when k >= 2).
        # 
        # Why threshold doesn't work with softmax:
        # - If true labels = [A, B] (2 labels), softmax can output at most P(A)=0.5, P(B)=0.5
        # - With threshold > 0.5, neither A nor B would be selected
        # - Solution: Select top-k predictions where k = number of true labels
        with torch.no_grad():
            # Get the number of true labels for each sample
            num_true_labels = p_probs.sum(dim=1).long()  # [batch_size]
            
            # For each sample, select top-k predictions where k = number of true labels
            pred_binary = torch.zeros_like(q_probs)
            for i in range(q_probs.size(0)):
                k = num_true_labels[i].item()
                if k > 0:
                    # Get top-k indices for this sample
                    # topk returns (values, indices), we only need indices
                    _, top_k_indices = torch.topk(q_probs[i], k, dim=0)
                    pred_binary[i, top_k_indices] = 1.0
                # If k == 0 (no true labels), pred_binary[i] remains all zeros
            
            # Subset accuracy: exact match (all predicted labels match true labels)
            # This is strict: requires perfect prediction of all labels
            subset_accuracy = (pred_binary == p_probs).all(dim=1).float().mean().item()
            
            # Hamming accuracy: average over tasks (label-wise accuracy)
            # This is lenient: averages accuracy across individual labels
            hamming_accuracy = (pred_binary == p_probs).float().mean().item()
        
        result = {
            "total_loss": total_loss,
            "total_reconstruction_loss": total_reconstruction_loss,
            "total_mape": total_mape / len(metric_predictions.keys()),
            "kl_divergence": kl_div,
            "elbo": elbo,
            "subset_accuracy": subset_accuracy,
            "hamming_accuracy": hamming_accuracy
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
        Prepare data batch for multi-label variational training
        
        Args:
            data: Dictionary containing:
                - queries: List[str] - query list
                - cost: [n_examples, n_models] - cost for each example on each model
                - effectiveness: [n_examples, n_models] - effectiveness for each example on each model
                - difficulty_ids: [n_examples] - list of lists of difficulty IDs (multi-label)
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            dataloader: DataLoader for training
            queries: List of query strings
            metric_names: List of metric names included in the data
            num_tasks: Number of tasks (for creating binary labels)
        """
        query_indices = torch.arange(len(data['queries']))
        
        # Convert multi-label difficulty_ids to binary tensor
        # difficulty_ids is a list of lists, e.g., [["60", "84"], ["398", "399"]]
        # Get num_tasks from setup if available, otherwise infer from data
        if hasattr(self, 'num_tasks') and self.num_tasks is not None:
            num_tasks = self.num_tasks
        else:
            # Infer from data: collect all unique task IDs
            all_task_ids = set()
            for diff_id_list in data['difficulty_ids']:
                if isinstance(diff_id_list, list):
                    all_task_ids.update([str(item) for item in diff_id_list])
                else:
                    all_task_ids.add(str(diff_id_list))
            num_tasks = len(all_task_ids) if all_task_ids else 100  # Default to 100 if empty
        
        # Create binary label tensor: (n_examples, num_tasks)
        binary_labels = torch.zeros(len(data['queries']), num_tasks, dtype=torch.float32)
        for i, diff_ids in enumerate(data['difficulty_ids']):
            # Handle both list and single value cases
            if isinstance(diff_ids, list):
                diff_id_list = diff_ids
            else:
                diff_id_list = [diff_ids] if diff_ids else []
            
            for diff_id in diff_id_list:
                try:
                    task_idx = int(diff_id)
                    if 0 <= task_idx < num_tasks:
                        binary_labels[i, task_idx] = 1.0
                except (ValueError, TypeError):
                    continue
        
        # Prepare metric data
        metrics = {}
        for metric in self.variational_config["metrics"]:
            metrics[metric] = torch.tensor(data[metric], dtype=torch.float32)  # [n_examples, n_models]
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            query_indices,  # [n_examples]
            binary_labels,  # [n_examples, num_tasks]
            *[metrics[m] for m in self.variational_config["metrics"]]  # [n_examples, n_models] for each metric
        )
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        return dataloader, data['queries'], self.variational_config["metrics"], num_tasks
    
    def load_model(self, save_dir: str):
        """Load saved model"""
        best_state = torch.load(os.path.join(save_dir, "variational_model_multiple.pt"))
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
        Train variational model using ELBO objective with multi-label support
        
        Args:
            train_data: Dictionary containing training data with multi-label difficulty_ids
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
        if save_dir and os.path.exists(os.path.join(save_dir, "variational_model_multiple.pt")):
            self.load_model(save_dir)
            print("Loaded variational model from", save_dir)
            return self.task_embeddings
        
        # Prepare optimizers
        variational_params = list(self.variational_encoder.parameters()) + \
                           list(self.metric_decoders.parameters())
        
        if self.variational_config["use_implicit_tasks"]:
            variational_params.append(self.task_embeddings)
        
        optimizer = torch.optim.AdamW(variational_params, lr=learning_rate)
        
        train_loader, train_queries, metric_names, num_tasks = self.prepare_variational_data_batch(
            train_data, batch_size, shuffle=True
        )
        
        # Training loop
        best_loss = float('inf')
        best_state = None
        patience = 3
        patience_counter = 0
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            total_subset_accuracy = 0
            total_hamming_accuracy = 0
            epoch_loss_outputs = []
            
            # Set models to training mode
            self.variational_encoder.train()
            for decoder in self.metric_decoders.values():
                decoder.train()
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Parse batch data
                query_indices = batch[0].to(self.device)  # [batch_size]
                task_labels = batch[1].to(self.device)    # [batch_size, num_tasks] - binary labels
                
                # Get metric values [batch_size, n_models]
                batch_metrics = {}
                for i, metric_name in enumerate(metric_names):
                    batch_metrics[metric_name] = batch[i + 2].to(self.device)
                
                # Get query embeddings
                batch_queries = [train_queries[i] for i in query_indices.cpu()]
                query_embeddings = self.encode_text(batch_queries, use_base_model=True, use_grad=False)
                
                # Forward pass
                forward_output = self.forward_variational(
                    query_embeddings=query_embeddings,
                    true_metrics=batch_metrics,
                    true_task_ids=task_labels
                )
                
                # Compute ELBO loss
                loss_output = self.compute_elbo_loss_multilabel(
                    forward_output, batch_metrics, task_labels
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss_output["total_loss"].backward()
                optimizer.step()
                
                epoch_loss_outputs.append(loss_output)
                
                # Accumulate losses
                total_loss += loss_output["total_loss"].item()
                total_recon_loss += loss_output["total_reconstruction_loss"].item()
                total_kl_loss += loss_output["kl_divergence"].item()
                total_subset_accuracy += loss_output["subset_accuracy"]
                total_hamming_accuracy += loss_output["hamming_accuracy"]
            
            # Average losses
            avg_loss = total_loss / len(train_loader)
            avg_recon_loss = total_recon_loss / len(train_loader)
            avg_kl_loss = total_kl_loss / len(train_loader)
            avg_subset_accuracy = total_subset_accuracy / len(train_loader)
            avg_hamming_accuracy = total_hamming_accuracy / len(train_loader)
            
            # Calculate MAPE for each metric
            metric_mapes = {}
            for metric in metric_names:
                metric_mapes[metric] = sum(batch_loss_output[f"{metric}_mape"].item() 
                                        for batch_loss_output in epoch_loss_outputs) / len(train_loader)
            
            print(f"\nEpoch {epoch+1}:")
            print(f"  Training Metrics:")
            print(f"    Total Loss: {avg_loss:.4f}, Reconstruction Loss: {avg_recon_loss:.4f}, KL Divergence: {avg_kl_loss:.4f}")
            print(f"    Subset Accuracy: {avg_subset_accuracy:.4f}, Hamming Accuracy: {avg_hamming_accuracy:.4f}")
            print("    MAPE for each metric:")
            for metric, mape in metric_mapes.items():
                print(f"      {metric}: {mape:.2f}")
            
            # Validation
            val_metrics = self.evaluate_variational(val_data, model_names, batch_size)
            print(f"\n  Validation Metrics:")
            print(f"    Total Loss: {val_metrics['loss']:.4f}, Reconstruction Loss: {val_metrics['recon_loss']:.4f}")
            print(f"    KL Divergence: {val_metrics['kl_loss']:.4f}")
            print(f"    Subset Accuracy: {val_metrics['subset_accuracy']:.4f}, Hamming Accuracy: {val_metrics['hamming_accuracy']:.4f}")
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
            torch.save(best_state, os.path.join(save_dir, "variational_model_multiple.pt"))
        
        return self.task_embeddings
    
    def evaluate_variational(self, val_data: Dict[str, Any], model_names: List[str], 
                           batch_size: int = 32) -> Dict[str, Any]:
        """Evaluate variational model on validation data"""
        self.variational_encoder.eval()
        for decoder in self.metric_decoders.values():
            decoder.eval()
        
        val_loader, val_queries, val_metric_names, num_tasks = self.prepare_variational_data_batch(
            val_data, batch_size, shuffle=False
        )
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_subset_accuracy = 0
        total_hamming_accuracy = 0
        metric_mapes = {metric: 0.0 for metric in val_metric_names}
        
        with torch.no_grad():
            for batch in val_loader:
                query_indices = batch[0].to(self.device)
                task_labels = batch[1].to(self.device)
                
                batch_metrics = {}
                for i, metric_name in enumerate(val_metric_names):
                    batch_metrics[metric_name] = batch[i + 2].to(self.device)
                
                batch_queries = [val_queries[i] for i in query_indices.cpu()]
                query_embeddings = self.encode_text(batch_queries, use_base_model=True, use_grad=False)
                
                forward_output = self.forward_variational(
                    query_embeddings=query_embeddings,
                    true_metrics=batch_metrics,
                    true_task_ids=task_labels
                )
                
                loss_output = self.compute_elbo_loss_multilabel(forward_output, batch_metrics, task_labels)
                total_loss += loss_output["total_loss"].item()
                total_recon_loss += loss_output["total_reconstruction_loss"].item()
                total_kl_loss += loss_output["kl_divergence"].item()
                total_subset_accuracy += loss_output["subset_accuracy"]
                total_hamming_accuracy += loss_output["hamming_accuracy"]
                
                for metric in val_metric_names:
                    metric_mapes[metric] += loss_output[f"{metric}_mape"].item()
        
        num_batches = len(val_loader)
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
            'subset_accuracy': total_subset_accuracy / num_batches,
            'hamming_accuracy': total_hamming_accuracy / num_batches,
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
            
            forward_output = self.forward_variational(
                query_embeddings=query_embeddings,
                true_metrics=None,
                true_task_ids=None
            )
            
            predictions = forward_output["metric_predictions"][metric]
            return predictions
    
    def get_task_distribution(self, query: Union[str, List[str]]) -> torch.Tensor:
        """
        Get task distribution q_φ(t|q) using variational inference
        
        Args:
            query: Input query or list of queries
            
        Returns:
            Tensor with shape (batch_size, num_tasks) containing task probabilities
        """
        if isinstance(query, str):
            query = [query]
        
        self.variational_encoder.eval()
        with torch.no_grad():
            query_embeddings = self.encode_text(query, use_base_model=True, use_grad=False)
            forward_output = self.forward_variational(
                query_embeddings=query_embeddings,
                true_metrics=None,
                true_task_ids=None
            )
            return forward_output["task_probs"]

