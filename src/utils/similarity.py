"""
Similarity calculation utilities for text embeddings and cosine similarity
"""
import numpy as np
from typing import List, Tuple, Union, Optional, Dict, Any
import torch

from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import AutoTokenizer, AutoModel
from src.utils.finetune_trainer import SentenceTransformerModel
import torch.nn.functional as F
from safetensors.torch import load_file


class SimilarityTool:
    """Text similarity calculation tool using sentence transformers"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize similarity tool
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
            
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.model is None:
            # Fallback: simple word overlap similarity
            return self._fallback_similarity(text1, text2)
            
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(emb1, emb2)[0][0]
        return float(similarity_score)
    
    def batch_similarity(self, query: str, texts: List[str]) -> List[float]:
        """
        Calculate similarity between query and multiple texts
        
        Args:
            query: Query text
            texts: List of texts to compare with
            
        Returns:
            List of similarity scores
        """
        if self.model is None:
            return [self._fallback_similarity(query, text) for text in texts]
            
        query_emb = self.model.encode(query, convert_to_tensor=True)
        text_embs = self.model.encode(texts, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_emb, text_embs)[0]
        return [float(sim) for sim in similarities]
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """
        Fallback similarity calculation using word overlap
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0 

    def encode_batch(self, texts: List[str], batch_size: int = 32, convert_to_numpy: bool = False) -> np.ndarray:
        """
        Encode a list of texts in batches
        
        Args:
            texts: List of texts to encode
            batch_size: Size of each batch for encoding
            convert_to_numpy: Whether to convert the output to numpy array
            
        Returns:
            Array of text embeddings with shape (n_texts, embedding_dim)
            If model not available, returns None
        """
        if self.model is None:
            return None
            
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            try:
                batch_texts = texts[i:i + batch_size]
                embeddings = self.model.encode(
                    batch_texts, 
                    convert_to_tensor=not convert_to_numpy,
                    batch_size=len(batch_texts),
                    # max_length=512
                )
                all_embeddings.append(embeddings)
            except Exception as e:
                print(texts)
                
        if convert_to_numpy:
            return np.vstack(all_embeddings)
        else:
            return torch.cat(all_embeddings, dim=0) 

    def encode(self, text: str, convert_to_numpy: bool = False) -> torch.Tensor:
        """
        Encode a single text
        """
        if len(text.split(" ")) > 2024:
            text = " ".join(text.split(" ")[:2024])
        return self.encode_batch([text], convert_to_numpy=convert_to_numpy)[0]


def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    
    Args:
        model_output: Model output containing token embeddings
        attention_mask: Attention mask for tokens
        
    Returns:
        Pooled sentence embeddings
    """
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SimilarityTool_hugg:
    """Text similarity calculation tool using HuggingFace transformers"""
    
    def __init__(
        self, 
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        device: Optional[str] = None,
        max_length: int = 512,
        model_path: str = None
    ):
        """
        Initialize similarity tool with HuggingFace transformers
        
        Args:
            model_name: Name of the HuggingFace model
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        if model_path is not None:
            self.load_model_from_path(model_path)
        else:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded HuggingFace model: {model_name} on {self.device}")

    def encode_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        convert_to_numpy: bool = False,
        normalize_embeddings: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode a list of texts in batches
        
        Args:
            texts: List of texts to encode
            batch_size: Size of each batch for encoding
            convert_to_numpy: Whether to convert the output to numpy array
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            Array of text embeddings with shape (n_texts, embedding_dim)
            If model not available, returns None
        """

        # Truncate texts to max_length (by words)
        truncated_texts = []
        for text in texts:
            if len(text.split()) > self.max_length:
                text = " ".join(text.split()[:self.max_length])
            truncated_texts.append(text)
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(truncated_texts), batch_size):
            
            batch_texts = truncated_texts[i:i + batch_size]
            
            # Tokenize batch
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt',
                max_length=self.max_length  # Model's max sequence length
            )
            
            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Compute embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Perform mean pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings if requested
            if normalize_embeddings:
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            # Move back to CPU for storage
            sentence_embeddings = sentence_embeddings.cpu()
            all_embeddings.append(sentence_embeddings)
                

        # Concatenate all embeddings
        final_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return final_embeddings.numpy()
        else:
            return final_embeddings
    
    def encode(
        self, 
        text: str, 
        convert_to_numpy: bool = False,
        normalize_embeddings: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode a single text
        
        Args:
            text: Text to encode
            convert_to_numpy: Whether to convert the output to numpy array
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            Text embedding tensor or array
        """
        result = self.encode_batch([text], convert_to_numpy=convert_to_numpy, normalize_embeddings=normalize_embeddings)
    
        return result[0]
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between -1 and 1 (normalized embeddings give 0 to 1)
        """
        if self.model is None:
            # Fallback: simple word overlap similarity
            return self._fallback_similarity(text1, text2)
        
        # Encode both texts
        emb1 = self.encode(text1, convert_to_numpy=False, normalize_embeddings=True)
        emb2 = self.encode(text2, convert_to_numpy=False, normalize_embeddings=True)
        
        if emb1 is None or emb2 is None:
            return self._fallback_similarity(text1, text2)
        
        # Calculate cosine similarity
        similarity_score = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return float(similarity_score)
    
    def batch_similarity(self, query: str, texts: List[str]) -> List[float]:
        """
        Calculate similarity between query and multiple texts
        
        Args:
            query: Query text
            texts: List of texts to compare with
            
        Returns:
            List of similarity scores
        """
        if self.model is None:
            return [self._fallback_similarity(query, text) for text in texts]
        
        # Encode query
        query_emb = self.encode(query, convert_to_numpy=False, normalize_embeddings=True)
        if query_emb is None:
            return [self._fallback_similarity(query, text) for text in texts]
        
        # Encode all texts
        text_embs = self.encode_batch(texts, convert_to_numpy=False, normalize_embeddings=True)
        if text_embs is None:
            return [self._fallback_similarity(query, text) for text in texts]
        
        # Calculate similarities
        query_emb = query_emb.unsqueeze(0)  # Add batch dimension
        similarities = torch.cosine_similarity(query_emb, text_embs, dim=1)
        
        return [float(sim) for sim in similarities]
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """
        Fallback similarity calculation using word overlap
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_length": self.max_length,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }
    
    def load_model_from_path(self, model_path: str) -> None:
        """
        Load model weights from a local directory
        
        Args:
            model_path: Path to the directory containing model weights
        """
        try:
            # Create instance of our custom SentenceTransformerModel
            sentence_transformer = SentenceTransformerModel(model_name=self.model_name)
            
            # Load the saved state dict using safetensors
            state_dict = load_file(f"{model_path}/model.safetensors")
            sentence_transformer.load_state_dict(state_dict)
            
            # Get the tokenizer and model
            self.tokenizer = sentence_transformer.tokenizer
            self.model = sentence_transformer.model
            
            # Move model to specified device
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Successfully loaded model from: {model_path}")
            print(f"Model loaded on device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            raise



def masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate standard deviation of values considering a mask tensor.
    
    Args:
        values: Input tensor of values
        mask: Tensor of same shape as values with 1s for valid values and 0s for masked values
        
    Returns:
        Standard deviation of unmasked values. Returns 0 if all values are masked.
    """     
    # Count number of unmasked elements
    valid_count = mask.sum()
    
    # If all elements are masked or only one element, return 0
    if valid_count <= 1:
        return torch.tensor(0.0)
        
    # Calculate mean of unmasked values
    mean = masked_mean(values, mask)
    
    # Calculate squared differences from mean
    squared_diff = (values - mean) ** 2
    
    # Apply mask and calculate mean of squared differences
    masked_squared_diff = squared_diff * mask
    variance = masked_squared_diff.sum() / (valid_count - 1)  # Using n-1 for sample standard deviation
    
    return torch.sqrt(variance)

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate mean of values considering a mask tensor.
    
    Args:
        values: Input tensor of values
        mask: Tensor of same shape as values with 1s for valid values and 0s for masked values
        
    Returns:
        Mean of unmasked values. Returns 0 if all values are masked.
    """
    # Multiply values by mask to zero out masked values
    masked_values = values * mask
    
    # Count number of unmasked elements
    valid_count = mask.sum()
    
    # If all elements are masked, return 0
    if valid_count == 0:
        return torch.tensor(0.0)
        
    # Sum masked values and divide by count of unmasked elements
    return masked_values.sum() / valid_count
