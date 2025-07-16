"""
Configuration settings for the hybrid ICD/HCC disease prediction system.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """Configuration class for model parameters and settings."""
    
    # Embedding Configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    fallback_embedding: bool = True
    
    # Patient Vector Configuration
    patient_vector_aggregation: str = "mean"  # "mean", "attention", "max"
    demographic_features: bool = True
    
    # Candidate Generation
    max_candidates: int = 1000
    min_frequency_threshold: int = 10
    filter_rare_codes: bool = True
    
    # Feature Engineering
    feature_types: List[str] = None
    
    def __post_init__(self):
        if self.feature_types is None:
            self.feature_types = [
                "cosine_similarity",
                "conditional_probability", 
                "cms_risk_score",
                "age",
                "gender",
                "icd_count",
                "entropy",
                "rule_triggers",
                "cluster_overlap",
                "embedding_variance"
            ]
        
        if self.lgb_params is None:
            self.lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        
        if self.mlp_hidden_sizes is None:
            self.mlp_hidden_sizes = [512, 256, 128]
        
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                "accuracy_at_k",
                "map_at_k", 
                "recall_at_k",
                "hit_rate",
                "ndcg_at_k"
            ]
        
        if self.evaluation_k_values is None:
            self.evaluation_k_values = [5, 10, 15, 20]
    
    # Rule Engine Configuration
    enable_rules: bool = True
    rule_confidence_threshold: float = 0.7
    
    # ML Model Configuration
    primary_model: str = "lightgbm"  # "lightgbm", "mlp", "transformer"
    
    # LightGBM Parameters
    lgb_params: Dict[str, Any] = None
    
    # MLP Parameters
    mlp_hidden_sizes: List[int] = None
    mlp_learning_rate: float = 0.001
    mlp_epochs: int = 100
    mlp_batch_size: int = 256
    
    # Training Configuration
    train_test_split_ratio: float = 0.8
    negative_sampling_ratio: int = 7  # false candidates per true label
    min_patient_codes: int = 2
    max_patient_codes: int = 20
    cross_validation_folds: int = 5
    
    # Prediction Configuration
    top_k_predictions: int = 15
    confidence_threshold: float = 0.1
    return_explanations: bool = True
    
    # Evaluation Configuration
    evaluation_metrics: List[str] = None
    
    evaluation_k_values: List[int] = None
    
    # File Paths
    data_dir: str = "data"
    model_dir: str = "models"
    cache_dir: str = "cache"
    
    # Data Leakage Prevention
    strict_patient_split: bool = True
    exclude_target_from_input: bool = True
    validate_no_future_leakage: bool = True
    
    # Logging and Debugging
    log_level: str = "INFO"
    enable_feature_importance: bool = True
    save_intermediate_results: bool = False


# Global configuration instance
config = ModelConfig()

def update_config(**kwargs):
    """Update configuration parameters."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

def get_config():
    """Get the current configuration."""
    return config

def reset_config():
    """Reset configuration to defaults."""
    global config
    config = ModelConfig()