"""
Scorer: ML models for scoring disease prediction candidates.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
import pickle
from pathlib import Path

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from config.model_config import get_config

logger = logging.getLogger(__name__)

class Scorer:
    """
    ML models for scoring and ranking disease prediction candidates.
    Supports LightGBM (primary) and MLP (alternative) models.
    """
    
    def __init__(self, model_type: Optional[str] = None):
        """
        Initialize the Scorer.
        
        Args:
            model_type: Type of model to use ('lightgbm', 'mlp', 'ensemble')
        """
        self.config = get_config()
        self.model_type = model_type or self.config.primary_model
        
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        
        # Model performance tracking
        self.training_metrics = {}
        self.validation_metrics = {}
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on configuration."""
        
        if self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                logger.error("LightGBM not available, falling back to MLP")
                self.model_type = 'mlp'
            else:
                self.model = lgb.LGBMClassifier(**self.config.lgb_params)
                logger.info("Initialized LightGBM classifier")
        
        if self.model_type == 'mlp':
            if not SKLEARN_AVAILABLE:
                raise ImportError("Neither LightGBM nor sklearn available")
            
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(self.config.mlp_hidden_sizes),
                learning_rate_init=self.config.mlp_learning_rate,
                max_iter=self.config.mlp_epochs,
                batch_size=self.config.mlp_batch_size,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            self.scaler = StandardScaler()
            logger.info("Initialized MLP classifier")
        
        elif self.model_type == 'ensemble':
            # Initialize both models for ensemble
            if LIGHTGBM_AVAILABLE and SKLEARN_AVAILABLE:
                self.lgb_model = lgb.LGBMClassifier(**self.config.lgb_params)
                self.mlp_model = MLPClassifier(
                    hidden_layer_sizes=tuple(self.config.mlp_hidden_sizes),
                    learning_rate_init=self.config.mlp_learning_rate,
                    max_iter=self.config.mlp_epochs,
                    batch_size=self.config.mlp_batch_size,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
                self.scaler = StandardScaler()
                logger.info("Initialized ensemble (LightGBM + MLP)")
            else:
                logger.error("Cannot create ensemble, falling back to available model")
                self.model_type = 'lightgbm' if LIGHTGBM_AVAILABLE else 'mlp'
                self._initialize_model()
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray, 
              feature_names: Optional[List[str]] = None,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels (binary: 1 = relevant disease, 0 = not relevant)
            feature_names: Names of features
            validation_data: Optional validation data (X_val, y_val)
            sample_weight: Optional sample weights
            
        Returns:
            Training metrics
        """
        if X.shape[0] == 0:
            raise ValueError("Training data is empty")
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info(f"Training {self.model_type} model with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split validation data if not provided
        if validation_data is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            if sample_weight is not None:
                sample_weight_train = sample_weight[:len(X_train)]
            else:
                sample_weight_train = None
        else:
            X_train, y_train = X, y
            X_val, y_val = validation_data
            sample_weight_train = sample_weight
        
        # Train based on model type
        if self.model_type == 'lightgbm':
            self._train_lightgbm(X_train, y_train, X_val, y_val, sample_weight_train)
        elif self.model_type == 'mlp':
            self._train_mlp(X_train, y_train, X_val, y_val, sample_weight_train)
        elif self.model_type == 'ensemble':
            self._train_ensemble(X_train, y_train, X_val, y_val, sample_weight_train)
        
        self.is_trained = True
        
        # Calculate training metrics
        train_probs = self.predict_proba(X_train)
        val_probs = self.predict_proba(X_val)
        
        self.training_metrics = self._calculate_metrics(y_train, train_probs)
        self.validation_metrics = self._calculate_metrics(y_val, val_probs)
        
        logger.info(f"Training completed. Validation AUC: {self.validation_metrics.get('auc', 0.0):.4f}")
        
        return self.validation_metrics
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, sample_weight):
        """Train LightGBM model."""
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
    
    def _train_mlp(self, X_train, y_train, X_val, y_val, sample_weight):
        """Train MLP model."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    
    def _train_ensemble(self, X_train, y_train, X_val, y_val, sample_weight):
        """Train ensemble model."""
        # Train LightGBM
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.lgb_model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Train MLP
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.mlp_model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability scores for positive class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.model_type == 'lightgbm':
            probs = self.model.predict_proba(X)[:, 1]
        elif self.model_type == 'mlp':
            X_scaled = self.scaler.transform(X)
            probs = self.model.predict_proba(X_scaled)[:, 1]
        elif self.model_type == 'ensemble':
            # Average predictions from both models
            lgb_probs = self.lgb_model.predict_proba(X)[:, 1]
            X_scaled = self.scaler.transform(X)
            mlp_probs = self.mlp_model.predict_proba(X_scaled)[:, 1]
            probs = 0.7 * lgb_probs + 0.3 * mlp_probs  # Weighted average
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return probs
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
            
        Returns:
            Binary predictions
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}
        
        try:
            # AUC
            if len(np.unique(y_true)) > 1:
                metrics['auc'] = roc_auc_score(y_true, y_probs)
            else:
                metrics['auc'] = 0.5
            
            # Accuracy at different thresholds
            for threshold in [0.3, 0.5, 0.7]:
                y_pred = (y_probs >= threshold).astype(int)
                metrics[f'accuracy_at_{threshold}'] = accuracy_score(y_true, y_pred)
            
            # Precision-Recall metrics
            if len(np.unique(y_true)) > 1:
                precision, recall, _ = precision_recall_curve(y_true, y_probs)
                metrics['avg_precision'] = np.mean(precision)
                metrics['avg_recall'] = np.mean(recall)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            metrics = {'auc': 0.0, 'accuracy_at_0.5': 0.0}
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            return {}
        
        try:
            if self.model_type == 'lightgbm':
                importances = self.model.feature_importances_
            elif self.model_type == 'mlp':
                # For MLP, use the absolute values of the first layer weights
                if hasattr(self.model, 'coefs_'):
                    importances = np.abs(self.model.coefs_[0]).mean(axis=1)
                else:
                    return {}
            elif self.model_type == 'ensemble':
                # Average importances from both models
                lgb_importances = self.lgb_model.feature_importances_
                if hasattr(self.mlp_model, 'coefs_'):
                    mlp_importances = np.abs(self.mlp_model.coefs_[0]).mean(axis=1)
                    importances = 0.7 * lgb_importances + 0.3 * mlp_importances
                else:
                    importances = lgb_importances
            else:
                return {}
            
            # Normalize importances
            importances = importances / np.sum(importances) if np.sum(importances) > 0 else importances
            
            return dict(zip(self.feature_names, importances))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for cross-validation")
            return {}
        
        try:
            if self.model_type == 'lightgbm':
                model = lgb.LGBMClassifier(**self.config.lgb_params)
                scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            elif self.model_type == 'mlp':
                model = MLPClassifier(
                    hidden_layer_sizes=tuple(self.config.mlp_hidden_sizes),
                    learning_rate_init=self.config.mlp_learning_rate,
                    max_iter=self.config.mlp_epochs,
                    random_state=42
                )
                X_scaled = StandardScaler().fit_transform(X)
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
            else:
                return {}
            
            return {
                'cv_mean': np.mean(scores),
                'cv_std': np.std(scores),
                'cv_scores': scores.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return {}
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'config': self.config.__dict__
        }
        
        if self.model_type == 'lightgbm':
            model_data['model'] = self.model
        elif self.model_type == 'mlp':
            model_data['model'] = self.model
            model_data['scaler'] = self.scaler
        elif self.model_type == 'ensemble':
            model_data['lgb_model'] = self.lgb_model
            model_data['mlp_model'] = self.mlp_model
            model_data['scaler'] = self.scaler
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.validation_metrics = model_data['validation_metrics']
        
        if self.model_type == 'lightgbm':
            self.model = model_data['model']
        elif self.model_type == 'mlp':
            self.model = model_data['model']
            self.scaler = model_data['scaler']
        elif self.model_type == 'ensemble':
            self.lgb_model = model_data['lgb_model']
            self.mlp_model = model_data['mlp_model']
            self.scaler = model_data['scaler']
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names[:10],  # First 10 features
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics
        }