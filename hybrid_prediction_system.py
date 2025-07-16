"""
Hybrid Disease Prediction System Integration
Integrates the new prediction system with the existing Flask application.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any

# Import our new modules
from config.model_config import get_config, update_config
from models.embedder import ICDEmbedder
from models.patient_vector import PatientVectorBuilder
from models.candidate_generator import CandidateGenerator
from models.feature_builder import FeatureBuilder
from models.rule_engine import RuleTriggerEngine
from models.scorer import Scorer
from models.ranker import Ranker
from models.evaluator import Evaluator
from utils.data_loader import DataLoader
from utils.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)

class HybridDiseasePredictionSystem:
    """
    Main class that integrates all components of the hybrid disease prediction system.
    """
    
    def __init__(self, 
                 icd_data_path: str = "icd_hcc.csv",
                 model_cache_dir: str = "cache"):
        """
        Initialize the hybrid prediction system.
        
        Args:
            icd_data_path: Path to ICD-HCC mapping data
            model_cache_dir: Directory to cache models and embeddings
        """
        self.config = get_config()
        self.cache_dir = Path(model_cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        
        # Load ICD data
        try:
            self.icd_data = self.data_loader.load_icd_master_data(icd_data_path)
            logger.info(f"Loaded {len(self.icd_data)} ICD codes")
        except Exception as e:
            logger.error(f"Error loading ICD data: {str(e)}")
            # Use sample data as fallback
            self.icd_data = self.data_loader.load_icd_master_data("data/sample_data/sample_icd_hcc_mapping.csv")
        
        # Initialize prediction components
        self.embedder = None
        self.patient_vector_builder = None
        self.candidate_generator = None
        self.feature_builder = None
        self.rule_engine = None
        self.scorer = None
        self.ranker = None
        self.evaluator = None
        
        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components."""
        logger.info("Initializing hybrid prediction system...")
        
        # Initialize embedder and create embeddings
        self.embedder = ICDEmbedder(cache_dir=str(self.cache_dir))
        icd_embeddings = self.embedder.encode_icd_descriptions(self.icd_data)
        
        # Initialize patient vector builder
        self.patient_vector_builder = PatientVectorBuilder(icd_embeddings)
        
        # Initialize rule engine
        self.rule_engine = RuleTriggerEngine()
        
        # Initialize candidate generator (will be updated with patient data when available)
        self.candidate_generator = CandidateGenerator(
            icd_data=self.icd_data,
            patient_data=None,  # Will be set when patient data is loaded
            hcc_data=None
        )
        
        # Initialize feature builder
        self.feature_builder = FeatureBuilder(
            patient_vector_builder=self.patient_vector_builder,
            candidate_generator=self.candidate_generator,
            rule_engine=self.rule_engine,
            icd_embeddings=icd_embeddings
        )
        
        # Initialize scorer (will be trained when patient data is available)
        self.scorer = Scorer()
        
        # Initialize evaluator
        self.evaluator = Evaluator()
        
        logger.info("System initialization completed")
    
    def load_patient_data(self, patient_data: pd.DataFrame):
        """
        Load patient data and update system components.
        
        Args:
            patient_data: DataFrame with patient-ICD associations
        """
        logger.info(f"Loading patient data: {len(patient_data)} records")
        
        # Update candidate generator with patient data
        self.candidate_generator = CandidateGenerator(
            icd_data=self.icd_data,
            patient_data=patient_data,
            hcc_data=None
        )
        
        # Update feature builder
        self.feature_builder = FeatureBuilder(
            patient_vector_builder=self.patient_vector_builder,
            candidate_generator=self.candidate_generator,
            rule_engine=self.rule_engine,
            icd_embeddings=self.embedder.encode_icd_descriptions(self.icd_data)
        )
        
        # Initialize ranker (requires scorer to be trained first)
        self.ranker = Ranker(
            scorer=self.scorer,
            feature_builder=self.feature_builder,
            rule_engine=self.rule_engine
        )
    
    def predict_diseases(self, 
                        input_icds: List[str],
                        age: Optional[int] = None,
                        gender: Optional[str] = None,
                        top_k: int = 15,
                        include_explanations: bool = True) -> Dict[str, Any]:
        """
        Predict diseases for a patient based on input ICDs and demographics.
        
        Args:
            input_icds: List of input ICD codes
            age: Patient age (optional)
            gender: Patient gender (optional) 
            top_k: Number of top predictions to return
            include_explanations: Whether to include explanations
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.candidate_generator:
            raise ValueError("System not initialized with patient data")
        
        logger.info(f"Predicting diseases for input ICDs: {input_icds}")
        
        try:
            # Validate input ICDs
            valid_icds = [icd for icd in input_icds if icd in self.candidate_generator.all_icd_codes]
            if not valid_icds:
                return {
                    'success': False,
                    'error': 'No valid ICD codes found in the dataset',
                    'predictions': []
                }
            
            # Generate candidates
            candidates = self.candidate_generator.generate_mixed_candidates(
                input_icds=valid_icds,
                max_candidates=self.config.max_candidates
            )
            
            if not candidates:
                return {
                    'success': False,
                    'error': 'No candidates generated',
                    'predictions': []
                }
            
            # If scorer is trained, use ML ranking
            if self.scorer.is_trained and self.ranker:
                predictions = self.ranker.rank_candidates(
                    input_icds=valid_icds,
                    candidates=candidates,
                    age=age,
                    gender=gender,
                    top_k=top_k
                )
            else:
                # Fallback to relevance-based ranking
                predictions = self._fallback_ranking(candidates, valid_icds, top_k)
            
            # Add rule explanations if requested
            if include_explanations and self.rule_engine:
                for pred in predictions:
                    rule_explanations = self.rule_engine.get_rule_explanations(
                        valid_icds, pred['code'], age, gender
                    )
                    pred['rule_explanations'] = rule_explanations
            
            # Apply demographic filtering
            if age or gender:
                predictions = self.candidate_generator.filter_candidates_by_demographics(
                    predictions, age, gender
                )
            
            result = {
                'success': True,
                'input_icds': valid_icds,
                'invalid_icds': [icd for icd in input_icds if icd not in valid_icds],
                'age': age,
                'gender': gender,
                'predictions': predictions[:top_k],
                'model_info': {
                    'model_trained': self.scorer.is_trained,
                    'total_candidates_evaluated': len(candidates),
                    'system_version': '1.0'
                }
            }
            
            # Add summary if predictions exist
            if predictions and self.ranker:
                result['summary'] = self.ranker.get_prediction_summary(predictions, valid_icds)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in disease prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }
    
    def _fallback_ranking(self, candidates: List[Dict], input_icds: List[str], top_k: int) -> List[Dict]:
        """Fallback ranking when ML model is not trained."""
        # Sort by relevance score and frequency
        for candidate in candidates:
            relevance = candidate.get('relevance_score', 0.0)
            frequency = candidate.get('frequency', 0)
            # Combine relevance and log frequency
            combined_score = 0.7 * relevance + 0.3 * min(np.log(frequency + 1) / 10.0, 1.0)
            candidate['combined_score'] = combined_score
            candidate['probability'] = combined_score  # Use as probability estimate
            candidate['confidence_score'] = combined_score
        
        # Sort by combined score
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Add ranking information
        for i, candidate in enumerate(candidates[:top_k]):
            candidate['rank'] = i + 1
        
        return candidates[:top_k]
    
    def train_model(self, 
                   patient_data: pd.DataFrame,
                   max_samples: int = 10000,
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the ML model using patient data.
        
        Args:
            patient_data: Patient data for training
            max_samples: Maximum number of training samples
            validation_split: Fraction for validation
            
        Returns:
            Training results
        """
        logger.info("Starting model training...")
        
        try:
            # Load patient data into the system
            self.load_patient_data(patient_data)
            
            # Create training samples
            samples = self.data_loader.create_leave_one_out_samples(
                patient_data, max_samples_per_patient=5
            )
            
            # Limit samples for training
            if len(samples) > max_samples:
                samples = samples[:max_samples]
            
            # Create negative samples
            all_icds = list(self.candidate_generator.all_icd_codes)
            negative_samples = self.preprocessor.create_negative_samples(
                samples, all_icds, negative_ratio=3
            )
            
            # Combine positive and negative samples
            all_samples = samples + negative_samples
            all_samples = self.preprocessor.balance_dataset(all_samples, target_ratio=0.3)
            
            # Split into train and validation
            train_samples, val_samples, _ = self.preprocessor.create_stratified_splits(
                all_samples, test_size=0.0, val_size=validation_split
            )
            
            # Build feature matrices
            logger.info("Building features for training...")
            X_train = []
            y_train = []
            
            for sample in train_samples:
                try:
                    features = self.feature_builder.build_features(
                        input_icds=sample['input_codes'],
                        candidate_code=sample['target_code'],
                        candidate_type='ICD',
                        age=sample.get('age'),
                        gender=sample.get('gender')
                    )
                    X_train.append(features)
                    y_train.append(1 if sample['is_positive'] else 0)
                except Exception as e:
                    logger.warning(f"Error building features for sample: {str(e)}")
                    continue
            
            if not X_train:
                return {'success': False, 'error': 'No training features could be built'}
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Build validation features
            X_val = []
            y_val = []
            
            for sample in val_samples:
                try:
                    features = self.feature_builder.build_features(
                        input_icds=sample['input_codes'],
                        candidate_code=sample['target_code'],
                        candidate_type='ICD',
                        age=sample.get('age'),
                        gender=sample.get('gender')
                    )
                    X_val.append(features)
                    y_val.append(1 if sample['is_positive'] else 0)
                except Exception:
                    continue
            
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            
            # Train the model
            feature_names = self.feature_builder.get_feature_names()
            training_metrics = self.scorer.train(
                X_train, y_train, 
                feature_names=feature_names,
                validation_data=(X_val, y_val) if len(X_val) > 0 else None
            )
            
            # Initialize ranker now that scorer is trained
            self.ranker = Ranker(
                scorer=self.scorer,
                feature_builder=self.feature_builder,
                rule_engine=self.rule_engine
            )
            
            return {
                'success': True,
                'training_samples': len(train_samples),
                'validation_samples': len(val_samples),
                'feature_count': X_train.shape[1],
                'metrics': training_metrics,
                'feature_importance': self.scorer.get_feature_importance()
            }
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components."""
        return {
            'embedder': {
                'initialized': self.embedder is not None,
                'model_type': getattr(self.embedder, 'model_type', None),
                'embedding_dim': getattr(self.embedder, 'embedding_dim', None)
            },
            'candidate_generator': {
                'initialized': self.candidate_generator is not None,
                'icd_count': len(getattr(self.candidate_generator, 'all_icd_codes', [])),
                'hcc_count': len(getattr(self.candidate_generator, 'all_hcc_codes', []))
            },
            'rule_engine': {
                'initialized': self.rule_engine is not None,
                'rule_count': len(getattr(self.rule_engine, 'rules', {}))
            },
            'scorer': {
                'initialized': self.scorer is not None,
                'trained': getattr(self.scorer, 'is_trained', False),
                'model_type': getattr(self.scorer, 'model_type', None)
            },
            'ranker': {
                'initialized': self.ranker is not None,
                'top_k': getattr(self.config, 'top_k_predictions', 15)
            }
        }
    
    def validate_input(self, 
                      input_icds: List[str], 
                      age: Optional[int] = None, 
                      gender: Optional[str] = None) -> Dict[str, Any]:
        """Validate input parameters."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate ICDs
        if not input_icds:
            validation_result['valid'] = False
            validation_result['errors'].append("At least one ICD code is required")
        
        valid_icds = []
        invalid_icds = []
        
        for icd in input_icds:
            if icd in self.candidate_generator.all_icd_codes:
                valid_icds.append(icd)
            else:
                invalid_icds.append(icd)
        
        if not valid_icds:
            validation_result['valid'] = False
            validation_result['errors'].append("No valid ICD codes found")
        
        if invalid_icds:
            validation_result['warnings'].append(f"Invalid ICD codes: {invalid_icds}")
        
        # Validate age
        if age is not None:
            if not isinstance(age, int) or age < 0 or age > 120:
                validation_result['valid'] = False
                validation_result['errors'].append("Age must be between 0 and 120")
        
        # Validate gender
        if gender is not None:
            valid_genders = ['M', 'F', 'Male', 'Female', 'male', 'female']
            if gender not in valid_genders:
                validation_result['valid'] = False
                validation_result['errors'].append("Gender must be M, F, Male, or Female")
        
        validation_result['valid_icds'] = valid_icds
        validation_result['invalid_icds'] = invalid_icds
        
        return validation_result