"""
FeatureBuilder: Assembles comprehensive feature vectors for ML models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional, Tuple, Union, Any
import logging
from collections import defaultdict, Counter
import math

from config.model_config import get_config

logger = logging.getLogger(__name__)

class FeatureBuilder:
    """
    Assembles comprehensive feature vectors including semantic, statistical,
    demographic, and rule-based features for ML models.
    """
    
    def __init__(self, 
                 patient_vector_builder,
                 candidate_generator,
                 rule_engine=None,
                 icd_embeddings: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize the FeatureBuilder.
        
        Args:
            patient_vector_builder: PatientVectorBuilder instance
            candidate_generator: CandidateGenerator instance
            rule_engine: RuleTriggerEngine instance (optional)
            icd_embeddings: Dictionary of ICD embeddings
        """
        self.config = get_config()
        self.patient_vector_builder = patient_vector_builder
        self.candidate_generator = candidate_generator
        self.rule_engine = rule_engine
        self.icd_embeddings = icd_embeddings or {}
        
        # Cache for computed features
        self.feature_cache = {}
        
        # Build feature type mappings
        self.feature_builders = {
            'cosine_similarity': self._build_cosine_similarity_features,
            'conditional_probability': self._build_conditional_probability_features,
            'cms_risk_score': self._build_cms_risk_score_features,
            'age': self._build_age_features,
            'gender': self._build_gender_features,
            'icd_count': self._build_icd_count_features,
            'entropy': self._build_entropy_features,
            'rule_triggers': self._build_rule_trigger_features,
            'cluster_overlap': self._build_cluster_overlap_features,
            'embedding_variance': self._build_embedding_variance_features
        }
    
    def build_features(self, 
                      input_icds: List[str],
                      candidate_code: str,
                      candidate_type: str = 'ICD',
                      age: Optional[int] = None,
                      gender: Optional[str] = None,
                      patient_data: Optional[Dict] = None) -> np.ndarray:
        """
        Build comprehensive feature vector for a candidate prediction.
        
        Args:
            input_icds: List of input ICD codes
            candidate_code: Candidate ICD/HCC code
            candidate_type: Type of candidate ('ICD' or 'HCC')
            age: Patient age
            gender: Patient gender
            patient_data: Additional patient data
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        feature_names = []
        
        # Build patient vector
        patient_vector = self.patient_vector_builder.build_patient_vector(input_icds, age, gender)
        
        # Build features for each configured type
        for feature_type in self.config.feature_types:
            if feature_type in self.feature_builders:
                try:
                    feature_values, feature_labels = self.feature_builders[feature_type](
                        input_icds, candidate_code, candidate_type, age, gender, patient_data, patient_vector
                    )
                    features.extend(feature_values)
                    feature_names.extend([f"{feature_type}_{label}" for label in feature_labels])
                except Exception as e:
                    logger.error(f"Error building {feature_type} features: {str(e)}")
                    # Add zero features as fallback
                    features.extend([0.0])
                    feature_names.extend([f"{feature_type}_error"])
        
        return np.array(features, dtype=np.float32)
    
    def _build_cosine_similarity_features(self, input_icds, candidate_code, candidate_type, 
                                        age, gender, patient_data, patient_vector) -> Tuple[List[float], List[str]]:
        """Build cosine similarity features."""
        features = []
        labels = []
        
        if candidate_code in self.icd_embeddings:
            candidate_embedding = self.icd_embeddings[candidate_code]
            
            # Patient vector to candidate similarity
            patient_medical_vector = patient_vector[:self.patient_vector_builder.embedding_dim]
            cosine_sim = self._cosine_similarity(patient_medical_vector, candidate_embedding)
            features.append(cosine_sim)
            labels.append("patient_candidate")
            
            # Individual ICD similarities
            similarities = []
            for icd in input_icds:
                if icd in self.icd_embeddings:
                    icd_sim = self._cosine_similarity(self.icd_embeddings[icd], candidate_embedding)
                    similarities.append(icd_sim)
            
            if similarities:
                features.extend([
                    np.mean(similarities),  # Mean similarity
                    np.max(similarities),   # Max similarity
                    np.min(similarities),   # Min similarity
                    np.std(similarities)    # Std similarity
                ])
                labels.extend(["mean_icd_sim", "max_icd_sim", "min_icd_sim", "std_icd_sim"])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
                labels.extend(["mean_icd_sim", "max_icd_sim", "min_icd_sim", "std_icd_sim"])
        else:
            # No embedding available
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            labels.extend(["patient_candidate", "mean_icd_sim", "max_icd_sim", "min_icd_sim", "std_icd_sim"])
        
        return features, labels
    
    def _build_conditional_probability_features(self, input_icds, candidate_code, candidate_type,
                                              age, gender, patient_data, patient_vector) -> Tuple[List[float], List[str]]:
        """Build conditional probability features."""
        features = []
        labels = []
        
        # P(candidate | input_icds) using co-occurrence data
        if hasattr(self.candidate_generator, 'icd_cooccurrence'):
            cooccurrence_probs = []
            for input_icd in input_icds:
                if input_icd in self.candidate_generator.icd_cooccurrence:
                    cooccur_count = self.candidate_generator.icd_cooccurrence[input_icd].get(candidate_code, 0)
                    input_freq = self.candidate_generator.icd_frequencies.get(input_icd, 0)
                    
                    if input_freq > 0:
                        prob = cooccur_count / input_freq
                        cooccurrence_probs.append(prob)
            
            if cooccurrence_probs:
                features.extend([
                    np.mean(cooccurrence_probs),
                    np.max(cooccurrence_probs),
                    np.sum(cooccurrence_probs)
                ])
                labels.extend(["mean_cond_prob", "max_cond_prob", "sum_cond_prob"])
            else:
                features.extend([0.0, 0.0, 0.0])
                labels.extend(["mean_cond_prob", "max_cond_prob", "sum_cond_prob"])
        else:
            features.extend([0.0, 0.0, 0.0])
            labels.extend(["mean_cond_prob", "max_cond_prob", "sum_cond_prob"])
        
        # Frequency-based probability
        candidate_freq = self.candidate_generator.icd_frequencies.get(candidate_code, 0)
        total_freq = sum(self.candidate_generator.icd_frequencies.values()) or 1
        global_prob = candidate_freq / total_freq
        features.append(global_prob)
        labels.append("global_prob")
        
        return features, labels
    
    def _build_cms_risk_score_features(self, input_icds, candidate_code, candidate_type,
                                     age, gender, patient_data, patient_vector) -> Tuple[List[float], List[str]]:
        """Build CMS risk score features."""
        features = []
        labels = []
        
        if candidate_type == 'HCC':
            # Get CMS risk score for HCC
            cms_score = self.candidate_generator._get_cms_risk_score(candidate_code)
            features.append(cms_score)
            labels.append("hcc_risk_score")
        else:
            # For ICD, get mapped HCC scores
            hcc_mappings = self.candidate_generator.icd_to_hcc.get(candidate_code, [])
            if hcc_mappings:
                hcc_scores = [self.candidate_generator._get_cms_risk_score(hcc) for hcc in hcc_mappings]
                features.extend([
                    np.mean(hcc_scores),
                    np.max(hcc_scores)
                ])
                labels.extend(["mean_hcc_score", "max_hcc_score"])
            else:
                features.extend([0.0, 0.0])
                labels.extend(["mean_hcc_score", "max_hcc_score"])
        
        # Input ICDs risk scores
        input_risk_scores = []
        for icd in input_icds:
            hcc_mappings = self.candidate_generator.icd_to_hcc.get(icd, [])
            for hcc in hcc_mappings:
                score = self.candidate_generator._get_cms_risk_score(hcc)
                input_risk_scores.append(score)
        
        if input_risk_scores:
            features.append(np.mean(input_risk_scores))
            labels.append("input_risk_score")
        else:
            features.append(0.0)
            labels.append("input_risk_score")
        
        return features, labels
    
    def _build_age_features(self, input_icds, candidate_code, candidate_type,
                           age, gender, patient_data, patient_vector) -> Tuple[List[float], List[str]]:
        """Build age-related features."""
        features = []
        labels = []
        
        if age is not None:
            # Normalized age
            features.append(min(age / 100.0, 1.0))
            labels.append("normalized")
            
            # Age categories
            age_categories = {
                'pediatric': 1.0 if age < 18 else 0.0,
                'young_adult': 1.0 if 18 <= age < 35 else 0.0,
                'middle_age': 1.0 if 35 <= age < 50 else 0.0,
                'pre_senior': 1.0 if 50 <= age < 65 else 0.0,
                'senior': 1.0 if 65 <= age < 80 else 0.0,
                'elderly': 1.0 if age >= 80 else 0.0
            }
            
            for category, value in age_categories.items():
                features.append(value)
                labels.append(category)
        else:
            # Missing age
            features.extend([0.5] + [0.0] * 6)  # Unknown age
            labels.extend(['normalized', 'pediatric', 'young_adult', 'middle_age', 
                          'pre_senior', 'senior', 'elderly'])
        
        return features, labels
    
    def _build_gender_features(self, input_icds, candidate_code, candidate_type,
                              age, gender, patient_data, patient_vector) -> Tuple[List[float], List[str]]:
        """Build gender-related features."""
        features = []
        labels = []
        
        if gender is not None:
            gender_encoding = {'M': [1.0, 0.0], 'F': [0.0, 1.0], 
                             'Male': [1.0, 0.0], 'Female': [0.0, 1.0],
                             'male': [1.0, 0.0], 'female': [0.0, 1.0]}
            
            encoding = gender_encoding.get(gender, [0.5, 0.5])
            features.extend(encoding)
            labels.extend(['is_male', 'is_female'])
        else:
            features.extend([0.5, 0.5])  # Unknown gender
            labels.extend(['is_male', 'is_female'])
        
        return features, labels
    
    def _build_icd_count_features(self, input_icds, candidate_code, candidate_type,
                                 age, gender, patient_data, patient_vector) -> Tuple[List[float], List[str]]:
        """Build ICD count features."""
        features = []
        labels = []
        
        # Basic counts
        features.extend([
            len(input_icds),                    # Total ICDs
            len(set(input_icds)),               # Unique ICDs
            len(input_icds) / 20.0              # Normalized count (assuming max 20)
        ])
        labels.extend(['total_count', 'unique_count', 'normalized_count'])
        
        # Category distribution
        category_counts = defaultdict(int)
        for icd in input_icds:
            if icd and len(icd) > 0:
                category_counts[icd[0].upper()] += 1
        
        # Top categories representation
        major_categories = ['I', 'E', 'J', 'K', 'M', 'N', 'Z']  # Common categories
        for category in major_categories:
            features.append(category_counts.get(category, 0))
            labels.append(f'category_{category}_count')
        
        return features, labels
    
    def _build_entropy_features(self, input_icds, candidate_code, candidate_type,
                               age, gender, patient_data, patient_vector) -> Tuple[List[float], List[str]]:
        """Build entropy-based diversity features."""
        features = []
        labels = []
        
        if input_icds:
            # ICD code entropy
            code_counts = Counter(input_icds)
            total_codes = len(input_icds)
            entropy = 0
            for count in code_counts.values():
                prob = count / total_codes
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            
            features.append(entropy)
            labels.append('icd_entropy')
            
            # Category entropy
            categories = [icd[0] if icd else 'X' for icd in input_icds]
            category_counts = Counter(categories)
            category_entropy = 0
            for count in category_counts.values():
                prob = count / len(categories)
                if prob > 0:
                    category_entropy -= prob * math.log2(prob)
            
            features.append(category_entropy)
            labels.append('category_entropy')
        else:
            features.extend([0.0, 0.0])
            labels.extend(['icd_entropy', 'category_entropy'])
        
        return features, labels
    
    def _build_rule_trigger_features(self, input_icds, candidate_code, candidate_type,
                                   age, gender, patient_data, patient_vector) -> Tuple[List[float], List[str]]:
        """Build rule trigger features."""
        features = []
        labels = []
        
        if self.rule_engine is not None:
            try:
                rule_results = self.rule_engine.evaluate_rules(input_icds, candidate_code, age, gender)
                
                for rule_name, triggered in rule_results.items():
                    features.append(1.0 if triggered else 0.0)
                    labels.append(rule_name)
            except Exception as e:
                logger.error(f"Error evaluating rules: {str(e)}")
                # Add default rule features
                default_rules = ['diabetes_complications', 'cardiac_risk', 'respiratory_comorbidity', 
                               'renal_complications', 'oncology_progression']
                features.extend([0.0] * len(default_rules))
                labels.extend(default_rules)
        else:
            # No rule engine available
            default_rules = ['diabetes_complications', 'cardiac_risk', 'respiratory_comorbidity',
                           'renal_complications', 'oncology_progression']
            features.extend([0.0] * len(default_rules))
            labels.extend(default_rules)
        
        return features, labels
    
    def _build_cluster_overlap_features(self, input_icds, candidate_code, candidate_type,
                                      age, gender, patient_data, patient_vector) -> Tuple[List[float], List[str]]:
        """Build cluster overlap features."""
        features = []
        labels = []
        
        # Disease system clustering (simplified)
        disease_systems = {
            'cardiovascular': ['I'],
            'endocrine': ['E'],
            'respiratory': ['J'],
            'digestive': ['K'],
            'musculoskeletal': ['M'],
            'genitourinary': ['N'],
            'neoplasms': ['C', 'D'],
            'infectious': ['A', 'B'],
            'mental': ['F'],
            'nervous': ['G']
        }
        
        # Get input systems
        input_systems = set()
        for icd in input_icds:
            if icd and len(icd) > 0:
                first_char = icd[0].upper()
                for system, chars in disease_systems.items():
                    if first_char in chars:
                        input_systems.add(system)
        
        # Check candidate system
        candidate_systems = set()
        if candidate_code and len(candidate_code) > 0:
            first_char = candidate_code[0].upper()
            for system, chars in disease_systems.items():
                if first_char in chars:
                    candidate_systems.add(system)
        
        # Calculate overlap
        if input_systems and candidate_systems:
            overlap = len(input_systems & candidate_systems) / len(input_systems | candidate_systems)
        else:
            overlap = 0.0
        
        features.append(overlap)
        labels.append('system_overlap')
        
        # Individual system features
        for system in disease_systems.keys():
            input_has_system = 1.0 if system in input_systems else 0.0
            candidate_has_system = 1.0 if system in candidate_systems else 0.0
            
            features.extend([input_has_system, candidate_has_system])
            labels.extend([f'input_has_{system}', f'candidate_has_{system}'])
        
        return features, labels
    
    def _build_embedding_variance_features(self, input_icds, candidate_code, candidate_type,
                                         age, gender, patient_data, patient_vector) -> Tuple[List[float], List[str]]:
        """Build embedding variance features."""
        features = []
        labels = []
        
        if len(input_icds) > 1:
            # Get embeddings for input ICDs
            input_embeddings = []
            for icd in input_icds:
                if icd in self.icd_embeddings:
                    input_embeddings.append(self.icd_embeddings[icd])
            
            if len(input_embeddings) > 1:
                embeddings_matrix = np.array(input_embeddings)
                
                # Calculate variance metrics
                feature_variances = np.var(embeddings_matrix, axis=0)
                features.extend([
                    np.mean(feature_variances),    # Mean variance
                    np.max(feature_variances),     # Max variance
                    np.std(feature_variances)      # Std of variances
                ])
                labels.extend(['mean_embedding_var', 'max_embedding_var', 'std_embedding_var'])
                
                # Pairwise distances
                distances = []
                for i in range(len(input_embeddings)):
                    for j in range(i + 1, len(input_embeddings)):
                        dist = np.linalg.norm(input_embeddings[i] - input_embeddings[j])
                        distances.append(dist)
                
                if distances:
                    features.extend([
                        np.mean(distances),
                        np.std(distances)
                    ])
                    labels.extend(['mean_pairwise_dist', 'std_pairwise_dist'])
                else:
                    features.extend([0.0, 0.0])
                    labels.extend(['mean_pairwise_dist', 'std_pairwise_dist'])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                labels.extend(['mean_embedding_var', 'max_embedding_var', 'std_embedding_var',
                             'mean_pairwise_dist', 'std_pairwise_dist'])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            labels.extend(['mean_embedding_var', 'max_embedding_var', 'std_embedding_var',
                         'mean_pairwise_dist', 'std_pairwise_dist'])
        
        return features, labels
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return np.dot(vec1, vec2) / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {str(e)}")
            return 0.0
    
    def build_batch_features(self, 
                           batch_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """
        Build features for a batch of samples.
        
        Args:
            batch_data: List of dictionaries containing sample data
            
        Returns:
            Feature matrix and feature names
        """
        feature_matrix = []
        feature_names = None
        
        for sample in batch_data:
            input_icds = sample.get('input_icds', [])
            candidate_code = sample.get('candidate_code')
            candidate_type = sample.get('candidate_type', 'ICD')
            age = sample.get('age')
            gender = sample.get('gender')
            patient_data = sample.get('patient_data')
            
            features = self.build_features(
                input_icds, candidate_code, candidate_type, age, gender, patient_data
            )
            feature_matrix.append(features)
            
            # Get feature names from first sample
            if feature_names is None:
                feature_names = self.get_feature_names()
        
        return np.array(feature_matrix), feature_names or []
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        # This would require building features once to get the names
        # For now, return a placeholder
        feature_names = []
        
        for feature_type in self.config.feature_types:
            if feature_type == 'cosine_similarity':
                feature_names.extend([
                    'cosine_similarity_patient_candidate',
                    'cosine_similarity_mean_icd_sim',
                    'cosine_similarity_max_icd_sim',
                    'cosine_similarity_min_icd_sim',
                    'cosine_similarity_std_icd_sim'
                ])
            elif feature_type == 'conditional_probability':
                feature_names.extend([
                    'conditional_probability_mean_cond_prob',
                    'conditional_probability_max_cond_prob',
                    'conditional_probability_sum_cond_prob',
                    'conditional_probability_global_prob'
                ])
            # Add other feature types as needed...
        
        return feature_names
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
                return dict(zip(feature_names, importances))
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}