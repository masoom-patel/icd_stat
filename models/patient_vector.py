"""
PatientVectorBuilder: Builds patient vectors from ICD patterns with demographic integration.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional, Tuple, Union
import logging
from collections import defaultdict

from config.model_config import get_config

logger = logging.getLogger(__name__)

class PatientVectorBuilder:
    """
    Builds comprehensive patient vectors from ICD patterns using embedding aggregation
    and demographic feature integration.
    """
    
    def __init__(self, icd_embeddings: Dict[str, np.ndarray]):
        """
        Initialize the PatientVectorBuilder.
        
        Args:
            icd_embeddings: Dictionary mapping ICD codes to their embeddings
        """
        self.config = get_config()
        self.icd_embeddings = icd_embeddings
        self.embedding_dim = len(next(iter(icd_embeddings.values()))) if icd_embeddings else 384
        
        # Demographic encoding
        self.gender_encoding = {'M': 0, 'F': 1, 'Male': 0, 'Female': 1, 'male': 0, 'female': 1}
        self.age_bins = self._create_age_bins()
        
    def _create_age_bins(self) -> List[Tuple[int, int]]:
        """Create age bins for categorical encoding."""
        return [
            (0, 17),    # Pediatric
            (18, 34),   # Young Adult
            (35, 49),   # Middle Age
            (50, 64),   # Pre-senior
            (65, 79),   # Senior
            (80, 120)   # Elderly
        ]
    
    def build_patient_vector(self, 
                           icd_codes: List[str], 
                           age: Optional[int] = None, 
                           gender: Optional[str] = None) -> np.ndarray:
        """
        Build a comprehensive patient vector from ICD codes and demographics.
        
        Args:
            icd_codes: List of ICD codes for the patient
            age: Patient age (optional)
            gender: Patient gender (optional)
            
        Returns:
            Combined patient vector including medical and demographic features
        """
        # Get medical vector from ICD embeddings
        medical_vector = self._aggregate_icd_embeddings(icd_codes)
        
        # Get demographic features
        demographic_vector = self._build_demographic_vector(age, gender)
        
        # Get pattern-based features
        pattern_vector = self._build_pattern_vector(icd_codes)
        
        # Combine all features
        if self.config.demographic_features:
            patient_vector = np.concatenate([medical_vector, demographic_vector, pattern_vector])
        else:
            patient_vector = np.concatenate([medical_vector, pattern_vector])
        
        return patient_vector
    
    def _aggregate_icd_embeddings(self, icd_codes: List[str]) -> np.ndarray:
        """
        Aggregate ICD embeddings using the configured aggregation method.
        
        Args:
            icd_codes: List of ICD codes
            
        Returns:
            Aggregated embedding vector
        """
        if not icd_codes:
            return np.zeros(self.embedding_dim)
        
        # Get embeddings for valid codes
        valid_embeddings = []
        for code in icd_codes:
            if code in self.icd_embeddings:
                valid_embeddings.append(self.icd_embeddings[code])
            else:
                logger.warning(f"ICD code {code} not found in embeddings")
        
        if not valid_embeddings:
            return np.zeros(self.embedding_dim)
        
        embeddings_matrix = np.array(valid_embeddings)
        
        # Apply aggregation method
        if self.config.patient_vector_aggregation == "mean":
            return np.mean(embeddings_matrix, axis=0)
        elif self.config.patient_vector_aggregation == "max":
            return np.max(embeddings_matrix, axis=0)
        elif self.config.patient_vector_aggregation == "attention":
            return self._attention_aggregation(embeddings_matrix)
        else:
            logger.warning(f"Unknown aggregation method: {self.config.patient_vector_aggregation}")
            return np.mean(embeddings_matrix, axis=0)
    
    def _attention_aggregation(self, embeddings_matrix: np.ndarray) -> np.ndarray:
        """
        Apply simple attention-based aggregation.
        
        Args:
            embeddings_matrix: Matrix of embeddings (n_codes x embedding_dim)
            
        Returns:
            Attention-weighted aggregated embedding
        """
        try:
            # Compute attention weights (simple dot-product attention)
            mean_embedding = np.mean(embeddings_matrix, axis=0)
            attention_scores = np.dot(embeddings_matrix, mean_embedding)
            attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores))
            
            # Apply weights
            weighted_embedding = np.sum(embeddings_matrix * attention_weights.reshape(-1, 1), axis=0)
            return weighted_embedding
        except Exception as e:
            logger.error(f"Error in attention aggregation: {str(e)}")
            return np.mean(embeddings_matrix, axis=0)
    
    def _build_demographic_vector(self, age: Optional[int], gender: Optional[str]) -> np.ndarray:
        """
        Build demographic feature vector.
        
        Args:
            age: Patient age
            gender: Patient gender
            
        Returns:
            Demographic feature vector
        """
        features = []
        
        # Age features
        if age is not None:
            # Normalized age
            features.append(min(age / 100.0, 1.0))  # Normalize to [0, 1]
            
            # Age bins (one-hot encoding)
            age_bin_features = [0] * len(self.age_bins)
            for i, (min_age, max_age) in enumerate(self.age_bins):
                if min_age <= age <= max_age:
                    age_bin_features[i] = 1
                    break
            features.extend(age_bin_features)
        else:
            # Missing age features
            features.append(0.5)  # Unknown age
            features.extend([0] * len(self.age_bins))
        
        # Gender features
        if gender is not None and gender in self.gender_encoding:
            # Binary gender encoding
            features.append(self.gender_encoding[gender])
            features.append(1 - self.gender_encoding[gender])  # Complement
        else:
            # Missing gender features
            features.extend([0.5, 0.5])  # Unknown gender
        
        return np.array(features, dtype=np.float32)
    
    def _build_pattern_vector(self, icd_codes: List[str]) -> np.ndarray:
        """
        Build pattern-based features from ICD codes.
        
        Args:
            icd_codes: List of ICD codes
            
        Returns:
            Pattern feature vector
        """
        features = []
        
        # Basic count features
        features.append(len(icd_codes))  # Total number of codes
        features.append(len(set(icd_codes)))  # Unique codes (in case of duplicates)
        
        # ICD code diversity (entropy)
        if icd_codes:
            code_counts = defaultdict(int)
            for code in icd_codes:
                code_counts[code] += 1
            
            total_codes = len(icd_codes)
            entropy = 0
            for count in code_counts.values():
                prob = count / total_codes
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            features.append(entropy)
        else:
            features.append(0.0)
        
        # Embedding variance (semantic diversity)
        if len(icd_codes) > 1:
            valid_embeddings = [self.icd_embeddings[code] for code in icd_codes 
                              if code in self.icd_embeddings]
            if len(valid_embeddings) > 1:
                embeddings_matrix = np.array(valid_embeddings)
                # Compute variance across embeddings
                variance = np.mean(np.var(embeddings_matrix, axis=0))
                features.append(variance)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # ICD code categories (based on first character)
        category_counts = defaultdict(int)
        for code in icd_codes:
            if code and len(code) > 0:
                category = code[0].upper()
                category_counts[category] += 1
        
        # One-hot encoding for major ICD categories (A-Z)
        major_categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                          'U', 'V', 'W', 'X', 'Y', 'Z']
        
        for category in major_categories:
            features.append(category_counts.get(category, 0))
        
        return np.array(features, dtype=np.float32)
    
    def build_batch_vectors(self, 
                           patients_data: List[Dict[str, Union[List[str], int, str]]]) -> np.ndarray:
        """
        Build patient vectors for a batch of patients.
        
        Args:
            patients_data: List of dictionaries with 'icd_codes', 'age', 'gender' keys
            
        Returns:
            Matrix of patient vectors (n_patients x vector_dim)
        """
        patient_vectors = []
        
        for patient_data in patients_data:
            icd_codes = patient_data.get('icd_codes', [])
            age = patient_data.get('age')
            gender = patient_data.get('gender')
            
            patient_vector = self.build_patient_vector(icd_codes, age, gender)
            patient_vectors.append(patient_vector)
        
        return np.array(patient_vectors)
    
    def get_vector_dimension(self) -> int:
        """Get the total dimension of patient vectors."""
        # Calculate dimensions
        medical_dim = self.embedding_dim
        
        if self.config.demographic_features:
            demographic_dim = 1 + len(self.age_bins) + 2  # age + age_bins + gender
        else:
            demographic_dim = 0
        
        pattern_dim = 4 + len(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                              'U', 'V', 'W', 'X', 'Y', 'Z'])  # pattern features + categories
        
        return medical_dim + demographic_dim + pattern_dim
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features in the patient vector."""
        feature_names = []
        
        # Medical embedding features
        feature_names.extend([f"embedding_{i}" for i in range(self.embedding_dim)])
        
        # Demographic features
        if self.config.demographic_features:
            feature_names.append("age_normalized")
            feature_names.extend([f"age_bin_{i}" for i in range(len(self.age_bins))])
            feature_names.extend(["gender_male", "gender_female"])
        
        # Pattern features
        feature_names.extend([
            "icd_count", "unique_icd_count", "icd_entropy", "embedding_variance"
        ])
        
        # ICD category features
        major_categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                          'U', 'V', 'W', 'X', 'Y', 'Z']
        feature_names.extend([f"category_{cat}" for cat in major_categories])
        
        return feature_names
    
    def analyze_patient_similarity(self, 
                                 patient1_codes: List[str], 
                                 patient2_codes: List[str],
                                 age1: Optional[int] = None,
                                 gender1: Optional[str] = None,
                                 age2: Optional[int] = None,
                                 gender2: Optional[str] = None) -> Dict[str, float]:
        """
        Analyze similarity between two patients.
        
        Args:
            patient1_codes: ICD codes for first patient
            patient2_codes: ICD codes for second patient
            age1, gender1: Demographics for first patient
            age2, gender2: Demographics for second patient
            
        Returns:
            Dictionary with various similarity metrics
        """
        # Build patient vectors
        vector1 = self.build_patient_vector(patient1_codes, age1, gender1)
        vector2 = self.build_patient_vector(patient2_codes, age2, gender2)
        
        # Compute similarities
        cosine_sim = self._cosine_similarity(vector1, vector2)
        
        # Code overlap similarity
        set1 = set(patient1_codes)
        set2 = set(patient2_codes)
        jaccard_sim = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0
        
        # Demographic similarity
        demo_sim = 0.0
        if age1 is not None and age2 is not None:
            age_diff = abs(age1 - age2)
            demo_sim += max(0, 1 - age_diff / 50.0) * 0.5  # Age similarity
        
        if gender1 is not None and gender2 is not None:
            demo_sim += 0.5 if gender1 == gender2 else 0.0  # Gender similarity
        
        return {
            'cosine_similarity': float(cosine_sim),
            'jaccard_similarity': float(jaccard_sim),
            'demographic_similarity': float(demo_sim),
            'combined_similarity': float(0.5 * cosine_sim + 0.3 * jaccard_sim + 0.2 * demo_sim)
        }
    
    def _cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return np.dot(vector1, vector2) / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {str(e)}")
            return 0.0