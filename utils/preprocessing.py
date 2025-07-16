"""
Data preprocessing utilities for the disease prediction system.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from collections import defaultdict, Counter
import random

from config.model_config import get_config

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Utility class for preprocessing data for the disease prediction system.
    Handles negative sampling, feature scaling, and data augmentation.
    """
    
    def __init__(self):
        """Initialize DataPreprocessor."""
        self.config = get_config()
        self.random_state = 42
        random.seed(self.random_state)
        np.random.seed(self.random_state)
    
    def create_negative_samples(self, 
                              positive_samples: List[Dict[str, Any]],
                              all_icd_codes: List[str],
                              negative_ratio: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Create negative samples for training.
        
        Args:
            positive_samples: List of positive training samples
            all_icd_codes: All available ICD codes for negative sampling
            negative_ratio: Number of negative samples per positive sample
            
        Returns:
            List of negative samples
        """
        negative_ratio = negative_ratio or self.config.negative_sampling_ratio
        negative_samples = []
        
        # Build patient code sets for faster lookup
        patient_codes = {}
        for sample in positive_samples:
            patient_id = sample['patient_id']
            if patient_id not in patient_codes:
                patient_codes[patient_id] = set()
            patient_codes[patient_id].update(sample['input_codes'])
            patient_codes[patient_id].add(sample['target_code'])
        
        logger.info(f"Creating negative samples with ratio {negative_ratio}:1")
        
        for sample in positive_samples:
            patient_id = sample['patient_id']
            input_codes = sample['input_codes']
            patient_all_codes = patient_codes[patient_id]
            
            # Candidate negative codes (not in patient's history)
            negative_candidates = [code for code in all_icd_codes 
                                 if code not in patient_all_codes]
            
            if not negative_candidates:
                continue
            
            # Sample negative codes
            num_negatives = min(negative_ratio, len(negative_candidates))
            sampled_negatives = random.sample(negative_candidates, num_negatives)
            
            for neg_code in sampled_negatives:
                neg_sample = {
                    'patient_id': patient_id,
                    'input_codes': input_codes,
                    'target_code': neg_code,
                    'age': sample.get('age'),
                    'gender': sample.get('gender'),
                    'is_positive': False  # This is a negative sample
                }
                negative_samples.append(neg_sample)
        
        logger.info(f"Created {len(negative_samples)} negative samples")
        return negative_samples
    
    def balance_dataset(self, 
                       samples: List[Dict[str, Any]], 
                       target_ratio: float = 0.5) -> List[Dict[str, Any]]:
        """
        Balance the dataset to achieve target positive/negative ratio.
        
        Args:
            samples: List of training samples
            target_ratio: Target ratio of positive samples
            
        Returns:
            Balanced list of samples
        """
        positive_samples = [s for s in samples if s['is_positive']]
        negative_samples = [s for s in samples if not s['is_positive']]
        
        current_pos_ratio = len(positive_samples) / len(samples) if samples else 0
        
        logger.info(f"Current positive ratio: {current_pos_ratio:.3f}, target: {target_ratio:.3f}")
        
        if current_pos_ratio < target_ratio:
            # Need more positive samples - oversample positives
            target_pos_count = int(len(negative_samples) * target_ratio / (1 - target_ratio))
            if target_pos_count > len(positive_samples):
                # Oversample positive samples
                additional_pos = target_pos_count - len(positive_samples)
                oversampled_pos = random.choices(positive_samples, k=additional_pos)
                balanced_samples = positive_samples + oversampled_pos + negative_samples
            else:
                balanced_samples = positive_samples + negative_samples
        else:
            # Need fewer positive samples or more negative samples
            target_neg_count = int(len(positive_samples) * (1 - target_ratio) / target_ratio)
            if target_neg_count > len(negative_samples):
                # Oversample negative samples
                additional_neg = target_neg_count - len(negative_samples)
                oversampled_neg = random.choices(negative_samples, k=additional_neg)
                balanced_samples = positive_samples + negative_samples + oversampled_neg
            else:
                # Downsample negative samples
                sampled_neg = random.sample(negative_samples, target_neg_count)
                balanced_samples = positive_samples + sampled_neg
        
        # Shuffle the balanced dataset
        random.shuffle(balanced_samples)
        
        final_pos_ratio = len([s for s in balanced_samples if s['is_positive']]) / len(balanced_samples)
        logger.info(f"Balanced dataset: {len(balanced_samples)} samples, positive ratio: {final_pos_ratio:.3f}")
        
        return balanced_samples
    
    def augment_patient_data(self, 
                           samples: List[Dict[str, Any]], 
                           augmentation_factor: float = 0.1) -> List[Dict[str, Any]]:
        """
        Augment training data by creating variations of existing samples.
        
        Args:
            samples: Original training samples
            augmentation_factor: Fraction of samples to augment
            
        Returns:
            Augmented list of samples
        """
        augmented_samples = samples.copy()
        num_to_augment = int(len(samples) * augmentation_factor)
        
        samples_to_augment = random.sample(samples, min(num_to_augment, len(samples)))
        
        logger.info(f"Augmenting {len(samples_to_augment)} samples")
        
        for sample in samples_to_augment:
            input_codes = sample['input_codes']
            
            if len(input_codes) <= 1:
                continue
            
            # Create variations by removing some input codes (dropout)
            for dropout_prob in [0.1, 0.2]:
                if random.random() < 0.5:  # 50% chance to create this variation
                    augmented_codes = [code for code in input_codes 
                                     if random.random() > dropout_prob]
                    
                    if augmented_codes:  # Ensure at least one code remains
                        aug_sample = sample.copy()
                        aug_sample['input_codes'] = augmented_codes
                        aug_sample['augmented'] = True
                        augmented_samples.append(aug_sample)
        
        logger.info(f"Created {len(augmented_samples) - len(samples)} augmented samples")
        return augmented_samples
    
    def filter_samples_by_criteria(self, 
                                 samples: List[Dict[str, Any]],
                                 min_input_codes: Optional[int] = None,
                                 max_input_codes: Optional[int] = None,
                                 required_demographics: bool = False) -> List[Dict[str, Any]]:
        """
        Filter samples based on various criteria.
        
        Args:
            samples: List of samples to filter
            min_input_codes: Minimum number of input codes required
            max_input_codes: Maximum number of input codes allowed
            required_demographics: Whether age and gender are required
            
        Returns:
            Filtered list of samples
        """
        filtered_samples = []
        
        min_codes = min_input_codes or self.config.min_patient_codes
        max_codes = max_input_codes or self.config.max_patient_codes
        
        for sample in samples:
            input_codes = sample.get('input_codes', [])
            
            # Check input code count
            if len(input_codes) < min_codes or len(input_codes) > max_codes:
                continue
            
            # Check demographics if required
            if required_demographics:
                if sample.get('age') is None or sample.get('gender') is None:
                    continue
            
            filtered_samples.append(sample)
        
        logger.info(f"Filtered samples: {len(samples)} -> {len(filtered_samples)}")
        return filtered_samples
    
    def create_stratified_splits(self, 
                               samples: List[Dict[str, Any]], 
                               test_size: float = 0.2,
                               val_size: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create stratified train/validation/test splits.
        
        Args:
            samples: List of samples to split
            test_size: Fraction for test set
            val_size: Fraction for validation set
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        # Separate positive and negative samples
        positive_samples = [s for s in samples if s['is_positive']]
        negative_samples = [s for s in samples if not s['is_positive']]
        
        # Shuffle each group
        random.shuffle(positive_samples)
        random.shuffle(negative_samples)
        
        # Split positive samples
        pos_test_count = int(len(positive_samples) * test_size)
        pos_val_count = int(len(positive_samples) * val_size)
        
        pos_test = positive_samples[:pos_test_count]
        pos_val = positive_samples[pos_test_count:pos_test_count + pos_val_count]
        pos_train = positive_samples[pos_test_count + pos_val_count:]
        
        # Split negative samples
        neg_test_count = int(len(negative_samples) * test_size)
        neg_val_count = int(len(negative_samples) * val_size)
        
        neg_test = negative_samples[:neg_test_count]
        neg_val = negative_samples[neg_test_count:neg_test_count + neg_val_count]
        neg_train = negative_samples[neg_test_count + neg_val_count:]
        
        # Combine and shuffle
        train_samples = pos_train + neg_train
        val_samples = pos_val + neg_val
        test_samples = pos_test + neg_test
        
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        logger.info(f"Split samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
        
        return train_samples, val_samples, test_samples
    
    def encode_categorical_features(self, 
                                  samples: List[Dict[str, Any]]) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Encode categorical features and return encoding mappings.
        
        Args:
            samples: List of samples with categorical features
            
        Returns:
            Tuple of (encoded_samples, encoding_mappings)
        """
        encoding_mappings = {
            'gender': {'M': 0, 'F': 1, 'Male': 0, 'Female': 1, 'male': 0, 'female': 1},
            'age_bins': {}
        }
        
        # Create age bins
        ages = [s.get('age') for s in samples if s.get('age') is not None]
        if ages:
            age_percentiles = np.percentile(ages, [20, 40, 60, 80])
            encoding_mappings['age_bins'] = {
                'very_young': (0, age_percentiles[0]),
                'young': (age_percentiles[0], age_percentiles[1]),
                'middle': (age_percentiles[1], age_percentiles[2]),
                'older': (age_percentiles[2], age_percentiles[3]),
                'elderly': (age_percentiles[3], float('inf'))
            }
        
        # Apply encodings
        encoded_samples = []
        for sample in samples:
            encoded_sample = sample.copy()
            
            # Encode gender
            gender = sample.get('gender')
            if gender:
                encoded_sample['gender_encoded'] = encoding_mappings['gender'].get(gender, 0.5)
            
            # Encode age bins
            age = sample.get('age')
            if age is not None and encoding_mappings['age_bins']:
                for bin_name, (min_age, max_age) in encoding_mappings['age_bins'].items():
                    encoded_sample[f'age_bin_{bin_name}'] = 1 if min_age <= age < max_age else 0
            
            encoded_samples.append(encoded_sample)
        
        return encoded_samples, encoding_mappings
    
    def normalize_features(self, 
                         feature_matrix: np.ndarray, 
                         fit_on_train: bool = True,
                         existing_stats: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Normalize feature matrix using z-score normalization.
        
        Args:
            feature_matrix: Feature matrix to normalize
            fit_on_train: Whether to fit normalization on this data
            existing_stats: Pre-computed normalization statistics
            
        Returns:
            Tuple of (normalized_matrix, normalization_stats)
        """
        if feature_matrix.shape[0] == 0:
            return feature_matrix, {}
        
        if fit_on_train or existing_stats is None:
            # Compute normalization statistics
            mean = np.mean(feature_matrix, axis=0)
            std = np.std(feature_matrix, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            
            normalization_stats = {
                'mean': mean,
                'std': std,
                'feature_count': feature_matrix.shape[1]
            }
        else:
            # Use existing statistics
            normalization_stats = existing_stats
            mean = normalization_stats['mean']
            std = normalization_stats['std']
        
        # Apply normalization
        normalized_matrix = (feature_matrix - mean) / std
        
        return normalized_matrix, normalization_stats
    
    def handle_missing_values(self, 
                            feature_matrix: np.ndarray, 
                            strategy: str = 'mean') -> np.ndarray:
        """
        Handle missing values in feature matrix.
        
        Args:
            feature_matrix: Feature matrix with potential missing values
            strategy: Strategy for handling missing values ('mean', 'median', 'zero')
            
        Returns:
            Feature matrix with missing values handled
        """
        if not np.isnan(feature_matrix).any():
            return feature_matrix
        
        logger.info(f"Handling missing values using strategy: {strategy}")
        
        if strategy == 'mean':
            col_means = np.nanmean(feature_matrix, axis=0)
            inds = np.where(np.isnan(feature_matrix))
            feature_matrix[inds] = np.take(col_means, inds[1])
        elif strategy == 'median':
            col_medians = np.nanmedian(feature_matrix, axis=0)
            inds = np.where(np.isnan(feature_matrix))
            feature_matrix[inds] = np.take(col_medians, inds[1])
        elif strategy == 'zero':
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")
        
        return feature_matrix
    
    def create_feature_groups(self, 
                            feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Group features by type for analysis.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping group names to feature lists
        """
        feature_groups = {
            'embedding': [],
            'demographic': [],
            'statistical': [],
            'rule_based': [],
            'similarity': [],
            'other': []
        }
        
        for feature_name in feature_names:
            name_lower = feature_name.lower()
            
            if 'embedding' in name_lower or 'cosine' in name_lower:
                feature_groups['similarity'].append(feature_name)
            elif any(demo in name_lower for demo in ['age', 'gender', 'demographic']):
                feature_groups['demographic'].append(feature_name)
            elif any(stat in name_lower for stat in ['count', 'entropy', 'variance', 'frequency']):
                feature_groups['statistical'].append(feature_name)
            elif 'rule' in name_lower or 'trigger' in name_lower:
                feature_groups['rule_based'].append(feature_name)
            elif any(emb in name_lower for emb in ['embed', 'vector', 'similarity']):
                feature_groups['embedding'].append(feature_name)
            else:
                feature_groups['other'].append(feature_name)
        
        return feature_groups
    
    def validate_preprocessed_data(self, 
                                 samples: List[Dict[str, Any]], 
                                 feature_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate preprocessed data for consistency and quality.
        
        Args:
            samples: List of preprocessed samples
            feature_matrix: Optional feature matrix to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Basic sample validation
        positive_count = len([s for s in samples if s.get('is_positive', False)])
        negative_count = len(samples) - positive_count
        
        validation_results['statistics'] = {
            'total_samples': len(samples),
            'positive_samples': positive_count,
            'negative_samples': negative_count,
            'positive_ratio': positive_count / len(samples) if samples else 0
        }
        
        # Check for data leakage
        patient_targets = defaultdict(set)
        for sample in samples:
            patient_id = sample.get('patient_id')
            target = sample.get('target_code')
            if patient_id and target:
                patient_targets[patient_id].add(target)
        
        # Check if any patient has same target in multiple samples
        patients_with_duplicates = [pid for pid, targets in patient_targets.items() 
                                  if len(targets) < len([s for s in samples 
                                                       if s.get('patient_id') == pid])]
        
        if patients_with_duplicates:
            validation_results['warnings'].append(
                f"Found {len(patients_with_duplicates)} patients with duplicate target codes"
            )
        
        # Feature matrix validation
        if feature_matrix is not None:
            validation_results['statistics'].update({
                'feature_matrix_shape': feature_matrix.shape,
                'features_with_nan': np.isnan(feature_matrix).any(axis=0).sum(),
                'features_with_inf': np.isinf(feature_matrix).any(axis=0).sum(),
                'features_constant': (feature_matrix.std(axis=0) == 0).sum()
            })
            
            # Check for problematic features
            if validation_results['statistics']['features_with_nan'] > 0:
                validation_results['warnings'].append("Features contain NaN values")
            
            if validation_results['statistics']['features_with_inf'] > 0:
                validation_results['warnings'].append("Features contain infinite values")
            
            if validation_results['statistics']['features_constant'] > 0:
                validation_results['warnings'].append("Some features are constant")
        
        logger.info(f"Data validation completed: {validation_results['statistics']}")
        return validation_results
    
    def get_preprocessing_summary(self, 
                                original_samples: List[Dict[str, Any]], 
                                processed_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of preprocessing operations.
        
        Args:
            original_samples: Original samples before preprocessing
            processed_samples: Samples after preprocessing
            
        Returns:
            Preprocessing summary
        """
        return {
            'original_count': len(original_samples),
            'processed_count': len(processed_samples),
            'samples_removed': len(original_samples) - len(processed_samples),
            'removal_rate': (len(original_samples) - len(processed_samples)) / len(original_samples) 
                           if original_samples else 0,
            'augmentation_applied': len([s for s in processed_samples if s.get('augmented', False)]),
            'positive_ratio_original': len([s for s in original_samples if s.get('is_positive', False)]) / len(original_samples) 
                                     if original_samples else 0,
            'positive_ratio_processed': len([s for s in processed_samples if s.get('is_positive', False)]) / len(processed_samples) 
                                      if processed_samples else 0
        }