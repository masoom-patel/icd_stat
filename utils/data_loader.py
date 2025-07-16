"""
Data loading utilities for the disease prediction system.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Utility class for loading and validating data for the disease prediction system.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def load_icd_master_data(self, filepath: str) -> pd.DataFrame:
        """
        Load ICD master table with codes, descriptions, and HCC mappings.
        
        Args:
            filepath: Path to ICD master CSV file
            
        Returns:
            DataFrame with ICD master data
        """
        try:
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_columns = ['ICDCode', 'Description']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean data
            df = df.dropna(subset=['ICDCode'])
            df['ICDCode'] = df['ICDCode'].astype(str).str.strip()
            df['Description'] = df['Description'].fillna('Unknown').astype(str).str.strip()
            
            # Validate ICD codes (basic format check)
            valid_mask = df['ICDCode'].str.match(r'^[A-Z][0-9]', na=False)
            invalid_count = (~valid_mask).sum()
            
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} invalid ICD codes, filtering them out")
                df = df[valid_mask]
            
            logger.info(f"Loaded {len(df)} ICD codes from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading ICD master data from {filepath}: {str(e)}")
            raise
    
    def load_patient_data(self, filepath: str) -> pd.DataFrame:
        """
        Load patient-ICD association data.
        
        Args:
            filepath: Path to patient data CSV file
            
        Returns:
            DataFrame with patient data
        """
        try:
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_columns = ['MemberID', 'ICDCode']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean data
            df = df.dropna(subset=['MemberID', 'ICDCode'])
            df['MemberID'] = df['MemberID'].astype(str).str.strip()
            df['ICDCode'] = df['ICDCode'].astype(str).str.strip()
            
            # Optional demographic columns
            if 'Age' in df.columns:
                df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
                df = df.dropna(subset=['Age'])  # Remove rows with invalid ages
                df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]  # Reasonable age range
            
            if 'Gender' in df.columns:
                df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()
                # Standardize gender values
                gender_mapping = {
                    'MALE': 'M', 'M': 'M', '1': 'M',
                    'FEMALE': 'F', 'F': 'F', '0': 'F'
                }
                df['Gender'] = df['Gender'].map(gender_mapping).fillna(df['Gender'])
                
                # Keep only valid genders
                df = df[df['Gender'].isin(['M', 'F'])]
            
            logger.info(f"Loaded {len(df)} patient records from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading patient data from {filepath}: {str(e)}")
            raise
    
    def load_hcc_data(self, filepath: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load HCC master data (optional).
        
        Args:
            filepath: Path to HCC data file
            
        Returns:
            DataFrame with HCC data or None if not available
        """
        if filepath is None:
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            # Validate expected columns
            expected_columns = ['HCCCode', 'Description', 'CMS_Risk_Score']
            available_columns = [col for col in expected_columns if col in df.columns]
            
            if not available_columns:
                logger.warning(f"No expected HCC columns found in {filepath}")
                return None
            
            # Clean data
            df = df.dropna(subset=available_columns[:1])  # At least HCC code should be present
            
            logger.info(f"Loaded {len(df)} HCC codes from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading HCC data from {filepath}: {str(e)}")
            return None
    
    def validate_data_consistency(self, 
                                 icd_data: pd.DataFrame, 
                                 patient_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate consistency between ICD master and patient data.
        
        Args:
            icd_data: ICD master DataFrame
            patient_data: Patient data DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Check ICD code overlap
        master_codes = set(icd_data['ICDCode'].unique())
        patient_codes = set(patient_data['ICDCode'].unique())
        
        missing_in_master = patient_codes - master_codes
        unused_in_master = master_codes - patient_codes
        
        validation_results['statistics'] = {
            'master_codes_count': len(master_codes),
            'patient_codes_count': len(patient_codes),
            'overlap_count': len(master_codes & patient_codes),
            'missing_in_master_count': len(missing_in_master),
            'unused_in_master_count': len(unused_in_master)
        }
        
        # Add warnings for missing codes
        if missing_in_master:
            validation_results['warnings'].append(
                f"{len(missing_in_master)} ICD codes in patient data not found in master data"
            )
            if len(missing_in_master) <= 10:
                validation_results['warnings'].append(
                    f"Missing codes: {list(missing_in_master)[:10]}"
                )
        
        # Check for duplicate patient-ICD combinations
        if 'MemberID' in patient_data.columns:
            patient_icd_combinations = patient_data[['MemberID', 'ICDCode']].drop_duplicates()
            if len(patient_icd_combinations) < len(patient_data):
                duplicate_count = len(patient_data) - len(patient_icd_combinations)
                validation_results['warnings'].append(
                    f"{duplicate_count} duplicate patient-ICD combinations found"
                )
        
        # Check patient statistics
        if 'MemberID' in patient_data.columns:
            patient_stats = patient_data.groupby('MemberID')['ICDCode'].count()
            validation_results['statistics'].update({
                'total_patients': len(patient_stats),
                'avg_codes_per_patient': patient_stats.mean(),
                'median_codes_per_patient': patient_stats.median(),
                'max_codes_per_patient': patient_stats.max(),
                'min_codes_per_patient': patient_stats.min()
            })
        
        logger.info(f"Data validation completed: {validation_results['statistics']}")
        return validation_results
    
    def prepare_training_data(self, 
                            patient_data: pd.DataFrame,
                            test_size: float = 0.2,
                            min_codes_per_patient: int = 2,
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and test datasets with proper patient-level splitting.
        
        Args:
            patient_data: Patient data DataFrame
            test_size: Fraction of patients for testing
            min_codes_per_patient: Minimum codes required per patient
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        # Group by patient and filter patients with minimum codes
        patient_groups = patient_data.groupby('MemberID')
        valid_patients = []
        
        for patient_id, group in patient_groups:
            if len(group) >= min_codes_per_patient:
                valid_patients.append(patient_id)
        
        logger.info(f"Found {len(valid_patients)} patients with >= {min_codes_per_patient} codes")
        
        # Random split of patients
        np.random.seed(random_state)
        np.random.shuffle(valid_patients)
        
        split_index = int(len(valid_patients) * (1 - test_size))
        train_patients = set(valid_patients[:split_index])
        test_patients = set(valid_patients[split_index:])
        
        # Split data by patients
        train_data = patient_data[patient_data['MemberID'].isin(train_patients)]
        test_data = patient_data[patient_data['MemberID'].isin(test_patients)]
        
        logger.info(f"Training data: {len(train_patients)} patients, {len(train_data)} records")
        logger.info(f"Test data: {len(test_patients)} patients, {len(test_data)} records")
        
        return train_data, test_data
    
    def create_leave_one_out_samples(self, 
                                   patient_data: pd.DataFrame,
                                   max_samples_per_patient: int = 5) -> List[Dict[str, Any]]:
        """
        Create leave-one-out samples for training.
        
        Args:
            patient_data: Patient data DataFrame
            max_samples_per_patient: Maximum samples to create per patient
            
        Returns:
            List of training samples
        """
        samples = []
        
        # Group by patient
        patient_groups = patient_data.groupby('MemberID')
        
        for patient_id, group in patient_groups:
            patient_codes = group['ICDCode'].tolist()
            
            if len(patient_codes) < 2:  # Need at least 2 codes for leave-one-out
                continue
            
            # Get patient demographics if available
            age = group['Age'].iloc[0] if 'Age' in group.columns else None
            gender = group['Gender'].iloc[0] if 'Gender' in group.columns else None
            
            # Create leave-one-out samples
            sample_count = 0
            for i, target_code in enumerate(patient_codes):
                if sample_count >= max_samples_per_patient:
                    break
                
                # Input codes are all codes except the target
                input_codes = [code for j, code in enumerate(patient_codes) if j != i]
                
                sample = {
                    'patient_id': patient_id,
                    'input_codes': input_codes,
                    'target_code': target_code,
                    'age': age,
                    'gender': gender,
                    'is_positive': True  # This is a positive sample
                }
                
                samples.append(sample)
                sample_count += 1
        
        logger.info(f"Created {len(samples)} leave-one-out samples")
        return samples
    
    def save_processed_data(self, data: Any, filepath: str):
        """Save processed data to file."""
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif filepath.endswith('.csv'):
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            else:
                raise ValueError("CSV format only supports DataFrames")
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        logger.info(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, filepath: str) -> Any:
        """Load processed data from file."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        elif filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        logger.info(f"Loaded processed data from {filepath}")
        return data
    
    def get_data_summary(self, patient_data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive summary of loaded data."""
        summary = {
            'total_records': len(patient_data),
            'unique_patients': patient_data['MemberID'].nunique() if 'MemberID' in patient_data.columns else 0,
            'unique_icd_codes': patient_data['ICDCode'].nunique() if 'ICDCode' in patient_data.columns else 0,
            'date_range': None,
            'demographic_info': {}
        }
        
        # Patient statistics
        if 'MemberID' in patient_data.columns:
            patient_stats = patient_data.groupby('MemberID')['ICDCode'].count()
            summary['patient_statistics'] = {
                'avg_codes_per_patient': float(patient_stats.mean()),
                'median_codes_per_patient': float(patient_stats.median()),
                'std_codes_per_patient': float(patient_stats.std()),
                'min_codes_per_patient': int(patient_stats.min()),
                'max_codes_per_patient': int(patient_stats.max())
            }
        
        # Age statistics
        if 'Age' in patient_data.columns:
            age_data = patient_data['Age'].dropna()
            summary['demographic_info']['age'] = {
                'mean': float(age_data.mean()),
                'median': float(age_data.median()),
                'std': float(age_data.std()),
                'min': int(age_data.min()),
                'max': int(age_data.max()),
                'count': len(age_data)
            }
        
        # Gender distribution
        if 'Gender' in patient_data.columns:
            gender_dist = patient_data['Gender'].value_counts().to_dict()
            summary['demographic_info']['gender'] = gender_dist
        
        # ICD category distribution
        if 'ICDCode' in patient_data.columns:
            categories = patient_data['ICDCode'].str[0].value_counts().head(10).to_dict()
            summary['top_icd_categories'] = categories
        
        return summary