"""
Sample patient data for testing the disease prediction system.
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict
from pathlib import Path

def create_sample_patient_data(num_patients: int = 1000, 
                             avg_codes_per_patient: int = 5) -> pd.DataFrame:
    """
    Create sample patient data for testing.
    
    Args:
        num_patients: Number of patients to generate
        avg_codes_per_patient: Average ICD codes per patient
        
    Returns:
        DataFrame with sample patient data
    """
    # Common ICD codes for realistic sampling
    common_icds = [
        'I10', 'E11.9', 'J44.1', 'N18.9', 'F32.9',  # Hypertension, Diabetes, COPD, CKD, Depression
        'I25.9', 'I48.91', 'K21.9', 'M79.3', 'Z87.891',  # CAD, AFib, GERD, Pain, Personal history
        'E78.5', 'I50.9', 'J45.9', 'G47.33', 'F17.210',  # Dyslipidemia, Heart failure, Asthma, Sleep apnea, Nicotine
        'M25.9', 'R06.02', 'R50.9', 'R53.83', 'Z51.11',  # Joint pain, Shortness of breath, Fever, Fatigue, Chemotherapy
        'C78.0', 'C79.9', 'N17.9', 'J96.90', 'R65.20'   # Metastatic cancer, AKI, Respiratory failure, Sepsis
    ]
    
    # Additional ICD codes for variety
    additional_icds = [
        'A41.9', 'B95.8', 'C25.9', 'D64.9', 'E10.9',
        'F33.9', 'G93.1', 'H25.9', 'I21.9', 'J18.9',
        'K59.00', 'L03.90', 'M06.9', 'N39.0', 'O99.89',
        'P07.30', 'Q21.9', 'R31.9', 'S72.90', 'T81.9',
        'V09.20', 'W19.XXX', 'Y92.9', 'Z98.89'
    ]
    
    all_icds = common_icds + additional_icds
    
    # Gender distribution
    genders = ['M', 'F']
    
    # Age ranges with realistic distributions
    def sample_age():
        # Realistic age distribution for medical data
        if random.random() < 0.3:  # 30% younger patients
            return random.randint(18, 45)
        elif random.random() < 0.7:  # 40% middle-aged
            return random.randint(46, 65)
        else:  # 30% older patients
            return random.randint(66, 90)
    
    records = []
    
    for patient_id in range(1, num_patients + 1):
        # Generate patient demographics
        age = sample_age()
        gender = random.choice(genders)
        
        # Generate number of codes for this patient (Poisson distribution)
        num_codes = max(1, np.random.poisson(avg_codes_per_patient))
        
        # Sample ICD codes (weighted towards common codes)
        weights = [3] * len(common_icds) + [1] * len(additional_icds)
        patient_icds = random.choices(all_icds, weights=weights, k=num_codes)
        
        # Remove duplicates while preserving some
        unique_icds = list(set(patient_icds))
        
        # Add some disease progression patterns
        if 'E11.9' in unique_icds and random.random() < 0.4:  # Diabetes complications
            unique_icds.extend(['H36.0', 'N08.3'])  # Diabetic retinopathy, nephropathy
        
        if 'I10' in unique_icds and random.random() < 0.3:  # Hypertension complications
            unique_icds.append('I50.9')  # Heart failure
        
        if 'J44.1' in unique_icds and random.random() < 0.3:  # COPD exacerbation
            unique_icds.append('J96.90')  # Respiratory failure
        
        # Create records for each ICD code
        for icd_code in unique_icds:
            records.append({
                'MemberID': f'PAT_{patient_id:06d}',
                'ICDCode': icd_code,
                'Age': age,
                'Gender': gender
            })
    
    df = pd.DataFrame(records)
    
    # Sort by patient ID for better organization
    df = df.sort_values(['MemberID', 'ICDCode']).reset_index(drop=True)
    
    return df

def create_sample_icd_hcc_mapping() -> pd.DataFrame:
    """
    Create a sample ICD-HCC mapping for testing (subset of real data).
    
    Returns:
        DataFrame with sample ICD-HCC mappings
    """
    # Sample mappings based on common medical conditions
    sample_mappings = [
        # Diabetes
        {'ICDCode': 'E10.9', 'Description': 'Type 1 diabetes mellitus without complications', 'HCC_V24': 17, 'HCC_V28': 37, 'HCC_ESRD_V24': 17},
        {'ICDCode': 'E11.9', 'Description': 'Type 2 diabetes mellitus without complications', 'HCC_V24': 19, 'HCC_V28': 38, 'HCC_ESRD_V24': 19},
        
        # Cardiovascular
        {'ICDCode': 'I10', 'Description': 'Essential hypertension', 'HCC_V24': np.nan, 'HCC_V28': np.nan, 'HCC_ESRD_V24': np.nan},
        {'ICDCode': 'I25.9', 'Description': 'Chronic ischemic heart disease', 'HCC_V24': 83, 'HCC_V28': 186, 'HCC_ESRD_V24': 83},
        {'ICDCode': 'I50.9', 'Description': 'Heart failure, unspecified', 'HCC_V24': 85, 'HCC_V28': 188, 'HCC_ESRD_V24': 85},
        {'ICDCode': 'I48.91', 'Description': 'Unspecified atrial fibrillation', 'HCC_V24': 96, 'HCC_V28': 195, 'HCC_ESRD_V24': 96},
        
        # Respiratory
        {'ICDCode': 'J44.1', 'Description': 'Chronic obstructive pulmonary disease with acute exacerbation', 'HCC_V24': 111, 'HCC_V28': 276, 'HCC_ESRD_V24': 111},
        {'ICDCode': 'J45.9', 'Description': 'Asthma, unspecified', 'HCC_V24': np.nan, 'HCC_V28': np.nan, 'HCC_ESRD_V24': np.nan},
        {'ICDCode': 'J96.90', 'Description': 'Respiratory failure, unspecified, unspecified with hypoxia or hypercapnia', 'HCC_V24': 84, 'HCC_V28': 187, 'HCC_ESRD_V24': 84},
        
        # Renal
        {'ICDCode': 'N18.9', 'Description': 'Chronic kidney disease, unspecified', 'HCC_V24': 137, 'HCC_V28': 326, 'HCC_ESRD_V24': 137},
        {'ICDCode': 'N17.9', 'Description': 'Acute kidney failure, unspecified', 'HCC_V24': 135, 'HCC_V28': 324, 'HCC_ESRD_V24': 135},
        
        # Mental Health
        {'ICDCode': 'F32.9', 'Description': 'Major depressive disorder, single episode, unspecified', 'HCC_V24': 58, 'HCC_V28': 151, 'HCC_ESRD_V24': 58},
        {'ICDCode': 'F33.9', 'Description': 'Major depressive disorder, recurrent, unspecified', 'HCC_V24': 58, 'HCC_V28': 151, 'HCC_ESRD_V24': 58},
        
        # Cancer
        {'ICDCode': 'C78.0', 'Description': 'Secondary malignant neoplasm of lung', 'HCC_V24': 8, 'HCC_V28': 17, 'HCC_ESRD_V24': 8},
        {'ICDCode': 'C79.9', 'Description': 'Secondary malignant neoplasm of unspecified site', 'HCC_V24': 8, 'HCC_V28': 17, 'HCC_ESRD_V24': 8},
        {'ICDCode': 'C25.9', 'Description': 'Malignant neoplasm of pancreas, unspecified', 'HCC_V24': 8, 'HCC_V28': 17, 'HCC_ESRD_V24': 8},
        
        # Other common conditions
        {'ICDCode': 'E78.5', 'Description': 'Hyperlipidemia, unspecified', 'HCC_V24': np.nan, 'HCC_V28': np.nan, 'HCC_ESRD_V24': np.nan},
        {'ICDCode': 'K21.9', 'Description': 'Gastro-esophageal reflux disease without esophagitis', 'HCC_V24': np.nan, 'HCC_V28': np.nan, 'HCC_ESRD_V24': np.nan},
        {'ICDCode': 'M79.3', 'Description': 'Panniculitis, unspecified', 'HCC_V24': np.nan, 'HCC_V28': np.nan, 'HCC_ESRD_V24': np.nan},
        {'ICDCode': 'G47.33', 'Description': 'Obstructive sleep apnea (adult) (pediatric)', 'HCC_V24': 77, 'HCC_V28': 180, 'HCC_ESRD_V24': 77},
        
        # Diabetic complications
        {'ICDCode': 'H36.0', 'Description': 'Diabetic retinopathy', 'HCC_V24': 18, 'HCC_V28': 37, 'HCC_ESRD_V24': 18},
        {'ICDCode': 'N08.3', 'Description': 'Glomerular disorders in diabetes mellitus', 'HCC_V24': 18, 'HCC_V28': 37, 'HCC_ESRD_V24': 18},
        
        # Infectious diseases
        {'ICDCode': 'A41.9', 'Description': 'Sepsis, unspecified organism', 'HCC_V24': 2, 'HCC_V28': 2, 'HCC_ESRD_V24': 2},
        {'ICDCode': 'R65.20', 'Description': 'Severe sepsis without septic shock', 'HCC_V24': 2, 'HCC_V28': 2, 'HCC_ESRD_V24': 2},
        
        # Symptoms and signs
        {'ICDCode': 'R50.9', 'Description': 'Fever, unspecified', 'HCC_V24': np.nan, 'HCC_V28': np.nan, 'HCC_ESRD_V24': np.nan},
        {'ICDCode': 'R53.83', 'Description': 'Other fatigue', 'HCC_V24': np.nan, 'HCC_V28': np.nan, 'HCC_ESRD_V24': np.nan},
        {'ICDCode': 'R06.02', 'Description': 'Shortness of breath', 'HCC_V24': np.nan, 'HCC_V28': np.nan, 'HCC_ESRD_V24': np.nan}
    ]
    
    return pd.DataFrame(sample_mappings)

def save_sample_data(data_dir: str = "data/sample_data"):
    """
    Save sample data files for testing.
    
    Args:
        data_dir: Directory to save sample data
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample patient data
    print("Creating sample patient data...")
    patient_data = create_sample_patient_data(num_patients=1000, avg_codes_per_patient=5)
    patient_data.to_csv(data_path / "sample_patient_data.csv", index=False)
    print(f"Saved {len(patient_data)} patient records to sample_patient_data.csv")
    
    # Create sample ICD-HCC mapping
    print("Creating sample ICD-HCC mapping...")
    icd_hcc_data = create_sample_icd_hcc_mapping()
    icd_hcc_data.to_csv(data_path / "sample_icd_hcc_mapping.csv", index=False)
    print(f"Saved {len(icd_hcc_data)} ICD-HCC mappings to sample_icd_hcc_mapping.csv")
    
    # Create a smaller test dataset
    print("Creating small test dataset...")
    small_patient_data = create_sample_patient_data(num_patients=50, avg_codes_per_patient=3)
    small_patient_data.to_csv(data_path / "small_test_data.csv", index=False)
    print(f"Saved {len(small_patient_data)} records to small_test_data.csv")
    
    print(f"Sample data saved to {data_path}")

if __name__ == "__main__":
    save_sample_data()