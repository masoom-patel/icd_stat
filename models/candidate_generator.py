"""
CandidateGenerator: Lists candidate ICDs/HCCs for prediction with filtering and relevance scoring.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional, Tuple, Union
import logging
from collections import defaultdict, Counter

from config.model_config import get_config

logger = logging.getLogger(__name__)

class CandidateGenerator:
    """
    Generates and filters candidate ICDs/HCCs for disease prediction.
    Implements relevance-based filtering and frequency thresholds.
    """
    
    def __init__(self, 
                 icd_data: pd.DataFrame, 
                 patient_data: Optional[pd.DataFrame] = None,
                 hcc_data: Optional[pd.DataFrame] = None):
        """
        Initialize the CandidateGenerator.
        
        Args:
            icd_data: DataFrame with ICD codes and descriptions
            patient_data: DataFrame with patient-ICD associations
            hcc_data: DataFrame with HCC information
        """
        self.config = get_config()
        self.icd_data = icd_data
        self.patient_data = patient_data
        self.hcc_data = hcc_data
        
        # Prepare candidate pools
        self.all_icd_codes = set(icd_data['ICDCode'].tolist()) if 'ICDCode' in icd_data.columns else set()
        self.all_hcc_codes = set()
        
        # Build frequency and co-occurrence statistics
        self._build_statistics()
        
        # Build HCC mappings
        self._build_hcc_mappings()
        
    def _build_statistics(self):
        """Build frequency and co-occurrence statistics from patient data."""
        self.icd_frequencies = Counter()
        self.icd_cooccurrence = defaultdict(Counter)
        self.patient_icd_counts = defaultdict(int)
        
        if self.patient_data is not None and 'ICDCode' in self.patient_data.columns:
            # Calculate ICD frequencies
            self.icd_frequencies = Counter(self.patient_data['ICDCode'].tolist())
            
            # Calculate patient-level ICD counts
            if 'MemberID' in self.patient_data.columns:
                patient_icds = self.patient_data.groupby('MemberID')['ICDCode'].apply(list).to_dict()
                
                for patient_id, icds in patient_icds.items():
                    self.patient_icd_counts[patient_id] = len(set(icds))
                    
                    # Build co-occurrence matrix
                    unique_icds = list(set(icds))
                    for i, icd1 in enumerate(unique_icds):
                        for icd2 in unique_icds[i+1:]:
                            self.icd_cooccurrence[icd1][icd2] += 1
                            self.icd_cooccurrence[icd2][icd1] += 1
        
        logger.info(f"Built statistics for {len(self.icd_frequencies)} ICD codes")
    
    def _build_hcc_mappings(self):
        """Build mappings between ICD and HCC codes."""
        self.icd_to_hcc = {}
        self.hcc_to_icds = defaultdict(list)
        
        if self.icd_data is not None:
            # Extract HCC mappings from ICD data
            hcc_columns = [col for col in self.icd_data.columns if 'HCC' in col]
            
            for _, row in self.icd_data.iterrows():
                icd_code = row.get('ICDCode')
                if pd.notna(icd_code):
                    hccs = []
                    for hcc_col in hcc_columns:
                        hcc_val = row.get(hcc_col)
                        if pd.notna(hcc_val) and str(hcc_val).strip() and str(hcc_val) != 'nan':
                            hcc_code = f"{hcc_col}_{hcc_val}"
                            hccs.append(hcc_code)
                            self.hcc_to_icds[hcc_code].append(icd_code)
                    
                    if hccs:
                        self.icd_to_hcc[icd_code] = hccs
        
        self.all_hcc_codes = set(self.hcc_to_icds.keys())
        logger.info(f"Built HCC mappings for {len(self.icd_to_hcc)} ICD codes and {len(self.all_hcc_codes)} HCC codes")
    
    def generate_icd_candidates(self, 
                               input_icds: List[str], 
                               exclude_input: bool = True,
                               max_candidates: Optional[int] = None) -> List[Dict[str, Union[str, float, int]]]:
        """
        Generate candidate ICD codes for prediction.
        
        Args:
            input_icds: List of input ICD codes
            exclude_input: Whether to exclude input ICDs from candidates
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of candidate dictionaries with relevance scores
        """
        input_set = set(input_icds)
        max_candidates = max_candidates or self.config.max_candidates
        
        candidates = []
        
        for icd_code in self.all_icd_codes:
            # Exclude input codes if requested
            if exclude_input and icd_code in input_set:
                continue
            
            # Apply frequency filter
            frequency = self.icd_frequencies.get(icd_code, 0)
            if self.config.filter_rare_codes and frequency < self.config.min_frequency_threshold:
                continue
            
            # Calculate relevance score
            relevance_score = self._calculate_icd_relevance(icd_code, input_icds)
            
            # Get ICD information
            icd_info = self._get_icd_info(icd_code)
            
            candidate = {
                'code': icd_code,
                'type': 'ICD',
                'description': icd_info.get('description', 'Unknown'),
                'frequency': frequency,
                'relevance_score': relevance_score,
                'hcc_mappings': self.icd_to_hcc.get(icd_code, [])
            }
            
            candidates.append(candidate)
        
        # Sort by relevance score (descending) and limit
        candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return candidates[:max_candidates]
    
    def generate_hcc_candidates(self, 
                               input_icds: List[str],
                               max_candidates: Optional[int] = None) -> List[Dict[str, Union[str, float, int]]]:
        """
        Generate candidate HCC codes for prediction.
        
        Args:
            input_icds: List of input ICD codes
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of HCC candidate dictionaries with relevance scores
        """
        input_set = set(input_icds)
        max_candidates = max_candidates or self.config.max_candidates
        
        # Get HCCs already covered by input ICDs
        input_hccs = set()
        for icd in input_icds:
            input_hccs.update(self.icd_to_hcc.get(icd, []))
        
        candidates = []
        
        for hcc_code in self.all_hcc_codes:
            # Exclude HCCs already covered by input
            if hcc_code in input_hccs:
                continue
            
            # Calculate frequency (sum of constituent ICD frequencies)
            constituent_icds = self.hcc_to_icds.get(hcc_code, [])
            frequency = sum(self.icd_frequencies.get(icd, 0) for icd in constituent_icds)
            
            # Apply frequency filter
            if self.config.filter_rare_codes and frequency < self.config.min_frequency_threshold:
                continue
            
            # Calculate relevance score
            relevance_score = self._calculate_hcc_relevance(hcc_code, input_icds)
            
            # Get HCC information
            hcc_info = self._get_hcc_info(hcc_code)
            
            candidate = {
                'code': hcc_code,
                'type': 'HCC',
                'description': hcc_info.get('description', 'Unknown'),
                'frequency': frequency,
                'relevance_score': relevance_score,
                'constituent_icds': constituent_icds,
                'cms_risk_score': hcc_info.get('cms_risk_score', 0.0)
            }
            
            candidates.append(candidate)
        
        # Sort by relevance score (descending) and limit
        candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return candidates[:max_candidates]
    
    def generate_mixed_candidates(self, 
                                 input_icds: List[str],
                                 icd_ratio: float = 0.7,
                                 max_candidates: Optional[int] = None) -> List[Dict[str, Union[str, float, int]]]:
        """
        Generate mixed ICD and HCC candidates.
        
        Args:
            input_icds: List of input ICD codes
            icd_ratio: Ratio of ICD to HCC candidates
            max_candidates: Maximum total candidates to return
            
        Returns:
            Combined list of ICD and HCC candidates
        """
        max_candidates = max_candidates or self.config.max_candidates
        
        # Calculate split
        max_icd_candidates = int(max_candidates * icd_ratio)
        max_hcc_candidates = max_candidates - max_icd_candidates
        
        # Generate candidates
        icd_candidates = self.generate_icd_candidates(input_icds, max_candidates=max_icd_candidates)
        hcc_candidates = self.generate_hcc_candidates(input_icds, max_candidates=max_hcc_candidates)
        
        # Combine and sort by relevance
        all_candidates = icd_candidates + hcc_candidates
        all_candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return all_candidates[:max_candidates]
    
    def _calculate_icd_relevance(self, candidate_icd: str, input_icds: List[str]) -> float:
        """Calculate relevance score for an ICD candidate."""
        relevance_score = 0.0
        
        # Base frequency score (normalized)
        frequency = self.icd_frequencies.get(candidate_icd, 0)
        max_frequency = max(self.icd_frequencies.values()) if self.icd_frequencies else 1
        frequency_score = frequency / max_frequency if max_frequency > 0 else 0
        relevance_score += 0.3 * frequency_score
        
        # Co-occurrence score
        cooccurrence_score = 0.0
        for input_icd in input_icds:
            cooccur_count = self.icd_cooccurrence[input_icd].get(candidate_icd, 0)
            input_frequency = self.icd_frequencies.get(input_icd, 0)
            
            if input_frequency > 0:
                conditional_prob = cooccur_count / input_frequency
                cooccurrence_score += conditional_prob
        
        # Average co-occurrence across input ICDs
        if input_icds:
            cooccurrence_score /= len(input_icds)
        
        relevance_score += 0.5 * cooccurrence_score
        
        # Category similarity bonus (same ICD category)
        category_score = 0.0
        candidate_category = candidate_icd[0] if candidate_icd else ''
        for input_icd in input_icds:
            input_category = input_icd[0] if input_icd else ''
            if candidate_category == input_category:
                category_score += 1
        
        if input_icds:
            category_score /= len(input_icds)
        
        relevance_score += 0.2 * category_score
        
        return relevance_score
    
    def _calculate_hcc_relevance(self, candidate_hcc: str, input_icds: List[str]) -> float:
        """Calculate relevance score for an HCC candidate."""
        relevance_score = 0.0
        
        # Get constituent ICDs for the HCC
        constituent_icds = self.hcc_to_icds.get(candidate_hcc, [])
        
        if not constituent_icds:
            return 0.0
        
        # Calculate average ICD relevance for constituent ICDs
        icd_relevances = []
        for constituent_icd in constituent_icds:
            icd_rel = self._calculate_icd_relevance(constituent_icd, input_icds)
            icd_relevances.append(icd_rel)
        
        # Use max relevance of constituent ICDs
        relevance_score = max(icd_relevances) if icd_relevances else 0.0
        
        # Bonus for HCC complexity (more constituent ICDs might indicate broader relevance)
        complexity_bonus = min(len(constituent_icds) / 10.0, 0.1)  # Cap at 0.1
        relevance_score += complexity_bonus
        
        return relevance_score
    
    def _get_icd_info(self, icd_code: str) -> Dict[str, str]:
        """Get information about an ICD code."""
        if self.icd_data is not None:
            matching_rows = self.icd_data[self.icd_data['ICDCode'] == icd_code]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                return {
                    'description': str(row.get('Description', 'Unknown')),
                    'code': icd_code
                }
        
        return {'description': 'Unknown', 'code': icd_code}
    
    def _get_hcc_info(self, hcc_code: str) -> Dict[str, Union[str, float]]:
        """Get information about an HCC code."""
        # Parse HCC code (format: HCC_column_value)
        parts = hcc_code.split('_')
        if len(parts) >= 3:
            hcc_column = '_'.join(parts[:-1])
            hcc_value = parts[-1]
            
            # Try to find description from constituent ICDs
            constituent_icds = self.hcc_to_icds.get(hcc_code, [])
            descriptions = []
            
            for icd in constituent_icds[:3]:  # Limit to first 3 for description
                icd_info = self._get_icd_info(icd)
                descriptions.append(icd_info['description'])
            
            description = f"HCC {hcc_value}: " + "; ".join(descriptions) if descriptions else f"HCC {hcc_value}"
            
            return {
                'description': description,
                'code': hcc_code,
                'hcc_column': hcc_column,
                'hcc_value': hcc_value,
                'cms_risk_score': self._get_cms_risk_score(hcc_code)
            }
        
        return {
            'description': 'Unknown HCC',
            'code': hcc_code,
            'cms_risk_score': 0.0
        }
    
    def _get_cms_risk_score(self, hcc_code: str) -> float:
        """Get CMS risk score for HCC (placeholder - would need actual CMS data)."""
        # This is a simplified scoring - in reality, CMS scores are complex
        parts = hcc_code.split('_')
        if len(parts) >= 3:
            try:
                hcc_value = int(parts[-1])
                # Simple scoring based on HCC number (lower numbers often = higher risk)
                if hcc_value <= 10:
                    return 2.5
                elif hcc_value <= 50:
                    return 1.5
                elif hcc_value <= 100:
                    return 1.0
                else:
                    return 0.5
            except ValueError:
                pass
        
        return 0.0
    
    def get_candidate_statistics(self) -> Dict[str, Union[int, float]]:
        """Get statistics about the candidate generation process."""
        return {
            'total_icd_codes': len(self.all_icd_codes),
            'total_hcc_codes': len(self.all_hcc_codes),
            'frequent_icd_codes': len([code for code, freq in self.icd_frequencies.items() 
                                     if freq >= self.config.min_frequency_threshold]),
            'total_patients': len(self.patient_icd_counts) if self.patient_icd_counts else 0,
            'avg_icds_per_patient': np.mean(list(self.patient_icd_counts.values())) if self.patient_icd_counts else 0,
            'max_candidates_config': self.config.max_candidates,
            'min_frequency_threshold': self.config.min_frequency_threshold
        }
    
    def filter_candidates_by_demographics(self, 
                                        candidates: List[Dict], 
                                        age: Optional[int] = None,
                                        gender: Optional[str] = None) -> List[Dict]:
        """
        Filter candidates based on demographic appropriateness.
        
        Args:
            candidates: List of candidate dictionaries
            age: Patient age
            gender: Patient gender
            
        Returns:
            Filtered list of candidates
        """
        filtered_candidates = []
        
        for candidate in candidates:
            if self._is_demographically_appropriate(candidate, age, gender):
                filtered_candidates.append(candidate)
        
        return filtered_candidates
    
    def _is_demographically_appropriate(self, 
                                     candidate: Dict, 
                                     age: Optional[int] = None,
                                     gender: Optional[str] = None) -> bool:
        """Check if a candidate is demographically appropriate."""
        code = candidate.get('code', '')
        description = candidate.get('description', '').lower()
        
        # Age-based filtering
        if age is not None:
            # Pediatric conditions (simplified rules)
            if age < 18:
                if any(term in description for term in ['pregnancy', 'childbirth', 'prostate']):
                    return False
            
            # Adult conditions
            if age >= 18:
                if any(term in description for term in ['congenital', 'birth', 'newborn']):
                    # Allow some congenital conditions that persist into adulthood
                    if not any(term in description for term in ['heart', 'defect', 'anomaly']):
                        return False
        
        # Gender-based filtering
        if gender is not None:
            gender_lower = gender.lower()
            
            # Male-specific exclusions
            if gender_lower in ['m', 'male']:
                if any(term in description for term in ['pregnancy', 'childbirth', 'ovarian', 'cervical', 'uterine']):
                    return False
            
            # Female-specific exclusions  
            if gender_lower in ['f', 'female']:
                if any(term in description for term in ['prostate', 'testicular']):
                    return False
        
        return True