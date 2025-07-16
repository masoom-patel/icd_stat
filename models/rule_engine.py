"""
RuleTriggerEngine: Implements domain-specific medical rules for disease prediction.
"""

import re
from typing import List, Dict, Set, Optional, Tuple, Union
import logging
from collections import defaultdict

from config.model_config import get_config

logger = logging.getLogger(__name__)

class RuleTriggerEngine:
    """
    Implements medical domain rules for predicting disease complications and comorbidities.
    Uses only current input ICDs to prevent data leakage.
    """
    
    def __init__(self):
        """Initialize the RuleTriggerEngine with medical rules."""
        self.config = get_config()
        self.rules = {}
        self._build_medical_rules()
        
    def _build_medical_rules(self):
        """Build the medical rule knowledge base."""
        
        # Diabetes complications and progression rules
        self.rules['diabetes_complications'] = {
            'trigger_codes': ['E10', 'E11', 'E12', 'E13', 'E14'],  # Diabetes codes
            'target_patterns': ['E1*.9', 'H36*', 'N08*', 'I79.2'],  # Diabetic complications
            'description': 'Diabetes leads to retinopathy, nephropathy, and vascular complications',
            'confidence': 0.8
        }
        
        # Cardiovascular disease progression
        self.rules['cardiac_complications'] = {
            'trigger_codes': ['I10', 'I20', 'I21', 'I25'],  # Hypertension, angina, MI, CAD
            'target_patterns': ['I50*', 'I42*', 'I48*'],  # Heart failure, cardiomyopathy, AFib
            'description': 'Cardiovascular disease progression to heart failure',
            'confidence': 0.75
        }
        
        # Respiratory comorbidities
        self.rules['respiratory_progression'] = {
            'trigger_codes': ['J44', 'J45'],  # COPD, Asthma
            'target_patterns': ['J44.0', 'J44.1', 'J96*', 'Z99.1*'],  # COPD exacerbations, respiratory failure
            'description': 'Chronic respiratory disease progression',
            'confidence': 0.7
        }
        
        # Chronic kidney disease progression
        self.rules['renal_progression'] = {
            'trigger_codes': ['N18.1', 'N18.2', 'N18.3'],  # Early CKD stages
            'target_patterns': ['N18.4', 'N18.5', 'N18.6', 'Z49*'],  # Advanced CKD, dialysis
            'description': 'Chronic kidney disease progression to ESRD',
            'confidence': 0.85
        }
        
        # Cancer progression and complications
        self.rules['oncology_complications'] = {
            'trigger_codes': ['C78*', 'C79*'],  # Metastatic cancer
            'target_patterns': ['R50*', 'R63*', 'Z51*'],  # Fever, cachexia, chemotherapy
            'description': 'Cancer complications and treatment-related conditions',
            'confidence': 0.6
        }
        
        # Metabolic syndrome cluster
        self.rules['metabolic_syndrome'] = {
            'trigger_codes': ['E78*', 'I10', 'E66*'],  # Dyslipidemia, hypertension, obesity
            'target_patterns': ['E11*', 'I25*'],  # Diabetes, coronary artery disease
            'description': 'Metabolic syndrome leading to diabetes and CAD',
            'confidence': 0.65
        }
        
        # Infectious disease complications
        self.rules['infectious_complications'] = {
            'trigger_codes': ['A41*', 'R65*'],  # Sepsis
            'target_patterns': ['J96*', 'N17*', 'R57*'],  # Respiratory failure, acute kidney injury, shock
            'description': 'Sepsis leading to organ failure',
            'confidence': 0.8
        }
        
        # Mental health comorbidities
        self.rules['psychiatric_comorbidity'] = {
            'trigger_codes': ['F32*', 'F33*'],  # Depression
            'target_patterns': ['F41*', 'F10*', 'Z87.891'],  # Anxiety, substance abuse, personal history
            'description': 'Depression associated with anxiety and substance use',
            'confidence': 0.55
        }
        
        # Surgical complications
        self.rules['surgical_complications'] = {
            'trigger_codes': ['T81*', 'T84*'],  # Surgical complications
            'target_patterns': ['T82*', 'T85*', 'Z98*'],  # Device complications, post-surgical states
            'description': 'Post-surgical complications and device issues',
            'confidence': 0.7
        }
        
        # Age-related conditions (elderly)
        self.rules['geriatric_conditions'] = {
            'trigger_codes': ['M79.3', 'R26*', 'R54'],  # Muscle weakness, gait issues, age-related frailty
            'target_patterns': ['S72*', 'F03*', 'R41*'],  # Hip fractures, dementia, cognitive issues
            'description': 'Age-related frailty leading to falls and cognitive decline',
            'confidence': 0.6,
            'age_requirement': 65
        }
        
        # Pregnancy-related conditions
        self.rules['pregnancy_complications'] = {
            'trigger_codes': ['O10*', 'O11*', 'O24*'],  # Gestational hypertension, diabetes
            'target_patterns': ['O36*', 'O60*', 'O71*'],  # Fetal complications, preterm labor
            'description': 'Pregnancy complications affecting mother and fetus',
            'confidence': 0.75,
            'gender_requirement': 'F'
        }
        
        # Pediatric developmental conditions
        self.rules['pediatric_development'] = {
            'trigger_codes': ['F84*', 'F90*'],  # Autism, ADHD
            'target_patterns': ['F80*', 'F81*', 'F82*'],  # Speech, learning, motor disorders
            'description': 'Developmental disorders with associated learning difficulties',
            'confidence': 0.6,
            'age_requirement': (0, 18)
        }
        
        logger.info(f"Built {len(self.rules)} medical rules")
    
    def evaluate_rules(self, 
                      input_icds: List[str], 
                      candidate_code: str,
                      age: Optional[int] = None, 
                      gender: Optional[str] = None) -> Dict[str, bool]:
        """
        Evaluate all rules against input ICDs and candidate.
        
        Args:
            input_icds: List of input ICD codes
            candidate_code: Candidate ICD/HCC code to predict
            age: Patient age (optional)
            gender: Patient gender (optional)
            
        Returns:
            Dictionary mapping rule names to boolean trigger status
        """
        results = {}
        
        for rule_name, rule_data in self.rules.items():
            try:
                triggered = self._evaluate_single_rule(
                    rule_data, input_icds, candidate_code, age, gender
                )
                results[rule_name] = triggered
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {str(e)}")
                results[rule_name] = False
        
        return results
    
    def _evaluate_single_rule(self, 
                             rule_data: Dict, 
                             input_icds: List[str], 
                             candidate_code: str,
                             age: Optional[int] = None, 
                             gender: Optional[str] = None) -> bool:
        """Evaluate a single rule."""
        
        # Check demographic requirements
        if not self._check_demographics(rule_data, age, gender):
            return False
        
        # Check if any trigger codes are present in input
        trigger_codes = rule_data.get('trigger_codes', [])
        has_trigger = self._has_matching_codes(input_icds, trigger_codes)
        
        if not has_trigger:
            return False
        
        # Check if candidate matches target patterns
        target_patterns = rule_data.get('target_patterns', [])
        matches_target = self._matches_patterns(candidate_code, target_patterns)
        
        return matches_target
    
    def _check_demographics(self, rule_data: Dict, age: Optional[int], gender: Optional[str]) -> bool:
        """Check if demographics meet rule requirements."""
        
        # Age requirements
        age_req = rule_data.get('age_requirement')
        if age_req is not None and age is not None:
            if isinstance(age_req, tuple):
                min_age, max_age = age_req
                if not (min_age <= age <= max_age):
                    return False
            elif isinstance(age_req, int):
                if age < age_req:
                    return False
        
        # Gender requirements
        gender_req = rule_data.get('gender_requirement')
        if gender_req is not None and gender is not None:
            if gender.upper() not in [gender_req.upper(), gender_req[0].upper()]:
                return False
        
        return True
    
    def _has_matching_codes(self, input_icds: List[str], trigger_codes: List[str]) -> bool:
        """Check if any input ICD matches trigger codes (supports wildcards)."""
        for input_icd in input_icds:
            for trigger_code in trigger_codes:
                if self._code_matches_pattern(input_icd, trigger_code):
                    return True
        return False
    
    def _matches_patterns(self, candidate_code: str, target_patterns: List[str]) -> bool:
        """Check if candidate code matches any target pattern."""
        for pattern in target_patterns:
            if self._code_matches_pattern(candidate_code, pattern):
                return True
        return False
    
    def _code_matches_pattern(self, code: str, pattern: str) -> bool:
        """Check if a code matches a pattern (supports wildcards)."""
        if not code or not pattern:
            return False
        
        # Direct match
        if code == pattern:
            return True
        
        # Wildcard patterns
        if '*' in pattern:
            # Convert pattern to regex
            regex_pattern = pattern.replace('*', '.*')
            regex_pattern = '^' + regex_pattern + '$'
            
            try:
                return bool(re.match(regex_pattern, code))
            except re.error:
                logger.warning(f"Invalid regex pattern: {regex_pattern}")
                return False
        
        # Prefix match
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            return code.startswith(prefix)
        
        return False
    
    def get_rule_explanations(self, 
                            input_icds: List[str], 
                            candidate_code: str,
                            age: Optional[int] = None, 
                            gender: Optional[str] = None) -> List[Dict[str, Union[str, float, bool]]]:
        """
        Get detailed explanations for triggered rules.
        
        Returns:
            List of rule explanations with confidence scores
        """
        explanations = []
        rule_results = self.evaluate_rules(input_icds, candidate_code, age, gender)
        
        for rule_name, triggered in rule_results.items():
            if triggered:
                rule_data = self.rules[rule_name]
                
                explanation = {
                    'rule_name': rule_name,
                    'description': rule_data.get('description', ''),
                    'confidence': rule_data.get('confidence', 0.5),
                    'triggered': triggered,
                    'trigger_codes': rule_data.get('trigger_codes', []),
                    'target_patterns': rule_data.get('target_patterns', []),
                    'matching_inputs': self._get_matching_inputs(input_icds, rule_data.get('trigger_codes', []))
                }
                
                explanations.append(explanation)
        
        # Sort by confidence (descending)
        explanations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return explanations
    
    def _get_matching_inputs(self, input_icds: List[str], trigger_codes: List[str]) -> List[str]:
        """Get input ICDs that match trigger codes."""
        matching = []
        for input_icd in input_icds:
            for trigger_code in trigger_codes:
                if self._code_matches_pattern(input_icd, trigger_code):
                    matching.append(input_icd)
                    break
        return matching
    
    def add_custom_rule(self, 
                       rule_name: str, 
                       trigger_codes: List[str], 
                       target_patterns: List[str],
                       description: str = "",
                       confidence: float = 0.5,
                       age_requirement: Optional[Union[int, Tuple[int, int]]] = None,
                       gender_requirement: Optional[str] = None):
        """Add a custom medical rule."""
        
        self.rules[rule_name] = {
            'trigger_codes': trigger_codes,
            'target_patterns': target_patterns,
            'description': description,
            'confidence': confidence
        }
        
        if age_requirement is not None:
            self.rules[rule_name]['age_requirement'] = age_requirement
        
        if gender_requirement is not None:
            self.rules[rule_name]['gender_requirement'] = gender_requirement
        
        logger.info(f"Added custom rule: {rule_name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a rule from the engine."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed rule: {rule_name}")
        else:
            logger.warning(f"Rule not found: {rule_name}")
    
    def get_rule_statistics(self) -> Dict[str, Union[int, List[str]]]:
        """Get statistics about the rule engine."""
        return {
            'total_rules': len(self.rules),
            'rule_names': list(self.rules.keys()),
            'rules_with_age_requirements': len([r for r in self.rules.values() 
                                              if 'age_requirement' in r]),
            'rules_with_gender_requirements': len([r for r in self.rules.values() 
                                                 if 'gender_requirement' in r])
        }
    
    def test_rule_coverage(self, 
                          sample_icds: List[List[str]], 
                          ages: Optional[List[int]] = None,
                          genders: Optional[List[str]] = None) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Test rule coverage across a sample of ICD patterns.
        
        Args:
            sample_icds: List of ICD code lists
            ages: Optional list of ages
            genders: Optional list of genders
            
        Returns:
            Dictionary with rule coverage statistics
        """
        rule_triggers = defaultdict(int)
        total_samples = len(sample_icds)
        
        for i, icd_list in enumerate(sample_icds):
            age = ages[i] if ages and i < len(ages) else None
            gender = genders[i] if genders and i < len(genders) else None
            
            # Test against all possible candidates (simplified)
            all_icds = set()
            for icds in sample_icds:
                all_icds.update(icds)
            
            for candidate in all_icds:
                if candidate not in icd_list:  # Don't test against input codes
                    rule_results = self.evaluate_rules(icd_list, candidate, age, gender)
                    
                    for rule_name, triggered in rule_results.items():
                        if triggered:
                            rule_triggers[rule_name] += 1
        
        # Calculate statistics
        statistics = {}
        for rule_name in self.rules.keys():
            trigger_count = rule_triggers.get(rule_name, 0)
            statistics[rule_name] = {
                'trigger_count': trigger_count,
                'trigger_rate': trigger_count / total_samples if total_samples > 0 else 0,
                'description': self.rules[rule_name].get('description', ''),
                'confidence': self.rules[rule_name].get('confidence', 0.5)
            }
        
        return statistics
    
    def validate_rules(self) -> Dict[str, List[str]]:
        """Validate all rules for correctness."""
        validation_results = {
            'valid_rules': [],
            'invalid_rules': [],
            'warnings': []
        }
        
        for rule_name, rule_data in self.rules.items():
            try:
                # Check required fields
                if 'trigger_codes' not in rule_data or 'target_patterns' not in rule_data:
                    validation_results['invalid_rules'].append(
                        f"{rule_name}: Missing trigger_codes or target_patterns"
                    )
                    continue
                
                # Validate trigger codes
                trigger_codes = rule_data.get('trigger_codes', [])
                if not trigger_codes:
                    validation_results['warnings'].append(
                        f"{rule_name}: No trigger codes defined"
                    )
                
                # Validate target patterns
                target_patterns = rule_data.get('target_patterns', [])
                if not target_patterns:
                    validation_results['warnings'].append(
                        f"{rule_name}: No target patterns defined"
                    )
                
                # Test pattern matching
                for pattern in target_patterns:
                    try:
                        self._code_matches_pattern("TEST", pattern)
                    except Exception as e:
                        validation_results['invalid_rules'].append(
                            f"{rule_name}: Invalid pattern '{pattern}': {str(e)}"
                        )
                        continue
                
                validation_results['valid_rules'].append(rule_name)
                
            except Exception as e:
                validation_results['invalid_rules'].append(
                    f"{rule_name}: Validation error: {str(e)}"
                )
        
        return validation_results