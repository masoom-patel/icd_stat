"""
Ranker: Ranks candidates by probability and returns top-K predictions with confidence scores.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from collections import defaultdict

from config.model_config import get_config

logger = logging.getLogger(__name__)

class Ranker:
    """
    Ranks disease prediction candidates and returns top-K predictions with confidence scores.
    Provides comprehensive ranking with explanations and confidence estimation.
    """
    
    def __init__(self, scorer, feature_builder, rule_engine=None):
        """
        Initialize the Ranker.
        
        Args:
            scorer: Trained Scorer instance
            feature_builder: FeatureBuilder instance
            rule_engine: Optional RuleTriggerEngine instance
        """
        self.config = get_config()
        self.scorer = scorer
        self.feature_builder = feature_builder
        self.rule_engine = rule_engine
        
        # Ranking parameters
        self.top_k = self.config.top_k_predictions
        self.confidence_threshold = self.config.confidence_threshold
        self.return_explanations = self.config.return_explanations
    
    def rank_candidates(self, 
                       input_icds: List[str],
                       candidates: List[Dict[str, Any]],
                       age: Optional[int] = None,
                       gender: Optional[str] = None,
                       patient_data: Optional[Dict] = None,
                       top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rank candidates and return top-K predictions.
        
        Args:
            input_icds: List of input ICD codes
            candidates: List of candidate dictionaries
            age: Patient age
            gender: Patient gender  
            patient_data: Additional patient data
            top_k: Number of top predictions to return
            
        Returns:
            List of ranked predictions with scores and explanations
        """
        top_k = top_k or self.top_k
        
        if not candidates:
            return []
        
        logger.info(f"Ranking {len(candidates)} candidates for input ICDs: {input_icds}")
        
        # Build features and get predictions
        scored_candidates = []
        
        for candidate in candidates:
            try:
                # Build features for this candidate
                features = self.feature_builder.build_features(
                    input_icds=input_icds,
                    candidate_code=candidate['code'],
                    candidate_type=candidate.get('type', 'ICD'),
                    age=age,
                    gender=gender,
                    patient_data=patient_data
                )
                
                # Get probability score
                probability = self.scorer.predict_proba(features.reshape(1, -1))[0]
                
                # Apply confidence threshold
                if probability < self.confidence_threshold:
                    continue
                
                # Calculate confidence metrics
                confidence_score = self._calculate_confidence(probability, candidate, input_icds)
                
                # Get rule explanations if available
                rule_explanations = []
                if self.rule_engine is not None and self.return_explanations:
                    rule_explanations = self.rule_engine.get_rule_explanations(
                        input_icds, candidate['code'], age, gender
                    )
                
                # Build result
                scored_candidate = {
                    'code': candidate['code'],
                    'type': candidate.get('type', 'ICD'),
                    'description': candidate.get('description', 'Unknown'),
                    'probability': float(probability),
                    'confidence_score': float(confidence_score),
                    'frequency': candidate.get('frequency', 0),
                    'relevance_score': candidate.get('relevance_score', 0.0),
                    'combined_score': self._calculate_combined_score(
                        probability, confidence_score, candidate.get('relevance_score', 0.0)
                    )
                }
                
                # Add explanations if requested
                if self.return_explanations:
                    scored_candidate['rule_explanations'] = rule_explanations
                    scored_candidate['feature_contributions'] = self._get_feature_contributions(features)
                
                # Add HCC information if available
                if 'hcc_mappings' in candidate:
                    scored_candidate['hcc_mappings'] = candidate['hcc_mappings']
                if 'cms_risk_score' in candidate:
                    scored_candidate['cms_risk_score'] = candidate['cms_risk_score']
                
                scored_candidates.append(scored_candidate)
                
            except Exception as e:
                logger.error(f"Error scoring candidate {candidate.get('code', 'unknown')}: {str(e)}")
                continue
        
        # Sort by combined score (descending)
        scored_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top-K results
        top_predictions = scored_candidates[:top_k]
        
        # Add ranking metadata
        for i, prediction in enumerate(top_predictions):
            prediction['rank'] = i + 1
            prediction['percentile'] = self._calculate_percentile(
                prediction['probability'], [p['probability'] for p in scored_candidates]
            )
        
        logger.info(f"Returning {len(top_predictions)} top predictions")
        return top_predictions
    
    def _calculate_confidence(self, probability: float, candidate: Dict, input_icds: List[str]) -> float:
        """Calculate confidence score for a prediction."""
        confidence_factors = []
        
        # Base probability confidence
        prob_confidence = min(probability * 2, 1.0)  # Higher probability = higher confidence
        confidence_factors.append(prob_confidence)
        
        # Frequency confidence (more frequent = more confident)
        frequency = candidate.get('frequency', 0)
        if frequency > 0:
            freq_confidence = min(np.log(frequency + 1) / 10.0, 1.0)
            confidence_factors.append(freq_confidence)
        
        # Relevance confidence
        relevance_score = candidate.get('relevance_score', 0.0)
        confidence_factors.append(relevance_score)
        
        # Rule-based confidence boost
        if self.rule_engine is not None:
            rule_results = self.rule_engine.evaluate_rules(
                input_icds, candidate['code']
            )
            rule_confidence = 0.0
            for rule_name, triggered in rule_results.items():
                if triggered:
                    rule_data = self.rule_engine.rules.get(rule_name, {})
                    rule_conf = rule_data.get('confidence', 0.5)
                    rule_confidence = max(rule_confidence, rule_conf)
            confidence_factors.append(rule_confidence)
        
        # Combine confidence factors (weighted average)
        if confidence_factors:
            weights = [0.4, 0.2, 0.2, 0.2][:len(confidence_factors)]
            weights = weights / np.sum(weights)  # Normalize weights
            confidence = np.average(confidence_factors, weights=weights)
        else:
            confidence = probability
        
        return min(confidence, 1.0)
    
    def _calculate_combined_score(self, probability: float, confidence: float, relevance: float) -> float:
        """Calculate combined ranking score."""
        # Weighted combination of different scores
        combined = (
            0.5 * probability +      # Model prediction
            0.3 * confidence +       # Confidence estimate
            0.2 * relevance         # Relevance score
        )
        return min(combined, 1.0)
    
    def _get_feature_contributions(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature contributions for explanation (simplified)."""
        # This is a simplified version - in practice, you'd use SHAP or similar
        feature_names = self.feature_builder.get_feature_names()
        
        if len(feature_names) != len(features):
            return {}
        
        # Get feature importance from model
        feature_importance = self.scorer.get_feature_importance()
        
        contributions = {}
        for i, (name, value) in enumerate(zip(feature_names, features)):
            importance = feature_importance.get(name, 0.0)
            contribution = float(value * importance)
            contributions[name] = contribution
        
        # Return top contributing features
        sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_contributions[:10])  # Top 10 features
    
    def _calculate_percentile(self, score: float, all_scores: List[float]) -> float:
        """Calculate percentile rank of a score."""
        if not all_scores:
            return 100.0
        
        rank = len([s for s in all_scores if s < score])
        percentile = (rank / len(all_scores)) * 100
        return percentile
    
    def get_prediction_summary(self, 
                             predictions: List[Dict[str, Any]], 
                             input_icds: List[str]) -> Dict[str, Any]:
        """Get summary statistics for predictions."""
        if not predictions:
            return {
                'total_predictions': 0,
                'avg_probability': 0.0,
                'avg_confidence': 0.0,
                'rule_triggered_count': 0,
                'top_categories': [],
                'risk_distribution': {}
            }
        
        # Basic statistics
        probabilities = [p['probability'] for p in predictions]
        confidences = [p['confidence_score'] for p in predictions]
        
        # Count rule triggers
        rule_triggered_count = 0
        for pred in predictions:
            if 'rule_explanations' in pred and pred['rule_explanations']:
                rule_triggered_count += 1
        
        # Category analysis
        categories = defaultdict(int)
        for pred in predictions:
            code = pred.get('code', '')
            if code and len(code) > 0:
                category = code[0].upper()
                categories[category] += 1
        
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Risk distribution
        risk_distribution = {
            'low_risk': len([p for p in predictions if p['probability'] < 0.3]),
            'medium_risk': len([p for p in predictions if 0.3 <= p['probability'] < 0.7]),
            'high_risk': len([p for p in predictions if p['probability'] >= 0.7])
        }
        
        return {
            'total_predictions': len(predictions),
            'avg_probability': np.mean(probabilities),
            'max_probability': np.max(probabilities),
            'min_probability': np.min(probabilities),
            'avg_confidence': np.mean(confidences),
            'rule_triggered_count': rule_triggered_count,
            'top_categories': top_categories,
            'risk_distribution': risk_distribution,
            'input_icd_count': len(input_icds)
        }
    
    def explain_ranking(self, 
                       predictions: List[Dict[str, Any]], 
                       input_icds: List[str]) -> Dict[str, Any]:
        """Provide detailed explanation of ranking methodology."""
        
        explanation = {
            'ranking_methodology': {
                'primary_score': 'ML model probability (50% weight)',
                'confidence_boost': 'Frequency and relevance confidence (30% weight)',
                'relevance_factor': 'Co-occurrence and similarity relevance (20% weight)',
                'rule_integration': 'Medical rules provide additional confidence',
                'threshold_filtering': f'Minimum probability threshold: {self.confidence_threshold}'
            },
            'model_info': self.scorer.get_model_info(),
            'feature_types': self.config.feature_types,
            'prediction_summary': self.get_prediction_summary(predictions, input_icds)
        }
        
        # Add top features from most confident prediction
        if predictions:
            top_prediction = predictions[0]
            if 'feature_contributions' in top_prediction:
                explanation['top_contributing_features'] = top_prediction['feature_contributions']
        
        return explanation
    
    def filter_predictions_by_demographics(self, 
                                         predictions: List[Dict[str, Any]],
                                         age: Optional[int] = None,
                                         gender: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter predictions based on demographic appropriateness."""
        filtered_predictions = []
        
        for pred in predictions:
            if self._is_demographically_appropriate(pred, age, gender):
                filtered_predictions.append(pred)
            else:
                # Reduce probability for inappropriate predictions
                pred['probability'] *= 0.5
                pred['confidence_score'] *= 0.5
                pred['demographic_warning'] = True
                filtered_predictions.append(pred)
        
        return filtered_predictions
    
    def _is_demographically_appropriate(self, 
                                      prediction: Dict[str, Any], 
                                      age: Optional[int] = None,
                                      gender: Optional[str] = None) -> bool:
        """Check if prediction is demographically appropriate."""
        code = prediction.get('code', '')
        description = prediction.get('description', '').lower()
        
        # Age-based filtering
        if age is not None:
            if age < 18 and any(term in description for term in ['pregnancy', 'prostate']):
                return False
            if age >= 18 and any(term in description for term in ['congenital', 'birth defect']):
                # Allow some congenital conditions in adults
                if not any(term in description for term in ['heart', 'cardiac']):
                    return False
        
        # Gender-based filtering
        if gender is not None:
            gender_lower = gender.lower()
            
            if gender_lower in ['m', 'male']:
                if any(term in description for term in ['pregnancy', 'ovarian', 'cervical']):
                    return False
            
            if gender_lower in ['f', 'female']:
                if any(term in description for term in ['prostate', 'testicular']):
                    return False
        
        return True
    
    def batch_rank_candidates(self, 
                            batch_requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Rank candidates for multiple patients in batch."""
        results = []
        
        for request in batch_requests:
            try:
                input_icds = request.get('input_icds', [])
                candidates = request.get('candidates', [])
                age = request.get('age')
                gender = request.get('gender')
                patient_data = request.get('patient_data')
                top_k = request.get('top_k')
                
                predictions = self.rank_candidates(
                    input_icds, candidates, age, gender, patient_data, top_k
                )
                results.append(predictions)
                
            except Exception as e:
                logger.error(f"Error in batch ranking: {str(e)}")
                results.append([])
        
        return results
    
    def compare_predictions(self, 
                          predictions1: List[Dict[str, Any]], 
                          predictions2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare two sets of predictions."""
        codes1 = set(p['code'] for p in predictions1)
        codes2 = set(p['code'] for p in predictions2)
        
        overlap = codes1 & codes2
        unique_to_1 = codes1 - codes2
        unique_to_2 = codes2 - codes1
        
        # Calculate ranking correlation for overlapping codes
        if overlap:
            ranks1 = {p['code']: p['rank'] for p in predictions1 if p['code'] in overlap}
            ranks2 = {p['code']: p['rank'] for p in predictions2 if p['code'] in overlap}
            
            rank_pairs = [(ranks1[code], ranks2[code]) for code in overlap]
            if len(rank_pairs) > 1:
                from scipy.stats import spearmanr
                correlation, _ = spearmanr([r[0] for r in rank_pairs], [r[1] for r in rank_pairs])
            else:
                correlation = 1.0 if len(rank_pairs) == 1 else 0.0
        else:
            correlation = 0.0
        
        return {
            'overlap_count': len(overlap),
            'unique_to_first': len(unique_to_1),
            'unique_to_second': len(unique_to_2),
            'jaccard_similarity': len(overlap) / len(codes1 | codes2) if (codes1 | codes2) else 0,
            'rank_correlation': correlation,
            'overlapping_codes': list(overlap)
        }