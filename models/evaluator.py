"""
Evaluator: Comprehensive evaluation metrics for the disease prediction system.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional, Tuple, Union, Any
import logging
from collections import defaultdict
import math

try:
    from sklearn.metrics import ndcg_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from config.model_config import get_config

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Comprehensive evaluation metrics for disease prediction including:
    - Top-K Accuracy
    - MAP@K (Mean Average Precision)
    - Recall@K 
    - Hit Rate
    - NDCG@K
    """
    
    def __init__(self):
        """Initialize the Evaluator."""
        self.config = get_config()
        self.k_values = self.config.evaluation_k_values
        
    def evaluate_predictions(self, 
                           predictions: List[List[Dict[str, Any]]], 
                           ground_truth: List[List[str]],
                           patient_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate prediction performance against ground truth.
        
        Args:
            predictions: List of prediction lists for each patient
            ground_truth: List of true ICD lists for each patient  
            patient_ids: Optional patient identifiers
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        logger.info(f"Evaluating {len(predictions)} prediction sets")
        
        results = {}
        
        # Calculate metrics for each K value
        for k in self.k_values:
            k_results = {}
            
            # Top-K Accuracy
            k_results['accuracy_at_k'] = self._calculate_accuracy_at_k(predictions, ground_truth, k)
            
            # Mean Average Precision at K
            k_results['map_at_k'] = self._calculate_map_at_k(predictions, ground_truth, k)
            
            # Recall at K
            k_results['recall_at_k'] = self._calculate_recall_at_k(predictions, ground_truth, k)
            
            # Hit Rate at K
            k_results['hit_rate_at_k'] = self._calculate_hit_rate_at_k(predictions, ground_truth, k)
            
            # NDCG at K (if sklearn available)
            if SKLEARN_AVAILABLE:
                k_results['ndcg_at_k'] = self._calculate_ndcg_at_k(predictions, ground_truth, k)
            
            results[f'k_{k}'] = k_results
        
        # Overall metrics
        results['overall'] = self._calculate_overall_metrics(predictions, ground_truth)
        
        # Detailed analysis
        results['detailed_analysis'] = self._detailed_analysis(predictions, ground_truth, patient_ids)
        
        return results
    
    def _calculate_accuracy_at_k(self, predictions: List[List[Dict]], ground_truth: List[List[str]], k: int) -> float:
        """Calculate top-K accuracy."""
        correct_predictions = 0
        total_predictions = 0
        
        for pred_list, true_list in zip(predictions, ground_truth):
            if not true_list:  # Skip if no ground truth
                continue
            
            # Get top-K predicted codes
            top_k_codes = [p['code'] for p in pred_list[:k]]
            true_codes = set(true_list)
            
            # Check if any predicted code is correct
            if any(code in true_codes for code in top_k_codes):
                correct_predictions += 1
            
            total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _calculate_map_at_k(self, predictions: List[List[Dict]], ground_truth: List[List[str]], k: int) -> float:
        """Calculate Mean Average Precision at K."""
        average_precisions = []
        
        for pred_list, true_list in zip(predictions, ground_truth):
            if not true_list:  # Skip if no ground truth
                continue
            
            top_k_codes = [p['code'] for p in pred_list[:k]]
            true_codes = set(true_list)
            
            # Calculate average precision for this instance
            if not top_k_codes:
                average_precisions.append(0.0)
                continue
            
            precisions = []
            relevant_found = 0
            
            for i, code in enumerate(top_k_codes):
                if code in true_codes:
                    relevant_found += 1
                    precision_at_i = relevant_found / (i + 1)
                    precisions.append(precision_at_i)
            
            if precisions:
                avg_precision = np.mean(precisions)
            else:
                avg_precision = 0.0
            
            average_precisions.append(avg_precision)
        
        return np.mean(average_precisions) if average_precisions else 0.0
    
    def _calculate_recall_at_k(self, predictions: List[List[Dict]], ground_truth: List[List[str]], k: int) -> float:
        """Calculate recall at K."""
        recalls = []
        
        for pred_list, true_list in zip(predictions, ground_truth):
            if not true_list:  # Skip if no ground truth
                continue
            
            top_k_codes = set([p['code'] for p in pred_list[:k]])
            true_codes = set(true_list)
            
            # Calculate recall for this instance
            if true_codes:
                recall = len(top_k_codes & true_codes) / len(true_codes)
            else:
                recall = 0.0
            
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def _calculate_hit_rate_at_k(self, predictions: List[List[Dict]], ground_truth: List[List[str]], k: int) -> float:
        """Calculate hit rate at K (same as accuracy@K for single relevant items)."""
        return self._calculate_accuracy_at_k(predictions, ground_truth, k)
    
    def _calculate_ndcg_at_k(self, predictions: List[List[Dict]], ground_truth: List[List[str]], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K."""
        if not SKLEARN_AVAILABLE:
            return 0.0
        
        ndcg_scores = []
        
        for pred_list, true_list in zip(predictions, ground_truth):
            if not true_list:  # Skip if no ground truth
                continue
            
            # Create relevance scores (1 for relevant, 0 for irrelevant)
            top_k_codes = [p['code'] for p in pred_list[:k]]
            true_codes = set(true_list)
            
            if not top_k_codes:
                ndcg_scores.append(0.0)
                continue
            
            # Binary relevance (1 if relevant, 0 if not)
            relevance_scores = [1 if code in true_codes else 0 for code in top_k_codes]
            
            # Create ideal ranking (all relevant items first)
            ideal_scores = sorted(relevance_scores, reverse=True)
            
            if sum(relevance_scores) == 0:
                ndcg_scores.append(0.0)
            else:
                try:
                    # sklearn expects shape (n_samples, n_items)
                    y_true = np.array([relevance_scores])
                    y_score = np.array([ideal_scores])  # Use ideal as scores for now
                    
                    ndcg = ndcg_score(y_true, y_score, k=k)
                    ndcg_scores.append(ndcg)
                except Exception as e:
                    logger.warning(f"Error calculating NDCG: {str(e)}")
                    ndcg_scores.append(0.0)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def _calculate_overall_metrics(self, predictions: List[List[Dict]], ground_truth: List[List[str]]) -> Dict[str, float]:
        """Calculate overall system metrics."""
        total_predictions = sum(len(pred_list) for pred_list in predictions)
        total_ground_truth = sum(len(true_list) for true_list in ground_truth)
        
        # Coverage metrics
        all_predicted_codes = set()
        all_true_codes = set()
        
        for pred_list in predictions:
            all_predicted_codes.update(p['code'] for p in pred_list)
        
        for true_list in ground_truth:
            all_true_codes.update(true_list)
        
        code_coverage = len(all_predicted_codes & all_true_codes) / len(all_true_codes) if all_true_codes else 0.0
        
        # Average prediction confidence
        all_confidences = []
        all_probabilities = []
        
        for pred_list in predictions:
            for pred in pred_list:
                if 'confidence_score' in pred:
                    all_confidences.append(pred['confidence_score'])
                if 'probability' in pred:
                    all_probabilities.append(pred['probability'])
        
        return {
            'total_predictions': total_predictions,
            'total_ground_truth': total_ground_truth,
            'unique_predicted_codes': len(all_predicted_codes),
            'unique_true_codes': len(all_true_codes),
            'code_coverage': code_coverage,
            'avg_predictions_per_patient': total_predictions / len(predictions) if predictions else 0,
            'avg_true_codes_per_patient': total_ground_truth / len(ground_truth) if ground_truth else 0,
            'avg_confidence': np.mean(all_confidences) if all_confidences else 0.0,
            'avg_probability': np.mean(all_probabilities) if all_probabilities else 0.0
        }
    
    def _detailed_analysis(self, predictions: List[List[Dict]], ground_truth: List[List[str]], 
                          patient_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform detailed analysis of predictions."""
        
        # Category-wise performance
        category_stats = defaultdict(lambda: {'predicted': 0, 'true': 0, 'correct': 0})
        
        for i, (pred_list, true_list) in enumerate(zip(predictions, ground_truth)):
            predicted_codes = set(p['code'] for p in pred_list)
            true_codes = set(true_list)
            
            # Count by ICD category (first character)
            for code in predicted_codes:
                if code and len(code) > 0:
                    category = code[0].upper()
                    category_stats[category]['predicted'] += 1
                    if code in true_codes:
                        category_stats[category]['correct'] += 1
            
            for code in true_codes:
                if code and len(code) > 0:
                    category = code[0].upper()
                    category_stats[category]['true'] += 1
        
        # Calculate precision/recall by category
        category_metrics = {}
        for category, stats in category_stats.items():
            precision = stats['correct'] / stats['predicted'] if stats['predicted'] > 0 else 0.0
            recall = stats['correct'] / stats['true'] if stats['true'] > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            category_metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predicted_count': stats['predicted'],
                'true_count': stats['true'],
                'correct_count': stats['correct']
            }
        
        # Performance by patient characteristics
        patient_analysis = self._analyze_by_patient_characteristics(predictions, ground_truth, patient_ids)
        
        # Error analysis
        error_analysis = self._analyze_prediction_errors(predictions, ground_truth)
        
        return {
            'category_performance': category_metrics,
            'patient_analysis': patient_analysis,
            'error_analysis': error_analysis
        }
    
    def _analyze_by_patient_characteristics(self, predictions: List[List[Dict]], 
                                          ground_truth: List[List[str]], 
                                          patient_ids: Optional[List[str]]) -> Dict[str, Any]:
        """Analyze performance by patient characteristics."""
        
        # Group by number of input conditions
        performance_by_complexity = defaultdict(list)
        
        for pred_list, true_list in zip(predictions, ground_truth):
            # Approximate complexity by number of predictions
            complexity = len(pred_list)
            complexity_bin = 'low' if complexity <= 5 else 'medium' if complexity <= 15 else 'high'
            
            # Calculate accuracy for this patient
            predicted_codes = set(p['code'] for p in pred_list[:10])  # Top 10
            true_codes = set(true_list)
            
            accuracy = len(predicted_codes & true_codes) / len(true_codes) if true_codes else 0.0
            performance_by_complexity[complexity_bin].append(accuracy)
        
        # Calculate average performance by complexity
        complexity_metrics = {}
        for complexity, accuracies in performance_by_complexity.items():
            complexity_metrics[complexity] = {
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'patient_count': len(accuracies)
            }
        
        return {
            'performance_by_complexity': complexity_metrics
        }
    
    def _analyze_prediction_errors(self, predictions: List[List[Dict]], 
                                 ground_truth: List[List[str]]) -> Dict[str, Any]:
        """Analyze common prediction errors."""
        
        false_positives = defaultdict(int)  # Predicted but not true
        false_negatives = defaultdict(int)  # True but not predicted
        
        for pred_list, true_list in zip(predictions, ground_truth):
            predicted_codes = set(p['code'] for p in pred_list)
            true_codes = set(true_list)
            
            # Count false positives
            for code in predicted_codes - true_codes:
                false_positives[code] += 1
            
            # Count false negatives
            for code in true_codes - predicted_codes:
                false_negatives[code] += 1
        
        # Get most common errors
        top_false_positives = sorted(false_positives.items(), key=lambda x: x[1], reverse=True)[:10]
        top_false_negatives = sorted(false_negatives.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'top_false_positives': top_false_positives,
            'top_false_negatives': top_false_negatives,
            'total_false_positives': sum(false_positives.values()),
            'total_false_negatives': sum(false_negatives.values())
        }
    
    def evaluate_model_calibration(self, predictions: List[List[Dict]], 
                                 ground_truth: List[List[str]], 
                                 n_bins: int = 10) -> Dict[str, Any]:
        """Evaluate model calibration (reliability of predicted probabilities)."""
        
        # Collect all predictions with their outcomes
        prediction_outcomes = []
        
        for pred_list, true_list in zip(predictions, ground_truth):
            true_codes = set(true_list)
            
            for pred in pred_list:
                probability = pred.get('probability', 0.0)
                is_correct = pred['code'] in true_codes
                prediction_outcomes.append((probability, is_correct))
        
        if not prediction_outcomes:
            return {}
        
        # Sort by probability
        prediction_outcomes.sort(key=lambda x: x[0])
        
        # Create bins
        bin_size = len(prediction_outcomes) // n_bins
        calibration_data = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(prediction_outcomes)
            
            bin_predictions = prediction_outcomes[start_idx:end_idx]
            
            if bin_predictions:
                avg_probability = np.mean([p[0] for p in bin_predictions])
                accuracy = np.mean([p[1] for p in bin_predictions])
                bin_count = len(bin_predictions)
                
                calibration_data.append({
                    'bin_id': i,
                    'avg_probability': avg_probability,
                    'accuracy': accuracy,
                    'count': bin_count,
                    'calibration_error': abs(avg_probability - accuracy)
                })
        
        # Calculate Expected Calibration Error (ECE)
        total_predictions = len(prediction_outcomes)
        ece = sum(bin_data['count'] / total_predictions * bin_data['calibration_error'] 
                 for bin_data in calibration_data)
        
        return {
            'expected_calibration_error': ece,
            'calibration_data': calibration_data,
            'total_predictions': total_predictions
        }
    
    def compare_model_performances(self, 
                                 results1: Dict[str, Any], 
                                 results2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance between two model results."""
        
        comparison = {}
        
        # Compare metrics for each K value
        for k in self.k_values:
            k_key = f'k_{k}'
            if k_key in results1 and k_key in results2:
                k_comparison = {}
                
                for metric in ['accuracy_at_k', 'map_at_k', 'recall_at_k', 'hit_rate_at_k']:
                    if metric in results1[k_key] and metric in results2[k_key]:
                        val1 = results1[k_key][metric]
                        val2 = results2[k_key][metric]
                        
                        k_comparison[metric] = {
                            'model1': val1,
                            'model2': val2,
                            'difference': val2 - val1,
                            'improvement': ((val2 - val1) / val1 * 100) if val1 > 0 else 0.0
                        }
                
                comparison[k_key] = k_comparison
        
        # Overall comparison
        if 'overall' in results1 and 'overall' in results2:
            overall_comparison = {}
            
            for metric in ['code_coverage', 'avg_confidence', 'avg_probability']:
                if metric in results1['overall'] and metric in results2['overall']:
                    val1 = results1['overall'][metric]
                    val2 = results2['overall'][metric]
                    
                    overall_comparison[metric] = {
                        'model1': val1,
                        'model2': val2,
                        'difference': val2 - val1,
                        'improvement': ((val2 - val1) / val1 * 100) if val1 > 0 else 0.0
                    }
            
            comparison['overall'] = overall_comparison
        
        return comparison
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict[str, Any], 
                                 model_name: str = "Model") -> str:
        """Generate a comprehensive evaluation report."""
        
        report = []
        report.append(f"# Disease Prediction Model Evaluation Report: {model_name}")
        report.append("=" * 60)
        report.append("")
        
        # Overall metrics
        if 'overall' in evaluation_results:
            overall = evaluation_results['overall']
            report.append("## Overall Performance Metrics")
            report.append(f"- Total Predictions: {overall.get('total_predictions', 0):,}")
            report.append(f"- Total Ground Truth: {overall.get('total_ground_truth', 0):,}")
            report.append(f"- Code Coverage: {overall.get('code_coverage', 0):.3f}")
            report.append(f"- Average Confidence: {overall.get('avg_confidence', 0):.3f}")
            report.append(f"- Average Probability: {overall.get('avg_probability', 0):.3f}")
            report.append("")
        
        # Performance by K values
        report.append("## Performance by Top-K")
        for k in self.k_values:
            k_key = f'k_{k}'
            if k_key in evaluation_results:
                k_metrics = evaluation_results[k_key]
                report.append(f"### Top-{k} Metrics")
                report.append(f"- Accuracy@{k}: {k_metrics.get('accuracy_at_k', 0):.3f}")
                report.append(f"- MAP@{k}: {k_metrics.get('map_at_k', 0):.3f}")
                report.append(f"- Recall@{k}: {k_metrics.get('recall_at_k', 0):.3f}")
                report.append(f"- Hit Rate@{k}: {k_metrics.get('hit_rate_at_k', 0):.3f}")
                if 'ndcg_at_k' in k_metrics:
                    report.append(f"- NDCG@{k}: {k_metrics.get('ndcg_at_k', 0):.3f}")
                report.append("")
        
        # Category performance
        if 'detailed_analysis' in evaluation_results:
            detailed = evaluation_results['detailed_analysis']
            if 'category_performance' in detailed:
                report.append("## Performance by ICD Category")
                category_perf = detailed['category_performance']
                
                for category, metrics in sorted(category_perf.items()):
                    report.append(f"### Category {category}")
                    report.append(f"- Precision: {metrics.get('precision', 0):.3f}")
                    report.append(f"- Recall: {metrics.get('recall', 0):.3f}")
                    report.append(f"- F1 Score: {metrics.get('f1_score', 0):.3f}")
                    report.append(f"- Predictions: {metrics.get('predicted_count', 0)}")
                    report.append(f"- True Cases: {metrics.get('true_count', 0)}")
                    report.append("")
        
        return "\n".join(report)