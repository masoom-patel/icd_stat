from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations, chain
import warnings
from datetime import datetime
import os
import io
from werkzeug.utils import secure_filename
import json
import tempfile
warnings.filterwarnings('ignore')
import threading
import webbrowser


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")
class DiseaseProbabilityCalculator:
    def __init__(self, csv_data=None):
        """
        Initialize the calculator with patient data
        
        Args:
            csv_data (DataFrame): DataFrame with columns MemberID, ICDCode
        """
        self.patient_data = csv_data
        self.patient_diseases = {}
        self.disease_patterns = defaultdict(list)
        self.probability_matrix = {}
        self.hcc_mapping = None
        self.load_hcc_mapping()
        
# Add this debugging code to your load_hcc_mapping method
    def load_hcc_mapping(self):
        """Load HCC mapping from icd_hcc.csv file"""
        try:
            hcc_file_path = 'icd_hcc.csv'
            if os.path.exists(hcc_file_path):
                self.hcc_mapping = pd.read_csv(hcc_file_path)
                print(f"HCC mapping loaded successfully with {len(self.hcc_mapping)} records")
                print(f"HCC mapping columns: {list(self.hcc_mapping.columns)}")
                
                # Debug: Print actual column names with their exact characters
                print("Exact column names:")
                for i, col in enumerate(self.hcc_mapping.columns):
                    print(f"  {i}: '{col}' (length: {len(col)})")
                
                # Check if we can find a sample ICD code
                if 'ICDCode' in self.hcc_mapping.columns:
                    sample_codes = self.hcc_mapping['ICDCode'].head(3).tolist()
                    print(f"Sample ICD codes: {sample_codes}")
                
            else:
                print(f"Warning: HCC mapping file not found at {hcc_file_path}")
                self.hcc_mapping = None
        except Exception as e:
            print(f"Error loading HCC mapping: {str(e)}")
            self.hcc_mapping = None
    

    def get_hcc_info(self, icd_code):
        """Get HCC information for a given ICD code"""
        if self.hcc_mapping is None:
            print(f"HCC mapping is None for code: {icd_code}")
            return {
                'description': 'No HCC mapping loaded',
                'HCC_ESRD_V24': 'N/A',
                'HCC_V24': 'N/A', 
                'HCC_V28': 'N/A'
            }
        
        try:
            # Debug: Print what we're looking for
            print(f"Looking for ICD code: '{icd_code}'")
            
            # Find matching row for the ICD code
            matching_rows = self.hcc_mapping[self.hcc_mapping['ICDCode'] == icd_code]
            
            print(f"Found {len(matching_rows)} matching rows")
            
            if matching_rows.empty:
                # Debug: Try to find similar codes
                similar_codes = self.hcc_mapping[self.hcc_mapping['ICDCode'].str.contains(icd_code, na=False)]
                print(f"Similar codes found: {len(similar_codes)}")
                if len(similar_codes) > 0:
                    print(f"Similar codes: {similar_codes['ICDCode'].head(3).tolist()}")
                
                return {
                    'description': 'Code not found in HCC mapping',
                    'HCC_ESRD_V24': 'Not Found',
                    'HCC_V24': 'Not Found',
                    'HCC_V28': 'Not Found'
                }
            
            row = matching_rows.iloc[0]
            print(f"Found row: {row.to_dict()}")
            
            # Try different possible column names for description
            description_value = 'N/A'
            possible_desc_columns = ['Description', 'description', 'DESC', 'desc', 'DESCRIPTION', 'ICD_Description', 'icd_description']
            
            print(f"Available columns: {list(self.hcc_mapping.columns)}")
            
            for col in possible_desc_columns:
                if col in self.hcc_mapping.columns:
                    desc_val = row.get(col)
                    print(f"Column '{col}' value: '{desc_val}' (type: {type(desc_val)})")
                    if pd.notna(desc_val) and str(desc_val).strip():
                        description_value = str(desc_val).strip()
                        print(f"Using description from column '{col}': '{description_value}'")
                        break
            
            # Extract HCC information using exact column names
            hcc_info = {
                'description': description_value,
                'HCC_ESRD_V24': str(row['HCC_ESRD_V24']) if pd.notna(row.get('HCC_ESRD_V24')) else 'N/A',
                'HCC_V24': str(row['HCC_V24']) if pd.notna(row.get('HCC_V24')) else 'N/A',
                'HCC_V28': str(row['HCC_V28']) if pd.notna(row.get('HCC_V28')) else 'N/A'
            }
            
            print(f"Final HCC info: {hcc_info}")
            return hcc_info
            
        except Exception as e:
            print(f"Error in get_hcc_info for code {icd_code}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'description': f'Error: {str(e)}',
                'HCC_ESRD_V24': 'Error',
                'HCC_V24': 'Error',
                'HCC_V28': 'Error'
            }        

        
    def load_data(self):
        """Load and process the patient data"""
        try:
            # Check if required columns exist
            required_columns = ['MemberID', 'ICDCode']
            if not all(col in self.patient_data.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Clean the data
            self.patient_data = self.patient_data.dropna()
            
            # Group diseases by patient
            self.patient_diseases = self.patient_data.groupby('MemberID')['ICDCode'].apply(set).to_dict()
            
            return {
                'success': True,
                'total_patients': len(self.patient_diseases),
                'total_records': len(self.patient_data),
                'unique_codes': len(self.patient_data['ICDCode'].unique()),
                'hcc_mapping_loaded': self.hcc_mapping is not None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_available_codes(self):
        """Get all available ICD codes"""
        if self.patient_data is not None:
            return sorted(self.patient_data['ICDCode'].unique().tolist())
        return []
    
    def generate_patterns_for_codes(self, input_codes):
        """Generate patterns based on specific input codes - ONLY patterns with 2+ diseases"""
        input_codes_set = set(input_codes)
        
        # Clear previous patterns
        self.disease_patterns = defaultdict(list)
        
        # Generate all possible combinations of input codes as patterns
        patterns_to_analyze = []
        
        # Generate combinations of different lengths from input codes (minimum 2 diseases)
        for r in range(2, len(input_codes) + 1):  # Changed from range(1, ...) to range(2, ...)
            for combo in combinations(input_codes, r):
                patterns_to_analyze.append(tuple(sorted(combo)))
        
        # For each pattern, find patients who have ALL diseases in that pattern
        for pattern in patterns_to_analyze:
            pattern_set = set(pattern)
            
            # Find all patients who have this exact pattern (or superset)
            matching_patients = []
            for patient_id, patient_diseases in self.patient_diseases.items():
                if pattern_set.issubset(patient_diseases):
                    matching_patients.append({
                        'patient_id': patient_id,
                        'all_diseases': patient_diseases,
                        'additional_diseases': patient_diseases - pattern_set
                    })
            
            if matching_patients:
                self.disease_patterns[pattern] = matching_patients
        
        return len(self.disease_patterns)
    
    def calculate_probabilities_for_patterns(self, input_codes):
        """Calculate probability of diseases occurring given the generated patterns with filtering"""
        all_diseases = set(self.patient_data['ICDCode'].unique())
        input_codes_set = set(input_codes)
        
        results = {}
        
        for pattern, patients_with_pattern in self.disease_patterns.items():
            if len(patients_with_pattern) < 1:
                continue
                
            pattern_results = {}
            total_patients_with_pattern = len(patients_with_pattern)
            
            # For each possible disease (excluding those in the pattern)
            for target_disease in all_diseases:
                if target_disease in pattern:
                    continue
                    
                # Count patients with this pattern who also have the target disease
                patients_with_target = sum(1 for patient in patients_with_pattern 
                                         if target_disease in patient['all_diseases'])
                
                # FILTER: Only include if patients_with_target >= 4
                if patients_with_target >= 4:
                    probability = patients_with_target / total_patients_with_pattern
                    confidence = self._calculate_confidence(patients_with_target, total_patients_with_pattern)
                    
                    # FILTER: Only include high confidence predictions
                    if confidence in ['High', 'Medium']:  # Only High and Medium confidence
                        # Get HCC information for the target disease
                        hcc_info = self.get_hcc_info(target_disease)
                        
                        pattern_results[target_disease] = {
                            'probability': probability,
                            'patients_with_disease': patients_with_target,
                            'total_patients_with_pattern': total_patients_with_pattern,
                            'confidence': confidence,
                            'hcc_info': hcc_info
                        }
            
            if pattern_results:
                results[pattern] = pattern_results
        
        return results
    
    def _deduplicate_predictions(self, all_predictions):
        """Remove duplicate disease predictions, keeping the one with highest confidence"""
        # Group predictions by disease
        disease_predictions = defaultdict(list)
        for pred in all_predictions:
            disease_predictions[pred['predicts']].append(pred)
        
        # For each disease, keep only the best prediction
        deduplicated_predictions = []
        confidence_order = {'High': 3, 'Medium': 2, 'Low': 1, 'Very Low': 0}
        
        for disease, predictions in disease_predictions.items():
            # Sort by confidence (descending), then by probability (descending), then by evidence count (descending)
            best_prediction = max(predictions, key=lambda x: (
                confidence_order.get(x['confidence'], 0),
                x['probability'],
                x['patients_with_disease']
            ))
            deduplicated_predictions.append(best_prediction)
        
        return deduplicated_predictions

    def analyze_input_codes(self, input_codes):
        """Complete analysis for specific input codes with enhanced filtering and deduplication"""
        # Check if we have enough codes for meaningful patterns
        if len(input_codes) < 2:
            return {
                'success': False,
                'error': 'Need at least 2 ICD codes to generate meaningful patterns (minimum 2 diseases per pattern).'
            }
        
        # Generate patterns for the input codes (minimum 2 diseases per pattern)
        num_patterns = self.generate_patterns_for_codes(input_codes)
        
        if num_patterns == 0:
            return {
                'success': False,
                'error': 'No patterns with 2+ diseases found for these ICD codes in the dataset.'
            }
        
        # Calculate probabilities with filtering
        probability_results = self.calculate_probabilities_for_patterns(input_codes)
        
        if not probability_results:
            return {
                'success': False,
                'error': 'No high-confidence probability calculations possible with minimum 4 patients per prediction.'
            }
        
        # Process results for web display
        all_predictions = []
        
        for pattern, disease_probs in probability_results.items():
            for disease, prob_info in disease_probs.items():
                all_predictions.append({
                    'pattern': ' + '.join(pattern),
                    'pattern_tuple': pattern,
                    'predicts': disease,
                    'probability': prob_info['probability'],
                    'evidence': f"{prob_info['patients_with_disease']}/{prob_info['total_patients_with_pattern']}",
                    'confidence': prob_info['confidence'],
                    'patients_with_disease': prob_info['patients_with_disease'],
                    'total_patients_with_pattern': prob_info['total_patients_with_pattern'],
                    'pattern_size': len(pattern),
                    'hcc_esrd_v24': prob_info['hcc_info']['HCC_ESRD_V24'],
                    'hcc_v24': prob_info['hcc_info']['HCC_V24'],
                    'hcc_v28': prob_info['hcc_info']['HCC_V28'],
                    'description': prob_info['hcc_info']['description']  # ← THIS WAS THE MISSING LINE!
                })
        
        # DEDUPLICATION: Remove duplicate disease predictions
        deduplicated_predictions = self._deduplicate_predictions(all_predictions)
        
        # Sort by probability
        deduplicated_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Group by pattern (using deduplicated predictions)
        pattern_groups = {}
        for pred in deduplicated_predictions:
            pattern = pred['pattern_tuple']
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(pred)
        
        # Calculate summary statistics using deduplicated predictions
        disease_max_probs = {}
        for pred in deduplicated_predictions:
            disease = pred['predicts']
            disease_max_probs[disease] = pred['probability']  # Since we deduplicated, each disease appears only once
        
        sorted_diseases = sorted(disease_max_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Enhanced summary with filtering and deduplication information
        summary = {
            'total_patterns': len(pattern_groups),
            'total_outcomes': len(deduplicated_predictions),
            'total_outcomes_before_dedup': len(all_predictions),
            'duplicates_removed': len(all_predictions) - len(deduplicated_predictions),
            'unique_diseases_predicted': len(set(pred['predicts'] for pred in deduplicated_predictions)),
            'highest_probability': max(pred['probability'] for pred in deduplicated_predictions) if deduplicated_predictions else 0,
            'average_probability': np.mean([pred['probability'] for pred in deduplicated_predictions]) if deduplicated_predictions else 0,
            'top_diseases': sorted_diseases[:10],
            'min_pattern_size': 2,
            'min_patients_per_prediction': 4,
            'confidence_levels_included': ['High', 'Medium'],
            'total_high_confidence': sum(1 for pred in deduplicated_predictions if pred['confidence'] == 'High'),
            'total_medium_confidence': sum(1 for pred in deduplicated_predictions if pred['confidence'] == 'Medium'),
            'hcc_mapping_available': self.hcc_mapping is not None,
            'deduplication_applied': True
        }
        
        return {
            'success': True,
            'input_codes': input_codes,
            'all_predictions': deduplicated_predictions,
            'pattern_groups': {' + '.join(k): v for k, v in pattern_groups.items()},
            'summary': summary,
            'filtering_applied': {
                'min_pattern_size': 2,
                'min_patients_with_disease': 4,
                'confidence_filter': 'High and Medium only',
                'deduplication': 'Each disease predicted only once (highest confidence kept)'
            }
        }
    
    def export_results_to_excel(self, results, input_codes):
        """Export detailed results to Excel with multiple sheets including HCC information"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"disease_probability_analysis_with_hcc_{timestamp}.xlsx"
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        
        # Create Excel writer
        with pd.ExcelWriter(temp_file.name, engine='openpyxl') as writer:
            
            # Sheet 1: Summary Report
            self.create_summary_sheet(writer, results, input_codes)
            
            # Sheet 2: Detailed Probability Results with HCC
            self.create_probability_sheet(writer, results, input_codes)
            
            # Sheet 3: Patient Details with Member IDs
            self.create_patient_details_sheet(writer, results, input_codes)
            
            # Sheet 4: Disease Pairings Analysis
            self.create_disease_pairings_sheet(writer, results, input_codes)
            
            # Sheet 5: HCC Analysis
            self.create_hcc_analysis_sheet(writer, results, input_codes)
            
            # Sheet 6: Raw Data Summary
            self.create_raw_data_sheet(writer)
        
        return temp_file.name, filename

    def create_raw_data_sheet(self, writer):
        """Create raw data summary sheet - this method was missing"""
        if self.patient_data is not None:
            # Create a summary of the raw data
            summary_data = []
            summary_data.append(['Total Records', len(self.patient_data)])
            summary_data.append(['Unique Patients', len(self.patient_diseases)])
            summary_data.append(['Unique ICD Codes', len(self.patient_data['ICDCode'].unique())])
            summary_data.append(['', ''])
            summary_data.append(['Top 20 Most Common ICD Codes', 'Count'])
            
            # Get top ICD codes
            top_codes = self.patient_data['ICDCode'].value_counts().head(20)
            for code, count in top_codes.items():
                summary_data.append([code, count])
            
            # Create DataFrame and save
            raw_data_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            raw_data_df.to_excel(writer, sheet_name='Raw_Data_Summary', index=False)
    
    def create_summary_sheet(self, writer, results, input_codes):
        """Create summary sheet with high-level analysis including filtering and deduplication info"""
        summary_data = []
        
        # Add input information
        summary_data.append(['Analysis Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        summary_data.append(['Input ICD Codes', ', '.join(input_codes)])
        summary_data.append(['Total Patients in Dataset', len(self.patient_diseases)])
        summary_data.append(['Total Unique ICD Codes', len(set(self.patient_data['ICDCode']))])
        summary_data.append(['Patterns Analyzed', len(results)])
        summary_data.append(['HCC Mapping Available', 'Yes' if self.hcc_mapping is not None else 'No'])
        summary_data.append(['', ''])
        
        # Add filtering information
        summary_data.append(['FILTERING CRITERIA APPLIED', ''])
        summary_data.append(['Minimum Pattern Size', '2 diseases'])
        summary_data.append(['Minimum Patients per Prediction', '4 patients'])
        summary_data.append(['Confidence Levels Included', 'High and Medium only'])
        summary_data.append(['Deduplication Applied', 'Yes - Each disease predicted only once'])
        summary_data.append(['', ''])
        
        # Collect all predictions for summary
        all_predictions = []
        for pattern, disease_probs in results.items():
            for disease, prob_info in disease_probs.items():
                all_predictions.append({
                    'pattern': ' + '.join(pattern),
                    'pattern_size': len(pattern),
                    'disease': disease,
                    'probability': prob_info['probability'],
                    'evidence': f"{prob_info['patients_with_disease']}/{prob_info['total_patients_with_pattern']}",
                    'confidence': prob_info['confidence'],
                    'patients_with_disease': prob_info['patients_with_disease']
                })
        
        # Apply deduplication to summary data as well
        deduplicated_predictions = self._deduplicate_predictions([{
            'predicts': pred['disease'],
            'probability': pred['probability'],
            'confidence': pred['confidence'],
            'patients_with_disease': pred['patients_with_disease'],
            'pattern_size': pred['pattern_size']
        } for pred in all_predictions])
        
        if deduplicated_predictions:
            summary_data.append(['Total Outcomes (Before Deduplication)', len(all_predictions)])
            summary_data.append(['Total Outcomes (After Deduplication)', len(deduplicated_predictions)])
            summary_data.append(['Duplicates Removed', len(all_predictions) - len(deduplicated_predictions)])
            summary_data.append(['Unique Diseases Predicted', len(set(pred['predicts'] for pred in deduplicated_predictions))])
            summary_data.append(['Highest Probability', f"{max(pred['probability'] for pred in deduplicated_predictions):.3f}"])
            summary_data.append(['Average Probability', f"{np.mean([pred['probability'] for pred in deduplicated_predictions]):.3f}"])
            summary_data.append(['High Confidence Predictions', len([p for p in deduplicated_predictions if p['confidence'] == 'High'])])
            summary_data.append(['Medium Confidence Predictions', len([p for p in deduplicated_predictions if p['confidence'] == 'Medium'])])
            summary_data.append(['Average Pattern Size', f"{np.mean([pred['pattern_size'] for pred in deduplicated_predictions]):.1f}"])
            summary_data.append(['', ''])
            
            # Top 10 most likely diseases (already deduplicated)
            summary_data.append(['TOP 10 MOST LIKELY DISEASES (DEDUPLICATED)', ''])
            disease_probs = [(pred['predicts'], pred['probability']) for pred in deduplicated_predictions]
            sorted_diseases = sorted(disease_probs, key=lambda x: x[1], reverse=True)
            
            for i, (disease, prob) in enumerate(sorted_diseases[:10], 1):
                summary_data.append([f"{i}. {disease}", f"{prob:.3f}"])
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    def create_probability_sheet(self, writer, results, input_codes):
        """Create detailed probability results sheet with HCC information and ICD descriptions"""
        prob_data = []
        
        for pattern, disease_probs in results.items():
            for disease, prob_info in disease_probs.items():
                prob_data.append({
                    'Pattern': ' + '.join(pattern),
                    'Pattern_Size': len(pattern),
                    'Predicted_Disease': disease,
                    'ICD_Description': prob_info['hcc_info']['description'],  # Added ICD description
                    'Probability': prob_info['probability'],
                    'Patients_With_Disease': prob_info['patients_with_disease'],
                    'Total_Patients_With_Pattern': prob_info['total_patients_with_pattern'],
                    'Evidence_Ratio': f"{prob_info['patients_with_disease']}/{prob_info['total_patients_with_pattern']}",
                    'Confidence_Level': prob_info['confidence'],
                    'CMS_HCC_ESRD_V24': prob_info['hcc_info']['HCC_ESRD_V24'],
                    'CMS_HCC_V24': prob_info['hcc_info']['HCC_V24'],
                    'CMS_HCC_V28': prob_info['hcc_info']['HCC_V28'],
                    'Meets_Min_Criteria': 'Yes'
                })
        
        if prob_data:
            # Apply deduplication to Excel data
            deduplicated_prob_data = self._deduplicate_excel_data(prob_data)
            
            prob_df = pd.DataFrame(deduplicated_prob_data)
            prob_df = prob_df.sort_values('Probability', ascending=False)
            
            # Add a note about deduplication
            prob_df.loc[len(prob_df)] = {
                'Pattern': 'NOTE: Deduplication applied - each disease appears only once',
                'Pattern_Size': '',
                'Predicted_Disease': 'with highest confidence prediction kept',
                'ICD_Description': '',
                'Probability': '',
                'Patients_With_Disease': '',
                'Total_Patients_With_Pattern': '',
                'Evidence_Ratio': '',
                'Confidence_Level': '',
                'CMS_HCC_ESRD_V24': '',
                'CMS_HCC_V24': '',
                'CMS_HCC_V28': '',
                'Meets_Min_Criteria': ''
            }
            
            prob_df.to_excel(writer, sheet_name='Probability_Results_HCC', index=False)
    
    def _deduplicate_excel_data(self, prob_data):
        """Deduplicate Excel data by disease - Updated to preserve ICD_Description"""
        # Group by disease
        disease_groups = defaultdict(list)
        for item in prob_data:
            disease_groups[item['Predicted_Disease']].append(item)
        
        # Keep best prediction for each disease
        deduplicated_data = []
        confidence_order = {'High': 3, 'Medium': 2, 'Low': 1, 'Very Low': 0}
        
        for disease, items in disease_groups.items():
            best_item = max(items, key=lambda x: (
                confidence_order.get(x['Confidence_Level'], 0),
                x['Probability'],
                x['Patients_With_Disease']
            ))
            deduplicated_data.append(best_item)
        
        return deduplicated_data
    
    def create_hcc_analysis_sheet(self, writer, results, input_codes):
        """Create HCC-specific analysis sheet with ICD descriptions"""
        hcc_data = []
        
        # Collect HCC information for analysis
        for pattern, disease_probs in results.items():
            for disease, prob_info in disease_probs.items():
                hcc_info = prob_info['hcc_info']
                hcc_data.append({
                    'Pattern': ' + '.join(pattern),
                    'Predicted_Disease': disease,
                    'ICD_Description': hcc_info['description'],  # Added ICD description
                    'Probability': prob_info['probability'],
                    'CMS_HCC_ESRD_V24': hcc_info['HCC_ESRD_V24'],
                    'CMS_HCC_V24': hcc_info['HCC_V24'],
                    'CMS_HCC_V28': hcc_info['HCC_V28'],
                    'Confidence': prob_info['confidence'],
                    'Evidence_Count': prob_info['patients_with_disease']
                })
        
        if hcc_data:
            # Apply deduplication
            deduplicated_hcc_data = self._deduplicate_excel_data([{
                'Pattern': item['Pattern'],
                'Predicted_Disease': item['Predicted_Disease'],
                'ICD_Description': item['ICD_Description'],  # Include in deduplication
                'Probability': item['Probability'],
                'CMS_HCC_ESRD_V24': item['CMS_HCC_ESRD_V24'],
                'CMS_HCC_V24': item['CMS_HCC_V24'],
                'CMS_HCC_V28': item['CMS_HCC_V28'],
                'Confidence_Level': item['Confidence'],
                'Patients_With_Disease': item['Evidence_Count']
            } for item in hcc_data])
            
            # Convert back to original format
            final_hcc_data = []
            for item in deduplicated_hcc_data:
                final_hcc_data.append({
                    'Pattern': item['Pattern'],
                    'Predicted_Disease': item['Predicted_Disease'],
                    'ICD_Description': item['ICD_Description'],  # Include description
                    'Probability': item['Probability'],
                    'CMS_HCC_ESRD_V24': item['CMS_HCC_ESRD_V24'],
                    'CMS_HCC_V24': item['CMS_HCC_V24'],
                    'CMS_HCC_V28': item['CMS_HCC_V28'],
                    'Confidence': item['Confidence_Level'],
                    'Evidence_Count': item['Patients_With_Disease']
                })
            
            hcc_df = pd.DataFrame(final_hcc_data)
            
            # Sort by HCC categories and probability
            hcc_df = hcc_df.sort_values(['CMS_HCC_V28', 'CMS_HCC_V24', 'Probability'], 
                                    ascending=[True, True, False])
            hcc_df.to_excel(writer, sheet_name='HCC_Analysis', index=False)
            
            # Create HCC summary statistics using deduplicated data
            hcc_summary = []
            hcc_summary.append(['HCC CATEGORY SUMMARY (DEDUPLICATED)', ''])
            
            # Count by HCC V28 categories
            v28_counts = Counter([item['CMS_HCC_V28'] for item in final_hcc_data if item['CMS_HCC_V28'] != 'N/A'])
            hcc_summary.append(['', ''])
            hcc_summary.append(['CMS-HCC V28 Categories', 'Count'])
            for category, count in v28_counts.most_common():
                hcc_summary.append([category, count])
            
            # Count by HCC V24 categories
            v24_counts = Counter([item['CMS_HCC_V24'] for item in final_hcc_data if item['CMS_HCC_V24'] != 'N/A'])
            hcc_summary.append(['', ''])
            hcc_summary.append(['CMS-HCC V24 Categories', 'Count'])
            for category, count in v24_counts.most_common():
                hcc_summary.append([category, count])
            
            # Save HCC summary
            hcc_summary_df = pd.DataFrame(hcc_summary, columns=['Category', 'Value'])
            hcc_summary_df.to_excel(writer, sheet_name='HCC_Summary', index=False)

    
    def create_patient_details_sheet(self, writer, results, input_codes):
        """Create detailed patient information with Member IDs"""
        patient_details = []
        
        for pattern, disease_probs in results.items():
            pattern_str = ' + '.join(pattern)
            
            # Get patients with this pattern
            patients_with_pattern = self.disease_patterns[pattern]
            
            for patient_info in patients_with_pattern:
                patient_id = patient_info['patient_id']
                all_diseases = list(patient_info['all_diseases'])
                additional_diseases = list(patient_info['additional_diseases'])
                
                patient_details.append({
                    'Pattern': pattern_str,
                    'Pattern_Size': len(pattern),
                    'Member_ID': patient_id,
                    'All_Diseases': ', '.join(sorted(all_diseases)),
                    'Additional_Diseases': ', '.join(sorted(additional_diseases)) if additional_diseases else 'None',
                    'Total_Diseases': len(all_diseases),
                    'Pattern_Diseases': ', '.join(sorted(pattern)),
                    'Additional_Disease_Count': len(additional_diseases)
                })
        
        if patient_details:
            patient_df = pd.DataFrame(patient_details)
            patient_df = patient_df.sort_values(['Pattern_Size', 'Pattern', 'Member_ID'])
            patient_df.to_excel(writer, sheet_name='Patient_Details', index=False)
    
    def create_disease_pairings_sheet(self, writer, results, input_codes):
        """Create disease pairings analysis sheet with ICD descriptions"""
        pairings_data = []
        
        # Analyze co-occurrence patterns
        for pattern, disease_probs in results.items():
            pattern_str = ' + '.join(pattern)
            patients_with_pattern = self.disease_patterns[pattern]
            
            # Count co-occurrences of diseases
            disease_combinations = defaultdict(int)
            
            for patient_info in patients_with_pattern:
                additional_diseases = list(patient_info['additional_diseases'])
                
                # Count individual diseases
                for disease in additional_diseases:
                    disease_combinations[disease] += 1
                
                # Count disease pairs
                if len(additional_diseases) > 1:
                    for pair in combinations(sorted(additional_diseases), 2):
                        pair_str = ' + '.join(pair)
                        disease_combinations[pair_str] += 1
            
            # Add to pairings data
            total_patients = len(patients_with_pattern)
            for disease_combo, count in disease_combinations.items():
                if count >= 4:  # Apply same minimum criteria
                    # Get HCC and description information for single diseases or pairs
                    hcc_info = {}
                    descriptions = []
                    combo_parts = disease_combo.split(' + ')
                    
                    if len(combo_parts) == 1:
                        # Single disease
                        hcc_info = self.get_hcc_info(combo_parts[0])
                        descriptions.append(hcc_info['description'])
                    else:
                        # For pairs, get HCC info for each disease
                        hcc_info = {
                            'HCC_ESRD_V24': [],
                            'HCC_V24': [],
                            'HCC_V28': []
                        }
                        for disease in combo_parts:
                            disease_hcc = self.get_hcc_info(disease)
                            descriptions.append(disease_hcc['description'])
                            for key in hcc_info:
                                hcc_info[key].append(disease_hcc[key])
                        # Join HCC info for pairs
                        hcc_info = {key: ' & '.join([str(val) for val in hcc_info[key]]) for key in hcc_info}
                    
                    # Join descriptions
                    combined_description = ' & '.join(descriptions)
                    
                    confidence = self._calculate_confidence(count, total_patients)
                    if confidence in ['High', 'Medium']:  # Only include High/Medium confidence
                        pairings_data.append({
                            'Base_Pattern': pattern_str,
                            'Base_Pattern_Size': len(pattern),
                            'Disease_Combination': disease_combo,
                            'Disease_Description': combined_description,  # Added descriptions
                            'Occurrence_Count': count,
                            'Total_Patients_With_Pattern': total_patients,
                            'Co_occurrence_Rate': count / total_patients if total_patients > 0 else 0,
                            'Confidence_Level': confidence,
                            'Is_Single_Disease': '+' not in disease_combo,
                            'Combination_Size': len(combo_parts),
                            'CMS_HCC_ESRD_V24': hcc_info['HCC_ESRD_V24'],
                            'CMS_HCC_V24': hcc_info['HCC_V24'],
                            'CMS_HCC_V28': hcc_info['HCC_V28'],
                            'Meets_Min_Criteria': 'Yes'
                        })
        
        if pairings_data:
            # Apply deduplication to disease combinations
            deduplicated_pairings = self._deduplicate_pairings_data(pairings_data)
            
            pairings_df = pd.DataFrame(deduplicated_pairings)
            pairings_df = pairings_df.sort_values(['Base_Pattern_Size', 'Base_Pattern', 'Co_occurrence_Rate'], ascending=[True, True, False])
            
            # Add a note about deduplication
            pairings_df.loc[len(pairings_df)] = {
                'Base_Pattern': 'NOTE: Deduplication applied - each disease combination appears only once',
                'Base_Pattern_Size': '',
                'Disease_Combination': 'with highest confidence prediction kept',
                'Disease_Description': '',
                'Occurrence_Count': '',
                'Total_Patients_With_Pattern': '',
                'Co_occurrence_Rate': '',
                'Confidence_Level': '',
                'Is_Single_Disease': '',
                'Combination_Size': '',
                'CMS_HCC_ESRD_V24': '',
                'CMS_HCC_V24': '',
                'CMS_HCC_V28': '',
                'Meets_Min_Criteria': ''
            }
            
            pairings_df.to_excel(writer, sheet_name='Disease_Pairings', index=False)

    def _deduplicate_pairings_data(self, pairings_data):
        """Deduplicate disease pairings data by Disease_Combination"""
        # Group by disease combination
        combo_groups = defaultdict(list)
        for item in pairings_data:
            combo_groups[item['Disease_Combination']].append(item)
        
        # Keep best prediction for each disease combination
        deduplicated_data = []
        confidence_order = {'High': 3, 'Medium': 2, 'Low': 1, 'Very Low': 0}
        
        for combo, items in combo_groups.items():
            best_item = max(items, key=lambda x: (
                confidence_order.get(x['Confidence_Level'], 0),
                x['Co_occurrence_Rate'],
                x['Occurrence_Count']
            ))
            deduplicated_data.append(best_item)
        
        return deduplicated_data
    def _calculate_confidence(self, successes, total):
        """Calculate confidence level based on sample size - Updated thresholds"""
        if total < 4:
            return "Very Low"  # This will be filtered out
        elif total < 6:
            return "Low"       # This will be filtered out
        elif total < 10:
            return "Medium"    # This will be included
        else:
            return "High"      # This will be included
# Global variable to store the calculator instance
calculator = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global calculator
    
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and file.filename.endswith('.csv'):
        try:
            # Read CSV file
            df = pd.read_csv(file)
            
            # Initialize calculator
            calculator = DiseaseProbabilityCalculator(df)
            
            # Load data
            result = calculator.load_data()
            
            if result['success']:
                available_codes = calculator.get_available_codes()
                return render_template('analysis.html', 
                                     data_loaded=True,
                                     total_patients=result['total_patients'],
                                     total_records=result['total_records'],
                                     unique_codes=result['unique_codes'],
                                     available_codes=available_codes,
                                     hcc_mapping_loaded=result['hcc_mapping_loaded'])
            else:
                flash(f'Error processing file: {result["error"]}')
                return redirect(url_for('index'))
                
        except Exception as e:
            flash(f'Error reading file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Please upload a CSV file')
        return redirect(url_for('index'))

@app.route('/analyze', methods=['POST'])
def analyze():
    global calculator
    
    if calculator is None:
        return jsonify({'success': False, 'error': 'No data loaded. Please upload a CSV file first.'})
    
    try:
        # Get input codes from form
        input_codes = request.form.get('codes', '').strip()
        
        if not input_codes:
            return jsonify({'success': False, 'error': 'Please enter at least one ICD code.'})
        
        # Parse input codes
        codes = [code.strip() for code in input_codes.split(',')]
        codes = [code for code in codes if code]  # Remove empty strings
        
        # Validate codes
        available_codes = set(calculator.get_available_codes())
        valid_codes = [code for code in codes if code in available_codes]
        invalid_codes = [code for code in codes if code not in available_codes]
        
        if not valid_codes:
            return jsonify({'success': False, 'error': 'No valid ICD codes found in the dataset.'})
        
        # Analyze patterns
        results = calculator.analyze_input_codes(valid_codes)
        
        if results['success']:
            results['invalid_codes'] = invalid_codes
            return jsonify(results)
        else:
            return jsonify(results)
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export_excel', methods=['POST'])
def export_excel():
    global calculator
    
    if calculator is None:
        return jsonify({'success': False, 'error': 'No data loaded.'})
    
    try:
        # Get input codes from form
        input_codes = request.form.get('codes', '').strip()
        
        if not input_codes:
            return jsonify({'success': False, 'error': 'Please enter at least one ICD code.'})
        
        # Parse input codes
        codes = [code.strip() for code in input_codes.split(',')]
        codes = [code for code in codes if code]
        
        # Validate codes
        available_codes = set(calculator.get_available_codes())
        valid_codes = [code for code in codes if code in available_codes]
        
        if not valid_codes:
            return jsonify({'success': False, 'error': 'No valid ICD codes found.'})
        
        # Generate patterns for the input codes (this populates calculator.disease_patterns)
        num_patterns = calculator.generate_patterns_for_codes(valid_codes)
        
        if num_patterns == 0:
            return jsonify({'success': False, 'error': 'No patterns found for these ICD codes.'})
        
        # Calculate probabilities with filtering
        probability_results = calculator.calculate_probabilities_for_patterns(valid_codes)
        
        if not probability_results:
            return jsonify({'success': False, 'error': 'No high-confidence probability calculations possible.'})
        
        # Export to Excel using the raw probability_results (with tuple keys)
        temp_file_path, filename = calculator.export_results_to_excel(probability_results, valid_codes)
        
        return send_file(temp_file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_codes')
def get_codes():
    global calculator
    
    if calculator is None:
        return jsonify({'success': False, 'error': 'No data loaded.'})
    
    try:
        codes = calculator.get_available_codes()
        return jsonify({'success': True, 'codes': codes})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(host='127.0.0.1', port=5000, use_reloader=False)

