"""
Enhanced Flask application integrating the hybrid disease prediction system.
"""

from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import warnings
from datetime import datetime
import os
import io
from werkzeug.utils import secure_filename
import json
import tempfile
import threading
import webbrowser
import logging

warnings.filterwarnings('ignore')

# Import the new hybrid prediction system
from hybrid_prediction_system import HybridDiseasePredictionSystem

# Import the original calculator for backward compatibility
import sys
sys.path.append('.')

app = Flask(__name__)
app.secret_key = 'hybrid-disease-prediction-secret-key-2024'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

class DiseaseProbabilityCalculator:
    """Original calculator class for backward compatibility."""
    
    def __init__(self, csv_data=None):
        self.patient_data = csv_data
        self.patient_diseases = {}
        self.disease_patterns = defaultdict(list)
        self.probability_matrix = {}
        self.hcc_mapping = None
        self.load_hcc_mapping()
        
    def load_hcc_mapping(self):
        """Load HCC mapping from icd_hcc.csv file"""
        try:
            hcc_file_path = 'icd_hcc.csv'
            if os.path.exists(hcc_file_path):
                self.hcc_mapping = pd.read_csv(hcc_file_path)
                logger.info(f"HCC mapping loaded with {len(self.hcc_mapping)} records")
            else:
                logger.warning(f"HCC mapping file not found at {hcc_file_path}")
                self.hcc_mapping = None
        except Exception as e:
            logger.error(f"Error loading HCC mapping: {str(e)}")
            self.hcc_mapping = None
    
    def get_hcc_info(self, icd_code):
        """Get HCC information for a given ICD code"""
        if self.hcc_mapping is None:
            return {
                'description': 'No HCC mapping loaded',
                'HCC_ESRD_V24': 'N/A',
                'HCC_V24': 'N/A', 
                'HCC_V28': 'N/A'
            }
        
        try:
            matching_rows = self.hcc_mapping[self.hcc_mapping['ICDCode'] == icd_code]
            
            if matching_rows.empty:
                return {
                    'description': 'Code not found in HCC mapping',
                    'HCC_ESRD_V24': 'Not Found',
                    'HCC_V24': 'Not Found',
                    'HCC_V28': 'Not Found'
                }
            
            row = matching_rows.iloc[0]
            
            # Extract HCC information
            hcc_info = {
                'description': str(row.get('Description', 'N/A')),
                'HCC_ESRD_V24': str(row['HCC_ESRD_V24']) if pd.notna(row.get('HCC_ESRD_V24')) else 'N/A',
                'HCC_V24': str(row['HCC_V24']) if pd.notna(row.get('HCC_V24')) else 'N/A',
                'HCC_V28': str(row['HCC_V28']) if pd.notna(row.get('HCC_V28')) else 'N/A'
            }
            
            return hcc_info
            
        except Exception as e:
            logger.error(f"Error in get_hcc_info for code {icd_code}: {str(e)}")
            return {
                'description': f'Error: {str(e)}',
                'HCC_ESRD_V24': 'Error',
                'HCC_V24': 'Error',
                'HCC_V28': 'Error'
            }
    
    def load_data(self):
        """Load and process the patient data"""
        try:
            required_columns = ['MemberID', 'ICDCode']
            if not all(col in self.patient_data.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            self.patient_data = self.patient_data.dropna()
            self.patient_diseases = self.patient_data.groupby('MemberID')['ICDCode'].apply(set).to_dict()
            
            return {
                'success': True,
                'total_patients': len(self.patient_diseases),
                'total_records': len(self.patient_data),
                'unique_codes': len(self.patient_data['ICDCode'].unique()),
                'hcc_mapping_loaded': self.hcc_mapping is not None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_available_codes(self):
        """Get all available ICD codes"""
        if self.patient_data is not None:
            return sorted(self.patient_data['ICDCode'].unique().tolist())
        return []
    
    def analyze_input_codes(self, input_codes):
        """Analyze input codes using legacy method"""
        # Simplified version of the original analysis
        try:
            available_codes = set(self.get_available_codes())
            valid_codes = [code for code in input_codes if code in available_codes]
            
            if not valid_codes:
                return {'success': False, 'error': 'No valid codes found'}
            
            # Simple co-occurrence analysis
            results = []
            for input_code in valid_codes:
                patients_with_code = self.patient_data[self.patient_data['ICDCode'] == input_code]['MemberID'].unique()
                
                if len(patients_with_code) == 0:
                    continue
                
                # Find other codes for these patients
                other_codes = self.patient_data[
                    (self.patient_data['MemberID'].isin(patients_with_code)) & 
                    (self.patient_data['ICDCode'] != input_code)
                ]['ICDCode'].value_counts()
                
                for other_code, count in other_codes.head(10).items():
                    probability = count / len(patients_with_code)
                    hcc_info = self.get_hcc_info(other_code)
                    
                    results.append({
                        'trigger_code': input_code,
                        'predicted_code': other_code,
                        'probability': probability,
                        'supporting_patients': count,
                        'total_patients': len(patients_with_code),
                        'hcc_info': hcc_info
                    })
            
            # Sort by probability
            results.sort(key=lambda x: x['probability'], reverse=True)
            
            return {
                'success': True,
                'results': results[:20],  # Top 20 results
                'valid_codes': valid_codes,
                'analysis_type': 'legacy'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Global variables
calculator = None
hybrid_system = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global calculator, hybrid_system
    
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
            
            # Initialize original calculator for backward compatibility
            calculator = DiseaseProbabilityCalculator(df)
            result = calculator.load_data()
            
            # Initialize hybrid system
            try:
                hybrid_system = HybridDiseasePredictionSystem()
                hybrid_system.load_patient_data(df)
                
                # Try to train the model (with limited samples for demo)
                training_result = hybrid_system.train_model(df, max_samples=1000)
                hybrid_available = training_result['success']
                
                logger.info(f"Hybrid system initialized: {hybrid_available}")
                
            except Exception as e:
                logger.warning(f"Hybrid system initialization failed: {str(e)}")
                hybrid_system = None
                hybrid_available = False
            
            if result['success']:
                available_codes = calculator.get_available_codes()
                system_status = hybrid_system.get_system_status() if hybrid_system else {}
                
                return render_template('analysis.html', 
                                     data_loaded=True,
                                     total_patients=result['total_patients'],
                                     total_records=result['total_records'],
                                     unique_codes=result['unique_codes'],
                                     available_codes=available_codes,
                                     hcc_mapping_loaded=result['hcc_mapping_loaded'],
                                     hybrid_available=hybrid_available,
                                     system_status=system_status)
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
    global calculator, hybrid_system
    
    if calculator is None:
        return jsonify({'success': False, 'error': 'No data loaded. Please upload a CSV file first.'})
    
    try:
        # Get input parameters
        input_codes = request.form.get('codes', '').strip()
        age = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        method = request.form.get('method', 'hybrid')  # 'legacy' or 'hybrid'
        
        if not input_codes:
            return jsonify({'success': False, 'error': 'Please enter at least one ICD code.'})
        
        # Parse input codes
        codes = [code.strip() for code in input_codes.split(',')]
        codes = [code for code in codes if code]
        
        # Parse demographics
        age_int = None
        if age:
            try:
                age_int = int(age)
                if age_int < 0 or age_int > 120:
                    return jsonify({'success': False, 'error': 'Age must be between 0 and 120'})
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid age format'})
        
        gender_str = None
        if gender and gender.upper() in ['M', 'F', 'MALE', 'FEMALE']:
            gender_str = gender.upper()[0]  # Convert to M or F
        
        # Choose analysis method
        if method == 'hybrid' and hybrid_system is not None:
            # Use new hybrid system
            try:
                result = hybrid_system.predict_diseases(
                    input_icds=codes,
                    age=age_int,
                    gender=gender_str,
                    top_k=20,
                    include_explanations=True
                )
                
                if result['success']:
                    # Format results for frontend
                    formatted_predictions = []
                    for pred in result['predictions']:
                        formatted_pred = {
                            'predicted_code': pred['code'],
                            'description': pred['description'],
                            'probability': pred['probability'],
                            'confidence_score': pred.get('confidence_score', pred['probability']),
                            'rank': pred.get('rank', 0),
                            'type': pred.get('type', 'ICD'),
                            'rule_explanations': pred.get('rule_explanations', []),
                            'hcc_mappings': pred.get('hcc_mappings', []),
                            'cms_risk_score': pred.get('cms_risk_score', 0)
                        }
                        formatted_predictions.append(formatted_pred)
                    
                    return jsonify({
                        'success': True,
                        'results': formatted_predictions,
                        'valid_codes': result['input_icds'],
                        'invalid_codes': result['invalid_icds'],
                        'analysis_type': 'hybrid',
                        'model_info': result['model_info'],
                        'summary': result.get('summary', {}),
                        'demographics': {
                            'age': age_int,
                            'gender': gender_str
                        }
                    })
                else:
                    # Fall back to legacy method
                    result = calculator.analyze_input_codes(codes)
                    result['analysis_type'] = 'legacy_fallback'
                    result['fallback_reason'] = 'Hybrid prediction failed'
                    return jsonify(result)
                    
            except Exception as e:
                logger.error(f"Hybrid analysis error: {str(e)}")
                # Fall back to legacy method
                result = calculator.analyze_input_codes(codes)
                result['analysis_type'] = 'legacy_fallback'
                result['fallback_reason'] = f'Hybrid error: {str(e)}'
                return jsonify(result)
        else:
            # Use legacy method
            result = calculator.analyze_input_codes(codes)
            return jsonify(result)
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/hybrid_predict', methods=['POST'])
def hybrid_predict():
    """Dedicated endpoint for hybrid predictions."""
    global hybrid_system
    
    if hybrid_system is None:
        return jsonify({'success': False, 'error': 'Hybrid system not available'})
    
    try:
        data = request.get_json()
        
        input_icds = data.get('input_icds', [])
        age = data.get('age')
        gender = data.get('gender')
        top_k = data.get('top_k', 15)
        include_explanations = data.get('include_explanations', True)
        
        # Validate input
        validation = hybrid_system.validate_input(input_icds, age, gender)
        if not validation['valid']:
            return jsonify({
                'success': False,
                'error': '; '.join(validation['errors']),
                'warnings': validation['warnings']
            })
        
        # Make prediction
        result = hybrid_system.predict_diseases(
            input_icds=input_icds,
            age=age,
            gender=gender,
            top_k=top_k,
            include_explanations=include_explanations
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/system_status')
def system_status():
    """Get status of both systems."""
    global calculator, hybrid_system
    
    status = {
        'legacy_system': {
            'available': calculator is not None,
            'data_loaded': calculator is not None and calculator.patient_data is not None,
            'total_patients': len(calculator.patient_diseases) if calculator and calculator.patient_diseases else 0,
            'hcc_mapping': calculator.hcc_mapping is not None if calculator else False
        },
        'hybrid_system': {
            'available': hybrid_system is not None,
            'status': hybrid_system.get_system_status() if hybrid_system else {}
        }
    }
    
    return jsonify(status)

@app.route('/get_codes')
def get_codes():
    """Get available ICD codes."""
    global calculator
    
    try:
        if calculator:
            codes = calculator.get_available_codes()
            return jsonify({'success': True, 'codes': codes})
        else:
            return jsonify({'success': False, 'error': 'No data loaded'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export_excel', methods=['POST'])
def export_excel():
    """Export results to Excel (legacy functionality)."""
    global calculator
    
    if calculator is None:
        return jsonify({'success': False, 'error': 'No data loaded'})
    
    try:
        # Get the last analysis results from session or request
        results_data = request.get_json()
        
        if not results_data or 'results' not in results_data:
            return jsonify({'success': False, 'error': 'No results to export'})
        
        # Create DataFrame from results
        df = pd.DataFrame(results_data['results'])
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Disease_Predictions', index=False)
            
            # Add metadata sheet
            metadata = pd.DataFrame([
                ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Input Codes', ', '.join(results_data.get('valid_codes', []))],
                ['Analysis Type', results_data.get('analysis_type', 'Unknown')],
                ['Total Results', len(df)]
            ], columns=['Metric', 'Value'])
            
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        output.seek(0)
        
        return send_file(
            output,
            as_attachment=True,
            download_name=f'disease_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(host='127.0.0.1', port=5000, use_reloader=False, debug=True)