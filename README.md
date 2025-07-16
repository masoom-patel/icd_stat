# Hybrid ICD/HCC Disease Prediction System

A comprehensive AI-powered disease prediction system that combines statistical analysis, machine learning, medical embeddings, and rule-based clinical logic to predict the most probable future diseases for patients based on their current ICD patterns and demographics.

## 🌟 Features

### Core Capabilities
- **Hybrid Prediction Models**: Combines LightGBM, MLP, and rule-based systems
- **Medical Text Embeddings**: Uses sentence transformers for semantic similarity
- **Demographics Integration**: Age and gender-aware predictions
- **Clinical Rules Engine**: Domain-specific medical rules and comorbidity patterns
- **HCC Risk Scoring**: CMS risk score integration
- **Top-K Predictions**: Ranked predictions with confidence scores
- **Comprehensive Evaluation**: Multiple metrics including MAP@K, NDCG@K, accuracy

### Technical Features
- **Data Leakage Prevention**: Strict patient-level splits and validation
- **Modular Architecture**: Pluggable components for easy extension
- **Fallback Systems**: TF-IDF embeddings when transformer models unavailable
- **Real-time Predictions**: Fast inference with caching
- **Web Interface**: Enhanced Flask app with both legacy and hybrid systems

## 🏗️ System Architecture

```
├── models/
│   ├── embedder.py           # ICDEmbedder - Medical text embeddings
│   ├── patient_vector.py     # PatientVectorBuilder - Demographics + ICD vectors
│   ├── candidate_generator.py # CandidateGenerator - ICD/HCC filtering
│   ├── feature_builder.py    # FeatureBuilder - Comprehensive features
│   ├── rule_engine.py        # RuleTriggerEngine - Medical rules
│   ├── scorer.py             # Scorer - ML models (LightGBM/MLP)
│   ├── ranker.py             # Ranker - Top-K ranking with confidence
│   └── evaluator.py          # Evaluator - Comprehensive metrics
├── config/
│   └── model_config.py       # Configuration management
├── utils/
│   ├── data_loader.py        # Data loading and validation
│   └── preprocessing.py      # Data preprocessing and augmentation
├── data/
│   └── sample_data/          # Sample datasets for testing
├── templates/
│   ├── index.html            # Original upload interface
│   ├── analysis.html         # Original analysis interface
│   └── analysis_enhanced.html # Enhanced hybrid interface
├── app.py                    # Original Flask application
├── app_enhanced.py           # Enhanced Flask with hybrid system
└── hybrid_prediction_system.py # Main system integration
```

## 🚀 Quick Start

### Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Prepare Data**
```bash
# Use provided sample data or prepare your own
python data/sample_data/create_sample_data.py
```

3. **Run the Application**
```bash
# Enhanced hybrid system
python app_enhanced.py

# Or original system
python app.py
```

### Basic Usage

#### Web Interface
1. Open http://127.0.0.1:5000
2. Upload CSV file with patient data (MemberID, ICDCode, Age, Gender)
3. Choose prediction method (Hybrid AI or Legacy Statistical)
4. Enter ICD codes and demographics
5. View ranked predictions with confidence scores

#### Python API
```python
from hybrid_prediction_system import HybridDiseasePredictionSystem
import pandas as pd

# Load patient data
patient_data = pd.read_csv('data/sample_data/sample_patient_data.csv')

# Initialize system
hybrid_system = HybridDiseasePredictionSystem()
hybrid_system.load_patient_data(patient_data)

# Make prediction
result = hybrid_system.predict_diseases(
    input_icds=['I10', 'E11.9'],  # Hypertension, Diabetes
    age=65,
    gender='M',
    top_k=10,
    include_explanations=True
)

print(f"Predictions: {len(result['predictions'])}")
for pred in result['predictions'][:5]:
    print(f"- {pred['code']}: {pred['description']} (prob: {pred['probability']:.3f})")
```

## 📊 Data Format

### Patient Data CSV
```csv
MemberID,ICDCode,Age,Gender
PAT_000001,I10,65,M
PAT_000001,E11.9,65,M
PAT_000002,J44.1,72,F
```

### ICD Master Data CSV
```csv
ICDCode,Description,HCC_ESRD_V24,HCC_V24,HCC_V28
I10,Essential hypertension,,,
E11.9,Type 2 diabetes mellitus without complications,19,19,38
J44.1,Chronic obstructive pulmonary disease with acute exacerbation,111,111,276
```

## 🧠 Model Components

### 1. ICDEmbedder
- **Purpose**: Convert ICD descriptions to vector embeddings
- **Models**: SentenceTransformers, Clinical BERT, TF-IDF fallback
- **Features**: Semantic similarity, caching, medical domain adaptation

### 2. PatientVectorBuilder  
- **Purpose**: Build comprehensive patient representations
- **Features**: 
  - Embedding aggregation (mean, attention, max)
  - Demographics encoding (age bins, gender)
  - Pattern features (entropy, diversity)
  - ICD category distribution

### 3. CandidateGenerator
- **Purpose**: Generate and filter candidate diseases
- **Features**:
  - Frequency-based filtering
  - Co-occurrence analysis
  - HCC mapping
  - Demographic appropriateness

### 4. FeatureBuilder
- **Purpose**: Create ML-ready feature vectors
- **Feature Types**:
  - `cosine_similarity`: Semantic similarity scores
  - `conditional_probability`: Co-occurrence probabilities
  - `cms_risk_score`: HCC risk scores
  - `demographics`: Age and gender features
  - `statistical`: Counts, entropy, variance
  - `rule_triggers`: Medical rule activations

### 5. RuleTriggerEngine
- **Purpose**: Apply medical domain knowledge
- **Rules**: 
  - Diabetes complications
  - Cardiovascular progression  
  - Respiratory comorbidities
  - Renal disease progression
  - Cancer complications
  - Age/gender-specific conditions

### 6. Scorer
- **Purpose**: ML models for probability scoring
- **Models**:
  - **LightGBM**: Primary model for speed and interpretability
  - **MLP**: Neural network for complex patterns
  - **Ensemble**: Weighted combination

### 7. Ranker
- **Purpose**: Rank candidates and provide explanations
- **Features**:
  - Confidence estimation
  - Rule explanations
  - Feature importance
  - Demographic filtering

## 📈 Evaluation

### Metrics
- **Accuracy@K**: Fraction of patients with ≥1 correct prediction in top-K
- **MAP@K**: Mean Average Precision at K
- **Recall@K**: Average recall across patients
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate**: Binary hit rate for top predictions

### Cross-Validation
- Patient-level splits (no data leakage)
- Leave-one-out validation
- Stratified sampling for balanced evaluation

## ⚙️ Configuration

### Model Configuration
```python
from config.model_config import update_config

update_config(
    primary_model='lightgbm',
    top_k_predictions=15,
    confidence_threshold=0.1,
    negative_sampling_ratio=5,
    min_frequency_threshold=10
)
```

### Available Settings
- **Models**: `lightgbm`, `mlp`, `ensemble`
- **Embeddings**: Various sentence-transformer models
- **Aggregation**: `mean`, `attention`, `max`
- **Thresholds**: Confidence, frequency, demographic filters

## 🔒 Data Security

### Privacy Protection
- No PHI storage in models
- Patient ID anonymization
- Secure data handling practices
- HIPAA-compliant architecture ready

### Data Leakage Prevention
- Strict patient-level train/test splits
- Target exclusion from input features
- Temporal validation support
- Cross-patient validation only

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Test specific modules
python -m pytest tests/test_embedder.py
python -m pytest tests/test_rules.py
```

### Sample Data Testing
```bash
# Generate sample data
python data/sample_data/create_sample_data.py

# Test with sample data
python -c "
from hybrid_prediction_system import HybridDiseasePredictionSystem
import pandas as pd
df = pd.read_csv('data/sample_data/sample_patient_data.csv')
system = HybridDiseasePredictionSystem()
system.load_patient_data(df)
result = system.predict_diseases(['E11.9', 'I10'], age=65, gender='M')
print(f'Success: {result[\"success\"]}, Predictions: {len(result[\"predictions\"])}')
"
```

## 🔧 Advanced Usage

### Custom Rules
```python
from models.rule_engine import RuleTriggerEngine

rule_engine = RuleTriggerEngine()
rule_engine.add_custom_rule(
    rule_name='custom_diabetes_rule',
    trigger_codes=['E11.*'],
    target_patterns=['H36.*', 'N08.*'],
    description='Diabetes leading to retinopathy and nephropathy',
    confidence=0.8
)
```

### Model Training
```python
# Train with custom data
training_result = hybrid_system.train_model(
    patient_data=df,
    max_samples=5000,
    validation_split=0.2
)

print(f"Training AUC: {training_result['metrics']['auc']:.3f}")
```

### Batch Predictions
```python
# Process multiple patients
batch_requests = [
    {
        'input_icds': ['I10', 'E11.9'],
        'age': 65,
        'gender': 'M'
    },
    {
        'input_icds': ['J44.1', 'I50.9'],
        'age': 72,
        'gender': 'F'
    }
]

results = hybrid_system.ranker.batch_rank_candidates(batch_requests)
```

## 📝 API Reference

### Core Methods

#### HybridDiseasePredictionSystem
- `predict_diseases(input_icds, age, gender, top_k)`: Main prediction method
- `train_model(patient_data, max_samples)`: Train ML models
- `get_system_status()`: Check component status
- `validate_input(input_icds, age, gender)`: Input validation

#### Web API Endpoints
- `POST /upload`: Upload patient data CSV
- `POST /analyze`: Analyze with method selection
- `POST /hybrid_predict`: Direct hybrid predictions
- `GET /system_status`: System component status
- `GET /get_codes`: Available ICD codes

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Install development dependencies
4. Run tests before committing
5. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Add type hints for new functions
- Include docstrings for public methods
- Maintain test coverage >80%

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on medical coding standards (ICD-10-CM, HCC)
- Inspired by clinical decision support systems
- Uses state-of-the-art NLP and ML techniques
- Designed for healthcare interoperability

## 📞 Support

For questions, issues, or contributions:
- Create GitHub issues for bugs
- Use discussions for questions
- Follow contribution guidelines
- Check documentation first

---

**Disclaimer**: This system is for research and educational purposes. Not intended for direct clinical use without proper validation and regulatory approval.