README.md
# CreditWise: Comprehensive Credit Assessment Solution with Responsible AI

## Project Overview

CreditWise is an academic project demonstrating a responsible and transparent credit scoring system built with Python and machine learning. The application combines predictive modeling with explainability, fairness auditing, and regulatory compliance features to enable informed lending decisions while maintaining ethical standards.

## Table of Contents
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Architecture](#technical-architecture)
- [Model Training](#model-training)
- [Model Details](#model-details)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Versioning](#model-versioning)
- [Fairness & Ethics](#fairness--ethics)
- [Security & Privacy](#security--privacy)
- [Testing](#testing)
- [Deployment Considerations](#deployment-considerations)
- [Contributing](#contributing)
- [License](#license)
- [Support and Resources](#support-and-resources)

## Features

- **Real-time Credit Assessment**: Instant evaluation of credit applications using machine learning algorithms
- **Explainable AI**: Detailed decision explanations using SHAP values to highlight key factors influencing creditworthiness
- **Fairness Auditing**: Built-in bias detection and fairness metrics to ensure equitable treatment across demographic groups
- **Advanced Fairness Controls**: Threshold calibration and bias mitigation specifically designed to reduce discrimination across protected groups
- **Performance Dashboard**: Comprehensive analytics including accuracy, precision, recall, F1 score, and confusion matrix
- **Regulatory Compliance**: Tools and documentation supporting compliance with financial regulations
- **User-friendly Interface**: Intuitive Streamlit-based web application for loan officers and analysts
- **Model Versioning**: System for managing and tracking different model versions
- **Comprehensive Logging**: Detailed logging of all predictions and model decisions

## Directory Structure

```
├── assets
│   └── style.css
├── data
├── docs
├── logs
├── models
├── src
│   ├── __init__.py
│   ├── auth.py
│   ├── config.py
│   ├── database.py
│   ├── model_versioning.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── security.py
│   └── utils.py
├── tests
│   └── test_app.py
├── README.md
├── api.py
├── app.py
├── download_data.py
├── initialize_models.py
├── retrain_model.py
└── whole_project.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd credit_scoring_app
   ```

2. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Initialize the models
   ```bash
   py initialize_models.py
   ```

5. Run the application
   ```bash
   streamlit run app.py
   ```

## Usage

### Running the Application

1. Start the main application:
   ```bash
   streamlit run app.py
   ```

2. Access the application through your web browser at `http://localhost:8501`

### Application Tabs

The application features five main tabs:

#### 1. Applicant Assessment
- Input form for credit application data
- Real-time validation of input values
- Credit decision (Approved/Declined) with confidence metrics

#### 2. Decision Explanation
- SHAP-based feature importance visualization
- Bar charts showing positive/negative influences
- Plain-language explanation of key factors

#### 3. Model Integrity Report
- Fairness and bias audit results
- Demographic parity and equal opportunity metrics
- Advanced fairness controls and bias mitigation indicators
- Regulatory compliance indicators

#### 4. Performance Metrics Dashboard
- Accuracy, precision, recall, and F1-score metrics
- Confusion matrix and feature importance charts
- Model evaluation statistics

#### 5. Audit Log
- Recent predictions with timestamps
- Approval rate statistics
- Prediction distribution charts

## Technical Architecture

### Core Components

1. **Main Application** ([`app.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\app.py))
   - Streamlit-based web interface
   - Multi-tab interface for different functionalities
   - Session management for persistent state
   - Error handling and user feedback mechanisms

2. **REST API** ([`api.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\api.py))
   - Flask-based REST API with CORS support
   - Health check endpoint
   - Prediction endpoint for credit risk assessment
   - Model information endpoint
   - Performance metrics endpoint
   - Fairness metrics endpoint
   - Input validation endpoint
   - Authentication and security layers

3. **Data Preprocessing Module** ([`src/preprocess.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\src\preprocess.py))
   - Handles categorical encoding and numerical scaling
   - Implements validation checks for input data integrity
   - Manages feature engineering and selection processes
   - Defines feature groups matching the German Credit Dataset structure
   - Creates preprocessing pipeline with scalers and encoders
   - Validates input data against acceptable ranges and values
   - Converts user input to preprocessed arrays for prediction

4. **Prediction Engine** ([`src/predict.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\src\predict.py))
   - Implements the trained machine learning model
   - Provides probability estimates for default risk
   - Calculates confidence intervals for predictions
   - Generates risk prediction with probability and confidence
   - Implements advanced fairness controls including threshold calibration and bias mitigation
   - Implements equalized odds by adjusting decision thresholds per demographic group
   - Creates SHAP-based feature contribution explanations
   - Generates plain-language reason codes for the decision

5. **Utility Functions** ([`src/utils.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\src/utils.py))
   - Helper functions for calculations and data transformations
   - Fairness metrics computation
   - Model performance evaluation
   - Logging utilities for monitoring
   - Configure logging for the application
   - Compute demographic parity and equal opportunity differences
   - Calculate comprehensive model performance metrics
   - Log prediction events for monitoring
   - Detect simple data drift
   - Save/load model artifacts to/from disk
   - Check if required artifacts exist
   - Format probability into human-readable risk statement
   - Return key model card points for UI display

6. **Database Layer** ([`src/database.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\src\database.py))
   - SQLite database for storing predictions
   - Model performance tracking
   - Fairness audit results
   - Database tables:
     - `PredictionLog`: Stores prediction events for monitoring and analysis
     - `ModelPerformance`: Stores model performance metrics over time
     - `FairnessAudit`: Stores fairness audit results
   - Key functions:
     - `log_prediction_event()`: Log prediction events to the database
     - `save_model_performance()`: Save model performance metrics
     - `save_fairness_audit()`: Save fairness audit results
     - `get_recent_predictions()`: Retrieve recent prediction logs
     - `get_model_performance_history()`: Retrieve historical model performance

7. **Model Versioning System** ([`src/model_versioning.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\src\model_versioning.py))
   - Version management for models
   - Integrity verification with hash checks
   - Rollback capabilities
   - Classes:
     - `ModelVersion`: Represents a specific version of a model
     - `ModelVersionManager`: Manages different versions of models
   - Key features:
     - Version tracking with creation dates and performance metrics
     - Rollback capability to previous model versions
     - A/B testing support for running multiple models
     - Metadata storage for training conditions
     - File integrity verification using hashes

8. **Authentication Module** ([`src/auth.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\src\auth.py))
   - User authentication and authorization
   - Password hashing and verification
   - Session management
   - Role-based access control

9. **Security Module** ([`src/security.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\src\security.py))
   - Input sanitization
   - SQL injection prevention
   - Cross-site scripting (XSS) protection
   - Cross-site request forgery (CSRF) protection

10. **Configuration Module** ([`src/config.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\src\config.py))
    - Centralized configuration management for the application
    - Application-wide constants and settings
    - File paths for data, models, and artifacts
    - Model parameters and hyperparameters
    - Thresholds and limits for input validation
    - Logging configuration
    - Directory initialization

### Supporting Scripts

- **Download Data Script** ([`download_data.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\download_data.py)): Script to download the German Credit dataset from the UCI repository
- **Initialize Models Script** ([`initialize_models.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\initialize_models.py)): Initialization script to create model artifacts if they don't exist
- **Retrain Model Script** ([`retrain_model.py`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\retrain_model.py)): Model retraining script for automatic model updates with new data and performance monitoring

### Assets and Styling

- **CSS Styles** ([`assets/style.css`](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\assets\style.css)): Custom CSS styles for enhancing the appearance of the Streamlit application

## Model Training

The project includes automated model training capabilities:

- **Data Loading**: Automatically loads from `data/` directory
- **Preprocessing**: Applies standard transformations and encodings
- **Model Training**: Fits logistic regression with cross-validation
- **Fairness-Aware Training**: Incorporates bias mitigation during model training
- **Evaluation**: Computes performance metrics
- **Artifact Storage**: Saves model, preprocessor, and explainer

To retrain the model, run:
```bash
python initialize_models.py
```

## Model Details

- **Algorithm**: Logistic Regression with L2 regularization
- **Dataset**: German Credit Dataset (academic demonstration)
- **Features**: 20 financial and demographic attributes
- **Performance Metrics**: 
  - Accuracy ~76%
  - Precision ~71%
  - Recall ~76%
  - F1-Score ~73%
  - AUC-ROC ~0.82
- **Fairness Features**: Threshold calibration, bias mitigation, equalized odds
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical features
- **Validation**: Cross-validation with stratified sampling
- **Regularization**: L2 regularization to prevent overfitting

## Machine Learning Pipeline

### Model Type
- **Algorithm**: Logistic Regression with L2 regularization
- **Purpose**: Binary classification (creditworthy/not creditworthy)
- **Interpretability**: High - coefficients are directly interpretable
- **Fairness**: Implements equalized odds and bias mitigation

### Features
- **Total Count**: 20 financial and demographic attributes
- **Numerical**: Duration, credit amount, installment commitment, residence since, age, etc.
- **Categorical**: Employment status, credit history, purpose, savings status, employment, etc.

### Performance Metrics
- **Accuracy**: ~76%
- **Precision**: ~71%
- **Recall**: ~76%
- **F1-Score**: ~73%
- **AUC-ROC**: ~0.82

### Preprocessing
- **Numerical Features**: StandardScaler for normalization
- **Categorical Features**: OneHotEncoder for encoding
- **Pipeline**: Combined preprocessing steps

### Fairness Mechanisms
- **Threshold Calibration**: Group-specific thresholds to promote demographic parity
- **Bias Mitigation**: Adjustments to reduce discrimination across protected groups
- **Equalized Odds**: Adjusting decision thresholds per demographic group
- **Individual Fairness**: Considering similar individuals when making predictions

### Explainability
- **Method**: SHAP (SHapley Additive exPlanations)
- **Purpose**: Explain individual predictions by showing feature contributions
- **Output**: Visualizations showing which features pushed the decision toward approval or denial

## Model Versioning

The application includes a comprehensive model versioning system:

- **Version Tracking**: Each model iteration is tracked with metadata
- **Rollback Capability**: Ability to revert to previous model versions
- **Integrity Verification**: SHA256 hash verification of model files
- **Fairness Tracking**: Monitor fairness metrics across different model versions
- **Metadata Storage**: Training metrics and conditions stored with each version
- **A/B Testing Support**: Capability to run multiple model versions simultaneously

## Fairness & Ethics

### Protected Attributes
The system monitors for potential bias across:
- Age groups (categorized as young, middle-aged, elderly)
- Gender (derived from personal status)
- Foreign worker status

### Fairness Metrics
- **Demographic Parity**: Similar positive decision rates across groups
- **Equal Opportunity**: Similar true positive rates across groups
- **Disparate Impact**: Ratio of favorable outcomes between groups
- **Threshold Calibration**: Group-specific thresholds to reduce bias

### Fairness Mechanisms
- **Threshold Calibration**: Adjusting decision thresholds for different demographic groups
- **Bias Mitigation**: Post-processing techniques to reduce discrimination
- **Equalized Odds**: Ensuring similar true positive and false positive rates across groups
- **Individual Fairness**: Treating similar individuals similarly regardless of group membership

### Ethical Guidelines
- No direct use of protected attributes in the model
- Full explainability of decisions per GDPR Article 22
- Continuous bias monitoring and reporting
- Regular fairness audits
- Proactive bias detection and mitigation

## Security & Privacy

- **Input Validation**: All user inputs are validated against acceptable ranges/values
- **Privacy Protection**: No applicant data is stored persistently beyond session time
- **Error Handling**: Detailed error messages are sanitized for client display
- **Authentication**: Basic authentication layer available in [src/auth.py](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\src\auth.py)
- **Security Measures**: Input sanitization, XSS protection in [src/security.py](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\src\security.py)
- **Database Security**: Encrypted storage of prediction logs
- **Input Validation**: Comprehensive validation of all user inputs, range checks for numerical values, enum validation for categorical values, sanitization of user inputs
- **Privacy Protection**: No applicant data is stored persistently, session-based temporary storage, encrypted communication channels, anonymized data processing
- **Error Handling**: Sanitized error messages for clients, detailed logging for developers, graceful failure modes, secure fallback mechanisms

## Testing

The application includes both unit and integration tests in the [tests/](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\tests) directory:

- **Unit Tests**: Verify individual function behavior
- **Integration Tests**: Test component interactions
- **Performance Tests**: Evaluate prediction speed and accuracy
- **Fairness Tests**: Validate bias detection algorithms
- **Input Validation Tests**: Ensure proper validation of user inputs
- **Fairness-Specific Tests**: Validate threshold calibration and bias mitigation

**Test Classes**:
- `TestPreprocessing`: Tests preprocessing functions
- `TestPrediction`: Tests prediction functions including fairness controls
- `TestUtils`: Tests utility functions
- `TestDatabase`: Tests database functions

To run the full test suite:
```bash
pytest tests/
```

## Deployment Considerations

### Production Readiness
- Academic demonstration only - not for production use
- Requires validation with institution-specific data
- Regulatory approval processes needed
- Integration with bank's core systems required

### Scalability
- Single-user application currently
- Database backend supports concurrent access
- API endpoints ready for system integration
- Docker containerization possible

### Monitoring
- Comprehensive logging of all predictions
- Performance metrics tracking
- Drift detection for input distributions
- Bias monitoring and alerting
- Fairness metrics tracking over time

## Contributing

We welcome contributions to improve the CreditWise application:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 Python style guide
- Include docstrings for all functions and classes
- Write unit tests for new functionality
- Ensure code maintains backward compatibility
- Document any changes to the fairness or ethical aspects of the model

## License

This academic project is made available for educational and research purposes. For commercial use, please contact the development team.

## Support and Resources

For questions, suggestions, or support:

- **Documentation**: Check the detailed documentation in [whole_project.md](file:///c%3A/Users/Lenovo/Desktop/SCHOOL%2.2\PROJECT_credit_scoring_ml\credit_scoring_app\whole_project.md)
- **Issues**: Submit issues via GitHub for bugs or feature requests
- **Email**: Contact the development team at creditwise.academic.project@example.com