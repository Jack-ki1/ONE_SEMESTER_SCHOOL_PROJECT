"""
API module for the credit scoring application.

This module provides REST API endpoints for system integration,
allowing external applications to access the credit scoring functionality.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import check_password_hash
import pandas as pd
import numpy as np
import pickle
import traceback
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add the src directory to the path to import modules
sys.path.append(str(Path(__file__).parent / "src"))

from src.preprocess import validate_input, transform_single_input
from src.predict import predict_risk, format_probability
from src.utils import calculate_fairness_metrics, log_prediction
from src.database import log_prediction_event, save_model_performance
from src.config import Config
from src.auth import auth_manager, get_current_user
from src.model_versioning import version_manager
from src.pickle_fix import load_with_compatibility  # Added import for compatibility fix

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'Credit Scoring API'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to get credit risk prediction for an applicant."""
    try:
        # Extract JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'checking_status', 'duration', 'credit_history', 'purpose',
            'credit_amount', 'savings_status', 'employment', 'installment_commitment',
            'personal_status', 'other_parties', 'residence_since', 'property_magnitude',
            'age', 'other_payment_plans', 'housing', 'existing_credits',
            'job', 'num_dependents', 'own_telephone', 'foreign_worker'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate inputs
        errors = validate_input(data)
        if errors:
            return jsonify({'error': 'Input validation failed', 'details': errors}), 400
        
        # Load model artifacts
        try:
            model, preprocessor, explainer, feature_names = version_manager.load_version(
                version_manager.current_version
            ) if version_manager.current_version else (
                Config.MODEL_PATH,
                Config.PREPROCESSOR_PATH,
                Config.EXPLAINER_PATH,
                Config.FEATURE_NAMES_PATH
            )
            
            # If we got strings instead of loaded objects, load the actual artifacts
            if isinstance(model, str):
                with open(Config.MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                # Use the compatibility loader for the preprocessor
                preprocessor = load_with_compatibility(Config.PREPROCESSOR_PATH)
                with open(Config.EXPLAINER_PATH, 'rb') as f:
                    explainer = pickle.load(f)
                with open(Config.FEATURE_NAMES_PATH, 'rb') as f:
                    feature_names = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model artifacts: {str(e)}")
            return jsonify({'error': 'Failed to load model artifacts'}), 500
        
        # Transform input for prediction
        try:
            preprocessed = transform_single_input(data, preprocessor, Config.DEFAULT_FEATURES)
        except Exception as e:
            logger.error(f"Error transforming input: {str(e)}")
            return jsonify({'error': 'Error processing input data'}), 400
        
        # Make prediction
        prob_default, decision, confidence = predict_risk(model, preprocessed, original_features=data)
        risk_statement = format_probability(prob_default)
        
        # Calculate fairness metrics
        try:
            fairness_metrics = calculate_fairness_metrics(
                y_true=[0],  # placeholder
                y_pred=[int(decision == "DECLINED")],  # convert decision to binary
                sensitive_attr="age_group",  # example attribute
                attr_value=1 if data['age'] < 35 else 0  # young vs older
            )
        except Exception as e:
            logger.warning(f"Fairness metrics calculation failed: {str(e)}")
            fairness_metrics = {
                'demographic_parity': 0.0,
                'equal_opportunity': 0.0
            }
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'decision': decision,
                'confidence': confidence,
                'probability_of_default': float(prob_default),
                'risk_assessment': risk_statement
            },
            'fairness_metrics': fairness_metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Log the prediction
        try:
            log_prediction(data, {
                'decision': decision,
                'prob_default': prob_default,
                'confidence': confidence
            })
            
            # Also save to database
            explanation_text = f"Decision: {decision}, Probability: {prob_default:.2%}"
            log_prediction_event(
                applicant_features=data,
                prediction=decision,
                default_probability=float(prob_default),
                confidence=float(confidence),
                explanation=explanation_text
            )
        except Exception as log_error:
            logger.warning(f"Logging error: {str(log_error)}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Endpoint to get information about the current model."""
    try:
        # Get model version info
        version_info = None
        if version_manager.current_version:
            version_info = version_manager.get_version_details(version_manager.current_version)
        
        response = {
            'current_model_version': version_manager.current_version,
            'model_algorithm': 'Logistic Regression',
            'dataset_used': 'German Credit Dataset',
            'features_count': 20,
            'expected_input_fields': Config.DEFAULT_FEATURES,
            'version_details': version_info,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in model info endpoint: {str(e)}")
        return jsonify({'error': 'Failed to retrieve model info'}), 500


@app.route('/models', methods=['GET'])
def list_models():
    """Endpoint to list all available model versions."""
    try:
        versions = version_manager.list_versions()
        response = {
            'available_versions': versions,
            'current_active_version': version_manager.current_version,
            'count': len(versions),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in list models endpoint: {str(e)}")
        return jsonify({'error': 'Failed to retrieve model list'}), 500


@app.route('/performance', methods=['GET'])
def get_performance():
    """Endpoint to get model performance metrics."""
    try:
        # For now, return static metrics based on training
        # In a real implementation, this would fetch from the database
        response = {
            'accuracy': 0.76,
            'precision': 0.71,
            'recall': 0.76,
            'f1_score': 0.73,
            'auc_roc': 0.82,
            'timestamp': datetime.utcnow().isoformat(),
            'note': 'These are baseline metrics. Actual metrics should come from validation on production data.'
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in performance endpoint: {str(e)}")
        return jsonify({'error': 'Failed to retrieve performance metrics'}), 500


@app.route('/fairness', methods=['GET'])
def get_fairness():
    """Endpoint to get overall fairness metrics."""
    try:
        # For now, return example metrics
        # In a real implementation, this would aggregate fairness metrics from the database
        response = {
            'demographic_parity_overall': 0.05,
            'equal_opportunity_overall': 0.03,
            'protected_attributes_monitored': ['age_group', 'foreign_worker'],
            'sample_size': 1000,
            'last_evaluation_date': datetime.utcnow().isoformat(),
            'note': 'These are example metrics. Actual metrics should come from aggregated evaluations.'
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in fairness endpoint: {str(e)}")
        return jsonify({'error': 'Failed to retrieve fairness metrics'}), 500


@app.route('/validate-input', methods=['POST'])
def validate_input_api():
    """Endpoint to validate input without making a prediction."""
    try:
        data = request.get_json()
        
        # Validate inputs
        errors = validate_input(data)
        
        response = {
            'valid': len(errors) == 0,
            'errors': errors,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in validate input endpoint: {str(e)}")
        return jsonify({'error': 'Failed to validate input'}), 500


if __name__ == '__main__':
    # Check if model artifacts exist, if not initialize them
    if not (Config.MODEL_PATH.exists() and 
            Config.PREPROCESSOR_PATH.exists() and 
            Config.EXPLAINER_PATH.exists() and 
            Config.FEATURE_NAMES_PATH.exists()):
        print("Model artifacts not found. Please run initialize_models.py first.")
        print("Exiting...")
        exit(1)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)