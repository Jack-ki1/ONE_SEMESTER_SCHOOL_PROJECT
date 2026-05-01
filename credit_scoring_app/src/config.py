"""
Configuration module for the credit scoring application.

This module provides centralized configuration management for the application,
including paths, model parameters, and system settings.
"""

import os
from pathlib import Path


class Config:
    """Application configuration class."""
    
    # Base directory of the application
    BASE_DIR = Path(__file__).parent.parent
    
    # Data paths
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    
    # Model artifact paths
    MODEL_PATH = MODELS_DIR / "credit_model.pkl"
    PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
    EXPLAINER_PATH = MODELS_DIR / "shap_explainer.pkl"
    FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
    
    # Dataset path
    DATASET_PATH = DATA_DIR / "german_credit_data.csv"
    
    # Default feature names
    DEFAULT_FEATURES = [
        'checking_status', 'duration', 'credit_history', 'purpose',
        'credit_amount', 'savings_status', 'employment', 'installment_commitment',
        'personal_status', 'other_parties', 'residence_since', 'property_magnitude',
        'age', 'other_payment_plans', 'housing', 'existing_credits',
        'job', 'num_dependents', 'own_telephone', 'foreign_worker'
    ]
    
    # Model parameters
    MODEL_PARAMS = {
        'random_state': 42,
        'max_iter': 1000,
        'C': 0.1  # Regularization parameter
    }
    
    # Thresholds and limits
    DEFAULT_PROBABILITY_THRESHOLD = 0.35  # Changed to match predict.py
    MAX_CREDIT_AMOUNT = 100000  # Maximum credit amount in EUR
    MIN_AGE = 18
    MAX_AGE = 75
    
    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = BASE_DIR / "logs" / "application.log"
    
    # Create directories if they don't exist
    @classmethod
    def initialize_dirs(cls):
        """Initialize required directories."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


# Initialize directories when module is loaded
Config.initialize_dirs()