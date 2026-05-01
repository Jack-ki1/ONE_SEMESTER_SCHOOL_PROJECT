initialize_models.py
"""
Initialization script to create model artifacts if they don't exist.
This ensures the app runs smoothly without needing to trigger the training flow.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import shap

# Import our modules
from src.preprocess import create_preprocessor, get_feature_config
from src.predict import predict_risk, generate_shap_explanation, generate_reason_codes
from src.utils import get_model_card_summary, format_probability
from src.pickle_fix import load_with_compatibility  # Added import for compatibility fix


def initialize_models():
    """Initialize and save model artifacts."""
    print("Initializing model artifacts...")
    
    # Create models directory if it doesn't exist
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Check if artifacts already exist
    artifacts = {
        'model': model_dir / "credit_model.pkl",
        'preprocessor': model_dir / "preprocessor.pkl",
        'explainer': model_dir / "shap_explainer.pkl",
        'feature_names': model_dir / "feature_names.pkl"
    }
    
    if all(f.exists() for f in artifacts.values()):
        print("Model artifacts already exist. Skipping initialization.")
        return
    
    print("Creating model artifacts...")
    
    # Load data
    data_path = Path("data/german_credit_data.csv")
    if not data_path.exists():
        print("Data file not found! Please ensure 'data/german_credit_data.csv' exists.")
        return
    
    df = pd.read_csv(data_path)
    
    # Preprocessing
    num_feats, cat_feats = get_feature_config()
    feature_cols = num_feats + cat_feats
    
    X = df[feature_cols]
    y = (df['target'] == 1).astype(int)  # Using 'target' as per CSV header
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build pipeline
    preprocessor = create_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(
        C=1.0, 
        class_weight='balanced', 
        max_iter=1000, 
        random_state=42
    )
    model.fit(X_train_proc, y_train)
    
    # Evaluate model performance
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    y_pred = model.predict(X_train_proc)
    y_pred_proba = model.predict_proba(X_train_proc)[:, 1]
    
    train_accuracy = accuracy_score(y_train, y_pred)
    train_precision = precision_score(y_train, y_pred, zero_division=0)
    train_recall = recall_score(y_train, y_pred, zero_division=0)
    train_auc = roc_auc_score(y_train, y_pred_proba)
    
    print(f"Training completed. Accuracy: {train_accuracy:.2f}, Precision: {train_precision:.2f}, "
          f"Recall: {train_recall:.2f}, AUC: {train_auc:.2f}")
    
    # Create SHAP explainer
    background = shap.utils.sample(X_train_proc, min(50, len(X_train_proc)))
    explainer = shap.LinearExplainer(model, background)
    
    # Save artifacts
    with open(artifacts['model'], 'wb') as f: pickle.dump(model, f)
    with open(artifacts['preprocessor'], 'wb') as f: pickle.dump(preprocessor, f)
    with open(artifacts['explainer'], 'wb') as f: pickle.dump(explainer, f)
    with open(artifacts['feature_names'], 'wb') as f: 
        pickle.dump(preprocessor.get_feature_names_out(), f)
    
    print("Model artifacts created successfully!")


def load_models_with_compatibility():
    """Load models with compatibility fixes applied."""
    model_dir = Path("models")
    
    artifacts = {
        'model': model_dir / "credit_model.pkl",
        'preprocessor': model_dir / "preprocessor.pkl",
        'explainer': model_dir / "shap_explainer.pkl",
        'feature_names': model_dir / "feature_names.pkl"
    }
    
    # Load each artifact
    with open(artifacts['model'], 'rb') as f:
        model = pickle.load(f)
    
    # Use compatibility loader for preprocessor
    preprocessor = load_with_compatibility(artifacts['preprocessor'])
    
    with open(artifacts['explainer'], 'rb') as f:
        explainer = pickle.load(f)
    
    with open(artifacts['feature_names'], 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, preprocessor, explainer, feature_names


if __name__ == "__main__":
    initialize_models()