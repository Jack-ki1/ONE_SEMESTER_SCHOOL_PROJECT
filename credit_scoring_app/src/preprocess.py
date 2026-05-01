"""
Preprocessing module for credit application data.
Ensures consistent transformation between training and inference.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_feature_config():
    """Define feature groups matching German Credit Dataset structure."""
    numerical_features = [
        'duration', 'credit_amount', 'installment_commitment', 
        'residence_since', 'age', 'existing_credits', 'num_dependents'
    ]
    categorical_features = [
        'checking_status', 'credit_history', 'purpose', 'savings_status',
        'employment', 'personal_status', 'other_parties', 'property_magnitude',
        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'
    ]
    return numerical_features, categorical_features

def create_preprocessor():
    """Build preprocessing pipeline with scaling and encoding."""
    num_features, cat_features = get_feature_config()
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='drop'
    )

def validate_input(input_dict):
    """Validate user inputs against expected ranges and types."""
    errors = []
    
    # Numerical validations
    if not (18 <= input_dict['age'] <= 75):
        errors.append("Age must be between 18-75 years")
    if input_dict['credit_amount'] <= 0:
        errors.append("Credit amount must be positive")
    if not (1 <= input_dict['duration'] <= 72):
        errors.append("Loan duration must be 1-72 months")
    if input_dict['installment_commitment'] < 1 or input_dict['installment_commitment'] > 10:
        errors.append("Installment commitment must be between 1-10")
    if input_dict['residence_since'] < 0:
        errors.append("Residence since cannot be negative")
    if input_dict['existing_credits'] < 0 or input_dict['existing_credits'] > 5:
        errors.append("Existing credits must be between 0-5")
    if input_dict['num_dependents'] < 0 or input_dict['num_dependents'] > 5:
        errors.append("Number of dependents must be between 0-5")
    
    # Categorical validations (ensure values match training categories)
    valid_checking_status = ["<0", "0<=X<200", ">=200", "no checking"]
    if input_dict['checking_status'] not in valid_checking_status:
        errors.append(f"Checking status must be one of: {valid_checking_status}")
    
    valid_credit_history = ["no credits", "all paid", "existing paid", "critical", "delayed"]
    if input_dict['credit_history'] not in valid_credit_history:
        errors.append(f"Credit history must be one of: {valid_credit_history}")
    
    valid_purpose = ["radio/tv", "education", "furniture", "car new", "car used", 
                     "business", "repairs", "other", "retraining", "domestic appliance"]
    if input_dict['purpose'] not in valid_purpose:
        errors.append(f"Purpose must be one of: {valid_purpose}")
    
    valid_savings_status = ["<100", "100<=X<500", "500<=X<1000", ">=1000", "unknown"]
    if input_dict['savings_status'] not in valid_savings_status:
        errors.append(f"Savings status must be one of: {valid_savings_status}")
    
    valid_employment = ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]
    if input_dict['employment'] not in valid_employment:
        errors.append(f"Employment duration must be one of: {valid_employment}")
    
    valid_personal_status = ["male single", "female div/dep/mar", "male div/sep", "male mar/wid"]
    if input_dict['personal_status'] not in valid_personal_status:
        errors.append(f"Personal status must be one of: {valid_personal_status}")
    
    valid_other_parties = ["none", "co-applicant", "guarantor"]
    if input_dict['other_parties'] not in valid_other_parties:
        errors.append(f"Other parties must be one of: {valid_other_parties}")
    
    valid_property_magnitude = ["real estate", "life insurance", "car", "no property"]
    if input_dict['property_magnitude'] not in valid_property_magnitude:
        errors.append(f"Property magnitude must be one of: {valid_property_magnitude}")
    
    valid_other_payment_plans = ["none", "bank", "stores"]
    if input_dict['other_payment_plans'] not in valid_other_payment_plans:
        errors.append(f"Other payment plans must be one of: {valid_other_payment_plans}")
    
    valid_housing = ["rent", "own", "for free"]
    if input_dict['housing'] not in valid_housing:
        errors.append(f"Housing status must be one of: {valid_housing}")
    
    valid_job = ["unskilled non-resident", "unskilled resident", "skilled", "highly skilled"]
    if input_dict['job'] not in valid_job:
        errors.append(f"Job type must be one of: {valid_job}")
    
    valid_own_telephone = ["yes", "no"]
    if input_dict['own_telephone'] not in valid_own_telephone:
        errors.append(f"Own telephone must be one of: {valid_own_telephone}")
    
    valid_foreign_worker = ["yes", "no"]
    if input_dict['foreign_worker'] not in valid_foreign_worker:
        errors.append(f"Foreign worker must be one of: {valid_foreign_worker}")
    
    return errors

def transform_single_input(input_dict, preprocessor, feature_order):
    """Convert user input dictionary to preprocessed array for prediction."""
    # Create a DataFrame with the correct column order
    row_data = {col: [input_dict[col]] for col in feature_order}
    df = pd.DataFrame(row_data, columns=feature_order)
    return preprocessor.transform(df)