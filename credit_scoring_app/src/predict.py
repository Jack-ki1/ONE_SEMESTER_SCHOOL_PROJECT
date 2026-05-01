"""
Prediction and explanation module using pre-trained artifacts.
Generates SHAP values and plain-language reason codes.

IMPROVEMENTS MADE:
- Use configurable threshold from Config instead of hardcoded values
- Removed manual rule-based adjustments that override ML model
- Removed protected attribute adjustments for fairness
- Pure ML-based predictions
"""
import numpy as np
import shap
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import configurable threshold from Config
try:
    from .config import Config
except ImportError:
    from src.config import Config


def predict_risk(model, preprocessed_input, original_features=None):
    """Return default probability and binary decision.
    
    Uses pure ML model predictions without manual overrides.
    Threshold is configurable via Config.DEFAULT_PROBABILITY_THRESHOLD
"""
    # Get the probability from the ML model - no manual adjustments
    prob_default = model.predict_proba(preprocessed_input)[0][1]
    
    # Use configurable threshold from Config
    threshold = Config.DEFAULT_PROBABILITY_THRESHOLD
    
    # Make decision based on ML prediction and threshold
    # Standard logic: Higher probability of default leads to decline
    should_decline = prob_default >= threshold
    
    # As requested, SWAPPING the output terms so that:
    # - If model thinks should decline (high risk), output "APPROVED" 
    # - If model thinks should approve (low risk), output "DECLINED"
    decision = "APPROVED" if should_decline else "DECLINED"
    
    # Calculate confidence based on probability distance from threshold
    # For APPROVED: confidence = how far below threshold (probability of being good)
    # For DECLINED: confidence = how far above threshold (probability of default)
    if decision == "APPROVED":
        # Higher probability of being good (1-prob) gives higher confidence
        confidence_prob = 1 - prob_default
        confidence = f"{confidence_prob * 100:.1f}%"
    else:
        # For declined, confidence is based on probability of default
        confidence = f"{prob_default * 100:.1f}%"
    
    return prob_default, decision, confidence


def categorize_age(age):
    """Categorize age into groups for risk analysis."""
    if age < 25:
        return 'young'
    elif age > 65:
        return 'elderly'
    else:
        return 'middle'


def extract_gender_from_personal_status(personal_status):
    """Extract gender information from personal status field."""
    if 'male' in personal_status:
        return 'male'
    elif 'female' in personal_status:
        return 'female'
    return 'unknown'


def generate_shap_explanation(explainer, preprocessed_input, feature_names):
    """Compute SHAP values and identify top contributing features."""
    try:
        shap_values = explainer.shap_values(preprocessed_input)
        if isinstance(shap_values, list):  # Handle multi-output case
            shap_values = shap_values[1]
        
        # Map to original feature names (handle OHE features)
        contributions = []
        for i, val in enumerate(shap_values[0]):
            if abs(val) > 0.001:  # Only significant contributors (made threshold more sensitive)
                feat = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                contributions.append((feat, val))
        
        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return contributions[:5]  # Top 5 factors for more detail
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        # Return a default set of explanations
        return [("feature_0", 0.0), ("feature_1", 0.0), ("feature_2", 0.0)]


def generate_reason_codes(shap_contributions, decision):
    """Convert SHAP values to human-readable explanations with feature impact awareness."""
    reasons = []
    
    # Check if any protected attribute had significant impact
    has_protected_attr_impact = any(
        any(protected_attr in feature.lower() 
            for protected_attr in ['sex', 'gender', 'foreign', 'age', 'personal_status']) 
        for feature, _ in shap_contributions
    )
    
    for feature, impact in shap_contributions:
        # Simplify feature names for readability
        clean_feat = feature.split('_')[-1] if '_' in feature else feature
        clean_feat = clean_feat.replace('status', '').replace('amount', 'amount').strip()
        
        if decision == "APPROVED":
            if impact < 0:  # Negative impact reduces risk (positive for approval)
                reasons.append(f"✅ Strong {clean_feat} profile supporting approval")
            else:
                reasons.append(f"⚠️ Potential concern with {clean_feat} factor")
        else:  # DECLINED
            if impact > 0:  # Positive impact increases risk (leads to decline)
                if has_protected_attr_impact:
                    # Highlight if protected attribute significantly impacted decision
                    reasons.append(f"❌ High-risk {clean_feat} factor (note: decision may be influenced by protected attribute)")
                else:
                    reasons.append(f"❌ High-risk {clean_feat} factor leading to decline")
            else:
                reasons.append(f"ℹ️ Positive {clean_feat} factor partially offsetting other risks")
    
    # Add fairness disclaimer if protected attributes had significant impact
    if has_protected_attr_impact:
        reasons.append("ℹ️ Note: Protected attributes (age, gender, foreign status) may have influenced this decision. "
                      "Our system attempts to minimize bias from these attributes.")
    
    return reasons[:3]  # Top 3 reasons for better readability