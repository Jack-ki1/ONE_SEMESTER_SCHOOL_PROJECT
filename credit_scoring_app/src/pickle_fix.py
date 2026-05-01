"""
Fix for sklearn compatibility issue with preprocessor pickle loading.

This module addresses the issue where pickled preprocessors from older sklearn versions
fail to load in newer versions due to missing classes like '_RemainderColsList'.
"""

import sys
import pickle
import importlib
import warnings

def patch_sklearn_modules():
    """
    Patch sklearn modules to handle compatibility issues with older pickle files.
    """
    # Suppress sklearn version warnings during loading
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    
    # Import the modules we need to patch
    try:
        from sklearn import compose
        import sklearn.compose._column_transformer
        import sklearn.preprocessing._data
        import sklearn.linear_model._logistic
    except ImportError:
        pass  # Not all modules might be available
    
    # Add placeholders for missing classes to prevent AttributeError
    if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
        class _RemainderColsList:
            """Placeholder for the missing class in newer sklearn versions."""
            def __init__(self, *args, **kwargs):
                pass
        
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

    # Patch StandardScaler if needed
    try:
        from sklearn.preprocessing import StandardScaler
        # Ensure backward compatibility
        if not hasattr(StandardScaler, '_get_tags'):
            def _get_tags(self):
                return {"allow_nan": True, "stateless": False}
            StandardScaler._get_tags = _get_tags
    except:
        pass

    # Patch LogisticRegression if needed
    try:
        from sklearn.linear_model import LogisticRegression
        # Ensure backward compatibility
        if not hasattr(LogisticRegression, '_get_tags'):
            def _get_tags(self):
                return {"allow_nan": True, "stateless": False}
            LogisticRegression._get_tags = _get_tags
    except:
        pass


def load_with_compatibility(path):
    """
    Load a pickle file with sklearn compatibility fixes applied.
    
    Args:
        path: Path to the pickle file
        
    Returns:
        Loaded object
    """
    # Apply the compatibility patches
    patch_sklearn_modules()
    
    # Now attempt to load the pickle file
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    
    # Reset warnings to default behavior
    warnings.resetwarnings()
    
    return obj