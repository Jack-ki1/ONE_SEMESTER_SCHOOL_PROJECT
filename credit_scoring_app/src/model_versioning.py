"""
Model versioning system for the credit scoring application.

This module provides functionality for managing different model versions,
tracking performance metrics, and enabling rollback capabilities.
"""

import os
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import our configuration
try:
    from .config import Config  # Support importing as a module within a package
    from .pickle_fix import load_with_compatibility  # Added import for compatibility fix
except ImportError:
    from src.config import Config  # Support running as a script directly
    from src.pickle_fix import load_with_compatibility  # Added import for compatibility fix

# Try to import optional dependencies, but don't fail if they're missing
try:
    import joblib
except ImportError:
    joblib = None


class ModelVersion:
    """Represents a specific version of a model."""
    
    def __init__(self, version_id: str, model_path: str, preprocessor_path: str, 
                 explainer_path: str, feature_names_path: str, metadata: Dict = None):
        self.version_id = version_id
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.explainer_path = explainer_path
        self.feature_names_path = feature_names_path
        self.created_at = datetime.now()
        self.metadata = metadata or {}
        
        # Calculate hash for integrity verification
        self.model_hash = self._calculate_file_hash(model_path)
        self.preprocessor_hash = self._calculate_file_hash(preprocessor_path)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        if not os.path.exists(file_path):
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class ModelVersionManager:
    """Manages different versions of models."""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path) if storage_path else Config.ARTIFACTS_DIR / "versions"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.storage_path / "versions.json"
        
        # Load existing versions
        self.versions = self._load_versions()
        self.current_version = self._get_current_version()
    
    def _load_versions(self) -> Dict[str, ModelVersion]:
        """Load version information from file."""
        if not self.versions_file.exists():
            return {}
        
        try:
            with open(self.versions_file, 'r') as f:
                versions_data = json.load(f)
            
            versions = {}
            for version_id, data in versions_data.items():
                versions[version_id] = ModelVersion(
                    version_id=version_id,
                    model_path=data['model_path'],
                    preprocessor_path=data['preprocessor_path'],
                    explainer_path=data['explainer_path'],
                    feature_names_path=data['feature_names_path'],
                    metadata=data.get('metadata', {})
                )
            
            return versions
        except Exception as e:
            print(f"Error loading versions: {e}")
            return {}
    
    def _save_versions(self):
        """Save version information to file."""
        versions_data = {}
        for version_id, version_obj in self.versions.items():
            versions_data[version_id] = {
                'version_id': version_obj.version_id,
                'model_path': version_obj.model_path,
                'preprocessor_path': version_obj.preprocessor_path,
                'explainer_path': version_obj.explainer_path,
                'feature_names_path': version_obj.feature_names_path,
                'created_at': version_obj.created_at.isoformat(),
                'metadata': version_obj.metadata
            }
        
        with open(self.versions_file, 'w') as f:
            json.dump(versions_data, f, indent=2)
    
    def _get_current_version(self) -> Optional[str]:
        """Get the current active version."""
        if not self.versions:
            return None
        
        # Find the version with the most recent creation date
        latest_version = max(
            self.versions.values(),
            key=lambda v: v.created_at
        )
        return latest_version.version_id
    
    def save_version(self, version_id: str, model, preprocessor, explainer, 
                     feature_names, metadata: Dict = None) -> bool:
        """
        Save a new model version.
        
        Args:
            version_id: Unique identifier for the version
            model: The trained model to save
            preprocessor: The fitted preprocessor to save
            explainer: The SHAP explainer to save
            feature_names: Feature names to save
            metadata: Additional metadata about this version
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create version directory
            version_dir = self.storage_path / version_id
            version_dir.mkdir(exist_ok=True)
            
            # Save model artifacts
            model_path = str(version_dir / "model.pkl")
            preprocessor_path = str(version_dir / "preprocessor.pkl")
            explainer_path = str(version_dir / "explainer.pkl")
            feature_names_path = str(version_dir / "feature_names.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            
            with open(explainer_path, 'wb') as f:
                pickle.dump(explainer, f)
            
            with open(feature_names_path, 'wb') as f:
                pickle.dump(feature_names, f)
            
            # Create version object
            version = ModelVersion(
                version_id=version_id,
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                explainer_path=explainer_path,
                feature_names_path=feature_names_path,
                metadata=metadata or {}
            )
            
            # Add to versions and save
            self.versions[version_id] = version
            self._save_versions()
            
            # Set as current version if it's the first or we don't have a current version
            if not self.current_version:
                self.current_version = version_id
            
            return True
        except Exception as e:
            print(f"Error saving model version: {e}")
            return False
    
    def load_version(self, version_id: str) -> Optional[Tuple]:
        """
        Load a specific model version.
        
        Args:
            version_id: ID of the version to load
        
        Returns:
            Tuple of (model, preprocessor, explainer, feature_names) or None if not found
        """
        if version_id not in self.versions:
            return None
        
        version = self.versions[version_id]
        
        # Verify file integrity using hashes
        if version.model_hash != self._calculate_file_hash(version.model_path):
            print(f"Model file integrity check failed for version {version_id}")
            return None
        
        if version.preprocessor_hash != self._calculate_file_hash(version.preprocessor_path):
            print(f"Preprocessor file integrity check failed for version {version_id}")
            return None
        
        try:
            # Load model artifacts
            with open(version.model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Use the compatibility loader for the preprocessor
            preprocessor = load_with_compatibility(version.preprocessor_path)
            
            with open(version.explainer_path, 'rb') as f:
                explainer = pickle.load(f)
            
            with open(version.feature_names_path, 'rb') as f:
                feature_names = pickle.load(f)
            
            return model, preprocessor, explainer, feature_names
        except Exception as e:
            print(f"Error loading model version: {e}")
            return None
    
    def get_version_metadata(self, version_id: str) -> Optional[Dict]:
        """Get metadata for a specific version."""
        if version_id not in self.versions:
            return None
        
        return self.versions[version_id].metadata
    
    def list_versions(self) -> List[str]:
        """List all available versions."""
        return list(self.versions.keys())
    
    def delete_version(self, version_id: str) -> bool:
        """Delete a specific model version."""
        if version_id not in self.versions:
            return False
        
        try:
            # Remove the version directory
            version_dir = self.storage_path / version_id
            import shutil
            shutil.rmtree(version_dir)
            
            # Remove from versions dict
            del self.versions[version_id]
            
            # Update current version if needed
            if self.current_version == version_id:
                # Set to the most recent version
                if self.versions:
                    self.current_version = self._get_current_version()
                else:
                    self.current_version = None
            
            # Save versions
            self._save_versions()
            
            return True
        except Exception as e:
            print(f"Error deleting model version: {e}")
            return False
    
    def activate_version(self, version_id: str) -> bool:
        """
        Activate a specific version as the current version.
        
        Args:
            version_id: ID of the version to activate
        
        Returns:
            True if successful, False otherwise
        """
        if version_id not in self.versions:
            return False
        
        self.current_version = version_id
        return True
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        if not os.path.exists(file_path):
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash
    
    def get_version_details(self, version_id: str) -> Optional[Dict]:
        """Get detailed information about a specific version."""
        if version_id not in self.versions:
            return None
        
        version = self.versions[version_id]
        return {
            'version_id': version.version_id,
            'created_at': version.created_at.isoformat(),
            'model_path': version.model_path,
            'preprocessor_path': version.preprocessor_path,
            'explainer_path': version.explainer_path,
            'feature_names_path': version.feature_names_path,
            'model_hash': version.model_hash,
            'preprocessor_hash': version.preprocessor_hash,
            'metadata': version.metadata
        }


# Global instance for convenience
version_manager = ModelVersionManager()


def save_current_model_version(version_id: str, metadata: Dict = None):
    """
    Save the current model artifacts as a new version.
    
    Args:
        version_id: Unique identifier for the version
        metadata: Additional metadata about this version
    """
    # Load current artifacts
    model, preprocessor, explainer, feature_names = load_current_model_artifacts()
    
    # Save as new version
    return version_manager.save_version(
        version_id=version_id,
        model=model,
        preprocessor=preprocessor,
        explainer=explainer,
        feature_names=feature_names,
        metadata=metadata
    )


def load_current_model_artifacts():
    """
    Load the current model artifacts.
    
    Returns:
        Tuple of (model, preprocessor, explainer, feature_names)
    """
    # Try to load from the current version if versioning is enabled
    # Otherwise, fall back to the default locations
    
    if version_manager.current_version:
        result = version_manager.load_version(version_manager.current_version)
        if result:
            return result
    
    # Fall back to default locations
    with open(Config.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(Config.PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(Config.EXPLAINER_PATH, 'rb') as f:
        explainer = pickle.load(f)
    
    with open(Config.FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, preprocessor, explainer, feature_names