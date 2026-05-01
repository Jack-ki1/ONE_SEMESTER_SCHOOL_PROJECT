"""
Database module for the credit scoring application.

This module provides database functionality for storing and retrieving
predictions, model metrics, and other application data.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
import os
from pathlib import Path
import hashlib
import logging

# Import our configuration
try:
    from .config import Config  # Support importing as a module within a package
except ImportError:
    from src.config import Config  # Support running as a script directly

# Database setup
DATABASE_URL = f"sqlite:///{Config.BASE_DIR}/credit_scoring.db"

engine = create_engine(DATABASE_URL, echo=False)  # Set echo=True for SQL debug logs
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

logger = logging.getLogger(__name__)

class PredictionLog(Base):
    """
    Table to store prediction events for monitoring and analysis.
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    applicant_features = Column(Text)  # JSON string of features
    prediction = Column(String)  # APPROVED/DECLINED
    default_probability = Column(Float)
    confidence = Column(Float)
    explanation = Column(Text)  # SHAP explanation or reason codes
    processed = Column(Boolean, default=False)  # For batch processing
    drift_detected = Column(Boolean, default=False)  # Flag for data drift detection


class ModelPerformance(Base):
    """
    Table to store model performance metrics over time.
    """
    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    model_version = Column(String, default="1.0.0")
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float, nullable=True)
    avg_precision = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)  # Additional notes about the evaluation


class FairnessAudit(Base):
    """
    Table to store fairness audit results.
    """
    __tablename__ = "fairness_audits"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    demographic_parity = Column(Float)
    equal_opportunity = Column(Float)
    sample_size = Column(Integer)
    sensitive_attribute = Column(String)  # e.g., "gender", "age_group"
    notes = Column(Text, nullable=True)  # Additional notes about the audit


# Create tables
Base.metadata.create_all(bind=engine)


def get_db():
    """
    Dependency to get database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_prediction_event(applicant_features: dict, prediction: str, 
                        default_probability: float, confidence: float, 
                        explanation: str = None):
    """
    Log a prediction event to the database.
    
    Parameters:
    - applicant_features: Dictionary of applicant features
    - prediction: The prediction result (APPROVED/DECLINED)
    - default_probability: The probability of default
    - confidence: The confidence level of the prediction
    - explanation: Optional explanation for the decision
    """
    db = SessionLocal()
    try:
        # Convert features to JSON string
        import json
        features_json = json.dumps(applicant_features)
        
        log_entry = PredictionLog(
            applicant_features=features_json,
            prediction=prediction,
            default_probability=default_probability,
            confidence=confidence,
            explanation=explanation
        )
        
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)  # Refresh to get the assigned ID
    except Exception as e:
        db.rollback()
        logger.error(f"Error logging prediction event: {str(e)}")
        raise e
    finally:
        db.close()


def save_model_performance(accuracy: float, precision: float, recall: float, 
                          f1_score: float, auc_roc: float = None, 
                          avg_precision: float = None, notes: str = None,
                          model_version: str = "1.0.0"):
    """
    Save model performance metrics to the database.
    
    Parameters:
    - accuracy: Model accuracy
    - precision: Model precision
    - recall: Model recall
    - f1_score: Model F1 score
    - auc_roc: Area under ROC curve (optional)
    - avg_precision: Average precision (optional)
    - notes: Additional notes (optional)
    - model_version: Version of the model being evaluated
    """
    db = SessionLocal()
    try:
        perf_entry = ModelPerformance(
            model_version=model_version,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc_roc=auc_roc,
            avg_precision=avg_precision,
            notes=notes
        )
        
        db.add(perf_entry)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving model performance: {str(e)}")
        raise e
    finally:
        db.close()


def save_fairness_audit(demographic_parity: float, equal_opportunity: float,
                       sample_size: int, sensitive_attribute: str, notes: str = None):
    """
    Save fairness audit results to the database.
    
    Parameters:
    - demographic_parity: Demographic parity difference
    - equal_opportunity: Equal opportunity difference
    - sample_size: Number of samples used in the audit
    - sensitive_attribute: The sensitive attribute tested (e.g., "gender", "age_group", "employment")
    - notes: Additional notes (optional)
    """
    db = SessionLocal()
    try:
        audit_entry = FairnessAudit(
            demographic_parity=demographic_parity,
            equal_opportunity=equal_opportunity,
            sample_size=sample_size,
            sensitive_attribute=sensitive_attribute,
            notes=notes
        )
        
        db.add(audit_entry)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving fairness audit: {str(e)}")
        raise e
    finally:
        db.close()


def get_recent_predictions(limit: int = 100):
    """
    Retrieve recent prediction logs.
    
    Parameters:
    - limit: Number of records to return
    
    Returns:
    - List of recent prediction logs
    """
    db = SessionLocal()
    try:
        predictions = db.query(PredictionLog).order_by(
            PredictionLog.timestamp.desc()
        ).limit(limit).all()
        return predictions
    except Exception as e:
        logger.error(f"Error getting recent predictions: {str(e)}")
        return []
    finally:
        db.close()


def get_model_performance_history(limit: int = 50):
    """
    Retrieve historical model performance metrics.
    
    Parameters:
    - limit: Number of records to return
    
    Returns:
    - List of model performance records
    """
    db = SessionLocal()
    try:
        performance_records = db.query(ModelPerformance).order_by(
            ModelPerformance.timestamp.desc()
        ).limit(limit).all()
        return performance_records
    except Exception as e:
        logger.error(f"Error getting model performance history: {str(e)}")
        return []
    finally:
        db.close()