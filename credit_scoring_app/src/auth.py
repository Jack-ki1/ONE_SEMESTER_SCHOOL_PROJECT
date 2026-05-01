"""
Authentication module for the credit scoring application.

This module provides user authentication and authorization functionality.
"""

import bcrypt
import jwt
import os
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, ValidationError
from enum import Enum


class UserRole(str, Enum):
    """Enumeration of user roles in the system."""
    ADMIN = "admin"
    ANALYST = "analyst"
    LOAN_OFFICER = "loan_officer"
    AUDITOR = "auditor"


class User(BaseModel):
    """User model representing a system user."""
    username: str
    email: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = datetime.utcnow()
    last_login: Optional[datetime] = None


class AuthManager:
    """Manages authentication and authorization for the application."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize the authentication manager.
        
        Args:
            secret_key: Secret key for JWT encoding/decoding. If not provided,
                       will use environment variable or default.
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
        self.users = {}  # In production, this would be a database
        self._initialize_default_users()
    
    def _initialize_default_users(self):
        """Initialize default users for the system."""
        # Hashed password for default users: "password123"
        default_password_hash = bcrypt.hashpw("password123".encode('utf-8'), bcrypt.gensalt())
        
        # Create default admin user
        admin_user = User(
            username="admin",
            email="admin@creditwise.example.com",
            role=UserRole.ADMIN
        )
        self.users["admin"] = {
            "user": admin_user,
            "password_hash": default_password_hash
        }
        
        # Create default analyst user
        analyst_user = User(
            username="analyst",
            email="analyst@creditwise.example.com",
            role=UserRole.ANALYST
        )
        self.users["analyst"] = {
            "user": analyst_user,
            "password_hash": default_password_hash
        }
    
    def register_user(self, username: str, email: str, password: str, role: UserRole) -> bool:
        """
        Register a new user in the system.
        
        Args:
            username: Unique username
            email: User's email address
            password: Plain text password (will be hashed)
            role: User role in the system
        
        Returns:
            True if registration successful, False otherwise
        """
        if username in self.users:
            return False
        
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        user = User(
            username=username,
            email=email,
            role=role
        )
        
        self.users[username] = {
            "user": user,
            "password_hash": password_hash
        }
        
        return True
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate a user and return a JWT token if successful.
        
        Args:
            username: Username to authenticate
            password: Plain text password to verify
        
        Returns:
            JWT token if authentication successful, None otherwise
        """
        if username not in self.users:
            return None
        
        user_data = self.users[username]
        password_hash = user_data["password_hash"]
        
        if bcrypt.checkpw(password.encode('utf-8'), password_hash):
            # Update last login
            user_data["user"].last_login = datetime.utcnow()
            
            # Create JWT token
            payload = {
                "sub": username,
                "role": user_data["user"].role.value,
                "exp": datetime.utcnow() + timedelta(hours=24),
                "iat": datetime.utcnow()
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm="HS256")
            return token
        
        return None
    
    def verify_token(self, token: str) -> Optional[dict]:
        """
        Verify a JWT token and return user info if valid.
        
        Args:
            token: JWT token to verify
        
        Returns:
            User info dictionary if token is valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            username = payload["sub"]
            
            if username not in self.users:
                return None
                
            user_data = self.users[username]
            return {
                "username": username,
                "role": user_data["user"].role.value,
                "is_active": user_data["user"].is_active
            }
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def authorize_user(self, token: str, required_role: UserRole) -> bool:
        """
        Check if a user has the required role.
        
        Args:
            token: JWT token to check
            required_role: Required role for authorization
        
        Returns:
            True if user is authorized, False otherwise
        """
        user_info = self.verify_token(token)
        if not user_info:
            return False
        
        user_role = UserRole(user_info["role"])
        
        # Define role hierarchy (admin can do everything)
        if required_role == UserRole.ADMIN:
            return user_role == UserRole.ADMIN
        elif required_role == UserRole.ANALYST:
            return user_role in [UserRole.ADMIN, UserRole.ANALYST]
        elif required_role == UserRole.LOAN_OFFICER:
            return user_role in [UserRole.ADMIN, UserRole.LOAN_OFFICER]
        elif required_role == UserRole.AUDITOR:
            return user_role in [UserRole.ADMIN, UserRole.AUDITOR]
        
        return False
    
    def get_user(self, username: str) -> Optional[User]:
        """
        Get user information by username.
        
        Args:
            username: Username to look up
        
        Returns:
            User object if found, None otherwise
        """
        if username in self.users:
            return self.users[username]["user"]
        return None


# Global instance of AuthManager
auth_manager = AuthManager()


def get_current_user(token: str) -> Optional[User]:
    """
    Helper function to get the current user from a token.
    
    Args:
        token: JWT token
    
    Returns:
        User object if token is valid, None otherwise
    """
    user_info = auth_manager.verify_token(token)
    if user_info:
        return auth_manager.get_user(user_info["username"])
    return None