"""
Security module for the credit scoring application.

This module provides security enhancements including input validation,
sanitization, and protection against common vulnerabilities.
"""

import re
import html
from typing import Any, Dict, Union
import logging
from urllib.parse import urlparse
import bleach

# Import our configuration
try:
    from .config import Config  # Support importing as a module within a package
except ImportError:
    from src.config import Config  # Support running as a script directly


class InputValidator:
    """Class to handle input validation and sanitization."""
    
    @staticmethod
    def sanitize_string(value: str) -> str:
        """
        Sanitize a string input to prevent injection attacks.
        
        Args:
            value: Input string to sanitize
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            raise TypeError("Value must be a string")
        
        # Remove HTML tags and encode dangerous characters
        sanitized = bleach.clean(value, strip=True)
        
        # Decode any HTML entities to prevent double encoding
        sanitized = html.unescape(sanitized)
        
        return sanitized
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_val: Union[int, float], 
                             max_val: Union[int, float], field_name: str) -> bool:
        """
        Validate that a numeric value is within a specified range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            field_name: Name of the field for error messages
            
        Returns:
            True if valid, raises ValueError if not
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be a number")
        
        if value < min_val or value > max_val:
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
        
        return True
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(email, str):
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(url, str):
            return False
        
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def validate_alphanumeric(value: str) -> bool:
        """
        Validate that a string contains only alphanumeric characters.
        
        Args:
            value: String to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, str):
            return False
        
        return value.replace('_', '').replace('-', '').replace(' ', '').isalnum()


class SecurityMiddleware:
    """Middleware class to handle security checks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.suspicious_patterns = [
            r'<script',  # Script tags
            r'javascript:',  # JavaScript protocol
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # Eval function
            r'expression\s*\(',  # CSS expression
            r'vbscript:',  # VBScript protocol
            r'&#x',  # Hex entities
            r'&#\d+'  # Numeric entities
        ]
    
    def check_for_suspicious_content(self, data: Union[str, Dict[str, Any]]) -> bool:
        """
        Check if data contains suspicious content.
        
        Args:
            data: Data to check (string or dict)
            
        Returns:
            True if suspicious content is found, False otherwise
        """
        if isinstance(data, str):
            content = data.lower()
        elif isinstance(data, dict):
            content = str(data).lower()
        else:
            content = str(data).lower()
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, content):
                self.logger.warning(f"Suspicious pattern detected: {pattern}")
                return True
        
        return False
    
    def sanitize_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize request data to remove potentially harmful content.
        
        Args:
            data: Request data to sanitize
            
        Returns:
            Sanitized data
        """
        sanitized_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Sanitize string values
                sanitized_data[key] = self._sanitize_value(value)
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized_data[key] = self.sanitize_request_data(value)
            elif isinstance(value, list):
                # Sanitize list items
                sanitized_list = []
                for item in value:
                    if isinstance(item, str):
                        sanitized_list.append(self._sanitize_value(item))
                    elif isinstance(item, dict):
                        sanitized_list.append(self.sanitize_request_data(item))
                    else:
                        sanitized_list.append(item)
                sanitized_data[key] = sanitized_list
            else:
                # Non-string values are kept as-is
                sanitized_data[key] = value
        
        return sanitized_data
    
    def _sanitize_value(self, value: str) -> str:
        """
        Internal method to sanitize a single value.
        
        Args:
            value: String value to sanitize
            
        Returns:
            Sanitized string
        """
        # Apply bleach cleaning
        sanitized = bleach.clean(value, strip=True)
        
        # Remove any remaining dangerous patterns
        dangerous_patterns = [
            r'(?:\b)(on\w+|javascript|vbscript|data|eval|expression)\s*[:=]',
            r'<\/?(?:script|iframe|frame|embed|object|meta|link)',
            r'&#[x]?(?:60|3c|62|3e|21|2d|2f|5c)',
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized


class RateLimiter:
    """Simple rate limiting to prevent abuse."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # Dictionary to track requests by IP
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if a request from an identifier is allowed.
        
        Args:
            identifier: Identifier for the requester (e.g., IP address)
            
        Returns:
            True if request is allowed, False otherwise
        """
        import time
        
        current_time = time.time()
        
        # Clean old entries
        self.requests = {
            ip: reqs for ip, reqs in self.requests.items()
            if current_time - reqs[0] < self.window_seconds
        }
        
        # Check if identifier exists
        if identifier in self.requests:
            count, timestamp = self.requests[identifier]
            
            # If time window has passed, reset
            if current_time - timestamp >= self.window_seconds:
                self.requests[identifier] = [1, current_time]
                return True
            
            # Check if limit exceeded
            if count >= self.max_requests:
                self.logger.warning(f"Rate limit exceeded for {identifier}")
                return False
            
            # Increment request count
            self.requests[identifier] = [count + 1, timestamp]
        else:
            # New identifier
            self.requests[identifier] = [1, current_time]
        
        return True


# Global instances
validator = InputValidator()
middleware = SecurityMiddleware()
rate_limiter = RateLimiter(max_requests=50, window_seconds=3600)  # 50 requests per hour


def validate_and_sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize input data.
    
    Args:
        data: Input data to validate and sanitize
        
    Returns:
        Validated and sanitized data
    """
    # Check for suspicious content
    if middleware.check_for_suspicious_content(data):
        raise ValueError("Suspicious content detected in input data")
    
    # Sanitize the data
    sanitized_data = middleware.sanitize_request_data(data)
    
    # Validate specific fields if needed
    # Add custom validation logic here based on your requirements
    
    return sanitized_data


def validate_ip(ip: str) -> bool:
    """
    Validate IP address format.
    
    Args:
        ip: IP address to validate
        
    Returns:
        True if valid, False otherwise
    """
    import ipaddress
    
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False