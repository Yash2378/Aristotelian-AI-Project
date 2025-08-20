"""
Custom exception classes for the Aristotelian AI system
"""

from typing import Optional, Dict, Any

class AristotelianAIException(Exception):
    """Base exception for all Aristotelian AI errors"""
    
    def __init__(
        self,
        message: str,
        error_type: str = "aristotelian_ai_error",
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ChatException(AristotelianAIException):
    """Exception raised during chat interactions"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="chat_error",
            status_code=400,
            details=details
        )

class PhilosophicalAccuracyException(AristotelianAIException):
    """Exception raised when philosophical accuracy is compromised"""
    
    def __init__(self, message: str, concept: str, accuracy_score: float):
        super().__init__(
            message=message,
            error_type="philosophical_accuracy_error",
            status_code=422,
            details={
                "concept": concept,
                "accuracy_score": accuracy_score,
                "threshold": 0.7
            }
        )

class CulturalAdaptationException(AristotelianAIException):
    """Exception raised when cultural adaptation fails"""
    
    def __init__(self, message: str, language: str, cultural_score: float):
        super().__init__(
            message=message,
            error_type="cultural_adaptation_error",
            status_code=422,
            details={
                "language": language,
                "cultural_score": cultural_score,
                "threshold": 0.6
            }
        )

class RateLimitException(AristotelianAIException):
    """Exception raised when rate limits are exceeded"""
    
    def __init__(self, user_id: str, current_requests: int, limit: int):
        super().__init__(
            message=f"Rate limit exceeded for user {user_id}",
            error_type="rate_limit_exceeded",
            status_code=429,
            details={
                "user_id": user_id,
                "current_requests": current_requests,
                "limit": limit
            }
        )

class EvaluationException(AristotelianAIException):
    """Exception raised during evaluation processes"""
    
    def __init__(self, message: str, evaluation_type: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="evaluation_error",
            status_code=500,
            details={
                "evaluation_type": evaluation_type,
                **(details or {})
            }
        )

class ModelException(AristotelianAIException):
    """Exception raised when interacting with language models"""
    
    def __init__(self, message: str, model_id: str, api_error: Optional[str] = None):
        super().__init__(
            message=message,
            error_type="model_error",
            status_code=502,
            details={
                "model_id": model_id,
                "api_error": api_error
            }
        )

class DatabaseException(AristotelianAIException):
    """Exception raised for database-related errors"""
    
    def __init__(self, message: str, operation: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="database_error",
            status_code=500,
            details={
                "operation": operation,
                **(details or {})
            }
        )

class ValidationException(AristotelianAIException):
    """Exception raised for input validation errors"""
    
    def __init__(self, message: str, field: str, value: Any):
        super().__init__(
            message=message,
            error_type="validation_error",
            status_code=422,
            details={
                "field": field,
                "value": str(value)
            }
        )

class ExpertValidationException(AristotelianAIException):
    """Exception raised when expert validation fails"""
    
    def __init__(self, message: str, expert_id: str, validation_score: float):
        super().__init__(
            message=message,
            error_type="expert_validation_error",
            status_code=422,
            details={
                "expert_id": expert_id,
                "validation_score": validation_score,
                "threshold": 0.8
            }
        )
