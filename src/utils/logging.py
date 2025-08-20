"""
Centralized logging configuration
"""

import logging
import logging.config
import sys
from typing import Dict, Any
from datetime import datetime

from src.utils.config import get_settings

def setup_logging():
    """Setup application logging configuration"""
    settings = get_settings()
    
    # Logging configuration
    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "[{asctime}] {levelname} {name}:{lineno} - {message}",
                "style": "{",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "{levelname} {name} - {message}",
                "style": "{"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "detailed" if settings.ENVIRONMENT == "development" else "json",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": "logs/aristotelian_ai.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "src": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console", "file"] if settings.ENVIRONMENT == "production" else ["console"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"]
        }
    }
    
    # Create logs directory if it doesn't exist
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured for environment: {settings.ENVIRONMENT}")
    logger.info(f"Log level: {settings.LOG_LEVEL}")

class StructuredLogger:
    """Structured logger for philosophical AI interactions"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_chat_interaction(
        self,
        user_id: str,
        message: str,
        response: str,
        language: str,
        processing_time: float,
        confidence: float,
        philosophical_concepts: list = None,
        cultural_notes: str = "",
        evaluation_scores: dict = None
    ):
        """Log a chat interaction with structured data"""
        self.logger.info(
            "Chat interaction",
            extra={
                "event_type": "chat_interaction",
                "user_id": user_id,
                "message_length": len(message),
                "response_length": len(response),
                "language": language,
                "processing_time": processing_time,
                "confidence": confidence,
                "philosophical_concepts": philosophical_concepts or [],
                "cultural_notes": cultural_notes,
                "evaluation_scores": evaluation_scores or {},
                "timestamp": datetime.utcnow().isoformat()
            }
        )