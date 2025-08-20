"""
Configuration management using Pydantic settings
"""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic import BaseSettings, Field, validator

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database Configuration
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    REDIS_URL: str = Field(..., env="REDIS_URL")
    
    # API Keys
    COHERE_API_KEY: str = Field(..., env="COHERE_API_KEY")
    FINE_TUNED_MODEL_ID: Optional[str] = Field(None, env="FINE_TUNED_MODEL_ID")
    
    # Feature Flags
    AYA_INTEGRATION_ENABLED: bool = Field(default=True, env="AYA_INTEGRATION_ENABLED")
    EXPERT_VALIDATION_ENABLED: bool = Field(default=True, env="EXPERT_VALIDATION_ENABLED")
    REAL_TIME_EVALUATION_ENABLED: bool = Field(default=True, env="REAL_TIME_EVALUATION_ENABLED")
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    
    # Application Settings
    SUPPORTED_LANGUAGES: List[str] = Field(
        default=["en", "es", "fr", "de", "it"],
        env="SUPPORTED_LANGUAGES"
    )
    MAX_TOKENS: int = Field(default=15000000, env="MAX_TOKENS")
    TRAINING_BATCH_SIZE: int = Field(default=32, env="TRAINING_BATCH_SIZE")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_HOUR: int = Field(default=1000, env="RATE_LIMIT_REQUESTS_PER_HOUR")
    RATE_LIMIT_BURST_SIZE: int = Field(default=50, env="RATE_LIMIT_BURST_SIZE")
    
    # Performance Settings
    REQUEST_TIMEOUT: int = Field(default=30, env="REQUEST_TIMEOUT")
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")
    
    # Security
    JWT_SECRET_KEY: Optional[str] = Field(None, env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_EXPIRATION_HOURS: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="ALLOWED_ORIGINS"
    )
    
    # Monitoring
    METRICS_EXPORT_INTERVAL: int = Field(default=60, env="METRICS_EXPORT_INTERVAL")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Evaluation Settings
    EVALUATION_FREQUENCY: str = Field(default="daily", env="EVALUATION_FREQUENCY")
    TARGET_PHILOSOPHICAL_ACCURACY: float = Field(default=0.8, env="TARGET_PHILOSOPHICAL_ACCURACY")
    TARGET_CULTURAL_APPROPRIATENESS: float = Field(default=0.75, env="TARGET_CULTURAL_APPROPRIATENESS")
    TARGET_ENGAGEMENT_IMPROVEMENT: float = Field(default=0.38, env="TARGET_ENGAGEMENT_IMPROVEMENT")
    TARGET_BLEU_PARITY: float = Field(default=0.85, env="TARGET_BLEU_PARITY")
    
    # Aya Integration
    AYA_API_KEY: Optional[str] = Field(None, env="AYA_API_KEY")
    AYA_DATASET_SHARING_ENABLED: bool = Field(default=True, env="AYA_DATASET_SHARING_ENABLED")
    AYA_COMMUNITY_FEEDBACK_ENABLED: bool = Field(default=True, env="AYA_COMMUNITY_FEEDBACK_ENABLED")
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        valid_environments = ["development", "staging", "production", "testing"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator("SUPPORTED_LANGUAGES")
    def validate_supported_languages(cls, v):
        if isinstance(v, str):
            v = [lang.strip() for lang in v.split(",")]
        
        valid_languages = ["en", "es", "fr", "de", "it", "ar", "el", "la"]
        for lang in v:
            if lang not in valid_languages:
                raise ValueError(f"Language {lang} not supported. Valid languages: {valid_languages}")
        return v
    
    @validator("ALLOWED_ORIGINS")
    def validate_allowed_origins(cls, v):
        if isinstance(v, str):
            v = [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()

# Environment-specific configurations
def get_database_config() -> dict:
    """Get database configuration"""
    settings = get_settings()
    return {
        "url": settings.DATABASE_URL,
        "pool_size": 20 if settings.ENVIRONMENT == "production" else 5,
        "max_overflow": 30 if settings.ENVIRONMENT == "production" else 10,
        "pool_timeout": 30,
        "pool_recycle": 3600
    }

def get_redis_config() -> dict:
    """Get Redis configuration"""
    settings = get_settings()
    return {
        "url": settings.REDIS_URL,
        "encoding": "utf-8",
        "decode_responses": True,
        "socket_timeout": 5,
        "socket_connect_timeout": 5,
        "retry_on_timeout": True,
        "health_check_interval": 30
    }

def get_cohere_config() -> dict:
    """Get Cohere API configuration"""
    settings = get_settings()
    return {
        "api_key": settings.COHERE_API_KEY,
        "model_id": settings.FINE_TUNED_MODEL_ID or "command-r-plus",
        "max_tokens": 500,
        "temperature": 0.7,
        "k": 0,
        "p": 0.75,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "timeout": settings.REQUEST_TIMEOUT
    }