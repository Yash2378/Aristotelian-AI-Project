"""
Pydantic response models for API endpoints
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="AI generated response")
    conversation_id: str = Field(..., description="Conversation identifier")
    language: str = Field(..., description="Detected/used language")
    cultural_notes: str = Field(default="", description="Cultural adaptation notes")
    philosophical_concepts: List[str] = Field(default_factory=list, description="Identified philosophical concepts")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    evaluation_scores: Dict[str, float] = Field(default_factory=dict, description="Real-time evaluation scores")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "La virtud, según Aristóteles, es una disposición del carácter que nos inclina a actuar de manera excelente...",
                "conversation_id": "conv_123",
                "language": "es",
                "cultural_notes": "Adapted with Latin American cultural examples",
                "philosophical_concepts": ["virtue", "character", "excellence"],
                "confidence": 0.92,
                "processing_time": 1.8,
                "evaluation_scores": {
                    "philosophical_accuracy": 0.89,
                    "cultural_appropriateness": 0.85
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

class StreamChatResponse(BaseModel):
    """Response model for streaming chat chunks"""
    chunk: str = Field(..., description="Response chunk")
    conversation_id: str = Field(..., description="Conversation identifier")
    chunk_type: str = Field(..., description="Type of chunk (text, concept, cultural_note)")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")
    
    class Config:
        schema_extra = {
            "example": {
                "chunk": "La virtud es",
                "conversation_id": "conv_123",
                "chunk_type": "text",
                "is_final": False
            }
        }

class EvaluationResponse(BaseModel):
    """Response model for evaluation endpoint"""
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    overall_performance: Dict[str, float] = Field(..., description="Overall performance metrics")
    detailed_results: Dict[str, Any] = Field(..., description="Detailed evaluation results")
    recommendations: List[Dict[str, str]] = Field(..., description="Improvement recommendations")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Evaluation timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "evaluation_id": "eval_20240115_103000",
                "overall_performance": {
                    "philosophical_accuracy": 0.84,
                    "cultural_appropriateness": 0.78,
                    "cross_lingual_consistency": 0.87
                },
                "detailed_results": {
                    "philosophical_accuracy": {
                        "en": {"concept_accuracy": 0.85, "argument_validity": 0.82}
                    }
                },
                "recommendations": [
                    {
                        "category": "Cultural Adaptation",
                        "priority": "High",
                        "recommendation": "Enhance metaphor database for German language"
                    }
                ],
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="System status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="System version")
    environment: str = Field(..., description="Environment name")
    components: Dict[str, str] = Field(..., description="Component health status")
    metrics: Optional[Dict[str, Any]] = Field(None, description="System metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "2.0.0",
                "environment": "production",
                "components": {
                    "database": "healthy",
                    "redis": "healthy",
                    "cohere_api": "healthy",
                    "evaluation_system": "healthy"
                },
                "metrics": {
                    "response_time_p95": 185,
                    "active_users_24h": 1247,
                    "philosophical_accuracy_avg": 0.84
                }
            }
        }

class MetricsResponse(BaseModel):
    """Response model for metrics endpoint"""
    philosophical_accuracy: float = Field(..., description="Average philosophical accuracy")
    cultural_appropriateness: float = Field(..., description="Average cultural appropriateness")
    educational_effectiveness: float = Field(..., description="Average educational effectiveness")
    cross_lingual_consistency: float = Field(..., description="Average cross-lingual consistency")
    bleu_score_average: float = Field(..., description="Average BLEU score")
    active_users_24h: int = Field(..., description="Active users in last 24 hours")
    total_interactions: int = Field(..., description="Total interactions")
    response_time_p95: float = Field(..., description="95th percentile response time")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "philosophical_accuracy": 0.84,
                "cultural_appropriateness": 0.78,
                "educational_effectiveness": 0.81,
                "cross_lingual_consistency": 0.87,
                "bleu_score_average": 0.86,
                "active_users_24h": 1247,
                "total_interactions": 45782,
                "response_time_p95": 185.0,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }