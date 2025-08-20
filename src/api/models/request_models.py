"""
Pydantic request models for API endpoints
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    user_id: str = Field(..., min_length=1, max_length=100, description="Unique user identifier")
    language: str = Field(default="auto", description="Language code (auto-detect if 'auto')")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    cultural_context: Optional[Dict[str, Any]] = Field(None, description="Cultural context information")
    enable_real_time_evaluation: bool = Field(default=False, description="Enable real-time evaluation")
    
    @validator("language")
    def validate_language(cls, v):
        supported_languages = ["auto", "en", "es", "fr", "de", "it", "ar", "el"]
        if v not in supported_languages:
            raise ValueError(f"Language must be one of {supported_languages}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "message": "¿Qué es la virtud según Aristóteles?",
                "user_id": "user123",
                "language": "es",
                "cultural_context": {
                    "region": "latin_america",
                    "education_level": "university"
                },
                "enable_real_time_evaluation": True
            }
        }

class StreamChatRequest(BaseModel):
    """Request model for streaming chat endpoint"""
    message: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., min_length=1, max_length=100)
    language: str = Field(default="auto")
    conversation_id: Optional[str] = None
    cultural_context: Optional[Dict[str, Any]] = None
    
    @validator("language")
    def validate_language(cls, v):
        supported_languages = ["auto", "en", "es", "fr", "de", "it", "ar", "el"]
        if v not in supported_languages:
            raise ValueError(f"Language must be one of {supported_languages}")
        return v

class EvaluationRequest(BaseModel):
    """Request model for evaluation endpoint"""
    responses: Dict[str, str] = Field(..., description="Responses by language")
    expected_concepts: List[str] = Field(..., description="Expected philosophical concepts")
    evaluation_type: str = Field(default="comprehensive", description="Type of evaluation")
    cultural_contexts: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Cultural contexts by language")
    reference_translations: Optional[Dict[str, List[str]]] = Field(None, description="Reference translations")
    
    @validator("evaluation_type")
    def validate_evaluation_type(cls, v):
        valid_types = ["comprehensive", "philosophical_accuracy", "cultural_appropriateness", 
                      "educational_effectiveness", "cross_lingual_consistency"]
        if v not in valid_types:
            raise ValueError(f"Evaluation type must be one of {valid_types}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "responses": {
                    "en": "Virtue is a disposition to act in ways that promote human flourishing...",
                    "es": "La virtud es una disposición para actuar de maneras que promuevan el florecimiento humano..."
                },
                "expected_concepts": ["virtue", "disposition", "flourishing"],
                "evaluation_type": "comprehensive",
                "cultural_contexts": {
                    "es": {"region": "latin_america", "education_level": "university"}
                }
            }
        }

class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    user_id: str = Field(..., min_length=1, max_length=100)
    conversation_id: str = Field(..., min_length=1, max_length=100)
    satisfaction_score: int = Field(..., ge=1, le=5, description="Satisfaction score (1-5)")
    feedback_text: Optional[str] = Field(None, max_length=1000, description="Optional feedback text")
    categories: Optional[List[str]] = Field(None, description="Feedback categories")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "conversation_id": "conv456",
                "satisfaction_score": 4,
                "feedback_text": "Very helpful explanation of virtue ethics",
                "categories": ["philosophical_accuracy", "cultural_relevance"]
            }
        }