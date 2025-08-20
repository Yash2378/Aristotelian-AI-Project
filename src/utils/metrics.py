"""
Metrics collection utilities
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

async def record_chat_interaction(
    user_id: str,
    message: str,
    response: str,
    language: str,
    processing_time: float,
    confidence: float,
    philosophical_concepts: Optional[list] = None,
    cultural_notes: str = "",
    evaluation_scores: Optional[Dict[str, float]] = None
):
    """Record chat interaction metrics"""
    try:
        # This would integrate with your metrics system (Prometheus, etc.)
        logger.info(
            "Chat interaction recorded",
            extra={
                "user_id": user_id,
                "language": language,
                "message_length": len(message),
                "response_length": len(response),
                "processing_time": processing_time,
                "confidence": confidence,
                "philosophical_concepts_count": len(philosophical_concepts or []),
                "has_cultural_notes": bool(cultural_notes),
                "evaluation_scores": evaluation_scores or {},
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Additional metrics recording logic would go here
        # (e.g., updating Prometheus counters, sending to analytics service)
        
    except Exception as e:
        logger.error(f"Failed to record chat interaction metrics: {e}")
