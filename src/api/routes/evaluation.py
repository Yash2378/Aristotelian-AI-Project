"""
Evaluation API endpoints
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks

from src.api.models.request_models import EvaluationRequest, FeedbackRequest
from src.api.models.response_models import EvaluationResponse, MetricsResponse
from src.api.middleware.auth import get_current_user
from src.evaluation.orchestrator import ComprehensiveEvaluationOrchestrator
from src.utils.exceptions import EvaluationException

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_evaluation_orchestrator() -> ComprehensiveEvaluationOrchestrator:
    """Dependency to get evaluation orchestrator"""
    from src.api.main import evaluation_orchestrator
    if evaluation_orchestrator is None:
        raise HTTPException(status_code=503, detail="Evaluation system not available")
    return evaluation_orchestrator

@router.post("/", response_model=EvaluationResponse)
async def run_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    orchestrator: ComprehensiveEvaluationOrchestrator = Depends(get_evaluation_orchestrator)
):
    """
    Run comprehensive evaluation on provided responses
    
    Requires authentication and appropriate permissions.
    """
    try:
        # Prepare test data from request
        test_data = {
            "responses": request.responses,
            "expected_concepts": request.expected_concepts,
            "cultural_context": request.cultural_contexts or {},
            "reference_translations": request.reference_translations or {},
            "interaction_data": [],  # Could be provided in future versions
            "assessment_results": {},  # Could be provided in future versions
            "user_feedback": []  # Could be provided in future versions
        }
        
        # Run evaluation based on type
        if request.evaluation_type == "comprehensive":
            results = await orchestrator.run_comprehensive_evaluation(test_data)
        elif request.evaluation_type == "philosophical_accuracy":
            results = await run_philosophical_accuracy_evaluation(orchestrator, test_data)
        elif request.evaluation_type == "cultural_appropriateness":
            results = await run_cultural_appropriateness_evaluation(orchestrator, test_data)
        elif request.evaluation_type == "cross_lingual_consistency":
            results = await run_cross_lingual_evaluation(orchestrator, test_data)
        else:
            raise EvaluationException(f"Unsupported evaluation type: {request.evaluation_type}", request.evaluation_type)
        
        return EvaluationResponse(
            evaluation_id=results.get("evaluation_id", "unknown"),
            overall_performance=results.get("overall_performance", {}),
            detailed_results=results.get("detailed_results", {}),
            recommendations=results.get("report", {}).get("recommendations", [])
        )
        
    except EvaluationException as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Evaluation failed")

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    current_user: dict = Depends(get_current_user),
    orchestrator: ComprehensiveEvaluationOrchestrator = Depends(get_evaluation_orchestrator)
):
    """Get current system metrics"""
    try:
        metrics = await orchestrator.get_system_metrics()
        
        performance = metrics.get("performance", {})
        
        return MetricsResponse(
            philosophical_accuracy=performance.get("philosophical_accuracy", 0.0),
            cultural_appropriateness=performance.get("cultural_appropriateness", 0.0),
            educational_effectiveness=performance.get("educational_effectiveness", 0.0),
            cross_lingual_consistency=performance.get("cross_lingual_consistency", 0.0),
            bleu_score_average=performance.get("bleu_score_avg", 0.0),
            active_users_24h=performance.get("total_users_7d", 0),
            total_interactions=performance.get("total_interactions", 0),
            response_time_p95=performance.get("avg_processing_time", 0) * 1000  # Convert to ms
        )
        
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    orchestrator: ComprehensiveEvaluationOrchestrator = Depends(get_evaluation_orchestrator)
):
    """Submit user feedback for system improvement"""
    try:
        # Store feedback
        await orchestrator.metrics_collector.record_feedback(request)
        
        return {"message": "Feedback recorded successfully"}
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@router.get("/history/{evaluation_id}")
async def get_evaluation_history(
    evaluation_id: str,
    current_user: dict = Depends(get_current_user),
    orchestrator: ComprehensiveEvaluationOrchestrator = Depends(get_evaluation_orchestrator)
):
    """Get historical evaluation results"""
    try:
        # Implementation would retrieve evaluation history from database
        # This is a placeholder for the full implementation
        return {"message": f"Evaluation history for {evaluation_id}"}
        
    except Exception as e:
        logger.error(f"Error retrieving evaluation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve evaluation history")

# Helper functions for specific evaluation types
async def run_philosophical_accuracy_evaluation(orchestrator, test_data):
    """Run philosophical accuracy focused evaluation"""
    # Implementation would focus only on philosophical accuracy metrics
    return await orchestrator.run_comprehensive_evaluation(test_data)

async def run_cultural_appropriateness_evaluation(orchestrator, test_data):
    """Run cultural appropriateness focused evaluation"""
    # Implementation would focus only on cultural appropriateness metrics
    return await orchestrator.run_comprehensive_evaluation(test_data)

async def run_cross_lingual_evaluation(orchestrator, test_data):
    """Run cross-lingual consistency focused evaluation"""
    # Implementation would focus only on cross-lingual consistency metrics
    return await orchestrator.run_comprehensive_evaluation(test_data)