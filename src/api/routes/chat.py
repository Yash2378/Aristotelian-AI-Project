"""
Chat API endpoints for philosophical dialogue
"""

import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse

from src.api.models.request_models import ChatRequest, StreamChatRequest
from src.api.models.response_models import ChatResponse, StreamChatResponse
from src.api.middleware.rate_limiter import RateLimiter, check_rate_limit
from src.core.philosophical_dialogue import PhilosophicalDialogueManager
from src.evaluation.orchestrator import ComprehensiveEvaluationOrchestrator
from src.utils.exceptions import ChatException, RateLimitException
from src.utils.metrics import record_chat_interaction

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_dialogue_manager() -> PhilosophicalDialogueManager:
    """Dependency to get dialogue manager"""
    from src.api.main import dialogue_manager
    if dialogue_manager is None:
        raise HTTPException(status_code=503, detail="Dialogue manager not available")
    return dialogue_manager

async def get_evaluation_orchestrator() -> ComprehensiveEvaluationOrchestrator:
    """Dependency to get evaluation orchestrator"""
    from src.api.main import evaluation_orchestrator
    if evaluation_orchestrator is None:
        raise HTTPException(status_code=503, detail="Evaluation orchestrator not available")
    return evaluation_orchestrator

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    dialogue_manager: PhilosophicalDialogueManager = Depends(get_dialogue_manager),
    evaluation_orchestrator: ComprehensiveEvaluationOrchestrator = Depends(get_evaluation_orchestrator)
):
    """
    Engage in philosophical dialogue with cultural adaptation
    
    Returns culturally-adapted, philosophically-accurate responses
    with comprehensive evaluation metrics.
    """
    try:
        # Check rate limits
        await check_rate_limit(request.user_id)
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid4())
        
        # Generate response
        response_data = await dialogue_manager.generate_response(
            user_input=request.message,
            user_id=request.user_id,
            language=request.language,
            cultural_context=request.cultural_context,
            conversation_id=conversation_id
        )
        
        # Create response
        chat_response = ChatResponse(
            response=response_data["response"],
            conversation_id=conversation_id,
            language=response_data["language"],
            cultural_notes=response_data.get("cultural_notes", ""),
            philosophical_concepts=response_data.get("philosophical_concepts", []),
            confidence=response_data.get("confidence", 0.8),
            processing_time=response_data.get("processing_time", 0),
            evaluation_scores=response_data.get("evaluation_scores", {})
        )
        
        # Background tasks
        background_tasks.add_task(
            record_chat_interaction,
            user_id=request.user_id,
            message=request.message,
            response=chat_response.response,
            language=chat_response.language,
            processing_time=chat_response.processing_time,
            confidence=chat_response.confidence
        )
        
        # Optional real-time evaluation
        if request.enable_real_time_evaluation:
            background_tasks.add_task(
                run_real_time_evaluation,
                evaluation_orchestrator,
                request,
                chat_response
            )
        
        return chat_response
        
    except RateLimitException as e:
        logger.warning(f"Rate limit exceeded for user {request.user_id}")
        raise HTTPException(status_code=429, detail=str(e))
    
    except ChatException as e:
        logger.error(f"Chat error for user {request.user_id}: {e}")
        raise HTTPException(
            raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/stream", response_class=StreamingResponse)
async def stream_chat(
    request: StreamChatRequest,
    dialogue_manager: PhilosophicalDialogueManager = Depends(get_dialogue_manager)
):
    """
    Stream philosophical dialogue responses for real-time interaction
    """
    try:
        # Check rate limits
        await check_rate_limit(request.user_id)
        
        # Generate streaming response
        async def generate_stream():
            conversation_id = request.conversation_id or str(uuid4())
            
            async for chunk in dialogue_manager.generate_streaming_response(
                user_input=request.message,
                user_id=request.user_id,
                language=request.language,
                cultural_context=request.cultural_context,
                conversation_id=conversation_id
            ):
                yield f"data: {chunk.json()}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in stream chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Streaming error")

@router.get("/conversations/{user_id}")
async def get_user_conversations(
    user_id: str,
    limit: int = 10,
    offset: int = 0,
    dialogue_manager: PhilosophicalDialogueManager = Depends(get_dialogue_manager)
):
    """Get user's conversation history"""
    try:
        conversations = await dialogue_manager.get_user_conversations(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        return conversations
    except Exception as e:
        logger.error(f"Error retrieving conversations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving conversations")

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_id: str,
    dialogue_manager: PhilosophicalDialogueManager = Depends(get_dialogue_manager)
):
    """Delete a specific conversation"""
    try:
        await dialogue_manager.delete_conversation(
            conversation_id=conversation_id,
            user_id=user_id
        )
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting conversation")

async def run_real_time_evaluation(
    orchestrator: ComprehensiveEvaluationOrchestrator,
    request: ChatRequest,
    response: ChatResponse
):
    """Background task for real-time evaluation"""
    try:
        # Prepare evaluation data
        test_data = {
            "responses": {response.language: response.response},
            "expected_concepts": response.philosophical_concepts,
            "cultural_context": {response.language: request.cultural_context or {}},
            "interaction_data": [{
                "user_id": request.user_id,
                "message": request.message,
                "response": response.response,
                "language": response.language,
                "timestamp": "now",
                "processing_time": response.processing_time
            }]
        }
        
        # Run evaluation
        evaluation_results = await orchestrator.run_comprehensive_evaluation(test_data)
        
        # Store results for analytics
        await orchestrator.store_real_time_evaluation(
            user_id=request.user_id,
            conversation_id=response.conversation_id,
            evaluation_results=evaluation_results
        )
        
    except Exception as e:
        logger.error(f"Error in real-time evaluation: {e}")