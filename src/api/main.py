"""
FastAPI Application - Multilingual Aristotelian AI
Production-ready API with comprehensive evaluation and monitoring
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.routes import chat, evaluation, health
from src.api.middleware.rate_limiter import RateLimiter
from src.api.middleware.auth import get_current_user
from src.core.multilingual_processor import MultilingualPhilosophicalProcessor
from src.core.philosophical_dialogue import PhilosophicalDialogueManager
from src.evaluation.orchestrator import ComprehensiveEvaluationOrchestrator
from src.utils.config import get_settings
from src.utils.logging import setup_logging
from src.utils.exceptions import AristotelianAIException
from src.database.connection import get_database_pool

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global instances
processor: MultilingualPhilosophicalProcessor = None
dialogue_manager: PhilosophicalDialogueManager = None
evaluation_orchestrator: ComprehensiveEvaluationOrchestrator = None
rate_limiter: RateLimiter = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Multilingual Aristotelian AI System...")
    
    settings = get_settings()
    
    # Initialize global instances
    global processor, dialogue_manager, evaluation_orchestrator, rate_limiter
    
    try:
        # Initialize multilingual processor
        processor = MultilingualPhilosophicalProcessor(settings.dict())
        await processor.initialize()
        logger.info("Multilingual processor initialized")
        
        # Initialize dialogue manager
        dialogue_manager = PhilosophicalDialogueManager(
            processor=processor,
            model_id=settings.FINE_TUNED_MODEL_ID
        )
        logger.info("Dialogue manager initialized")
        
        # Initialize evaluation orchestrator
        evaluation_orchestrator = ComprehensiveEvaluationOrchestrator(settings.dict())
        await evaluation_orchestrator.initialize(settings.DATABASE_URL)
        logger.info("Evaluation orchestrator initialized")
        
        # Initialize rate limiter
        rate_limiter = RateLimiter(
            redis_url=settings.REDIS_URL,
            default_limit=settings.RATE_LIMIT_REQUESTS_PER_HOUR
        )
        await rate_limiter.initialize()
        logger.info("Rate limiter initialized")
        
        logger.info("All systems initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multilingual Aristotelian AI System...")
    
    if processor:
        await processor.cleanup()
    if evaluation_orchestrator:
        await evaluation_orchestrator.cleanup()
    if rate_limiter:
        await rate_limiter.cleanup()
    
    logger.info("System shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Multilingual Aristotelian AI",
    description="Advanced multilingual conversational AI for philosophical dialogue",
    version="2.0.0",
    docs_url="/docs" if get_settings().ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if get_settings().ENVIRONMENT != "production" else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add Prometheus monitoring
if get_settings().PROMETHEUS_ENABLED:
    instrumentator = Instrumentator()
    instrumentator.instrument(app)
    instrumentator.expose(app, endpoint="/metrics")

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(AristotelianAIException)
async def aristotelian_ai_exception_handler(request: Request, exc: AristotelianAIException):
    logger.error(f"AristotelianAI exception: {exc}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_type,
            "message": exc.message,
            "details": exc.details
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An internal error occurred"
        }
    )

# Dependency to get global instances
async def get_processor() -> MultilingualPhilosophicalProcessor:
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    return processor

async def get_dialogue_manager() -> PhilosophicalDialogueManager:
    if dialogue_manager is None:
        raise HTTPException(status_code=503, detail="Dialogue manager not initialized")
    return dialogue_manager

async def get_evaluation_orchestrator() -> ComprehensiveEvaluationOrchestrator:
    if evaluation_orchestrator is None:
        raise HTTPException(status_code=503, detail="Evaluation orchestrator not initialized")
    return evaluation_orchestrator

async def get_rate_limiter() -> RateLimiter:
    if rate_limiter is None:
        raise HTTPException(status_code=503, detail="Rate limiter not initialized")
    return rate_limiter

# Include routers
app.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

app.include_router(
    chat.router,
    prefix="/chat",
    tags=["chat"],
    dependencies=[Depends(get_rate_limiter)]
)

app.include_router(
    evaluation.router,
    prefix="/evaluation",
    tags=["evaluation"],
    dependencies=[Depends(get_current_user)]
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    settings = get_settings()
    return {
        "name": "Multilingual Aristotelian AI",
        "version": "2.0.0",
        "description": "Advanced multilingual conversational AI for philosophical dialogue",
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "features": {
            "philosophical_accuracy_evaluation": True,
            "cultural_appropriateness_adaptation": True,
            "cross_lingual_consistency": True,
            "educational_effectiveness_tracking": True,
            "aya_integration": settings.AYA_INTEGRATION_ENABLED,
            "expert_validation": settings.EXPERT_VALIDATION_ENABLED
        },
        "status": "operational",
        "environment": settings.ENVIRONMENT
    }

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS if settings.ENVIRONMENT == "production" else 1,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )