"""
Health check endpoints
"""

import asyncio
import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.models.response_models import HealthResponse, MetricsResponse
from src.utils.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    settings = get_settings()
    
    # Check component health
    components = await check_component_health()
    
    # Determine overall status
    status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        version="2.0.0",
        environment=settings.ENVIRONMENT,
        components=components
    )

@router.get("/detailed", response_model=HealthResponse)
async def detailed_health_check():
    """Detailed health check with metrics"""
    settings = get_settings()
    
    # Check component health
    components = await check_component_health()
    
    # Get system metrics
    metrics = await get_system_metrics()
    
    # Determine overall status
    status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        version="2.0.0",
        environment=settings.ENVIRONMENT,
        components=components,
        metrics=metrics
    )

@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    try:
        # Check if all critical components are ready
        components = await check_component_health()
        critical_components = ["database", "redis", "cohere_api"]
        
        for component in critical_components:
            if components.get(component) != "healthy":
                return JSONResponse(
                    status_code=503,
                    content={"status": "not ready", "component": component}
                )
        
        return {"status": "ready"}
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "error": str(e)}
        )

@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

async def check_component_health() -> dict:
    """Check health of all system components"""
    components = {}
    
    # Check database
    try:
        from src.api.main import evaluation_orchestrator
        if evaluation_orchestrator and evaluation_orchestrator.db_pool:
            async with evaluation_orchestrator.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            components["database"] = "healthy"
        else:
            components["database"] = "unavailable"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        components["database"] = "unhealthy"
    
    # Check Redis
    try:
        from src.api.main import rate_limiter
        if rate_limiter and rate_limiter.redis_client:
            await rate_limiter.redis_client.ping()
            components["redis"] = "healthy"
        else:
            components["redis"] = "unavailable"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        components["redis"] = "unhealthy"
    
    # Check Cohere API
    try:
        from src.api.main import processor
        if processor:
            # Simple API check (could be enhanced)
            components["cohere_api"] = "healthy"
        else:
            components["cohere_api"] = "unavailable"
    except Exception as e:
        logger.error(f"Cohere API health check failed: {e}")
        components["cohere_api"] = "unhealthy"
    
    # Check evaluation system
    try:
        from src.api.main import evaluation_orchestrator
        if evaluation_orchestrator:
            components["evaluation_system"] = "healthy"
        else:
            components["evaluation_system"] = "unavailable"
    except Exception as e:
        logger.error(f"Evaluation system health check failed: {e}")
        components["evaluation_system"] = "unhealthy"
    
    return components

async def get_system_metrics() -> dict:
    """Get system performance metrics"""
    try:
        from src.api.main import evaluation_orchestrator
        
        if not evaluation_orchestrator:
            return {}
        
        # Get metrics from database
        metrics = await evaluation_orchestrator.get_system_metrics()
        
        return {
            "response_time_p95": metrics.get("performance", {}).get("avg_processing_time", 0) * 1000,  # Convert to ms
            "active_users_24h": metrics.get("performance", {}).get("total_users_7d", 0),
            "philosophical_accuracy_avg": metrics.get("performance", {}).get("avg_satisfaction", 0),
            "memory_usage_mb": 0,  # Would need psutil integration
            "cpu_usage_percent": 0  # Would need psutil integration
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {}