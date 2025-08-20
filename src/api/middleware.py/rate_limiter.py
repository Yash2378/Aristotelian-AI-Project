"""
Rate limiting middleware using Redis
"""

import asyncio
import logging
import time
from typing import Optional

import redis.asyncio as redis
from fastapi import HTTPException

from src.utils.exceptions import RateLimitException

logger = logging.getLogger(__name__)

class RateLimiter:
    """Redis-based rate limiter for API endpoints"""
    
    def __init__(self, redis_url: str, default_limit: int = 1000):
        self.redis_url = redis_url
        self.default_limit = default_limit
        self.redis_client: Optional[redis.Redis] = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Rate limiter Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis for rate limiting: {e}")
            # Continue without rate limiting if Redis is unavailable
            self.redis_client = None
    
    async def cleanup(self):
        """Cleanup Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def check_rate_limit(
        self, 
        user_id: str, 
        limit: Optional[int] = None,
        window_seconds: int = 3600
    ) -> bool:
        """
        Check if user is within rate limits
        
        Args:
            user_id: User identifier
            limit: Request limit (uses default if None)
            window_seconds: Time window in seconds (default: 1 hour)
            
        Returns:
            True if within limits, raises RateLimitException if exceeded
        """
        if not self.redis_client:
            # No rate limiting if Redis unavailable
            return True
        
        limit = limit or self.default_limit
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        try:
            # Use sliding window rate limiting
            key = f"rate_limit:{user_id}"
            
            # Remove expired entries
            await self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            current_requests = await self.redis_client.zcard(key)
            
            if current_requests >= limit:
                raise RateLimitException(user_id, current_requests, limit)
            
            # Add current request
            await self.redis_client.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            await self.redis_client.expire(key, window_seconds)
            
            return True
            
        except RateLimitException:
            raise
        except Exception as e:
            logger.error(f"Rate limiting error for user {user_id}: {e}")
            # Allow request if rate limiting fails
            return True

# Dependency function for FastAPI
async def check_rate_limit(user_id: str):
    """FastAPI dependency for rate limiting"""
    from src.api.main import rate_limiter
    if rate_limiter:
        await rate_limiter.check_rate_limit(user_id)