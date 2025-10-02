"""
Rate limiting middleware for API endpoints using Redis
"""
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request
import redis.asyncio as aioredis
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Create Redis-backed limiter for distributed rate limiting
def create_limiter() -> Limiter:
    """Create a Limiter instance with Redis storage"""
    redis_url = f"redis://:{os.getenv('REDIS_PASSWORD', '')}@{os.getenv('REDIS_HOST', 'redis')}:{os.getenv('REDIS_PORT', '6379')}/2"

    # Initialize limiter with Redis storage
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=redis_url,
        strategy="fixed-window",
        default_limits=["100 per minute"],  # Global default limit
        headers_enabled=True,  # Add rate limit headers to responses
    )

    return limiter

# Create global limiter instance
limiter = create_limiter()

# Specific rate limits for different endpoint types
INFERENCE_RATE_LIMITS = {
    "image": "10 per minute",  # Images: 10 requests per minute
    "video": "3 per minute",   # Videos: 3 requests per minute (more resource intensive)
    "status": "60 per minute",  # Status checks: 60 per minute
    "media_list": "30 per minute",  # Media list: 30 per minute
}

# User-based rate limits (for authenticated endpoints)
USER_RATE_LIMITS = {
    "default": "100 per hour",  # Default user limit
    "admin": "500 per hour",    # Admin users get higher limits
    "premium": "250 per hour",  # Premium users (future feature)
}

def get_user_key_func(request: Request) -> str:
    """Get rate limit key based on authenticated user"""
    # Try to get user from request state (set by authentication middleware)
    user = getattr(request.state, "user", None)
    if user and hasattr(user, "username"):
        return f"user:{user.username}"
    # Fall back to IP address
    return get_remote_address(request)

def get_combined_key_func(request: Request) -> str:
    """Get rate limit key combining user and IP for stricter limits"""
    user = getattr(request.state, "user", None)
    ip = get_remote_address(request)
    if user and hasattr(user, "username"):
        return f"user:{user.username}:ip:{ip}"
    return f"ip:{ip}"

# Create specialized limiters for different scenarios
user_limiter = Limiter(
    key_func=get_user_key_func,
    storage_uri=f"redis://:{os.getenv('REDIS_PASSWORD', '')}@{os.getenv('REDIS_HOST', 'redis')}:{os.getenv('REDIS_PORT', '6379')}/2",
    strategy="fixed-window",
    headers_enabled=True,
)

combined_limiter = Limiter(
    key_func=get_combined_key_func,
    storage_uri=f"redis://:{os.getenv('REDIS_PASSWORD', '')}@{os.getenv('REDIS_HOST', 'redis')}:{os.getenv('REDIS_PORT', '6379')}/2",
    strategy="fixed-window",
    headers_enabled=True,
)

async def check_rate_limit_health() -> bool:
    """Check if Redis rate limiting is working"""
    try:
        redis = aioredis.from_url(
            f"redis://:{os.getenv('REDIS_PASSWORD', '')}@{os.getenv('REDIS_HOST', 'redis')}:{os.getenv('REDIS_PORT', '6379')}/2",
            decode_responses=True
        )
        await redis.ping()
        await redis.close()
        return True
    except Exception as e:
        logger.error(f"Rate limiter Redis health check failed: {e}")
        return False