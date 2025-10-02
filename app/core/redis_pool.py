"""
Redis connection pool with circuit breaker pattern for resilient connections
"""
import redis
from redis.exceptions import RedisError
import time
import os
import logging
from typing import Optional
from threading import Lock
import socket

logger = logging.getLogger(__name__)

def _keepalive_opts():
    opts = {}
    # Linux
    if hasattr(socket, "TCP_KEEPIDLE"):
        opts[socket.TCP_KEEPIDLE] = int(os.getenv("REDIS_TCP_KEEPIDLE", "60"))
    if hasattr(socket, "TCP_KEEPINTVL"):
        opts[socket.TCP_KEEPINTVL] = int(os.getenv("REDIS_TCP_KEEPINTVL", "30"))
    if hasattr(socket, "TCP_KEEPCNT"):
        opts[socket.TCP_KEEPCNT] = int(os.getenv("REDIS_TCP_KEEPCNT", "5"))
    # macOS (no KEEPIDLE/INTVL/KEEPCNT)
    if not opts and hasattr(socket, "TCP_KEEPALIVE"):
        opts[socket.TCP_KEEPALIVE] = int(os.getenv("REDIS_TCP_KEEPALIVE", "60"))
    return opts or None

class RedisConnectionPool:
    """
    Redis connection pool with circuit breaker pattern to handle failures gracefully
    """
    def __init__(self):
        self.pool = None
        self.last_failure = None
        self.failure_count = 0
        self.circuit_open = False
        self.lock = Lock()
        self.circuit_open_duration = 30  # seconds
        self.max_failures = 3

    def get_connection(self) -> Optional[redis.Redis]:
        """
        Get a Redis connection with circuit breaker protection

        Returns:
            Redis connection or None if circuit is open
        """
        with self.lock:
            # Check if circuit breaker is open
            if self.circuit_open:
                if time.time() - self.last_failure > self.circuit_open_duration:
                    # Try to close the circuit
                    self.circuit_open = False
                    self.failure_count = 0
                    logger.info("Circuit breaker closed, retrying Redis connection")
                else:
                    logger.warning("Circuit breaker is open, Redis unavailable")
                    return None

            try:
                if not self.pool:
                    self.pool = redis.ConnectionPool(
                        host=os.getenv("REDIS_HOST", "redis"),
                        port=int(os.getenv("REDIS_PORT", "6379")),
                        db=int(os.getenv("REDIS_DB", "0")),
                        password=os.getenv("REDIS_PASSWORD"),
                        max_connections=50,
                        socket_keepalive=True,
                        socket_keepalive_options=_keepalive_opts(),
                        socket_connect_timeout=5,
                        socket_timeout=5,
                        retry_on_timeout=True,
                        health_check_interval=30,
                    )

                # Test the connection
                conn = redis.Redis(connection_pool=self.pool)
                conn.ping()

                # Reset failure count on success
                self.failure_count = 0
                return conn

            except RedisError as e:
                self.failure_count += 1
                self.last_failure = time.time()

                logger.error(f"Redis connection failed (attempt {self.failure_count}): {e}")

                if self.failure_count >= self.max_failures:
                    self.circuit_open = True
                    logger.error(f"Circuit breaker opened after {self.max_failures} failures")

                return None
            except Exception as e:
                logger.error(f"Unexpected error in Redis connection: {e}")
                return None

    def close(self):
        """Close the connection pool"""
        if self.pool:
            self.pool.disconnect()
            self.pool = None


# Global singleton instance
redis_pool = RedisConnectionPool()


def get_redis() -> Optional[redis.Redis]:
    """
    Get a Redis connection from the pool

    Returns:
        Redis connection or None if unavailable
    """
    return redis_pool.get_connection()


def send_websocket_update_safe(task_id: str, status: str, error_message: str = None, progress: int = None):
    """
    Send WebSocket update via Redis pub/sub with circuit breaker protection

    Args:
        task_id: Celery task ID
        status: Task status
        error_message: Optional error message
        progress: Optional progress percentage
    """
    import json

    r = get_redis()
    if not r:
        logger.warning(f"Cannot send WebSocket update, Redis unavailable (task_id={task_id})")
        return

    try:
        message = {
            "task_id": task_id,
            "status": status,
            "error": error_message
        }
        if progress is not None:
            message["progress"] = progress

        r.publish(f"task_updates:{task_id}", json.dumps(message))

    except RedisError as e:
        logger.warning(f"Failed to send WebSocket update via Redis: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending WebSocket update: {e}")


# Progress throttling to prevent spam
class ProgressThrottler:
    """
    Throttle progress updates to prevent overwhelming WebSocket clients
    """
    def __init__(self, min_delta: int = 5, min_interval: float = 0.5):
        self.min_delta = min_delta  # Minimum percentage change
        self.min_interval = min_interval  # Minimum time between updates
        self.last_progress = {}
        self.last_time = {}
        self.lock = Lock()

    def should_send(self, task_id: str, progress: int) -> bool:
        """
        Check if progress update should be sent

        Args:
            task_id: Task identifier
            progress: Current progress percentage

        Returns:
            True if update should be sent
        """
        with self.lock:
            now = time.time()

            # Always send 0%, 100%
            if progress in (0, 100):
                self.last_progress[task_id] = progress
                self.last_time[task_id] = now
                return True

            # Check time and delta constraints
            last_p = self.last_progress.get(task_id, -10)
            last_t = self.last_time.get(task_id, 0)

            if (progress - last_p >= self.min_delta and
                now - last_t >= self.min_interval):
                self.last_progress[task_id] = progress
                self.last_time[task_id] = now
                return True

            return False

    def cleanup(self, task_id: str):
        """Remove task from tracking"""
        with self.lock:
            self.last_progress.pop(task_id, None)
            self.last_time.pop(task_id, None)


# Global progress throttler
progress_throttler = ProgressThrottler()