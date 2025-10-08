from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Dict, Set, Optional
import json
import logging
import asyncio
from app.core.database import get_db
from app.models import media as dbm
from sqlalchemy.orm import Session
from app.core.celery_app import celery_app
from celery.result import AsyncResult
from jose import jwt, JWTError
import os
from dotenv import load_dotenv
from redis.asyncio import Redis
from app.core.redis_pool import _keepalive_opts

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["WebSocket"])
_bridge_task: asyncio.Task | None = None
_bridge_healthy: bool = False
_last_bridge_error: Optional[str] = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.task_subscriptions: Dict[str, Set[str]] = {}  # task_id -> set of user_ids

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        logger.info(f"WebSocket connected for user: {user_id}")

    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

        # Clean up all task subscriptions for this user to prevent memory leak
        tasks_to_clean = []
        for task_id, subscribers in self.task_subscriptions.items():
            if user_id in subscribers:
                subscribers.discard(user_id)
                if not subscribers:  # No more subscribers for this task
                    tasks_to_clean.append(task_id)

        # Remove empty task subscription entries
        for task_id in tasks_to_clean:
            del self.task_subscriptions[task_id]

        logger.info(f"WebSocket disconnected for user: {user_id}")

    def subscribe_to_task(self, task_id: str, user_id: str):
        if task_id not in self.task_subscriptions:
            self.task_subscriptions[task_id] = set()
        self.task_subscriptions[task_id].add(user_id)
        logger.info(f"User {user_id} subscribed to task {task_id}")

    def unsubscribe_from_task(self, task_id: str, user_id: str):
        if task_id in self.task_subscriptions:
            self.task_subscriptions[task_id].discard(user_id)
            if not self.task_subscriptions[task_id]:
                del self.task_subscriptions[task_id]

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_text(message)
                except:
                    disconnected.add(connection)

            # Clean up disconnected connections
            for conn in disconnected:
                self.active_connections[user_id].discard(conn)

    async def broadcast_task_update(self, task_id: str, update: dict):
        if task_id in self.task_subscriptions:
            message = json.dumps({
                "type": "task_update",
                "task_id": task_id,
                "update": update
            })
            for user_id in self.task_subscriptions[task_id]:
                await self.send_personal_message(message, user_id)


manager = ConnectionManager()


@router.websocket("/inference")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time inference updates.
    Client should send JWT token as query parameter.
    """
    user = None
    try:
        # Verify JWT token
        SECRET_KEY = os.getenv("SECRET_KEY", "dev-key")
        ALGORITHM = "HS256"

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if not username:
                await websocket.close(code=1008, reason="Invalid token")
                return

            from app.models.user import User
            user = db.query(User).filter(User.username == username).first()
            if not user:
                await websocket.close(code=1008, reason="User not found")
                return
        except JWTError:
            await websocket.close(code=1008, reason="Invalid token")
            return

        # Connect the client
        await manager.connect(websocket, user.username)

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    message_type = message.get("type")

                    if message_type == "subscribe":
                        # Subscribe to task updates
                        task_id = message.get("task_id")
                        if task_id:
                            # Verify user owns the task
                            media = db.query(dbm.Media).filter(
                                dbm.Media.task_id == task_id
                            ).first()
                            if media and (media.user_username == user.username or user.role == "admin"):
                                manager.subscribe_to_task(task_id, user.username)

                                # Send current status
                                result = AsyncResult(task_id, app=celery_app)
                                status_str = "processing" if result.state == "PROGRESS" else result.state
                                status_update = {
                                    "status": status_str,
                                    "result": result.result if result.state == "SUCCESS" else None,
                                    "error": str(result.info) if result.state == "FAILURE" else None,
                                }
                                if result.state == "PROGRESS" and isinstance(result.info, dict):
                                    status_update["progress"] = int(result.info.get("current", 0))
                                await websocket.send_text(json.dumps({
                                    "type": "task_update",
                                    "task_id": task_id,
                                    "update": status_update
                                }))
                            else:
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": "Access denied or task not found"
                                }))

                    elif message_type == "unsubscribe":
                        # Unsubscribe from task updates
                        task_id = message.get("task_id")
                        if task_id:
                            manager.unsubscribe_from_task(task_id, user.username)

                    elif message_type == "ping":
                        # Respond to ping with pong
                        await websocket.send_text(json.dumps({"type": "pong"}))

                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))

        except WebSocketDisconnect:
            manager.disconnect(websocket, user.username)
            logger.info(f"WebSocket disconnected for user: {user.username}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if user:
            manager.disconnect(websocket, user.username)


async def _redis_ws_bridge():
    """Subscribe to Redis pub/sub and relay messages to WebSocket clients with reconnection logic"""
    global _bridge_healthy, _last_bridge_error

    max_retries = -1  # Infinite retries
    base_delay = 1  # Start with 1 second
    max_delay = 60  # Cap at 60 seconds
    retry_count = 0

    while True:
        r = None
        pubsub = None

        try:
            # Create Redis connection
            r = Redis(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                socket_keepalive_options=_keepalive_opts(),
                retry_on_timeout=True,
            )

            # Test connection
            await r.ping()

            # Subscribe to channels
            pubsub = r.pubsub()
            await pubsub.psubscribe("task_updates:*")
            await pubsub.psubscribe("notifications:*")

            # Reset retry count on successful connection
            if retry_count > 0:
                logger.info(f"Redis WebSocket bridge reconnected after {retry_count} attempts")
            retry_count = 0
            _bridge_healthy = True
            _last_bridge_error = None

            # Listen for messages
            async for msg in pubsub.listen():
                if msg.get("type") not in ("message", "pmessage"):
                    continue
                try:
                    payload = json.loads(msg["data"])
                except Exception:
                    continue

                # Handle task updates
                task_id = payload.get("task_id")
                if task_id:
                    update = {
                        "status": payload.get("status"),
                        "error": payload.get("error"),
                        "progress": payload.get("progress"),
                    }
                    await manager.broadcast_task_update(task_id, update)
                    continue

                # Handle notification updates
                username = payload.get("username")
                if username and payload.get("type") == "notification":
                    notification_message = json.dumps({
                        "type": "notification",
                        "event_type": payload.get("event_type"),
                        "message": payload.get("message"),
                        "notification_id": payload.get("notification_id"),
                    })
                    await manager.send_personal_message(notification_message, username)

        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            logger.info("Redis WebSocket bridge task cancelled")
            break

        except (ConnectionError, TimeoutError, OSError) as e:
            retry_count += 1
            delay = min(base_delay * (2 ** min(retry_count - 1, 10)), max_delay)  # Exponential backoff with cap
            _bridge_healthy = False
            _last_bridge_error = str(e)
            logger.error(
                f"Redis connection failed (attempt {retry_count}): {e}. "
                f"Retrying in {delay} seconds..."
            )
            await asyncio.sleep(delay)

        except Exception as e:
            retry_count += 1
            delay = min(base_delay * (2 ** min(retry_count - 1, 10)), max_delay)
            logger.exception(
                f"Unexpected error in Redis WebSocket bridge (attempt {retry_count}): {e}. "
                f"Retrying in {delay} seconds..."
            )
            await asyncio.sleep(delay)

        finally:
            # Clean up connections
            try:
                if pubsub:
                    await pubsub.close()
            except Exception:
                pass
            try:
                if r:
                    await r.close()
            except Exception:
                pass


async def start_ws_redis_bridge():
    """Start the Redis-WebSocket bridge task"""
    global _bridge_task
    if _bridge_task is None or _bridge_task.done():
        _bridge_task = asyncio.create_task(_redis_ws_bridge())
        logger.info("Started Redis-WebSocket bridge")


async def stop_ws_redis_bridge():
    """Stop the Redis-WebSocket bridge task"""
    global _bridge_task, _bridge_healthy
    if _bridge_task:
        _bridge_task.cancel()
        try:
            await _bridge_task
        except asyncio.CancelledError:
            pass
        _bridge_task = None
        _bridge_healthy = False
        logger.info("Stopped Redis-WebSocket bridge")


def get_bridge_health() -> dict:
    """Get the health status of the Redis-WebSocket bridge"""
    return {
        "healthy": _bridge_healthy,
        "task_running": _bridge_task is not None and not _bridge_task.done(),
        "last_error": _last_bridge_error,
    }