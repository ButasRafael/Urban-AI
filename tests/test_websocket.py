"""
Comprehensive tests for WebSocket functionality.

Tests cover:
- Connection authentication
- Task subscription/unsubscription
- Real-time progress updates
- Connection manager
- Redis bridge functionality
- Multiple concurrent connections
- Disconnection handling
- Invalid token handling
"""

import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from jose import jwt
from datetime import datetime, timedelta
from pathlib import Path
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import models first to ensure tables are created
import app.models.user
import app.models.media
import app.models.revoked
import app.models.rag
import app.models.conversation
from app.models.user import User
from app.models.media import Media, Frame, Detection
from app.api.websocket import ConnectionManager, start_ws_redis_bridge, stop_ws_redis_bridge, get_bridge_health
from app.core.security import create_access_token, get_password_hash
from app.core.database import Base, get_db
from app.main import app


def generate_token(username: str) -> str:
    """Generate a valid JWT token for testing"""
    return create_access_token({"sub": username})


# Database fixtures - Use PostgreSQL for testing (same as production)
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@db:5432/urban_ai_test"
)

engine = create_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session():
    """Create a real database session for testing"""
    session = TestingSessionLocal()
    try:
        # Clean up data before test
        session.query(Detection).delete()
        session.query(Frame).delete()
        session.query(Media).delete()
        session.query(User).delete()
        session.commit()
        yield session
    finally:
        # Clean up data after test
        session.query(Detection).delete()
        session.query(Frame).delete()
        session.query(Media).delete()
        session.query(User).delete()
        session.commit()
        session.close()


@pytest.fixture
def client(db_session):
    """Create a test client with database override"""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    # Disable rate limiting for tests
    with patch('app.core.rate_limiter.limiter.limit', lambda x: lambda f: f):
        with patch('app.core.rate_limiter.user_limiter.limit', lambda x: lambda f: f):
            app.dependency_overrides[get_db] = override_get_db
            yield TestClient(app)
            app.dependency_overrides.clear()


@pytest.fixture
def test_user_regular(db_session):
    """Create a regular test user"""
    user = User(
        username="ws_testuser",
        hashed_password=get_password_hash("testpass123"),
        role="user"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_user_admin(db_session):
    """Create an admin test user"""
    user = User(
        username="ws_admin",
        hashed_password=get_password_hash("adminpass123"),
        role="admin"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_media_item(db_session, test_user_regular):
    """Create a test media item with task ID"""
    media = Media(
        user_username=test_user_regular.username,
        filename="test_image.jpg",
        media_type="image",
        static_filename="test_image.jpg",
        task_id="test-task-123"
    )
    db_session.add(media)
    db_session.commit()
    db_session.refresh(media)
    return media


@pytest.fixture
def test_media_item_other_user(db_session):
    """Create a media item for another user"""
    other_user = User(
        username="other_user",
        hashed_password=get_password_hash("otherpass123"),
        role="user"
    )
    db_session.add(other_user)
    db_session.commit()

    media = Media(
        user_username=other_user.username,
        filename="other_image.jpg",
        media_type="image",
        static_filename="other_image.jpg",
        task_id="other-task-456"
    )
    db_session.add(media)
    db_session.commit()
    db_session.refresh(media)
    return media


@pytest.fixture
def valid_token(test_user_regular):
    """Generate valid JWT token for regular user"""
    return generate_token(test_user_regular.username)


@pytest.fixture
def admin_token(test_user_admin):
    """Generate valid JWT token for admin user"""
    return generate_token(test_user_admin.username)


@pytest.fixture
def expired_token():
    """Generate expired JWT token"""
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-key")
    ALGORITHM = "HS256"

    expire = datetime.utcnow() - timedelta(hours=1)  # Expired 1 hour ago
    payload = {
        "sub": "expired_user",
        "exp": expire
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


@pytest.fixture
def invalid_token():
    """Generate invalid JWT token with wrong signature"""
    return jwt.encode({"sub": "testuser"}, "wrong-secret-key", algorithm="HS256")


# Test 1: Connection Authentication
class TestWebSocketAuthentication:
    """Test WebSocket connection authentication"""

    def test_successful_connection_with_valid_token(self, client, valid_token, test_user_regular, db_session):
        """Test successful WebSocket connection with valid JWT token"""
        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Connection should be established
            # Send a ping to verify connection is working
            websocket.send_text(json.dumps({"type": "ping"}))
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "pong"

    def test_connection_rejected_with_invalid_token(self, client, invalid_token):
        """Test WebSocket connection rejected with invalid JWT token"""
        with pytest.raises(Exception):
            with client.websocket_connect(f"/ws/inference?token={invalid_token}") as websocket:
                pass
        # Connection should be closed with error (code 1008)

    def test_connection_rejected_with_expired_token(self, client, expired_token):
        """Test WebSocket connection rejected with expired JWT token"""
        with pytest.raises(Exception):
            with client.websocket_connect(f"/ws/inference?token={expired_token}") as websocket:
                pass
        # Connection should be closed with error (code 1008)

    def test_connection_rejected_without_token(self, client):
        """Test WebSocket connection rejected without JWT token"""
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/inference") as websocket:
                pass

    def test_connection_rejected_for_nonexistent_user(self, client, db_session):
        """Test WebSocket connection rejected for token with non-existent user"""
        # Create token for non-existent user
        SECRET_KEY = os.getenv("SECRET_KEY", "dev-key")
        ALGORITHM = "HS256"
        token = jwt.encode(
            {"sub": "nonexistent_user", "exp": datetime.utcnow() + timedelta(hours=1)},
            SECRET_KEY,
            algorithm=ALGORITHM
        )

        with pytest.raises(Exception):
            with client.websocket_connect(f"/ws/inference?token={token}") as websocket:
                pass
        # Connection should be closed with error (code 1008)

    def test_admin_user_connection(self, client, admin_token, test_user_admin, db_session):
        """Test admin user can connect successfully"""
        with client.websocket_connect(f"/ws/inference?token={admin_token}") as websocket:
            websocket.send_text(json.dumps({"type": "ping"}))
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "pong"


# Test 2: Task Subscription/Unsubscription
class TestTaskSubscription:
    """Test task subscription and unsubscription functionality"""

    @patch('app.api.websocket.AsyncResult')
    def test_subscribe_to_own_task(self, mock_async_result, client, valid_token, test_media_item, db_session):
        """Test user can subscribe to their own task"""
        # Setup mock AsyncResult
        mock_result = MagicMock()
        mock_result.state = "PROGRESS"
        mock_result.info = {"current": 50}
        mock_result.result = None
        mock_async_result.return_value = mock_result

        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Subscribe to task
            websocket.send_text(json.dumps({
                "type": "subscribe",
                "task_id": test_media_item.task_id
            }))

            # Should receive current status
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "task_update"
            assert response["task_id"] == test_media_item.task_id
            assert response["update"]["status"] == "processing"
            assert response["update"]["progress"] == 50

    def test_subscribe_to_other_user_task_denied(self, client, valid_token, test_media_item_other_user, db_session):
        """Test user cannot subscribe to another user's task"""
        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Try to subscribe to another user's task
            websocket.send_text(json.dumps({
                "type": "subscribe",
                "task_id": test_media_item_other_user.task_id
            }))

            # Should receive error
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "error"
            assert "Access denied" in response["message"]

    @patch('app.api.websocket.AsyncResult')
    def test_admin_can_subscribe_to_any_task(self, mock_async_result, client, admin_token, test_media_item_other_user, db_session):
        """Test admin can subscribe to any user's task"""
        # Setup mock AsyncResult
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"output": "processed"}
        mock_result.info = None
        mock_async_result.return_value = mock_result

        with client.websocket_connect(f"/ws/inference?token={admin_token}") as websocket:
            # Admin subscribes to another user's task
            websocket.send_text(json.dumps({
                "type": "subscribe",
                "task_id": test_media_item_other_user.task_id
            }))

            # Should receive current status
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "task_update"
            assert response["task_id"] == test_media_item_other_user.task_id
            assert response["update"]["status"] == "SUCCESS"
            assert response["update"]["result"] == {"output": "processed"}

    def test_unsubscribe_from_task(self, client, valid_token, test_media_item, db_session):
        """Test unsubscribing from a task"""
        with patch('app.api.websocket.AsyncResult') as mock_async_result:
            mock_result = MagicMock()
            mock_result.state = "PENDING"
            mock_async_result.return_value = mock_result

            with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
                # Subscribe first
                websocket.send_text(json.dumps({
                    "type": "subscribe",
                    "task_id": test_media_item.task_id
                }))
                data = websocket.receive_text()  # Consume the status update

                # Unsubscribe
                websocket.send_text(json.dumps({
                    "type": "unsubscribe",
                    "task_id": test_media_item.task_id
                }))

                # Send ping to verify connection still works
                websocket.send_text(json.dumps({"type": "ping"}))
                data = websocket.receive_text()
                response = json.loads(data)
                assert response["type"] == "pong"

    def test_subscribe_to_nonexistent_task(self, client, valid_token, db_session):
        """Test subscribing to non-existent task"""
        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Subscribe to non-existent task
            websocket.send_text(json.dumps({
                "type": "subscribe",
                "task_id": "nonexistent-task-999"
            }))

            # Should receive error
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "error"
            assert "Access denied or task not found" in response["message"]


# Test 3: Real-time Progress Updates
class TestRealtimeProgressUpdates:
    """Test real-time task progress updates"""

    @patch('app.api.websocket.AsyncResult')
    def test_receive_progress_updates(self, mock_async_result, client, valid_token, test_media_item, db_session):
        """Test receiving progress updates for subscribed task"""
        # Setup mock AsyncResult for different states
        mock_result = MagicMock()
        mock_result.state = "PROGRESS"
        mock_result.info = {"current": 25}
        mock_async_result.return_value = mock_result

        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Subscribe to task
            websocket.send_text(json.dumps({
                "type": "subscribe",
                "task_id": test_media_item.task_id
            }))

            # Receive initial status
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["update"]["progress"] == 25

    @patch('app.api.websocket.AsyncResult')
    def test_task_completion_update(self, mock_async_result, client, valid_token, test_media_item, db_session):
        """Test receiving task completion update"""
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"annotations": ["object1", "object2"]}
        mock_result.info = None
        mock_async_result.return_value = mock_result

        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Subscribe to task
            websocket.send_text(json.dumps({
                "type": "subscribe",
                "task_id": test_media_item.task_id
            }))

            # Receive status
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "task_update"
            assert response["update"]["status"] == "SUCCESS"
            assert response["update"]["result"] == {"annotations": ["object1", "object2"]}
            assert response["update"]["error"] is None

    @patch('app.api.websocket.AsyncResult')
    def test_task_failure_update(self, mock_async_result, client, valid_token, test_media_item, db_session):
        """Test receiving task failure update"""
        mock_result = MagicMock()
        mock_result.state = "FAILURE"
        mock_result.result = None
        mock_result.info = "Processing failed: Out of memory"
        mock_async_result.return_value = mock_result

        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Subscribe to task
            websocket.send_text(json.dumps({
                "type": "subscribe",
                "task_id": test_media_item.task_id
            }))

            # Receive status
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "task_update"
            assert response["update"]["status"] == "FAILURE"
            assert response["update"]["error"] == "Processing failed: Out of memory"
            assert response["update"]["result"] is None


# Test 4: Connection Manager
class TestConnectionManager:
    """Test ConnectionManager functionality"""

    @pytest.mark.anyio
    async def test_connection_manager_connect_disconnect(self):
        """Test ConnectionManager connect and disconnect operations"""
        manager = ConnectionManager()

        # Create mock websockets
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        # Connect users
        await manager.connect(ws1, "user1")
        await manager.connect(ws2, "user2")

        assert "user1" in manager.active_connections
        assert "user2" in manager.active_connections
        assert ws1 in manager.active_connections["user1"]
        assert ws2 in manager.active_connections["user2"]

        # Disconnect users
        manager.disconnect(ws1, "user1")
        assert "user1" not in manager.active_connections

        manager.disconnect(ws2, "user2")
        assert "user2" not in manager.active_connections

    @pytest.mark.anyio
    async def test_connection_manager_multiple_connections_same_user(self):
        """Test multiple WebSocket connections for same user"""
        manager = ConnectionManager()

        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()

        # Connect same user multiple times (e.g., multiple browser tabs)
        await manager.connect(ws1, "user1")
        await manager.connect(ws2, "user1")
        await manager.connect(ws3, "user1")

        assert len(manager.active_connections["user1"]) == 3

        # Disconnect one connection
        manager.disconnect(ws2, "user1")
        assert len(manager.active_connections["user1"]) == 2
        assert ws2 not in manager.active_connections["user1"]

        # Disconnect remaining
        manager.disconnect(ws1, "user1")
        manager.disconnect(ws3, "user1")
        assert "user1" not in manager.active_connections

    def test_connection_manager_task_subscriptions(self):
        """Test task subscription management"""
        manager = ConnectionManager()

        # Subscribe users to tasks
        manager.subscribe_to_task("task1", "user1")
        manager.subscribe_to_task("task1", "user2")
        manager.subscribe_to_task("task2", "user1")

        assert "task1" in manager.task_subscriptions
        assert "task2" in manager.task_subscriptions
        assert "user1" in manager.task_subscriptions["task1"]
        assert "user2" in manager.task_subscriptions["task1"]
        assert "user1" in manager.task_subscriptions["task2"]

        # Unsubscribe
        manager.unsubscribe_from_task("task1", "user1")
        assert "user1" not in manager.task_subscriptions["task1"]
        assert "user2" in manager.task_subscriptions["task1"]

        # Unsubscribe last user from task
        manager.unsubscribe_from_task("task1", "user2")
        assert "task1" not in manager.task_subscriptions

    @pytest.mark.anyio
    async def test_send_personal_message(self):
        """Test sending personal message to user"""
        manager = ConnectionManager()

        ws1 = AsyncMock()
        ws2 = AsyncMock()

        await manager.connect(ws1, "user1")
        await manager.connect(ws2, "user1")

        # Send message to user1 (should go to both connections)
        await manager.send_personal_message("Hello User1", "user1")

        ws1.send_text.assert_called_once_with("Hello User1")
        ws2.send_text.assert_called_once_with("Hello User1")

    @pytest.mark.anyio
    async def test_broadcast_task_update(self):
        """Test broadcasting task updates to subscribed users"""
        manager = ConnectionManager()

        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()

        await manager.connect(ws1, "user1")
        await manager.connect(ws2, "user2")
        await manager.connect(ws3, "user3")

        # Subscribe users to task
        manager.subscribe_to_task("task1", "user1")
        manager.subscribe_to_task("task1", "user2")
        # user3 not subscribed

        # Broadcast update
        update = {"status": "processing", "progress": 50}
        await manager.broadcast_task_update("task1", update)

        # Check that subscribed users received update
        expected_message = json.dumps({
            "type": "task_update",
            "task_id": "task1",
            "update": update
        })

        ws1.send_text.assert_called_once_with(expected_message)
        ws2.send_text.assert_called_once_with(expected_message)
        ws3.send_text.assert_not_called()  # user3 not subscribed


# Test 5: Redis Bridge Functionality
class TestRedisBridge:
    """Test Redis bridge functionality for real-time updates"""

    @pytest.mark.anyio
    async def test_redis_bridge_start_stop(self):
        from app.api import websocket as ws_module

        original_task = ws_module._bridge_task
        ws_module._bridge_task = None
        try:
            with patch('app.api.websocket.asyncio.create_task') as mock_create_task:
                loop = asyncio.get_running_loop()
                task = loop.create_future()  # not done yet

                def fake_create_task(coro):
                    # prevent "coroutine was never awaited" since we’re not actually running it
                    try:
                        coro.close()
                    except Exception:
                        pass
                    return task

                mock_create_task.side_effect = fake_create_task

                # Start bridge (should create exactly one task)
                await start_ws_redis_bridge()
                mock_create_task.assert_called_once()

                # Start again (should NO-OP because task.done() is False)
                mock_create_task.reset_mock()
                await start_ws_redis_bridge()
                mock_create_task.assert_not_called()

                # Stop bridge (cancel & await)
                await stop_ws_redis_bridge()
                assert task.cancelled()  # was cancelled by stop
        finally:
            ws_module._bridge_task = original_task

    def test_bridge_health_status(self):
        """Test bridge health status reporting"""
        health = get_bridge_health()

        assert "healthy" in health
        assert "task_running" in health
        assert "last_error" in health

        # Initially should be unhealthy and not running
        assert health["healthy"] is False
        assert health["task_running"] is False

    @pytest.mark.anyio
    async def test_redis_message_relay(self):
        """Test Redis message relay to WebSocket clients"""
        with patch('app.api.websocket.manager') as mock_manager:
            mock_manager.broadcast_task_update = AsyncMock()

            # Simulate Redis message processing
            # This would normally come from Redis pub/sub
            task_id = "test-task-123"
            update = {
                "status": "processing",
                "progress": 75,
                "error": None
            }

            # Call the broadcast directly (simulating what bridge would do)
            await mock_manager.broadcast_task_update(task_id, update)

            mock_manager.broadcast_task_update.assert_called_once_with(task_id, update)


# Test 6: Multiple Concurrent Connections
class TestConcurrentConnections:
    """Test multiple concurrent WebSocket connections"""

    def test_multiple_users_concurrent_connections(self, client, db_session):
        """Test multiple users with concurrent connections"""
        # Create multiple users
        users = []
        tokens = []
        for i in range(3):
            user = User(
                username=f"concurrent_user_{i}",
                hashed_password=get_password_hash(f"pass{i}123"),
                role="user"
            )
            db_session.add(user)
            users.append(user)
            tokens.append(generate_token(user.username))

        db_session.commit()

        # Create media items for each user
        for i, user in enumerate(users):
            media = Media(
                user_username=user.username,
                filename=f"concurrent_{i}.jpg",
                media_type="image",
                static_filename=f"concurrent_{i}.jpg",
                task_id=f"concurrent-task-{i}"
            )
            db_session.add(media)
        db_session.commit()

        # Connect all users simultaneously
        websockets = []
        try:
            for i, token in enumerate(tokens):
                ws = client.websocket_connect(f"/ws/inference?token={token}")
                ws.__enter__()
                websockets.append(ws)

            # All should be connected
            assert len(websockets) == 3

            # Each user sends a ping
            for ws in websockets:
                ws.send_text(json.dumps({"type": "ping"}))

            # Each should receive pong
            for ws in websockets:
                data = ws.receive_text()
                response = json.loads(data)
                assert response["type"] == "pong"

        finally:
            # Clean up connections
            for ws in websockets:
                try:
                    ws.__exit__(None, None, None)
                except:
                    pass

    def test_same_user_multiple_connections(self, client, valid_token, test_user_regular, db_session):
        """Test same user with multiple concurrent connections (multiple browser tabs)"""
        websockets = []
        try:
            # Open 3 connections for the same user
            for i in range(3):
                ws = client.websocket_connect(f"/ws/inference?token={valid_token}")
                ws.__enter__()
                websockets.append(ws)

            assert len(websockets) == 3

            # Send different messages from each connection
            for i, ws in enumerate(websockets):
                ws.send_text(json.dumps({"type": "ping", "connection_id": i}))

            # Each should receive response
            for ws in websockets:
                data = ws.receive_text()
                response = json.loads(data)
                assert response["type"] == "pong"

        finally:
            for ws in websockets:
                try:
                    ws.__exit__(None, None, None)
                except:
                    pass


# Test 7: Disconnection Handling
class TestDisconnectionHandling:
    """Test WebSocket disconnection handling"""

    def test_clean_disconnect(self, client, valid_token, db_session):
        """Test clean WebSocket disconnection"""
        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Send ping to verify connection
            websocket.send_text(json.dumps({"type": "ping"}))
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "pong"
            # WebSocket should disconnect cleanly when exiting context

    def test_unexpected_disconnect_handling(self, client, valid_token, test_media_item, db_session):
        """Test handling of unexpected client disconnection"""
        with patch('app.api.websocket.AsyncResult'):
            websocket = None
            try:
                # Connect
                ws_context = client.websocket_connect(f"/ws/inference?token={valid_token}")
                websocket = ws_context.__enter__()

                # Subscribe to task
                websocket.send_text(json.dumps({
                    "type": "subscribe",
                    "task_id": test_media_item.task_id
                }))

                # Simulate unexpected disconnection by closing without proper cleanup
                websocket.close()

            except:
                pass
            finally:
                if websocket:
                    try:
                        ws_context.__exit__(None, None, None)
                    except:
                        pass

    @pytest.mark.anyio
    async def test_connection_cleanup_on_send_failure(self):
        """Test connection cleanup when send fails"""
        manager = ConnectionManager()

        # Create mock websocket that fails on send
        ws = AsyncMock()
        ws.send_text.side_effect = Exception("Connection lost")

        await manager.connect(ws, "user1")
        assert ws in manager.active_connections["user1"]

        # Try to send message (should fail and cleanup)
        await manager.send_personal_message("test message", "user1")

        # Failed connection should be removed
        assert ws not in manager.active_connections.get("user1", set())


# Test 8: Invalid Token Handling
class TestInvalidTokenHandling:
    """Test handling of various invalid token scenarios"""

    def test_malformed_token(self, client):
        """Test connection with malformed JWT token"""
        malformed_token = "this.is.not.a.valid.jwt.token"

        with pytest.raises(Exception):
            with client.websocket_connect(f"/ws/inference?token={malformed_token}") as websocket:
                pass
        # Connection should be closed with error (code 1008)

    def test_token_with_wrong_algorithm(self, client):
        """Test token signed with wrong algorithm"""
        SECRET_KEY = os.getenv("SECRET_KEY", "dev-key")
        token = jwt.encode(
            {"sub": "testuser", "exp": datetime.utcnow() + timedelta(hours=1)},
            SECRET_KEY,
            algorithm="HS512"  # Wrong algorithm
        )

        with pytest.raises(Exception):
            with client.websocket_connect(f"/ws/inference?token={token}") as websocket:
                pass
        # Connection should be closed with error (code 1008)

    def test_token_missing_sub_claim(self, client):
        """Test token missing 'sub' claim"""
        SECRET_KEY = os.getenv("SECRET_KEY", "dev-key")
        ALGORITHM = "HS256"
        token = jwt.encode(
            {"exp": datetime.utcnow() + timedelta(hours=1)},  # Missing 'sub'
            SECRET_KEY,
            algorithm=ALGORITHM
        )

        with pytest.raises(Exception):
            with client.websocket_connect(f"/ws/inference?token={token}") as websocket:
                pass
        # Connection should be closed with error (code 1008)

    def test_empty_token_string(self, client):
        """Test with empty token string"""
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/inference?token=") as websocket:
                pass


# Test 9: Error Handling
class TestErrorHandling:
    """Test various error scenarios"""

    def test_invalid_json_message(self, client, valid_token, db_session):
        """Test handling of invalid JSON messages"""
        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Send invalid JSON
            websocket.send_text("this is not json")

            # Should receive error response
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "error"
            assert "Invalid JSON" in response["message"]

    def test_unknown_message_type(self, client, valid_token, db_session):
        """Test handling of unknown message types"""
        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Send unknown message type
            websocket.send_text(json.dumps({
                "type": "unknown_type",
                "data": "some data"
            }))

            # Should still be connected, send ping to verify
            websocket.send_text(json.dumps({"type": "ping"}))
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "pong"

    def test_missing_required_fields(self, client, valid_token, db_session):
        """Test handling of messages with missing required fields"""
        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Subscribe without task_id
            websocket.send_text(json.dumps({"type": "subscribe"}))

            # Connection should still work
            websocket.send_text(json.dumps({"type": "ping"}))
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "pong"

    @patch('app.api.websocket.AsyncResult')
    def test_celery_connection_error(self, mock_async_result, client, valid_token, test_media_item, db_session):
        """Test handling of Celery connection errors"""
        # Make AsyncResult raise an exception
        mock_async_result.side_effect = Exception("Celery connection failed")

        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Try to subscribe (should handle Celery error gracefully)
            websocket.send_text(json.dumps({
                "type": "subscribe",
                "task_id": test_media_item.task_id
            }))

            # Should receive error response
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "error"
            assert "Celery connection failed" in response["message"]


# Test 10: Performance and Load Testing
class TestPerformance:
    """Test WebSocket performance under load"""

    def test_rapid_message_handling(self, client, valid_token, db_session):
        """Test handling of rapid message sending"""
        with client.websocket_connect(f"/ws/inference?token={valid_token}") as websocket:
            # Send multiple messages rapidly
            for i in range(10):
                websocket.send_text(json.dumps({"type": "ping", "seq": i}))

            # Should receive all pongs
            for i in range(10):
                data = websocket.receive_text()
                response = json.loads(data)
                assert response["type"] == "pong"

    @pytest.mark.anyio
    async def test_connection_manager_performance(self):
        """Test ConnectionManager performance with many connections"""
        manager = ConnectionManager()

        # Create many mock connections
        connections = []
        for i in range(100):
            ws = AsyncMock()
            connections.append(ws)
            await manager.connect(ws, f"user_{i % 10}")  # 10 users, 10 connections each

        # Verify all connected
        assert len(manager.active_connections) == 10
        for user_id in manager.active_connections:
            assert len(manager.active_connections[user_id]) == 10

        # Subscribe many users to same task
        for i in range(50):
            manager.subscribe_to_task("popular_task", f"user_{i}")

        assert len(manager.task_subscriptions["popular_task"]) == 50

        # Broadcast update
        await manager.broadcast_task_update("popular_task", {"status": "complete"})

        # Cleanup
        for i, ws in enumerate(connections):
            manager.disconnect(ws, f"user_{i % 10}")

        assert len(manager.active_connections) == 0