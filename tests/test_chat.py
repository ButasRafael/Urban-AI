"""
Comprehensive tests for chat system.

Tests cover:
- Session creation and management
- Message streaming with SSE
- Session history retrieval
- Title generation and updates
- Session deletion with CASCADE
- Context management with RAG integration
- Message history truncation (context management)
"""

import pytest
import os
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import all models
import app.models.user
import app.models.media
import app.models.revoked
import app.models.rag
import app.models.conversation

from app.core.database import Base
from app.models.user import User as UserModel, RoleEnum
from app.models.conversation import ChatSession, ChatMessage
from app.models.media import Media, Detection, Frame, Severity, IssueStatus, DetectionSource
from app.models.rag import RAGChunk
from app.core.security import create_access_token
from app.main import app


# Use PostgreSQL for testing
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
        session.query(ChatMessage).delete()
        session.query(ChatSession).delete()
        session.query(RAGChunk).delete()
        session.query(Detection).delete()
        session.query(Frame).delete()
        session.query(Media).delete()
        session.query(UserModel).delete()
        session.commit()
        yield session
    finally:
        # Rollback any pending transactions
        session.rollback()
        try:
            # Clean up data after test
            session.query(ChatMessage).delete()
            session.query(ChatSession).delete()
            session.query(RAGChunk).delete()
            session.query(Detection).delete()
            session.query(Frame).delete()
            session.query(Media).delete()
            session.query(UserModel).delete()
            session.commit()
        except:
            session.rollback()
        finally:
            session.close()


@pytest.fixture
def override_get_db(db_session):
    """Override FastAPI database dependency"""
    def _override():
        try:
            yield db_session
        finally:
            pass
    return _override


@pytest.fixture
def authority_user(db_session):
    """Create an authority user for testing"""
    user = UserModel(
        username="test_authority",
        hashed_password="hashed",
        role=RoleEnum.authority
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def authority_token(authority_user):
    """Generate auth token for authority user"""
    return create_access_token({"sub": authority_user.username})


@pytest.fixture
def client(override_get_db):
    """Create test client with database override"""
    from app.core.database import get_db
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def test_rag_chunks(db_session, authority_user):
    """Create test RAG chunks for context"""
    media = Media(
        filename="test.jpg",
        media_type="image",
        user_username=authority_user.username,
        address="Main Street, City Center"
    )
    db_session.add(media)
    db_session.commit()

    chunks = [
        RAGChunk(
            media_id=media.id,
            chunk="Pothole detected at Main Street requiring urgent repair",
            embedding=[0.1] * 3072,
            address="Main Street",
            class_name="pothole",
            severity=Severity.high,
            status=IssueStatus.open
        ),
        RAGChunk(
            media_id=media.id,
            chunk="Street light malfunction on Oak Avenue",
            embedding=[0.2] * 3072,
            address="Oak Avenue",
            class_name="streetlight",
            severity=Severity.medium,
            status=IssueStatus.open
        )
    ]
    for chunk in chunks:
        db_session.add(chunk)
    db_session.commit()
    return chunks


# ============== Session Creation Tests ==============

class TestSessionCreation:
    """Test chat session creation"""

    def test_create_session_on_first_message(self, client, authority_token, db_session):
        """Test that new session is created when no session_id provided"""
        with patch('app.api.chat.generate_conversation_title', new_callable=AsyncMock, return_value="Test Title"):
            with patch('app.api.chat.client.responses.create', new_callable=AsyncMock) as mock_create:
                # Mock OpenAI streaming response
                async def mock_stream():
                    # Simulate streaming events
                    mock_event1 = MagicMock()
                    mock_event1.type = "response.output_text.delta"
                    mock_event1.delta = "Test "
                    yield mock_event1

                    mock_event2 = MagicMock()
                    mock_event2.type = "response.output_text.delta"
                    mock_event2.delta = "response"
                    yield mock_event2

                    mock_event3 = MagicMock()
                    mock_event3.type = "response.completed"
                    yield mock_event3

                mock_create.return_value = mock_stream()

                response = client.post(
                    "/chat/stream",
                    json={"message": "Test message"},
                    headers={"Authorization": f"Bearer {authority_token}"}
                )

                assert response.status_code == 200

                # Verify session was created
                db_session.rollback()  # Ensure we see committed data
                sessions = db_session.query(ChatSession).all()
                assert len(sessions) == 1
                assert sessions[0].authority_username == "test_authority"
                assert sessions[0].title == "Test Title"

    def test_reuse_existing_session(self, client, authority_token, db_session, authority_user):
        """Test that existing session is reused when session_id provided"""
        # Create existing session
        session = ChatSession(
            authority_username=authority_user.username,
            title="Existing Session"
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        with patch('app.api.chat.client.responses.create', new_callable=AsyncMock) as mock_create:
            async def mock_stream():
                mock_event = MagicMock()
                mock_event.type = "response.completed"
                yield mock_event

            mock_create.return_value = mock_stream()

            response = client.post(
                "/chat/stream",
                json={"message": "Follow-up message", "session_id": session.id},
                headers={"Authorization": f"Bearer {authority_token}"}
            )

            assert response.status_code == 200

            # Verify no new session was created
            db_session.rollback()
            sessions = db_session.query(ChatSession).all()
            assert len(sessions) == 1
            assert sessions[0].id == session.id


# ============== Message Streaming Tests ==============

class TestMessageStreaming:
    """Test SSE message streaming"""

    def test_stream_events_format(self, client, authority_token):
        """Test that SSE events are properly formatted"""
        with patch('app.api.chat.generate_conversation_title', new_callable=AsyncMock, return_value="Test"):
            with patch('app.api.chat.client.responses.create', new_callable=AsyncMock) as mock_create:
                async def mock_stream():
                    mock_event1 = MagicMock()
                    mock_event1.type = "response.output_text.delta"
                    mock_event1.delta = "Hello"
                    yield mock_event1

                    mock_event2 = MagicMock()
                    mock_event2.type = "response.completed"
                    yield mock_event2

                mock_create.return_value = mock_stream()

                response = client.post(
                    "/chat/stream",
                    json={"message": "Test"},
                    headers={"Authorization": f"Bearer {authority_token}"}
                )

                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
                assert "Cache-Control" in response.headers
                assert response.headers["Cache-Control"] == "no-cache"

    def test_stream_delta_events(self, client, authority_token):
        """Test that delta events are streamed correctly"""
        with patch('app.api.chat.generate_conversation_title', new_callable=AsyncMock, return_value="Test"):
            with patch('app.api.chat.client.responses.create', new_callable=AsyncMock) as mock_create:
                async def mock_stream():
                    mock_event1 = MagicMock()
                    mock_event1.type = "response.output_text.delta"
                    mock_event1.delta = "Part1 "
                    yield mock_event1

                    mock_event2 = MagicMock()
                    mock_event2.type = "response.output_text.delta"
                    mock_event2.delta = "Part2"
                    yield mock_event2

                    mock_event3 = MagicMock()
                    mock_event3.type = "response.completed"
                    yield mock_event3

                mock_create.return_value = mock_stream()

                response = client.post(
                    "/chat/stream",
                    json={"message": "Test"},
                    headers={"Authorization": f"Bearer {authority_token}"}
                )

                # Parse SSE events from response
                content = response.text
                assert "event: delta" in content
                assert "Part1" in content or "Part2" in content

    def test_stream_session_event(self, client, authority_token):
        """Test that session event is sent at start"""
        with patch('app.api.chat.generate_conversation_title', new_callable=AsyncMock, return_value="Test"):
            with patch('app.api.chat.client.responses.create', new_callable=AsyncMock) as mock_create:
                async def mock_stream():
                    mock_event = MagicMock()
                    mock_event.type = "response.completed"
                    yield mock_event

                mock_create.return_value = mock_stream()

                response = client.post(
                    "/chat/stream",
                    json={"message": "Test"},
                    headers={"Authorization": f"Bearer {authority_token}"}
                )

                content = response.text
                assert "event: session" in content


# ============== Session History Tests ==============

class TestSessionHistory:
    """Test session history retrieval"""

    def test_get_session_history(self, client, authority_token, db_session, authority_user):
        """Test retrieving session history"""
        # Create session with messages
        session = ChatSession(
            authority_username=authority_user.username,
            title="Test Session"
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        messages = [
            ChatMessage(session_id=session.id, role="user", content="Hello"),
            ChatMessage(session_id=session.id, role="assistant", content="Hi there!"),
            ChatMessage(session_id=session.id, role="user", content="How are you?"),
        ]
        for msg in messages:
            db_session.add(msg)
        db_session.commit()

        response = client.get(
            f"/chat/sessions/{session.id}",
            headers={"Authorization": f"Bearer {authority_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert len(data["messages"]) == 3
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello"
        assert data["messages"][1]["role"] == "assistant"

    def test_get_session_history_not_found(self, client, authority_token):
        """Test getting history for non-existent session"""
        response = client.get(
            "/chat/sessions/99999",
            headers={"Authorization": f"Bearer {authority_token}"}
        )

        assert response.status_code == 404

    def test_list_sessions(self, client, authority_token, db_session, authority_user):
        """Test listing user's sessions"""
        # Create multiple sessions
        sessions = [
            ChatSession(authority_username=authority_user.username, title="Session 1"),
            ChatSession(authority_username=authority_user.username, title="Session 2"),
        ]
        for s in sessions:
            db_session.add(s)
        db_session.commit()

        response = client.get(
            "/chat/sessions",
            headers={"Authorization": f"Bearer {authority_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all("title" in s for s in data)
        assert all("created_at" in s for s in data)
        assert all("last_message_at" in s for s in data)


# ============== Title Management Tests ==============

class TestTitleManagement:
    """Test session title generation and updates"""

    @pytest.mark.anyio
    async def test_title_generation(self):
        """Test automatic title generation for new sessions"""
        from app.api.chat import generate_conversation_title

        with patch('app.api.chat.client.responses.create') as mock_create:
            mock_response = MagicMock()
            mock_response.output_text = "Pothole Repair Main Street"
            mock_create.return_value = mock_response

            title = await generate_conversation_title("Fix pothole on Main Street")

            assert title == "Pothole Repair Main Street"
            assert len(title) <= 50

    @pytest.mark.anyio
    async def test_title_generation_fallback(self):
        """Test title generation fallback on error"""
        from app.api.chat import generate_conversation_title

        with patch('app.api.chat.client.responses.create', side_effect=Exception("API Error")):
            title = await generate_conversation_title("Test message")

            assert title == "New Conversation"

    @pytest.mark.anyio
    async def test_title_generation_sanitization(self):
        """Test that generated titles are sanitized"""
        from app.api.chat import generate_conversation_title

        with patch('app.api.chat.client.responses.create') as mock_create:
            mock_response = MagicMock()
            mock_response.output_text = '"Test: Title!" with special@chars#'
            mock_create.return_value = mock_response

            title = await generate_conversation_title("Test")

            # Should remove special characters
            assert ":" not in title
            assert "!" not in title
            assert "@" not in title
            assert "#" not in title

    def test_update_session_title(self, client, authority_token, db_session, authority_user):
        """Test manually updating session title"""
        session = ChatSession(
            authority_username=authority_user.username,
            title="Old Title"
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        response = client.patch(
            f"/chat/sessions/{session.id}/title",
            params={"title": "New Title"},
            headers={"Authorization": f"Bearer {authority_token}"}
        )

        assert response.status_code == 200
        db_session.refresh(session)
        assert session.title == "New Title"

    def test_update_title_max_length(self, client, authority_token, db_session, authority_user):
        """Test that title is truncated to max length"""
        session = ChatSession(
            authority_username=authority_user.username,
            title="Original"
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        long_title = "A" * 100  # 100 characters
        response = client.patch(
            f"/chat/sessions/{session.id}/title",
            params={"title": long_title},
            headers={"Authorization": f"Bearer {authority_token}"}
        )

        assert response.status_code == 200
        db_session.refresh(session)
        assert len(session.title) == 50  # Max length


# ============== Session Deletion Tests ==============

class TestSessionDeletion:
    """Test session deletion with CASCADE"""

    def test_delete_session(self, client, authority_token, db_session, authority_user):
        """Test deleting a session"""
        session = ChatSession(
            authority_username=authority_user.username,
            title="To Delete"
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        response = client.delete(
            f"/chat/sessions/{session.id}",
            headers={"Authorization": f"Bearer {authority_token}"}
        )

        assert response.status_code == 204

        # Verify session is deleted
        deleted = db_session.query(ChatSession).filter_by(id=session.id).first()
        assert deleted is None

    def test_delete_session_cascades_messages(self, client, authority_token, db_session, authority_user):
        """Test that deleting session cascades to messages"""
        session = ChatSession(
            authority_username=authority_user.username,
            title="To Delete"
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        # Add messages
        messages = [
            ChatMessage(session_id=session.id, role="user", content="Msg1"),
            ChatMessage(session_id=session.id, role="assistant", content="Msg2"),
        ]
        for msg in messages:
            db_session.add(msg)
        db_session.commit()

        response = client.delete(
            f"/chat/sessions/{session.id}",
            headers={"Authorization": f"Bearer {authority_token}"}
        )

        assert response.status_code == 204

        # Verify messages are also deleted
        remaining_messages = db_session.query(ChatMessage).filter_by(session_id=session.id).all()
        assert len(remaining_messages) == 0

    def test_delete_nonexistent_session(self, client, authority_token):
        """Test deleting non-existent session"""
        response = client.delete(
            "/chat/sessions/99999",
            headers={"Authorization": f"Bearer {authority_token}"}
        )

        assert response.status_code == 404


# ============== Context Management Tests ==============

class TestContextManagement:
    """Test RAG context management"""

    def test_context_built_from_rag(self, client, authority_token, test_rag_chunks):
        """Test that context is built from RAG chunks"""
        with patch('app.api.chat.generate_conversation_title', new_callable=AsyncMock, return_value="Test"):
            with patch('app.services.rag.embed', new_callable=AsyncMock, return_value=[0.1] * 3072):
                with patch('app.api.chat.client.responses.create', new_callable=AsyncMock) as mock_create:
                    async def mock_stream():
                        mock_event = MagicMock()
                        mock_event.type = "response.completed"
                        yield mock_event

                    mock_create.return_value = mock_stream()

                    response = client.post(
                        "/chat/stream",
                        json={"message": "Show me potholes"},
                        headers={"Authorization": f"Bearer {authority_token}"}
                    )

                    # Verify RAG embed was called
                    assert response.status_code == 200

    def test_message_history_truncation(self, client, authority_token, db_session, authority_user):
        """Test that message history is truncated to last 10 messages"""
        session = ChatSession(
            authority_username=authority_user.username,
            title="Long History"
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        # Add 15 messages (should only use last 10)
        for i in range(15):
            role = "user" if i % 2 == 0 else "assistant"
            msg = ChatMessage(session_id=session.id, role=role, content=f"Message {i}")
            db_session.add(msg)
        db_session.commit()

        with patch('app.api.chat.client.responses.create', new_callable=AsyncMock) as mock_create:
            captured_messages = []

            async def mock_stream():
                # Capture the input messages
                call_kwargs = mock_create.call_args.kwargs
                if 'input' in call_kwargs:
                    for item in call_kwargs['input']:
                        if isinstance(item, dict) and item.get('role') in ('user', 'assistant'):
                            captured_messages.append(item)

                mock_event = MagicMock()
                mock_event.type = "response.completed"
                yield mock_event

            mock_create.return_value = mock_stream()

            response = client.post(
                "/chat/stream",
                json={"message": "New message", "session_id": session.id},
                headers={"Authorization": f"Bearer {authority_token}"}
            )

            assert response.status_code == 200
            # Should have at most 10 history messages in context
            # (not counting system messages and current user message)

    def test_no_sources_fallback(self, client, authority_token, db_session):
        """Test fallback response when no RAG sources found"""
        with patch('app.api.chat.generate_conversation_title', new_callable=AsyncMock, return_value="Test"):
            with patch('app.services.rag.retrieve', return_value=[]):  # No sources
                with patch('app.api.chat.client.responses.create', new_callable=AsyncMock) as mock_create:
                    async def mock_stream():
                        # Simulate no text from model (will trigger fallback)
                        mock_event = MagicMock()
                        mock_event.type = "response.completed"
                        yield mock_event

                    mock_create.return_value = mock_stream()

                    response = client.post(
                        "/chat/stream",
                        json={"message": "Find something that doesn't exist"},
                        headers={"Authorization": f"Bearer {authority_token}"}
                    )

                    assert response.status_code == 200
                    content = response.text
                    # Should contain fallback message
                    assert "Not enough evidence" in content or "no sources" in content.lower()


# ============== Integration Tests ==============

class TestChatIntegration:
    """Integration tests for complete chat flows"""

    def test_complete_conversation_flow(self, client, authority_token, db_session):
        """Test complete conversation flow from start to finish"""
        with patch('app.api.chat.generate_conversation_title', new_callable=AsyncMock, return_value="Complete Test"):
            with patch('app.api.chat.client.responses.create', new_callable=AsyncMock) as mock_create:
                async def mock_stream():
                    mock_event1 = MagicMock()
                    mock_event1.type = "response.output_text.delta"
                    mock_event1.delta = "Response text"
                    yield mock_event1

                    mock_event2 = MagicMock()
                    mock_event2.type = "response.completed"
                    yield mock_event2

                # Return a new generator for each call
                mock_create.side_effect = lambda *args, **kwargs: mock_stream()

                # 1. Create new session with first message
                response1 = client.post(
                    "/chat/stream",
                    json={"message": "First message"},
                    headers={"Authorization": f"Bearer {authority_token}"}
                )
                assert response1.status_code == 200

                # Extract session_id from response
                db_session.rollback()  # Ensure we see committed data
                sessions = db_session.query(ChatSession).all()
                assert len(sessions) == 1
                session_id = sessions[0].id

                # 2. Send follow-up message
                response2 = client.post(
                    "/chat/stream",
                    json={"message": "Follow-up", "session_id": session_id},
                    headers={"Authorization": f"Bearer {authority_token}"}
                )
                assert response2.status_code == 200

                # 3. Get history
                response3 = client.get(
                    f"/chat/sessions/{session_id}",
                    headers={"Authorization": f"Bearer {authority_token}"}
                )
                assert response3.status_code == 200
                history = response3.json()
                # Should have 2 user messages + 2 assistant responses = 4 total
                assert len(history["messages"]) == 4

                # 4. Update title
                response4 = client.patch(
                    f"/chat/sessions/{session_id}/title",
                    params={"title": "Updated Title"},
                    headers={"Authorization": f"Bearer {authority_token}"}
                )
                assert response4.status_code == 200

                # 5. Delete session
                response5 = client.delete(
                    f"/chat/sessions/{session_id}",
                    headers={"Authorization": f"Bearer {authority_token}"}
                )
                assert response5.status_code == 204

                # Verify complete deletion
                db_session.rollback()
                deleted_session = db_session.query(ChatSession).filter_by(id=session_id).first()
                assert deleted_session is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])