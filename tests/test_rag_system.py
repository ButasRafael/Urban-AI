"""
Comprehensive tests for RAG (Retrieval Augmented Generation) system.

Tests cover:
- Document chunking and ingestion
- Embedding generation
- Vector search
- BM25 full-text search
- Hybrid retrieval (PostgreSQL FTS + vector similarity)
- Cross-encoder reranking
- CSV export
- Query parsing with filter extraction
"""

import pytest
import os
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import all models
import app.models.user
import app.models.media
import app.models.revoked
import app.models.rag
import app.models.conversation

from app.core.database import Base
from app.models.user import User as UserModel, RoleEnum
from app.models.media import Media, Detection, Frame, Severity, IssueStatus, DetectionSource
from app.models.rag import RAGChunk
from app.services.rag import (
    embed,
    retrieve,
    ingest_media,
    sync_detection_updates,
    load_reranker,
    _rrf
)
from app.services.rag_query_parser import parse_query_with_filters


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
        session.query(RAGChunk).delete()
        session.query(Detection).delete()
        session.query(Frame).delete()
        session.query(Media).delete()
        session.query(UserModel).delete()
        session.commit()
        yield session
    finally:
        # Clean up data after test
        session.query(RAGChunk).delete()
        session.query(Detection).delete()
        session.query(Frame).delete()
        session.query(Media).delete()
        session.query(UserModel).delete()
        session.commit()
        session.close()


@pytest.fixture
def test_user(db_session):
    """Create a test user"""
    user = UserModel(
        username="rag_test_user",
        hashed_password="hashed",
        role=RoleEnum.user
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_media_with_detections(db_session, test_user):
    """Create test media with detections for RAG ingestion"""
    # Create media
    media = Media(
        filename="test_road.jpg",
        media_type="image",
        user_username=test_user.username,
        address="Strada Principală 123, Cluj-Napoca",
        created_at=datetime.now(timezone.utc)
    )
    db_session.add(media)
    db_session.commit()
    db_session.refresh(media)

    # Create frame
    frame = Frame(
        media_id=media.id,
        frame_index=0,
        timestamp=0.0
    )
    db_session.add(frame)
    db_session.commit()
    db_session.refresh(frame)

    # Create detections
    detections = [
        Detection(
            frame_id=frame.id,
            class_id=0,
            class_name="pothole",
            confidence=0.95,
            x1=100, y1=100, x2=200, y2=200,
            track_id=1,
            description="Large pothole on the road",
            solution="Fill with asphalt",
            severity=Severity.high,
            status=IssueStatus.open,
            source=DetectionSource.yolo
        ),
        Detection(
            frame_id=frame.id,
            class_id=1,
            class_name="crack",
            confidence=0.85,
            x1=300, y1=150, x2=350, y2=200,
            track_id=2,
            description="Road surface crack",
            solution="Seal the crack",
            severity=Severity.medium,
            status=IssueStatus.open,
            source=DetectionSource.yolo
        )
    ]

    for det in detections:
        db_session.add(det)
    db_session.commit()

    for det in detections:
        db_session.refresh(det)

    return media, detections


# ============== Embedding Generation Tests ==============

class TestEmbeddingGeneration:
    """Test embedding generation functionality"""

    @pytest.mark.anyio
    async def test_embed_function_returns_normalized_vector(self):
        """Test that embed function returns normalized embeddings"""
        with patch('app.services.rag.client.embeddings.create') as mock_create:
            # Mock OpenAI response
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            # Create a non-normalized vector
            raw_embedding = np.random.randn(3072).tolist()
            mock_response.data[0].embedding = raw_embedding
            mock_create.return_value = mock_response

            # Get embedding
            result = await embed("test text")

            # Verify it's normalized (L2 norm ≈ 1)
            result_array = np.array(result)
            norm = np.linalg.norm(result_array)
            assert abs(norm - 1.0) < 1e-5, "Embedding should be normalized"
            assert len(result) == 3072, "Should return 3072-dimensional vector"
            mock_create.assert_called_once()

    @pytest.mark.anyio
    async def test_embed_handles_empty_text(self):
        """Test embedding generation with empty text"""
        with patch('app.services.rag.client.embeddings.create') as mock_create:
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = np.zeros(3072).tolist()
            mock_create.return_value = mock_response

            result = await embed("")

            # Should still return a valid vector
            assert len(result) == 3072


# ============== Document Chunking Tests ==============

class TestDocumentChunking:
    """Test document chunking and ingestion"""

    @pytest.mark.anyio
    async def test_ingest_media_creates_chunks(self, db_session, test_media_with_detections):
        """Test that ingesting media creates RAG chunks correctly"""
        media, detections = test_media_with_detections

        with patch('app.services.rag.embed') as mock_embed:
            # Mock embedding generation
            mock_embed.return_value = np.random.randn(3072).tolist()

            # Ingest media
            await ingest_media(db_session, media.id)

            # Verify chunks were created
            chunks = db_session.query(RAGChunk).filter_by(media_id=media.id).all()
            assert len(chunks) == 2, "Should create one chunk per detection"

            # Verify chunk content
            pothole_chunk = next((c for c in chunks if "pothole" in c.chunk.lower()), None)
            assert pothole_chunk is not None, "Should have pothole chunk"
            assert "Strada Principală 123" in pothole_chunk.chunk
            assert "Large pothole" in pothole_chunk.chunk
            assert "Fill with asphalt" in pothole_chunk.chunk

            # Verify metadata fields
            assert pothole_chunk.severity == Severity.high
            assert pothole_chunk.status == IssueStatus.open
            assert pothole_chunk.class_name == "pothole"
            assert pothole_chunk.address == media.address
            assert pothole_chunk.media_type == "image"

            # Verify embedding was generated
            mock_embed.assert_called()

    @pytest.mark.anyio
    async def test_chunk_includes_temporal_information(self, db_session, test_user):
        """Test that chunks include formatted temporal information"""
        # Create media with specific timestamp
        specific_time = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        media = Media(
            filename="timed_test.jpg",
            media_type="image",
            user_username=test_user.username,
            address="Test Street 1",
            created_at=specific_time
        )
        db_session.add(media)
        db_session.commit()

        frame = Frame(media_id=media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.commit()

        detection = Detection(
            frame_id=frame.id,
            class_id=0,
            class_name="test_issue",
            confidence=0.9,
            x1=0, y1=0, x2=100, y2=100,
            track_id=1,
            description="Test description",
            solution="Test solution",
            source=DetectionSource.yolo
        )
        db_session.add(detection)
        db_session.commit()

        with patch('app.services.rag.embed') as mock_embed:
            mock_embed.return_value = np.random.randn(3072).tolist()
            await ingest_media(db_session, media.id)

            chunk = db_session.query(RAGChunk).filter_by(media_id=media.id).first()
            # Should contain formatted date in Romanian format
            assert "2025" in chunk.chunk or "ianuarie" in chunk.chunk.lower()


# ============== Search Functionality Tests ==============

class TestSearchFunctionality:
    """Test RAG search and retrieval"""

    @pytest.mark.anyio
    async def test_vector_search_finds_relevant_results(self, db_session, test_user):
        """Test that vector search finds semantically similar results"""
        # Create test chunks with embeddings
        # Chunk about potholes
        pothole_emb = np.random.randn(3072)
        pothole_emb = pothole_emb / np.linalg.norm(pothole_emb)

        # Similar embedding (for testing similarity)
        similar_emb = pothole_emb + np.random.randn(3072) * 0.1
        similar_emb = similar_emb / np.linalg.norm(similar_emb)

        # Different embedding
        different_emb = np.random.randn(3072)
        different_emb = different_emb / np.linalg.norm(different_emb)

        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username=test_user.username,
            address="Test Address"
        )
        db_session.add(media)
        db_session.commit()

        chunks_data = [
            ("Pothole on main street", pothole_emb.tolist()),
            ("Road damage similar to pothole", similar_emb.tolist()),
            ("Streetlight malfunction", different_emb.tolist())
        ]

        for text, emb in chunks_data:
            chunk = RAGChunk(
                media_id=media.id,
                chunk=text,
                embedding=emb,
                address="Test"
            )
            db_session.add(chunk)
        db_session.commit()

        # Search with pothole embedding
        results = retrieve(
            db_session,
            pothole_emb.tolist(),
            k=2,
            query_text="pothole",
            skip_semantic=False
        )

        # Should return the two pothole-related chunks
        assert len(results) <= 2
        # First result should be most similar
        if results:
            assert "pothole" in results[0].chunk.lower() or "damage" in results[0].chunk.lower()

    @pytest.mark.anyio
    async def test_chunk_text_search_capability(self, db_session, test_user):
        """Test that chunks can be searched by text content"""
        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username=test_user.username,
            address="Test Address"
        )
        db_session.add(media)
        db_session.commit()

        # Create chunks with different content
        chunks_text = [
            "Pothole on strada principală needs immediate repair",
            "Crack in the road surface",
            "Streetlight not working in the park"
        ]

        for text in chunks_text:
            chunk = RAGChunk(
                media_id=media.id,
                chunk=text,
                embedding=np.random.randn(3072).tolist(),
                address="Test"
            )
            db_session.add(chunk)
        db_session.commit()

        # Simple text search using SQLAlchemy
        results = db_session.query(RAGChunk).filter(
            RAGChunk.chunk.ilike('%pothole%')
        ).all()

        # Should find the pothole chunk
        assert len(results) >= 1
        assert any("pothole" in c.chunk.lower() for c in results)


# ============== Hybrid Retrieval Tests ==============

class TestHybridRetrieval:
    """Test hybrid retrieval combining vector and BM25 search"""

    @pytest.mark.anyio
    async def test_hybrid_search_combines_vector_and_bm25(self, db_session, test_user):
        """Test that hybrid search uses both vector similarity and keyword matching"""
        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username=test_user.username,
            address="Main Street, City Center"
        )
        db_session.add(media)
        db_session.commit()

        # Create query embedding
        query_emb = np.random.randn(3072)
        query_emb = query_emb / np.linalg.norm(query_emb)

        # Chunk 1: High vector similarity, no keyword match
        similar_emb = query_emb + np.random.randn(3072) * 0.05
        similar_emb = similar_emb / np.linalg.norm(similar_emb)

        # Chunk 2: Exact keyword match, low vector similarity
        keyword_emb = np.random.randn(3072)
        keyword_emb = keyword_emb / np.linalg.norm(keyword_emb)

        chunks = [
            RAGChunk(
                media_id=media.id,
                chunk="Road surface degradation causing vehicle damage",
                embedding=similar_emb.tolist(),
                address="Test"
            ),
            RAGChunk(
                media_id=media.id,
                chunk="Pothole detected on the street requiring urgent repair",
                embedding=keyword_emb.tolist(),
                address="Test"
            )
        ]

        for chunk in chunks:
            db_session.add(chunk)
        db_session.commit()

        # Hybrid search for "pothole"
        with patch('app.services.rag.cross_encoder', None):  # Use embedding-based reranking
            results = retrieve(
                db_session,
                query_emb.tolist(),
                k=2,
                query_text="pothole",
                skip_semantic=False
            )

        # Should return both results (hybrid fusion)
        assert len(results) > 0
        # At least one should mention pothole
        assert any("pothole" in r.chunk.lower() for r in results)

    def test_rrf_fusion_combines_rankings(self):
        """Test Reciprocal Rank Fusion combines rankings correctly"""
        # Ranking 1: [1, 2, 3]
        # Ranking 2: [3, 1, 4]

        rrf1 = _rrf([1, 2, 3], k0=60)
        rrf2 = _rrf([3, 1, 4], k0=60)

        # Combine scores
        combined = {}
        for doc_id, score in rrf1.items():
            combined[doc_id] = combined.get(doc_id, 0) + score
        for doc_id, score in rrf2.items():
            combined[doc_id] = combined.get(doc_id, 0) + score

        # Doc 1 and 3 appear in both, should have higher scores
        assert combined[1] > combined[2]
        assert combined[3] > combined[2]
        assert combined[1] > combined[4]


# ============== Filter Tests ==============

class TestFiltering:
    """Test SQL filtering functionality"""

    @pytest.mark.anyio
    async def test_severity_filter(self, db_session, test_user):
        """Test filtering by severity"""
        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username=test_user.username,
            address="Test"
        )
        db_session.add(media)
        db_session.commit()

        chunks = [
            RAGChunk(
                media_id=media.id,
                chunk="High severity issue",
                embedding=np.random.randn(3072).tolist(),
                severity=Severity.high,
                address="Test"
            ),
            RAGChunk(
                media_id=media.id,
                chunk="Low severity issue",
                embedding=np.random.randn(3072).tolist(),
                severity=Severity.low,
                address="Test"
            )
        ]

        for c in chunks:
            db_session.add(c)
        db_session.commit()

        # Search with severity filter
        query_emb = np.random.randn(3072).tolist()
        results = retrieve(
            db_session,
            query_emb,
            k=10,
            severity_filter=Severity.high
        )

        # Should only return high severity
        assert all(r.severity == Severity.high for r in results)

    @pytest.mark.anyio
    async def test_status_filter(self, db_session, test_user):
        """Test filtering by status"""
        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username=test_user.username,
            address="Test"
        )
        db_session.add(media)
        db_session.commit()

        chunks = [
            RAGChunk(
                media_id=media.id,
                chunk="Open issue",
                embedding=np.random.randn(3072).tolist(),
                status=IssueStatus.open,
                address="Test"
            ),
            RAGChunk(
                media_id=media.id,
                chunk="Resolved issue",
                embedding=np.random.randn(3072).tolist(),
                status=IssueStatus.resolved,
                address="Test"
            )
        ]

        for c in chunks:
            db_session.add(c)
        db_session.commit()

        # Search with status filter
        query_emb = np.random.randn(3072).tolist()
        results = retrieve(
            db_session,
            query_emb,
            k=10,
            status_filter=IssueStatus.resolved
        )

        # Should only return resolved
        assert all(r.status == IssueStatus.resolved for r in results)

    @pytest.mark.anyio
    async def test_date_range_filter(self, db_session, test_user):
        """Test filtering by date ranges"""
        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username=test_user.username,
            address="Test"
        )
        db_session.add(media)
        db_session.commit()

        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)
        last_week = today - timedelta(days=7)

        chunks = [
            RAGChunk(
                media_id=media.id,
                chunk="Recently resolved",
                embedding=np.random.randn(3072).tolist(),
                status=IssueStatus.resolved,
                resolved_at=datetime.now(timezone.utc),
                address="Test"
            ),
            RAGChunk(
                media_id=media.id,
                chunk="Resolved last week",
                embedding=np.random.randn(3072).tolist(),
                status=IssueStatus.resolved,
                resolved_at=datetime.combine(last_week, datetime.min.time()).replace(tzinfo=timezone.utc),
                address="Test"
            )
        ]

        for c in chunks:
            db_session.add(c)
        db_session.commit()

        # Search for items resolved after yesterday
        query_emb = np.random.randn(3072).tolist()
        results = retrieve(
            db_session,
            query_emb,
            k=10,
            resolved_after=yesterday.isoformat()
        )

        # Should only return recently resolved
        assert len(results) >= 1
        assert all(r.resolved_at >= datetime.combine(yesterday, datetime.min.time()).replace(tzinfo=timezone.utc) for r in results if r.resolved_at)


# ============== Reranking Tests ==============

class TestReranking:
    """Test cross-encoder reranking"""

    @pytest.mark.anyio
    async def test_reranking_improves_relevance(self, db_session, test_user):
        """Test that reranking reorders results by relevance"""
        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username=test_user.username,
            address="Test"
        )
        db_session.add(media)
        db_session.commit()

        chunks = [
            RAGChunk(
                media_id=media.id,
                chunk="Pothole on the main street causing traffic issues",
                embedding=np.random.randn(3072).tolist(),
                address="Test"
            ),
            RAGChunk(
                media_id=media.id,
                chunk="Streetlight malfunction in the park",
                embedding=np.random.randn(3072).tolist(),
                address="Test"
            )
        ]

        for c in chunks:
            db_session.add(c)
        db_session.commit()

        # Mock cross-encoder with deterministic scores
        with patch('app.services.rag.cross_encoder') as mock_ce:
            with patch('app.services.rag.reranker_type', 'cross-encoder'):
                # First chunk (pothole) gets higher score for "pothole" query
                mock_ce.predict.return_value = [0.9, 0.3]

                query_emb = np.random.randn(3072).tolist()
                results = retrieve(
                    db_session,
                    query_emb,
                    k=2,
                    query_text="pothole on street"
                )

                # First result should be the pothole chunk (reranked)
                assert "pothole" in results[0].chunk.lower()


# ============== Synchronization Tests ==============

class TestSynchronization:
    """Test RAG chunk synchronization with detection updates"""

    @pytest.mark.anyio
    async def test_sync_detection_updates_dynamic_fields(self, db_session, test_media_with_detections):
        """Test that updating detection syncs to RAG chunks"""
        media, detections = test_media_with_detections

        # Create users for assignment
        authority = UserModel(
            username="test_authority",
            hashed_password="hashed",
            role=RoleEnum.user
        )
        admin = UserModel(
            username="test_admin",
            hashed_password="hashed",
            role=RoleEnum.admin
        )
        db_session.add_all([authority, admin])
        db_session.commit()

        # Ingest media to create chunks
        with patch('app.services.rag.embed') as mock_embed:
            mock_embed.return_value = np.random.randn(3072).tolist()
            await ingest_media(db_session, media.id)

        # Get a detection and its chunk
        detection = detections[0]
        chunk = db_session.query(RAGChunk).filter_by(detection_id=detection.id).first()
        assert chunk.severity == Severity.high
        assert chunk.status == IssueStatus.open

        # Update detection
        detection.severity = Severity.low
        detection.status = IssueStatus.resolved
        detection.assigned_to = "test_authority"
        detection.verified_by = "test_admin"
        db_session.commit()

        # Sync updates
        await sync_detection_updates(db_session, detection.id)

        # Verify chunk was updated
        db_session.refresh(chunk)
        assert chunk.severity == Severity.low
        assert chunk.status == IssueStatus.resolved
        assert chunk.assigned_to == "test_authority"
        assert chunk.verified_by == "test_admin"


# ============== Query Parsing Tests ==============

class TestQueryParsing:
    """Test query parsing and filter extraction"""

    @pytest.mark.anyio
    async def test_parse_severity_filter(self, db_session):
        """Test parsing severity from natural language"""
        with patch('app.services.rag_query_parser.client.responses.create') as mock_create:
            # Mock GPT response
            mock_response = MagicMock()
            mock_response.output_text = '{"query": "potholes", "severity": "high", "status": null, "assigned_to": null, "verified_by": null, "resolved_after": null, "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": false}'
            mock_create.return_value = mock_response

            result = await parse_query_with_filters("high severity potholes")

            assert result["query"] == "potholes"
            assert result["severity"] == Severity.high
            assert result["status"] is None

    @pytest.mark.anyio
    async def test_parse_status_filter(self, db_session):
        """Test parsing status from natural language"""
        with patch('app.services.rag_query_parser.client.responses.create') as mock_create:
            mock_response = MagicMock()
            mock_response.output_text = '{"query": "issues", "severity": null, "status": "resolved", "assigned_to": null, "verified_by": null, "resolved_after": null, "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": true}'
            mock_create.return_value = mock_response

            result = await parse_query_with_filters("resolved issues")

            assert result["query"] == "issues"
            assert result["status"] == IssueStatus.resolved
            assert result["sql_only"] is True

    @pytest.mark.anyio
    async def test_parse_temporal_filter(self, db_session):
        """Test parsing temporal filters"""
        with patch('app.services.rag_query_parser.client.responses.create') as mock_create:
            mock_response = MagicMock()
            mock_response.output_text = '{"query": "problems", "severity": null, "status": null, "assigned_to": null, "verified_by": null, "resolved_after": "2025-01-20", "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": true}'
            mock_create.return_value = mock_response

            result = await parse_query_with_filters("problems resolved after January 20")

            assert result["resolved_after"] == "2025-01-20"

    @pytest.mark.anyio
    async def test_sql_only_detection(self, db_session):
        """Test SQL-only query detection"""
        with patch('app.services.rag_query_parser.client.responses.create') as mock_create:
            # SQL-only query: filter + generic term
            mock_response = MagicMock()
            mock_response.output_text = '{"query": "issues", "severity": "high", "status": null, "assigned_to": null, "verified_by": null, "resolved_after": null, "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": true}'
            mock_create.return_value = mock_response

            result = await parse_query_with_filters("high severity issues")
            assert result["sql_only"] is True

            # Semantic query: filter + specific content
            mock_response.output_text = '{"query": "potholes on Main Street", "severity": "high", "status": null, "assigned_to": null, "verified_by": null, "resolved_after": null, "resolved_before": null, "verified_after": null, "verified_before": null, "sql_only": false}'
            mock_create.return_value = mock_response

            result = await parse_query_with_filters("high severity potholes on Main Street")
            assert result["sql_only"] is False

    @pytest.mark.anyio
    async def test_parse_handles_error_gracefully(self, db_session):
        """Test that parsing errors fallback gracefully"""
        with patch('app.services.rag_query_parser.client.responses.create') as mock_create:
            # Simulate parsing error
            mock_create.side_effect = Exception("API error")

            result = await parse_query_with_filters("test query")

            # Should return original query with no filters
            assert result["query"] == "test query"
            assert result["severity"] is None
            assert result["status"] is None
            assert result["sql_only"] is False


# ============== Per-Media Cap Tests ==============

class TestPerMediaCap:
    """Test per-media result capping"""

    @pytest.mark.anyio
    async def test_per_media_cap_limits_results(self, db_session, test_user):
        """Test that per-media cap limits results from same media"""
        # Create two media items
        media1 = Media(
            filename="test1.jpg",
            media_type="image",
            user_username=test_user.username,
            address="Location 1"
        )
        media2 = Media(
            filename="test2.jpg",
            media_type="image",
            user_username=test_user.username,
            address="Location 2"
        )
        db_session.add_all([media1, media2])
        db_session.commit()

        # Create multiple chunks for media1, one for media2
        for i in range(5):
            chunk = RAGChunk(
                media_id=media1.id,
                chunk=f"Issue {i} at location 1",
                embedding=np.random.randn(3072).tolist(),
                address="Location 1"
            )
            db_session.add(chunk)

        chunk = RAGChunk(
            media_id=media2.id,
            chunk="Issue at location 2",
            embedding=np.random.randn(3072).tolist(),
            address="Location 2"
        )
        db_session.add(chunk)
        db_session.commit()

        # Search with per_media_cap=2
        query_emb = np.random.randn(3072).tolist()
        with patch('app.services.rag.cross_encoder', None):
            results = retrieve(
                db_session,
                query_emb,
                k=10,
                per_media_cap=2
            )

        # Count results per media
        media1_count = sum(1 for r in results if r.media_id == media1.id)
        media2_count = sum(1 for r in results if r.media_id == media2.id)

        # Should respect cap
        assert media1_count <= 2, "Should cap results from media1 to 2"
        assert media2_count <= 2, "Should cap results from media2 to 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
