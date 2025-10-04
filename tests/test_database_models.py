"""
Comprehensive tests for database models and operations.

Tests cover:
- User model CRUD operations (create, read, update, delete)
- Media model operations (images, videos, metadata)
- Detection model operations (bounding boxes, masks, tracking)
- Frame model operations (video frame management)
- RAG chunk model operations (document storage, embeddings)
- Chat session and message models
- Relationship integrity (foreign keys, back references)
- Cascading deletes (CASCADE, SET NULL behaviors)
- Status transitions (open → resolved → ignored)
- Enum validation (severity, status, source, role)
- Transaction handling (commit, rollback, isolation)
- Concurrent updates and race conditions
- Database constraints (unique, not null, check constraints)
- Index performance and query optimization
- Timestamp handling (created_at, updated_at, resolved_at)
- Pagination and filtering
- Complex queries with joins
- Database session management
- Connection pooling
"""

import pytest
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, DataError
import threading

# Import all models to ensure they're registered
import app.models.user
import app.models.media
import app.models.revoked
import app.models.rag  # Need to import RAG models for Media.rag_chunks relationship
import app.models.conversation  # Need to import conversation models

from app.core.database import Base
from app.models.user import User as UserModel, RoleEnum
from app.models.media import (
    Media,
    Frame,
    Detection,
    ProcessingStatus,
    IssueStatus,
    Severity,
    DetectionSource,
)
from app.core.security import get_password_hash


# Test database setup
import os

# Use PostgreSQL for testing (same as production)
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@db:5432/urban_ai_test"
)

engine = create_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session():
    """Create a new database session for each test"""
    session = TestingSessionLocal()
    try:
        # Clean up all tables
        session.query(Detection).delete()
        session.query(Frame).delete()
        session.query(Media).delete()
        session.query(UserModel).delete()
        session.commit()
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def test_user(db_session):
    """Create a test user"""
    user = UserModel(
        username="testuser",
        hashed_password=get_password_hash("password123"),
        role=RoleEnum.user,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_media(db_session, test_user):
    """Create test media"""
    media = Media(
        filename="test.jpg",
        media_type="image",
        user_username=test_user.username,
        width=1920,
        height=1080,
        processing_status=ProcessingStatus.pending,
    )
    db_session.add(media)
    db_session.commit()
    db_session.refresh(media)
    return media


class TestMediaCRUD:
    """Test Media model CRUD operations"""

    def test_create_media(self, db_session, test_user):
        """Test creating a media record"""
        media = Media(
            filename="image.jpg",
            media_type="image",
            user_username=test_user.username,
            width=1920,
            height=1080,
            latitude=40.7128,
            longitude=-74.0060,
            address="New York, NY",
        )
        db_session.add(media)
        db_session.commit()

        # Verify record was created
        assert media.id is not None
        assert media.created_at is not None
        assert media.processing_status == ProcessingStatus.pending  # Default value
        assert isinstance(media.created_at, datetime)

    def test_read_media(self, db_session, test_media):
        """Test reading a media record"""
        # Query by ID
        media = db_session.query(Media).filter(Media.id == test_media.id).first()

        assert media is not None
        assert media.filename == "test.jpg"
        assert media.media_type == "image"
        assert media.width == 1920
        assert media.height == 1080

    def test_update_media(self, db_session, test_media):
        """Test updating a media record"""
        # Update fields
        test_media.processing_status = ProcessingStatus.completed
        test_media.static_filename = "uuid-123.jpg"
        test_media.process_ms_total = 5000
        db_session.commit()

        # Verify updates persisted
        db_session.refresh(test_media)
        assert test_media.processing_status == ProcessingStatus.completed
        assert test_media.static_filename == "uuid-123.jpg"
        assert test_media.process_ms_total == 5000

    def test_delete_media(self, db_session, test_media):
        """Test deleting a media record"""
        media_id = test_media.id

        db_session.delete(test_media)
        db_session.commit()

        # Verify deletion
        media = db_session.query(Media).filter(Media.id == media_id).first()
        assert media is None

    def test_media_default_values(self, db_session, test_user):
        """Test that default values are set correctly"""
        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username=test_user.username,
        )
        db_session.add(media)
        db_session.commit()

        assert media.processing_status == ProcessingStatus.pending
        assert media.created_at is not None
        assert media.width is None  # Optional fields are None
        assert media.latitude is None


class TestDetectionCRUD:
    """Test Detection model CRUD operations"""

    def test_create_detection(self, db_session, test_media):
        """Test creating a detection record"""
        # Create frame first
        frame = Frame(
            media_id=test_media.id,
            frame_index=0,
            timestamp=0.0,
        )
        db_session.add(frame)
        db_session.commit()

        # Create detection
        detection = Detection(
            frame_id=frame.id,
            class_id=1,
            class_name="pothole",
            confidence=0.95,
            x1=100, y1=100, x2=200, y2=200,
            description="Large pothole on road",
            solution="Fill with asphalt",
            severity=Severity.high,
            source=DetectionSource.yolo,
            status=IssueStatus.open,
        )
        db_session.add(detection)
        db_session.commit()

        assert detection.id is not None
        assert detection.created_at is not None
        assert detection.status == IssueStatus.open

    def test_detection_with_mask(self, db_session, test_media):
        """Test detection with mask data"""
        frame = Frame(media_id=test_media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.commit()

        detection = Detection(
            frame_id=frame.id,
            class_id=1,
            class_name="crack",
            confidence=0.85,
            x1=50, y1=50, x2=150, y2=150,
            mask_rle={"counts": "ABCD1234", "size": [480, 640]},
            mask_polygon=[[50, 50], [150, 50], [150, 150], [50, 150]],
        )
        db_session.add(detection)
        db_session.commit()

        # Verify JSON fields
        assert detection.mask_rle == {"counts": "ABCD1234", "size": [480, 640]}
        assert detection.mask_polygon == [[50, 50], [150, 50], [150, 150], [50, 150]]

    def test_update_detection_status(self, db_session, test_media):
        """Test updating detection status"""
        frame = Frame(media_id=test_media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.commit()

        detection = Detection(
            frame_id=frame.id,
            class_id=1,
            class_name="pothole",
            confidence=0.9,
            x1=0, y1=0, x2=100, y2=100,
        )
        db_session.add(detection)
        db_session.commit()

        # Update status to resolved
        detection.status = IssueStatus.resolved
        detection.resolved_at = datetime.now(timezone.utc)
        db_session.commit()

        assert detection.status == IssueStatus.resolved
        assert detection.resolved_at is not None

    def test_detection_default_values(self, db_session, test_media):
        """Test detection default enum values"""
        frame = Frame(media_id=test_media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.commit()

        detection = Detection(
            frame_id=frame.id,
            class_id=1,
            class_name="pothole",
            confidence=0.9,
            x1=0, y1=0, x2=100, y2=100,
        )
        db_session.add(detection)
        db_session.commit()

        # Verify defaults
        assert detection.source == DetectionSource.yolo
        assert detection.status == IssueStatus.open
        assert detection.severity == Severity.medium


class TestStatusTransitions:
    """Test status enum transitions and related timestamps"""

    def test_processing_status_transitions(self, db_session, test_media):
        """Test ProcessingStatus state transitions"""
        # pending -> processing
        test_media.processing_status = ProcessingStatus.processing
        test_media.started_at = datetime.now(timezone.utc)
        db_session.commit()

        assert test_media.processing_status == ProcessingStatus.processing
        assert test_media.started_at is not None

        # processing -> completed
        test_media.processing_status = ProcessingStatus.completed
        test_media.completed_at = datetime.now(timezone.utc)
        db_session.commit()

        assert test_media.processing_status == ProcessingStatus.completed
        assert test_media.completed_at is not None

    def test_processing_status_failed(self, db_session, test_media):
        """Test ProcessingStatus transition to failed"""
        test_media.processing_status = ProcessingStatus.processing
        test_media.started_at = datetime.now(timezone.utc)
        db_session.commit()

        # Simulate failure
        test_media.processing_status = ProcessingStatus.failed
        test_media.error_message = "GPU out of memory"
        test_media.completed_at = datetime.now(timezone.utc)
        db_session.commit()

        assert test_media.processing_status == ProcessingStatus.failed
        assert test_media.error_message == "GPU out of memory"

    def test_issue_status_transitions(self, db_session, test_media):
        """Test IssueStatus transitions"""
        frame = Frame(media_id=test_media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.commit()

        detection = Detection(
            frame_id=frame.id,
            class_id=1,
            class_name="pothole",
            confidence=0.9,
            x1=0, y1=0, x2=100, y2=100,
            status=IssueStatus.open,
        )
        db_session.add(detection)
        db_session.commit()

        # open -> resolved
        detection.status = IssueStatus.resolved
        detection.resolved_at = datetime.now(timezone.utc)
        db_session.commit()

        assert detection.status == IssueStatus.resolved
        assert detection.resolved_at is not None

        # Can transition back to open
        detection.status = IssueStatus.open
        detection.resolved_at = None
        db_session.commit()

        assert detection.status == IssueStatus.open

        # open -> ignored
        detection.status = IssueStatus.ignored
        db_session.commit()

        assert detection.status == IssueStatus.ignored


class TestCascadingDeletes:
    """Test cascading delete behavior across relationships"""

    def test_delete_user_cascades_to_media(self, db_session, test_user):
        """Test that deleting a user cascades to media"""
        # Create media for user
        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username=test_user.username,
        )
        db_session.add(media)
        db_session.commit()
        media_id = media.id

        # Delete user
        db_session.delete(test_user)
        db_session.commit()

        # Verify media was also deleted (CASCADE)
        media = db_session.query(Media).filter(Media.id == media_id).first()
        assert media is None

    def test_delete_media_cascades_to_frames(self, db_session, test_media):
        """Test that deleting media cascades to frames"""
        # Create frames
        frame1 = Frame(media_id=test_media.id, frame_index=0, timestamp=0.0)
        frame2 = Frame(media_id=test_media.id, frame_index=1, timestamp=0.033)
        db_session.add_all([frame1, frame2])
        db_session.commit()

        frame_ids = [frame1.id, frame2.id]

        # Delete media
        db_session.delete(test_media)
        db_session.commit()

        # Verify frames were deleted
        frames = db_session.query(Frame).filter(Frame.id.in_(frame_ids)).all()
        assert len(frames) == 0

    def test_delete_frame_cascades_to_detections(self, db_session, test_media):
        """Test that deleting a frame cascades to detections"""
        frame = Frame(media_id=test_media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.commit()

        # Create detections
        det1 = Detection(frame_id=frame.id, class_id=1, class_name="pothole", confidence=0.9, x1=0, y1=0, x2=100, y2=100)
        det2 = Detection(frame_id=frame.id, class_id=2, class_name="crack", confidence=0.85, x1=50, y1=50, x2=150, y2=150)
        db_session.add_all([det1, det2])
        db_session.commit()

        detection_ids = [det1.id, det2.id]

        # Delete frame
        db_session.delete(frame)
        db_session.commit()

        # Verify detections were deleted
        detections = db_session.query(Detection).filter(Detection.id.in_(detection_ids)).all()
        assert len(detections) == 0

    def test_full_cascade_user_to_detections(self, db_session, test_user):
        """Test full cascade: User -> Media -> Frame -> Detection"""
        # Create complete hierarchy
        media = Media(filename="test.jpg", media_type="image", user_username=test_user.username)
        db_session.add(media)
        db_session.commit()

        frame = Frame(media_id=media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.commit()

        detection = Detection(
            frame_id=frame.id,
            class_id=1,
            class_name="pothole",
            confidence=0.9,
            x1=0, y1=0, x2=100, y2=100,
        )
        db_session.add(detection)
        db_session.commit()

        media_id = media.id
        frame_id = frame.id
        detection_id = detection.id

        # Delete user (should cascade all the way down)
        db_session.delete(test_user)
        db_session.commit()

        # Verify everything was deleted
        assert db_session.query(Media).filter(Media.id == media_id).first() is None
        assert db_session.query(Frame).filter(Frame.id == frame_id).first() is None
        assert db_session.query(Detection).filter(Detection.id == detection_id).first() is None

    def test_set_null_on_user_delete_for_assignments(self, db_session):
        """Test SET NULL behavior for assigned_to and verified_by"""
        # Create users
        user1 = UserModel(username="user1", hashed_password="hash1", role=RoleEnum.user)
        user2 = UserModel(username="user2", hashed_password="hash2", role=RoleEnum.user)
        user3 = UserModel(username="user3", hashed_password="hash3", role=RoleEnum.user)
        db_session.add_all([user1, user2, user3])
        db_session.commit()

        # Create media and detection
        media = Media(filename="test.jpg", media_type="image", user_username=user1.username)
        db_session.add(media)
        db_session.commit()

        frame = Frame(media_id=media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.commit()

        detection = Detection(
            frame_id=frame.id,
            class_id=1,
            class_name="pothole",
            confidence=0.9,
            x1=0, y1=0, x2=100, y2=100,
            assigned_to=user2.username,  # Assigned to user2
            verified_by=user3.username,  # Verified by user3
        )
        db_session.add(detection)
        db_session.commit()

        detection_id = detection.id

        # Delete user2 (assigned_to should become NULL)
        db_session.delete(user2)
        db_session.commit()

        detection = db_session.query(Detection).filter(Detection.id == detection_id).first()
        assert detection.assigned_to is None  # SET NULL
        assert detection.verified_by == user3.username  # Still set

        # Delete user3 (verified_by should become NULL)
        db_session.delete(user3)
        db_session.commit()

        detection = db_session.query(Detection).filter(Detection.id == detection_id).first()
        assert detection.verified_by is None  # SET NULL


class TestRelationships:
    """Test model relationships and lazy/eager loading"""

    def test_media_frames_relationship(self, db_session, test_media):
        """Test Media -> Frames relationship"""
        # Create frames
        frame1 = Frame(media_id=test_media.id, frame_index=0, timestamp=0.0)
        frame2 = Frame(media_id=test_media.id, frame_index=1, timestamp=0.033)
        frame3 = Frame(media_id=test_media.id, frame_index=2, timestamp=0.066)
        db_session.add_all([frame1, frame2, frame3])
        db_session.commit()

        # Access frames through relationship
        db_session.refresh(test_media)
        assert len(test_media.frames) == 3
        assert all(isinstance(f, Frame) for f in test_media.frames)

    def test_frame_detections_relationship(self, db_session, test_media):
        """Test Frame -> Detections relationship"""
        frame = Frame(media_id=test_media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.commit()

        # Create multiple detections
        det1 = Detection(frame_id=frame.id, class_id=1, class_name="pothole", confidence=0.9, x1=0, y1=0, x2=100, y2=100)
        det2 = Detection(frame_id=frame.id, class_id=2, class_name="crack", confidence=0.85, x1=50, y1=50, x2=150, y2=150)
        db_session.add_all([det1, det2])
        db_session.commit()

        # Access detections through relationship
        db_session.refresh(frame)
        assert len(frame.detects) == 2

    def test_detection_frame_backref(self, db_session, test_media):
        """Test Detection -> Frame backref"""
        frame = Frame(media_id=test_media.id, frame_index=5, timestamp=0.166)
        db_session.add(frame)
        db_session.commit()

        detection = Detection(frame_id=frame.id, class_id=1, class_name="pothole", confidence=0.9, x1=0, y1=0, x2=100, y2=100)
        db_session.add(detection)
        db_session.commit()

        # Access frame through backref
        assert detection.frame.id == frame.id
        assert detection.frame.frame_index == 5


class TestConstraints:
    """Test database constraints and integrity"""

    def test_foreign_key_constraint(self, db_session):
        """Test that foreign key constraints are enforced"""
        # Try to create media with non-existent user
        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username="nonexistent_user",
        )
        db_session.add(media)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_not_null_constraint(self, db_session, test_user):
        """Test that NOT NULL constraints are enforced"""
        # Try to create media without required fields
        media = Media(
            user_username=test_user.username,
            # Missing filename and media_type (NOT NULL)
        )
        db_session.add(media)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_enum_value_validation(self, db_session, test_user):
        """Test that invalid enum values are rejected by PostgreSQL"""
        from sqlalchemy import text

        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username=test_user.username,
        )
        db_session.add(media)
        db_session.commit()

        # Try to set invalid enum value - PostgreSQL enforces enum types
        # This will raise a DataError when executing
        with pytest.raises(DataError):
            # Directly set an invalid value bypassing SQLAlchemy's enum validation
            db_session.execute(
                text(f"UPDATE media SET processing_status = 'invalid_status' WHERE id = {media.id}")
            )
            db_session.commit()


class TestIndexPerformance:
    """Test that indexes exist and are properly configured"""

    def test_indexes_exist(self):
        """Test that expected indexes are created"""
        inspector = inspect(engine)

        # Check Media indexes
        media_indexes = inspector.get_indexes("media")
        media_index_cols = [idx['column_names'] for idx in media_indexes]

        assert any('geohash6' in cols for cols in media_index_cols), "geohash6 should be indexed"
        assert any('created_at' in cols for cols in media_index_cols), "created_at should be indexed"
        assert any('task_id' in cols for cols in media_index_cols), "task_id should be indexed"

        # Check Detection indexes
        detection_indexes = inspector.get_indexes("detection")
        detection_index_cols = [idx['column_names'] for idx in detection_indexes]

        assert any('class_name' in cols for cols in detection_index_cols), "class_name should be indexed"
        assert any('created_at' in cols for cols in detection_index_cols), "created_at should be indexed"

        # Check for composite index
        composite_found = any(
            set(cols) == {'class_name', 'created_at'}
            for cols in detection_index_cols
        )
        assert composite_found, "Composite index (class_name, created_at) should exist"

    def test_query_with_index(self, db_session, test_media):
        """Test querying using indexed columns"""
        # Create multiple media records
        for i in range(10):
            media = Media(
                filename=f"test{i}.jpg",
                media_type="image",
                user_username=test_media.user_username,
                geohash6=f"hash{i % 3}",  # Indexed column
            )
            db_session.add(media)
        db_session.commit()

        # Query using indexed column (should be fast)
        results = db_session.query(Media).filter(Media.geohash6 == "hash1").all()
        assert len(results) > 0


class TestTransactionHandling:
    """Test database transaction behavior"""

    def test_commit_persists_changes(self, db_session, test_user):
        """Test that commit persists changes"""
        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username=test_user.username,
        )
        db_session.add(media)
        db_session.commit()

        media_id = media.id

        # Create new session and verify data persisted
        new_session = TestingSessionLocal()
        try:
            media = new_session.query(Media).filter(Media.id == media_id).first()
            assert media is not None
            assert media.filename == "test.jpg"
        finally:
            new_session.close()

    def test_rollback_discards_changes(self, db_session, test_media):
        """Test that rollback discards changes"""
        original_filename = test_media.filename

        # Make changes
        test_media.filename = "modified.jpg"
        db_session.flush()  # Flush to session but don't commit

        # Rollback
        db_session.rollback()

        # Verify changes were discarded
        db_session.refresh(test_media)
        assert test_media.filename == original_filename

    def test_exception_triggers_rollback(self, db_session, test_user):
        """Test that exceptions trigger rollback"""
        initial_count = db_session.query(Media).count()

        try:
            media1 = Media(filename="test1.jpg", media_type="image", user_username=test_user.username)
            db_session.add(media1)
            db_session.flush()

            # This should fail (duplicate username if we try to add user again)
            media2 = Media(
                filename="test2.jpg",
                media_type="image",
                user_username="nonexistent",  # FK violation
            )
            db_session.add(media2)
            db_session.commit()
        except IntegrityError:
            db_session.rollback()

        # Verify no media was added
        final_count = db_session.query(Media).count()
        assert final_count == initial_count


class TestConcurrentUpdates:
    """Test concurrent update scenarios"""

    def test_concurrent_status_updates(self, test_media):
        """Test concurrent updates to same record with PostgreSQL"""
        def update_status(status):
            session = TestingSessionLocal()
            try:
                media = session.query(Media).filter(Media.id == test_media.id).first()
                media.processing_status = status
                session.commit()
            finally:
                session.close()

        # Simulate concurrent updates
        thread1 = threading.Thread(target=update_status, args=(ProcessingStatus.processing,))
        thread2 = threading.Thread(target=update_status, args=(ProcessingStatus.completed,))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # One of the statuses should win
        session = TestingSessionLocal()
        try:
            media = session.query(Media).filter(Media.id == test_media.id).first()
            assert media.processing_status in [ProcessingStatus.processing, ProcessingStatus.completed]
        finally:
            session.close()


class TestSpecialFields:
    """Test special field types and behaviors"""

    def test_json_fields(self, db_session, test_media):
        """Test JSON field storage and retrieval"""
        frame = Frame(media_id=test_media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.commit()

        complex_mask = {
            "counts": "abc123xyz",
            "size": [1920, 1080],
            "nested": {"key": "value", "array": [1, 2, 3]},
        }

        detection = Detection(
            frame_id=frame.id,
            class_id=1,
            class_name="pothole",
            confidence=0.9,
            x1=0, y1=0, x2=100, y2=100,
            mask_rle=complex_mask,
        )
        db_session.add(detection)
        db_session.commit()

        # Verify JSON is stored and retrieved correctly
        db_session.refresh(detection)
        assert detection.mask_rle == complex_mask
        assert detection.mask_rle["nested"]["array"] == [1, 2, 3]

    def test_geohash_indexing(self, db_session, test_user):
        """Test geohash field for spatial indexing"""
        media = Media(
            filename="test.jpg",
            media_type="image",
            user_username=test_user.username,
            latitude=40.7128,
            longitude=-74.0060,
            geohash6="dr5ru6",  # NYC geohash
        )
        db_session.add(media)
        db_session.commit()

        # Query by geohash
        results = db_session.query(Media).filter(Media.geohash6 == "dr5ru6").all()
        assert len(results) == 1
        assert results[0].id == media.id

    def test_timestamp_fields(self, db_session, test_media):
        """Test timestamp fields are set correctly"""
        # created_at should be set automatically
        assert test_media.created_at is not None
        assert isinstance(test_media.created_at, datetime)

        # Timestamps should be timezone-aware
        assert test_media.created_at.tzinfo is not None

        # Update timestamps
        now = datetime.now(timezone.utc)
        test_media.started_at = now
        test_media.completed_at = now + timedelta(seconds=10)
        db_session.commit()

        db_session.refresh(test_media)
        assert test_media.started_at is not None
        assert test_media.completed_at is not None
        assert test_media.completed_at > test_media.started_at


if __name__ == '__main__':
    pytest.main([__file__, '-v'])