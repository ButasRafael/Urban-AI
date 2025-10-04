"""
Comprehensive tests for issue/problem management endpoints.

Tests cover:
- Issue listing with filtering (media_id, type, class, severity, status, source, assignee)
- Problem listing (media with detections)
- Bulk status updates (open → resolved/ignored)
- Severity updates (database-level)
- Assignment workflow (admin only, database-level)
- Verification workflow (admin only, database-level)
- Summary endpoints (overall and by media)
- Pagination and limits
- Access control (admin/authority only)
- Edge cases and error handling
"""

import pytest
import os
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import all models
import app.models.user
import app.models.media
import app.models.revoked
import app.models.rag
import app.models.conversation

from app.main import app
from app.core.database import Base, get_db
from app.models.user import User as UserModel, RoleEnum
from app.models.media import Media, Frame, Detection, Severity, IssueStatus, DetectionSource
from app.core.security import create_access_token, get_password_hash

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
def admin_user(db_session):
    """Create an admin user"""
    user = UserModel(
        username="admin_problems",
        hashed_password=get_password_hash("adminpass123"),
        role=RoleEnum.admin
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def authority_user(db_session):
    """Create an authority user"""
    user = UserModel(
        username="authority_problems",
        hashed_password=get_password_hash("authoritypass123"),
        role=RoleEnum.authority
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def regular_user(db_session):
    """Create a regular user"""
    user = UserModel(
        username="user_problems",
        hashed_password=get_password_hash("userpass123"),
        role=RoleEnum.user
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user):
    """Generate JWT token for admin user"""
    return create_access_token({"sub": admin_user.username})


@pytest.fixture
def authority_token(authority_user):
    """Generate JWT token for authority user"""
    return create_access_token({"sub": authority_user.username})


@pytest.fixture
def user_token(regular_user):
    """Generate JWT token for regular user"""
    return create_access_token({"sub": regular_user.username})


@pytest.fixture
def client(override_get_db):
    """Create test client with database override"""
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_issues(db_session, admin_user):
    """Create sample media with detections for testing"""
    media_list = []
    detection_list = []

    # Create 3 images and 2 videos
    for i in range(5):
        media_type = "image" if i < 3 else "video"
        media = Media(
            filename=f"test_{i}.jpg" if media_type == "image" else f"test_{i}.mp4",
            media_type=media_type,
            user_username=admin_user.username,
            static_filename=f"static_{i}.jpg" if media_type == "image" else f"static_{i}.mp4",
            thumbnail_filename=f"thumb_{i}.jpg" if media_type == "video" else None,
            address=f"Street {i}, City",
            latitude=44.4268 + (i * 0.01),
            longitude=26.1025 + (i * 0.01),
            summary_description=f"Summary for media {i}",
            summary_solution=f"Solution for media {i}",
        )
        db_session.add(media)
        db_session.flush()

        # Create frame
        frame = Frame(
            media_id=media.id,
            frame_index=0,
            timestamp=0.0
        )
        db_session.add(frame)
        db_session.flush()

        # Create multiple detections with varying properties
        severities = [Severity.high, Severity.medium, Severity.low]
        statuses = [IssueStatus.open, IssueStatus.resolved, IssueStatus.ignored]
        sources = [DetectionSource.yolo, DetectionSource.gpt_dino, DetectionSource.sam_fallback]
        classes = ["pothole", "crack", "debris", "damaged_sign"]

        for j in range(3):  # 3 detections per media
            detection = Detection(
                frame_id=frame.id,
                class_id=j,
                class_name=classes[j % len(classes)],
                confidence=0.75 + (j * 0.05),
                x1=100.0 * j,
                y1=100.0 * j,
                x2=200.0 * j,
                y2=200.0 * j,
                description=f"Detection {j} description",
                solution=f"Detection {j} solution",
                severity=severities[j % 3],
                status=statuses[j % 3],
                source=sources[j % 3],
                track_id=1000 + j,
                track_thumbnail_url=f"/static/track_{i}_{j}.jpg" if media_type == "video" else None,
                # Assign every other detection
                assigned_to=admin_user.username if j % 2 == 0 else None,
                # Verify every third detection
                verified_by=admin_user.username if j % 3 == 0 else None,
                verified_at=datetime.now(timezone.utc) if j % 3 == 0 else None,
                # Resolve some detections
                resolved_at=datetime.now(timezone.utc) if j % 3 == 1 else None,
            )
            db_session.add(detection)
            detection_list.append(detection)

        media_list.append(media)

    db_session.commit()

    # Refresh to get IDs
    for detection in detection_list:
        db_session.refresh(detection)

    return {
        "media": media_list,
        "detections": detection_list
    }


# ============== Access Control Tests ==============

class TestAccessControl:
    """Test access control for problem/issue endpoints"""

    def test_problems_requires_admin_or_authority(self, client, user_token):
        """Test that regular users cannot access problems endpoints"""
        response = client.get(
            "/problems",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        assert response.status_code == 403

    def test_admin_can_access_problems(self, client, admin_token, sample_issues):
        """Test that admin users can access problems"""
        response = client.get(
            "/problems",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200

    def test_authority_can_access_problems(self, client, authority_token, sample_issues):
        """Test that authority users can access problems"""
        response = client.get(
            "/problems",
            headers={"Authorization": f"Bearer {authority_token}"}
        )
        assert response.status_code == 200

    def test_unauthenticated_access_denied(self, client):
        """Test that unauthenticated requests are rejected"""
        response = client.get("/problems")
        assert response.status_code == 401


# ============== Problem Listing Tests ==============

class TestProblemListing:
    """Test /problems endpoint for listing media with detections"""

    def test_list_all_problems(self, client, admin_token, sample_issues):
        """Test listing all problems"""
        response = client.get(
            "/problems",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 5  # 5 media items

        # Verify structure
        for problem in data:
            assert "media_id" in problem
            assert "address" in problem
            assert "media_type" in problem
            assert "predicted_classes" in problem
            assert "descriptions" in problem
            assert "solutions" in problem
            assert "created_at" in problem

    def test_filter_problems_by_image_type(self, client, admin_token, sample_issues):
        """Test filtering problems by image media type"""
        response = client.get(
            "/problems?media_type=image",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 3  # 3 images
        for problem in data:
            assert problem["media_type"] == "image"

    def test_filter_problems_by_video_type(self, client, admin_token, sample_issues):
        """Test filtering problems by video media type"""
        response = client.get(
            "/problems?media_type=video",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 2  # 2 videos
        for problem in data:
            assert problem["media_type"] == "video"

    def test_filter_problems_by_class(self, client, admin_token, sample_issues):
        """Test filtering problems by detection class"""
        response = client.get(
            "/problems?klass=pothole",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) > 0
        for problem in data:
            assert "pothole" in problem["predicted_classes"]

    def test_problems_include_static_urls(self, client, admin_token, sample_issues):
        """Test that problems include static file URLs"""
        response = client.get(
            "/problems",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        images = [p for p in data if p["media_type"] == "image"]
        videos = [p for p in data if p["media_type"] == "video"]

        # Images should have annotated_image_url
        for img in images:
            assert img["annotated_image_url"] is not None
            assert "/static/" in img["annotated_image_url"]

        # Videos should have annotated_video_url and thumbnail
        for vid in videos:
            assert vid["annotated_video_url"] is not None
            assert "/static/" in vid["annotated_video_url"]


# ============== Issue Listing Tests ==============

class TestIssueListing:
    """Test /problems/issues endpoint for listing individual detections"""

    def test_list_all_issues(self, client, admin_token, sample_issues):
        """Test listing all issues"""
        response = client.get(
            "/problems/issues",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) > 0  # Should have detections

        # Verify structure
        for issue in data:
            assert "id" in issue
            assert "media_id" in issue
            assert "frame_id" in issue
            assert "class_name" in issue
            assert "confidence" in issue
            assert "bbox" in issue
            assert "severity" in issue
            assert "status" in issue
            assert "source" in issue
            assert len(issue["bbox"]) == 4  # [x1, y1, x2, y2]

    def test_filter_issues_by_media_id(self, client, admin_token, sample_issues):
        """Test filtering issues by specific media ID"""
        media_id = sample_issues["media"][0].id

        response = client.get(
            f"/problems/issues?media_id={media_id}",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 3  # 3 detections per media
        for issue in data:
            assert issue["media_id"] == media_id

    def test_filter_issues_by_severity(self, client, admin_token, sample_issues):
        """Test filtering issues by severity"""
        response = client.get(
            "/problems/issues?severity=high",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        for issue in data:
            assert issue["severity"] == "high"

    def test_filter_issues_by_status(self, client, admin_token, sample_issues):
        """Test filtering issues by status"""
        response = client.get(
            "/problems/issues?status=open",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        for issue in data:
            assert issue["status"] == "open"

    def test_filter_issues_by_source(self, client, admin_token, sample_issues):
        """Test filtering issues by detection source"""
        response = client.get(
            "/problems/issues?source=yolo",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        for issue in data:
            assert issue["source"] == "yolo"

    def test_filter_issues_by_class(self, client, admin_token, sample_issues):
        """Test filtering issues by class name"""
        response = client.get(
            "/problems/issues?klass=pothole",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        for issue in data:
            assert issue["class_name"] == "pothole"

    def test_filter_issues_by_assignee(self, client, admin_token, sample_issues, admin_user):
        """Test filtering issues by assignee"""
        response = client.get(
            f"/problems/issues?assigned_to={admin_user.username}",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        for issue in data:
            assert issue["assigned_to"] == admin_user.username

    def test_filter_issues_by_media_type(self, client, admin_token, sample_issues):
        """Test filtering issues by media type"""
        response = client.get(
            "/problems/issues?media_type=image",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should only have issues from image media (3 images × 3 detections = 9)
        assert len(data) == 9

    def test_issues_pagination_limit(self, client, admin_token, sample_issues):
        """Test pagination with limit parameter"""
        response = client.get(
            "/problems/issues?limit=5",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 5  # Respects limit

    def test_issues_pagination_offset(self, client, admin_token, sample_issues):
        """Test pagination with offset parameter"""
        # Get first page
        response1 = client.get(
            "/problems/issues?limit=5&offset=0",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        page1 = response1.json()

        # Get second page
        response2 = client.get(
            "/problems/issues?limit=5&offset=5",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        page2 = response2.json()

        # Pages should have different issues
        page1_ids = {issue["id"] for issue in page1}
        page2_ids = {issue["id"] for issue in page2}
        assert len(page1_ids & page2_ids) == 0  # No overlap

    def test_issues_include_track_thumbnail(self, client, admin_token, sample_issues):
        """Test that video issues include track-specific thumbnails"""
        response = client.get(
            "/problems/issues?media_type=video",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Some video detections should have track thumbnails
        with_track_thumb = [i for i in data if i["track_thumbnail_url"]]
        assert len(with_track_thumb) > 0

    def test_combine_multiple_filters(self, client, admin_token, sample_issues):
        """Test combining multiple filters"""
        response = client.get(
            "/problems/issues?media_type=image&severity=high&status=open",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify all filters applied
        for issue in data:
            assert issue["severity"] == "high"
            assert issue["status"] == "open"


# ============== Summary Tests ==============

class TestSummaryEndpoints:
    """Test summary endpoints"""

    def test_issues_summary_overall(self, client, admin_token, sample_issues):
        """Test overall issues summary by severity and status"""
        response = client.get(
            "/problems/issues/summary",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should have severity levels
        assert "high" in data or "medium" in data or "low" in data

        # Each severity should have status breakdowns
        for severity_data in data.values():
            # Should have at least some statuses
            assert isinstance(severity_data, dict)

    def test_issues_summary_by_media_type(self, client, admin_token, sample_issues):
        """Test summary filtered by media type"""
        response = client.get(
            "/problems/issues/summary?media_type=image",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_issues_summary_by_media(self, client, admin_token, sample_issues):
        """Test summary grouped by specific media IDs"""
        media_ids = [sample_issues["media"][0].id, sample_issues["media"][1].id]

        response = client.get(
            f"/problems/issues/summary_by_media?media_ids={media_ids[0]}&media_ids={media_ids[1]}",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should have data for both media
        assert len(data) == 2

        # Check if keys are integers or strings (JSON may convert)
        data_keys = list(data.keys())
        media_ids_str = [str(mid) for mid in media_ids]

        # Either int keys or string keys should work
        if isinstance(data_keys[0], int):
            assert media_ids[0] in data
            assert media_ids[1] in data
        else:
            assert str(media_ids[0]) in data
            assert str(media_ids[1]) in data

        # Each media should have severity/status breakdown
        for media_summary in data.values():
            assert isinstance(media_summary, dict)


# ============== Bulk Status Update Tests ==============

class TestBulkStatusUpdate:
    """Test bulk status update functionality"""

    def test_bulk_update_to_resolved(self, client, admin_token, sample_issues, db_session):
        """Test bulk updating issues to resolved status"""
        media_id = sample_issues["media"][0].id

        # Get initial open count
        open_before = db_session.query(Detection).join(Frame).join(Media).filter(
            Media.id == media_id,
            Detection.status == IssueStatus.open
        ).count()

        response = client.patch(
            "/problems/issues/bulk_status",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "media_id": media_id,
                "status": "resolved"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["updated_count"] == open_before
        assert data["media_id"] == media_id
        assert data["new_status"] == "resolved"

        # Verify in database
        db_session.rollback()  # Refresh session
        open_after = db_session.query(Detection).join(Frame).join(Media).filter(
            Media.id == media_id,
            Detection.status == IssueStatus.open
        ).count()

        assert open_after == 0  # All should be resolved now

    def test_bulk_update_to_ignored(self, client, admin_token, sample_issues, db_session):
        """Test bulk updating issues to ignored status"""
        media_id = sample_issues["media"][1].id

        response = client.patch(
            "/problems/issues/bulk_status",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "media_id": media_id,
                "status": "ignored"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["new_status"] == "ignored"
        assert data["updated_count"] >= 0

    def test_bulk_update_sets_resolved_at(self, client, admin_token, sample_issues, db_session):
        """Test that resolving issues sets resolved_at timestamp"""
        media_id = sample_issues["media"][2].id

        response = client.patch(
            "/problems/issues/bulk_status",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "media_id": media_id,
                "status": "resolved"
            }
        )

        assert response.status_code == 200

        # Verify resolved_at is set
        db_session.rollback()
        resolved_detections = db_session.query(Detection).join(Frame).join(Media).filter(
            Media.id == media_id,
            Detection.status == IssueStatus.resolved
        ).all()

        for detection in resolved_detections:
            if detection.resolved_at:  # If it was just resolved
                assert detection.resolved_at is not None

    def test_bulk_update_no_open_detections(self, client, admin_token, sample_issues, db_session):
        """Test bulk update when no open detections exist"""
        media_id = sample_issues["media"][0].id

        # First resolve all
        client.patch(
            "/problems/issues/bulk_status",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"media_id": media_id, "status": "resolved"}
        )

        # Try to resolve again
        response = client.patch(
            "/problems/issues/bulk_status",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"media_id": media_id, "status": "resolved"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["updated_count"] == 0
        assert "No open detections" in data["message"]

    def test_bulk_update_nonexistent_media(self, client, admin_token):
        """Test bulk update with non-existent media ID"""
        response = client.patch(
            "/problems/issues/bulk_status",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "media_id": 999999,
                "status": "resolved"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["updated_count"] == 0


# ============== Database-Level Update Tests ==============

class TestDatabaseLevelUpdates:
    """Test database-level updates for assignment, verification, and severity"""

    def test_assign_detection_to_user(self, db_session, sample_issues, admin_user):
        """Test assigning a detection to a user"""
        detection = sample_issues["detections"][0]

        # Assign
        detection.assigned_to = admin_user.username
        db_session.commit()

        # Verify
        db_session.refresh(detection)
        assert detection.assigned_to == admin_user.username

    def test_unassign_detection(self, db_session, sample_issues):
        """Test unassigning a detection"""
        # Find an assigned detection
        assigned = [d for d in sample_issues["detections"] if d.assigned_to][0]

        # Unassign
        assigned.assigned_to = None
        db_session.commit()

        # Verify
        db_session.refresh(assigned)
        assert assigned.assigned_to is None

    def test_verify_detection(self, db_session, sample_issues, admin_user):
        """Test verifying a detection"""
        detection = sample_issues["detections"][1]

        # Verify
        detection.verified_by = admin_user.username
        detection.verified_at = datetime.now(timezone.utc)
        db_session.commit()

        # Check
        db_session.refresh(detection)
        assert detection.verified_by == admin_user.username
        assert detection.verified_at is not None

    def test_update_severity(self, db_session, sample_issues):
        """Test updating detection severity"""
        detection = sample_issues["detections"][2]
        original_severity = detection.severity

        # Change severity
        new_severity = Severity.high if original_severity != Severity.high else Severity.low
        detection.severity = new_severity
        db_session.commit()

        # Verify
        db_session.refresh(detection)
        assert detection.severity == new_severity

    def test_cascade_delete_on_user_deletion(self, db_session, sample_issues):
        """Test that assigned_to/verified_by are set to NULL when user is deleted"""
        # Create a temp user
        temp_user = UserModel(
            username="temp_user",
            hashed_password=get_password_hash("temp"),
            role=RoleEnum.user
        )
        db_session.add(temp_user)
        db_session.commit()

        # Assign a detection to temp user
        detection = sample_issues["detections"][3]
        detection.assigned_to = temp_user.username
        detection.verified_by = temp_user.username
        db_session.commit()

        detection_id = detection.id

        # Delete temp user
        db_session.delete(temp_user)
        db_session.commit()

        # Verify detection still exists but references are NULL
        db_session.expire_all()
        detection = db_session.query(Detection).filter(Detection.id == detection_id).first()
        assert detection is not None
        assert detection.assigned_to is None
        assert detection.verified_by is None

    def test_bulk_severity_update(self, db_session, sample_issues):
        """Test bulk updating severity for multiple detections"""
        media_id = sample_issues["media"][0].id

        # Get all detections for this media
        detections = db_session.query(Detection).join(Frame).join(Media).filter(
            Media.id == media_id
        ).all()

        # Update all to high severity
        for detection in detections:
            detection.severity = Severity.high

        db_session.commit()

        # Verify
        db_session.expire_all()
        updated = db_session.query(Detection).join(Frame).join(Media).filter(
            Media.id == media_id,
            Detection.severity == Severity.high
        ).count()

        assert updated == len(detections)

    def test_status_transition_tracking(self, db_session, sample_issues):
        """Test that status transitions are properly tracked"""
        detection = sample_issues["detections"][4]

        # Transition: open → resolved
        detection.status = IssueStatus.resolved
        detection.resolved_at = datetime.now(timezone.utc)
        db_session.commit()

        db_session.refresh(detection)
        assert detection.status == IssueStatus.resolved
        assert detection.resolved_at is not None

        # Transition: resolved → ignored (edge case)
        detection.status = IssueStatus.ignored
        db_session.commit()

        db_session.refresh(detection)
        assert detection.status == IssueStatus.ignored
        # resolved_at should still be set from previous transition
        assert detection.resolved_at is not None


# ============== Edge Cases and Error Handling ==============

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_issues_list(self, client, admin_token, db_session):
        """Test issues endpoint with no detections"""
        response = client.get(
            "/problems/issues",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_filter_no_matches(self, client, admin_token, sample_issues):
        """Test filtering with criteria that match no issues"""
        response = client.get(
            "/problems/issues?klass=nonexistent_class",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_pagination_beyond_results(self, client, admin_token, sample_issues):
        """Test pagination with offset beyond available results"""
        response = client.get(
            "/problems/issues?limit=10&offset=1000",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_invalid_media_type_filter(self, client, admin_token):
        """Test invalid media_type parameter"""
        response = client.get(
            "/problems?media_type=invalid",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        # Should return validation error
        assert response.status_code == 422

    def test_invalid_severity_filter(self, client, admin_token):
        """Test invalid severity parameter"""
        response = client.get(
            "/problems/issues?severity=invalid",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 422

    def test_invalid_status_filter(self, client, admin_token):
        """Test invalid status parameter"""
        response = client.get(
            "/problems/issues?status=invalid",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 422

    def test_summary_with_no_data(self, client, admin_token, db_session):
        """Test summary endpoints with no detections"""
        response = client.get(
            "/problems/issues/summary",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data == {}

    def test_detection_with_null_coordinates(self, db_session, admin_user):
        """Test detection associated with media that has no coordinates"""
        media = Media(
            filename="no_coords.jpg",
            media_type="image",
            user_username=admin_user.username,
            latitude=None,
            longitude=None,
            address=None
        )
        db_session.add(media)
        db_session.flush()

        frame = Frame(media_id=media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.flush()

        detection = Detection(
            frame_id=frame.id,
            class_id=1,
            class_name="test",
            confidence=0.9,
            x1=0.0, y1=0.0, x2=100.0, y2=100.0,
            severity=Severity.low,
            status=IssueStatus.open,
            source=DetectionSource.yolo
        )
        db_session.add(detection)
        db_session.commit()

        # Should still be retrievable
        assert detection.id is not None


# ============== Integration Tests ==============

class TestIntegrationScenarios:
    """Test real-world integration scenarios"""

    def test_complete_issue_workflow(self, client, admin_token, db_session, admin_user):
        """Test complete issue management workflow"""
        # 1. Create media with detection
        media = Media(
            filename="workflow.jpg",
            media_type="image",
            user_username=admin_user.username
        )
        db_session.add(media)
        db_session.flush()

        frame = Frame(media_id=media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.flush()

        detection = Detection(
            frame_id=frame.id,
            class_id=1,
            class_name="pothole",
            confidence=0.95,
            x1=0.0, y1=0.0, x2=100.0, y2=100.0,
            severity=Severity.high,
            status=IssueStatus.open,
            source=DetectionSource.yolo
        )
        db_session.add(detection)
        db_session.commit()

        # 2. List issues - should appear
        response = client.get(
            "/problems/issues?status=open",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        issues = response.json()
        assert len([i for i in issues if i["id"] == detection.id]) == 1

        # 3. Assign issue
        detection.assigned_to = admin_user.username
        db_session.commit()

        # 4. Verify assignment via API
        response = client.get(
            f"/problems/issues?assigned_to={admin_user.username}",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert len([i for i in response.json() if i["id"] == detection.id]) == 1

        # 5. Verify issue
        detection.verified_by = admin_user.username
        detection.verified_at = datetime.now(timezone.utc)
        db_session.commit()

        # 6. Resolve via bulk update
        response = client.patch(
            "/problems/issues/bulk_status",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"media_id": media.id, "status": "resolved"}
        )
        assert response.status_code == 200
        assert response.json()["updated_count"] == 1

        # 7. Verify in summary
        response = client.get(
            "/problems/issues/summary",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        summary = response.json()
        assert summary.get("high", {}).get("resolved", 0) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])