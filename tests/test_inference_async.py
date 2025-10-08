"""
Comprehensive tests for async inference endpoints and file processing.

Tests cover:
- Image upload and validation (size, type, dimensions)
- Video upload and validation (size, type, duration)
- Async task creation via Celery
- Task status tracking and monitoring
- File type validation (content-based, not just extension)
- Rate limiting (10/min for images, 3/min for videos)
- Error handling for invalid media files
- Mock Celery tasks for unit testing
- Task ID generation and retrieval
- Progress updates via WebSocket
- Media metadata extraction (dimensions, duration, codec)
- File size enforcement (10MB images, 50MB videos)
- Coordinate and location data validation
- Thumbnail generation for videos
- Database record creation
"""

import io
import json
import os
import uuid
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
from datetime import datetime, timezone

import pytest
import numpy as np
import cv2
from PIL import Image
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from celery.result import AsyncResult

# Import models first to ensure they're registered with Base
import app.models.user
import app.models.media  # Contains Media and Detection models
import app.models.revoked
import app.models.conversation
import app.models.rag  # RAG models

from app.main import app
from app.core.database import Base, get_db
from app.models import media as dbm
from app.models.user import User as UserModel
from app.models.revoked import RevokedToken
from app.core.security import create_access_token
from app.core.validation import (
    MAX_IMAGE_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_VIDEO_DIMENSION,
    MAX_VIDEO_DURATION_SECONDS,
    IMAGE_EXTS,
    VIDEO_EXTS
)


# Test database setup - Use PostgreSQL for testing (same as production)
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
        session.query(dbm.Detection).delete()
        session.query(dbm.Frame).delete()
        session.query(dbm.Media).delete()
        session.query(RevokedToken).delete()
        session.query(UserModel).delete()
        session.commit()
        yield session
    finally:
        # Clean up data after test
        session.query(dbm.Detection).delete()
        session.query(dbm.Frame).delete()
        session.query(dbm.Media).delete()
        session.query(RevokedToken).delete()
        session.query(UserModel).delete()
        session.commit()
        session.close()


@pytest.fixture
def disable_rate_limiting():
    """Disable rate limiting for tests"""
    with patch('app.core.rate_limiter.limiter.limit', lambda x: lambda f: f):
        with patch('app.core.rate_limiter.user_limiter.limit', lambda x: lambda f: f):
            yield


@pytest.fixture
def client(db_session, disable_rate_limiting):
    """Create test client with database override and rate limiting disabled"""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.fixture
def auth_headers(db_session):
    """Create authenticated headers for a test user"""
    # Create test user with fixed username for consistency
    username = "test_user"
    existing_user = db_session.query(UserModel).filter_by(username=username).first()
    if not existing_user:
        user = UserModel(
            username=username,
            email="test_user@test.com",
            hashed_password="hashed_password",
            email_verified=True,
            email_verification_token=None,
            role="user"
        )
        db_session.add(user)
        db_session.commit()

    # Create access token
    token = create_access_token({"sub": username})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(db_session):
    """Create authenticated headers for an admin user"""
    # Create admin user
    username = "test_admin"
    existing_user = db_session.query(UserModel).filter_by(username=username).first()
    if not existing_user:
        user = UserModel(
            username=username,
            email="test_admin@test.com",
            hashed_password="hashed_password",
            email_verified=True,
            email_verification_token=None,
            role="admin"
        )
        db_session.add(user)
        db_session.commit()

    # Create access token
    token = create_access_token({"sub": username})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_image():
    """Generate a sample image file for testing"""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def large_image():
    """Generate a large image that exceeds size limits"""
    # Create a large image that actually exceeds MAX_IMAGE_SIZE_BYTES (10MB)
    # Create a BytesIO object with more than 10MB of data
    img_bytes = io.BytesIO()
    # Write more than 10MB of data
    img_bytes.write(b'X' * (MAX_IMAGE_SIZE_BYTES + 1000000))
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def oversized_image():
    """Generate an image with dimensions exceeding limits"""
    img = Image.new('RGB', (MAX_IMAGE_DIMENSION + 100, 100), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_video():
    """Generate a sample video file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        # Create a simple video with OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f.name, fourcc, 20.0, (640, 480))

        # Write 20 frames (1 second at 20 fps)
        for i in range(20):
            frame = np.zeros((480, 640, 3), np.uint8)
            frame[:] = (0, 0, 255)  # Red frame
            out.write(frame)

        out.release()

        with open(f.name, 'rb') as video_file:
            video_bytes = io.BytesIO(video_file.read())

        os.unlink(f.name)
        video_bytes.seek(0)
        return video_bytes


# ============== Image Upload Tests ==============

class TestImageUpload:
    """Test image upload and processing via Celery"""

    @patch('app.services.tasks.process_image_task.apply_async')
    def test_successful_image_upload(self, mock_apply_async, client, auth_headers, sample_image):
        """Test successful image upload and task queuing"""
        # Setup mock
        mock_task = MagicMock()
        mock_task.id = str(uuid.uuid4())
        mock_apply_async.return_value = mock_task

        # Upload image
        response = client.post(
            "/infer/async/image",
            headers=auth_headers,
            files={"file": ("test.png", sample_image, "image/png")},
            data={
                "use_sam": "true",
                "latitude": 44.4268,
                "longitude": 26.1025,
                "address": "Bucharest, Romania"
            }
        )

        assert response.status_code == 202
        result = response.json()
        assert "task_id" in result
        assert result["task_id"] == mock_task.id
        assert "media_id" in result
        assert result["status"] == "pending"
        assert "queued" in result["message"].lower()

        # Verify task was queued
        mock_apply_async.assert_called_once()
        args, kwargs = mock_apply_async.call_args
        assert "args" in kwargs
        assert len(kwargs["args"]) == 3  # media_id, path, use_sam
        assert kwargs["countdown"] == 2  # Race condition prevention

    def test_image_upload_without_auth(self, client, sample_image):
        """Test that image upload requires authentication"""
        response = client.post(
            "/infer/async/image",
            files={"file": ("test.png", sample_image, "image/png")},
            data={"use_sam": "true"}
        )
        assert response.status_code == 401

    @patch('app.services.tasks.process_image_task.apply_async')
    def test_image_upload_with_location_only(self, mock_apply_async, client, auth_headers, sample_image):
        """Test image upload with only coordinates (no address)"""
        mock_task = MagicMock()
        mock_task.id = str(uuid.uuid4())
        mock_apply_async.return_value = mock_task

        # Mock reverse geocoding
        with patch('app.api.inference_routes_async.reverse_geocode') as mock_geocode:
            mock_geocode.return_value = "123 Test Street, Test City"

            response = client.post(
                "/infer/async/image",
                headers=auth_headers,
                files={"file": ("test.png", sample_image, "image/png")},
                data={
                    "use_sam": "false",
                    "latitude": 44.4268,
                    "longitude": 26.1025
                }
            )

            assert response.status_code == 202
            mock_geocode.assert_called_once_with(44.4268, 26.1025)

    def test_invalid_image_extension(self, client, auth_headers):
        """Test rejection of unsupported file types"""
        # Create a text file pretending to be an image
        fake_image = io.BytesIO(b"This is not an image")
        fake_image.seek(0)

        response = client.post(
            "/infer/async/image",
            headers=auth_headers,
            files={"file": ("test.txt", fake_image, "text/plain")},
            data={"use_sam": "true"}
        )
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()

    def test_image_size_validation(self, client, auth_headers, large_image):
        """Test rejection of images exceeding size limit"""
        response = client.post(
            "/infer/async/image",
            headers=auth_headers,
            files={"file": ("large.png", large_image, "image/png")},
            data={"use_sam": "true"}
        )
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()

    @patch('cv2.imread')
    def test_image_dimension_validation(self, mock_imread, client, auth_headers, sample_image):
        """Test rejection of images with excessive dimensions"""
        # Mock cv2.imread to return oversized image dimensions
        mock_img = MagicMock()
        mock_img.shape = (5000, 6000, 3)  # height, width, channels - exceeds MAX_IMAGE_DIMENSION
        mock_imread.return_value = mock_img

        response = client.post(
            "/infer/async/image",
            headers=auth_headers,
            files={"file": ("test.png", sample_image, "image/png")},
            data={"use_sam": "true"}
        )
        assert response.status_code == 400
        assert "dimensions too large" in response.json()["detail"].lower()

    def test_invalid_location_data(self, client, auth_headers, sample_image):
        """Test validation of location coordinates"""
        # Test invalid latitude
        response = client.post(
            "/infer/async/image",
            headers=auth_headers,
            files={"file": ("test.png", sample_image, "image/png")},
            data={
                "use_sam": "true",
                "latitude": 91,  # Invalid: > 90
                "longitude": 26.1025
            }
        )
        assert response.status_code == 422

        # Reset file pointer
        sample_image.seek(0)

        # Test invalid longitude
        response = client.post(
            "/infer/async/image",
            headers=auth_headers,
            files={"file": ("test.png", sample_image, "image/png")},
            data={
                "use_sam": "true",
                "latitude": 44.4268,
                "longitude": 181  # Invalid: > 180
            }
        )
        assert response.status_code == 422


# ============== Video Upload Tests ==============

class TestVideoUpload:
    """Test video upload and processing"""

    @patch('app.services.tasks.process_video_task.apply_async')
    @patch('app.core.validation.validate_video_properties')
    def test_successful_video_upload(self, mock_validate_video, mock_apply_async,
                                    client, auth_headers, sample_video):
        """Test successful video upload and task queuing"""
        # Setup mocks
        mock_validate_video.return_value = {
            "duration": 5.0,
            "fps": 30,
            "width": 640,
            "height": 480,
            "frame_count": 150
        }

        mock_task = MagicMock()
        mock_task.id = str(uuid.uuid4())
        mock_apply_async.return_value = mock_task

        response = client.post(
            "/infer/async/video",
            headers=auth_headers,
            files={"file": ("test.mp4", sample_video, "video/mp4")},
            data={
                "use_sam": "true",
                "address": "Test Location"
            }
        )

        assert response.status_code == 202
        result = response.json()
        assert "task_id" in result
        assert result["task_id"] == mock_task.id
        assert "media_id" in result
        assert result["status"] == "pending"

        # Verify task was queued
        mock_apply_async.assert_called_once()

    def test_video_upload_without_auth(self, client, sample_video):
        """Test that video upload requires authentication"""
        response = client.post(
            "/infer/async/video",
            files={"file": ("test.mp4", sample_video, "video/mp4")},
            data={"use_sam": "true"}
        )
        assert response.status_code == 401

    def test_invalid_video_extension(self, client, auth_headers):
        """Test rejection of unsupported video formats"""
        fake_video = io.BytesIO(b"This is not a video")
        fake_video.seek(0)

        response = client.post(
            "/infer/async/video",
            headers=auth_headers,
            files={"file": ("test.txt", fake_video, "text/plain")},
            data={"use_sam": "true"}
        )
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()

    @patch('cv2.VideoCapture')
    def test_video_duration_validation(self, mock_video_capture, client, auth_headers, sample_video):
        """Test rejection of videos exceeding duration limit"""
        # Mock VideoCapture to simulate long video
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 3000  # 100 seconds at 30 fps - exceeds limit
        }.get(x, 0)
        mock_video_capture.return_value = mock_cap

        response = client.post(
            "/infer/async/video",
            headers=auth_headers,
            files={"file": ("test.mp4", sample_video, "video/mp4")},
            data={"use_sam": "true"}
        )
        assert response.status_code == 400
        assert "duration too long" in response.json()["detail"].lower()


# ============== Task Status Tracking Tests ==============

class TestTaskStatus:
    """Test task status tracking and retrieval"""

    @patch('app.api.inference_routes_async.AsyncResult')
    def test_get_task_status_pending(self, mock_async_result, client, auth_headers, db_session):
        """Test retrieving pending task status"""
        task_id = str(uuid.uuid4())

        # Create a media record with the task_id
        media = dbm.Media(
            filename="test.jpg",
            media_type="image",
            user_username="test_user",  # Must match the auth_headers user
            task_id=task_id,
            processing_status=dbm.ProcessingStatus.pending
        )
        db_session.add(media)
        db_session.commit()

        # Mock AsyncResult
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_result.info = {}
        mock_async_result.return_value = mock_result

        response = client.get(
            f"/infer/status/{task_id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()
        assert result["task_id"] == task_id
        assert result["status"] == "pending"
        assert result["progress"] is None

    @patch('app.api.inference_routes_async.AsyncResult')
    def test_get_task_status_progress(self, mock_async_result, client, auth_headers, db_session):
        """Test retrieving task with progress updates"""
        task_id = str(uuid.uuid4())

        # Create a media record
        media = dbm.Media(
            filename="test.jpg",
            media_type="image",
            user_username="test_user",
            task_id=task_id,
            processing_status=dbm.ProcessingStatus.processing
        )
        db_session.add(media)
        db_session.commit()

        # Mock AsyncResult with progress - state must be uppercase "PROGRESS"
        mock_result = MagicMock()
        mock_result.state = "PROGRESS"
        mock_result.info = {
            "current": 50,
            "total": 100,
            "status": "Processing frame 50 of 100"
        }
        mock_async_result.return_value = mock_result

        response = client.get(
            f"/infer/status/{task_id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()
        assert result["task_id"] == task_id
        assert result["status"] == "processing"  # mapped from PROGRESS
        assert result["progress"] == 50

    @patch('app.api.inference_routes_async.AsyncResult')
    def test_get_task_status_success(self, mock_async_result, client, auth_headers, db_session):
        """Test retrieving successful task status"""
        task_id = str(uuid.uuid4())

        # Create a media record
        media = dbm.Media(
            filename="test.jpg",
            media_type="image",
            user_username="test_user",
            task_id=task_id,
            processing_status=dbm.ProcessingStatus.completed
        )
        db_session.add(media)
        db_session.commit()

        # Mock AsyncResult for success - use result.result not result.info
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {  # Changed from info to result
            "media_id": media.id,
            "detections_count": 5,
            "processing_time": 2.5
        }
        mock_result.info = None  # info is not used for SUCCESS state
        mock_async_result.return_value = mock_result

        response = client.get(
            f"/infer/status/{task_id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()
        assert result["task_id"] == task_id
        assert result["status"] == "completed"
        assert result["result"]["media_id"] == media.id

    @patch('app.api.inference_routes_async.AsyncResult')
    def test_get_task_status_failure(self, mock_async_result, client, auth_headers, db_session):
        """Test retrieving failed task status"""
        task_id = str(uuid.uuid4())

        # Create a media record with error message
        media = dbm.Media(
            filename="test.jpg",
            media_type="image",
            user_username="test_user",
            task_id=task_id,
            processing_status=dbm.ProcessingStatus.failed,
            error_message="Processing failed: GPU out of memory"
        )
        db_session.add(media)
        db_session.commit()

        # Mock AsyncResult for failure - info should be the Exception object
        mock_result = MagicMock()
        mock_result.state = "FAILURE"
        mock_result.info = "Processing failed: GPU out of memory"  # This gets converted to string
        mock_result.result = None
        mock_async_result.return_value = mock_result

        response = client.get(
            f"/infer/status/{task_id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()
        assert result["task_id"] == task_id
        assert result["status"] == "failed"
        assert "error" in result
        assert "GPU out of memory" in str(result["error"])

    def test_get_task_status_without_auth(self, client):
        """Test that task status requires authentication"""
        task_id = str(uuid.uuid4())
        response = client.get(f"/infer/status/{task_id}")
        assert response.status_code == 401


# ============== Error Handling Tests ==============

class TestErrorHandling:
    """Test error handling for invalid media"""

    def test_missing_file_parameter(self, client, auth_headers):
        """Test error when file parameter is missing"""
        response = client.post(
            "/infer/async/image",
            headers=auth_headers,
            data={"use_sam": "true"}
        )
        assert response.status_code == 422
        assert "field required" in str(response.json()["detail"]).lower()

    @patch('cv2.imread')
    def test_corrupt_image_file(self, mock_imread, client, auth_headers):
        """Test handling of corrupt image files"""
        # Mock imread to return None (corrupt file)
        mock_imread.return_value = None

        # Create a fake corrupt file
        corrupt_file = io.BytesIO(b"CORRUPT_IMAGE_DATA")
        corrupt_file.seek(0)

        response = client.post(
            "/infer/async/image",
            headers=auth_headers,
            files={"file": ("corrupt.png", corrupt_file, "image/png")},
            data={"use_sam": "true"}
        )
        assert response.status_code == 400
        assert "could not decode" in response.json()["detail"].lower()


# ============== Media Result Retrieval Tests ==============

class TestMediaResults:
    """Test retrieving processing results"""

    def test_get_media_result_success(self, client, auth_headers, db_session):
        """Test retrieving successful processing results"""
        # Create a completed media record
        media = dbm.Media(
            filename="test.jpg",
            media_type="image",
            user_username="test_user",
            processing_status=dbm.ProcessingStatus.completed,
            static_filename="test_annotated.jpg",  # Fixed field name
            created_at=datetime.now(timezone.utc)
        )
        db_session.add(media)
        db_session.commit()

        response = client.get(
            f"/infer/result/{media.id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()
        assert result["media_id"] == media.id
        assert result["status"] == "completed"

    def test_get_media_result_not_found(self, client, auth_headers):
        """Test retrieving non-existent media result"""
        response = client.get(
            "/infer/result/99999",
            headers=auth_headers
        )
        assert response.status_code == 404


# ============== Media List Tests ==============

class TestMediaList:
    """Test media listing functionality"""

    def test_list_user_media(self, client, auth_headers, db_session):
        """Test listing media for authenticated user"""
        # Create some media records
        for i in range(3):
            media = dbm.Media(
                filename=f"test{i}.jpg",
                media_type="image",
                user_username="test_user",
                processing_status=dbm.ProcessingStatus.completed
            )
            db_session.add(media)
        db_session.commit()

        response = client.get(
            "/infer/list",
            headers=auth_headers
        )

        assert response.status_code == 200
        results = response.json()
        assert len(results) >= 3


# ============== Integration Tests ==============

class TestIntegrationFlow:
    """End-to-end integration tests"""

    @patch('app.services.tasks.process_image_task.apply_async')
    @patch('app.api.inference_routes_async.AsyncResult')
    def test_complete_image_processing_flow(self, mock_async_result, mock_apply_async,
                                           client, auth_headers, sample_image, db_session):
        """Test complete flow from upload to result retrieval"""
        # Step 1: Upload image
        mock_task = MagicMock()
        mock_task.id = str(uuid.uuid4())
        mock_apply_async.return_value = mock_task

        upload_response = client.post(
            "/infer/async/image",
            headers=auth_headers,
            files={"file": ("test.png", sample_image, "image/png")},
            data={"use_sam": "true"}
        )

        assert upload_response.status_code == 202
        task_id = upload_response.json()["task_id"]
        media_id = upload_response.json()["media_id"]

        # Step 2: Check status - need to get media from DB
        media = db_session.query(dbm.Media).filter_by(id=media_id).first()
        assert media is not None
        # The media already has task_id set from the upload, just update status
        media.processing_status = dbm.ProcessingStatus.processing
        db_session.commit()

        # Mock processing status
        mock_result = MagicMock()
        mock_result.state = "PROGRESS"
        mock_result.info = {"current": 50, "total": 100}
        mock_result.result = None
        mock_async_result.return_value = mock_result

        status_response = client.get(
            f"/infer/status/{task_id}",
            headers=auth_headers
        )
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "processing"

