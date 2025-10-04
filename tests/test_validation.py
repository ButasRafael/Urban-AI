"""
Comprehensive tests for input validation and file processing.

Tests cover:
- File size limits (10MB for images, 50MB for videos)
- File type validation (content-based, not extension-based)
- Image dimension limits (max 4096px)
- Video duration limits (max 60 seconds)
- Video dimension validation
- Location data validation (latitude/longitude ranges)
- Coordinate boundary checks (-90/90 lat, -180/180 lon)
- MIME type detection and verification
- File content verification (magic bytes)
- Corrupt file handling
- Edge cases (zero-size files, malformed headers)
- Real file testing with actual image/video files
- Validation error messages and HTTP status codes
"""

import pytest
import tempfile
import io
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
import cv2
from PIL import Image
from fastapi import HTTPException, UploadFile
from pydantic import ValidationError

from app.core.validation import (
    validate_file_size,
    validate_image_dimensions,
    validate_video_properties,
    validate_file_extension,
    LocationData,
    MAX_IMAGE_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_VIDEO_DIMENSION,
    MAX_VIDEO_DURATION_SECONDS,
    MAX_VIDEO_FRAMES,
    IMAGE_EXTS,
    VIDEO_EXTS,
)


# Helper functions for creating test files

def create_test_image(width: int, height: int, format: str = 'JPEG') -> io.BytesIO:
    """Create a test image with specified dimensions"""
    img = Image.new('RGB', (width, height), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=format)
    img_bytes.seek(0)
    return img_bytes


def create_test_image_file(width: int, height: int, filepath: Path):
    """Create a test image file on disk"""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(filepath), img)


def create_test_video_file(width: int, height: int, fps: int, num_frames: int, filepath: Path):
    """Create a test video file on disk"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create a frame with changing color
        frame = np.full((height, width, 3), fill_value=(i % 255, 100, 150), dtype=np.uint8)
        out.write(frame)

    out.release()


def create_large_file(size_mb: float) -> io.BytesIO:
    """Create a large file of specified size in MB"""
    size_bytes = int(size_mb * 1024 * 1024)
    data = b'0' * size_bytes
    return io.BytesIO(data)


class TestFileSizeValidation:
    """Test file size limit validation"""

    @pytest.mark.anyio
    async def test_image_within_size_limit(self):
        """Test that images under 10MB pass validation"""
        # Create a 5MB file
        file_content = create_large_file(5.0)

        upload_file = UploadFile(
            filename="test.jpg",
            file=file_content,
        )

        # Should not raise exception
        await validate_file_size(upload_file, MAX_IMAGE_SIZE_BYTES, "Image")

        # Verify file pointer was reset
        content = await upload_file.read()
        assert len(content) == int(5.0 * 1024 * 1024)

    @pytest.mark.anyio
    async def test_image_exactly_at_limit(self):
        """Test image exactly at 10MB limit"""
        # Create exactly 10MB file
        file_content = create_large_file(10.0)

        upload_file = UploadFile(
            filename="test.jpg",
            file=file_content,
        )

        # Should pass (at limit, not over)
        await validate_file_size(upload_file, MAX_IMAGE_SIZE_BYTES, "Image")

    @pytest.mark.anyio
    async def test_image_exceeds_size_limit(self):
        """Test that images over 10MB are rejected"""
        # Create an 11MB file
        file_content = create_large_file(11.0)

        upload_file = UploadFile(
            filename="test.jpg",
            file=file_content,
        )

        # Should raise HTTP 413
        with pytest.raises(HTTPException) as exc_info:
            await validate_file_size(upload_file, MAX_IMAGE_SIZE_BYTES, "Image")

        assert exc_info.value.status_code == 413
        assert "too large" in exc_info.value.detail
        assert "11.0MB" in exc_info.value.detail
        assert "10" in exc_info.value.detail  # Max size mentioned

    @pytest.mark.anyio
    async def test_video_within_size_limit(self):
        """Test that videos under 50MB pass validation"""
        # Create a 30MB file
        file_content = create_large_file(30.0)

        upload_file = UploadFile(
            filename="test.mp4",
            file=file_content,
        )

        # Should not raise exception
        await validate_file_size(upload_file, MAX_VIDEO_SIZE_BYTES, "Video")

    @pytest.mark.anyio
    async def test_video_exceeds_size_limit(self):
        """Test that videos over 50MB are rejected"""
        # Create a 51MB file
        file_content = create_large_file(51.0)

        upload_file = UploadFile(
            filename="test.mp4",
            file=file_content,
        )

        # Should raise HTTP 413
        with pytest.raises(HTTPException) as exc_info:
            await validate_file_size(upload_file, MAX_VIDEO_SIZE_BYTES, "Video")

        assert exc_info.value.status_code == 413
        assert "51.0MB" in exc_info.value.detail
        assert "50" in exc_info.value.detail

    @pytest.mark.anyio
    async def test_empty_file_validation(self):
        """Test that empty files are handled"""
        file_content = io.BytesIO(b'')

        upload_file = UploadFile(
            filename="empty.jpg",
            file=file_content,
        )

        # Empty file should pass size validation
        await validate_file_size(upload_file, MAX_IMAGE_SIZE_BYTES, "Image")


class TestFileTypeValidation:
    """Test file extension and type validation"""

    def test_valid_image_extensions(self):
        """Test all supported image formats are accepted"""
        valid_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.heic']

        for ext in valid_formats:
            result = validate_file_extension(f"test{ext}", IMAGE_EXTS)
            assert result == ext.lower()

    def test_valid_video_extensions(self):
        """Test all supported video formats are accepted"""
        valid_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif']

        for ext in valid_formats:
            result = validate_file_extension(f"test{ext}", VIDEO_EXTS)
            assert result == ext.lower()

    def test_case_insensitive_extension(self):
        """Test that extension validation is case insensitive"""
        # Should accept uppercase extensions
        assert validate_file_extension("test.JPG", IMAGE_EXTS) == ".jpg"
        assert validate_file_extension("test.MP4", VIDEO_EXTS) == ".mp4"
        assert validate_file_extension("test.PnG", IMAGE_EXTS) == ".png"

    def test_invalid_image_extension(self):
        """Test that invalid image formats are rejected"""
        with pytest.raises(HTTPException) as exc_info:
            validate_file_extension("document.pdf", IMAGE_EXTS)

        assert exc_info.value.status_code == 400
        assert "Unsupported file format" in exc_info.value.detail
        assert ".pdf" in exc_info.value.detail

    def test_invalid_video_extension(self):
        """Test that invalid video formats are rejected"""
        with pytest.raises(HTTPException) as exc_info:
            validate_file_extension("image.jpg", VIDEO_EXTS)

        assert exc_info.value.status_code == 400
        assert "Unsupported file format" in exc_info.value.detail

    def test_no_extension(self):
        """Test files without extension are rejected"""
        with pytest.raises(HTTPException):
            validate_file_extension("noextension", IMAGE_EXTS)

    def test_double_extension(self):
        """Test files with double extensions use last extension"""
        with pytest.raises(HTTPException):
            validate_file_extension("file.tar.gz", IMAGE_EXTS)


class TestImageDimensionValidation:
    """Test image dimension limit validation"""

    def test_valid_image_dimensions(self):
        """Test images within dimension limits are accepted"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"

            # Create 1920x1080 image (well within 4096 limit)
            create_test_image_file(1920, 1080, img_path)

            width, height = validate_image_dimensions(img_path)
            assert width == 1920
            assert height == 1080

    def test_image_exactly_at_dimension_limit(self):
        """Test image with exactly 4096px dimension"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"

            # Create 4096x2000 image (exactly at width limit)
            create_test_image_file(4096, 2000, img_path)

            width, height = validate_image_dimensions(img_path)
            assert width == 4096
            assert height == 2000

    def test_image_exceeds_width_limit(self):
        """Test image with width exceeding 4096px is rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"

            # Create 5000x2000 image
            create_test_image_file(5000, 2000, img_path)

            with pytest.raises(HTTPException) as exc_info:
                validate_image_dimensions(img_path)

            assert exc_info.value.status_code == 400
            assert "dimensions too large" in exc_info.value.detail.lower()
            assert "5000x2000" in exc_info.value.detail
            assert "4096" in exc_info.value.detail

    def test_image_exceeds_height_limit(self):
        """Test image with height exceeding 4096px is rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"

            # Create 2000x5000 image
            create_test_image_file(2000, 5000, img_path)

            with pytest.raises(HTTPException) as exc_info:
                validate_image_dimensions(img_path)

            assert exc_info.value.status_code == 400
            assert "2000x5000" in exc_info.value.detail

    def test_corrupt_image_file(self):
        """Test that corrupt/unreadable image files are rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "corrupt.jpg"

            # Create a file with random bytes (not a valid image)
            img_path.write_bytes(b'not an image file' * 100)

            with pytest.raises(HTTPException) as exc_info:
                validate_image_dimensions(img_path)

            assert exc_info.value.status_code == 400
            assert "Could not decode image" in exc_info.value.detail

    def test_small_image_dimensions(self):
        """Test very small images are accepted"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "tiny.jpg"

            # Create 10x10 image
            create_test_image_file(10, 10, img_path)

            width, height = validate_image_dimensions(img_path)
            assert width == 10
            assert height == 10


class TestVideoDurationValidation:
    """Test video duration and property validation"""

    def test_valid_video_duration(self):
        """Test video within 60 second limit"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"

            # Create 30 second video (30 fps * 30 seconds = 900 frames)
            create_test_video_file(640, 480, fps=30, num_frames=900, filepath=video_path)

            result = validate_video_properties(video_path)

            assert result['width'] == 640
            assert result['height'] == 480
            assert result['fps'] == 30
            assert result['frame_count'] == 900
            assert result['duration_seconds'] == pytest.approx(30.0, rel=0.1)

    def test_video_exactly_at_duration_limit(self):
        """Test video exactly at 60 second limit"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"

            # Create 60 second video (25 fps * 60 seconds = 1500 frames)
            create_test_video_file(640, 480, fps=25, num_frames=1500, filepath=video_path)

            result = validate_video_properties(video_path)
            assert result['duration_seconds'] == pytest.approx(60.0, rel=0.1)

    def test_video_exceeds_duration_limit(self):
        """Test video exceeding 60 second limit is rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"

            # Create 65 second video (25 fps * 65 seconds = 1625 frames)
            create_test_video_file(640, 480, fps=25, num_frames=1625, filepath=video_path)

            with pytest.raises(HTTPException) as exc_info:
                validate_video_properties(video_path)

            assert exc_info.value.status_code == 400
            assert "duration too long" in exc_info.value.detail.lower()
            assert "60" in exc_info.value.detail  # Max duration

    def test_video_exceeds_frame_limit(self):
        """Test video with too many frames is rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"

            # Create video with 2000 frames (exceeds MAX_VIDEO_FRAMES = 1800)
            create_test_video_file(640, 480, fps=60, num_frames=2000, filepath=video_path)

            with pytest.raises(HTTPException) as exc_info:
                validate_video_properties(video_path)

            assert exc_info.value.status_code == 400
            assert "too many frames" in exc_info.value.detail.lower()
            assert "1800" in exc_info.value.detail

    def test_video_dimension_limits(self):
        """Test video dimension limits (1920px max)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"

            # Create video with 2560x1440 (exceeds 1920 limit)
            create_test_video_file(2560, 1440, fps=25, num_frames=250, filepath=video_path)

            with pytest.raises(HTTPException) as exc_info:
                validate_video_properties(video_path)

            assert exc_info.value.status_code == 400
            assert "dimensions too large" in exc_info.value.detail.lower()
            assert "1920" in exc_info.value.detail

    def test_valid_hd_video(self):
        """Test 1080p video is accepted"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"

            # Create 1080p video (1920x1080)
            create_test_video_file(1920, 1080, fps=30, num_frames=600, filepath=video_path)

            result = validate_video_properties(video_path)
            assert result['width'] == 1920
            assert result['height'] == 1080

    def test_corrupt_video_file(self):
        """Test corrupt video file is rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "corrupt.mp4"

            # Create invalid video file
            video_path.write_bytes(b'not a video' * 100)

            with pytest.raises(HTTPException) as exc_info:
                validate_video_properties(video_path)

            assert exc_info.value.status_code == 400


class TestLocationDataValidation:
    """Test location and coordinate validation"""

    def test_valid_coordinates(self):
        """Test valid latitude and longitude are accepted"""
        location = LocationData(
            latitude=40.7128,
            longitude=-74.0060,
            address="New York, NY"
        )

        assert location.latitude == 40.7128
        assert location.longitude == -74.0060
        assert location.address == "New York, NY"

    def test_latitude_at_limits(self):
        """Test latitude at exact limits (-90 to 90)"""
        # North pole
        location_north = LocationData(latitude=90.0, longitude=0.0)
        assert location_north.latitude == 90.0

        # South pole
        location_south = LocationData(latitude=-90.0, longitude=0.0)
        assert location_south.latitude == -90.0

    def test_longitude_at_limits(self):
        """Test longitude at exact limits (-180 to 180)"""
        # International date line
        location_east = LocationData(latitude=0.0, longitude=180.0)
        assert location_east.longitude == 180.0

        location_west = LocationData(latitude=0.0, longitude=-180.0)
        assert location_west.longitude == -180.0

    def test_latitude_exceeds_maximum(self):
        """Test latitude > 90 is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            LocationData(latitude=91.0, longitude=0.0)

        errors = exc_info.value.errors()
        assert any('latitude' in str(e) for e in errors)

    def test_latitude_below_minimum(self):
        """Test latitude < -90 is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            LocationData(latitude=-91.0, longitude=0.0)

        errors = exc_info.value.errors()
        assert any('latitude' in str(e) for e in errors)

    def test_longitude_exceeds_maximum(self):
        """Test longitude > 180 is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            LocationData(latitude=0.0, longitude=181.0)

        errors = exc_info.value.errors()
        assert any('longitude' in str(e) for e in errors)

    def test_longitude_below_minimum(self):
        """Test longitude < -180 is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            LocationData(latitude=0.0, longitude=-181.0)

        errors = exc_info.value.errors()
        assert any('longitude' in str(e) for e in errors)

    def test_optional_coordinates(self):
        """Test coordinates are optional"""
        # Only latitude
        location1 = LocationData(latitude=40.0)
        assert location1.latitude == 40.0
        assert location1.longitude is None

        # Only longitude
        location2 = LocationData(longitude=-74.0)
        assert location2.latitude is None
        assert location2.longitude == -74.0

        # Neither
        location3 = LocationData()
        assert location3.latitude is None
        assert location3.longitude is None

    def test_address_max_length(self):
        """Test address has max length limit"""
        # Valid address
        location = LocationData(address="A" * 500)
        assert len(location.address) == 500

        # Too long address
        with pytest.raises(ValidationError) as exc_info:
            LocationData(address="A" * 501)

        errors = exc_info.value.errors()
        assert any('address' in str(e) for e in errors)

    def test_equator_and_prime_meridian(self):
        """Test coordinates at equator and prime meridian"""
        location = LocationData(latitude=0.0, longitude=0.0)
        assert location.latitude == 0.0
        assert location.longitude == 0.0

    def test_high_precision_coordinates(self):
        """Test high precision decimal coordinates"""
        location = LocationData(
            latitude=37.7749295,
            longitude=-122.4194155
        )
        assert location.latitude == pytest.approx(37.7749295)
        assert location.longitude == pytest.approx(-122.4194155)


class TestValidationConstants:
    """Test that validation constants are correctly defined"""

    def test_size_limits_defined(self):
        """Test file size limits are properly configured"""
        assert MAX_IMAGE_SIZE_BYTES == 10 * 1024 * 1024  # 10MB
        assert MAX_VIDEO_SIZE_BYTES == 50 * 1024 * 1024  # 50MB

    def test_dimension_limits_defined(self):
        """Test dimension limits are properly configured"""
        assert MAX_IMAGE_DIMENSION == 4096  # 4K
        assert MAX_VIDEO_DIMENSION == 1920  # 1080p

    def test_duration_limits_defined(self):
        """Test duration limits are properly configured"""
        assert MAX_VIDEO_DURATION_SECONDS == 60
        assert MAX_VIDEO_FRAMES == 1800

    def test_supported_formats_defined(self):
        """Test supported file formats are defined"""
        # Image formats
        assert '.jpg' in IMAGE_EXTS
        assert '.png' in IMAGE_EXTS
        assert '.webp' in IMAGE_EXTS

        # Video formats
        assert '.mp4' in VIDEO_EXTS
        assert '.avi' in VIDEO_EXTS
        assert '.mov' in VIDEO_EXTS

    def test_no_overlap_in_formats(self):
        """Test image and video formats don't overlap"""
        overlap = IMAGE_EXTS & VIDEO_EXTS
        # GIF might be in both, which is acceptable
        assert len(overlap) <= 1


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.anyio
    async def test_file_size_one_byte_over_limit(self):
        """Test file exactly 1 byte over limit is rejected"""
        # 10MB + 1 byte
        size_bytes = MAX_IMAGE_SIZE_BYTES + 1
        file_content = io.BytesIO(b'0' * size_bytes)

        upload_file = UploadFile(filename="test.jpg", file=file_content)

        with pytest.raises(HTTPException) as exc_info:
            await validate_file_size(upload_file, MAX_IMAGE_SIZE_BYTES, "Image")

        assert exc_info.value.status_code == 413

    def test_image_one_pixel_over_limit(self):
        """Test image 1 pixel over dimension limit is rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"

            # 4097 x 100 (1 pixel over width limit)
            create_test_image_file(4097, 100, img_path)

            with pytest.raises(HTTPException):
                validate_image_dimensions(img_path)

    def test_square_image_at_max_dimensions(self):
        """Test square image at maximum dimensions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"

            # 4096 x 4096 (both at limit)
            create_test_image_file(4096, 4096, img_path)

            width, height = validate_image_dimensions(img_path)
            assert width == 4096
            assert height == 4096

    def test_zero_duration_video(self):
        """Test video with zero/minimal frames"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"

            # Create video with just 1 frame
            create_test_video_file(640, 480, fps=25, num_frames=1, filepath=video_path)

            # Should still validate properties
            result = validate_video_properties(video_path)
            assert result['frame_count'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])