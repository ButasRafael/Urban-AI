"""
Input validation utilities for file uploads and API parameters
"""
import cv2
from pathlib import Path
from typing import Optional, Tuple
from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
import numpy as np
import logging

logger = logging.getLogger(__name__)

# File size limits
MAX_IMAGE_SIZE_MB = 10
MAX_VIDEO_SIZE_MB = 50
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024

# Resolution limits
MAX_IMAGE_DIMENSION = 4096  # 4K max width or height
MAX_VIDEO_DIMENSION = 1920  # 1080p max width or height
MAX_VIDEO_DURATION_SECONDS = 60  # Max 1 minute videos
MAX_VIDEO_FRAMES = 1800  # Max frames (60 fps * 30 seconds)

# Supported file extensions
IMAGE_EXTS = {
    ".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff",
    ".webp", ".pfm", ".heic",
}
VIDEO_EXTS = {
    ".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg",
    ".mpg", ".ts", ".wmv", ".webm",
}


class LocationData(BaseModel):
    """Validated location data with proper ranges"""
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude (-90 to 90)")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude (-180 to 180)")
    address: Optional[str] = Field(None, max_length=500, description="Address string")

    @field_validator('latitude', 'longitude')
    @classmethod
    def validate_coordinates(cls, v, info):
        if v is not None:
            # Additional precision check
            field_name = info.field_name
            if field_name == 'latitude' and not -90 <= v <= 90:
                raise ValueError(f"Latitude must be between -90 and 90, got {v}")
            if field_name == 'longitude' and not -180 <= v <= 180:
                raise ValueError(f"Longitude must be between -180 and 180, got {v}")
        return v


async def validate_file_size(file: UploadFile, max_size_bytes: int, file_type: str) -> None:
    """Validate file size by reading content"""
    # Read file to check size (FastAPI doesn't give us size directly)
    content = await file.read()
    file_size = len(content)

    # Reset file pointer for later use
    await file.seek(0)

    if file_size > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"{file_type} file too large: {file_size / (1024 * 1024):.1f}MB. Maximum allowed: {max_mb}MB"
        )

    logger.info(f"File size validated: {file_size / (1024 * 1024):.2f}MB")


def validate_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Validate image dimensions and return (width, height)"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise HTTPException(400, "Could not decode image file")

    height, width = img.shape[:2]

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            400,
            f"Image dimensions too large: {width}x{height}. Maximum dimension: {MAX_IMAGE_DIMENSION}px"
        )

    logger.info(f"Image dimensions validated: {width}x{height}")
    return width, height


def validate_video_properties(video_path: Path) -> dict:
    """Validate video properties and return metadata"""
    cap = cv2.VideoCapture(str(video_path))

    try:
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video file")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Validate FPS
        if fps <= 0:
            fps = 25  # Default fallback

        # Calculate duration
        duration_seconds = frame_count / fps if fps > 0 else 0

        # Validate duration
        if duration_seconds > MAX_VIDEO_DURATION_SECONDS:
            raise HTTPException(
                400,
                f"Video duration too long: {duration_seconds:.1f}s. Maximum allowed: {MAX_VIDEO_DURATION_SECONDS}s"
            )

        # Validate frame count
        if frame_count > MAX_VIDEO_FRAMES:
            raise HTTPException(
                400,
                f"Video has too many frames: {frame_count}. Maximum allowed: {MAX_VIDEO_FRAMES}"
            )

        # Validate dimensions
        if width > MAX_VIDEO_DIMENSION or height > MAX_VIDEO_DIMENSION:
            raise HTTPException(
                400,
                f"Video dimensions too large: {width}x{height}. Maximum dimension: {MAX_VIDEO_DIMENSION}px"
            )

        # Read first frame to ensure video is readable
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(400, "Could not read video frames")

        logger.info(
            f"Video validated: {width}x{height}, {fps:.1f}fps, "
            f"{frame_count} frames, {duration_seconds:.1f}s"
        )

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration_seconds
        }

    finally:
        cap.release()


def validate_file_extension(filename: str, allowed_extensions: set) -> str:
    """Validate file extension and return lowercase extension"""
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        allowed_str = ", ".join(sorted(allowed_extensions))
        raise HTTPException(
            400,
            f"Unsupported file format: {ext}. Allowed formats: {allowed_str}"
        )
    return ext