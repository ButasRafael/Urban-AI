from fastapi import (
    APIRouter, UploadFile, File, HTTPException, Depends, Query, Form, BackgroundTasks, Request, Response
)
from pathlib import Path
import shutil, uuid, logging, time
from app.models.schemas_inference import MediaListItem
from app.core.database import get_db
from app.models import media as dbm
from app.core.security import require_roles, get_current_user
from sqlalchemy.orm import Session, selectinload
from typing import List
import httpx
from app.services.tasks import process_image_task, process_video_task
from celery.result import AsyncResult
from app.core.celery_app import celery_app
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2
from starlette.concurrency import run_in_threadpool
from app.core.validation import (
    LocationData,
    validate_file_size,
    validate_image_dimensions,
    validate_video_properties,
    validate_file_extension,
    MAX_IMAGE_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    IMAGE_EXTS,
    VIDEO_EXTS
)
from app.core.rate_limiter import limiter, user_limiter, INFERENCE_RATE_LIMITS

try:
    import geohash2 as _geohash_mod
except Exception:
    _geohash_mod = None


def _geohash6(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None:
        return None
    if _geohash_mod:
        try:
            return _geohash_mod.encode(lat, lon, precision=6)
        except Exception:
            pass
    return None


# File extensions are now imported from validation module

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/infer", tags=["Async Inference"])

STATIC_DIR = Path("static")
UPLOADS_DIR = STATIC_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MAX_DIM = 1024


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= MAX_DIM:
        return img
    scale = MAX_DIM / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _save_temp(f: UploadFile) -> Path:
    dst = UPLOADS_DIR / f"{uuid.uuid4()}_{Path(f.filename).name}"
    with dst.open("wb") as w:
        shutil.copyfileobj(f.file, w)
    return dst


async def reverse_geocode(lat, lon):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"format": "jsonv2", "lat": lat, "lon": lon},
                timeout=5.0
            )
            r.raise_for_status()
            return r.json().get("display_name", "")
    except:
        return ""


class InferenceSubmitResponse(BaseModel):
    task_id: str
    media_id: int
    status: str = "pending"
    message: str = "Processing started"


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[int] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class MediaResultResponse(BaseModel):
    media_id: int
    status: dbm.ProcessingStatus
    media_type: Optional[str] = None
    created_at: Optional[str] = None
    annotated_image_url: Optional[str] = None
    annotated_video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    frames: Optional[List[dict]] = None
    detections: Optional[List[dict]] = None
    summary_description: Optional[str] = None
    summary_solution: Optional[str] = None
    error_message: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@router.post(
    "/async/image",
    response_model=InferenceSubmitResponse,
    dependencies=[require_roles("user", "admin")],
    status_code=202,
)
@limiter.limit(INFERENCE_RATE_LIMITS["image"])
@user_limiter.limit("20 per hour")  # Per-user limit for images
async def detect_image_async(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    use_sam: bool = Query(True, description="Use SAM for segmentation masks"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    latitude: float | None = Form(None, ge=-90, le=90),
    longitude: float | None = Form(None, ge=-180, le=180),
    address: str | None = Form(None, max_length=500),
):
    """Submit image for asynchronous processing"""
    logger.info(f"Received async image inference request: {file.filename}")

    # Validate file extension
    ext = validate_file_extension(file.filename, IMAGE_EXTS)

    # Validate file size
    await validate_file_size(file, MAX_IMAGE_SIZE_BYTES, "Image")

    # Validate location data
    location = LocationData(latitude=latitude, longitude=longitude, address=address)

    # Save the uploaded file
    path = _save_temp(file)

    # Validate image dimensions
    try:
        width, height = await run_in_threadpool(validate_image_dimensions, path)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise

    # Geocode if needed
    if address:
        final_address = address
    elif location.latitude is not None and location.longitude is not None:
        final_address = await reverse_geocode(location.latitude, location.longitude)
    else:
        final_address = ""

    # Create Media record with pending status
    media = dbm.Media(
        filename=file.filename,
        media_type="image",
        user_username=current_user.username,
        address=final_address,
        latitude=location.latitude,
        longitude=location.longitude,
        geohash6=_geohash6(location.latitude, location.longitude),
        processing_status=dbm.ProcessingStatus.pending,
    )
    db.add(media)
    db.flush()  # Get media.id without committing

    # Queue the task with countdown to ensure DB commit happens first
    task = process_image_task.apply_async(
        args=(media.id, str(path), use_sam),
        countdown=2  # 2-second delay to prevent race condition
    )
    media.task_id = task.id

    # Single commit with both media and task_id
    db.commit()
    db.refresh(media)

    logger.info(f"Queued image task {task.id} for media_id={media.id}")

    return InferenceSubmitResponse(
        task_id=task.id,
        media_id=media.id,
        status="pending",
        message="Image processing has been queued"
    )


@router.post(
    "/async/video",
    response_model=InferenceSubmitResponse,
    dependencies=[require_roles("user", "admin")],
    status_code=202,
)
@limiter.limit(INFERENCE_RATE_LIMITS["video"])
@user_limiter.limit("5 per hour")  # Per-user limit for videos
async def detect_video_async(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    use_sam: bool = Query(True, description="Use SAM for segmentation masks"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    latitude: float | None = Form(None, ge=-90, le=90),
    longitude: float | None = Form(None, ge=-180, le=180),
    address: str | None = Form(None, max_length=500),
):
    """Submit video for asynchronous processing"""
    logger.info(f"Received async video inference request: {file.filename}")

    # Validate file extension
    ext = validate_file_extension(file.filename, VIDEO_EXTS)

    # Validate file size
    await validate_file_size(file, MAX_VIDEO_SIZE_BYTES, "Video")

    # Validate location data
    location = LocationData(latitude=latitude, longitude=longitude, address=address)

    # Save the uploaded file
    tmp_path = await run_in_threadpool(_save_temp, file)

    # Validate video properties (duration, resolution, etc.)
    try:
        video_info = await run_in_threadpool(validate_video_properties, tmp_path)
        logger.info(f"Video properties: {video_info}")
    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise

    # Geocode if needed
    if address:
        final_address = address
    elif location.latitude is not None and location.longitude is not None:
        final_address = await reverse_geocode(location.latitude, location.longitude)
    else:
        final_address = ""

    # Create Media record with pending status
    media = dbm.Media(
        filename=file.filename,
        media_type="video",
        user_username=current_user.username,
        address=final_address,
        latitude=location.latitude,
        longitude=location.longitude,
        geohash6=_geohash6(location.latitude, location.longitude),
        processing_status=dbm.ProcessingStatus.pending,
    )
    db.add(media)
    db.flush()  # Get media.id without committing

    # Queue the task with countdown to ensure DB commit happens first
    task = process_video_task.apply_async(
        args=(media.id, str(tmp_path), use_sam),
        countdown=2  # 2-second delay to prevent race condition
    )
    media.task_id = task.id

    # Single commit with both media and task_id
    db.commit()
    db.refresh(media)

    logger.info(f"Queued video task {task.id} for media_id={media.id}")

    return InferenceSubmitResponse(
        task_id=task.id,
        media_id=media.id,
        status="pending",
        message="Video processing has been queued"
    )


@router.get(
    "/status/{task_id}",
    response_model=TaskStatusResponse,
    dependencies=[require_roles("user", "admin")],
)
@limiter.limit(INFERENCE_RATE_LIMITS["status"])
def get_task_status(
    request: Request,
    response: Response,
    task_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get the status of a processing task"""
    # Verify user owns the media
    media = db.query(dbm.Media).filter(dbm.Media.task_id == task_id).first()
    if not media:
        raise HTTPException(404, "Task not found")
    if media.user_username != current_user.username and current_user.role != "admin":
        raise HTTPException(403, "Access denied")

    # Get task result
    result = AsyncResult(task_id, app=celery_app)
    celery_state = (result.state or "PENDING").upper()

    status_map = {
        "PENDING": "pending",
        "STARTED": "processing",
        "RETRY": "processing",
        "PROGRESS": "processing",
        "SUCCESS": "completed",
        "FAILURE": "failed",
    }
    status = status_map.get(celery_state, celery_state.lower())

    progress: int | None = None
    error: str | None = None
    payload: dict | None = None

    if celery_state == "PROGRESS":
        info = result.info or {}
        if isinstance(info, dict) and "current" in info:
            try:
                progress = int(info["current"])
            except Exception:
                progress = None
    elif celery_state == "SUCCESS":
        payload = result.result
    elif celery_state == "FAILURE":
        error = str(result.info)

    # ---- DB fallback when Celery looks "lost" (broker restart / result expired) ----
    if celery_state in ("PENDING", "REVOKED", None):
        db_status = media.processing_status
        if db_status == dbm.ProcessingStatus.completed:
            status = "completed"
            progress = 100
            # (payload stays None; client should call /infer/result/{media_id})
        elif db_status == dbm.ProcessingStatus.failed:
            status = "failed"
            error = media.error_message or error
        elif db_status == dbm.ProcessingStatus.processing:
            status = "processing"   # better than showing "pending"
        # if db says pending, leave as-is

    return TaskStatusResponse(
        task_id=task_id,
        status=status,
        progress=progress,
        result=payload if celery_state == "SUCCESS" else None,
        error=error,
    )


@router.get(
    "/result/{media_id}",
    response_model=MediaResultResponse,
    dependencies=[require_roles("user", "admin")],
)
@limiter.limit(INFERENCE_RATE_LIMITS["status"])
def get_media_result(
    request: Request,
    response: Response,
    media_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get the processing result for a media item"""
    # Get media record
    media = db.query(dbm.Media).filter(dbm.Media.id == media_id).first()
    if not media:
        raise HTTPException(404, "Media not found")
    if media.user_username != current_user.username and current_user.role != "admin":
        raise HTTPException(403, "Access denied")

    response = MediaResultResponse(
        media_id=media.id,
        status=media.processing_status,
        media_type=media.media_type,
        created_at=media.created_at.isoformat() if media.created_at else None,
        address=media.address,
        latitude=media.latitude,
        longitude=media.longitude,
        summary_description=media.summary_description,
        summary_solution=media.summary_solution,
        error_message=media.error_message,
    )

    if media.processing_status == dbm.ProcessingStatus.completed:
        if media.media_type == "image":
            response.annotated_image_url = f"/static/{media.static_filename}" if media.static_filename else None

            # Get detections
            frame = db.query(dbm.Frame).filter(dbm.Frame.media_id == media.id).first()
            if frame:
                detections = db.query(dbm.Detection).filter(dbm.Detection.frame_id == frame.id).all()
                response.detections = [
                    {
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox": [d.x1, d.y1, d.x2, d.y2],
                        "description": d.description,
                        "solution": d.solution,
                        "severity": d.severity.value if d.severity else None,
                    }
                    for d in detections
                ]

        elif media.media_type == "video":
            response.annotated_video_url = f"/static/{media.static_filename}" if media.static_filename else None
            response.thumbnail_url = f"/static/{media.thumbnail_filename}" if media.thumbnail_filename else None

            # Get frames summary
            frames = db.query(dbm.Frame).filter(dbm.Frame.media_id == media.id).all()
            response.frames = [
                {
                    "frame_index": f.frame_index,
                    "timestamp": f.timestamp,
                    "num_detections": db.query(dbm.Detection).filter(
                        dbm.Detection.frame_id == f.id
                    ).count()
                }
                for f in frames
            ]

    return response


@router.get(
    "/list",
    response_model=List[MediaListItem],
    dependencies=[require_roles("user", "admin")],
)
@limiter.limit(INFERENCE_RATE_LIMITS["media_list"])
def list_my_uploads(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    include_pending: bool = Query(False, description="Include pending/processing items"),
):
    """List user's uploaded media with processing status"""
    # Build query with eager loading to avoid N+1 problem
    query = (
        db.query(dbm.Media)
        .options(
            selectinload(dbm.Media.frames).selectinload(dbm.Frame.detects)
        )
        .filter(dbm.Media.user_username == current_user.username)
    )

    if not include_pending:
        query = query.filter(dbm.Media.processing_status == dbm.ProcessingStatus.completed)

    rows = query.order_by(dbm.Media.created_at.desc()).all()

    out: List[MediaListItem] = []
    for m in rows:
        # Only provide URLs for completed items
        if m.processing_status == dbm.ProcessingStatus.completed and m.static_filename:
            if m.media_type == "image":
                img_url = f"/static/{m.static_filename}"
                video_url = None
            else:
                video_url = f"/static/{m.static_filename}"
                img_url = f"/static/{m.thumbnail_filename}" if m.thumbnail_filename else None
        else:
            img_url = None
            video_url = None

        # Get detection info only if processing is complete
        classes = []
        descriptions = []
        solutions = []
        if m.processing_status == dbm.ProcessingStatus.completed:
            # Access already-loaded frames instead of querying again
            if m.frames:
                # Get first frame (already loaded via eager loading)
                first_frame = sorted(m.frames, key=lambda f: f.frame_index)[0] if m.frames else None
                if first_frame:
                    # Use the already-loaded detections
                    classes = list(set(d.class_name for d in first_frame.detects if d.class_name))
                    descriptions = [d.description for d in first_frame.detects if d.description]
                    solutions = [d.solution for d in first_frame.detects if d.solution]

        out.append(MediaListItem(
            media_id=m.id,
            media_type=m.media_type,
            annotated_image_url=img_url,
            annotated_video_url=video_url,
            created_at=m.created_at,
            address=m.address,
            latitude=m.latitude,
            longitude=m.longitude,
            predicted_classes=classes,
            descriptions=descriptions,
            solutions=solutions,
            summary_description=m.summary_description,
            summary_solution=m.summary_solution,
            processing_status=m.processing_status.value,
            task_id=m.task_id,
        ))
    return out