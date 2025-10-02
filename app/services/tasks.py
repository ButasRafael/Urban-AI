import cv2
import numpy as np
import logging
import traceback
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from celery import Task
from celery.signals import worker_process_init, task_postrun
from sqlalchemy.orm import Session
from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.models import media as dbm
from app.models.user import User  # Import User model to register table
from app.services import inference as svc
from app.services.embedding_worker import process_media_embeddings
from app.core.redis_pool import (
    send_websocket_update_safe,
    progress_throttler
)
import uuid
import tempfile
import json
import time
import torch
import gc
import os
from contextlib import contextmanager, nullcontext
from typing import Optional
from prometheus_client import Counter, Histogram
from sqlalchemy import select, delete

logger = logging.getLogger(__name__)

# Prometheus metrics for Celery tasks
CELERY_INFERENCE_IMAGE_COUNT = Counter(
    'celery_inference_image_requests_total',
    'Total number of image inference tasks processed'
)
CELERY_INFERENCE_IMAGE_LATENCY = Histogram(
    'celery_inference_image_duration_seconds',
    'Time spent processing image inference tasks'
)
CELERY_INFERENCE_VIDEO_COUNT = Counter(
    'celery_inference_video_requests_total',
    'Total number of video inference tasks processed'
)
CELERY_INFERENCE_VIDEO_LATENCY = Histogram(
    'celery_inference_video_duration_seconds',
    'Time spent processing video inference tasks'
)

models_loaded = False
yolo_model = None
sam_predictor = None
mask_gen = None
grounder = None


@worker_process_init.connect
def init_worker_process(sender=None, **kwargs):
    """
    Initialize models when worker process starts.
    This keeps models in GPU memory across tasks.
    """
    global models_loaded, yolo_model, sam_predictor, mask_gen, grounder

    # Only load models in worker processes, not in web container
    if os.getenv("ROLE") != "worker":
        logger.info("Skipping model loading in non-worker process")
        return

    # Only load models in GPU workers
    if os.getenv("WORKER_KIND") != "gpu":
        logger.info(f"Skipping model loading in {os.getenv('WORKER_KIND')} worker")
        return

    if not models_loaded:
        logger.info("Loading models into GPU worker process...")
        try:
            # Set optimal CUDA settings
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                os.environ.setdefault("OMP_NUM_THREADS", "1")
                os.environ.setdefault("MKL_NUM_THREADS", "1")

            yolo_model, sam_predictor, mask_gen = svc._load_models()
            grounder = svc._load_grounder()
            models_loaded = True
            logger.info("Models loaded successfully in GPU worker process")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise


@task_postrun.connect
def cleanup_gpu_memory(sender=None, **kwargs):
    """Release GPU memory after each task"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


@contextmanager
def db_transaction():
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def _delete_media_detections(db: Session, media_id: int) -> None:
    """Delete Detection rows for a media_id"""
    frames_subq = select(dbm.Frame.id).where(dbm.Frame.media_id == media_id)
    db.execute(
        delete(dbm.Detection).where(dbm.Detection.frame_id.in_(frames_subq))
    )


class InferenceTask(Task):
    """Base task with custom error handling"""
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3, 'countdown': 60}
    retry_backoff = True
    retry_backoff_max = 700
    retry_jitter = True


def _clamp_pct(x: int) -> int:
    try:
        return max(0, min(100, int(x)))
    except Exception:
        return 0


def update_media_status(media_id: int, status: dbm.ProcessingStatus, error_message: str = None, progress: int | None = None):
    """Update media processing status in database"""
    with db_transaction() as db:
        media = db.query(dbm.Media).filter(dbm.Media.id == media_id).first()
        if media:
            media.processing_status = status
            task_id = media.task_id
            if error_message:
                media.error_message = error_message
            if status == dbm.ProcessingStatus.processing:
                media.started_at = datetime.now(timezone.utc)
            elif status in [dbm.ProcessingStatus.completed, dbm.ProcessingStatus.failed]:
                media.completed_at = datetime.now(timezone.utc)

            # Send WebSocket notification if we have a task_id
            if task_id:
                send_websocket_update(task_id, status.value, error_message, progress)


def send_websocket_update(task_id: str, status: str, error_message: str = None, progress: int | None = None):
    """Send WebSocket update via Redis pub/sub with throttling"""
    # Use throttling for progress updates
    if progress is not None and status == "processing":
        if not progress_throttler.should_send(task_id, progress):
            return

    send_websocket_update_safe(task_id, status, error_message, progress)


@celery_app.task(bind=True, base=InferenceTask, name="tasks.process_image")
def process_image_task(self, media_id: int, file_path: str, use_sam: bool = True):
    """
    Process image inference in background task.
    """
    logger.info(f"Starting image processing task for media_id={media_id}")
    CELERY_INFERENCE_IMAGE_COUNT.inc()
    start_time = time.time()

    # Update status to processing early for better user feedback
    update_media_status(media_id, dbm.ProcessingStatus.processing, progress=0)

    # local helper to report progress
    def _report(pct: int):
        pct = _clamp_pct(pct)
        try:
            self.update_state(state="PROGRESS", meta={"current": pct})
        except Exception:
            pass
        try:
            # publish to WS subscribers
            send_websocket_update(self.request.id, "processing", None, pct)
        except Exception:
            pass

    try:
        # Load and process image first (before DB operations)
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not decode image from {file_path}")

        # Resize if needed
        img = _resize_if_needed(img)

        # Run async inference with CUDA OOM protection
        try:
            with torch.inference_mode() if torch.cuda.is_available() else nullcontext():
                annotated, dets = asyncio.run(
                    svc.process_image_combined(img, use_sam, str(media_id))
                )
                _report(70)
                summary = asyncio.run(
                    svc.generate_comprehensive_summary(dets)
                )
                _report(80)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM for media_id={media_id}, attempting cleanup and retry")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            update_media_status(media_id, dbm.ProcessingStatus.failed, "GPU out of memory, retrying")
            raise self.retry(exc=e, countdown=120, max_retries=2)
        except Exception as e:
            logger.error(f"Inference failed for media_id={media_id}: {e}")
            raise

        # Save annotated image
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)
        file_uuid = str(uuid.uuid4())
        out_name = f"{file_uuid}.jpg"
        out_path = static_dir / out_name
        cv2.imwrite(str(out_path), annotated)

        # Now use a single transaction for all database operations
        with db_transaction() as db:
            # Update status to processing
            media = db.query(dbm.Media).filter(dbm.Media.id == media_id).first()
            if not media:
                raise ValueError(f"Media record {media_id} not found")

            media.processing_status = dbm.ProcessingStatus.processing
            media.started_at = datetime.now(timezone.utc)
            _report(5)

            # Idempotency guard: delete any existing rows for this media_id (in case of retry)
            _delete_media_detections(db, media_id)
            db.query(dbm.Frame).filter(dbm.Frame.media_id == media_id).delete(synchronize_session=False)

            # Update media with inference results
            media.width = annotated.shape[1]
            media.height = annotated.shape[0]
            media.summary_description = summary.get("description")
            media.summary_solution = summary.get("solution")
            _report(85)

            # Create frame and detection records
            fr = dbm.Frame(media_id=media_id, frame_index=0, timestamp=0.0)
            db.add(fr)
            db.flush()  # Get the ID without committing

            # Batch insert detections using bulk_insert_mappings
            if dets:
                detection_data = []
                for d in dets:
                    mask = d.get("mask", {}) or {}
                    detection_data.append({
                        "frame_id": fr.id,
                        "track_id": d.get("track_id"),
                        "class_id": d.get("class_id", -1),
                        "class_name": d.get("class_name", ""),
                        "confidence": d.get("confidence"),
                        "x1": d["bbox"][0],
                        "y1": d["bbox"][1],
                        "x2": d["bbox"][2],
                        "y2": d["bbox"][3],
                        "mask_rle": mask.get("rle", {}),
                        "mask_polygon": mask.get("polygon", []),
                        "description": d.get("description"),
                        "solution": d.get("solution"),
                        "source": _infer_source(d),
                        "severity": _to_severity_enum(d.get("severity")),
                    })
                db.bulk_insert_mappings(dbm.Detection, detection_data)
            _report(90)

            # Update media with output filename
            media.static_filename = out_name
            media.processing_status = dbm.ProcessingStatus.completed
            media.completed_at = datetime.now(timezone.utc)
            # Transaction commits automatically when exiting the context
            _report(95)

        # Queue embedding processing as separate task
        process_embeddings_task.delay(media_id)

        # Send completion notification via WebSocket
        update_media_status(media_id, dbm.ProcessingStatus.completed)
        # Push 100% explicitly:
        try:
            send_websocket_update(self.request.id, "completed", None, 100)
        except Exception:
            pass

        # Clean up temporary file
        try:
            Path(file_path).unlink(missing_ok=True)
            logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {file_path}: {e}")

        logger.info(f"Image processing completed for media_id={media_id}")
        CELERY_INFERENCE_IMAGE_LATENCY.observe(time.time() - start_time)
        return {
            "media_id": media_id,
            "status": "completed",
            "annotated_image_url": f"/static/{out_name}",
            "detections": len(dets)
        }

    except Exception as e:
        logger.error(f"Image processing failed for media_id={media_id}: {e}")
        logger.error(traceback.format_exc())
        CELERY_INFERENCE_IMAGE_LATENCY.observe(time.time() - start_time)
        update_media_status(media_id, dbm.ProcessingStatus.failed, str(e))
        raise
    finally:
        # Clean up throttler
        progress_throttler.cleanup(self.request.id)
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

@celery_app.task(bind=True, base=InferenceTask, name="tasks.process_video")
def process_video_task(self, media_id: int, file_path: str, use_sam: bool = True):
    """
    Process video inference in background task.
    """
    logger.info(f"Starting video processing task for media_id={media_id}")
    CELERY_INFERENCE_VIDEO_COUNT.inc()
    start_time = time.time()

    # Update status to processing early for better user feedback
    update_media_status(media_id, dbm.ProcessingStatus.processing, progress=0)

    def _report(pct: int):
        pct = _clamp_pct(pct)
        try:
            self.update_state(state="PROGRESS", meta={"current": pct})
        except Exception:
            pass
        try:
            send_websocket_update(self.request.id, "processing", None, pct)
        except Exception:
            pass

    try:
        # Update status to processing
        update_media_status(media_id, dbm.ProcessingStatus.processing)
        _report(5)

        # ==== Run inference (no DB I/O here) ====
        try:
            with torch.inference_mode() if torch.cuda.is_available() else nullcontext():
                annotated_tmp, frames_meta, track_thumbnails = svc.process_video(
                    Path(file_path), use_sam
                )
                _report(70)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM for media_id={media_id}, attempting cleanup and retry")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            update_media_status(media_id, dbm.ProcessingStatus.failed, "GPU out of memory, retrying")
            raise self.retry(exc=e, countdown=120, max_retries=2)
        except Exception as e:
            logger.error(f"Video inference failed for media_id={media_id}: {e}")
            raise

        # ==== Build frames, aggregates, and summary in ONE transaction ====
        track_aggregates = {}

        with db_transaction() as db:
            # Media row
            media = db.query(dbm.Media).filter(dbm.Media.id == media_id).first()
            if not media:
                raise ValueError(f"Media record {media_id} not found")

            media.num_frames = len(frames_meta)

            # Idempotency guard: delete any existing rows for this media_id (in case of retry)
            _delete_media_detections(db, media_id)
            db.query(dbm.Frame).filter(dbm.Frame.media_id == media_id).delete(synchronize_session=False)

            # 1) Create Frame rows and keep a mapping frame_index -> frame_id
            frame_id_by_index = {}
            frame_rows = []
            for fr in frames_meta:
                fr_row = dbm.Frame(
                    media_id=media_id,
                    frame_index=fr["frame_index"],
                    timestamp=fr["timestamp_ms"],
                )
                db.add(fr_row)
                frame_rows.append((fr["frame_index"], fr_row))

            # Single flush to get all frame IDs at once (much faster)
            db.flush()

            # Now build the mapping
            for frame_index, fr_row in frame_rows:
                frame_id_by_index[frame_index] = fr_row.id

            # 2) Build per-track/class aggregates with the correct first_frame_id
            for fr in frames_meta:
                fr_id = frame_id_by_index[fr["frame_index"]]
                for d in fr["objects"]:
                    class_name = d.get("class_name", "") or ""
                    if not class_name:
                        continue

                    track_id = d.get("track_id")
                    aggregate_key = (
                        f"track_{track_id}" if track_id is not None else f"class_{class_name}"
                    )

                    if aggregate_key not in track_aggregates:
                        mask = d.get("mask", {}) or {}
                        track_aggregates[aggregate_key] = {
                            "first_frame_id": fr_id,
                            "track_id": track_id,
                            "class_id": d.get("class_id", -1),
                            "class_name": class_name,
                            "max_confidence": float(d.get("confidence", 0.0)),
                            "bbox": d["bbox"],
                            "mask_rle": mask.get("rle", {}),
                            "mask_polygon": mask.get("polygon", []),
                            "description": d.get("description"),
                            "solution": d.get("solution"),
                            "source": _infer_source(d),
                            "severity": _to_severity_enum(d.get("severity")),
                            "total_frames": 1,
                        }
                    else:
                        agg = track_aggregates[aggregate_key]
                        current_conf = float(d.get("confidence", 0.0))
                        if current_conf > agg["max_confidence"]:
                            mask = d.get("mask", {}) or {}
                            agg["max_confidence"] = current_conf
                            agg["bbox"] = d["bbox"]
                            agg["mask_rle"] = mask.get("rle", {})
                            agg["mask_polygon"] = mask.get("polygon", [])
                            if d.get("description"):
                                agg["description"] = d["description"]
                            if d.get("solution"):
                                agg["solution"] = d["solution"]
                            if d.get("severity"):
                                agg["severity"] = _to_severity_enum(d.get("severity"))
                        agg["total_frames"] += 1

            # 3) Generate summary (no extra commit here)
            summary_input = [{
                "class_name": (agg.get("class_name") or "unknown"),
                "description": (agg.get("description") or "").strip(),
                "solution": (agg.get("solution") or "").strip(),
                "severity": (
                    agg["severity"].value if hasattr(agg.get("severity"), "value")
                    else str(agg.get("severity") or "medium").lower()
                ),
            } for agg in track_aggregates.values()]

            if summary_input:
                try:
                    summary = asyncio.run(
                        svc.generate_comprehensive_summary(summary_input)
                    )
                    media.summary_description = summary.get("description")
                    media.summary_solution = summary.get("solution")
                    _report(80)
                except Exception as e:
                    logger.error(f"Summary generation failed for media_id={media_id}: {e}")
                    # continue without summary

            # 4) Batch insert aggregated Detection rows using bulk_insert_mappings
            if track_aggregates:
                detection_data = []
                for agg in track_aggregates.values():
                    detection_data.append({
                        "frame_id": agg["first_frame_id"],
                        "track_id": agg.get("track_id"),
                        "class_id": agg["class_id"],
                        "class_name": agg["class_name"],
                        "confidence": agg["max_confidence"],
                        "x1": agg["bbox"][0],
                        "y1": agg["bbox"][1],
                        "x2": agg["bbox"][2],
                        "y2": agg["bbox"][3],
                        "mask_rle": agg["mask_rle"],
                        "mask_polygon": agg["mask_polygon"],
                        "description": agg.get("description"),
                        "solution": agg.get("solution"),
                        "source": agg["source"],
                        "severity": agg["severity"],
                        "frames_detected": agg["total_frames"],
                    })
                db.bulk_insert_mappings(dbm.Detection, detection_data)
            # (commit happens on context exit)

        _report(85)

        # ==== Save files (no DB locks held) ====
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)

        file_uuid = str(uuid.uuid4())
        out_name = f"{file_uuid}.mp4"
        out_path = static_dir / out_name

        thumbnail_uuid = str(uuid.uuid4())
        thumbnail_name = f"{thumbnail_uuid}.jpg"
        thumbnail_path = static_dir / thumbnail_name

        _generate_video_thumbnail(str(annotated_tmp), str(thumbnail_path))
        try:
            _to_h264(str(annotated_tmp), str(out_path))
        except Exception as e:
            logger.warning(f"H.264 transcoding failed: {e}, using original")
            import shutil
            shutil.move(str(annotated_tmp), str(out_path))

        # Track thumbnails: update Detection rows in a SHORT transaction
        if track_thumbnails:
            from PIL import Image
            video_uuid = Path(out_name).stem
            tracks_folder_name = f"{video_uuid}_tracks"
            tracks_folder_path = static_dir / tracks_folder_name
            tracks_folder_path.mkdir(exist_ok=True)

            for track_id, thumbnail_img in track_thumbnails.items():
                thumb_filename = f"track_{track_id}.jpg"
                thumb_path = tracks_folder_path / thumb_filename
                thumbnail_rgb = cv2.cvtColor(thumbnail_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(thumbnail_rgb)
                pil_image.save(str(thumb_path))

                with db_transaction() as db:
                    detections = db.query(dbm.Detection).join(dbm.Frame).filter(
                        dbm.Frame.media_id == media_id,
                        dbm.Detection.track_id == track_id
                    ).all()
                    for detection in detections:
                        detection.track_thumbnail_url = f"/static/{tracks_folder_name}/{thumb_filename}"

        # Final media fields in a SHORT transaction
        with db_transaction() as db:
            media = db.query(dbm.Media).filter(dbm.Media.id == media_id).first()
            if not media:
                raise ValueError(f"Media record {media_id} not found")
            media.static_filename = out_name
            media.thumbnail_filename = thumbnail_name
            media.processing_status = dbm.ProcessingStatus.completed

        _report(95)

        # Kick off embeddings separate from inference
        process_embeddings_task.delay(media_id)

        # Notify completion
        update_media_status(media_id, dbm.ProcessingStatus.completed)
        try:
            send_websocket_update(self.request.id, "completed", None, 100)
        except Exception:
            pass

        # Clean up temp files
        try:
            Path(file_path).unlink(missing_ok=True)
            if annotated_tmp != out_path:
                Path(annotated_tmp).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

        logger.info(f"Video processing completed for media_id={media_id}")
        CELERY_INFERENCE_VIDEO_LATENCY.observe(time.time() - start_time)
        return {
            "media_id": media_id,
            "status": "completed",
            "annotated_video_url": f"/static/{out_name}",
            "thumbnail_url": f"/static/{thumbnail_name}",
            "frames": len(frames_meta)
        }

    except Exception as e:
        logger.error(f"Video processing failed for media_id={media_id}: {e}")
        logger.error(traceback.format_exc())
        CELERY_INFERENCE_VIDEO_LATENCY.observe(time.time() - start_time)
        update_media_status(media_id, dbm.ProcessingStatus.failed, str(e))
        raise
    finally:
        progress_throttler.cleanup(self.request.id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

def _resize_if_needed(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Resize image if it exceeds maximum dimension"""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _infer_source(d: dict) -> dbm.DetectionSource:
    """Infer detection source from detection dict"""
    s = d.get("source")
    if s:
        try:
            return dbm.DetectionSource(s)
        except Exception:
            pass
    name = (d.get("class_name") or "").lower()
    if name.endswith("-gpt+dino"):
        return dbm.DetectionSource.gpt_dino
    if name == "clean":
        return dbm.DetectionSource.sam_fallback
    return dbm.DetectionSource.yolo


def _to_severity_enum(val: str | None) -> dbm.Severity:
    """Convert string to Severity enum"""
    try:
        return dbm.Severity((val or "medium").lower())
    except Exception:
        return dbm.Severity.medium


def _generate_video_thumbnail(video_path: str, thumbnail_path: str):
    """Generate thumbnail from first frame of video"""
    import os
    logger.info(f"Generating video thumbnail: {video_path} -> {thumbnail_path}")

    if not os.path.exists(video_path):
        raise RuntimeError(f"Video file does not exist: {video_path}")

    thumbnail_dir = Path(thumbnail_path).parent
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    try:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read first frame from video: {video_path}")

        success = cv2.imwrite(thumbnail_path, frame)
        if not success:
            raise RuntimeError(f"Failed to save thumbnail: {thumbnail_path}")

        logger.info(f"Successfully generated thumbnail: {thumbnail_path}")
    finally:
        cap.release()


@celery_app.task(name="tasks.process_embeddings", queue="cpu")
def process_embeddings_task(media_id: int):
    """
    Process RAG embeddings as a separate task.
    This allows inference to complete quickly while embeddings process in background.
    """
    try:
        logger.info(f"Starting embedding processing for media_id={media_id}")
        process_media_embeddings(media_id)
        logger.info(f"Completed embedding processing for media_id={media_id}")
        return {"media_id": media_id, "status": "embeddings_completed"}
    except Exception as e:
        logger.error(f"Failed to process embeddings for media_id={media_id}: {e}")
        # Don't fail the main task if embeddings fail
        return {"media_id": media_id, "status": "embeddings_failed", "error": str(e)}


def _to_h264(src: str, dst: str):
    """Transcode video to H.264 format"""
    import subprocess
    import os
    logger.info(f"Transcoding {src} to {dst}")

    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)

    if not os.path.exists(src_abs):
        raise RuntimeError(f"Source file does not exist: {src_abs}")

    dst_path = Path(dst_abs)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Find ffmpeg
    import shutil
    ffmpeg_exec = shutil.which("ffmpeg")
    if not ffmpeg_exec:
        raise RuntimeError("ffmpeg not found")

    cmd = [
        ffmpeg_exec, "-y", "-i", src_abs,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", "1",  # Limit CPU usage in GPU containers
        "-an",
        dst_abs,
    ]

    logger.info(f"Running ffmpeg command: {' '.join(cmd)}")

    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            logger.error(f"ffmpeg failed: {p.stderr}")
            raise RuntimeError(f"ffmpeg failed: {p.stderr[-400:]}")

        if not os.path.exists(dst_abs):
            raise RuntimeError(f"ffmpeg completed but output file not created: {dst_abs}")

        logger.info(f"Successfully transcoded to {dst_abs}")
    except subprocess.SubprocessError as e:
        logger.error(f"Subprocess error during ffmpeg: {e}")
        raise RuntimeError(f"ffmpeg subprocess error: {e}")


@celery_app.task(name="tasks.cleanup_temp_files", queue="cpu")
def cleanup_temp_files():
    """
    Periodic task to clean up old temporary files.
    Should be scheduled with Celery Beat.
    """
    from datetime import timedelta

    logger.info("Starting temporary file cleanup")
    temp_dir = Path("static/uploads")
    cutoff = datetime.now() - timedelta(days=30)

    cleaned = 0
    errors = 0

    if temp_dir.exists():
        for file in temp_dir.glob("*"):
            if file.is_file():
                stat = file.stat()
                if datetime.fromtimestamp(stat.st_mtime) < cutoff:
                    try:
                        file.unlink()
                        cleaned += 1
                        logger.debug(f"Deleted old temp file: {file}")
                    except Exception as e:
                        errors += 1
                        logger.error(f"Failed to delete {file}: {e}")

    logger.info(f"Cleanup complete: {cleaned} files deleted, {errors} errors")
    return {"cleaned_files": cleaned, "errors": errors}