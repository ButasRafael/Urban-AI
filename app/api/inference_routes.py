from fastapi import (
    APIRouter, UploadFile, File, HTTPException, Depends, Query, Form,
    BackgroundTasks
)
from pathlib import Path
import shutil, uuid, logging, time, subprocess
from app.models.schemas_inference import (
    ImageResponse, VideoResponse, FrameOut, MediaListItem
)
from app.services import inference as svc
from app.core.database import get_db
from app.models import media as dbm
from starlette.concurrency import run_in_threadpool
import cv2
from prometheus_client import Counter, Histogram
import sentry_sdk
from app.core.security import require_roles, get_current_user
from sqlalchemy.orm import Session
from typing import List
import httpx
from openai import InternalServerError
import numpy as np
from app.services.embedding_worker import enqueue_embeddings

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


_FFMPEG_PATH = None


def _find_ffmpeg():
    global _FFMPEG_PATH
    if _FFMPEG_PATH is not None:
        return _FFMPEG_PATH

    import shutil

    try:
        which_result = shutil.which("ffmpeg")
        if which_result:
            result = subprocess.run([which_result, "-version"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    timeout=5)
            if result.returncode == 0:
                _FFMPEG_PATH = which_result
                logger.info(f"Found ffmpeg at: {_FFMPEG_PATH}")
                return _FFMPEG_PATH
    except Exception as e:
        logger.debug(f"'which' command failed: {e}")

    fallback_paths = [
        "ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        r"C:\ffmpeg\ffmpeg\ffmpeg\bin\ffmpeg.exe",
        "ffmpeg.exe",
    ]

    for path in fallback_paths:
        try:
            result = subprocess.run([path, "-version"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    timeout=5)
            if result.returncode == 0:
                _FFMPEG_PATH = path
                logger.info(f"Found ffmpeg at: {_FFMPEG_PATH}")
                return _FFMPEG_PATH
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue

    raise RuntimeError(
        "ffmpeg not found. Please ensure ffmpeg is installed in your container. "
        "Add 'ffmpeg' to your Dockerfile's apt-get install command."
    )


def _generate_video_thumbnail(video_path: str, thumbnail_path: str):
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


def _to_h264(src: str, dst: str):
    import os
    logger.info(f"Transcoding {src} to {dst}")

    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)

    logger.info(f"Absolute paths: {src_abs} -> {dst_abs}")

    if not os.path.exists(src_abs):
        raise RuntimeError(f"Source file does not exist: {src_abs}")

    dst_path = Path(dst_abs)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_exec = _find_ffmpeg()

    cmd = [
        ffmpeg_exec, "-y", "-i", src_abs,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        dst_abs,
    ]

    logger.info(f"Running ffmpeg command: {' '.join(cmd)}")

    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info(f"ffmpeg return code: {p.returncode}")

        if p.returncode != 0:
            logger.error(f"ffmpeg failed with return code {p.returncode}")
            logger.error(f"ffmpeg stderr: {p.stderr}")
            logger.error(f"ffmpeg stdout: {p.stdout}")
            raise RuntimeError(f"ffmpeg failed: {p.stderr[-400:]}")

        if not os.path.exists(dst_abs):
            raise RuntimeError(f"ffmpeg completed but output file not created: {dst_abs}")

        logger.info(f"Successfully transcoded to {dst_abs}")

    except subprocess.SubprocessError as e:
        logger.error(f"Subprocess error during ffmpeg: {e}")
        raise RuntimeError(f"ffmpeg subprocess error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during ffmpeg: {e}")
        raise


IMAGE_EXTS = {
    ".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff",
    ".webp", ".pfm", ".heic",
}
VIDEO_EXTS = {
    ".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg",
    ".mpg", ".ts", ".wmv", ".webm",
}

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/infer", tags=["Inference"])

INFERENCE_IMAGE_COUNT = Counter(
    "inference_image_requests_total",
    "Total number of image inference requests"
)
INFERENCE_IMAGE_LATENCY = Histogram(
    "inference_image_latency_seconds",
    "Time spent doing image inference"
)
INFERENCE_VIDEO_COUNT = Counter(
    "inference_video_requests_total",
    "Total number of video inference requests"
)
INFERENCE_VIDEO_LATENCY = Histogram(
    "inference_video_latency_seconds",
    "Time spent doing video inference"
)

STATIC_DIR = Path("static")
MAX_DIM = 1024


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= MAX_DIM:
        return img
    scale = MAX_DIM / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _save_temp(f: UploadFile) -> Path:
    dst = Path("/tmp") / f"{uuid.uuid4()}_{f.filename}"
    with dst.open("wb") as w:
        shutil.copyfileobj(f.file, w)
    return dst


def reverse_geocode(lat, lon):
    try:
        r = httpx.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "jsonv2", "lat": lat, "lon": lon},
            timeout=5.0
        )
        r.raise_for_status()
        return r.json().get("display_name", "")
    except:
        return ""


# --- map detection dict -> DetectionSource enum (analytics) ---
def _infer_source(d: dict) -> dbm.DetectionSource:
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
    try:
        return dbm.Severity((val or "medium").lower())
    except Exception:
        return dbm.Severity.medium


@router.post(
    "/image",
    response_model=ImageResponse,
    dependencies=[require_roles("user", "admin")],
)
async def detect_image(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        use_sam: bool = Query(
            True,
            description="Set to False to draw only YOLO boxes; True to draw YOLO+SAM masks",
        ),
        db: Session = Depends(get_db),
        current_user=Depends(get_current_user),
        latitude: float | None = Form(None),
        longitude: float | None = Form(None),
        address: str | None = Form(None),
):
    logger.info("Received image inference request", extra={
        "upload_filename": file.filename, "use_sam": use_sam
    })
    INFERENCE_IMAGE_COUNT.inc()

    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, f"Unsupported image format: {ext}")

    path = _save_temp(file)
    img = await run_in_threadpool(cv2.imread, str(path))
    if img is None:
        raise HTTPException(400, "Could not decode image")
    img = _resize_if_needed(img)

    if address:
        final_address = address
    elif latitude is not None and longitude is not None:
        final_address = await run_in_threadpool(reverse_geocode, latitude, longitude)
    else:
        final_address = ""

    # 1) create Media row so we have media.id (+ geohash6)
    media = dbm.Media(
        filename=file.filename,
        media_type="image",
        user_username=current_user.username,
        address=final_address,
        latitude=latitude,
        longitude=longitude,
        geohash6=_geohash6(latitude, longitude),
    )
    db.add(media);
    db.commit();
    db.refresh(media)
    logger.debug("Inserted media row", extra={"media_id": media.id})

    # 2) run inference with real latency measurement
    t0 = time.perf_counter()
    try:
        annotated, dets = await svc.process_image_combined(img, use_sam, str(media.id))
        # Generate comprehensive summary
        summary = await svc.generate_comprehensive_summary(dets)
    except InternalServerError:
        raise HTTPException(502, "Upstream LLM error; please retry shortly.")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.exception("❌ Inference or DB write failed")
        db.delete(media);
        db.commit()
        raise HTTPException(500, f"Inference failed: {e!r}")
    dt_s = time.perf_counter() - t0
    INFERENCE_IMAGE_LATENCY.observe(dt_s)

    # 3) update media metadata (dims + total ms + summary)
    media.width = annotated.shape[1]
    media.height = annotated.shape[0]
    media.process_ms_total = int(dt_s * 1000)
    media.summary_description = summary.get("description")
    media.summary_solution = summary.get("solution")
    db.add(media);
    db.commit()

    # 4) persist Frame + Detection rows
    fr = dbm.Frame(media_id=media.id, frame_index=0, timestamp=0.0)
    db.add(fr);
    db.commit();
    db.refresh(fr)

    for d in dets:
        mask = d.get("mask", {}) or {}
        db.add(dbm.Detection(
            frame_id=fr.id,
            track_id=d.get("track_id"),
            class_id=d.get("class_id", -1),
            class_name=d.get("class_name", ""),
            confidence=d.get("confidence"),
            x1=d["bbox"][0], y1=d["bbox"][1],
            x2=d["bbox"][2], y2=d["bbox"][3],
            mask_rle=mask.get("rle", {}),
            mask_polygon=mask.get("polygon", []),
            description=d.get("description"),
            solution=d.get("solution"),
            source=_infer_source(d),
            severity=_to_severity_enum(d.get("severity")),
        ))
    db.commit()

    # 5) write annotated image file
    STATIC_DIR.mkdir(exist_ok=True)
    # Use UUID for file name to avoid cache issues
    file_uuid = str(uuid.uuid4())
    out_name = f"{file_uuid}.jpg"
    out_path = STATIC_DIR / out_name
    await run_in_threadpool(cv2.imwrite, str(out_path), annotated)

    # Store the UUID filename in the media record
    media.static_filename = out_name
    db.add(media)
    db.commit()

    enqueue_embeddings(background_tasks, media.id)

    return ImageResponse(
        media_id=media.id,
        annotated_image_url=f"/static/{out_name}",
        frames=[FrameOut(frame_index=0, timestamp_ms=0.0, objects=dets)],
        address=media.address,
        latitude=media.latitude,
        longitude=media.longitude,
    )


@router.post(
    "/video",
    response_model=VideoResponse,
    dependencies=[require_roles("user", "admin")],
)
async def detect_video(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        use_sam: bool = Query(
            True,
            description="Set to False to draw only YOLO boxes; True to draw YOLO+SAM masks",
        ),
        db: Session = Depends(get_db),
        current_user=Depends(get_current_user),
        latitude: float | None = Form(None),
        longitude: float | None = Form(None),
        address: str | None = Form(None),
):
    logger.info("Received video inference request", extra={"upload_filename": file.filename})
    INFERENCE_VIDEO_COUNT.inc()

    ext = Path(file.filename).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported video format: {ext}")

    tmp_path = await run_in_threadpool(_save_temp, file)

    if address:
        final_address = address
    elif latitude is not None and longitude is not None:
        final_address = await run_in_threadpool(reverse_geocode, latitude, longitude)
    else:
        final_address = ""

    media = dbm.Media(
        filename=file.filename,
        media_type="video",
        user_username=current_user.username,
        address=final_address,
        latitude=latitude,
        longitude=longitude,
        geohash6=_geohash6(latitude, longitude),
    )
    db.add(media);
    db.commit();
    db.refresh(media)

    t0 = time.perf_counter()
    try:
        annotated_tmp, frames_meta, track_thumbnails = await run_in_threadpool(
            svc.process_video, tmp_path, use_sam
        )
    except Exception as e:
        sentry_sdk.capture_exception(e)
        db.delete(media);
        db.commit()
        raise HTTPException(500, f"Inference failed: {e!r}")
    dt_s = time.perf_counter() - t0
    INFERENCE_VIDEO_LATENCY.observe(dt_s)

    media.num_frames = len(frames_meta)
    media.process_ms_total = int(dt_s * 1000)
    db.add(media);
    db.commit()

    # Track thumbnails will be saved later after we have the video UUID
    track_thumbnail_paths = {}

    # Group detections by track (for LLM-analyzed videos) or class (for legacy videos)
    track_aggregates = {}

    # Process frames and aggregate detections by track_id (preserving individual LLM annotations)
    for fr in frames_meta:
        fr_row = dbm.Frame(
            media_id=media.id,
            frame_index=fr["frame_index"],
            timestamp=fr["timestamp_ms"],
        )
        db.add(fr_row);
        db.commit();
        db.refresh(fr_row)

        for d in fr["objects"]:
            track_id = d.get("track_id")
            class_name = d.get("class_name", "")

            if not class_name:
                continue

            # Use track_id as key if available (for LLM-analyzed videos), otherwise use class_name (legacy)
            aggregate_key = f"track_{track_id}" if track_id is not None else f"class_{class_name}"

            # Initialize or update track/class aggregate
            if aggregate_key not in track_aggregates:
                mask = d.get("mask", {}) or {}
                # Get track thumbnail URL if available, fallback to video thumbnail
                thumbnail_url = track_thumbnail_paths.get(track_id) if track_id is not None else None
                if not thumbnail_url and media.thumbnail_filename:
                    # Fallback to video thumbnail if track thumbnail is missing
                    thumbnail_url = f"/static/{media.thumbnail_filename}"
                track_aggregates[aggregate_key] = {
                    "first_frame_id": fr_row.id,
                    "track_id": track_id,
                    "class_id": d.get("class_id", -1),
                    "class_name": class_name,
                    "max_confidence": d.get("confidence", 0.0),
                    "bbox": d["bbox"],  # bbox from highest confidence detection
                    "mask_rle": mask.get("rle", {}),
                    "mask_polygon": mask.get("polygon", []),
                    "description": d.get("description"),
                    "solution": d.get("solution"),
                    "source": _infer_source(d),
                    "severity": _to_severity_enum(d.get("severity")),
                    "total_frames": 1,
                    "track_ids": set([track_id]) if track_id else set(),
                    "track_thumbnail_url": thumbnail_url,
                }
            else:
                # Update aggregate with higher confidence detection
                current_conf = d.get("confidence", 0.0)
                if current_conf > track_aggregates[aggregate_key]["max_confidence"]:
                    mask = d.get("mask", {}) or {}
                    agg = track_aggregates[aggregate_key]
                    agg["max_confidence"] = current_conf
                    agg["bbox"] = d["bbox"]
                    agg["mask_rle"] = mask.get("rle", {})
                    agg["mask_polygon"] = mask.get("polygon", [])

                    # Only overwrite text fields if new value is non-empty
                    if d.get("description"):
                        agg["description"] = d["description"]
                    if d.get("solution"):
                        agg["solution"] = d["solution"]
                    if d.get("severity"):
                        agg["severity"] = _to_severity_enum(d.get("severity"))

                track_aggregates[aggregate_key]["total_frames"] += 1
                if track_id:
                    track_aggregates[aggregate_key]["track_ids"].add(track_id)

    # Generate comprehensive summary for video
    summary_input = []
    for aggregate_key, agg in track_aggregates.items():
        # Include ALL tracks, not just ones with descriptions
        summary_input.append({
            "class_name": agg.get("class_name") or "unknown",
            "description": (agg.get("description") or "").strip(),
            "solution": (agg.get("solution") or "").strip(),
            "severity": (
                agg["severity"].value if hasattr(agg.get("severity"), "value")
                else (str(agg.get("severity") or "medium")).lower()
            ),
        })

    # Generate summary if we have detections with descriptions
    if summary_input:
        try:
            summary = await svc.generate_comprehensive_summary(summary_input)
            media.summary_description = summary.get("description")
            media.summary_solution = summary.get("solution")
            db.add(media)
            db.commit()
        except Exception as e:
            logger.warning(f"Video summary generation failed: {e}")

    # Create one detection record per track (or class for legacy videos)
    for aggregate_key, agg in track_aggregates.items():
        # Use the track_id if available, otherwise None
        track_id = agg.get("track_id")

        db.add(dbm.Detection(
            frame_id=agg["first_frame_id"],  # Reference first frame where class was detected
            track_id=track_id,
            class_id=agg["class_id"],
            class_name=agg["class_name"],
            confidence=agg["max_confidence"],
            x1=agg["bbox"][0], y1=agg["bbox"][1],
            x2=agg["bbox"][2], y2=agg["bbox"][3],
            mask_rle=agg["mask_rle"],
            mask_polygon=agg["mask_polygon"],
            description=agg["description"],
            solution=agg["solution"],
            source=agg["source"],
            severity=agg["severity"],
            frames_detected=agg["total_frames"],  # Track how many frames contained this class
            track_thumbnail_url=agg.get("track_thumbnail_url"),
        ))

    db.commit()

    STATIC_DIR.mkdir(exist_ok=True)
    # Use UUID for file names to avoid cache issues
    file_uuid = str(uuid.uuid4())
    out_name = f"{file_uuid}.mp4"
    out_path = STATIC_DIR / out_name

    # Generate thumbnail from first frame with detections
    thumbnail_uuid = str(uuid.uuid4())
    thumbnail_name = f"{thumbnail_uuid}.jpg"
    thumbnail_path = STATIC_DIR / thumbnail_name
    await run_in_threadpool(_generate_video_thumbnail, str(annotated_tmp), str(thumbnail_path))

    # Store the UUID filenames in the media record
    media.static_filename = out_name
    media.thumbnail_filename = thumbnail_name

    # Now save track thumbnails using the video UUID for folder naming
    if track_thumbnails:
        from PIL import Image

        video_uuid = Path(out_name).stem
        tracks_folder_name = f"{video_uuid}_tracks"
        tracks_folder_path = STATIC_DIR / tracks_folder_name
        tracks_folder_path.mkdir(exist_ok=True)

        for track_id, thumbnail_img in track_thumbnails.items():
            thumb_filename = f"track_{track_id}.jpg"
            thumb_path = tracks_folder_path / thumb_filename
            # Convert OpenCV image (BGR) to PIL Image (RGB)
            thumbnail_rgb = cv2.cvtColor(thumbnail_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(thumbnail_rgb)
            await run_in_threadpool(pil_image.save, str(thumb_path))
            track_thumbnail_paths[track_id] = f"/static/{tracks_folder_name}/{thumb_filename}"
        
        # Update Detection records with track thumbnail URLs
        for detection in db.query(dbm.Detection).join(dbm.Frame).filter(dbm.Frame.media_id == media.id).all():
            if detection.track_id is not None and detection.track_id in track_thumbnail_paths:
                detection.track_thumbnail_url = track_thumbnail_paths[detection.track_id]
        db.commit()

    try:
        await run_in_threadpool(_to_h264, str(annotated_tmp), str(out_path))
    except Exception as e:
        logger.warning(f"H.264 transcoding failed: {e}, falling back to original video")
        sentry_sdk.capture_exception(e)
        await run_in_threadpool(shutil.move, str(annotated_tmp), str(out_path))
    else:
        try:
            annotated_tmp.unlink(missing_ok=True)
        except Exception:
            pass

    # Save the media with updated filenames
    db.add(media)
    db.commit()

    enqueue_embeddings(background_tasks, media.id)

    return VideoResponse(
        media_id=media.id,
        frames=[FrameOut(**f) for f in frames_meta],
        annotated_video_url=f"/static/{out_name}",
        address=media.address,
        latitude=media.latitude,
        longitude=media.longitude,
    )


@router.get(
    "/list",
    response_model=List[MediaListItem],
    dependencies=[require_roles("user", "admin")],
)
def list_my_uploads(
        db: Session = Depends(get_db),
        current_user=Depends(get_current_user),
):
    rows = (
        db.query(dbm.Media)
        .filter(dbm.Media.user_username == current_user.username)
        .order_by(dbm.Media.created_at.desc())
        .all()
    )

    out: List[MediaListItem] = []
    for m in rows:
        # Use stored UUID filenames if available, fallback to ID-based names for backward compatibility
        if m.static_filename:
            if m.media_type == "image":
                img_url = f"/static/{m.static_filename}"
                video_url = None
            else:
                video_url = f"/static/{m.static_filename}"
                img_url = f"/static/{m.thumbnail_filename}" if m.thumbnail_filename else f"/static/{m.id}.jpg"
        else:
            # Fallback for old records without UUID filenames
            img_url = f"/static/{m.id}.jpg" if m.media_type == "image" else None
            video_url = f"/static/{m.id}.mp4" if m.media_type == "video" else None

        first_frame = (
            db.query(dbm.Frame)
            .filter(dbm.Frame.media_id == m.id)
            .order_by(dbm.Frame.frame_index)
            .first()
        )
        classes: List[str] = []
        if first_frame:
            classes = [
                c[0] for c in
                db.query(dbm.Detection.class_name)
                .filter(dbm.Detection.frame_id == first_frame.id)
                .distinct()
                .all()
            ]
        descriptions, solutions = [], []
        if first_frame:
            descriptions = [
                d[0] for d in
                db.query(dbm.Detection.description)
                .filter(
                    dbm.Detection.frame_id == first_frame.id,
                    dbm.Detection.description.isnot(None),
                )
                .all()
            ]
            solutions = [
                s[0] for s in
                db.query(dbm.Detection.solution)
                .filter(
                    dbm.Detection.frame_id == first_frame.id,
                    dbm.Detection.solution.isnot(None),
                )
                .all()
            ]

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
        ))
    return out


@router.post(
    "/images",
    response_model=List[ImageResponse],
    dependencies=[require_roles("user", "admin")],
)
async def detect_images(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
        use_sam: bool = Query(
            True,
            description="Set to False to draw only YOLO boxes; True to draw YOLO+SAM masks",
        ),
        db: Session = Depends(get_db),
        current_user=Depends(get_current_user),
        latitude: float | None = Form(None),
        longitude: float | None = Form(None),
        address: str | None = Form(None),
):
    responses: List[ImageResponse] = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in IMAGE_EXTS:
            raise HTTPException(400, f"Unsupported image format: {ext}")

        tmp_path = _save_temp(file)
        img = await run_in_threadpool(cv2.imread, str(tmp_path))
        if img is None:
            raise HTTPException(400, "Could not decode image")
        img = _resize_if_needed(img)

        if address:
            final_address = address
        elif latitude is not None and longitude is not None:
            final_address = await run_in_threadpool(reverse_geocode, latitude, longitude)
        else:
            final_address = ""

        media = dbm.Media(
            filename=file.filename,
            media_type="image",
            user_username=current_user.username,
            address=final_address,
            latitude=latitude,
            longitude=longitude,
            geohash6=_geohash6(latitude, longitude),
        )
        db.add(media);
        db.commit();
        db.refresh(media)

        t0 = time.perf_counter()
        try:
            annotated, dets = await svc.process_image_combined(img, use_sam, str(media.id))
        except Exception as e:
            db.delete(media);
            db.commit()
            raise HTTPException(500, f"Inference failed for {file.filename}: {e!r}")
        dt_s = time.perf_counter() - t0

        media.width = annotated.shape[1]
        media.height = annotated.shape[0]
        media.process_ms_total = int(dt_s * 1000)
        db.add(media);
        db.commit()

        fr = dbm.Frame(media_id=media.id, frame_index=0, timestamp=0.0)
        db.add(fr);
        db.commit();
        db.refresh(fr)
        for d in dets:
            mask = d.get("mask", {}) or {}
            db.add(dbm.Detection(
                frame_id=fr.id,
                track_id=d.get("track_id"),
                class_id=d.get("class_id", -1),
                class_name=d.get("class_name", ""),
                confidence=d.get("confidence"),
                x1=d["bbox"][0], y1=d["bbox"][1],
                x2=d["bbox"][2], y2=d["bbox"][3],
                mask_rle=mask.get("rle", {}),
                mask_polygon=mask.get("polygon", []),
                description=d.get("description"),
                solution=d.get("solution"),
                source=_infer_source(d),
                severity=_to_severity_enum(d.get("severity")),
            ))
        db.commit()

        # Use UUID for file name to avoid cache issues
        file_uuid = str(uuid.uuid4())
        out_name = f"{file_uuid}.jpg"
        out_path = STATIC_DIR / out_name
        await run_in_threadpool(cv2.imwrite, str(out_path), annotated)

        # Store the UUID filename in the media record
        media.static_filename = out_name
        db.add(media)
        db.commit()

        enqueue_embeddings(background_tasks, media.id)

        responses.append(
            ImageResponse(
                media_id=media.id,
                annotated_image_url=f"/static/{out_name}",
                frames=[FrameOut(frame_index=0, timestamp_ms=0.0, objects=dets)],
                address=media.address,
                latitude=media.latitude,
                longitude=media.longitude,
                suggestions=[],
            )
        )

    return responses


import zipfile


@router.post(
    "/images_zip",
    response_model=List[ImageResponse],
    dependencies=[require_roles("user", "admin")],
)
async def detect_images_zip(
        background_tasks: BackgroundTasks,
        archive: UploadFile = File(..., media_type="application/zip"),
        use_sam: bool = Query(
            True,
            description="Set to False to draw only YOLO boxes; True to draw YOLO+SAM masks",
        ),
        db: Session = Depends(get_db),
        current_user=Depends(get_current_user),
        latitude: float | None = Form(None),
        longitude: float | None = Form(None),
        address: str | None = Form(None),
):
    tmp_zip = _save_temp(archive)

    images: List[tuple[str, np.ndarray]] = []
    with zipfile.ZipFile(tmp_zip, 'r') as zf:
        for name in zf.namelist():
            ext = Path(name).suffix.lower()
            if ext in IMAGE_EXTS:
                data = zf.read(name)
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    images.append((name, _resize_if_needed(img)))

    if not images:
        raise HTTPException(400, "No supported images found in zip")

    responses: List[ImageResponse] = []
    for filename, img in images:
        if address:
            final_address = address
        elif latitude is not None and longitude is not None:
            final_address = await run_in_threadpool(reverse_geocode, latitude, longitude)
        else:
            final_address = ""

        media = dbm.Media(
            filename=filename,
            media_type="image",
            user_username=current_user.username,
            address=final_address,
            latitude=latitude,
            longitude=longitude,
            geohash6=_geohash6(latitude, longitude),
        )
        db.add(media);
        db.commit();
        db.refresh(media)

        t0 = time.perf_counter()
        try:
            annotated, dets = await svc.process_image_combined(img, use_sam, str(media.id))
        except Exception as e:
            db.delete(media);
            db.commit()
            raise HTTPException(500, f"Inference failed for {filename}: {e!r}")
        dt_s = time.perf_counter() - t0

        media.width, media.height = annotated.shape[1], annotated.shape[0]
        media.process_ms_total = int(dt_s * 1000)
        db.add(media);
        db.commit()

        fr = dbm.Frame(media_id=media.id, frame_index=0, timestamp=0.0)
        db.add(fr);
        db.commit();
        db.refresh(fr)
        for d in dets:
            mask = d.get("mask", {}) or {}
            db.add(dbm.Detection(
                frame_id=fr.id,
                track_id=d.get("track_id"),
                class_id=d.get("class_id", -1),
                class_name=d.get("class_name", ""),
                confidence=d.get("confidence"),
                x1=d["bbox"][0], y1=d["bbox"][1],
                x2=d["bbox"][2], y2=d["bbox"][3],
                mask_rle=mask.get("rle", {}),
                mask_polygon=mask.get("polygon", []),
                description=d.get("description"),
                solution=d.get("solution"),
                source=_infer_source(d),
                severity=_to_severity_enum(d.get("severity")),
            ))
        db.commit()

        # Use UUID for file name to avoid cache issues
        file_uuid = str(uuid.uuid4())
        out_name = f"{file_uuid}.jpg"
        out_path = STATIC_DIR / out_name
        await run_in_threadpool(cv2.imwrite, str(out_path), annotated)

        # Store the UUID filename in the media record
        media.static_filename = out_name
        db.add(media)
        db.commit()

        enqueue_embeddings(background_tasks, media.id)

        responses.append(
            ImageResponse(
                media_id=media.id,
                annotated_image_url=f"/static/{out_name}",
                frames=[FrameOut(frame_index=0, timestamp_ms=0.0, objects=dets)],
                address=media.address,
                latitude=media.latitude,
                longitude=media.longitude,
                suggestions=[],
            )
        )

    return responses
