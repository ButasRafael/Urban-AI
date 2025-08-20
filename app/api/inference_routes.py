from fastapi import (
    APIRouter, UploadFile, File, HTTPException, Depends, Query, Form,
    BackgroundTasks
)
from pathlib import Path
import shutil, uuid, logging, time
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
            params={"format":"jsonv2","lat":lat,"lon":lon},
            timeout=5.0
        )
        r.raise_for_status()
        return r.json().get("display_name","")
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
    latitude:  float | None = Form(None),
    longitude: float | None = Form(None),
    address:   str   | None = Form(None),
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
    db.add(media); db.commit(); db.refresh(media)
    logger.debug("Inserted media row", extra={"media_id": media.id})

    # 2) run inference with real latency measurement
    t0 = time.perf_counter()
    try:
        annotated, dets = await svc.process_image_combined(img, use_sam, str(media.id))
    except InternalServerError:
        raise HTTPException(502, "Upstream LLM error; please retry shortly.")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.exception("❌ Inference or DB write failed")
        db.delete(media); db.commit()
        raise HTTPException(500, f"Inference failed: {e!r}")
    dt_s = time.perf_counter() - t0
    INFERENCE_IMAGE_LATENCY.observe(dt_s)

    # 3) update media metadata (dims + total ms)
    media.width = annotated.shape[1]
    media.height = annotated.shape[0]
    media.process_ms_total = int(dt_s * 1000)
    db.add(media); db.commit()

    # 4) persist Frame + Detection rows
    fr = dbm.Frame(media_id=media.id, frame_index=0, timestamp=0.0)
    db.add(fr); db.commit(); db.refresh(fr)

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
            mask_rle     = mask.get("rle", {}),
            mask_polygon = mask.get("polygon", []),
            description  = d.get("description"),
            solution     = d.get("solution"),
            source       = _infer_source(d),
            severity=_to_severity_enum(d.get("severity")),
        ))
    db.commit()

    # 5) write annotated image file
    STATIC_DIR.mkdir(exist_ok=True)
    out_name = f"{media.id}.jpg"
    out_path = STATIC_DIR / out_name
    await run_in_threadpool(cv2.imwrite, str(out_path), annotated)

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
    latitude:  float | None = Form(None),
    longitude: float | None = Form(None),
    address:   str   | None = Form(None),
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
    db.add(media); db.commit(); db.refresh(media)

    t0 = time.perf_counter()
    try:
        annotated_tmp, frames_meta = await run_in_threadpool(
            svc.process_video, tmp_path, use_sam
        )
    except Exception as e:
        sentry_sdk.capture_exception(e)
        db.delete(media); db.commit()
        raise HTTPException(500, f"Inference failed: {e!r}")
    dt_s = time.perf_counter() - t0
    INFERENCE_VIDEO_LATENCY.observe(dt_s)

    media.num_frames = len(frames_meta)
    media.process_ms_total = int(dt_s * 1000)
    db.add(media); db.commit()

    for fr in frames_meta:
        fr_row = dbm.Frame(
            media_id=media.id,
            frame_index=fr["frame_index"],
            timestamp=fr["timestamp_ms"],
        )
        db.add(fr_row); db.commit(); db.refresh(fr_row)
        for d in fr["objects"]:
            mask = d.get("mask", {}) or {}
            db.add(dbm.Detection(
                frame_id=fr_row.id,
                track_id=d.get("track_id"),
                class_id=d.get("class_id", -1),
                class_name=d.get("class_name", ""),
                confidence=d.get("confidence"),
                x1=d["bbox"][0], y1=d["bbox"][1],
                x2=d["bbox"][2], y2=d["bbox"][3],
                mask_rle     = mask.get("rle", {}),
                mask_polygon = mask.get("polygon", []),
                source       = _infer_source(d),
                severity=_to_severity_enum(d.get("severity")),

            ))
        db.commit()

    STATIC_DIR.mkdir(exist_ok=True)
    out_name = f"{media.id}.mp4"
    await run_in_threadpool(shutil.move, str(annotated_tmp), str(STATIC_DIR / out_name))

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
        img_url   = f"/static/{m.id}.jpg" if m.media_type == "image" else None
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
    latitude:  float | None = Form(None),
    longitude: float | None = Form(None),
    address:   str   | None = Form(None),
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
        db.add(media); db.commit(); db.refresh(media)

        t0 = time.perf_counter()
        try:
            annotated, dets = await svc.process_image_combined(img, use_sam, str(media.id))
        except Exception as e:
            db.delete(media); db.commit()
            raise HTTPException(500, f"Inference failed for {file.filename}: {e!r}")
        dt_s = time.perf_counter() - t0

        media.width  = annotated.shape[1]
        media.height = annotated.shape[0]
        media.process_ms_total = int(dt_s * 1000)
        db.add(media); db.commit()

        fr = dbm.Frame(media_id=media.id, frame_index=0, timestamp=0.0)
        db.add(fr); db.commit(); db.refresh(fr)
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
                mask_rle     = mask.get("rle", {}),
                mask_polygon = mask.get("polygon", []),
                description  = d.get("description"),
                solution     = d.get("solution"),
                source       = _infer_source(d),
                severity=_to_severity_enum(d.get("severity")),
            ))
        db.commit()

        out_name = f"{media.id}.jpg"
        out_path = STATIC_DIR / out_name
        await run_in_threadpool(cv2.imwrite, str(out_path), annotated)

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
    latitude:  float | None = Form(None),
    longitude: float | None = Form(None),
    address:   str   | None = Form(None),
):
    tmp_zip = _save_temp(archive)

    images: List[tuple[str, np.ndarray]] = []
    with zipfile.ZipFile(tmp_zip, 'r') as zf:
        for name in zf.namelist():
            ext = Path(name).suffix.lower()
            if ext in IMAGE_EXTS:
                data = zf.read(name)
                arr  = np.frombuffer(data, np.uint8)
                img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
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
        db.add(media); db.commit(); db.refresh(media)

        t0 = time.perf_counter()
        try:
            annotated, dets = await svc.process_image_combined(img, use_sam, str(media.id))
        except Exception as e:
            db.delete(media); db.commit()
            raise HTTPException(500, f"Inference failed for {filename}: {e!r}")
        dt_s = time.perf_counter() - t0

        media.width, media.height = annotated.shape[1], annotated.shape[0]
        media.process_ms_total = int(dt_s * 1000)
        db.add(media); db.commit()

        fr = dbm.Frame(media_id=media.id, frame_index=0, timestamp=0.0)
        db.add(fr); db.commit(); db.refresh(fr)
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
                mask_rle     = mask.get("rle", {}),
                mask_polygon = mask.get("polygon", []),
                description  = d.get("description"),
                solution     = d.get("solution"),
                source       = _infer_source(d),
                severity=_to_severity_enum(d.get("severity")),
            ))
        db.commit()

        out_name = f"{media.id}.jpg"
        out_path = STATIC_DIR / out_name
        await run_in_threadpool(cv2.imwrite, str(out_path), annotated)

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
