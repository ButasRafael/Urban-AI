# app/api/inference_routes.py
from fastapi import (
    APIRouter, UploadFile, File, HTTPException, Depends, Query, Form
)
from pathlib import Path
import shutil, uuid, logging
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
from app.utils.exif import get_image_gps
from sqlalchemy.orm import Session
from typing import List
import httpx
from openai import InternalServerError
import numpy as np

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
    # if both dimensions are already under the cap, do nothing
    if max(h, w) <= MAX_DIM:
        return img

    # compute scale factor so that the longer side == MAX_DIM
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

@router.post(
    "/image",
    response_model=ImageResponse,
    dependencies=[require_roles("user", "admin")],
)
async def detect_image(
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

    # save upload to temp
    path = _save_temp(file)
    #lat, lng = get_image_gps(str(path))
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
    # 1) create Media row so we have media.id
    media = dbm.Media(
        filename=file.filename,
        media_type="image",
        user_username=current_user.username,
        address=final_address,
        latitude=latitude,
        longitude=longitude,
    )
    db.add(media)
    db.commit()
    db.refresh(media)
    logger.debug("Inserted media row", extra={"media_id": media.id})

    # 2) run inference with run_id = media.id
    try:
        annotated, dets = await svc.process_image_combined(img, use_sam, str(media.id))
    except InternalServerError:
        raise HTTPException(502, "Upstream LLM error; please retry shortly.")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.exception("❌ Inference or DB write failed")
        db.delete(media)
        db.commit()
        raise HTTPException(500, f"Inference failed: {e!r}")


    INFERENCE_IMAGE_LATENCY.observe(0)  # or record actual timing if you capture it

    # 3) update width/height
    media.width = annotated.shape[1]
    media.height = annotated.shape[0]
    db.add(media)
    db.commit()

    # 4) persist Frame + Detection rows
    fr = dbm.Frame(media_id=media.id, frame_index=0, timestamp=0.0)
    db.add(fr); db.commit(); db.refresh(fr)
    for d in dets:
        db.add(dbm.Detection(
            frame_id=fr.id,
            track_id=d["track_id"],
            class_id=d.get("class_id", -1),
            class_name=d["class_name"],
            confidence=d["confidence"],
            x1=d["bbox"][0], y1=d["bbox"][1],
            x2=d["bbox"][2], y2=d["bbox"][3],
            mask_rle      = d.get("mask_rle", {}),
            mask_polygon  = d.get("mask_polygon", []),
            description=d.get("description"),
            solution=d.get("solution"),
        ))
    db.commit()

    # 5) write annotated image file
    STATIC_DIR.mkdir(exist_ok=True)
    out_name = f"{media.id}.jpg"
    out_path = STATIC_DIR / out_name
    await run_in_threadpool(cv2.imwrite, str(out_path), annotated)

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
    # create media row
    media = dbm.Media(
        filename=file.filename,
        media_type="video",
        user_username=current_user.username,
        address=final_address,
        latitude=latitude,
        longitude=longitude,
    )
    db.add(media); db.commit(); db.refresh(media)

    # run inference into static/<media.id> folder
    try:
        annotated_tmp, frames_meta = await run_in_threadpool(
            svc.process_video, tmp_path, use_sam
        )
    except Exception as e:
        sentry_sdk.capture_exception(e)
        db.delete(media); db.commit()
        raise HTTPException(500, "Inference failed")

    media.num_frames = len(frames_meta)
    db.add(media); db.commit()

    # persist frames & detections
    for fr in frames_meta:
        fr_row = dbm.Frame(
            media_id=media.id,
            frame_index=fr["frame_index"],
            timestamp=fr["timestamp_ms"],
        )
        db.add(fr_row); db.commit(); db.refresh(fr_row)
        for d in fr["objects"]:
            db.add(dbm.Detection(
                frame_id=fr_row.id,
                track_id=d["track_id"],
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                x1=d["bbox"][0], y1=d["bbox"][1],
                x2=d["bbox"][2], y2=d["bbox"][3],
            ))
        db.commit()

    # move final mp4
    STATIC_DIR.mkdir(exist_ok=True)
    out_name = f"{media.id}.mp4"
    await run_in_threadpool(shutil.move, str(annotated_tmp), str(STATIC_DIR / out_name))

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
        descriptions: List[Optional[str]] = []
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
        ))
    return out

@router.post(
    "/images",
    response_model=List[ImageResponse],
    dependencies=[require_roles("user", "admin")],
)
async def detect_images(
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
        # 1) validate extension
        ext = Path(file.filename).suffix.lower()
        if ext not in IMAGE_EXTS:
            raise HTTPException(400, f"Unsupported image format: {ext}")

        # 2) save temp, read & resize
        tmp_path = _save_temp(file)
        img = await run_in_threadpool(cv2.imread, str(tmp_path))
        if img is None:
            raise HTTPException(400, "Could not decode image")
        img = _resize_if_needed(img)

        # 3) reverse‐geocode if needed
        if address:
            final_address = address
        elif latitude is not None and longitude is not None:
            final_address = await run_in_threadpool(reverse_geocode, latitude, longitude)
        else:
            final_address = ""

        # 4) insert Media row
        media = dbm.Media(
            filename=file.filename,
            media_type="image",
            user_username=current_user.username,
            address=final_address,
            latitude=latitude,
            longitude=longitude,
        )
        db.add(media)
        db.commit()
        db.refresh(media)

        # 5) run your combined pipeline
        try:
            annotated, dets = await svc.process_image_combined(img, use_sam, str(media.id))
        except Exception as e:
            # on failure rollback this media and bubble up
            db.delete(media); db.commit()
            raise HTTPException(500, f"Inference failed for {file.filename}: {e!r}")

        # 6) update media dims
        media.width  = annotated.shape[1]
        media.height = annotated.shape[0]
        db.add(media); db.commit()

        # 7) persist Frame + Detection
        fr = dbm.Frame(media_id=media.id, frame_index=0, timestamp=0.0)
        db.add(fr); db.commit(); db.refresh(fr)
        for d in dets:
            db.add(dbm.Detection(
                frame_id=fr.id,
                track_id=d.get("track_id"),
                class_id=d.get("class_id", -1),
                class_name=d["class_name"],
                confidence=d["confidence"],
                x1=d["bbox"][0], y1=d["bbox"][1],
                x2=d["bbox"][2], y2=d["bbox"][3],
                mask_rle      = d.get("mask", {}).get("rle", {}),
                mask_polygon  = d.get("mask", {}).get("polygon", []),
                description   = d.get("description"),
                solution      = d.get("solution"),
            ))
        db.commit()

        # 8) write out the annotated image
        out_name = f"{media.id}.jpg"
        out_path = STATIC_DIR / out_name
        await run_in_threadpool(cv2.imwrite, str(out_path), annotated)

        # 9) collect response object
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
    # 1) save the zip
    tmp_zip = _save_temp(archive)

    # 2) extract supported images
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
        # — reverse‐geocode if needed
        if address:
            final_address = address
        elif latitude is not None and longitude is not None:
            final_address = await run_in_threadpool(reverse_geocode, latitude, longitude)
        else:
            final_address = ""

        # — insert Media row
        media = dbm.Media(
            filename=filename,
            media_type="image",
            user_username=current_user.username,
            address=final_address,
            latitude=latitude,
            longitude=longitude,
        )
        db.add(media); db.commit(); db.refresh(media)

        # — run your pipeline
        try:
            annotated, dets = await svc.process_image_combined(img, use_sam, str(media.id))
        except Exception as e:
            db.delete(media); db.commit()
            raise HTTPException(500, f"Inference failed for {filename}: {e!r}")

        # — update dims
        media.width, media.height = annotated.shape[1], annotated.shape[0]
        db.add(media); db.commit()

        # — persist Frame + Detections
        fr = dbm.Frame(media_id=media.id, frame_index=0, timestamp=0.0)
        db.add(fr); db.commit(); db.refresh(fr)
        for d in dets:
            db.add(dbm.Detection(
                frame_id=fr.id,
                track_id=d.get("track_id"),
                class_id=d.get("class_id", -1),
                class_name=d["class_name"],
                confidence=d["confidence"],
                x1=d["bbox"][0], y1=d["bbox"][1],
                x2=d["bbox"][2], y2=d["bbox"][3],
                mask_rle     = d.get("mask", {}).get("rle", {}),
                mask_polygon = d.get("mask", {}).get("polygon", []),
                description  = d.get("description"),
                solution     = d.get("solution"),
            ))
        db.commit()

        # — write annotated image
        out_name = f"{media.id}.jpg"
        out_path = STATIC_DIR / out_name
        await run_in_threadpool(cv2.imwrite, str(out_path), annotated)

        # — build response
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
