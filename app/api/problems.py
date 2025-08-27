from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional, Literal

from app.core.database import get_db
from app.core.security import require_roles
from app.models import media as dbm
from app.models.schemas_portal import ProblemOut

router = APIRouter(
    prefix="/problems",
    tags=["Problems"],
    dependencies=[require_roles("admin", "authority")],
)

@router.get("", response_model=List[ProblemOut])
def all_problems(
    media_type: str | None = Query(None, pattern="^(image|video)$"),
    klass: str | None = Query(None, description="Filter by YOLO/SAM class"),
    db: Session = Depends(get_db)
):
    q = db.query(dbm.Media)
    if media_type:
        q = q.filter(dbm.Media.media_type == media_type)
    rows = q.order_by(dbm.Media.created_at.desc()).all()

    out: list[ProblemOut] = []
    for m in rows:
        classes_q = (db.query(dbm.Detection.class_name)
                       .join(dbm.Frame, dbm.Frame.id == dbm.Detection.frame_id)
                       .filter(dbm.Frame.media_id == m.id)
                       .distinct())
        if klass:
            classes_q = classes_q.filter(dbm.Detection.class_name == klass)
        classes = [c[0] for c in classes_q]
        if klass and not classes:
            continue

        detects = (
            db.query(dbm.Detection)
              .join(dbm.Frame, dbm.Frame.id == dbm.Detection.frame_id)
              .filter(dbm.Frame.media_id == m.id)
              .all()
        )
        descriptions = [d.description or "n/a" for d in detects]
        solutions    = [d.solution    or "n/a" for d in detects]
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

        out.append(ProblemOut(
            media_id=m.id,
            address=m.address,
            latitude=m.latitude,
            longitude=m.longitude,
            user_username=m.user_username,
            media_type=m.media_type,
            annotated_image_url=img_url,
            annotated_video_url=video_url,
            created_at=m.created_at,
            predicted_classes=classes,
            descriptions=descriptions,
            solutions=solutions,
            summary_description=m.summary_description,
            summary_solution=m.summary_solution,
        ))
    return out


from pydantic import BaseModel

class IssueOut(BaseModel):
    id: int
    media_id: int
    frame_id: int
    created_at: str
    class_name: str
    confidence: float | None
    bbox: List[float]
    description: str | None
    solution: str | None
    severity: Literal["low", "medium", "high"]
    status: Literal["open", "resolved", "ignored"]
    source: Literal["yolo", "gpt_dino", "sam_fallback"]
    annotated_image_url: Optional[str] = None
    annotated_video_url: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    assigned_to: Optional[str] = None
    verified_by: Optional[str] = None
    verified_at: Optional[str] = None
    track_thumbnail_url: Optional[str] = None

@router.get("/issues", response_model=List[IssueOut])
def list_issues(
    db: Session = Depends(get_db),
    media_id: Optional[int] = Query(None),
    media_type: Optional[Literal["image", "video"]] = Query(None),
    klass: Optional[str] = Query(None, description="Exact class_name"),
    severity: Optional[Literal["low", "medium", "high"]] = Query(None),
    status: Optional[Literal["open", "resolved", "ignored"]] = Query(None),
    source: Optional[Literal["yolo", "gpt_dino", "sam_fallback"]] = Query(None),
    assigned_to: Optional[str] = Query(None, description="Filter by assignee username"),  # <-- NEW
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    q = (
        db.query(dbm.Detection, dbm.Frame, dbm.Media)
          .join(dbm.Frame, dbm.Frame.id == dbm.Detection.frame_id)
          .join(dbm.Media, dbm.Media.id == dbm.Frame.media_id)
    )

    if media_id is not None:
        q = q.filter(dbm.Media.id == media_id)
    if media_type:
        q = q.filter(dbm.Media.media_type == media_type)
    if klass:
        q = q.filter(dbm.Detection.class_name == klass)
    if severity:
        q = q.filter(dbm.Detection.severity == dbm.Severity(severity))
    if status:
        q = q.filter(dbm.Detection.status == dbm.IssueStatus(status))
    if source:
        q = q.filter(dbm.Detection.source == dbm.DetectionSource(source))
    if assigned_to:
        q = q.filter(dbm.Detection.assigned_to == assigned_to)

    rows = (
        q.order_by(desc(dbm.Detection.created_at))
         .limit(limit)
         .offset(offset)
         .all()
    )
    out: list[IssueOut] = []
    for det, fr, med in rows:
        # Use stored UUID filenames if available, fallback to ID-based names for backward compatibility
        if med.static_filename:
            if med.media_type == "image":
                img_url = f"/static/{med.static_filename}"
                video_url = None
            else:
                video_url = f"/static/{med.static_filename}"
                img_url = f"/static/{med.thumbnail_filename}" if med.thumbnail_filename else f"/static/{med.id}.jpg"
        else:
            # Fallback for old records without UUID filenames
            img_url = f"/static/{med.id}.jpg" if med.media_type == "image" else None
            video_url = f"/static/{med.id}.mp4" if med.media_type == "video" else None
        out.append(IssueOut(
            id=det.id,
            media_id=med.id,
            frame_id=fr.id,
            created_at=det.created_at.isoformat(),
            class_name=det.class_name,
            confidence=det.confidence,
            bbox=[det.x1, det.y1, det.x2, det.y2],
            description=det.description,
            solution=det.solution,
            severity=(det.severity.value if hasattr(det.severity, "value") else str(det.severity)),
            status=(det.status.value if hasattr(det.status, "value") else str(det.status)),
            source=(det.source.value if hasattr(det.source, "value") else str(det.source)),
            annotated_image_url=img_url,
            annotated_video_url=video_url,
            address=med.address,
            latitude=med.latitude,
            longitude=med.longitude,
            assigned_to=det.assigned_to,
            verified_by=det.verified_by,
            verified_at=det.verified_at.isoformat() if det.verified_at else None,
            track_thumbnail_url=det.track_thumbnail_url,
        ))
    return out


@router.get("/issues/summary")
def issues_summary(
    db: Session = Depends(get_db),
    media_type: Optional[Literal["image", "video"]] = Query(None),
):
    q = (
        db.query(
            dbm.Detection.severity,
            dbm.Detection.status,
            func.count().label("c"),
        )
        .join(dbm.Frame, dbm.Frame.id == dbm.Detection.frame_id)
        .join(dbm.Media, dbm.Media.id == dbm.Frame.media_id)
    )
    if media_type:
        q = q.filter(dbm.Media.media_type == media_type)

    rows = q.group_by(dbm.Detection.severity, dbm.Detection.status).all()

    summary = {}
    for sev, st, c in rows:
        sev_key = (sev.value if hasattr(sev, "value") else str(sev))
        st_key  = (st.value  if hasattr(st,  "value")  else str(st))
        summary.setdefault(sev_key, {}).setdefault(st_key, 0)
        summary[sev_key][st_key] += int(c)

    return summary

from typing import Dict

class BulkUpdateRequest(BaseModel):
    media_id: int
    status: Literal["resolved", "ignored"]  # Only allow resolved or ignored for bulk updates

@router.patch("/issues/bulk_status")
def bulk_update_status(
    request: BulkUpdateRequest,
    db: Session = Depends(get_db),
):
    """Update status of all open detections for a specific media item"""
    # Get all open detections for the media
    detections = (
        db.query(dbm.Detection)
        .join(dbm.Frame, dbm.Frame.id == dbm.Detection.frame_id)
        .join(dbm.Media, dbm.Media.id == dbm.Frame.media_id)
        .filter(
            dbm.Media.id == request.media_id,
            dbm.Detection.status == dbm.IssueStatus.open
        )
        .all()
    )
    
    if not detections:
        return {"updated_count": 0, "message": "No open detections found for this media"}
    
    # Update all detections
    updated_count = 0
    new_status = dbm.IssueStatus(request.status)
    
    for detection in detections:
        detection.status = new_status
        if new_status == dbm.IssueStatus.resolved:
            from datetime import datetime, timezone
            detection.resolved_at = datetime.now(timezone.utc)
        updated_count += 1
    
    db.commit()
    
    return {
        "updated_count": updated_count,
        "media_id": request.media_id,
        "new_status": request.status,
        "message": f"Updated {updated_count} detection(s) to {request.status}"
    }

@router.get("/issues/summary_by_media")
def issues_summary_by_media(
    db: Session = Depends(get_db),
    media_ids: List[int] = Query(..., description="Repeat: media_ids=1&media_ids=2"),
):
    rows = (
        db.query(
            dbm.Media.id.label("media_id"),
            dbm.Detection.severity,
            dbm.Detection.status,
            func.count().label("c"),
        )
        .join(dbm.Frame, dbm.Frame.id == dbm.Detection.frame_id)
        .join(dbm.Media, dbm.Media.id == dbm.Frame.media_id)
        .filter(dbm.Media.id.in_(media_ids))
        .group_by(dbm.Media.id, dbm.Detection.severity, dbm.Detection.status)
        .all()
    )

    out: Dict[int, dict] = {}
    for media_id, sev, st, c in rows:
        sev_key = sev.value if hasattr(sev, "value") else str(sev)
        st_key  = st.value  if hasattr(st,  "value") else str(st)
        out.setdefault(media_id, {}).setdefault(sev_key, {}).setdefault(st_key, 0)
        out[media_id][sev_key][st_key] += int(c)

    return out

