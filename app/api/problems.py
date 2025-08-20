# app/api/problems.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional, Literal

from app.core.database import get_db
from app.core.security import require_roles
from app.models import media as dbm
from app.models.schemas_portal import ProblemOut  # existing list-by-media DTO

router = APIRouter(
    prefix="/problems",
    tags=["Problems"],
    dependencies=[require_roles("admin", "authority")],
)

# ---------- Existing endpoint (unchanged) ----------
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
        if klass and not classes:           # filter out if class not found
            continue

        detects = (
            db.query(dbm.Detection)
              .join(dbm.Frame, dbm.Frame.id == dbm.Detection.frame_id)
              .filter(dbm.Frame.media_id == m.id)
              .all()
        )
        descriptions = [d.description or "n/a" for d in detects]
        solutions    = [d.solution    or "n/a" for d in detects]
        out.append(ProblemOut(
            media_id=m.id,
            address=m.address,
            latitude=m.latitude,
            longitude=m.longitude,
            user_username=m.user_username,
            media_type=m.media_type,
            annotated_image_url = f"/static/{m.id}.jpg" if m.media_type=="image" else None,
            annotated_video_url = f"/static/{m.id}.mp4" if m.media_type=="video" else None,
            created_at=m.created_at,
            predicted_classes=classes,
            descriptions=descriptions,
            solutions=solutions,
        ))
    return out


# ---------- New: issue list (detections) with lifecycle filters ----------
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
    if assigned_to:                                                     # <-- NEW
        q = q.filter(dbm.Detection.assigned_to == assigned_to)

    rows = (
        q.order_by(desc(dbm.Detection.created_at))
         .limit(limit)
         .offset(offset)
         .all()
    )
    out: list[IssueOut] = []
    for det, fr, med in rows:
        img_url   = f"/static/{med.id}.jpg" if med.media_type == "image" else None
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
        ))
    return out


# ---------- New: tiny summary for portal filters ----------
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

@router.get("/issues/summary_by_media")
def issues_summary_by_media(
    db: Session = Depends(get_db),
    media_ids: List[int] = Query(..., description="Repeat: media_ids=1&media_ids=2"),
):
    # media_id, severity, status, count
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

