from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_roles, get_current_user
from app.core.datetime_utils import format_datetime_iso_ro
from app.models import media as dbm
from app.models.user import User
from app.services.rag import sync_detection_updates

router = APIRouter(
    prefix="/issues",
    tags=["Issues"],
    dependencies=[require_roles("user", "authority", "admin")],
)

# ---------- Pydantic request models ----------

class StatusUpdate(BaseModel):
    status: Literal["open", "resolved", "ignored"]

class SeverityUpdate(BaseModel):
    severity: Literal["low", "medium", "high"]

class AssignUpdate(BaseModel):
    assigned_to: Optional[str] = None  # null = unassign

class VerifyUpdate(BaseModel):
    verified: bool


# ---------- Helpers ----------

def _get_detection_or_404(db: Session, detection_id: int) -> dbm.Detection:
    det = db.query(dbm.Detection).filter(dbm.Detection.id == detection_id).first()
    if not det:
        raise HTTPException(404, f"Detection {detection_id} not found")
    return det

def _ensure_user_exists_and_is_authority(db: Session, username: str):
    u = db.query(User).filter(User.username == username).first()
    if not u:
        raise HTTPException(404, f"User '{username}' not found")
    if u.role != "authority":
        raise HTTPException(409, "Can only assign to authority users")

def _is_verified(det: dbm.Detection) -> bool:
    return bool(det.verified_at)


# ---------- Endpoints ----------

@router.get("/{detection_id}")
def get_issue(
    detection_id: int,
    db: Session = Depends(get_db),
):
    det = _get_detection_or_404(db, detection_id)
    
    # Get media info for fallback thumbnails
    frame = db.query(dbm.Frame).filter(dbm.Frame.id == det.frame_id).first()
    media = db.query(dbm.Media).filter(dbm.Media.id == frame.media_id).first() if frame else None
    
    # Determine thumbnail URL with fallbacks
    thumbnail_url = None
    if det.track_thumbnail_url:
        thumbnail_url = det.track_thumbnail_url
    elif media and media.media_type == "video" and media.thumbnail_filename:
        thumbnail_url = f"/static/{media.thumbnail_filename}"
    
    return {
        "id": det.id,
        "frame_id": det.frame_id,
        "track_id": det.track_id,
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": det.confidence,
        "bbox": [det.x1, det.y1, det.x2, det.y2],
        "description": det.description,
        "solution": det.solution,
        "source": det.source.value,
        "status": det.status.value,
        "severity": det.severity.value,
        "assigned_to": det.assigned_to,
        "verified_by": det.verified_by,
        "verified_at": format_datetime_iso_ro(det.verified_at),
        "created_at": format_datetime_iso_ro(det.created_at),
        "resolved_at": format_datetime_iso_ro(det.resolved_at),
        "track_thumbnail_url": thumbnail_url,
    }


@router.patch("/{detection_id}/status")
async def update_status(
    detection_id: int,
    body: StatusUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    det = _get_detection_or_404(db, detection_id)
    try:
        new_status = dbm.IssueStatus(body.status)
    except Exception:
        raise HTTPException(400, "Invalid status")

    if new_status == dbm.IssueStatus.resolved:
        if not det.assigned_to:
            raise HTTPException(403, "Cannot resolve an unassigned issue")
        if det.assigned_to != current_user.username:
            raise HTTPException(403, "Only the assignee can resolve this issue")

    old_status = det.status
    if new_status == old_status:
        return {
            "id": det.id,
            "status": det.status.value,
            "resolved_at": format_datetime_iso_ro(det.resolved_at),
        }

    det.status = new_status
    det.resolved_at = (
        datetime.now(timezone.utc) if new_status == dbm.IssueStatus.resolved else None
    )
    db.add(det)
    db.commit()
    db.refresh(det)

    # Sync RAG chunks with updated status
    await sync_detection_updates(db, detection_id)

    # Publish event for notifications
    from app.services.event_helpers import create_issue_status_changed_event
    from app.services.event_publisher import publish_event
    event = create_issue_status_changed_event(
        issue_id=det.id,
        old_status=old_status.value,
        new_status=det.status.value,
        changed_by=current_user.username,
        severity=det.severity.value,
        class_name=det.class_name
    )
    publish_event(event)

    return {
        "id": det.id,
        "old_status": old_status.value,
        "new_status": det.status.value,
        "resolved_at": format_datetime_iso_ro(det.resolved_at),
    }


@router.patch("/{detection_id}/severity")
async def update_severity(
    detection_id: int,
    body: SeverityUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    det = _get_detection_or_404(db, detection_id)
    old_severity = det.severity
    try:
        det.severity = dbm.Severity(body.severity)
    except Exception:
        raise HTTPException(400, "Invalid severity")

    db.add(det)
    db.commit()
    db.refresh(det)

    # Sync RAG chunks with updated severity
    await sync_detection_updates(db, detection_id)

    # Publish event for notifications
    from app.services.event_helpers import create_issue_severity_changed_event
    from app.services.event_publisher import publish_event
    event = create_issue_severity_changed_event(
        issue_id=det.id,
        old_severity=old_severity.value,
        new_severity=det.severity.value,
        changed_by=current_user.username,
        class_name=det.class_name
    )
    publish_event(event)

    return {"id": det.id, "severity": det.severity.value}


@router.patch("/{detection_id}/assign", dependencies=[require_roles("admin")])  # <-- only admins
async def assign_issue(
    detection_id: int,
    body: AssignUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    det = _get_detection_or_404(db, detection_id)

    if body.assigned_to:
        if not _is_verified(det):
            raise HTTPException(409, "Issue must be verified before it can be assigned")
        _ensure_user_exists_and_is_authority(db, body.assigned_to)

    det.assigned_to = body.assigned_to or None
    db.add(det)
    db.commit()
    db.refresh(det)

    # Sync RAG chunks with updated assignment
    await sync_detection_updates(db, detection_id)

    # Publish event for notifications (only if assigned, not unassigned)
    if body.assigned_to:
        from app.services.event_helpers import create_issue_assigned_event
        from app.services.event_publisher import publish_event
        event = create_issue_assigned_event(
            issue_id=det.id,
            assigned_to=body.assigned_to,
            assigned_by=current_user.username,
            severity=det.severity.value,
            class_name=det.class_name
        )
        publish_event(event)

    return {"id": det.id, "assigned_to": det.assigned_to}

@router.patch("/{detection_id}/verify", dependencies=[require_roles("admin")])  # <-- admin only
async def verify_issue(
    detection_id: int,
    body: VerifyUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    det = _get_detection_or_404(db, detection_id)
    if body.verified:
        det.verified_by = current_user.username
        det.verified_at = datetime.now(timezone.utc)
    else:
        det.verified_by = None
        det.verified_at = None

    db.add(det)
    db.commit()
    db.refresh(det)

    # Sync RAG chunks with updated verification
    await sync_detection_updates(db, detection_id)

    # Publish event for notifications (only when verified, not unverified)
    if body.verified:
        from app.services.event_helpers import create_issue_verified_event
        from app.services.event_publisher import publish_event
        event = create_issue_verified_event(
            issue_id=det.id,
            verified_by=current_user.username,
            severity=det.severity.value,
            class_name=det.class_name
        )
        publish_event(event)

    return {
        "id": det.id,
        "verified_by": det.verified_by,
        "verified_at": format_datetime_iso_ro(det.verified_at),
    }
