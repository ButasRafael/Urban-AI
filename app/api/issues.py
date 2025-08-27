from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_roles, get_current_user
from app.models import media as dbm
from app.models.user import User

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
        "verified_at": det.verified_at.isoformat() if det.verified_at else None,
        "created_at": det.created_at.isoformat() if det.created_at else None,
        "resolved_at": det.resolved_at.isoformat() if det.resolved_at else None,
        "track_thumbnail_url": det.track_thumbnail_url,
    }


@router.patch("/{detection_id}/status")
def update_status(
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
            "resolved_at": det.resolved_at.isoformat() if det.resolved_at else None,
        }

    det.status = new_status
    det.resolved_at = (
        datetime.now(timezone.utc) if new_status == dbm.IssueStatus.resolved else None
    )
    db.add(det)
    db.commit()
    db.refresh(det)

    return {
        "id": det.id,
        "old_status": old_status.value,
        "new_status": det.status.value,
        "resolved_at": det.resolved_at.isoformat() if det.resolved_at else None,
    }


@router.patch("/{detection_id}/severity")
def update_severity(
    detection_id: int,
    body: SeverityUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    det = _get_detection_or_404(db, detection_id)
    try:
        det.severity = dbm.Severity(body.severity)
    except Exception:
        raise HTTPException(400, "Invalid severity")

    db.add(det)
    db.commit()
    db.refresh(det)
    return {"id": det.id, "severity": det.severity.value}


@router.patch("/{detection_id}/assign", dependencies=[require_roles("admin")])  # <-- only admins
def assign_issue(
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
    return {"id": det.id, "assigned_to": det.assigned_to}

@router.patch("/{detection_id}/verify", dependencies=[require_roles("admin")])  # <-- admin only
def verify_issue(
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
    return {
        "id": det.id,
        "verified_by": det.verified_by,
        "verified_at": det.verified_at.isoformat() if det.verified_at else None,
    }
