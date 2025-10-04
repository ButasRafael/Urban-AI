from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_roles, get_current_user
from app.models.user import User
from app.services.notification_preferences import NotificationPreferenceService

router = APIRouter(
    prefix="/notifications",
    tags=["Notifications"],
    dependencies=[require_roles("user", "authority", "admin")],
)


# ---------- Pydantic request models ----------

class UpdatePreferencesRequest(BaseModel):
    email_enabled: Optional[bool] = None
    push_enabled: Optional[bool] = None
    event_preferences: Optional[Dict[str, Dict[str, bool]]] = None
    min_severity: Optional[str] = None
    quiet_hours: Optional[Dict[str, Any]] = None
    digest_enabled: Optional[bool] = None
    digest_frequency: Optional[str] = None


class UpdateFCMTokenRequest(BaseModel):
    fcm_token: str


# ---------- Endpoints ----------

@router.get("/preferences")
def get_preferences(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get the current user's notification preferences
    """
    prefs = NotificationPreferenceService.get_or_create_preferences(db, current_user.username)

    return {
        "email_enabled": prefs.email_enabled,
        "push_enabled": prefs.push_enabled,
        "event_preferences": prefs.event_preferences,
        "min_severity": prefs.min_severity,
        "quiet_hours": prefs.quiet_hours,
        "digest_enabled": prefs.digest_enabled,
        "digest_frequency": prefs.digest_frequency,
        "has_fcm_token": bool(prefs.fcm_token),
    }


@router.patch("/preferences")
def update_preferences(
    body: UpdatePreferencesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update the current user's notification preferences
    """
    prefs = NotificationPreferenceService.get_or_create_preferences(db, current_user.username)

    # Update only the fields that are provided
    if body.email_enabled is not None:
        prefs.email_enabled = body.email_enabled

    if body.push_enabled is not None:
        prefs.push_enabled = body.push_enabled

    if body.event_preferences is not None:
        prefs.event_preferences = body.event_preferences

    if body.min_severity is not None:
        if body.min_severity not in ["low", "medium", "high"]:
            raise HTTPException(400, "Invalid severity level. Must be 'low', 'medium', or 'high'")
        prefs.min_severity = body.min_severity

    if body.quiet_hours is not None:
        prefs.quiet_hours = body.quiet_hours

    if body.digest_enabled is not None:
        prefs.digest_enabled = body.digest_enabled

    if body.digest_frequency is not None:
        if body.digest_frequency not in ["immediate", "hourly", "daily"]:
            raise HTTPException(400, "Invalid digest frequency. Must be 'immediate', 'hourly', or 'daily'")
        prefs.digest_frequency = body.digest_frequency

    db.commit()
    db.refresh(prefs)

    return {
        "email_enabled": prefs.email_enabled,
        "push_enabled": prefs.push_enabled,
        "event_preferences": prefs.event_preferences,
        "min_severity": prefs.min_severity,
        "quiet_hours": prefs.quiet_hours,
        "digest_enabled": prefs.digest_enabled,
        "digest_frequency": prefs.digest_frequency,
        "has_fcm_token": bool(prefs.fcm_token),
    }


@router.post("/fcm-token")
def update_fcm_token(
    body: UpdateFCMTokenRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update the current user's FCM device token for push notifications
    """
    NotificationPreferenceService.update_fcm_token(db, current_user.username, body.fcm_token)

    return {"message": "FCM token updated successfully"}


@router.delete("/fcm-token")
def delete_fcm_token(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete the current user's FCM device token
    """
    NotificationPreferenceService.update_fcm_token(db, current_user.username, None)

    return {"message": "FCM token deleted successfully"}
