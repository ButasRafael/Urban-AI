from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.notification import NotificationLog, NotificationStatus
from app.core.datetime_utils import format_datetime_iso_ro

router = APIRouter(
    prefix="/notifications",
    tags=["Notification Store"],
)


class NotificationResponse(BaseModel):
    id: int
    event_type: str
    channel: str
    status: str
    delivered_channels: Optional[List[str]]  # ["email"], ["push"], ["email", "push"], or []
    event_data: dict
    sent_at: Optional[str]
    read_at: Optional[str]
    created_at: str


class UnreadCountResponse(BaseModel):
    unread_count: int


@router.get("/store", response_model=List[NotificationResponse])
def get_notifications(
    limit: int = 50,
    offset: int = 0,
    unread_only: bool = False,
    status: Optional[str] = None,  # Filter by status: "pending", "sent", "failed"
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get user's notifications from the store

    Query parameters:
    - limit: Maximum number of notifications to return (default: 50, max: 200)
    - offset: Number of notifications to skip for pagination (default: 0)
    - unread_only: If true, only return unread notifications (default: false)
    - status: Filter by status - "pending", "sent", or "failed" (default: all statuses)
    """
    # Clamp limit to prevent abuse
    limit = min(limit, 200)

    query = db.query(NotificationLog).filter(
        NotificationLog.username == current_user.username
        # Show all statuses by default (pending, sent, failed) so users can see everything
    )

    # Optional status filter
    if status:
        try:
            status_enum = NotificationStatus(status)
            query = query.filter(NotificationLog.status == status_enum)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}. Must be 'pending', 'sent', or 'failed'")

    if unread_only:
        query = query.filter(NotificationLog.read_at == None)

    notifications = query.order_by(desc(NotificationLog.created_at)).limit(limit).offset(offset).all()

    return [
        NotificationResponse(
            id=n.id,
            event_type=n.event_type.value,
            channel=n.channel.value,
            status=n.status.value,
            delivered_channels=n.delivered_channels,
            event_data=n.event_data,
            sent_at=format_datetime_iso_ro(n.sent_at),
            read_at=format_datetime_iso_ro(n.read_at),
            created_at=format_datetime_iso_ro(n.created_at),
        )
        for n in notifications
    ]


@router.get("/store/unread-count", response_model=UnreadCountResponse)
def get_unread_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get count of unread notifications
    """
    count = db.query(NotificationLog).filter(
        NotificationLog.username == current_user.username,
        NotificationLog.status == NotificationStatus.sent,
        NotificationLog.read_at == None
    ).count()

    return UnreadCountResponse(unread_count=count)


@router.patch("/store/{notification_id}/read")
def mark_as_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Mark a notification as read
    """
    notification = db.query(NotificationLog).filter(
        NotificationLog.id == notification_id,
        NotificationLog.username == current_user.username
    ).first()

    if not notification:
        raise HTTPException(404, "Notification not found")

    if not notification.read_at:
        notification.read_at = datetime.now(timezone.utc)
        db.commit()

    return {"success": True, "read_at": format_datetime_iso_ro(notification.read_at)}


@router.post("/store/mark-all-read")
def mark_all_as_read(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Mark all notifications as read
    """
    db.query(NotificationLog).filter(
        NotificationLog.username == current_user.username,
        NotificationLog.read_at == None
    ).update({"read_at": datetime.now(timezone.utc)})

    db.commit()

    return {"success": True, "message": "All notifications marked as read"}


@router.delete("/store/{notification_id}")
def delete_notification(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a notification
    """
    notification = db.query(NotificationLog).filter(
        NotificationLog.id == notification_id,
        NotificationLog.username == current_user.username
    ).first()

    if not notification:
        raise HTTPException(404, "Notification not found")

    db.delete(notification)
    db.commit()

    return {"success": True, "message": "Notification deleted"}
