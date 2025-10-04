import enum
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Enum as SQLEnum, Text, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.ext.mutable import MutableDict, MutableList
from app.core.database import Base


class NotificationChannel(str, enum.Enum):
    email = "email"
    push = "push"


class EventType(str, enum.Enum):
    issue_created = "issue_created"
    issue_status_changed = "issue_status_changed"
    issue_assigned = "issue_assigned"
    issue_severity_changed = "issue_severity_changed"
    issue_verified = "issue_verified"


class NotificationStatus(str, enum.Enum):
    pending = "pending"
    sent = "sent"
    failed = "failed"


class NotificationPreference(Base):
    __tablename__ = "notification_preferences"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE"), nullable=False)

    # Channel preferences
    email_enabled = Column(Boolean, default=True, nullable=False)
    push_enabled = Column(Boolean, default=True, nullable=False)

    # Event type preferences (stored as JSON for flexibility)
    # Example: {"issue_created": {"email": true, "push": true}, "issue_assigned": {"email": true, "push": true}}
    event_preferences = Column(MutableDict.as_mutable(JSON), default=dict, nullable=False)

    # Severity thresholds (minimum severity to trigger notification)
    # "low", "medium", or "high"
    min_severity = Column(String, default="low", nullable=False)

    # Quiet hours (stored as JSON)
    # Example: {"enabled": true, "start": "22:00", "end": "08:00", "timezone": "UTC"}
    quiet_hours = Column(MutableDict.as_mutable(JSON), default=dict, nullable=False)

    # Digest preferences
    digest_enabled = Column(Boolean, default=False, nullable=False)
    digest_frequency = Column(String, default="daily", nullable=False)  # "immediate", "hourly", "daily"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # FCM device token for push notifications
    fcm_token = Column(String, nullable=True)

    __table_args__ = (
        Index('ix_notification_preferences_username', 'username'),
    )


class NotificationTemplate(Base):
    __tablename__ = "notification_templates"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_type = Column(SQLEnum(EventType, name="event_type_enum"), nullable=False)
    channel = Column(SQLEnum(NotificationChannel, name="notification_channel_enum"), nullable=False)

    # Template content
    subject = Column(String, nullable=True)  # For email
    body = Column(Text, nullable=False)

    # Template variables (stored as JSON list)
    # Example: ["issue_id", "old_status", "new_status", "assigned_to"]
    variables = Column(MutableList.as_mutable(JSON), default=list, nullable=False)

    # Language support
    language = Column(String, default="en", nullable=False)

    # Version control
    version = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('ix_notification_templates_event_channel', 'event_type', 'channel', 'is_active'),
    )


class NotificationLog(Base):
    __tablename__ = "notification_log"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE"), nullable=False)

    event_type = Column(SQLEnum(EventType, name="event_type_enum"), nullable=False)
    channel = Column(SQLEnum(NotificationChannel, name="notification_channel_enum"), nullable=False)

    # Notification status
    status = Column(SQLEnum(NotificationStatus, name="notification_status_enum"), default=NotificationStatus.pending, nullable=False)

    # Event data (stored as JSON)
    event_data = Column(MutableDict.as_mutable(JSON), nullable=False)

    # Delivery details
    sent_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)

    # Rate limiting tracking
    retry_count = Column(Integer, default=0, nullable=False)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('ix_notification_log_username_created', 'username', 'created_at'),
        Index('ix_notification_log_status', 'status'),
    )
