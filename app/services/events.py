from typing import Optional, Any, Dict
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from app.models.notification import EventType


@dataclass
class Event:
    """Base event class"""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['event_type'] = self.event_type.value
        return result


@dataclass
class MediaCreatedEvent(Event):
    """Event fired when new media is created with detections"""
    def __init__(
        self,
        media_id: int,
        num_detections: int,
        created_by: str,
        media_type: str,
        max_severity: str = "medium",
        timestamp: Optional[datetime] = None
    ):
        super().__init__(
            event_type=EventType.media_created,
            timestamp=timestamp or datetime.now(timezone.utc),
            data={
                "media_id": media_id,
                "num_detections": num_detections,
                "created_by": created_by,
                "media_type": media_type,
                "severity": max_severity,  # Add severity for notification filtering
            }
        )


@dataclass
class IssueStatusChangedEvent(Event):
    """Event fired when an issue's status changes"""
    def __init__(
        self,
        issue_id: int,
        old_status: str,
        new_status: str,
        changed_by: str,
        severity: str,
        class_name: str,
        media_id: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ):
        super().__init__(
            event_type=EventType.issue_status_changed,
            timestamp=timestamp or datetime.now(timezone.utc),
            data={
                "issue_id": issue_id,
                "old_status": old_status,
                "new_status": new_status,
                "changed_by": changed_by,
                "severity": severity,
                "class_name": class_name,
                "media_id": media_id,
            }
        )


@dataclass
class IssueAssignedEvent(Event):
    """Event fired when an issue is assigned to a user"""
    def __init__(
        self,
        issue_id: int,
        assigned_to: str,
        assigned_by: str,
        severity: str,
        class_name: str,
        media_id: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ):
        super().__init__(
            event_type=EventType.issue_assigned,
            timestamp=timestamp or datetime.now(timezone.utc),
            data={
                "issue_id": issue_id,
                "assigned_to": assigned_to,
                "assigned_by": assigned_by,
                "severity": severity,
                "class_name": class_name,
                "media_id": media_id,
            }
        )


@dataclass
class IssueSeverityChangedEvent(Event):
    """Event fired when an issue's severity changes"""
    def __init__(
        self,
        issue_id: int,
        old_severity: str,
        new_severity: str,
        changed_by: str,
        class_name: str,
        media_id: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ):
        super().__init__(
            event_type=EventType.issue_severity_changed,
            timestamp=timestamp or datetime.now(timezone.utc),
            data={
                "issue_id": issue_id,
                "old_severity": old_severity,
                "new_severity": new_severity,
                "changed_by": changed_by,
                "class_name": class_name,
                "media_id": media_id,
            }
        )


@dataclass
class IssueVerifiedEvent(Event):
    """Event fired when an issue is verified"""
    def __init__(
        self,
        issue_id: int,
        verified_by: str,
        severity: str,
        class_name: str,
        media_id: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ):
        super().__init__(
            event_type=EventType.issue_verified,
            timestamp=timestamp or datetime.now(timezone.utc),
            data={
                "issue_id": issue_id,
                "verified_by": verified_by,
                "severity": severity,
                "class_name": class_name,
                "media_id": media_id,
            }
        )
