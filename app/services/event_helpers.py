from typing import Optional
from datetime import datetime, timezone
from app.services.events import (
    MediaCreatedEvent,
    IssueStatusChangedEvent,
    IssueAssignedEvent,
    IssueSeverityChangedEvent,
    IssueVerifiedEvent,
    Event
)


def create_media_created_event(
    media_id: int,
    num_detections: int,
    created_by: str,
    media_type: str,
    max_severity: str = "medium"
) -> MediaCreatedEvent:
    """Helper to create a MediaCreatedEvent"""
    return MediaCreatedEvent(
        media_id=media_id,
        num_detections=num_detections,
        created_by=created_by,
        media_type=media_type,
        max_severity=max_severity,
        timestamp=datetime.now(timezone.utc)
    )


def create_issue_status_changed_event(
    issue_id: int,
    old_status: str,
    new_status: str,
    changed_by: str,
    severity: str,
    class_name: str,
    media_id: Optional[int] = None
) -> IssueStatusChangedEvent:
    """Helper to create an IssueStatusChangedEvent"""
    return IssueStatusChangedEvent(
        issue_id=issue_id,
        old_status=old_status,
        new_status=new_status,
        changed_by=changed_by,
        severity=severity,
        class_name=class_name,
        media_id=media_id,
        timestamp=datetime.now(timezone.utc)
    )


def create_issue_assigned_event(
    issue_id: int,
    assigned_to: str,
    assigned_by: str,
    severity: str,
    class_name: str,
    media_id: Optional[int] = None
) -> IssueAssignedEvent:
    """Helper to create an IssueAssignedEvent"""
    return IssueAssignedEvent(
        issue_id=issue_id,
        assigned_to=assigned_to,
        assigned_by=assigned_by,
        severity=severity,
        class_name=class_name,
        media_id=media_id,
        timestamp=datetime.now(timezone.utc)
    )


def create_issue_severity_changed_event(
    issue_id: int,
    old_severity: str,
    new_severity: str,
    changed_by: str,
    class_name: str,
    media_id: Optional[int] = None
) -> IssueSeverityChangedEvent:
    """Helper to create an IssueSeverityChangedEvent"""
    return IssueSeverityChangedEvent(
        issue_id=issue_id,
        old_severity=old_severity,
        new_severity=new_severity,
        changed_by=changed_by,
        class_name=class_name,
        media_id=media_id,
        timestamp=datetime.now(timezone.utc)
    )


def create_issue_verified_event(
    issue_id: int,
    verified_by: str,
    severity: str,
    class_name: str,
    media_id: Optional[int] = None
) -> IssueVerifiedEvent:
    """Helper to create an IssueVerifiedEvent"""
    return IssueVerifiedEvent(
        issue_id=issue_id,
        verified_by=verified_by,
        severity=severity,
        class_name=class_name,
        media_id=media_id,
        timestamp=datetime.now(timezone.utc)
    )
