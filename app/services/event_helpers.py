from typing import Optional
from datetime import datetime, timezone
from app.services.events import (
    IssueCreatedEvent,
    IssueStatusChangedEvent,
    IssueAssignedEvent,
    IssueSeverityChangedEvent,
    IssueVerifiedEvent,
    Event
)


def create_issue_created_event(
    issue_id: int,
    class_name: str,
    severity: str,
    description: Optional[str] = None,
    created_by: Optional[str] = None
) -> IssueCreatedEvent:
    """Helper to create an IssueCreatedEvent"""
    return IssueCreatedEvent(
        issue_id=issue_id,
        class_name=class_name,
        severity=severity,
        description=description,
        created_by=created_by,
        timestamp=datetime.now(timezone.utc)
    )


def create_issue_status_changed_event(
    issue_id: int,
    old_status: str,
    new_status: str,
    changed_by: str,
    severity: str,
    class_name: str
) -> IssueStatusChangedEvent:
    """Helper to create an IssueStatusChangedEvent"""
    return IssueStatusChangedEvent(
        issue_id=issue_id,
        old_status=old_status,
        new_status=new_status,
        changed_by=changed_by,
        severity=severity,
        class_name=class_name,
        timestamp=datetime.now(timezone.utc)
    )


def create_issue_assigned_event(
    issue_id: int,
    assigned_to: str,
    assigned_by: str,
    severity: str,
    class_name: str
) -> IssueAssignedEvent:
    """Helper to create an IssueAssignedEvent"""
    return IssueAssignedEvent(
        issue_id=issue_id,
        assigned_to=assigned_to,
        assigned_by=assigned_by,
        severity=severity,
        class_name=class_name,
        timestamp=datetime.now(timezone.utc)
    )


def create_issue_severity_changed_event(
    issue_id: int,
    old_severity: str,
    new_severity: str,
    changed_by: str,
    class_name: str
) -> IssueSeverityChangedEvent:
    """Helper to create an IssueSeverityChangedEvent"""
    return IssueSeverityChangedEvent(
        issue_id=issue_id,
        old_severity=old_severity,
        new_severity=new_severity,
        changed_by=changed_by,
        class_name=class_name,
        timestamp=datetime.now(timezone.utc)
    )


def create_issue_verified_event(
    issue_id: int,
    verified_by: str,
    severity: str,
    class_name: str
) -> IssueVerifiedEvent:
    """Helper to create an IssueVerifiedEvent"""
    return IssueVerifiedEvent(
        issue_id=issue_id,
        verified_by=verified_by,
        severity=severity,
        class_name=class_name,
        timestamp=datetime.now(timezone.utc)
    )
