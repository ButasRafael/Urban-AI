import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from jinja2 import Template
from app.models.notification import NotificationTemplate, EventType, NotificationChannel

logger = logging.getLogger(__name__)


class NotificationTemplateService:
    """
    Service for managing and rendering notification templates
    """

    @staticmethod
    def get_template(
        db: Session,
        event_type: EventType,
        channel: NotificationChannel,
        language: str = "en"
    ) -> Optional[NotificationTemplate]:
        """
        Get an active template for a given event type and channel

        Args:
            db: Database session
            event_type: Type of event
            channel: Notification channel
            language: Template language (default: "en")

        Returns:
            NotificationTemplate if found, None otherwise
        """
        template = db.query(NotificationTemplate).filter(
            NotificationTemplate.event_type == event_type,
            NotificationTemplate.channel == channel,
            NotificationTemplate.language == language,
            NotificationTemplate.is_active == True
        ).order_by(NotificationTemplate.version.desc()).first()

        return template

    @staticmethod
    def render_template(template: NotificationTemplate, context: Dict[str, Any]) -> Dict[str, str]:
        """
        Render a template with the given context

        Args:
            template: The notification template
            context: Dictionary of variables to render

        Returns:
            Dictionary with 'subject' and 'body' keys (subject may be None for push notifications)
        """
        try:
            # Render subject (if exists, for email)
            subject = None
            if template.subject:
                subject_template = Template(template.subject)
                subject = subject_template.render(**context)

            # Render body
            body_template = Template(template.body)
            body = body_template.render(**context)

            return {
                "subject": subject,
                "body": body
            }

        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return {
                "subject": "Notification",
                "body": "An error occurred rendering this notification."
            }

    @staticmethod
    def create_default_templates(db: Session) -> None:
        """
        Create default notification templates if they don't exist

        Args:
            db: Database session
        """
        default_templates = [
            # Email templates
            {
                "event_type": EventType.issue_created,
                "channel": NotificationChannel.email,
                "subject": "New Issue Created: {{ class_name }}",
                "body": """
Hello,

A new issue has been created in the Urban AI system.

Issue ID: {{ issue_id }}
Type: {{ class_name }}
Severity: {{ severity }}
Description: {{ description }}

Please review this issue in the dashboard.

Best regards,
Urban AI Team
                """.strip(),
                "variables": ["issue_id", "class_name", "severity", "description"]
            },
            {
                "event_type": EventType.issue_status_changed,
                "channel": NotificationChannel.email,
                "subject": "Issue Status Updated: {{ class_name }}",
                "body": """
Hello,

Issue #{{ issue_id }} status has been updated.

Type: {{ class_name }}
Old Status: {{ old_status }}
New Status: {{ new_status }}
Severity: {{ severity }}
Changed by: {{ changed_by }}

View the issue in the dashboard for more details.

Best regards,
Urban AI Team
                """.strip(),
                "variables": ["issue_id", "class_name", "old_status", "new_status", "severity", "changed_by"]
            },
            {
                "event_type": EventType.issue_assigned,
                "channel": NotificationChannel.email,
                "subject": "Issue Assigned to You: {{ class_name }}",
                "body": """
Hello {{ assigned_to }},

You have been assigned to issue #{{ issue_id }}.

Type: {{ class_name }}
Severity: {{ severity }}
Assigned by: {{ assigned_by }}

Please review and address this issue at your earliest convenience.

Best regards,
Urban AI Team
                """.strip(),
                "variables": ["issue_id", "class_name", "severity", "assigned_to", "assigned_by"]
            },
            {
                "event_type": EventType.issue_severity_changed,
                "channel": NotificationChannel.email,
                "subject": "Issue Severity Changed: {{ class_name }}",
                "body": """
Hello,

Issue #{{ issue_id }} severity has been updated.

Type: {{ class_name }}
Old Severity: {{ old_severity }}
New Severity: {{ new_severity }}
Changed by: {{ changed_by }}

View the issue in the dashboard for more details.

Best regards,
Urban AI Team
                """.strip(),
                "variables": ["issue_id", "class_name", "old_severity", "new_severity", "changed_by"]
            },
            {
                "event_type": EventType.issue_verified,
                "channel": NotificationChannel.email,
                "subject": "Issue Verified: {{ class_name }}",
                "body": """
Hello,

Issue #{{ issue_id }} has been verified.

Type: {{ class_name }}
Severity: {{ severity }}
Verified by: {{ verified_by }}

This issue is now ready for assignment.

Best regards,
Urban AI Team
                """.strip(),
                "variables": ["issue_id", "class_name", "severity", "verified_by"]
            },

            # Push notification templates
            {
                "event_type": EventType.issue_created,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "New {{ severity }} severity issue: {{ class_name }} (#{{ issue_id }})",
                "variables": ["issue_id", "class_name", "severity"]
            },
            {
                "event_type": EventType.issue_status_changed,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "Issue #{{ issue_id }} status changed: {{ old_status }} → {{ new_status }}",
                "variables": ["issue_id", "old_status", "new_status"]
            },
            {
                "event_type": EventType.issue_assigned,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "You've been assigned issue #{{ issue_id }}: {{ class_name }} ({{ severity }})",
                "variables": ["issue_id", "class_name", "severity"]
            },
            {
                "event_type": EventType.issue_severity_changed,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "Issue #{{ issue_id }} severity changed: {{ old_severity }} → {{ new_severity }}",
                "variables": ["issue_id", "old_severity", "new_severity"]
            },
            {
                "event_type": EventType.issue_verified,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "Issue #{{ issue_id }} verified by {{ verified_by }}",
                "variables": ["issue_id", "verified_by"]
            },
        ]

        for template_data in default_templates:
            # Check if template already exists
            existing = db.query(NotificationTemplate).filter(
                NotificationTemplate.event_type == template_data["event_type"],
                NotificationTemplate.channel == template_data["channel"],
                NotificationTemplate.language == "en"
            ).first()

            if not existing:
                template = NotificationTemplate(
                    event_type=template_data["event_type"],
                    channel=template_data["channel"],
                    subject=template_data["subject"],
                    body=template_data["body"],
                    variables=template_data["variables"],
                    language="en",
                    version=1,
                    is_active=True
                )
                db.add(template)

        db.commit()
        logger.info("Default notification templates created")
