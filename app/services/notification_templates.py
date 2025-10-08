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
        user_role: Optional[str] = None,
        language: str = "en"
    ) -> Optional[NotificationTemplate]:
        """
        Get an active template for a given event type and channel

        Args:
            db: Database session
            event_type: Type of event
            channel: Notification channel
            user_role: User role (user, admin, authority) - will try role-specific template first, then fallback to general
            language: Template language (default: "en")

        Returns:
            NotificationTemplate if found, None otherwise
        """
        # Try to get role-specific template first
        if user_role:
            template = db.query(NotificationTemplate).filter(
                NotificationTemplate.event_type == event_type,
                NotificationTemplate.channel == channel,
                NotificationTemplate.user_role == user_role,
                NotificationTemplate.language == language,
                NotificationTemplate.is_active == True
            ).order_by(NotificationTemplate.version.desc()).first()

            if template:
                return template

        # Fallback to general template (user_role is NULL)
        template = db.query(NotificationTemplate).filter(
            NotificationTemplate.event_type == event_type,
            NotificationTemplate.channel == channel,
            NotificationTemplate.user_role == None,
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
            # Email templates for admins/authorities
            {
                "event_type": EventType.media_created,
                "channel": NotificationChannel.email,
                "subject": "New Issues Detected - Urban AI",
                "body": """
Hello,

New media has been processed and {{ num_detections }} issue(s) detected.

Media ID: {{ media_id }}
Media Type: {{ media_type }}
Number of Detections: {{ num_detections }}
Uploaded by: {{ created_by }}

View the issues at: {{ web_app_url }}/issues?media_id={{ media_id }}

Best regards,
Urban AI Team
                """.strip(),
                "variables": ["media_id", "num_detections", "created_by", "media_type"],
                "user_role": None
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
                "variables": ["issue_id", "class_name", "old_status", "new_status", "severity", "changed_by"],
                "user_role": None
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
                "variables": ["issue_id", "class_name", "severity", "assigned_to", "assigned_by"],
                "user_role": None
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
                "variables": ["issue_id", "class_name", "old_severity", "new_severity", "changed_by"],
                "user_role": None
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
                "variables": ["issue_id", "class_name", "severity", "verified_by"],
                "user_role": None
            },

            # Push notification templates (general)
            {
                "event_type": EventType.media_created,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "New media created with {{ num_detections }} detection(s)",
                "variables": ["media_id", "num_detections", "created_by", "media_type"],
                "user_role": None
            },
            {
                "event_type": EventType.issue_status_changed,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "Issue #{{ issue_id }} status changed: {{ old_status }} → {{ new_status }}",
                "variables": ["issue_id", "old_status", "new_status"],
                "user_role": None
            },
            {
                "event_type": EventType.issue_assigned,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "You've been assigned issue #{{ issue_id }}: {{ class_name }} ({{ severity }})",
                "variables": ["issue_id", "class_name", "severity"],
                "user_role": None
            },
            {
                "event_type": EventType.issue_severity_changed,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "Issue #{{ issue_id }} severity changed: {{ old_severity }} → {{ new_severity }}",
                "variables": ["issue_id", "old_severity", "new_severity"],
                "user_role": None
            },
            {
                "event_type": EventType.issue_verified,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "Issue #{{ issue_id }} verified by {{ verified_by }}",
                "variables": ["issue_id", "verified_by"],
                "user_role": None
            },

            # USER-FRIENDLY EMAIL TEMPLATES FOR REGULAR USERS
            {
                "event_type": EventType.issue_status_changed,
                "channel": NotificationChannel.email,
                "subject": "Update on Your Report: {{ class_name }}",
                "body": """
Hi there!

Good news! We have an update on the {{ class_name }} issue you reported.

Your report has been {{ new_status }}.

You can view the details and track progress in your dashboard.

Thank you for helping keep our city clean and safe!

Best regards,
Urban AI Team
                """.strip(),
                "variables": ["class_name", "new_status"],
                "user_role": "user"
            },
            {
                "event_type": EventType.issue_assigned,
                "channel": NotificationChannel.email,
                "subject": "Your Report is Being Addressed!",
                "body": """
Great news!

Your report about {{ class_name }} has been assigned to our team and is being actively worked on.

We'll keep you updated on the progress!

Thank you for being an engaged citizen and helping improve our community.

Best regards,
Urban AI Team
                """.strip(),
                "variables": ["class_name"],
                "user_role": "user"
            },
            {
                "event_type": EventType.issue_severity_changed,
                "channel": NotificationChannel.email,
                "subject": "Update on Your {{ class_name }} Report",
                "body": """
Hello!

We've reviewed your report about {{ class_name }} and updated its priority level.

This helps us address issues more effectively and ensure the most urgent matters get the attention they need.

Thank you for your report!

Best regards,
Urban AI Team
                """.strip(),
                "variables": ["class_name"],
                "user_role": "user"
            },
            {
                "event_type": EventType.issue_verified,
                "channel": NotificationChannel.email,
                "subject": "Thank You! Your Report Has Been Verified",
                "body": """
Congratulations!

Your report about {{ class_name }} has been verified by our team. This is an important step in getting the issue addressed.

Thank you for taking the time to report this issue. Your contribution makes a real difference in keeping our community safe and well-maintained!

We'll notify you when work begins on resolving it.

Best regards,
Urban AI Team
                """.strip(),
                "variables": ["class_name"],
                "user_role": "user"
            },

            # USER-FRIENDLY PUSH TEMPLATES FOR REGULAR USERS
            {
                "event_type": EventType.issue_status_changed,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "Update: Your {{ class_name }} report is now {{ new_status }}!",
                "variables": ["class_name", "new_status"],
                "user_role": "user"
            },
            {
                "event_type": EventType.issue_assigned,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "Good news! Your {{ class_name }} report is being worked on",
                "variables": ["class_name"],
                "user_role": "user"
            },
            {
                "event_type": EventType.issue_severity_changed,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "Your {{ class_name }} report priority has been updated",
                "variables": ["class_name"],
                "user_role": "user"
            },
            {
                "event_type": EventType.issue_verified,
                "channel": NotificationChannel.push,
                "subject": None,
                "body": "Thank you! Your {{ class_name }} report has been verified ✓",
                "variables": ["class_name"],
                "user_role": "user"
            },
        ]

        for template_data in default_templates:
            # Check if template already exists (check user_role too)
            existing = db.query(NotificationTemplate).filter(
                NotificationTemplate.event_type == template_data["event_type"],
                NotificationTemplate.channel == template_data["channel"],
                NotificationTemplate.user_role == template_data.get("user_role"),
                NotificationTemplate.language == "en"
            ).first()

            if not existing:
                template = NotificationTemplate(
                    event_type=template_data["event_type"],
                    channel=template_data["channel"],
                    subject=template_data["subject"],
                    body=template_data["body"],
                    variables=template_data["variables"],
                    user_role=template_data.get("user_role"),
                    language="en",
                    version=1,
                    is_active=True
                )
                db.add(template)

        db.commit()
        logger.info("Default notification templates created")
