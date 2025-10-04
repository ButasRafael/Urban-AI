import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.notification import (
    NotificationChannel,
    EventType,
    NotificationLog,
    NotificationStatus
)
from app.models.user import User
from app.services.notification_preferences import NotificationPreferenceService
from app.services.notification_templates import NotificationTemplateService
from app.services.notification_channels import send_email, send_push_notification

logger = logging.getLogger(__name__)


class NotificationOrchestrator:
    """
    Orchestrates notification delivery based on events and user preferences
    """

    def __init__(self, db: Session):
        self.db = db
        self.pref_service = NotificationPreferenceService()
        self.template_service = NotificationTemplateService()

    def process_event(self, event_data: Dict[str, Any]) -> None:
        """
        Process an event and send appropriate notifications

        Args:
            event_data: Event data from Redis pub/sub
        """
        try:
            event_type_str = event_data.get("event_type")
            event_type = EventType(event_type_str)
            data = event_data.get("data", {})

            logger.info(f"Processing event: {event_type} with data: {data}")

            # Determine recipients based on event type
            recipients = self._get_recipients(event_type, data)

            # Process notifications for each recipient
            for username in recipients:
                self._process_user_notification(username, event_type, data)

        except Exception as e:
            logger.error(f"Error processing event: {e}", exc_info=True)

    def _get_recipients(self, event_type: EventType, data: Dict[str, Any]) -> List[str]:
        """
        Determine who should receive notifications for this event

        Args:
            event_type: Type of event
            data: Event data

        Returns:
            List of usernames to notify
        """
        recipients = []

        try:
            # Get the issue reporter for all events (except issue_created)
            issue_id = data.get("issue_id")
            reporter = None
            if issue_id and event_type != EventType.issue_created:
                reporter = self._get_issue_reporter(issue_id)

            if event_type == EventType.issue_assigned:
                # Notify the assignee and the reporter
                assigned_to = data.get("assigned_to")
                if assigned_to:
                    recipients.append(assigned_to)
                if reporter and reporter not in recipients:
                    recipients.append(reporter)

            elif event_type == EventType.issue_created:
                # Notify all admins and authorities (not the reporter since they just created it)
                users = self.db.query(User).filter(
                    User.role.in_(["admin", "authority"])
                ).all()
                recipients = [user.username for user in users]

            elif event_type == EventType.issue_status_changed:
                # Notify the assignee and the reporter
                if issue_id:
                    from app.models import media as dbm
                    detection = self.db.query(dbm.Detection).filter(
                        dbm.Detection.id == issue_id
                    ).first()
                    if detection and detection.assigned_to:
                        recipients.append(detection.assigned_to)

                if reporter and reporter not in recipients:
                    recipients.append(reporter)

            elif event_type == EventType.issue_severity_changed:
                # Notify the assignee/admins and the reporter
                if issue_id:
                    from app.models import media as dbm
                    detection = self.db.query(dbm.Detection).filter(
                        dbm.Detection.id == issue_id
                    ).first()
                    if detection and detection.assigned_to:
                        recipients.append(detection.assigned_to)
                    else:
                        # Notify admins
                        users = self.db.query(User).filter(User.role == "admin").all()
                        recipients.extend([user.username for user in users])

                if reporter and reporter not in recipients:
                    recipients.append(reporter)

            elif event_type == EventType.issue_verified:
                # Notify all admins and the reporter
                users = self.db.query(User).filter(User.role == "admin").all()
                recipients = [user.username for user in users]

                if reporter and reporter not in recipients:
                    recipients.append(reporter)

        except Exception as e:
            logger.error(f"Error determining recipients: {e}", exc_info=True)

        return recipients

    def _get_issue_reporter(self, issue_id: int) -> Optional[str]:
        """
        Get the username of the user who reported an issue

        Args:
            issue_id: Detection ID

        Returns:
            Username of the reporter, or None if not found
        """
        try:
            from app.models import media as dbm
            detection = self.db.query(dbm.Detection).filter(
                dbm.Detection.id == issue_id
            ).first()

            if detection and detection.frame and detection.frame.media:
                return detection.frame.media.user_username

        except Exception as e:
            logger.error(f"Error getting issue reporter: {e}", exc_info=True)

        return None

    def _process_user_notification(
        self,
        username: str,
        event_type: EventType,
        data: Dict[str, Any]
    ) -> None:
        """
        Process notification for a specific user

        Args:
            username: Username to notify
            event_type: Type of event
            data: Event data
        """
        try:
            # Get user preferences
            prefs = self.pref_service.get_or_create_preferences(self.db, username)

            # Get severity from event data (default to "medium" if not present)
            severity = data.get("severity", "medium")

            # Check if we're in quiet hours
            if self.pref_service.is_quiet_hours(prefs):
                logger.info(f"Skipping notification for {username} - in quiet hours")
                return

            # Determine which channels to use
            channels_to_use = []

            if self.pref_service.should_notify(prefs, event_type, NotificationChannel.email, severity):
                channels_to_use.append(NotificationChannel.email)

            if self.pref_service.should_notify(prefs, event_type, NotificationChannel.push, severity):
                channels_to_use.append(NotificationChannel.push)

            # Send notifications for each channel
            for channel in channels_to_use:
                self._send_notification(username, event_type, channel, data, prefs)

        except Exception as e:
            logger.error(f"Error processing notification for user {username}: {e}", exc_info=True)

    def _send_notification(
        self,
        username: str,
        event_type: EventType,
        channel: NotificationChannel,
        data: Dict[str, Any],
        prefs
    ) -> None:
        """
        Send a notification via a specific channel

        Args:
            username: Username to notify
            event_type: Type of event
            channel: Notification channel
            data: Event data
            prefs: User notification preferences
        """
        try:
            # Get the template
            template = self.template_service.get_template(self.db, event_type, channel)
            if not template:
                logger.warning(f"No template found for {event_type} on {channel}")
                return

            # Render the template
            rendered = self.template_service.render_template(template, data)

            # Send via appropriate channel
            success = False
            error_message = None

            if channel == NotificationChannel.email:
                # Get user email (assuming username is email or we need to fetch it)
                user = self.db.query(User).filter(User.username == username).first()
                if user:
                    # Create stable idempotency key to prevent duplicate sends on retry
                    issue_id = data.get("issue_id", "unknown")
                    idem_key = f"{issue_id}:{username}:email"

                    # Assuming username is the email address
                    # If not, you'll need to add an email field to the User model
                    success = send_email(
                        username,  # Using username as email
                        rendered["subject"],
                        rendered["body"],
                        idem_key=idem_key
                    )
                    if not success:
                        error_message = "Failed to send email"

            elif channel == NotificationChannel.push:
                # Get FCM token from preferences
                if prefs.fcm_token:
                    success = send_push_notification(
                        prefs.fcm_token,
                        f"Urban AI - {event_type.value.replace('_', ' ').title()}",
                        rendered["body"],
                        {"issue_id": str(data.get("issue_id", ""))}
                    )
                    if not success:
                        error_message = "Failed to send push notification"
                else:
                    logger.info(f"No FCM token for user {username}, skipping push notification")
                    return

            # Log the notification
            log_entry = NotificationLog(
                username=username,
                event_type=event_type,
                channel=channel,
                status=NotificationStatus.sent if success else NotificationStatus.failed,
                event_data=data,
                sent_at=datetime.now(timezone.utc) if success else None,
                error_message=error_message
            )
            self.db.add(log_entry)
            self.db.commit()

            if success:
                logger.info(f"Notification sent to {username} via {channel.value}")
            else:
                logger.error(f"Failed to send notification to {username} via {channel.value}: {error_message}")

        except Exception as e:
            logger.error(f"Error sending notification: {e}", exc_info=True)
            # Log the failed notification
            try:
                log_entry = NotificationLog(
                    username=username,
                    event_type=event_type,
                    channel=channel,
                    status=NotificationStatus.failed,
                    event_data=data,
                    error_message=str(e)
                )
                self.db.add(log_entry)
                self.db.commit()
            except Exception as log_error:
                logger.error(f"Failed to log notification error: {log_error}")


def handle_notification_event(event_json: str) -> None:
    """
    Handle a notification event from Redis pub/sub

    Args:
        event_json: JSON string of the event
    """
    db = SessionLocal()
    try:
        event_data = json.loads(event_json)
        orchestrator = NotificationOrchestrator(db)
        orchestrator.process_event(event_data)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse event JSON: {e}")
    except Exception as e:
        logger.error(f"Error handling notification event: {e}", exc_info=True)
    finally:
        db.close()
