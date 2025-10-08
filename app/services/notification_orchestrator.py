import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.core.redis_pool import get_redis
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
            # Get the issue reporter for relevant events
            issue_id = data.get("issue_id")
            reporter = None
            if issue_id and event_type != EventType.media_created:
                reporter = self._get_issue_reporter(issue_id)

            # Get the person making the change (to exclude them from notifications)
            changed_by = data.get("changed_by") or data.get("assigned_by") or data.get("verified_by")

            if event_type == EventType.issue_assigned:
                # Notify the assignee and the reporter, but exclude the assigner
                assigned_to = data.get("assigned_to")
                assigned_by = data.get("assigned_by")

                if assigned_to and assigned_to != assigned_by:
                    recipients.append(assigned_to)
                if reporter and reporter not in recipients and reporter != assigned_by:
                    recipients.append(reporter)

            elif event_type == EventType.media_created:
                # Notify only admins when new media is created with detections
                users = self.db.query(User).filter(
                    User.role == "admin"
                ).all()
                recipients = [user.username for user in users]

            elif event_type == EventType.issue_status_changed:
                # Notify uploader + (assignee OR verifying admin) - whoever didn't make the change
                # Always add reporter (uploader) unless they made the change
                if reporter and reporter != changed_by:
                    recipients.append(reporter)

                # Get detection to find assignee and verified_by
                if issue_id:
                    from app.models import media as dbm
                    detection = self.db.query(dbm.Detection).filter(
                        dbm.Detection.id == issue_id
                    ).first()

                    if detection:
                        # Add assignee if exists and didn't make the change
                        if detection.assigned_to and detection.assigned_to != changed_by:
                            if detection.assigned_to not in recipients:
                                recipients.append(detection.assigned_to)

                        # If no assignee or assignee made the change, notify the verifying admin
                        elif detection.verified_by and detection.verified_by != changed_by:
                            if detection.verified_by not in recipients:
                                recipients.append(detection.verified_by)

            elif event_type == EventType.issue_severity_changed:
                # Notify uploader + (assignee OR verifying admin) - whoever didn't make the change
                # Always add reporter (uploader) unless they made the change
                if reporter and reporter != changed_by:
                    recipients.append(reporter)

                # Get detection to find assignee and verified_by
                if issue_id:
                    from app.models import media as dbm
                    detection = self.db.query(dbm.Detection).filter(
                        dbm.Detection.id == issue_id
                    ).first()

                    if detection:
                        # Add assignee if exists and didn't make the change
                        if detection.assigned_to and detection.assigned_to != changed_by:
                            if detection.assigned_to not in recipients:
                                recipients.append(detection.assigned_to)

                        # If no assignee or assignee made the change, notify the verifying admin
                        elif detection.verified_by and detection.verified_by != changed_by:
                            if detection.verified_by not in recipients:
                                recipients.append(detection.verified_by)

            elif event_type == EventType.issue_verified:
                # Notify only the uploader (reporter)
                if reporter:
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
            # Add web_app_url to context for email templates
            data = {**data, "web_app_url": os.getenv("WEB_APP_URL", "http://localhost:5173")}
            # Get user preferences
            prefs = self.pref_service.get_or_create_preferences(self.db, username)

            # Get user for role-based template selection (query once, use everywhere)
            user = self.db.query(User).filter(User.username == username).first()
            if not user:
                logger.warning(f"User {username} not found, skipping notification")
                return

            # Get severity from event data (default to "medium" if not present)
            severity = data.get("severity", "medium")

            # Check if we're in quiet hours
            if self.pref_service.is_quiet_hours(prefs):
                # Still create a pending log entry for the inbox so users can read it
                # Also schedule for retry after quiet hours end

                # Determine preferred channel for inbox display
                email_enabled = self.pref_service.should_notify(prefs, event_type, NotificationChannel.email, severity)
                push_enabled = self.pref_service.should_notify(prefs, event_type, NotificationChannel.push, severity)
                preferred_channel = NotificationChannel.email if email_enabled else NotificationChannel.push

                # Get template for inbox body (prefer email for richer content)
                template = self.template_service.get_template(
                    self.db, event_type, preferred_channel, user_role=user.role
                ) or self.template_service.get_template(
                    self.db, event_type, preferred_channel, user_role=None
                )

                if template:
                    rendered = self.template_service.render_template(template, data)
                    inbox_body = rendered["body"]

                    # Create pending log entry for inbox viewing and later retry
                    log_entry = NotificationLog(
                        username=username,
                        event_type=event_type,
                        channel=preferred_channel,
                        status=NotificationStatus.pending,
                        event_data=data,
                        created_at=datetime.now(timezone.utc),
                        next_retry_at=datetime.now(timezone.utc) + timedelta(hours=1),
                    )
                    self.db.add(log_entry)
                    self.db.commit()
                    self.db.refresh(log_entry)

                    # Send WebSocket notification so it appears in inbox immediately
                    self._send_websocket_notification(username, event_type, inbox_body, log_entry.id)

                logger.info(f"Queued pending notification for {username} during quiet hours (retry scheduled)")
                return

            # User already queried at the start - use it for normal delivery
            user_role = user.role

            # Determine which channels to use for delivery
            email_enabled = self.pref_service.should_notify(
                prefs, event_type, NotificationChannel.email, severity
            ) and bool(user.email and user.email_verified)
            push_enabled = self.pref_service.should_notify(
                prefs, event_type, NotificationChannel.push, severity
            ) and bool(prefs.fcm_token)

            # Render per channel (role-aware), independently
            rendered_email = None
            rendered_push = None

            if email_enabled:
                tpl_email = self.template_service.get_template(
                    self.db, event_type, NotificationChannel.email, user_role=user_role
                ) or self.template_service.get_template(
                    self.db, event_type, NotificationChannel.email, user_role=None
                )
                if tpl_email:
                    rendered_email = self.template_service.render_template(tpl_email, data)

            if push_enabled:
                tpl_push = self.template_service.get_template(
                    self.db, event_type, NotificationChannel.push, user_role=user_role
                ) or self.template_service.get_template(
                    self.db, event_type, NotificationChannel.push, user_role=None
                )
                if tpl_push:
                    rendered_push = self.template_service.render_template(tpl_push, data)

            # If neither channel has a template, skip
            if not (rendered_email or rendered_push):
                logger.warning(f"No template found for {event_type} (role={user_role}); skipping {username}")
                return

            # Choose the inbox/store copy (prefer email body for richer text)
            inbox_body = (rendered_email or rendered_push)["body"]

            # Create ONE inbox/store log (represent the richer copy if available)
            preferred_channel_for_store = NotificationChannel.email if rendered_email else NotificationChannel.push
            log_entry = NotificationLog(
                username=username,
                event_type=event_type,
                channel=preferred_channel_for_store,
                status=NotificationStatus.pending,
                event_data=data,
                created_at=datetime.now(timezone.utc),
            )
            self.db.add(log_entry)
            self.db.commit()
            self.db.refresh(log_entry)

            # Try to deliver via enabled channels (don't create additional log rows)
            email_sent = False
            push_sent = False

            if email_enabled and rendered_email:
                issue_id = data.get("issue_id", "unknown")
                idem_key = f"{issue_id}:{username}:email"
                email_sent = send_email(
                    user.email,
                    rendered_email.get("subject") or "Urban AI Notification",
                    rendered_email.get("body") or inbox_body,
                    idem_key=idem_key,
                )
                if not email_sent:
                    logger.warning(f"Failed to send email to {username}")

            if push_enabled and rendered_push:
                # Build notification data based on event type
                web_app_url = os.getenv("WEB_APP_URL", "http://localhost:5173")
                if event_type == EventType.media_created:
                    media_id = str(data.get("media_id", ""))
                    notification_data = {
                        "media_id": media_id,
                        "url": f"{web_app_url}/issues?media_id={media_id}" if media_id else f"{web_app_url}/issues",
                        "tag": f"media:{media_id}" if media_id else "notification"
                    }
                else:
                    issue_id = str(data.get("issue_id", ""))
                    media_id = str(data.get("media_id", ""))
                    notification_data = {
                        "issue_id": issue_id,
                        "url": f"{web_app_url}/issues?media_id={media_id}&assignee=me" if media_id else f"{web_app_url}/issues?assignee=me",
                        "tag": f"issue:{issue_id}" if issue_id else "notification"
                    }

                push_sent, should_clear_token = send_push_notification(
                    prefs.fcm_token,
                    f"Urban AI - {event_type.value.replace('_', ' ').title()}",
                    rendered_push.get("body") or inbox_body,
                    notification_data
                )

                if should_clear_token:
                    logger.info(f"Clearing invalid FCM token for user {username}")
                    # Use raw SQL to avoid committing other pending changes in session
                    self.db.execute(
                        "UPDATE notification_preferences SET fcm_token = NULL WHERE username = :username",
                        {"username": username}
                    )
                    self.db.commit()

                if not push_sent:
                    logger.warning(f"Failed to send push notification to {username}")

            # Update log entry status based on delivery results
            any_delivered = email_sent or push_sent
            log_entry.status = NotificationStatus.sent if any_delivered else NotificationStatus.failed
            log_entry.sent_at = datetime.now(timezone.utc) if any_delivered else None

            # Track which channels successfully delivered
            delivered = []
            if email_sent:
                delivered.append("email")
            if push_sent:
                delivered.append("push")
            log_entry.delivered_channels = delivered

            if not any_delivered:
                log_entry.error_message = "All delivery channels failed"
            self.db.commit()

            # ALWAYS send WebSocket notification for real-time UI updates (even if delivery failed)
            # This ensures the user sees the notification in their inbox
            self._send_websocket_notification(username, event_type, inbox_body, log_entry.id)

            logger.info(f"Processed notification for {username}: event={event_type}, inbox_id={log_entry.id}, email={email_sent}, push={push_sent}")

        except Exception as e:
            logger.error(f"Error processing notification for user {username}: {e}", exc_info=True)

    def _attempt_delivery(
        self,
        user: User,
        prefs: "NotificationPreference",
        event_type: EventType,
        data: Dict[str, Any],
        log_entry: NotificationLog
    ) -> tuple[bool, bool]:
        """
        Attempt to deliver a notification via email and/or push channels

        Args:
            user: User object
            prefs: User notification preferences
            event_type: Type of event
            data: Event data
            log_entry: Existing log entry to track delivery (not used for new log creation)

        Returns:
            Tuple of (email_sent, push_sent)
        """
        # Add web_app_url to context for email templates
        data = {**data, "web_app_url": os.getenv("WEB_APP_URL", "http://localhost:5173")}
        severity = data.get("severity", "medium")
        user_role = user.role

        # Determine which channels to use for delivery
        email_enabled = self.pref_service.should_notify(
            prefs, event_type, NotificationChannel.email, severity
        ) and bool(user.email and user.email_verified)
        push_enabled = self.pref_service.should_notify(
            prefs, event_type, NotificationChannel.push, severity
        ) and bool(prefs.fcm_token)

        # Render per channel (role-aware), independently
        rendered_email = None
        rendered_push = None

        if email_enabled:
            tpl_email = self.template_service.get_template(
                self.db, event_type, NotificationChannel.email, user_role=user_role
            ) or self.template_service.get_template(
                self.db, event_type, NotificationChannel.email, user_role=None
            )
            if tpl_email:
                rendered_email = self.template_service.render_template(tpl_email, data)

        if push_enabled:
            tpl_push = self.template_service.get_template(
                self.db, event_type, NotificationChannel.push, user_role=user_role
            ) or self.template_service.get_template(
                self.db, event_type, NotificationChannel.push, user_role=None
            )
            if tpl_push:
                rendered_push = self.template_service.render_template(tpl_push, data)

        # If neither channel has a template, skip
        if not (rendered_email or rendered_push):
            logger.warning(f"No template found for {event_type} (role={user_role}); skipping {user.username}")
            return False, False

        # Try to deliver via enabled channels
        email_sent = False
        push_sent = False
        inbox_body = (rendered_email or rendered_push)["body"]

        if email_enabled and rendered_email:
            issue_id = data.get("issue_id", "unknown")
            idem_key = f"{issue_id}:{user.username}:email:{event_type.value}"
            email_sent = send_email(
                user.email,
                rendered_email.get("subject") or "Urban AI Notification",
                rendered_email.get("body") or inbox_body,
                idem_key=idem_key,
            )
            if not email_sent:
                logger.warning(f"Failed to send email to {user.username}")

        if push_enabled and rendered_push:
            # Build notification data based on event type
            web_app_url = os.getenv("WEB_APP_URL", "http://localhost:5173")
            if event_type == EventType.media_created:
                media_id = str(data.get("media_id", ""))
                notification_data = {
                    "media_id": media_id,
                    "url": f"{web_app_url}/issues?media_id={media_id}" if media_id else f"{web_app_url}/issues",
                    "tag": f"media:{media_id}" if media_id else "notification"
                }
            else:
                issue_id = str(data.get("issue_id", ""))
                media_id = str(data.get("media_id", ""))
                notification_data = {
                    "issue_id": issue_id,
                    "url": f"{web_app_url}/issues?media_id={media_id}&assignee=me" if media_id else f"{web_app_url}/issues?assignee=me",
                    "tag": f"issue:{issue_id}" if issue_id else "notification"
                }

            push_sent, should_clear_token = send_push_notification(
                prefs.fcm_token,
                f"Urban AI - {event_type.value.replace('_', ' ').title()}",
                rendered_push.get("body") or inbox_body,
                notification_data
            )

            if should_clear_token:
                logger.info(f"Clearing invalid FCM token for user {user.username}")
                # Use a separate update to avoid side effects
                self.db.execute(
                    "UPDATE notification_preferences SET fcm_token = NULL WHERE username = :username",
                    {"username": user.username}
                )
                self.db.commit()

            if not push_sent:
                logger.warning(f"Failed to send push notification to {user.username}")

        return email_sent, push_sent

    def retry_pending_notifications(self) -> int:
        """
        Retry pending notifications that are past their retry time and not in quiet hours

        Returns:
            Number of notifications retried
        """
        try:
            now = datetime.now(timezone.utc)

            # Find pending notifications that are ready for retry
            pending_notifications = self.db.query(NotificationLog).filter(
                NotificationLog.status == NotificationStatus.pending,
                NotificationLog.next_retry_at <= now,
                NotificationLog.retry_count < 10  # Max retry attempts
            ).all()

            retried_count = 0
            for log_entry in pending_notifications:
                try:
                    # Get user preferences to check if still in quiet hours
                    prefs = self.pref_service.get_or_create_preferences(self.db, log_entry.username)

                    # If still in quiet hours, schedule next retry
                    if self.pref_service.is_quiet_hours(prefs):
                        log_entry.next_retry_at = now + timedelta(minutes=30)
                        log_entry.retry_count += 1
                        self.db.commit()
                        logger.info(f"User {log_entry.username} still in quiet hours, rescheduling retry")
                        continue

                    # Reuse the same log entry (preserve read_at and retry_count)
                    username = log_entry.username
                    event_type = log_entry.event_type
                    event_data = log_entry.event_data

                    # Get user for role-based template selection
                    user = self.db.query(User).filter(User.username == username).first()
                    if not user:
                        logger.warning(f"User {username} not found, skipping notification retry")
                        log_entry.status = NotificationStatus.failed
                        log_entry.error_message = "User not found"
                        self.db.commit()
                        continue

                    # Try delivery via available channels
                    email_sent, push_sent = self._attempt_delivery(
                        user, prefs, event_type, event_data, log_entry
                    )

                    # Update log entry status based on delivery results
                    any_delivered = email_sent or push_sent
                    log_entry.status = NotificationStatus.sent if any_delivered else NotificationStatus.failed
                    log_entry.sent_at = now if any_delivered else None
                    log_entry.retry_count += 1

                    # Track which channels successfully delivered
                    delivered = []
                    if email_sent:
                        delivered.append("email")
                    if push_sent:
                        delivered.append("push")
                    log_entry.delivered_channels = delivered

                    if not any_delivered:
                        log_entry.error_message = "All delivery channels failed"
                        log_entry.next_retry_at = now + timedelta(hours=1)
                    self.db.commit()

                    # Send WebSocket notification for UI update
                    severity = event_data.get("severity", "medium")
                    email_enabled = self.pref_service.should_notify(prefs, event_type, NotificationChannel.email, severity)
                    preferred_channel = NotificationChannel.email if email_enabled else NotificationChannel.push

                    template = self.template_service.get_template(
                        self.db, event_type, preferred_channel, user_role=user.role
                    ) or self.template_service.get_template(
                        self.db, event_type, preferred_channel, user_role=None
                    )

                    if template:
                        rendered = self.template_service.render_template(template, event_data)
                        self._send_websocket_notification(username, event_type, rendered["body"], log_entry.id)

                    retried_count += 1
                    logger.info(f"Retried notification {log_entry.id} for {username}: email={email_sent}, push={push_sent}")

                except Exception as e:
                    logger.error(f"Error retrying notification {log_entry.id}: {e}", exc_info=True)
                    # Update the same log entry with error info
                    try:
                        self.db.refresh(log_entry)  # Ensure it's still in session
                        log_entry.retry_count += 1
                        log_entry.next_retry_at = now + timedelta(hours=1)
                        log_entry.error_message = str(e)
                        self.db.commit()
                    except Exception as update_error:
                        logger.error(f"Failed to update notification {log_entry.id} after error: {update_error}")

            logger.info(f"Retried {retried_count} pending notifications")
            return retried_count

        except Exception as e:
            logger.error(f"Error in retry_pending_notifications: {e}", exc_info=True)
            return 0

    def _send_websocket_notification(
        self,
        username: str,
        event_type: EventType,
        message: str,
        notification_id: Optional[int]
    ) -> None:
        """
        Send a WebSocket notification for real-time UI updates using resilient redis_pool

        Args:
            username: Username to notify
            event_type: Type of event
            message: Rendered notification message
            notification_id: ID of the notification log entry
        """
        try:
            redis_client = get_redis()
            if not redis_client:
                logger.warning(f"Cannot send WebSocket notification for {username}, Redis unavailable (circuit breaker open)")
                return

            ws_payload = {
                "type": "notification",
                "username": username,
                "event_type": event_type.value,
                "message": message[:200],  # Truncate for WebSocket
                "notification_id": notification_id,
            }
            redis_client.publish(f"notifications:{username}", json.dumps(ws_payload))
            logger.info(f"Published WebSocket notification for {username}: {event_type.value}")
        except Exception as e:
            logger.error(f"Failed to publish WebSocket notification: {e}", exc_info=True)


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
