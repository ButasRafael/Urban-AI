import logging
from typing import Optional, Dict, Any
from datetime import datetime, time, timezone
from sqlalchemy.orm import Session
from app.models.notification import NotificationPreference, EventType, NotificationChannel

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = {
    "low": 0,
    "medium": 1,
    "high": 2
}


class NotificationPreferenceService:
    """
    Service for managing user notification preferences
    """

    @staticmethod
    def get_or_create_preferences(db: Session, username: str) -> NotificationPreference:
        """
        Get user notification preferences, creating default ones if they don't exist

        Args:
            db: Database session
            username: Username

        Returns:
            NotificationPreference object
        """
        prefs = db.query(NotificationPreference).filter(
            NotificationPreference.username == username
        ).first()

        if not prefs:
            # Create default preferences
            prefs = NotificationPreference(
                username=username,
                email_enabled=True,
                push_enabled=True,
                event_preferences={
                    "issue_created": {"email": True, "push": True},
                    "issue_status_changed": {"email": True, "push": True},
                    "issue_assigned": {"email": True, "push": True},
                    "issue_severity_changed": {"email": True, "push": True},
                    "issue_verified": {"email": True, "push": True},
                },
                min_severity="low",
                quiet_hours={"enabled": False},
                digest_enabled=False,
                digest_frequency="immediate"
            )
            db.add(prefs)
            db.commit()
            db.refresh(prefs)
            logger.info(f"Created default notification preferences for user {username}")

        return prefs

    @staticmethod
    def should_notify(
        prefs: NotificationPreference,
        event_type: EventType,
        channel: NotificationChannel,
        severity: str
    ) -> bool:
        """
        Check if a notification should be sent based on user preferences

        Args:
            prefs: User notification preferences
            event_type: Type of event
            channel: Notification channel
            severity: Issue severity

        Returns:
            True if notification should be sent, False otherwise
        """
        # Check if channel is enabled globally
        if channel == NotificationChannel.email and not prefs.email_enabled:
            return False
        if channel == NotificationChannel.push and not prefs.push_enabled:
            return False

        # Check severity threshold
        if severity not in SEVERITY_LEVELS or prefs.min_severity not in SEVERITY_LEVELS:
            logger.warning(f"Invalid severity level: {severity} or {prefs.min_severity}")
            return True  # Default to sending if severity is invalid

        if SEVERITY_LEVELS[severity] < SEVERITY_LEVELS[prefs.min_severity]:
            return False

        # Check event-specific preferences
        event_prefs = prefs.event_preferences.get(event_type.value, {})
        channel_enabled = event_prefs.get(channel.value, True)  # Default to True if not set

        return channel_enabled

    @staticmethod
    def is_quiet_hours(prefs: NotificationPreference) -> bool:
        """
        Check if current time is within user's quiet hours

        Args:
            prefs: User notification preferences

        Returns:
            True if in quiet hours, False otherwise
        """
        quiet_hours = prefs.quiet_hours
        if not quiet_hours or not quiet_hours.get("enabled", False):
            return False

        try:
            # Parse quiet hours
            start_str = quiet_hours.get("start")
            end_str = quiet_hours.get("end")

            if not start_str or not end_str:
                return False

            # Parse time strings (format: "HH:MM")
            start_hour, start_minute = map(int, start_str.split(":"))
            end_hour, end_minute = map(int, end_str.split(":"))

            start_time = time(start_hour, start_minute)
            end_time = time(end_hour, end_minute)

            # Get current time (in UTC or user's timezone if specified)
            # TODO: Support user timezones
            current_time = datetime.now(timezone.utc).time()

            # Check if current time is within quiet hours
            if start_time <= end_time:
                # Normal case: e.g., 22:00 to 08:00 next day
                return start_time <= current_time <= end_time
            else:
                # Wraps around midnight: e.g., 22:00 to 08:00
                return current_time >= start_time or current_time <= end_time

        except Exception as e:
            logger.error(f"Error checking quiet hours: {e}")
            return False

    @staticmethod
    def update_fcm_token(db: Session, username: str, fcm_token: str) -> None:
        """
        Update user's FCM device token for push notifications

        Args:
            db: Database session
            username: Username
            fcm_token: Firebase Cloud Messaging device token
        """
        prefs = NotificationPreferenceService.get_or_create_preferences(db, username)
        prefs.fcm_token = fcm_token
        db.commit()
        logger.info(f"Updated FCM token for user {username}")
