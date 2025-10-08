import logging
import os
from typing import Optional, Dict, Any
from uuid import uuid4
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from firebase_admin import messaging, credentials, initialize_app, get_app, exceptions as fb_exceptions

logger = logging.getLogger(__name__)


class EmailChannel:
    """
    Email notification channel using MailHog SMTP
    """

    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "localhost")
        self.smtp_port = int(os.getenv("SMTP_PORT", "1025"))
        self.from_email = os.getenv("SMTP_FROM_EMAIL", "noreply@urbanai.com")

    def send(self, to_email: str, subject: str, body: str, idem_key: Optional[str] = None) -> bool:
        """
        Send an email notification via MailHog SMTP

        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body (plain text)
            idem_key: Optional idempotency key (not used in MailHog, kept for API compatibility)

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject

            # Add idempotency key as custom header for debugging
            if idem_key:
                msg['X-Idempotency-Key'] = idem_key
            else:
                msg['X-Idempotency-Key'] = f"notif:{uuid4()}"

            # Attach body
            msg.attach(MIMEText(body, 'plain'))

            # Connect to MailHog SMTP server and send
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.send_message(msg)

            logger.info(f"Email sent to {to_email} via MailHog ({self.smtp_host}:{self.smtp_port})")
            return True

        except Exception as e:
            logger.error(f"MailHog SMTP send failed: {e}")
            return False


class PushNotificationChannel:
    """
    Push notification channel using Firebase Cloud Messaging (FCM)
    """

    def __init__(self):
        self._initialized = False
        self._initialize_firebase()

    def _initialize_firebase(self):
        """
        Initialize Firebase Admin SDK
        """
        try:
            # Check if Firebase is already initialized
            try:
                get_app()
                self._initialized = True
                logger.info("Firebase Admin SDK already initialized")
                return
            except ValueError:
                # Not initialized, proceed with initialization
                pass

            # Get credentials from environment variable or file
            firebase_credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
            firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

            if firebase_credentials_json:
                # Use JSON string from environment variable
                import json
                cred_dict = json.loads(firebase_credentials_json)
                cred = credentials.Certificate(cred_dict)
                initialize_app(cred)
                logger.info("Firebase initialized from JSON credentials")
            elif firebase_credentials_path and os.path.exists(firebase_credentials_path):
                # Use credentials file
                cred = credentials.Certificate(firebase_credentials_path)
                initialize_app(cred)
                logger.info(f"Firebase initialized from file: {firebase_credentials_path}")
            else:
                logger.warning("Firebase credentials not configured. Push notifications will not work.")
                return

            self._initialized = True
            logger.info("Firebase Admin SDK initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            self._initialized = False

    def send(self, device_token: str, title: str, body: str, data: Optional[Dict[str, str]] = None) -> tuple[bool, bool]:
        """
        Send a push notification via FCM

        Args:
            device_token: FCM device token
            title: Notification title
            body: Notification body
            data: Optional additional data to send with the notification

        Returns:
            Tuple of (success, should_clear_token)
            - success: True if sent successfully, False otherwise
            - should_clear_token: True if token should be removed from DB (permanent failure)
        """
        if not self._initialized:
            logger.error("Firebase not initialized. Cannot send push notification.")
            return False, False

        try:
            # Create notification message
            message = messaging.Message(
                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),
                data=data or {},
                token=device_token,
            )

            # Send the message
            response = messaging.send(message)
            logger.info(f"Push notification sent (id={response}) to token={device_token[:8]}***")
            return True, False

        except fb_exceptions.UnregisteredError:
            logger.warning(f"FCM token unregistered: {device_token[:8]}***")
            return False, True  # Token is invalid, should be cleared
        except fb_exceptions.SenderIdMismatchError:
            logger.error(f"FCM sender mismatch: {device_token[:8]}***")
            return False, True  # Token is invalid, should be cleared
        except fb_exceptions.FirebaseError as e:
            logger.error(f"Firebase error: {e}")
            return False, False  # Transient error, keep token
        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
            return False, False  # Unknown error, keep token for safety


# Global channel instances
email_channel = EmailChannel()
push_channel = PushNotificationChannel()


def send_email(to_email: str, subject: str, body: str, idem_key: Optional[str] = None) -> bool:
    """
    Convenience function to send an email

    Args:
        to_email: Recipient email address
        subject: Email subject
        body: Email body
        idem_key: Optional idempotency key to prevent duplicate sends

    Returns:
        True if sent successfully, False otherwise
    """
    return email_channel.send(to_email, subject, body, idem_key)


def send_push_notification(device_token: str, title: str, body: str, data: Optional[Dict[str, str]] = None) -> tuple[bool, bool]:
    """
    Convenience function to send a push notification

    Args:
        device_token: FCM device token
        title: Notification title
        body: Notification body
        data: Optional additional data

    Returns:
        Tuple of (success, should_clear_token)
        - success: True if sent successfully, False otherwise
        - should_clear_token: True if token should be removed from DB (permanent failure)
    """
    return push_channel.send(device_token, title, body, data)
