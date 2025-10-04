import logging
import os
from typing import Optional, Dict, Any
from uuid import uuid4
import resend
from firebase_admin import messaging, credentials, initialize_app, get_app, exceptions as fb_exceptions

logger = logging.getLogger(__name__)


class EmailChannel:
    """
    Email notification channel using Resend SDK
    """

    def __init__(self):
        resend.api_key = os.getenv("RESEND_API_KEY")
        self.from_email = os.getenv("RESEND_FROM_EMAIL", "Urban AI <noreply@urbanai.com>")

    def send(self, to_email: str, subject: str, body: str, idem_key: Optional[str] = None) -> bool:
        """
        Send an email notification via Resend

        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body (plain text)
            idem_key: Optional idempotency key to prevent duplicate sends

        Returns:
            True if sent successfully, False otherwise
        """
        if not resend.api_key:
            logger.error("RESEND_API_KEY not configured")
            return False

        try:
            params: resend.Emails.SendParams = {
                "from": self.from_email,
                "to": [to_email],
                "subject": subject,
                "text": body,
                "tags": [{"name": "service", "value": "notifications"}],
            }

            # Idempotency key prevents duplicate sends on Celery retries
            headers = {"Idempotency-Key": idem_key or f"notif:{uuid4()}"}

            response = resend.Emails.send(params, headers=headers)
            email_id = getattr(response, "id", None)

            logger.info(f"Email sent to {to_email} (id={email_id})")
            return True

        except Exception as e:
            logger.error(f"Resend send failed: {e}")
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

    def send(self, device_token: str, title: str, body: str, data: Optional[Dict[str, str]] = None) -> bool:
        """
        Send a push notification via FCM

        Args:
            device_token: FCM device token
            title: Notification title
            body: Notification body
            data: Optional additional data to send with the notification

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._initialized:
            logger.error("Firebase not initialized. Cannot send push notification.")
            return False

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
            return True

        except fb_exceptions.UnregisteredError:
            logger.warning(f"FCM token unregistered: {device_token[:8]}***")
            return False
        except fb_exceptions.SenderIdMismatchError:
            logger.error(f"FCM sender mismatch: {device_token[:8]}***")
            return False
        except fb_exceptions.FirebaseError as e:
            logger.error(f"Firebase error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
            return False


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


def send_push_notification(device_token: str, title: str, body: str, data: Optional[Dict[str, str]] = None) -> bool:
    """
    Convenience function to send a push notification

    Args:
        device_token: FCM device token
        title: Notification title
        body: Notification body
        data: Optional additional data

    Returns:
        True if sent successfully, False otherwise
    """
    return push_channel.send(device_token, title, body, data)
