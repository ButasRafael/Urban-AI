import json
import logging
from app.services.events import Event

logger = logging.getLogger(__name__)


class EventPublisher:
    """
    Publishes domain events to Celery for notification processing
    """

    @staticmethod
    def publish(event: Event) -> bool:
        """
        Publish an event to the Celery notification queue

        Args:
            event: The event to publish

        Returns:
            True if published successfully, False otherwise
        """
        try:
            # Convert event to JSON
            event_dict = event.to_dict()
            event_json = json.dumps(event_dict)

            # Dispatch to Celery for processing
            from app.core.celery_app import celery_app
            celery_app.send_task(
                'tasks.process_notification_event',
                args=[event_json],
                queue='notifications'
            )

            logger.info(f"Published event: {event.event_type} with data: {event.data}")
            return True

        except Exception as e:
            logger.error(f"Unexpected error publishing event: {e}")
            return False


def publish_event(event: Event) -> bool:
    """
    Convenience function to publish an event

    Args:
        event: The event to publish

    Returns:
        True if published successfully, False otherwise
    """
    return EventPublisher.publish(event)
