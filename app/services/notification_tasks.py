import logging
from app.core.celery_app import celery_app
from app.services.notification_orchestrator import handle_notification_event

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.process_notification_event", bind=True, ignore_result=True)
def process_notification_event(self, event_json: str):
    """
    Process a notification event

    Args:
        event_json: JSON string of the event
    """
    try:
        logger.info(f"Processing notification event: {event_json}")
        handle_notification_event(event_json)
    except Exception as e:
        logger.error(f"Error processing notification event: {e}", exc_info=True)
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60, max_retries=3)
