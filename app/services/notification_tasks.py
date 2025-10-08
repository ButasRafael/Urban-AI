import logging
from app.core.celery_app import celery_app
from app.services.notification_orchestrator import handle_notification_event, NotificationOrchestrator
from app.core.database import SessionLocal

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


@celery_app.task(name="tasks.retry_pending_notifications", bind=True, ignore_result=True)
def retry_pending_notifications_task(self):
    """
    Periodic task to retry pending notifications that are past their retry time
    Runs every 15 minutes to check for notifications ready to be delivered
    """
    db = SessionLocal()
    try:
        logger.info("Running periodic retry of pending notifications")
        orchestrator = NotificationOrchestrator(db)
        retried_count = orchestrator.retry_pending_notifications()
        logger.info(f"Completed retry task: {retried_count} notifications processed")
        return {"retried_count": retried_count}
    except Exception as e:
        logger.error(f"Error in retry_pending_notifications_task: {e}", exc_info=True)
        # Don't retry this task - it will run again on schedule
    finally:
        db.close()
