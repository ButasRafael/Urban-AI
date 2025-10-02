from fastapi import BackgroundTasks
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.services import rag as rag_svc
import asyncio
import logging

logger = logging.getLogger(__name__)


def enqueue_embeddings(tasks: BackgroundTasks, media_id: int):
    tasks.add_task(_worker, media_id)


async def _worker(media_id: int):
    db: Session = SessionLocal()
    try:
        await rag_svc.ingest_media(db, media_id)
    finally:
        db.close()


def process_media_embeddings(media_id: int):
    """Synchronous wrapper for processing embeddings in Celery tasks"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_worker(media_id))
        finally:
            loop.close()
        logger.info(f"Successfully processed embeddings for media_id={media_id}")
    except Exception as e:
        logger.error(f"Failed to process embeddings for media_id={media_id}: {e}")
        raise

