from fastapi import BackgroundTasks
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.services import rag as rag_svc


def enqueue_embeddings(tasks: BackgroundTasks, media_id: int):
    tasks.add_task(_worker, media_id)


async def _worker(media_id: int):
    db: Session = SessionLocal()
    try:
        await rag_svc.ingest_media(db, media_id)
    finally:
        db.close()

