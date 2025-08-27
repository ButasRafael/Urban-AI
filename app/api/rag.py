from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session, joinedload
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import os

from app.core.database import get_db
from app.models.rag import RAGChunk

router = APIRouter(tags=["RAG"])

class ChunkOut(BaseModel):
    id: int
    media_id: int
    chunk: str
    image_url: Optional[str] = None

    class Config:
        orm_mode = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def _abs_static_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/static/{filename}"

def _image_url_for_media(media, request: Request, track_id: Optional[int] = None) -> Optional[str]:
    if not media:
        return None

    static_root = Path(os.getenv("STATIC_DIR", "static"))

    # 1) For videos with track_id: prefer track-specific thumbnail
    if track_id is not None and media.media_type == "video":
        tracks_folder = f"{media.id}_tracks"
        track_thumb = f"track_{track_id}.jpg"
        track_path = static_root / tracks_folder / track_thumb
        if track_path.exists():
            return _abs_static_url(request, f"{tracks_folder}/{track_thumb}")

    # 2) Prefer UUID-based static_filename if available (new system)
    if hasattr(media, 'static_filename') and media.static_filename:
        if (static_root / media.static_filename).exists():
            return _abs_static_url(request, media.static_filename)

    # 3) Fallback to annotated image <id>.jpg produced by inference (old system)
    annotated = f"{media.id}.jpg"
    if (static_root / annotated).exists():
        return _abs_static_url(request, annotated)

    # 4) Fallback to stored filename (normalize ext to a common image type)
    if media.filename:
        name = Path(media.filename).name
        ext = Path(name).suffix.lower()
        if ext not in IMAGE_EXTS:
            name = f"{Path(name).stem}.jpg"
        if (static_root / name).exists():
            return _abs_static_url(request, name)

    # 5) Last resort: still return a plausible absolute URL
    return _abs_static_url(request, media.static_filename if hasattr(media, 'static_filename') and media.static_filename else annotated)

@router.get("/chunk/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: int, request: Request, db: Session = Depends(get_db)):  # ⬅ add request
    row = (
        db.query(RAGChunk)
          .options(joinedload(RAGChunk.media))
          .filter(RAGChunk.id == chunk_id)
          .first()
    )
    if not row:
        raise HTTPException(404, "Chunk not found")

    return ChunkOut(
        id=row.id,
        media_id=row.media_id,
        chunk=row.chunk,
        image_url=_image_url_for_media(row.media, request, row.track_id),
    )
