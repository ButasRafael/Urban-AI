from sqlalchemy.orm import Session
from sqlalchemy import select
from openai import AsyncOpenAI
from app.models.rag import RAGChunk
import os
import math

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMB_MODEL = "text-embedding-3-small"
EMB_DIM = 1536
R = 6_371_000

async def embed(text: str) -> list[float]:
    resp = await client.embeddings.create(model=EMB_MODEL, input=text)
    return resp.data[0].embedding


def _within_radius(row, q_lat, q_lon, r_km) -> bool:

    if q_lat is None or q_lon is None or r_km is None:
        return True
    dlat = math.radians(row.latitude - q_lat)
    dlon = math.radians(row.longitude - q_lon)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(q_lat))
        * math.cos(math.radians(row.latitude))
        * math.sin(dlon / 2) ** 2
    )
    dist = 2 * R * math.asin(math.sqrt(a))  # metres
    return dist <= r_km * 1_000


def retrieve(db: Session, query_emb: list[float], *, k=8, lat=None, lon=None, radius_km=None):
    pool = (
        db.execute(
            select(RAGChunk)
            .order_by(RAGChunk.embedding.cosine_distance(query_emb))
            .limit(k * 4)
        )
        .scalars()
        .all()
    )
    filtered = [c for c in pool if _within_radius(c, lat, lon, radius_km)]
    return filtered[:k]


async def ingest_media(db: Session, media_id: int):
    """Generate & store RAG chunks for every detection belonging to *media*."""
    from app.models.media import Media, Detection  # local import to avoid cycles

    media = db.query(Media).filter(Media.id == media_id).first()
    if not media:
        return

    for det in (
        db.query(Detection).join(Detection.frame).filter(Detection.frame.has(media_id=media_id)).all()
    ):
        chunk_txt = (
            f"{det.class_name.title()} detected at "
            f"{media.address or f'({media.latitude:.5f},{media.longitude:.5f})'}. "
            f"Description: {det.description or 'n/a'}. "
            f"Suggest fix: {det.solution or 'n/a'}."
        )
        emb = await embed(chunk_txt)
        db.add(
            RAGChunk(
                media_id=media_id,
                chunk=chunk_txt,
                embedding=emb,
                latitude=media.latitude,
                longitude=media.longitude,
            )
        )
    db.commit()
