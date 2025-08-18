# app/services/rag.py
from __future__ import annotations

import os
from typing import List

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from openai import AsyncOpenAI

from app.models.rag import RAGChunk

# --------------------------
# Embeddings (always unit-norm)
# --------------------------
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMB_MODEL = "text-embedding-3-small"
EMB_DIM = 1536


async def embed(text: str) -> list[float]:
    """Create a unit-normalized embedding vector."""
    resp = await client.embeddings.create(model=EMB_MODEL, input=text)
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n > 0 else v.tolist()


# --------------------------
# MMR (deterministic)
# --------------------------
def _mmr(
    query_vec: np.ndarray,
    cand_vecs: np.ndarray,
    k: int,
    lam: float = 0.6,
) -> list[int]:
    """
    Maximal Marginal Relevance.
    - query_vec: (D,) unit vector
    - cand_vecs: (N,D) unit vectors
    - lam in [0,1]: 1.0 -> pure relevance; 0.0 -> pure diversity
    Returns indices into cand_vecs.
    """
    if cand_vecs.size == 0 or k <= 0:
        return []

    sim_q = cand_vecs @ query_vec
    selected: list[int] = []
    remaining = set(range(cand_vecs.shape[0]))

    while remaining and len(selected) < k:
        rem = np.array(sorted(remaining), dtype=int)
        if not selected:
            pick = int(rem[np.argmax(sim_q[rem])])
        else:
            sel = cand_vecs[selected]
            sim_to_sel = (cand_vecs[rem] @ sel.T).max(axis=1)
            scores = lam * sim_q[rem] - (1.0 - lam) * sim_to_sel
            pick = int(rem[np.argmax(scores)])
        selected.append(pick)
        remaining.remove(pick)

    return selected


# --------------------------
# Retrieval (no geo)
# --------------------------
def retrieve(
    db: Session,
    query_emb: List[float],
    *,
    k: int = 8,
    pool_factor: int = 6,     # pull more candidates, then MMR
    mmr_lambda: float = 0.6,  # 0.6 favors relevance; lower -> more diversity
    per_media_cap: int = 2,   # limit from same media
) -> list[RAGChunk]:
    """
    Pipeline (geo-agnostic):
      1) ANN order by cosine distance (ivfflat).
      2) MMR for diversity.
      3) Per-media cap + exact-k backfill.
    """
    pool_n = max(k * pool_factor, k + 2)

    # 1) ANN candidates
    stmt = (
        select(RAGChunk)
        .order_by(RAGChunk.embedding.cosine_distance(query_emb))
        .limit(pool_n)
    )
    cand_rows: list[RAGChunk] = db.execute(stmt).scalars().all()
    if not cand_rows:
        return []

    # 2) Prepare for MMR
    qv = np.asarray(query_emb, dtype=np.float32)
    qn = float(np.linalg.norm(qv))
    if qn > 0:
        qv /= qn

    mat = np.stack([np.asarray(r.embedding, dtype=np.float32) for r in cand_rows], axis=0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    mmr_idxs = _mmr(qv, mat, k=k, lam=mmr_lambda)

    # 3) Per-media cap + exact-k
    picked: list[RAGChunk] = []
    picked_ids: set[int] = set()
    per_media_counts: dict[int, int] = {}

    for i in mmr_idxs:
        r = cand_rows[i]
        if r.id in picked_ids:
            continue
        cnt = per_media_counts.get(r.media_id, 0)
        if cnt >= per_media_cap:
            continue
        picked.append(r)
        picked_ids.add(r.id)
        per_media_counts[r.media_id] = cnt + 1
        if len(picked) == k:
            return picked

    # backfill in ANN arrival order
    for r in cand_rows:
        if len(picked) == k:
            break
        if r.id in picked_ids:
            continue
        cnt = per_media_counts.get(r.media_id, 0)
        if cnt >= per_media_cap:
            continue
        picked.append(r)
        picked_ids.add(r.id)
        per_media_counts[r.media_id] = cnt + 1

    return picked[:k]


# --------------------------
# Ingestion
# --------------------------
async def ingest_media(db: Session, media_id: int):
    """
    Generate & store RAG chunks for every detection belonging to *media*.
    Each chunk is embedded (unit-norm). No geo recorded/used.
    """
    from app.models.media import Media, Detection  # avoid cycles

    media = db.query(Media).filter(Media.id == media_id).first()
    if not media:
        return

    detects = (
        db.query(Detection)
          .join(Detection.frame)
          .filter(Detection.frame.has(media_id=media_id))
          .all()
    )

    for det in detects:
        where_txt = media.address or "unspecified location"
        chunk_txt = (
            f"{det.class_name.title()} detected at {where_txt}. "
            f"Description: {det.description or 'n/a'}. "
            f"Suggest fix: {det.solution or 'n/a'}."
        )
        emb = await embed(chunk_txt)
        db.add(
            RAGChunk(
                media_id=media_id,
                chunk=chunk_txt,
                embedding=emb,
                # latitude/longitude intentionally omitted
            )
        )
    db.commit()
