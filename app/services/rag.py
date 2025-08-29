from __future__ import annotations

import os
from typing import List, Optional
from datetime import datetime, timezone
import json

import numpy as np
from sqlalchemy import select, and_
from sqlalchemy.orm import Session
from openai import AsyncOpenAI

from app.models.rag import RAGChunk
from app.models.media import IssueStatus, Severity
from app.core.datetime_utils import ro_date_bounds_to_utc, format_datetime_ro

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMB_MODEL = "text-embedding-3-large"
EMB_DIM = 3072


async def embed(text: str) -> list[float]:
    resp = await client.embeddings.create(model=EMB_MODEL, input=text)
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n > 0 else v.tolist()


def _mmr(
        query_vec: np.ndarray,
        cand_vecs: np.ndarray,
        k: int,
        lam: float = 0.6,
) -> list[int]:
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


def retrieve(
        db: Session,
        query_emb: List[float],
        *,
        k: int = 8,
        pool_factor: int = 6,
        mmr_lambda: float = 0.6,
        per_media_cap: int = 2,
        # Dynamic filters only (Phase 1 - SQL filters)
        severity_filter: Optional[Severity] = None,
        status_filter: Optional[IssueStatus] = None,
        assigned_to_filter: Optional[str] = None,
        verified_by_filter: Optional[str] = None,
        resolved_after: Optional[str] = None,
        resolved_before: Optional[str] = None,
        verified_after: Optional[str] = None,
        verified_before: Optional[str] = None,
        skip_semantic: bool = False,
) -> list[RAGChunk]:
    pool_n = max(k * pool_factor, k + 2)

    # Phase 1: Apply dynamic field filters (fast, indexed SQL)
    stmt = select(RAGChunk)

    filters = []
    if severity_filter:
        filters.append(RAGChunk.severity == severity_filter)
    if status_filter:
        filters.append(RAGChunk.status == status_filter)
    if assigned_to_filter:
        filters.append(RAGChunk.assigned_to == assigned_to_filter)
    if verified_by_filter:
        filters.append(RAGChunk.verified_by == verified_by_filter)

    if resolved_after:
        try:
            date_parsed = datetime.strptime(resolved_after, "%Y-%m-%d").date()
            start_utc, _ = ro_date_bounds_to_utc(date_parsed)
            filters.append(RAGChunk.resolved_at >= start_utc)
        except ValueError:
            pass
    if resolved_before:
        try:
            date_parsed = datetime.strptime(resolved_before, "%Y-%m-%d").date()
            _, end_utc = ro_date_bounds_to_utc(date_parsed)
            filters.append(RAGChunk.resolved_at <= end_utc)
        except ValueError:
            pass

    if verified_after:
        try:
            date_parsed = datetime.strptime(verified_after, "%Y-%m-%d").date()
            start_utc, _ = ro_date_bounds_to_utc(date_parsed)
            filters.append(RAGChunk.verified_at >= start_utc)
        except ValueError:
            pass
    if verified_before:
        try:
            date_parsed = datetime.strptime(verified_before, "%Y-%m-%d").date()
            _, end_utc = ro_date_bounds_to_utc(date_parsed)
            filters.append(RAGChunk.verified_at <= end_utc)
        except ValueError:
            pass

    if filters:
        stmt = stmt.where(and_(*filters))

    if skip_semantic:

        has_filters = bool(filters)
        if not has_filters:

            return []

        from sqlalchemy import case, desc
        severity_order = case(
            (RAGChunk.severity == Severity.high, 1),
            (RAGChunk.severity == Severity.medium, 2),
            (RAGChunk.severity == Severity.low, 3),
            else_=4
        )

        # Add a safety limit to prevent memory issues
        MAX_SQL_RESULTS = 1000

        stmt = stmt.order_by(severity_order, RAGChunk.uploaded_at.desc()).limit(MAX_SQL_RESULTS)
        all_results: list[RAGChunk] = db.execute(stmt).scalars().all()
        return all_results

    # Phase 2: Vector similarity search (handles location, area, class_name via embeddings)

    stmt = stmt.order_by(RAGChunk.embedding.cosine_distance(query_emb)).limit(pool_n)
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


async def ingest_media(db: Session, media_id: int):
    from app.models.media import Media, Detection

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
        chunk_txt = (
            f"{det.class_name.title()} issue detected at {media.address or 'unknown location'}. "
            f"Type: {det.class_name}. "
            f"Description: {det.description or 'No description available'}. "
            f"Suggested solution: {det.solution or 'No solution provided'}. "
            f"Captured: {format_datetime_ro(media.created_at, '%B %d, %Y at %H:%M')}. "
            f"Media type: {media.media_type}."
        )

        emb = await embed(chunk_txt)

        # Store in database
        db.add(
            RAGChunk(
                media_id=media_id,
                track_id=det.track_id,
                detection_id=det.id,
                chunk=chunk_txt,
                embedding=emb,

                address=media.address,
                media_type=media.media_type,
                class_name=det.class_name,
                uploaded_at=media.created_at,

                severity=det.severity,
                status=det.status,
                assigned_to=det.assigned_to,
                verified_by=det.verified_by,
                resolved_at=det.resolved_at,
                verified_at=det.verified_at,

                extra_metadata={
                    "confidence": det.confidence,
                    "frames_detected": det.frames_detected,
                    "verified_by": det.verified_by,
                    "source": det.source.value if det.source else None
                }
            )
        )
    db.commit()


async def sync_detection_updates(db: Session, detection_id: int):

    from app.models.media import Detection

    # Get the updated detection
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        return

    # Update all RAG chunks for this detection
    chunks = db.query(RAGChunk).filter(RAGChunk.detection_id == detection_id).all()

    for chunk in chunks:
        # Update dynamic fields only
        chunk.severity = detection.severity
        chunk.status = detection.status
        chunk.assigned_to = detection.assigned_to
        chunk.verified_by = detection.verified_by
        chunk.resolved_at = detection.resolved_at
        chunk.verified_at = detection.verified_at
        chunk.updated_at = datetime.now(timezone.utc)

    db.commit()
    print(f"Updated {len(chunks)} RAG chunks for detection {detection_id}")


async def bulk_reingest_all_media(db: Session):

    from app.models.media import Media

    # Clear existing chunks
    db.query(RAGChunk).delete()
    db.commit()

    # Get all media
    all_media = db.query(Media).all()

    for media in all_media:
        await ingest_media(db, media.id)
        print(f"Reingested media {media.id}: {media.filename}")

    print(f"Reingested {len(all_media)} media files")
