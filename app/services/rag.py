from __future__ import annotations

import os
from typing import List, Optional
from datetime import datetime, timezone
import json
from collections import defaultdict
import logging

import numpy as np
from sqlalchemy import select, and_, text as sqltext
from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder

from app.models.rag import RAGChunk
from app.models.media import IssueStatus, Severity
from app.core.datetime_utils import ro_date_bounds_to_utc, format_datetime_ro

# Set up file logging for RAG retrieval debugging
rag_logger = logging.getLogger('RAG_RETRIEVAL')
rag_logger.setLevel(logging.INFO)

# Clear any existing handlers to avoid duplicates
rag_logger.handlers = []

# Create formatter with more concise timestamp
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

# File handler - only for RAG logs
file_handler = logging.FileHandler('rag_retrieval.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to RAG logger only
rag_logger.addHandler(file_handler)
rag_logger.addHandler(console_handler)

# Prevent propagation to root logger
rag_logger.propagate = False

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMB_MODEL = "text-embedding-3-large"
EMB_DIM = 3072

# Initialize cross-encoder for reranking
# Using bge-reranker-v2-m3 for better multilingual (Romanian) support
cross_encoder = None
reranker_type = None

def load_reranker():
    """Load the reranker model. Called during app startup."""
    global cross_encoder, reranker_type
    
    if cross_encoder is not None:
        return  # Already loaded
    
    try:
        from FlagEmbedding import FlagReranker
        cross_encoder = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        reranker_type = 'bge-m3'
        rag_logger.info("BGE-M3 reranker loaded successfully for multilingual support")
    except ImportError:
        rag_logger.warning("FlagEmbedding not installed. Install with: pip install FlagEmbedding")
        try:
            # Fallback to sentence-transformers cross-encoder
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2', max_length=512)
            reranker_type = 'cross-encoder'
            rag_logger.info("Fallback cross-encoder loaded (limited multilingual support)")
        except Exception as e:
            rag_logger.warning(f"Failed to load any cross-encoder: {e}. Will fall back to embedding-based reranking.")


async def embed(text: str) -> list[float]:
    resp = await client.embeddings.create(model=EMB_MODEL, input=text)
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n > 0 else v.tolist()

def _rrf(ranking_ids: list[int], k0: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion scores: 1/(k0 + rank)."""
    return {doc_id: 1.0 / (k0 + rank) for rank, doc_id in enumerate(ranking_ids, start=1)}


def _bm25_candidates(
    db: Session,
    query_text: str,
    *,
    limit: int,
    severity_filter: Optional[Severity],
    status_filter: Optional[IssueStatus],
    assigned_to_filter: Optional[str],
    verified_by_filter: Optional[str],
    resolved_after: Optional[str],
    resolved_before: Optional[str],
    verified_after: Optional[str],
    verified_before: Optional[str],
) -> list[tuple[int, float]]:
    where_clauses = []
    params = {"q": query_text, "limit": limit}

    if severity_filter:
        where_clauses.append("rag_chunks.severity = :sev")
        params["sev"] = severity_filter.value
    if status_filter:
        where_clauses.append("rag_chunks.status = :st")
        params["st"] = status_filter.value
    if assigned_to_filter:
        where_clauses.append("rag_chunks.assigned_to = :asgn")
        params["asgn"] = assigned_to_filter
    if verified_by_filter:
        where_clauses.append("rag_chunks.verified_by = :vby")
        params["vby"] = verified_by_filter

    def _date_clause(col: str, op: str, val: Optional[str], key: str):
        if not val: return
        where_clauses.append(f"rag_chunks.{col} {op} (TIMESTAMP WITH TIME ZONE :{key})")
        params[key] = f"{val} 00:00:00+00" if op == ">=" else f"{val} 23:59:59+00"

    _date_clause("resolved_at", ">=", resolved_after,  "res_after")
    _date_clause("resolved_at", "<=", resolved_before, "res_before")
    _date_clause("verified_at", ">=", verified_after,  "ver_after")
    _date_clause("verified_at", "<=", verified_before, "ver_before")

    where_sql = " AND ".join(where_clauses)
    if where_sql:
        where_sql = " AND " + where_sql
    
    # Build a combined query for both languages
    sql = sqltext(f"""
        SELECT id,
               ts_rank_cd(tsv, 
                   plainto_tsquery('english', immutable_unaccent(:q)) ||
                   plainto_tsquery('romanian', immutable_unaccent(:q)), 
                   32) AS bm25
        FROM rag_chunks
        WHERE tsv @@ (plainto_tsquery('english', immutable_unaccent(:q)) ||
                     plainto_tsquery('romanian', immutable_unaccent(:q)))
        {where_sql}
        ORDER BY bm25 DESC
        LIMIT :limit
    """)

    rows = db.execute(sql, params).all()
    return [(int(r[0]), float(r[1])) for r in rows]


def retrieve(
        db: Session,
        query_emb: List[float],
        *,
        query_text: Optional[str] = None,
        k: int = 8,
        pool_factor: int = 8,
        per_media_cap: int = 2,
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
    """
    Hybrid retrieval with cross-encoder reranking (MMR disabled for better relevance):
      1) Dynamic SQL filters (fast)
      2) Vector pass to get top-N by cosine distance (unless skip_semantic)
      3) BM25 pass (FTS) over tsvector column (needs query_text)
      4) RRF fuse IDs, then fetch rows
      5) Cross-encoder rerank (bge-reranker-v2-m3 or ms-marco-MiniLM-L12-v2) or embedding-based fallback
      6) Per-media cap → top-k (preserving cross-encoder ranking)
    """

    # Safety
    pool_n = max(k * pool_factor, k + 2)
    
    # Log initial search parameters
    rag_logger.info(f"\n{'='*80}")
    rag_logger.info(f"NEW RAG SEARCH - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    rag_logger.info(f"{'='*80}")
    rag_logger.info(f"Query: '{query_text[:200]}...' (truncated)" if query_text and len(query_text) > 200 else f"Query: '{query_text or 'Embedding-only search'}'")
    rag_logger.info(f"Reranker: {'BAAI/bge-reranker-v2-m3 (Multilingual)' if reranker_type == 'bge-m3' else 'cross-encoder/ms-marco-MiniLM-L12-v2' if reranker_type == 'cross-encoder' else 'None - Embedding only'}")
    rag_logger.info(f"Params: k={k}, pool_n={pool_n}, per_media_cap={per_media_cap} (MMR disabled)")
    rag_logger.info(f"Filters: severity={severity_filter}, status={status_filter}, assigned={assigned_to_filter}")

    # ---- Phase 1: base SQL filter query builder (no ORDER yet)
    base_filters = []
    if severity_filter:
        base_filters.append(RAGChunk.severity == severity_filter)
    if status_filter:
        base_filters.append(RAGChunk.status == status_filter)
    if assigned_to_filter:
        base_filters.append(RAGChunk.assigned_to == assigned_to_filter)
    if verified_by_filter:
        base_filters.append(RAGChunk.verified_by == verified_by_filter)

    def _add_date_filter(col, op, val):
        if not val: return None
        try:
            dt = datetime.strptime(val, "%Y-%m-%d").date()
            start_utc, end_utc = ro_date_bounds_to_utc(dt)
            return (RAGChunk.__table__.c[col] >= start_utc) if op == ">=" else (RAGChunk.__table__.c[col] <= end_utc)
        except ValueError:
            return None

    for op, col, val in [
        (">=", "resolved_at", resolved_after),
        ("<=", "resolved_at", resolved_before),
        (">=", "verified_at", verified_after),
        ("<=", "verified_at", verified_before),
    ]:
        filt = _add_date_filter(col, op, val)
        if filt is not None:
            base_filters.append(filt)

    # ---- If skip_semantic: pure SQL only (no BM25)
    if skip_semantic:
        # Pure SQL filters ordered by severity and upload time
        from sqlalchemy import case, desc
        severity_order = case(
            (RAGChunk.severity == Severity.high, 1),
            (RAGChunk.severity == Severity.medium, 2),
            (RAGChunk.severity == Severity.low, 3),
            else_=4
        )
        stmt = select(RAGChunk).where(and_(*base_filters)).order_by(severity_order, RAGChunk.uploaded_at.desc()).limit(1000)
        return db.execute(stmt).scalars().all()

    # ---- Phase 2: vector lane (top-N by cosine distance)
    stmt_vec = select(RAGChunk).where(and_(*base_filters)).order_by(
        RAGChunk.embedding.cosine_distance(query_emb)
    ).limit(pool_n)
    vec_rows: list[RAGChunk] = db.execute(stmt_vec).scalars().all()
    vec_ids = [r.id for r in vec_rows]
    
    rag_logger.info(f"\n{'='*80}")
    rag_logger.info(f"VECTOR SEARCH RESULTS (Semantic Similarity)")
    rag_logger.info(f"{'='*80}")
    rag_logger.info(f"Query embedding: {EMB_MODEL} | Found: {len(vec_ids)} results | Showing top 10")
    rag_logger.info("-" * 80)
    
    for i, r in enumerate(vec_rows[:10]):
        # Format the output more clearly
        severity_badge = f"[{r.severity.upper()}]" if r.severity else "[N/A]"
        status_badge = f"[{r.status.upper()}]" if r.status else "[OPEN]"
        media_info = f"Media #{r.media_id}" if r.media_id else "No media"
        
        rag_logger.info(f"  {i+1}. {severity_badge} {status_badge} {r.class_name or 'Unknown class'}")
        rag_logger.info(f"     📍 Location: {r.address or 'No address'}")
        rag_logger.info(f"     📄 {media_info} | Detection #{r.detection_id if r.detection_id else 'N/A'}")
        rag_logger.info(f"     💬 Preview: {r.chunk[:120]}..." if len(r.chunk) > 120 else f"     💬 Content: {r.chunk}")
        rag_logger.info("")

    # ---- Phase 3: BM25 lane (needs query_text)
    bm25_ids = []
    bm25_scores = []
    if query_text and query_text.strip():
        bm25_results = _bm25_candidates(
            db, query_text,
            limit=min(pool_n, 500),
            severity_filter=severity_filter,
            status_filter=status_filter,
            assigned_to_filter=assigned_to_filter,
            verified_by_filter=verified_by_filter,
            resolved_after=resolved_after,
            resolved_before=resolved_before,
            verified_after=verified_after,
            verified_before=verified_before,
        )
        bm25_ids = [rid for rid, _ in bm25_results]
        bm25_scores = {rid: score for rid, score in bm25_results}
        
        rag_logger.info(f"\n{'='*80}")
        rag_logger.info(f"BM25 SEARCH RESULTS (Keyword Matching)")
        rag_logger.info(f"{'='*80}")
        rag_logger.info(f"Query text: '{query_text[:100]}...' | Found: {len(bm25_ids)} results")
        rag_logger.info("-" * 80)
        # Get chunk details for top BM25 results
        if bm25_ids:
            top_bm25 = db.execute(select(RAGChunk).where(RAGChunk.id.in_(bm25_ids[:10]))).scalars().all()
            bm25_dict = {r.id: r for r in top_bm25}
            for i, (rid, score) in enumerate(bm25_results[:10]):
                if rid in bm25_dict:
                    r = bm25_dict[rid]
                    severity_badge = f"[{r.severity.upper()}]" if r.severity else "[N/A]"
                    rag_logger.info(f"  {i+1}. BM25 Score: {score:.4f} | {severity_badge} {r.class_name or 'Unknown'}")
                    rag_logger.info(f"     📍 {r.address[:60]}..." if r.address and len(r.address) > 60 else f"     📍 {r.address or 'No address'}")
                    rag_logger.info(f"     💬 {r.chunk[:100]}...")
                    rag_logger.info("")

    # If BM25 found nothing (short query, stop-words), just use vectors
    if not bm25_ids:
        cand_rows = vec_rows
    else:
        # ---- Phase 4: RRF fuse ids
        scores = defaultdict(float)
        vec_rrf_scores = _rrf(vec_ids)
        bm25_rrf_scores = _rrf(bm25_ids)
        
        for d, s in vec_rrf_scores.items():   scores[d] += s
        for d, s in bm25_rrf_scores.items():  scores[d] += s

        fused_sorted = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
        fused_ids = fused_sorted[:pool_n]
        
        # Print RRF fusion results
        vec_only = set(vec_ids) - set(bm25_ids)
        bm25_only = set(bm25_ids) - set(vec_ids)
        both = set(vec_ids) & set(bm25_ids)
        
        rag_logger.info(f"\n{'='*80}")
        rag_logger.info(f"RECIPROCAL RANK FUSION (RRF) - Combining Vector & BM25")
        rag_logger.info(f"{'='*80}")
        rag_logger.info(f"📊 Distribution:")
        rag_logger.info(f"   • Vector-only results: {len(vec_only)} documents")
        rag_logger.info(f"   • BM25-only results: {len(bm25_only)} documents")
        rag_logger.info(f"   • Found in both: {len(both)} documents (highest priority)")
        rag_logger.info(f"\n🏆 Top 10 after fusion (k0=60):")
        
        # Get details for top fused results
        top_fused_chunks = db.execute(select(RAGChunk).where(RAGChunk.id.in_(fused_ids[:10]))).scalars().all()
        fused_dict = {r.id: r for r in top_fused_chunks}
        
        for i, doc_id in enumerate(fused_ids[:10]):
            if doc_id in fused_dict:
                r = fused_dict[doc_id]
                in_both = "✓ BOTH" if doc_id in both else "Vector" if doc_id in vec_only else "BM25"
                rag_logger.info(f"   {i+1}. [{in_both}] RRF={scores[doc_id]:.4f} | {r.class_name} @ {r.address[:40]}...")

        # Gather rows: we already have vec_rows; fetch missing (BM25-only) ids
        have = {r.id: r for r in vec_rows}
        missing_ids = [i for i in fused_ids if i not in have]
        if missing_ids:
            extra_rows = db.execute(select(RAGChunk).where(RAGChunk.id.in_(missing_ids))).scalars().all()
            have.update({r.id: r for r in extra_rows})
        cand_rows = [have[i] for i in fused_ids if i in have]

    if not cand_rows:
        return []

    # ---- Phase 5: Rerank using cross-encoder if available and query_text exists
    # Ensure reranker is loaded (in case it wasn't loaded at startup)
    if query_text and query_text.strip():
        if cross_encoder is None:
            load_reranker()
    
    if cross_encoder and query_text and query_text.strip():
        # Use cross-encoder for reranking
        rag_logger.info(f"\n{'='*80}")
        rag_logger.info(f"CROSS-ENCODER RERANKING (Neural Relevance Scoring)")
        rag_logger.info(f"{'='*80}")
        model_name = "BAAI/bge-reranker-v2-m3 (Multilingual)" if reranker_type == 'bge-m3' else "cross-encoder/ms-marco-MiniLM-L12-v2"
        rag_logger.info(f"Model: {model_name}")
        rag_logger.info(f"Reranking {len(cand_rows)} candidates based on query-document relevance")
        rag_logger.info("-" * 80)

        pairs = [(query_text, row.chunk) for row in cand_rows]

        try:
            if reranker_type == 'bge-m3':
                ce_scores = cross_encoder.compute_score(pairs)
            else:
                ce_scores = cross_encoder.predict(pairs)


            scored_rows = list(zip(cand_rows, ce_scores))
            reranked = [row for row, _ in sorted(scored_rows, key=lambda x: x[1], reverse=True)]


            top_reranked = sorted(scored_rows, key=lambda x: x[1], reverse=True)[:10]
            for i, (row, score) in enumerate(top_reranked):
                severity_badge = f"[{row.severity.upper()}]" if row.severity else "[N/A]"
                status_badge = f"[{row.status.upper()}]" if row.status else "[OPEN]"
                score_indicator = "🟢" if score > 2 else "🟡" if score > 0 else "🔴"
                
                rag_logger.info(f"  {i+1}. {score_indicator} CE Score: {score:+.3f} | {severity_badge} {status_badge}")
                rag_logger.info(f"     🏷️  {row.class_name or 'Unknown'} | Media #{row.media_id}")
                rag_logger.info(f"     📍 {row.address or 'No address'}")
                rag_logger.info(f"     💬 {row.chunk[:100]}...")
                rag_logger.info("")
            
            # Summary statistics
            ce_scores_array = np.array(ce_scores)
            rag_logger.info(f"\n📈 Score Statistics:")
            rag_logger.info(f"   • Max: {ce_scores_array.max():.3f}")
            rag_logger.info(f"   • Min: {ce_scores_array.min():.3f}")
            rag_logger.info(f"   • Mean: {ce_scores_array.mean():.3f}")
            rag_logger.info(f"   • Positive scores: {(ce_scores_array > 0).sum()}/{len(ce_scores_array)}")
                
        except Exception as e:
            rag_logger.warning(f"Cross-encoder failed: {e}. Falling back to embedding-based reranking.")
            # Fallback to embedding-based reranking
            qv = np.asarray(query_emb, dtype=np.float32)
            qn = float(np.linalg.norm(qv))
            if qn > 0: qv /= qn

            def _sim(row: RAGChunk) -> float:
                ev = np.asarray(row.embedding, dtype=np.float32)
                en = float(np.linalg.norm(ev))
                if en > 0: ev = ev / en
                return float(ev @ qv)

            reranked = sorted(cand_rows, key=_sim, reverse=True)
    else:
        # Fallback to embedding-based reranking when cross-encoder unavailable or no query text
        rag_logger.info(f"\n{'='*80}")
        rag_logger.info(f"FALLBACK: Embedding-based Reranking")
        rag_logger.info(f"{'='*80}")
        rag_logger.info(f"Using cosine similarity with {EMB_MODEL}")
        rag_logger.info(f"Reranking {len(cand_rows)} candidates")
        qv = np.asarray(query_emb, dtype=np.float32)
        qn = float(np.linalg.norm(qv))
        if qn > 0: qv /= qn

        def _sim(row: RAGChunk) -> float:
            ev = np.asarray(row.embedding, dtype=np.float32)
            en = float(np.linalg.norm(ev))
            if en > 0: ev = ev / en
            return float(ev @ qv)

        reranked = sorted(cand_rows, key=_sim, reverse=True)

    # ---- Phase 6: Apply per-media cap while preserving cross-encoder ranking
    selected: list[RAGChunk] = []
    per_media_counts: dict[int, int] = {}
    
    # Take top-k from reranked results, respecting per-media cap
    for row in reranked:
        cnt = per_media_counts.get(row.media_id, 0)
        if cnt >= per_media_cap:
            continue
        selected.append(row)
        per_media_counts[row.media_id] = cnt + 1
        if len(selected) == k:
            break
    
    final_results = selected[:k]
    
    # Log final results summary
    if final_results:
        rag_logger.info(f"\n{'='*80}")
        rag_logger.info(f"FINAL RESULTS - Top {len(final_results)} documents (Cross-Encoder ranking preserved)")
        rag_logger.info(f"{'='*80}")
        rag_logger.info(f"Per-media cap: {per_media_cap} | No MMR applied - using pure CE scores")
        rag_logger.info("-" * 80)
        
        media_distribution = {}
        for r in final_results:
            media_distribution[r.media_id] = media_distribution.get(r.media_id, 0) + 1
        
        for i, r in enumerate(final_results):
            severity_badge = f"[{r.severity.upper()}]" if r.severity else "[N/A]"
            rag_logger.info(f"  {i+1}. {severity_badge} {r.class_name} | Media #{r.media_id}")
            rag_logger.info(f"     📍 {r.address[:80]}..." if r.address and len(r.address) > 80 else f"     📍 {r.address or 'N/A'}")
        
        rag_logger.info(f"\n📊 Media distribution: {dict(media_distribution)}")
        rag_logger.info(f"{'='*80}\n")
    
    return final_results


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
