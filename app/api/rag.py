from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session, joinedload
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import os
import csv
import io
import json
import uuid
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.rag import RAGChunk
from app.models.media import IssueStatus, Severity
from app.services.rag import embed, retrieve
from app.services.rag_query_parser import parse_query_with_filters
from app.services.csv_export import (
    prepare_csv_data,
    store_csv_data,
    get_csv_data,
    image_url_for_media,
)

router = APIRouter(tags=["RAG"])


class ChunkOut(BaseModel):
    id: int
    media_id: int
    chunk: str
    image_url: Optional[str] = None
    # Include metadata for context
    severity: Optional[str] = None
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    address: Optional[str] = None
    class_name: Optional[str] = None

    class Config:
        orm_mode = True


class RAGSearchRequest(BaseModel):
    query: str
    k: Optional[int] = 8
    # Optional explicit filters (if not using GPT parsing)
    severity: Optional[str] = None
    status: Optional[str] = None
    assigned_to: Optional[str] = None


class RAGSearchResponse(BaseModel):
    chunks: List[ChunkOut]
    filters_applied: dict  # Show what filters were applied
    total_count: Optional[int] = None  # Total results when SQL-only mode
    csv_download_url: Optional[str] = None  # URL to download full CSV when truncated


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@router.get("/chunk/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: int, request: Request, db: Session = Depends(get_db)):
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
        image_url=image_url_for_media(row.media, request, row.track_id),
        severity=row.severity.value if row.severity else None,
        status=row.status.value if row.status else None,
        assigned_to=row.assigned_to,
        address=row.address,
        class_name=row.class_name,
    )


@router.get("/download-csv/{csv_id}", name="rag_download_csv")
def download_csv(csv_id: str):
    csv_data = get_csv_data(csv_id)
    if not csv_data:
        raise HTTPException(404, "CSV not found or expired")
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=issues_{csv_id[:8]}.csv"}
    )

@router.post("/search", response_model=RAGSearchResponse)
async def search_rag(
        search_request: RAGSearchRequest,
        request: Request,
        db: Session = Depends(get_db)
):
    """
    Smart RAG search with optimized retrieval:
    1. GPT parses query to extract semantic search and dynamic filters
    2. If SQL-only query (filters + generic terms):
       - Returns ALL matching results without limits
       - No semantic search, no k limit, no per-media cap
    3. Otherwise, two-phase retrieval:
       - Phase 1: Apply SQL filters on dynamic fields (severity, status, assigned_to)
       - Phase 2: Vector similarity search on filtered results (respects k limit)

    The semantic query handles: locations, areas, issue types, dates
    The SQL filters handle: severity, status, assigned_to
    """

    # Parse the query using GPT to extract filters
    parsed = await parse_query_with_filters(search_request.query)

    # Override with explicit filters if provided
    if search_request.severity:
        parsed["severity"] = Severity[search_request.severity.lower()]
    if search_request.status:
        parsed["status"] = IssueStatus[search_request.status.lower()]
    if search_request.assigned_to:
        parsed["assigned_to"] = search_request.assigned_to

    # Check if we should skip semantic search (SQL-only query)
    skip_semantic = parsed.get("sql_only", False)

    # Generate embedding for semantic search (only if needed)
    query_embedding = []
    if not skip_semantic:
        query_embedding = await embed(parsed["query"])

    # Two-phase retrieval (or SQL-only if skip_semantic is True)
    chunks = retrieve(
        db,
        query_embedding,
        k=search_request.k,
        severity_filter=parsed["severity"],
        status_filter=parsed["status"],
        assigned_to_filter=parsed["assigned_to"],
        verified_by_filter=parsed.get("verified_by"),
        resolved_after=parsed.get("resolved_after"),
        resolved_before=parsed.get("resolved_before"),
        verified_after=parsed.get("verified_after"),
        verified_before=parsed.get("verified_before"),
        skip_semantic=skip_semantic
    )

    # For SQL-only mode with many results, limit display but prepare CSV
    total_count = len(chunks)
    csv_download_id = None
    display_limit = 10

    # If SQL-only and we have many results, prepare for CSV download
    if skip_semantic and total_count > display_limit:
        # Store full results for CSV download (in memory cache or database)
        csv_download_id = str(uuid.uuid4())
        csv_data = prepare_csv_data(chunks, request)
        store_csv_data(csv_download_id, csv_data)

        # Limit displayed chunks to first 10
        display_chunks = chunks[:display_limit]
    else:
        display_chunks = chunks

    # Convert to response format
    chunk_results = []
    for chunk in display_chunks:
        chunk_results.append(ChunkOut(
            id=chunk.id,
            media_id=chunk.media_id,
            chunk=chunk.chunk,
            image_url=image_url_for_media(chunk.media, request, chunk.track_id),
            severity=chunk.severity.value if chunk.severity else None,
            status=chunk.status.value if chunk.status else None,
            assigned_to=chunk.assigned_to,
            address=chunk.address,
            class_name=chunk.class_name,
        ))

    # Build CSV download URL if we truncated results
    csv_url = None
    if csv_download_id:
        base_url = str(request.base_url).rstrip("/")
        csv_url = str(request.url_for("rag_download_csv", csv_id=csv_download_id))

    response = RAGSearchResponse(
        chunks=chunk_results,
        filters_applied={
            "semantic_query": parsed["query"],
            "severity": parsed["severity"].value if parsed["severity"] else None,
            "status": parsed["status"].value if parsed["status"] else None,
            "assigned_to": parsed["assigned_to"],
            "verified_by": parsed.get("verified_by"),
            "resolved_after": parsed.get("resolved_after"),
            "resolved_before": parsed.get("resolved_before"),
            "verified_after": parsed.get("verified_after"),
            "verified_before": parsed.get("verified_before"),
            "sql_only": skip_semantic,
            "returned_results": len(chunk_results),
            "k_requested": search_request.k
        }
    )

    # Add total count and CSV URL if applicable
    if skip_semantic:
        response.total_count = total_count
        response.csv_download_url = csv_url

        if total_count >= 1000:
            response.filters_applied["results_truncated"] = True
            response.filters_applied[
                "truncation_message"] = "Results limited to 1000 for performance. Apply more filters to narrow results."

    return response

