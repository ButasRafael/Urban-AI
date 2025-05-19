from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from app.core.security import require_roles, get_current_user
from app.core.database import get_db
from app.models.conversation import ChatSession, ChatMessage
from app.services import rag as rag_svc
from openai import AsyncOpenAI, InternalServerError
import asyncio
import os
from app.models.schemas_chat import (
    ChatRequest,
    ChatResponse,
    SessionSummary,
    SessionHistory,
    ChatMessageResponse,
)

router = APIRouter(tags=["Chat"], dependencies=[require_roles("authority")])
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def _build_context(db: Session, req: ChatRequest) -> str:
    emb = await rag_svc.embed(req.message)
    chunks = rag_svc.retrieve(
        db, emb, k=8, lat=req.latitude, lon=req.longitude, radius_km=req.radius_km
    )
    return "\n---\n".join(c.chunk for c in chunks)


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    # 1. conversation store
    if req.session_id:
        session = db.query(ChatSession).filter(ChatSession.id == req.session_id).first()
        if not session:
            raise HTTPException(404, "session_id not found")
    else:
        session = ChatSession(authority_username=current_user.username)
        db.add(session)
        db.commit(); db.refresh(session)

    # persist user message
    db.add(ChatMessage(session_id=session.id, role="user", content=req.message))
    db.commit()

    # 2. assemble prompt with retrieved context + last 5 turns
    context = await _build_context(db, req)

    history_msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(10)
        .all()[::-1] 
    )
    chat_roll = [
        {"role": m.role, "content": m.content} for m in history_msgs if m.role in ("user", "assistant")
    ]

    system = "You are an urban‑maintenance assistant. Use provided context strictly."
    messages = [{"role": "system", "content": system}] + chat_roll + [
        {"role": "system", "name": "context", "content": context},
        {"role": "user", "content": req.message},
    ]

    try:
        completion = await client.chat.completions.create(model="gpt-4o", messages=messages)
    except InternalServerError as e:
        raise HTTPException(502, "Upstream LLM error") from e

    answer = completion.choices[0].message.content

    # store assistant reply
    db.add(ChatMessage(session_id=session.id, role="assistant", content=answer))
    db.commit()

    return ChatResponse(session_id=session.id, answer=answer)

@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    sessions = (
        db.query(ChatSession)
          .filter(ChatSession.authority_username == current_user.username)
          .order_by(ChatSession.created_at.desc())
          .all()
    )
    out: List[SessionSummary] = []
    for s in sessions:
        last = s.messages[-1].created_at if s.messages else s.created_at
        out.append(SessionSummary(
            id=s.id,
            created_at=s.created_at,
            last_message_at=last
        ))
    return out

@router.get("/sessions/{session_id}", response_model=SessionHistory)
async def get_session_history(
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    session = (
        db.query(ChatSession)
          .filter(
              ChatSession.id == session_id,
              ChatSession.authority_username == current_user.username
          )
          .first()
    )
    if not session:
        raise HTTPException(404, "Session not found")

    msgs = (
        db.query(ChatMessage)
          .filter(ChatMessage.session_id == session_id)
          .order_by(ChatMessage.created_at)
          .all()
    )
    return SessionHistory(
        messages=[
            ChatMessageResponse(
                role=m.role,
                content=m.content,
                created_at=m.created_at
            )
            for m in msgs
        ]
    )

@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Delete one chat session and all its messages."""
    session = (
        db.query(ChatSession)
          .filter(
              ChatSession.id == session_id,
              ChatSession.authority_username == current_user.username
          )
          .first()
    )
    if not session:
        raise HTTPException(404, "Session not found")
    db.delete(session)
    db.commit()
    # 204 No Content – nothing to return
