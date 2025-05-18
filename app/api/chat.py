from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from app.core.security import require_roles, get_current_user
from app.core.database import get_db
from app.models.conversation import ChatSession, ChatMessage
from app.services import rag as rag_svc
from openai import AsyncOpenAI, InternalServerError
import asyncio
import os


router = APIRouter(tags=["Chat"], dependencies=[require_roles("authority")])
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: Optional[float] = 1.0
    session_id: Optional[int] = None

class ChatResponse(BaseModel):
    session_id: int
    answer: str


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
