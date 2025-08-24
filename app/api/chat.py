from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Tuple
import os

from app.core.security import require_roles, get_current_user
from app.core.database import get_db
from app.models.conversation import ChatSession, ChatMessage
from app.services import rag as rag_svc
from app.models.schemas_chat import (
    ChatRequest, ChatResponse, SessionSummary, SessionHistory, ChatMessageResponse
)

from openai import AsyncOpenAI, InternalServerError

router = APIRouter(tags=["Chat"], dependencies=[require_roles("authority")])
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def generate_conversation_title(first_message: str) -> str:
    try:
        completion = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": """# Role and Objective
You are a title generator that creates concise, descriptive conversation titles for an urban maintenance assistant chat system.

# Instructions
- Generate a title that captures the main topic or issue from the user's first message
- Keep titles between 20-45 characters for optimal display
- Use title case (capitalize first letter of each major word)
- Focus on the specific problem, location, or request mentioned
- Avoid generic words like "conversation", "chat", "question", or "help"
- Do not include punctuation marks, quotes, or special characters
- Return ONLY the title text, nothing else

# Title Style Guidelines
- For maintenance issues: focus on the problem type and location
- For reports: emphasize what is being reported
- For requests: highlight the specific service needed
- Use clear, actionable language that a city worker would understand

# Examples
## Input: "How do I fix a pothole on Main Street?"
## Output: Pothole Repair Main Street

## Input: "Report water leak at 123 Oak Avenue"
## Output: Water Leak 123 Oak Avenue

## Input: "Street light not working near Central Park"
## Output: Broken Street Light Central Park

## Input: "Graffiti removal request for downtown area"
## Output: Graffiti Removal Downtown

## Input: "Tree branch blocking sidewalk on Elm Street"
## Output: Tree Branch Blocking Elm Street

Generate a title following these guidelines:"""
                },
                {"role": "user", "content": first_message}
            ],
            temperature=0.2,
            max_tokens=25,
        )
        title = completion.choices[0].message.content or "New Conversation"
        return title.strip()[:50]  # Ensure max length and remove whitespace
    except Exception:
        return "New Conversation"


# ---------- Context builder with source tagging ----------
async def _build_context(db: Session, req: ChatRequest, k: int = 12) -> Tuple[str, bool]:

    emb = await rag_svc.embed(req.message)
    chunks = rag_svc.retrieve(db, emb, k=k)

    if not chunks:
        return "NO_SOURCES", False

    docs = []
    for c in chunks:
        docs.append(
            f"[#{c.id}]\n"
            f"<doc id=\"{c.id}\" media_id=\"{c.media_id}\">\n"
            f"{c.chunk}\n"
            f"</doc>"
        )

    ctx = (
        "## EXTERNAL CONTEXT (authoritative)\n"
        "Only use these sources for factual claims. Every non-trivial claim must cite [#ID].\n\n"
        "<external_context>\n" + "\n\n---\n\n".join(docs) + "\n</external_context>"
    )
    return ctx, True

# ---------- Chat endpoint ----------
@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # 1) ensure a session
    is_new_session = False
    if req.session_id:
        session = db.query(ChatSession).filter(ChatSession.id == req.session_id).first()
        if not session:
            raise HTTPException(404, "session_id not found")
    else:
        session = ChatSession(authority_username=current_user.username)
        db.add(session)
        db.commit()
        db.refresh(session)
        is_new_session = True

    db.add(ChatMessage(session_id=session.id, role="user", content=req.message))
    db.commit()
    
    # Generate title for new sessions from the first message
    if is_new_session:
        title = await generate_conversation_title(req.message)
        session.title = title
        db.commit()

    # 2) build strict context (geo-agnostic)
    context_block, has_sources = await _build_context(db, req, k=12)

    history_msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(10)
        .all()[::-1]
    )
    chat_roll = [
        {"role": m.role, "content": m.content}
        for m in history_msgs
        if m.role in ("user", "assistant")
    ]

    # 3) Optimized system rules for GPT-4.1
    system_rules_top = (
        "# Role & Objective\n"
        "You are an urban-maintenance assistant for city authorities.\n\n"
        "# Instructions\n"
        "- Use ONLY the External Context for factual claims; if unsupported, say: "
        "\"Not enough evidence in the sources.\" Do not invent details.\n"
        "- Attach bracket citations [#ID] immediately after each non-trivial claim; "
        "use multiple [#ID] if a point relies on multiple sources.\n"
        "- Stay concise. Professional tone.\n"
        "- Think step-by-step **silently** before answering; DO NOT reveal your internal analysis.\n"
        "- No web browsing, no tools. No policy/legal/medical advice.\n\n"
        "# Output Format (Markdown)\n"
        "#### Answer\n"
        "- Bullet points that directly answer the question, each with [#ID].\n\n"
        "#### Actions\n"
        "- 2–5 concrete next steps for city staff, each with [#ID] if applicable.\n\n"
        "#### Timeline\n"
        "- Bullets like \"within 24h\", \"this week\", \"this quarter\".\n\n"
        "#### Assumptions / Gaps\n"
        "- Any missing info or uncertainties.\n\n"
        "#### Citations\n"
        "- List unique [#ID] used, ascending.\n"
    )

    rules_after_context = (
        "# Final Instructions\n"
        "- Answer ONLY from External Context.\n"
        "- Every substantive statement must include [#ID].\n"
        "- Use the exact Output Format.\n"
        "- If sources are insufficient, say so and still provide Actions/Timeline noting gaps.\n"
    )

    messages = [
        {"role": "system", "content": system_rules_top},
        {"role": "system", "name": "SOURCES", "content": context_block},
        {"role": "system", "content": rules_after_context},
    ] + chat_roll + [
        {"role": "user", "content": f"QUESTION:\n{req.message}"}
    ]

    # 4) Call GPT-4.1
    try:
        completion = await client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.2,
            top_p=0.9,
        )
    except InternalServerError as e:
        raise HTTPException(502, "Upstream LLM error") from e

    raw_answer = completion.choices[0].message.content or ""

    # 5) Fallback if no sources
    if not has_sources:
        raw_answer = (
            "#### Answer\n"
            "- Not enough evidence in the sources to answer. Please add more data to the knowledge base.\n\n"
            "#### Actions\n"
            "- Ingest additional incidents and documents related to this topic.\n"
            "- Broaden retrieval scope by adding more datasets.\n\n"
            "#### Timeline\n"
            "- Collect items today; re-run analysis tomorrow.\n\n"
            "#### Assumptions / Gaps\n"
            "- No matching RAG sources.\n\n"
            "#### Citations\n"
            "- (none)"
        )

    # 6) store assistant reply
    db.add(ChatMessage(session_id=session.id, role="assistant", content=raw_answer))
    db.commit()

    return ChatResponse(session_id=session.id, answer=raw_answer)


# ---------- Session utilities ----------
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
        out.append(
            SessionSummary(
                id=s.id, 
                title=s.title,
                created_at=s.created_at, 
                last_message_at=last
            )
        )
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
            ChatSession.authority_username == current_user.username,
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
                role=m.role, content=m.content, created_at=m.created_at
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
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.authority_username == current_user.username,
        )
        .first()
    )
    if not session:
        raise HTTPException(404, "Session not found")
    db.delete(session)
    db.commit()


@router.patch("/sessions/{session_id}/title")
async def update_session_title(
    session_id: int,
    title: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.authority_username == current_user.username,
        )
        .first()
    )
    if not session:
        raise HTTPException(404, "Session not found")
    
    session.title = title[:50]  # Ensure max length
    db.commit()
    return {"message": "Title updated successfully"}
