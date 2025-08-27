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
import re

router = APIRouter(tags=["Chat"], dependencies=[require_roles("authority")])
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def generate_conversation_title(first_message: str) -> str:
    """
    Generate a concise, punctuation-free, Title Case conversation title.
    Target: 20-45 characters (hard cap 50) using the Responses API.
    """
    try:
        resp = await client.responses.create(
            model="gpt-4.1",
            # System-level guidance lives in `instructions`
            instructions="""
# Role & Objective
You generate concise, descriptive conversation titles for an urban maintenance assistant.

# Instructions
- Capture the main topic/issue from the user's first message
- Target 20-45 characters (hard cap 50 applied by caller)
- Use Title Case
- No punctuation or special characters
- Avoid generic words: conversation, chat, question, help
- Return ONLY the title text, no quotes or extra text

# Style Guidelines
- Maintenance issues: focus on problem type + location
- Reports: emphasize what is being reported
- Requests: highlight the specific service needed
- Use clear, actionable language for city workers

# Examples
- Input: "How do I fix a pothole on Main Street?"
  Output: Pothole Repair Main Street
- Input: "Report water leak at 123 Oak Avenue"
  Output: Water Leak 123 Oak Avenue
- Input: "Street light not working near Central Park"
  Output: Broken Street Light Central Park
- Input: "Graffiti removal request for downtown area"
  Output: Graffiti Removal Downtown
- Input: "Tree branch blocking sidewalk on Elm Street"
  Output: Tree Branch Blocking Elm Street

# Final Instruction
Return a single line with only the title. No punctuation. No quotes.
""".strip(),
            # For this simple case, pass the user input as a plain string
            input=first_message,
            temperature=0.2,
            max_output_tokens=30,   # Slightly higher to accommodate 45-char target
            # store=True is default - enables OpenAI dashboard logging
        )

        raw = (resp.output_text or "").strip()
        if not raw:
            return "New Conversation"

        title = raw.splitlines()[0].strip().strip('"\'')
        title = re.sub(r'[^\w\s\u00C0-\u024F]', ' ', title)
        title = re.sub(r'_', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        title = title.title()
        title = title[:50].strip()

        return title or "New Conversation"

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

    # 3) Optimized system rules for GPT-4.1 (top-level behavioral contract)
    system_rules_top = (
        "# Role & Objective\n"
        "You are an urban-maintenance assistant for city authorities.\n\n"
        "# Instructions\n"
        "- Use ONLY the External Context for factual claims; if unsupported, say "
        "\"Not enough evidence in the sources.\" Do not invent details.\n"
        "- Attach bracket citations [#ID] immediately after each non-trivial claim; "
        "use multiple [#ID] if a point relies on multiple sources.\n"
        "- Stay concise. Professional tone.\n"
        "- Think step-by-step internally before answering; do NOT expose this reasoning.\n"
        "- No web browsing, no tools. No policy/legal/medical advice.\n\n"
        "# Internal Reasoning Strategy (do privately)\n"
        "1. Identify what information the question needs\n"
        "2. Scan context documents for relevant evidence\n"
        "3. Synthesize findings with proper citations\n"
        "4. Formulate actionable next steps\n"
        "5. Note any information gaps\n\n"
        "# Output Format (strict Markdown)\n"
        "#### Answer\n"
        "- Bullet points that directly answer the question, each with [#ID]\n\n"
        "#### Actions\n"
        "- 2-5 concrete next steps for city staff, each with [#ID] if applicable\n\n"
        "#### Timeline\n"
        "- Bullets like \"within 24h\", \"this week\", \"this quarter\"\n\n"
        "#### Assumptions / Gaps\n"
        "- Any missing info or uncertainties\n\n"
        "#### Citations\n"
        "- List unique [#ID] used, ascending\n"
    )

    # repeat key reminders after the long context (guide: instructions at both top & bottom)
    rules_after_context = (
        "# Final Instructions\n"
        "- Answer ONLY from External Context\n"
        "- Every substantive statement must include [#ID]\n"
        "- Use the exact Output Format shown above\n"
        "- If sources are insufficient, acknowledge gaps but still provide Actions/Timeline\n"
        "- Output only the requested sections—no reasoning or prefaces\n"
    )

    # 4) Call GPT-4.1 via Responses API (stateless)
    try:
        resp = await client.responses.create(
            model="gpt-4.1",
            instructions=system_rules_top,   # top-level behavior
            input=[
                # long context first…
                {"role": "system", "content": context_block},
                # …then repeat the key reminders after the context
                {"role": "system", "content": rules_after_context},
                # recent conversation turns
                *chat_roll,
                # the current user question
                {"role": "user", "content": f"QUESTION:\n{req.message}"},
            ],
            temperature=0.2,
            # store=True is default - enables OpenAI dashboard logging
        )
    except InternalServerError as e:
        raise HTTPException(502, "Upstream LLM error") from e

    raw_answer = (getattr(resp, "output_text", None) or "").strip()

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
