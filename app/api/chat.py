from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Tuple, Optional, AsyncIterator
import os
import json
import asyncio

from app.core.security import require_roles, get_current_user
from app.core.database import get_db
from app.core.datetime_utils import format_datetime_ro
from app.models.conversation import ChatSession, ChatMessage
from app.services import rag as rag_svc
from app.models.schemas_chat import (
    ChatRequest, SessionSummary, SessionHistory, ChatMessageResponse
)

from openai import AsyncOpenAI, InternalServerError
import re
import uuid
from app.services.csv_export import prepare_csv_data, store_csv_data

router = APIRouter(tags=["Chat"], dependencies=[require_roles("authority")])
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _sse(event: str, data: dict | str) -> str:
    """Format a single SSE frame."""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\n" f"data: {payload}\n\n"


async def generate_conversation_title(first_message: str) -> str:

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
            input=first_message,
            temperature=0.2,
            max_output_tokens=30,
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


async def _build_context(
    db: Session, req: ChatRequest, request: Request, k: int = 12
) -> Tuple[str, bool, Optional[str], Optional[int]]:
    from app.services.rag_query_parser import parse_query_with_filters

    parsed = await parse_query_with_filters(req.message)
    skip_semantic = parsed.get("sql_only", False)

    emb = []
    if not skip_semantic:
        emb = await rag_svc.embed(parsed["query"])

    chunks = rag_svc.retrieve(
        db, emb, k=k,
        query_text=parsed["query"],
        severity_filter=parsed.get("severity"),
        status_filter=parsed.get("status"),
        assigned_to_filter=parsed.get("assigned_to"),
        verified_by_filter=parsed.get("verified_by"),
        resolved_after=parsed.get("resolved_after"),
        resolved_before=parsed.get("resolved_before"),
        verified_after=parsed.get("verified_after"),
        verified_before=parsed.get("verified_before"),
        skip_semantic=skip_semantic,
    )

    if not chunks:
        return "NO_SOURCES", False, None, None

    total_chunks = len(chunks)
    csv_url: Optional[str] = None

    if skip_semantic and total_chunks > 10:
        csv_id = str(uuid.uuid4())
        csv_data = prepare_csv_data(chunks, request)
        store_csv_data(csv_id, csv_data)
        root_path = request.scope.get("root_path", "").rstrip("/")
        path = request.app.url_path_for("rag_download_csv", csv_id=csv_id)
        csv_url = f"{str(request.base_url).rstrip('/')}{root_path}{path}"
        chunks = chunks[:10]

    # 3) Server-declared filters note (authoritative)
    def _val(v, enum=False):
        if v is None:
            return "null"
        return v.value if enum else v

    filters_note = (
        "## FILTERS APPLIED (server)\n"
        f"semantic_query={parsed['query']}\n"
        f"severity={_val(parsed.get('severity'), enum=True)}\n"
        f"status={_val(parsed.get('status'), enum=True)}\n"
        f"assigned_to={_val(parsed.get('assigned_to'))}\n"
        f"verified_by={_val(parsed.get('verified_by'))}\n"
        f"resolved_after={_val(parsed.get('resolved_after'))}\n"
        f"resolved_before={_val(parsed.get('resolved_before'))}\n"
        f"verified_after={_val(parsed.get('verified_after'))}\n"
        f"verified_before={_val(parsed.get('verified_before'))}\n"
    )

    # Add truncation note if applicable
    if skip_semantic and total_chunks > 10:
        filters_note += f"\nNOTE: Showing first 10 of {total_chunks} total results. Full dataset available via CSV export.\n"

    # 4) Build authoritative docs with metadata
    docs = []
    for c in chunks:
        docs.append(
            f"""[#{c.id}]
<doc id="{c.id}" media_id="{c.media_id}">
<metadata>
status: {c.status.value if c.status else "unknown"}
severity: {c.severity.value if c.severity else "unknown"}
assigned_to: {c.assigned_to or "null"}
verified_by: {c.verified_by or "null"}
resolved_at: {format_datetime_ro(c.resolved_at) or "null"}
verified_at: {format_datetime_ro(c.verified_at) or "null"}
class_name: {c.class_name or "unknown"}
address: {c.address or "unknown"}
</metadata>

<content>
{c.chunk}
</content>
</doc>"""
        )

    ctx = (
            filters_note
            + "\n"
            + "## EXTERNAL CONTEXT (authoritative)\n"
              "Only use these sources for factual claims. Every non-trivial claim must cite [#ID].\n\n"
              "<external_context>\n"
            + "\n\n---\n\n".join(docs)
            + "\n</external_context>"
    )
    return ctx, True, csv_url, total_chunks


# ---------- Streaming chat endpoint ----------
@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    user_text = req.message
    if not user_text.strip():
        raise HTTPException(400, "message is required")

    is_new_session = False
    session = None
    session_id = req.session_id
    if session_id:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise HTTPException(404, "session_id not found")
    else:
        session = ChatSession(authority_username=current_user.username)
        db.add(session); db.commit(); db.refresh(session)
        is_new_session = True

    # persist user message
    db.add(ChatMessage(session_id=session.id, role="user", content=user_text))
    db.commit()

    # title for new sessions
    if is_new_session:
        title = await generate_conversation_title(user_text)
        session.title = title
        db.commit()

    context_block, has_sources, csv_url, total_count = await _build_context(
        db, ChatRequest(message=user_text), request, k=12
    )

    history_msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(10)
        .all()[::-1]
    )
    chat_roll = [{"role": m.role, "content": m.content} for m in history_msgs if m.role in ("user","assistant")]

    system_rules_top = (
        "# Role & Objective\n"
        "You are an urban-maintenance assistant for city authorities.\n\n"
        "# Instructions\n"
        "- Use ONLY the External Context for factual claims; if unsupported, say "
        "\"Not enough evidence in the sources.\" Do not invent details.\n"
        "- Treat <metadata> inside each <doc> as authoritative facts about that source "
        "(status, severity, assignee, verification, dates, class_name, address). "
        "You may state these with a [#ID] citation to that doc.\n"
        "- You may summarize any property explicitly declared in \"FILTERS APPLIED (server)\" "
        "(e.g., \"All items are resolved\"). Still cite at least one [#ID] from the set when making such a statement.\n"
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
        "- For each matching issue, output: **Class** — **Address** — **Status**"
        " (add \"Resolved on <date>\" if available, using the human-readable format from metadata) [#ID]\n"
        "- Dates should be displayed as provided in metadata (e.g., \"27 August 2025, 18:40\")\n"
        "- If giving an overall summary (e.g., \"All are resolved\"), include a citation to at least one representative [#ID].\n\n"
        "#### Actions\n"
        "- 2-5 concrete next steps for city staff, each with [#ID] if applicable\n\n"
        "#### Timeline\n"
        "- Bullets like \"within 24h\", \"this week\", \"this quarter\"\n\n"
        "#### Assumptions / Gaps\n"
        "- Any missing info or uncertainties\n\n"
        "#### Citations\n"
        "- List unique [#ID] used, ascending\n"
    )

    rules_after_context = (
        "# Final Instructions\n"
        "- Answer ONLY from External Context\n"
        "- Treat <metadata> as authoritative for status/severity/assignee/verification/dates/class/address\n"
        "- Every substantive statement must include [#ID]\n"
        "- Use the exact Output Format shown above\n"
        "- If sources are insufficient, acknowledge gaps but still provide Actions/Timeline\n"
        "- Output only the requested sections—no reasoning or prefaces\n"
    )

    async def eventgen() -> AsyncIterator[bytes]:
        full_text = []
        try:
            # OpenAI Responses API streaming
            stream = await client.responses.create(
                model="gpt-4.1",
                instructions=system_rules_top,
                input=[
                    {"role": "system", "content": context_block},
                    {"role": "system", "content": rules_after_context},
                    *chat_roll,
                    {"role": "user", "content": f"QUESTION:\n{user_text}"},
                ],
                temperature=0.2,
                stream=True,  # <-- key difference
            )

            # initial priming event (optional)
            yield _sse("session", {"session_id": session.id}).encode("utf-8")

            # stream events as they arrive
            async for event in stream:
                et = getattr(event, "type", None)

                # Text token delta
                if et == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        full_text.append(delta)
                        yield _sse("delta", {"text": delta}).encode("utf-8")

                # Completed — finalize & persist
                elif et == "response.completed":
                    # Text that has already been streamed from OpenAI
                    base = "".join(full_text)
                    emitted_any = bool(base.strip())

                    # If there were no sources and the model produced nothing,
                    # build the fallback response AND STREAM it now.
                    if not has_sources and not emitted_any:
                        fallback = (
                            "#### Answer\n"
                            "- Not enough evidence in the sources to answer. Please add more data.\n\n"
                            "#### Actions\n"
                            "- Ingest additional incidents and documents.\n"
                            "- Broaden retrieval scope by adding more datasets.\n\n"
                            "#### Timeline\n"
                            "- Collect items today; re-run analysis tomorrow.\n\n"
                            "#### Assumptions / Gaps\n"
                            "- No matching RAG sources.\n\n"
                            "#### Citations\n"
                            "- (none)"
                        )
                        # stream the entire fallback now
                        yield _sse("delta", {"text": fallback}).encode("utf-8")
                        full_text.append(fallback)
                        base = fallback
                        emitted_any = True

                    # Build the extra tail (CSV note / link) and STREAM it too
                    tail = ""
                    if csv_url:
                        if total_count and total_count > 10:
                            tail = (
                                f"\n\n> **Note:** Showing first 10 of {total_count} results. "
                                f"[Download full CSV]({csv_url})"
                            )
                        else:
                            tail = f"\n\n[Download full CSV]({csv_url})"

                    if tail:
                        # keep in-memory buffer and the client in sync
                        full_text.append(tail)
                        yield _sse("delta", {"text": tail}).encode("utf-8")

                    # Persist exactly what the user saw
                    final = "".join(full_text).strip()
                    db.add(ChatMessage(session_id=session.id, role="assistant", content=final))
                    db.commit()

                    # Finish the stream
                    yield _sse("done", {"session_id": session.id}).encode("utf-8")
                    return

                # Forward errors
                elif et == "error":
                    # Some SDKs deliver errors as 'error' with 'message'
                    msg = getattr(event, "message", "unknown error")
                    yield _sse("error", {"message": msg}).encode("utf-8")
                    return

                # Ignore other event types, or forward generically if you want:
                # else:
                #     yield _sse("event", json.loads(event.model_dump_json())).encode("utf-8")

        except InternalServerError:
            yield _sse("error", {"message": "Upstream LLM error"}).encode("utf-8")
        except Exception as e:
            yield _sse("error", {"message": str(e)}).encode("utf-8")

    return StreamingResponse(
        eventgen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
