from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None

class ChatResponse(BaseModel):
    session_id: int
    answer: str

class SessionSummary(BaseModel):
    id: int
    title: Optional[str] = None
    created_at: datetime
    last_message_at: datetime

class ChatMessageResponse(BaseModel):
    role: str
    content: str
    created_at: datetime

class SessionHistory(BaseModel):
    messages: List[ChatMessageResponse]
