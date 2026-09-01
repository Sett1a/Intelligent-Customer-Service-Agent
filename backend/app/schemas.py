"""API 请求/响应模型。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SessionCreate(BaseModel):
    title: str | None = None


class SessionUpdate(BaseModel):
    title: str


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    handoff: bool
    created_at: datetime
    updated_at: datetime


class CitationOut(BaseModel):
    source: str = ""
    question: str = ""
    snippet: str = ""
    score: float = 0.0


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    citations: list[CitationOut] | None = None
    created_at: datetime


class ChatRequest(BaseModel):
    session_id: str
    content: str
