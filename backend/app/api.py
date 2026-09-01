"""API 路由:会话 CRUD + SSE 流式对话。"""

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import SupportDeps, agent
from app.config import HISTORY_MAX_MESSAGES, ZHIPU_API_KEY
from app.db import get_db, session_by_id
from app.models import MessageModel, SessionModel
from app.rag.embeddings import build_embed_model
from app.rag.retriever import get_retriever
from app.schemas import ChatRequest, MessageOut, SessionCreate, SessionOut, SessionUpdate

router = APIRouter(prefix="/api")


# ---------- 会话 CRUD ----------


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    rows = (
        await db.execute(select(SessionModel).order_by(SessionModel.updated_at.desc()))
    ).scalars().all()
    return rows


@router.post("/sessions", response_model=SessionOut, status_code=201)
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    s = SessionModel(id=str(uuid.uuid4()), title=(body.title or "新会话").strip() or "新会话")
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return s


@router.patch("/sessions/{session_id}", response_model=SessionOut)
async def rename_session(session_id: str, body: SessionUpdate, db: AsyncSession = Depends(get_db)):
    s = await session_by_id(db, session_id)
    if not s:
        raise HTTPException(404, "会话不存在")
    s.title = body.title.strip()[:200] or s.title
    s.updated_at = datetime.now()
    await db.commit()
    await db.refresh(s)
    return s


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    s = await session_by_id(db, session_id)
    if not s:
        raise HTTPException(404, "会话不存在")
    await db.execute(delete(MessageModel).where(MessageModel.session_id == session_id))
    await db.delete(s)
    await db.commit()
    return None


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
async def list_messages(session_id: str, db: AsyncSession = Depends(get_db)):
    if not await session_by_id(db, session_id):
        raise HTTPException(404, "会话不存在")
    rows = (
        await db.execute(select(MessageModel).where(MessageModel.session_id == session_id).order_by(MessageModel.id))
    ).scalars().all()
    return rows


# ---------- 对话(SSE) ----------


def _to_history(rows: list[MessageModel]):
    """DB 消息行 → Pydantic AI message_history。"""
    history = []
    for r in rows:
        if r.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=r.content)]))
        else:
            history.append(ModelResponse(parts=[TextPart(content=r.content)]))
    return history


def _dedupe_citations(citations: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for c in citations:
        key = (c.get("source", ""), c.get("question", ""))
        if key not in seen or c.get("score", 0) > seen[key].get("score", 0):
            seen[key] = c
    return sorted(seen.values(), key=lambda c: c.get("score", 0), reverse=True)[:6]


def _sse(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _chat_stream(db: AsyncSession, session: SessionModel, content: str):
    # 先取历史(不含本轮),再落库用户消息
    rows = (
        await db.execute(
            select(MessageModel)
            .where(MessageModel.session_id == session.id)
            .order_by(MessageModel.id)
        )
    ).scalars().all()
    history = _to_history(rows[-HISTORY_MAX_MESSAGES:])

    db.add(MessageModel(session_id=session.id, role="user", content=content))
    is_first_exchange = not rows
    await db.commit()

    if is_first_exchange and session.title in ("", "新会话"):
        session.title = content.strip()[:20]
        await db.commit()

    if not ZHIPU_API_KEY:
        yield _sse("error", {"message": "未配置 ZHIPU_API_KEY:请编辑 backend/.env 后重启后端"})
        return

    try:
        retriever = get_retriever()
    except RuntimeError as e:
        yield _sse("error", {"message": str(e)})
        return

    deps = SupportDeps(retriever=retriever)
    try:
        async with agent.run_stream(
            content, deps=deps, message_history=history
        ) as result:
            async for delta in result.stream_text(delta=True):
                if delta:
                    yield _sse("delta", {"text": delta})
            final_text = await result.get_output()
    except Exception as e:  # noqa: BLE001 - SSE 通道内统一转错误事件
        yield _sse("error", {"message": f"模型调用失败:{e}"})
        return

    citations = _dedupe_citations(deps.citations)
    db.add(
        MessageModel(
            session_id=session.id,
            role="assistant",
            content=final_text or "",
            citations=citations or None,
        )
    )
    if deps.escalated:
        session.handoff = True
    session.updated_at = datetime.now()
    await db.commit()
    await db.refresh(session)

    if citations:
        yield _sse("citations", {"citations": citations})
    yield _sse(
        "done",
        {"escalated": deps.escalated, "handoff": session.handoff, "title": session.title},
    )


@router.post("/chat")
async def chat(body: ChatRequest, db: AsyncSession = Depends(get_db)):
    session = await session_by_id(db, body.session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    content = body.content.strip()
    if not content:
        raise HTTPException(400, "消息内容不能为空")
    return StreamingResponse(
        _chat_stream(db, session, content),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
