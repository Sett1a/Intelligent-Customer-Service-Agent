"""应用入口:uv run uvicorn app.main:app --port 8000 --reload"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.config import CHAT_MODEL, ZHIPU_API_KEY
from app.db import init_db
from app.rag.retriever import collection_count


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="客服 Agent API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "chat_model": CHAT_MODEL,
        "has_key": bool(ZHIPU_API_KEY),
        "kb_docs": collection_count(),
    }
