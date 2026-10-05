"""FastAPI backend.

POST /session -> {"session_id": "..."}
POST /chat    {"session_id","message"} -> {"reply","citations","cache_hit","latency_ms"}
"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .cache import SemanticCache
from .config import get_settings
from .embeddings import get_embeddings
from .llm import Generator, LLMError
from .retriever import Retriever
from .service import ChatService

_service: ChatService | None = None
_lock = threading.Lock()


def build_service() -> ChatService:
    s = get_settings()
    emb = get_embeddings()
    retriever = Retriever(emb, s.index_dir)
    cache = SemanticCache(emb, s.cache_db, retriever.index_version, s.cache_sim_threshold, s.cache_search_k)
    emb.embed_sym("warm up")  # load the ONNX model now, not on the first student's request
    return ChatService(retriever, cache, Generator(s), s)


def get_service() -> ChatService:
    global _service
    with _lock:
        if _service is None:
            _service = build_service()
        return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_service()
    yield


app = FastAPI(title="NCERT Class 10 Science Chatbot", lifespan=lifespan)


class SessionOut(BaseModel):
    session_id: str


class ChatIn(BaseModel):
    session_id: str
    message: str = Field(min_length=1)


class ChatOut(BaseModel):
    reply: str
    citations: list[str]
    cache_hit: bool
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session", response_model=SessionOut)
def create_session():
    return {"session_id": get_service().new_session()}


@app.post("/chat", response_model=ChatOut)
def chat(req: ChatIn):
    svc = get_service()
    if svc.get_session(req.session_id) is None:
        raise HTTPException(404, "Unknown session_id. Create one with POST /session.")
    if not req.message.strip():
        raise HTTPException(422, "message must not be empty")
    try:
        return svc.chat(req.session_id, req.message)
    except LLMError as e:
        raise HTTPException(502, str(e))


@app.get("/cache/stats")
def cache_stats():
    c = get_service().cache
    return {"entries": len(c), **c.stats}
