"""Shared Pydantic schemas."""
from typing import Any, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    session_id: str | None = Field(
        default=None, description="Conversation/session identifier for tracing"
    )
    user_id: str | None = Field(default=None)
    stream: bool = False


class Citation(BaseModel):
    document_id: str
    snippet: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    latency_ms: float | None = None


class EvalSample(BaseModel):
    question: str
    ground_truth: str | None = None
    contexts: list[str] = Field(default_factory=list)
    answer: str | None = None


class EvalResult(BaseModel):
    metric: str
    score: float
    detail: dict[str, Any] = Field(default_factory=dict)


AgentDecision = Literal["retrieve", "calculate", "web_search", "respond"]
