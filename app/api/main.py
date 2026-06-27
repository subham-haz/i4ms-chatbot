"""FastAPI application exposing the i4MS assistant."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.graph import run_agent, stream_agent
from app.core.config import get_settings
from app.core.logging_config import configure_logging, get_logger
from app.core.schemas import ChatRequest, ChatResponse
from app.observability.langfuse_client import flush, get_langfuse, score_trace
from app.security.input_guard import check_input
from app.security.rbac import AccessContext, Role

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Starting i4MS Assistant (env=%s).", get_settings().environment)
    yield
    flush()
    logger.info("Shutdown complete; flushed Langfuse.")


app = FastAPI(
    title="i4MS Assistant",
    description=(
        "Agentic assistant for the Integrated Minor Mineral Mining Management "
        "System (Odisha). Answers data and policy questions with role-scoped "
        "access, read-only DB guards, prompt-injection detection, Odia language "
        "support, and Langfuse observability."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


def resolve_access_context(
    x_role: str | None,
    x_lessee_id: str | None,
    x_district: str | None,
) -> AccessContext:
    """Resolve the caller's scope from authenticated session headers.

    In production these come from a verified session/JWT issued at login, NOT
    from client-supplied values. The header names here stand in for that.
    """
    role_str = (x_role or "lessee").lower()
    try:
        role = Role(role_str)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=f"Unknown role: {role_str}") from exc

    if role is Role.LESSEE and not x_lessee_id:
        raise HTTPException(status_code=403, detail="Lessee session missing lessee_id.")
    if role is Role.OFFICER and not x_district:
        raise HTTPException(status_code=403, detail="Officer session missing district.")

    return AccessContext(role=role, lessee_id=x_lessee_id, district=x_district)


def _guard_input(query: str) -> None:
    """Run prompt-injection detection; raises HTTPException if blocked."""
    verdict = check_input(query)
    if not verdict.allowed:
        logger.warning("Prompt injection blocked: %s", verdict.reason)
        raise HTTPException(
            status_code=400,
            detail=f"Request blocked by input guard: {verdict.reason}",
        )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "environment": get_settings().environment,
        "langfuse": get_langfuse() is not None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    x_role: str | None = Header(default=None),
    x_lessee_id: str | None = Header(default=None),
    x_district: str | None = Header(default=None),
) -> ChatResponse:
    _guard_input(request.query)
    ctx = resolve_access_context(x_role, x_lessee_id, x_district)
    try:
        return run_agent(
            request.query,
            session_id=request.session_id,
            user_id=request.user_id,
            access_context=ctx,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    x_role: str | None = Header(default=None),
    x_lessee_id: str | None = Header(default=None),
    x_district: str | None = Header(default=None),
) -> StreamingResponse:
    """Stream agent responses as Server-Sent Events (SSE).

    Event types emitted:
      - thinking: agent is processing
      - tool_call: a tool invocation with name and input
      - tool_result: tool finished
      - token: a chunk of the final answer text
      - done: complete ChatResponse as JSON
      - error: failure details
    """
    _guard_input(request.query)
    ctx = resolve_access_context(x_role, x_lessee_id, x_district)
    return StreamingResponse(
        stream_agent(
            request.query,
            session_id=request.session_id,
            user_id=request.user_id,
            access_context=ctx,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class FeedbackRequest(BaseModel):
    trace_id: str
    score: float
    comment: str | None = None


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict:
    """Attach end-user feedback to a Langfuse trace (closes the eval loop)."""
    score_trace(request.trace_id, "user_feedback", request.score, request.comment)
    flush()
    return {"status": "recorded"}
