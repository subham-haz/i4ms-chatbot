"""Langfuse observability layer.

Provides a single source of truth for the Langfuse client plus thin helpers
so the rest of the codebase never imports Langfuse directly. This keeps tracing
optional (degrades gracefully when keys are absent) and testable.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

try:
    from langfuse import Langfuse, get_client
    from langfuse.langchain import CallbackHandler

    _LANGFUSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    Langfuse = None  # type: ignore
    get_client = None  # type: ignore
    CallbackHandler = None  # type: ignore
    _LANGFUSE_AVAILABLE = False


@lru_cache
def get_langfuse() -> Any | None:
    """Return a configured Langfuse client, or None when disabled/unavailable."""
    settings = get_settings()
    if not settings.langfuse_enabled:
        logger.info("Langfuse disabled via config.")
        return None
    if not _LANGFUSE_AVAILABLE:
        logger.warning("langfuse package not installed; tracing is a no-op.")
        return None
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.warning("Langfuse keys missing; tracing is a no-op.")
        return None

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        environment=settings.environment,
    )
    logger.info("Langfuse client initialized (host=%s).", settings.langfuse_host)
    return client


def get_callback_handler() -> Any | None:
    """LangChain/LangGraph callback handler that streams spans to Langfuse."""
    if not _LANGFUSE_AVAILABLE or get_langfuse() is None:
        return None
    return CallbackHandler()


@contextmanager
def trace_span(
    name: str,
    *,
    input: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Lightweight span context manager that no-ops without Langfuse."""
    client = get_langfuse()
    if client is None:
        yield None
        return

    with client.start_as_current_span(name=name, input=input, metadata=metadata) as span:
        try:
            yield span
        except Exception as exc:  # record errors on the span
            span.update(level="ERROR", status_message=str(exc))
            raise


def score_trace(
    trace_id: str, name: str, value: float, comment: str | None = None
) -> None:
    """Attach a numeric score (e.g. eval metric, user feedback) to a trace."""
    client = get_langfuse()
    if client is None:
        return
    client.create_score(trace_id=trace_id, name=name, value=value, comment=comment)


def flush() -> None:
    client = get_langfuse()
    if client is not None:
        client.flush()
