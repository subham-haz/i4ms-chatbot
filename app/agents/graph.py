"""LangGraph agent: a tool-calling ReAct loop with conditional routing.

Graph shape:

    START -> agent -> (tools? )-> tools -> agent -> ... -> END

The agent node decides whether to call a tool or produce a final answer.
A `should_continue` edge routes back to tools while tool calls are pending,
otherwise terminates. Langfuse traces the whole run via the callback handler.

Conversation memory is maintained per session using LangGraph's MemorySaver
checkpointer. Each unique session_id gets its own thread of message history,
enabling multi-turn follow-up questions.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agents.prompts import AGENT_SYSTEM_PROMPT
from app.agents.tools import ALL_TOOLS, set_access_context
from app.core.config import get_settings
from app.core.language import detect_language, translate_from_english, translate_to_english
from app.core.logging_config import get_logger
from app.core.schemas import ChatResponse, Citation
from app.observability.langfuse_client import get_callback_handler
from app.security.rbac import AccessContext

logger = get_logger(__name__)

_checkpointer = MemorySaver()


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@lru_cache
def _get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    ).bind_tools(ALL_TOOLS)


def _agent_node(state: AgentState) -> dict:
    response = _get_llm().invoke(state["messages"])
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


@lru_cache
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    compiled = graph.compile(checkpointer=_checkpointer)
    logger.info("Agent graph compiled with MemorySaver checkpointer.")
    return compiled


_CITATION_RE = re.compile(r"\[doc:([^\s|\]]+)")


def _extract_citations(messages: list[AnyMessage]) -> list[Citation]:
    """Pull cited doc ids from tool outputs that the final answer references."""
    citations: dict[str, Citation] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            for doc_id in _CITATION_RE.findall(content):
                snippet = content[:200]
                citations.setdefault(
                    doc_id, Citation(document_id=doc_id, snippet=snippet)
                )
    return list(citations.values())


def _tools_used(messages: list[AnyMessage]) -> list[str]:
    used: list[str] = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", []) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name and name not in used:
                used.append(name)
    return used


def _build_config(
    session_id: str | None, user_id: str | None
) -> dict:
    config: dict = {"configurable": {"thread_id": session_id or "default"}}
    handler = get_callback_handler()
    if handler is not None:
        config["callbacks"] = [handler]
        config["metadata"] = {
            "langfuse_session_id": session_id,
            "langfuse_user_id": user_id,
            "langfuse_tags": ["i4ms-assistant"],
        }
    return config


def run_agent(
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    access_context: "AccessContext | None" = None,
) -> ChatResponse:
    start = time.perf_counter()
    agent = build_agent()

    if access_context is not None:
        set_access_context(access_context)

    detected_lang = detect_language(query)
    processing_query = translate_to_english(query, detected_lang)

    config = _build_config(session_id, user_id)

    init = {
        "messages": [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=processing_query),
        ]
    }
    result = agent.invoke(init, config=config)
    messages = result["messages"]
    answer = str(messages[-1].content)

    answer = translate_from_english(answer, detected_lang)

    latency = (time.perf_counter() - start) * 1000

    trace_id = None
    handler = get_callback_handler()
    if handler is not None:
        try:
            from langfuse import get_client

            trace_id = get_client().get_current_trace_id()
        except Exception:
            trace_id = None

    return ChatResponse(
        answer=answer,
        citations=_extract_citations(messages),
        tools_used=_tools_used(messages),
        trace_id=trace_id,
        latency_ms=round(latency, 2),
    )


async def stream_agent(
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    access_context: "AccessContext | None" = None,
) -> AsyncGenerator[str, None]:
    """Yield Server-Sent Events as the agent processes a query.

    Event types:
      - thinking: agent is deciding which tool to call
      - tool_call: a tool is being invoked (name + args)
      - token: a chunk of the final answer
      - done: final assembled response as JSON
      - error: something went wrong
    """
    agent = build_agent()

    if access_context is not None:
        set_access_context(access_context)

    detected_lang = detect_language(query)
    processing_query = translate_to_english(query, detected_lang)

    config = _build_config(session_id, user_id)
    init = {
        "messages": [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=processing_query),
        ]
    }

    start = time.perf_counter()
    answer_chunks: list[str] = []
    all_messages: list[AnyMessage] = []

    try:
        yield _sse("thinking", {"status": "processing query..."})

        async for event in agent.astream_events(init, config=config, version="v2"):
            kind = event.get("event", "")

            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    answer_chunks.append(str(chunk.content))
                    yield _sse("token", {"content": str(chunk.content)})

            elif kind == "on_tool_start":
                name = event.get("name", "unknown")
                tool_input = event.get("data", {}).get("input", {})
                yield _sse("tool_call", {"tool": name, "input": tool_input})

            elif kind == "on_tool_end":
                yield _sse("tool_result", {"tool": event.get("name", "")})

            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output", {})
                msgs = output.get("messages", [])
                all_messages = msgs

        full_answer = "".join(answer_chunks) or (
            str(all_messages[-1].content) if all_messages else "No response generated."
        )

        full_answer = translate_from_english(full_answer, detected_lang)
        latency = (time.perf_counter() - start) * 1000

        response = ChatResponse(
            answer=full_answer,
            citations=_extract_citations(all_messages),
            tools_used=_tools_used(all_messages),
            latency_ms=round(latency, 2),
        )
        yield _sse("done", response.model_dump())

    except Exception as exc:
        logger.exception("stream_agent failed")
        yield _sse("error", {"detail": str(exc)})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
