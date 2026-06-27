"""Tools available to the i4MS agent.

Two retrieval surfaces:
  - query_i4ms_database: structured questions ("how many active passes for
    lease X", "pending returns in Cuttack") -> safe, scoped text-to-SQL.
  - search_minor_mineral_policy: rule/procedure questions ("how long is an
    e-transit pass valid", "renewal notice period for a quarry lease") -> RAG
    over OMMC 2016 / OMPTS 2007 / SOPs.

The database tool needs an AccessContext. Because LangChain tools take a flat
input, we bind the context via a ContextVar set per request rather than passing
it through the model (the model must never choose whose data it can see).
"""
from __future__ import annotations

from contextvars import ContextVar

from langchain_core.tools import tool

from app.core.logging_config import get_logger
from app.database.text_to_sql import query_database
from app.rag.retriever import hybrid_retrieve
from app.security.rbac import AccessContext, Role

logger = get_logger(__name__)

# Per-request access context. Set by the API layer from the authenticated session.
_ctx_var: ContextVar[AccessContext] = ContextVar(
    "access_context", default=AccessContext(role=Role.ADMIN)
)


def set_access_context(ctx: AccessContext) -> None:
    _ctx_var.set(ctx)


def get_access_context() -> AccessContext:
    return _ctx_var.get()


@tool
def query_i4ms_database(question: str) -> str:
    """Answer questions about specific i4MS records: leases/sairat sources,
    e-permits, e-transit passes, royalty payments, and statutory returns.
    Use for counts, lookups, statuses, totals, and 'which/how many' questions
    about actual data. Do NOT use for questions about rules or procedures."""
    ctx = get_access_context()
    result = query_database(question, ctx)
    if result.get("error"):
        return f"Could not retrieve data: {result['error']}"
    rows = result["rows"]
    if not rows:
        return "No matching records found (within your access scope)."
    header = list(rows[0].keys())
    lines = [" | ".join(header)]
    for r in rows[:50]:
        lines.append(" | ".join(str(r.get(c, "")) for c in header))
    return f"Returned {result['row_count']} row(s):\n" + "\n".join(lines)


@tool
def search_minor_mineral_policy(query: str) -> str:
    """Answer questions about minor-mineral RULES, PROCEDURES, and POLICY:
    OMMC Rules 2016, OMPTS Rules 2007, e-permit/e-transit-pass procedures,
    renewal timelines, royalty/dead-rent provisions, registration SOPs.
    Use for 'how do I', 'what is the rule', 'how long is X valid' questions."""
    chunks = hybrid_retrieve(query)
    if not chunks:
        return "No relevant policy provisions found."
    return "\n\n".join(
        f"[doc:{c.document_id} | source:{c.metadata.get('source', 'n/a')}]\n{c.content}"
        for c in chunks
    )


ALL_TOOLS = [query_i4ms_database, search_minor_mineral_policy]
