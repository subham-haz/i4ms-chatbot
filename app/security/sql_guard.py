"""SQL guardrails for a production database agent.

The single most important control when an LLM can query a government production
database: it must NEVER mutate data and must NEVER read across tenant boundaries.
This module validates generated SQL before execution. It is intentionally strict
and fails closed.

Controls enforced:
  1. Read-only: only a single SELECT (or WITH ... SELECT) is allowed.
  2. No multiple statements / no statement stacking (';' chaining).
  3. Denylist of mutating / DDL / admin keywords anywhere in the query.
  4. Mandatory LIMIT injection to bound result size.
  5. Tenant scoping is enforced separately at the connection layer via a
     parameterized predicate, not by trusting the LLM to add a WHERE clause.
"""
from __future__ import annotations

import re

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Keywords that must never appear in a read-only analytical query.
_FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "replace", "merge", "grant", "revoke", "attach", "detach", "pragma",
    "vacuum", "reindex", "commit", "rollback", "savepoint", "exec",
    "execute", "copy", "into",  # 'SELECT ... INTO' is a write
}

_COMMENT_RE = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.DOTALL)
_WORD_RE = re.compile(r"[a-zA-Z_]+")

MAX_ROWS = 200


class UnsafeSQLError(Exception):
    """Raised when generated SQL fails a safety check. Fails closed."""


def _strip_comments(sql: str) -> str:
    return _COMMENT_RE.sub(" ", sql)


def validate_select(sql: str) -> str:
    """Validate that `sql` is a single, read-only SELECT. Returns cleaned SQL.

    Raises UnsafeSQLError on any violation.
    """
    cleaned = _strip_comments(sql).strip().rstrip(";").strip()

    if not cleaned:
        raise UnsafeSQLError("Empty query.")

    # Reject statement stacking (anything after the first statement).
    if ";" in cleaned:
        raise UnsafeSQLError("Multiple statements are not allowed.")

    lowered = cleaned.lower()
    first_word = _WORD_RE.search(lowered)
    if first_word is None or first_word.group(0) not in {"select", "with"}:
        raise UnsafeSQLError("Only SELECT / WITH queries are permitted.")

    # Token-level denylist (so 'updated_at' as a column is fine, 'update' is not).
    tokens = set(_WORD_RE.findall(lowered))
    hits = tokens & _FORBIDDEN
    if hits:
        raise UnsafeSQLError(f"Forbidden keyword(s) in query: {sorted(hits)}")

    return cleaned


def enforce_limit(sql: str, max_rows: int = MAX_ROWS) -> str:
    """Append a LIMIT if the query lacks one, to bound result size."""
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql
    return f"{sql}\nLIMIT {max_rows}"
