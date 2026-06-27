"""Text-to-SQL: turn a natural-language question into a safe, scoped query.

Flow:
  question --LLM--> candidate SQL --guard--> validated SELECT
          --scope injection--> tenant-predicated SQL --execute--> rows

Tenant scoping is injected here in code (not by the LLM). We wrap the model's
SELECT as a subquery and apply the access predicate on the outer query, so the
caller can never widen their own scope regardless of what SQL the model writes.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.database.connection import get_schema_description, run_scoped_query
from app.observability.langfuse_client import trace_span
from app.security.rbac import AccessContext
from app.security.sql_guard import UnsafeSQLError, validate_select

logger = get_logger(__name__)

_SQL_SYSTEM = """You are a careful SQL analyst for the i4MS minor-minerals database.
Generate exactly ONE read-only SQLite SELECT query that answers the question.

Rules:
- SELECT only. Never write INSERT/UPDATE/DELETE/DDL.
- Always join through lease -> lessee when querying leases/permits/passes so the
  result includes lessee_id and district (needed for access control).
- Prefer explicit column lists over SELECT *.
- Return ONLY the SQL, no markdown fences, no commentary.

Schema:
{schema}
"""


@lru_cache
def _sql_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model, temperature=0.0, api_key=settings.openai_api_key
    )


def _generate_sql(question: str) -> str:
    prompt = _SQL_SYSTEM.format(schema=get_schema_description())
    raw = _sql_llm().invoke(
        [("system", prompt), ("user", question)]
    ).content
    sql = str(raw).strip().removeprefix("```sql").removeprefix("```").removesuffix("```")
    return sql.strip()


def _inject_scope(validated_sql: str, ctx: AccessContext) -> tuple[str, list]:
    """Wrap the validated SELECT and apply the tenant predicate on the outside.

    Requires the inner query to expose `lessee_id` and `district` columns; the
    system prompt instructs the model to include them. If they're absent the
    outer predicate fails closed (no rows / error), which is the safe direction.
    """
    predicate, params = ctx.scope_predicate(lessee_alias="scoped")
    wrapped = (
        f"SELECT * FROM (\n{validated_sql}\n) AS scoped\nWHERE {predicate}"
    )
    return wrapped, params


def query_database(question: str, ctx: AccessContext) -> dict:
    """Public entry point used by the agent tool."""
    with trace_span(
        "text_to_sql",
        input={"question": question},
        metadata={"role": ctx.role.value},
    ) as span:
        try:
            candidate = _generate_sql(question)
            validated = validate_select(candidate)
            scoped_sql, params = _inject_scope(validated, ctx)
            rows = run_scoped_query(scoped_sql, ctx, params)
            result = {"sql": candidate, "rows": rows, "row_count": len(rows)}
        except UnsafeSQLError as exc:
            logger.warning("Blocked unsafe SQL: %s", exc)
            result = {"error": f"Query blocked by safety guard: {exc}", "rows": []}
        except Exception as exc:  # noqa: BLE001
            logger.exception("text_to_sql failed")
            result = {"error": str(exc), "rows": []}

        if span is not None:
            span.update(output={k: v for k, v in result.items() if k != "rows"})
        return result
