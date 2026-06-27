"""Database access layer with read-only enforcement and tenant scoping.

In production this points at a READ REPLICA of the i4MS RDBMS using a DB role
that has been granted SELECT only (defense in depth: even if the SQL guard were
bypassed, the database itself would reject writes). Here we use SQLite with a
query-time guard so the project runs locally.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from app.core.logging_config import get_logger
from app.security.rbac import AccessContext, redact_row
from app.security.sql_guard import enforce_limit, validate_select

logger = get_logger(__name__)

DB_PATH = Path("data/i4ms.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Defense in depth: open the connection read-only at the SQLite level too.
    conn.execute("PRAGMA query_only = ON;")
    return conn


def run_scoped_query(
    sql: str, ctx: AccessContext, params: list | None = None
) -> list[dict]:
    """Validate, scope, bound, execute. Returns redacted rows.

    The caller's SQL is expected to be a SELECT that joins (or can join) the
    lessee table so the tenant predicate applies. The predicate is ANDed in by
    the agent tool layer; here we validate and execute.
    """
    safe_sql = enforce_limit(validate_select(sql))
    params = params or []

    with get_connection() as conn:
        cursor = conn.execute(safe_sql, params)
        rows = [dict(r) for r in cursor.fetchall()]

    redacted = [redact_row(r, ctx) for r in rows]
    logger.info("Query returned %d rows (role=%s).", len(redacted), ctx.role.value)
    return redacted


def get_schema_description() -> str:
    """Human-readable schema passed to the LLM for text-to-SQL grounding."""
    return (
        "Tables (SQLite):\n"
        "lessee(lessee_id, name, pan, mobile, district, category, registered_on)\n"
        "lease(lease_id, lessee_id, source_name, mineral, district, tahasil, "
        "area_hectares, lease_status, valid_from, valid_to, ec_valid_to, cto_valid_to)\n"
        "e_permit(permit_id, lease_id, issued_on, valid_to, quantity_cum, status)\n"
        "e_transit_pass(pass_id, permit_id, vehicle_number, quantity_cum, "
        "destination, generated_at, valid_until, status)\n"
        "royalty_payment(payment_id, lease_id, amount_inr, paid_on, period, mode)\n"
        "statutory_return(return_id, lease_id, period, filed_on, status)\n\n"
        "Notes:\n"
        "- 'sairat source' and 'quarry lease' both refer to the lease table.\n"
        "- e-Transit Pass is the per-vehicle pass; e-Permit (Form L) authorizes dispatch.\n"
        "- Always join through lease -> lessee so access scoping can apply.\n"
        "- Dates are ISO strings 'YYYY-MM-DD'. Money is INR in amount_inr.\n"
    )
