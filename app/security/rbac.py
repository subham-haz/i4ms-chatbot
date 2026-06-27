"""Role-based access control (RBAC) and tenant scoping.

Different callers see different slices of the database:

  - LESSEE: a regulated party. Sees ONLY their own rows (scoped by lessee_id).
  - OFFICER: a mining circle officer. Sees all leases within their district.
  - ADMIN: directorate-level. Sees everything (audited).

Scoping is applied as a parameterized predicate at the connection layer, NOT by
asking the LLM to add a WHERE clause. This is the difference between a demo and a
system you can put in front of a government auditor: access control cannot depend
on the model behaving.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    LESSEE = "lessee"
    OFFICER = "officer"
    ADMIN = "admin"


@dataclass(frozen=True)
class AccessContext:
    """Who is asking. Resolved from the authenticated session, never from input."""

    role: Role
    lessee_id: str | None = None   # required for LESSEE
    district: str | None = None    # required for OFFICER

    def scope_predicate(self, lessee_alias: str = "lessee") -> tuple[str, list]:
        """Return a (sql_fragment, params) pair to AND into queries.

        The fragment references the lessee table alias so it works for any query
        that joins back to lessee. Callers that don't join lessee must add it.
        """
        if self.role is Role.ADMIN:
            return ("1=1", [])
        if self.role is Role.OFFICER:
            if not self.district:
                raise PermissionError("Officer context missing district.")
            return (f"{lessee_alias}.district = ?", [self.district])
        if self.role is Role.LESSEE:
            if not self.lessee_id:
                raise PermissionError("Lessee context missing lessee_id.")
            return (f"{lessee_alias}.lessee_id = ?", [self.lessee_id])
        raise PermissionError(f"Unknown role: {self.role}")


# Columns considered PII; redacted unless the caller is ADMIN.
PII_COLUMNS = {"pan", "mobile"}


def redact_row(row: dict, ctx: AccessContext) -> dict:
    """Mask PII columns for non-admin callers."""
    if ctx.role is Role.ADMIN:
        return row
    return {
        k: ("****REDACTED****" if k in PII_COLUMNS and v is not None else v)
        for k, v in row.items()
    }
