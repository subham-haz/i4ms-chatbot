"""Initialize and seed the local i4MS database (stand-in for the production RDBMS)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from app.core.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

DB_PATH = Path("data/i4ms.db")
SCHEMA = Path("app/database/schema.sql")

LESSEES = [
    ("LS001", "Maa Tarini Stone Works", "ABCPN1234A", "9876500001", "Cuttack", "Lessee", "2023-04-10"),
    ("LS002", "Konark Aggregates Pvt Ltd", "ABCPN2234B", "9876500002", "Cuttack", "Crusher", "2023-06-22"),
    ("LS003", "Mahanadi Sand Suppliers", "ABCPN3234C", "9876500003", "Sambalpur", "Lessee", "2024-01-15"),
    ("LS004", "Utkal Minerals", "ABCPN4234D", "9876500004", "Sundargarh", "Licensee", "2024-03-05"),
]

LEASES = [
    ("LE001", "LS001", "Naraj Stone Quarry", "Stone", "Cuttack", "Athagarh", 3.5, "Active", "2023-05-01", "2028-04-30", "2027-03-31", "2026-12-31"),
    ("LE002", "LS002", "Tangi Crusher Source", "Stone", "Cuttack", "Tangi", 2.0, "Active", "2023-07-01", "2028-06-30", "2027-06-30", "2026-08-15"),
    ("LE003", "LS003", "Hirakud Sand Ghat", "Sand", "Sambalpur", "Hirakud", 5.2, "Active", "2024-02-01", "2029-01-31", "2028-01-31", "2027-01-31"),
    ("LE004", "LS003", "Burla Riverbed Source", "Sand", "Sambalpur", "Burla", 4.1, "Suspended", "2024-02-01", "2029-01-31", "2025-12-31", "2025-12-31"),
    ("LE005", "LS004", "Koira Murrum Pit", "Murrum", "Sundargarh", "Koira", 1.8, "Active", "2024-04-01", "2029-03-31", "2028-03-31", "2026-09-30"),
]

PERMITS = [
    ("PM001", "LE001", "2026-06-01", "2026-06-30", 500.0, "Issued"),
    ("PM002", "LE001", "2026-05-01", "2026-05-31", 400.0, "Exhausted"),
    ("PM003", "LE003", "2026-06-10", "2026-07-10", 800.0, "Issued"),
    ("PM004", "LE005", "2026-06-15", "2026-07-15", 300.0, "Issued"),
]

PASSES = [
    ("TP0001", "PM001", "OD05AB1234", 12.0, "Bhubaneswar", "2026-06-20 09:15", "2026-06-20 21:15", "Used"),
    ("TP0002", "PM001", "OD05AB1234", 12.0, "Cuttack", "2026-06-21 10:00", "2026-06-21 22:00", "Active"),
    ("TP0003", "PM001", "OD05CD5678", 10.0, "Jajpur", "2026-06-21 11:30", "2026-06-21 23:30", "Active"),
    ("TP0004", "PM003", "OD15EF9012", 18.0, "Sambalpur", "2026-06-22 08:45", "2026-06-22 20:45", "Active"),
    ("TP0005", "PM003", "OD15EF9012", 18.0, "Bargarh", "2026-06-19 07:00", "2026-06-19 19:00", "Flagged"),
    ("TP0006", "PM004", "OD16GH3456", 9.0, "Rourkela", "2026-06-23 14:20", "2026-06-23 23:59", "Active"),
]

ROYALTIES = [
    ("RP001", "LE001", 125000.0, "2026-04-05", "2026-Q1", "Online"),
    ("RP002", "LE003", 210000.0, "2026-04-10", "2026-Q1", "Online"),
    ("RP003", "LE005", 64000.0, "2026-04-12", "2026-Q1", "Challan"),
]

RETURNS = [
    ("RT001", "LE001", "2026-05", "2026-06-05", "Filed"),
    ("RT002", "LE001", "2026-06", None, "Pending"),
    ("RT003", "LE003", "2026-05", "2026-06-08", "Filed"),
    ("RT004", "LE003", "2026-06", None, "Pending"),
    ("RT005", "LE005", "2026-05", None, "Overdue"),
]


def main() -> None:
    configure_logging()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA.read_text())

    conn.executemany("INSERT INTO lessee VALUES (?,?,?,?,?,?,?)", LESSEES)
    conn.executemany("INSERT INTO lease VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", LEASES)
    conn.executemany("INSERT INTO e_permit VALUES (?,?,?,?,?,?)", PERMITS)
    conn.executemany("INSERT INTO e_transit_pass VALUES (?,?,?,?,?,?,?,?)", PASSES)
    conn.executemany("INSERT INTO royalty_payment VALUES (?,?,?,?,?,?)", ROYALTIES)
    conn.executemany("INSERT INTO statutory_return VALUES (?,?,?,?,?)", RETURNS)
    conn.commit()
    conn.close()
    logger.info("Seeded i4ms.db: %d lessees, %d leases, %d passes.",
                len(LESSEES), len(LEASES), len(PASSES))


if __name__ == "__main__":
    main()
