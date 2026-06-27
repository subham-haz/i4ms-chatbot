-- i4MS domain schema (Integrated Minor Mineral Mining Management System, Odisha)
-- This mirrors the real entity model: lessees, sairat sources / quarry leases,
-- e-permits, e-transit passes, royalty payments, and statutory returns.
-- In production this maps to the read-replica of the i4MS RDBMS; here it is
-- SQLite so the whole project runs locally without the government database.

-- A lessee/licensee (the regulated party). tenant_key scopes row-level access.
CREATE TABLE IF NOT EXISTS lessee (
    lessee_id        TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    pan              TEXT,                  -- treated as PII, redacted in outputs
    mobile           TEXT,                  -- PII
    district         TEXT NOT NULL,
    category         TEXT NOT NULL,         -- 'Lessee' | 'Licensee' | 'Crusher'
    registered_on    TEXT NOT NULL
);

-- A sairat source / quarry lease (the area from which minor minerals are won).
CREATE TABLE IF NOT EXISTS lease (
    lease_id         TEXT PRIMARY KEY,
    lessee_id        TEXT NOT NULL REFERENCES lessee(lessee_id),
    source_name      TEXT NOT NULL,
    mineral          TEXT NOT NULL,         -- e.g. 'Stone', 'Sand', 'Murrum', 'Gravel'
    district         TEXT NOT NULL,
    tahasil          TEXT NOT NULL,
    area_hectares    REAL NOT NULL,
    lease_status     TEXT NOT NULL,         -- 'Active' | 'Suspended' | 'Expired'
    valid_from       TEXT NOT NULL,
    valid_to         TEXT NOT NULL,
    ec_valid_to      TEXT,                  -- Environmental Clearance validity
    cto_valid_to     TEXT                   -- Consent To Operate validity
);

-- e-Permit (Form L): authorization to remove/dispatch mineral.
CREATE TABLE IF NOT EXISTS e_permit (
    permit_id        TEXT PRIMARY KEY,
    lease_id         TEXT NOT NULL REFERENCES lease(lease_id),
    issued_on        TEXT NOT NULL,
    valid_to         TEXT NOT NULL,
    quantity_cum     REAL NOT NULL,         -- permitted quantity (cubic metres)
    status           TEXT NOT NULL          -- 'Issued' | 'Exhausted' | 'Expired'
);

-- e-Transit Pass: per-vehicle barcoded pass tracking one consignment.
CREATE TABLE IF NOT EXISTS e_transit_pass (
    pass_id          TEXT PRIMARY KEY,
    permit_id        TEXT NOT NULL REFERENCES e_permit(permit_id),
    vehicle_number   TEXT NOT NULL,
    quantity_cum     REAL NOT NULL,
    destination      TEXT NOT NULL,
    generated_at     TEXT NOT NULL,
    valid_until      TEXT NOT NULL,
    status           TEXT NOT NULL          -- 'Active' | 'Used' | 'Expired' | 'Flagged'
);

-- Royalty / fee payments against a lease.
CREATE TABLE IF NOT EXISTS royalty_payment (
    payment_id       TEXT PRIMARY KEY,
    lease_id         TEXT NOT NULL REFERENCES lease(lease_id),
    amount_inr       REAL NOT NULL,
    paid_on          TEXT NOT NULL,
    period           TEXT NOT NULL,         -- e.g. '2026-Q1'
    mode             TEXT NOT NULL          -- 'Online' | 'Challan'
);

-- Statutory monthly/annual returns.
CREATE TABLE IF NOT EXISTS statutory_return (
    return_id        TEXT PRIMARY KEY,
    lease_id         TEXT NOT NULL REFERENCES lease(lease_id),
    period           TEXT NOT NULL,
    filed_on         TEXT,                  -- NULL = not yet filed
    status           TEXT NOT NULL          -- 'Filed' | 'Pending' | 'Overdue'
);

CREATE INDEX IF NOT EXISTS idx_lease_lessee ON lease(lessee_id);
CREATE INDEX IF NOT EXISTS idx_permit_lease ON e_permit(lease_id);
CREATE INDEX IF NOT EXISTS idx_pass_permit ON e_transit_pass(permit_id);
CREATE INDEX IF NOT EXISTS idx_royalty_lease ON royalty_payment(lease_id);
CREATE INDEX IF NOT EXISTS idx_return_lease ON statutory_return(lease_id);
