"""Extract / load layer.

The IBM file is a single flat CSV. To mimic a real warehouse, `build_warehouse`
splits it into four normalised tables in SQLite (customers, accounts,
services, churn_labels) and `load_from_warehouse` reassembles the modelling
table with a SQL join through SQLAlchemy -- the same shape of query you would
run against Snowflake/BigQuery/Postgres in production.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from . import config

TABLES = {
    "customers": [config.ID_COL] + config.DEMOGRAPHIC_COLS,
    "accounts": [config.ID_COL] + config.ACCOUNT_COLS,
    "services": [config.ID_COL] + config.SERVICE_COLS,
    "churn_labels": [config.ID_COL, config.TARGET_COL],
}

JOIN_SQL = """
SELECT c.*,
       a.tenure, a.Contract, a.PaperlessBilling, a.PaymentMethod,
       a.MonthlyCharges, a.TotalCharges,
       s.PhoneService, s.MultipleLines, s.InternetService, s.OnlineSecurity,
       s.OnlineBackup, s.DeviceProtection, s.TechSupport, s.StreamingTV,
       s.StreamingMovies,
       l.Churn
FROM customers c
JOIN accounts a      ON a.customerID = c.customerID
JOIN services s      ON s.customerID = c.customerID
JOIN churn_labels l  ON l.customerID = c.customerID
ORDER BY c.customerID
"""


# Path defaults are None and resolved from `config` at CALL time. A default of
# `config.WAREHOUSE_DB` is frozen when this module is imported, so a caller or
# test that repoints config.WAREHOUSE_DB would silently keep the original file.


def load_raw_csv(path: Path | str | None = None) -> pd.DataFrame:
    """Read the source CSV exactly as delivered (everything except two numeric
    columns arrives as text; TotalCharges has blanks)."""
    # dtype=str for TotalCharges keeps the raw blanks visible to the cleaner.
    return pd.read_csv(path or config.RAW_CSV, dtype={"TotalCharges": str})


def file_sha256(path: Path | str) -> str:
    """Content hash used as the data version in MLflow and the manifest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_warehouse(raw: pd.DataFrame | None = None, db_path: Path | str | None = None) -> Path:
    """Split the flat file into normalised tables and write them to SQLite."""
    raw = load_raw_csv() if raw is None else raw
    db_path = Path(db_path or config.WAREHOUSE_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        for name, cols in TABLES.items():
            raw[cols].to_sql(name, conn, index=False)
    engine.dispose()
    return db_path


class WarehouseIntegrityError(ValueError):
    """The simulated warehouse violates a key or join-cardinality rule."""


def check_warehouse_integrity(conn) -> dict:
    """Key uniqueness, orphan rows and join cardinality. Raises on failure.

    SQLite tables written by `to_sql` carry no PK/FK constraints, so these
    rules are enforced here instead (audit P2-D).
    """
    counts, problems = {}, []
    for table in TABLES:
        n, n_distinct, n_null = conn.execute(
            text(f"SELECT COUNT(*), COUNT(DISTINCT customerID), SUM(customerID IS NULL) FROM {table}")
        ).one()
        counts[table] = n
        if n != n_distinct or n_null:
            problems.append(f"{table}: customerID not unique / has nulls ({n} rows, {n_distinct} distinct)")
    for table in ("accounts", "services", "churn_labels"):
        missing = conn.execute(
            text(
                f"SELECT COUNT(*) FROM customers c LEFT JOIN {table} t ON t.customerID = c.customerID "
                "WHERE t.customerID IS NULL"
            )
        ).scalar()
        orphans = conn.execute(
            text(
                f"SELECT COUNT(*) FROM {table} t LEFT JOIN customers c ON c.customerID = t.customerID "
                "WHERE c.customerID IS NULL"
            )
        ).scalar()
        if missing:
            problems.append(f"{missing} customer(s) missing from {table}")
        if orphans:
            problems.append(f"{orphans} orphan row(s) in {table} with no matching customer")
    if problems:
        raise WarehouseIntegrityError("; ".join(problems))
    return counts


def load_from_warehouse(db_path: Path | str | None = None) -> pd.DataFrame:
    """Rebuild the modelling table from the warehouse with a SQL join."""
    db_path = Path(db_path or config.WAREHOUSE_DB)
    if not db_path.exists():  # SQLite would otherwise create an empty file
        raise FileNotFoundError(f"No warehouse at {db_path}; run etl.build_warehouse() first")
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        counts = check_warehouse_integrity(conn)
        df = pd.read_sql(text(JOIN_SQL), conn)
    engine.dispose()
    if len(df) != counts["customers"]:
        raise WarehouseIntegrityError(f"Join returned {len(df)} rows for {counts['customers']} customers")
    # SQLite stores TotalCharges as TEXT (it has blanks); keep it as string so
    # cleaning is identical whether data came from CSV or the warehouse.
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    return df


def write_data_manifest(path: Path | str | None = None) -> dict:
    """Record what data a model was trained on (lightweight data versioning)."""
    manifest = {
        "raw_csv": str(config.RAW_CSV.relative_to(config.PROJECT_ROOT)),
        "raw_sha256": file_sha256(config.RAW_CSV),
        "rows": int(len(load_raw_csv())),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "IBM Telco Customer Churn sample (CC BY 4.0)",
    }
    Path(path or config.DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
