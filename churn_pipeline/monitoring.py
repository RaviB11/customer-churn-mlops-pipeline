"""Monitoring: input schema checks, prediction logging and drift detection.

* `validate_records`  -- hard errors (missing columns, bad numbers) vs soft
                         warnings (a category the model never saw, e.g. a new
                         payment method). Warnings still get scored.
* `PredictionLogger`  -- appends one JSON line per prediction (timestamp,
                         request id, model version, inputs, score, flag).
* `drift_report`      -- compares logged traffic with the training profile
                         using the Population Stability Index (PSI).

PSI rule of thumb (an industry convention, not a law): < 0.10 stable,
0.10-0.25 moderate shift, > 0.25 significant shift worth investigating.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
_EPS = 1e-4


# ---------------------------------------------------------------- validation
def validate_records(df: pd.DataFrame) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    missing = [c for c in config.RAW_FEATURE_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return {"errors": errors, "warnings": warnings}

    for col, (lo, hi) in config.NUMERIC_RANGES.items():
        raw = df[col]
        vals = pd.to_numeric(raw.astype(str).str.strip(), errors="coerce")
        blank = raw.astype(str).str.strip().isin(["", "nan", "None"])
        if col == "TotalCharges":
            bad = vals.isna() & ~blank  # blanks are allowed (new customers)
        else:
            bad = vals.isna()
        if bad.any():
            errors.append(f"{col}: {int(bad.sum())} non-numeric or missing value(s)")
        out_of_range = vals.notna() & ((vals < lo) | (vals > hi))
        if out_of_range.any():
            errors.append(f"{col}: {int(out_of_range.sum())} value(s) outside [{lo}, {hi}]")

    for col, allowed in config.EXPECTED_CATEGORIES.items():
        series = df[col]
        if series.isna().any():
            errors.append(f"{col}: {int(series.isna().sum())} missing value(s)")
        if col == "SeniorCitizen":
            nums = pd.to_numeric(series, errors="coerce")
            bad = series.notna() & ~nums.isin([0, 1])
            if bad.any():
                errors.append(f"SeniorCitizen: must be 0 or 1, got {sorted(map(str, series[bad].unique()))}")
            continue
        else:
            seen = set(series.dropna().astype(str).str.strip())
            allowed_set = set(allowed)
        new = sorted(map(str, seen - allowed_set))
        if new:
            if col == "gender" or col in config.BINARY_YES_NO_COLS:
                errors.append(f"{col}: invalid value(s) {new}")
            else:
                warnings.append(f"{col}: unseen category {new} (scored, but monitor)")
    return {"errors": errors, "warnings": warnings}


# ---------------------------------------------------------- reference profile
def build_reference_profile(features: pd.DataFrame, scores: np.ndarray) -> dict:
    """Snapshot of the training distribution used later for drift checks."""
    numeric = {}
    for col in config.NUMERIC_COLS:
        x = features[col].astype(float).to_numpy()
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        counts, _ = np.histogram(x, bins=np.r_[-np.inf, edges[1:-1], np.inf])
        numeric[col] = {"edges": edges[1:-1].tolist(), "props": (counts / counts.sum()).tolist()}
    categorical = {
        col: features[col].astype(str).value_counts(normalize=True).to_dict()
        for col in config.CATEGORICAL_COLS + config.BINARY_MODEL_COLS
    }
    s_edges = np.linspace(0, 1, 11)
    s_counts, _ = np.histogram(scores, bins=s_edges)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "score": {
            "edges": s_edges.tolist(),
            "props": (s_counts / s_counts.sum()).tolist(),
            "mean": float(np.mean(scores)),
        },
        "n_rows": int(len(features)),
    }


def psi(expected_props, actual_props) -> float:
    e = np.clip(np.asarray(expected_props, float), _EPS, None)
    a = np.clip(np.asarray(actual_props, float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _status(value: float) -> str:
    if value >= PSI_SIGNIFICANT:
        return "significant"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


# ------------------------------------------------------------------- logging
def hash_customer_id(customer_id, salt: str = config.LOG_ID_SALT) -> str | None:
    if customer_id is None or (isinstance(customer_id, float) and np.isnan(customer_id)):
        return None
    return hashlib.sha256(f"{salt}:{customer_id}".encode()).hexdigest()[:20]


class PredictionLogger:
    """Append-only JSONL log. Thread-safe within one process (API workers).

    Minimal by default: hashed customer ID, non-demographic model inputs,
    score, flag, model version and timestamp. Retention, rotation and access
    control belong to the hosting environment (see LIMITATIONS_AND_PRODUCTION_PLAN.md).
    """

    def __init__(
        self, path: Path | str = config.PREDICTION_LOG, include_demographics: bool = config.LOG_DEMOGRAPHICS
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.logged_cols = [
            c for c in config.RAW_FEATURE_COLS if include_demographics or c not in config.DEMOGRAPHIC_COLS
        ]

    def log(self, records: pd.DataFrame, probs, flags, model_version: str, warnings=None):
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        lines = []
        for rec, p, f in zip(records.to_dict(orient="records"), probs, flags, strict=True):
            lines.append(
                json.dumps(
                    {
                        "timestamp_utc": now,
                        "request_id": str(uuid.uuid4()),
                        "model_version": model_version,
                        "customer_id_hash": hash_customer_id(rec.get(config.ID_COL)),
                        "features": {k: rec.get(k) for k in self.logged_cols},
                        "churn_probability": round(float(p), 6),
                        "churn_flag": bool(f),
                        "warnings": warnings or [],
                    },
                    default=str,
                )
            )
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def read_prediction_log(path: Path | str = config.PREDICTION_LOG) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.json_normalize(rows)


# --------------------------------------------------------------------- drift
def drift_report(log_df: pd.DataFrame, reference: dict) -> tuple[pd.DataFrame, dict]:
    """Per-feature drift table and a prediction-level summary."""
    from .features import clean_and_engineer

    logged = [c for c in config.RAW_FEATURE_COLS if f"features.{c}" in log_df.columns]
    feats = log_df[[f"features.{c}" for c in logged]].copy()
    feats.columns = logged
    # Fields not logged (demographics, by default) get a placeholder so the
    # feature code runs; their derived features are then left out of the report.
    not_logged = [c for c in config.RAW_FEATURE_COLS if c not in logged]
    for c in not_logged:
        feats[c] = config.EXPECTED_CATEGORIES[c][0]
    skip = {"gender_male" if c == "gender" else c for c in not_logged}
    current = clean_and_engineer(feats[config.RAW_FEATURE_COLS])

    rows = []
    for col, ref in reference["numeric"].items():
        bins = np.r_[-np.inf, ref["edges"], np.inf]
        counts, _ = np.histogram(current[col].astype(float), bins=bins)
        value = psi(ref["props"], counts / max(counts.sum(), 1))
        rows.append(
            {
                "feature": col,
                "type": "numeric",
                "psi": round(value, 4),
                "status": _status(value),
                "unseen_categories": "",
            }
        )
    for col, ref_props in reference["categorical"].items():
        if col in skip:
            continue
        cur = current[col].astype(str).value_counts(normalize=True).to_dict()
        levels = sorted(set(ref_props) | set(cur))
        value = psi([ref_props.get(lv, 0) for lv in levels], [cur.get(lv, 0) for lv in levels])
        unseen = sorted(set(cur) - set(ref_props))
        rows.append(
            {
                "feature": col,
                "type": "categorical",
                "psi": round(value, 4),
                "status": "significant" if unseen else _status(value),
                "unseen_categories": ", ".join(unseen),
            }
        )
    # Raw payment method is engineered away into auto_pay; check it directly too
    raw_pm = set(feats["PaymentMethod"].astype(str).str.strip())
    unseen_pm = sorted(raw_pm - set(config.EXPECTED_CATEGORIES["PaymentMethod"]))
    for row in rows:
        if row["feature"] == "PaymentMethod" and unseen_pm:
            row["unseen_categories"], row["status"] = ", ".join(unseen_pm), "significant"

    scores = log_df["churn_probability"].to_numpy()
    s_counts, _ = np.histogram(scores, bins=reference["score"]["edges"])
    score_psi = psi(reference["score"]["props"], s_counts / max(s_counts.sum(), 1))
    summary = {
        "n_predictions": int(len(log_df)),
        "first_request_utc": str(log_df["timestamp_utc"].min()),
        "last_request_utc": str(log_df["timestamp_utc"].max()),
        "mean_score_training": round(reference["score"]["mean"], 4),
        "mean_score_current": round(float(scores.mean()), 4),
        "flag_rate_current": round(float(log_df["churn_flag"].mean()), 4),
        "score_psi": round(score_psi, 4),
        "score_status": _status(score_psi),
        "features_significant": [r["feature"] for r in rows if r["status"] == "significant"],
        "features_moderate": [r["feature"] for r in rows if r["status"] == "moderate"],
    }
    table = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return table, summary
