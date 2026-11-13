"""Cleaning layer.

`clean_features` is used both at training time and inside the deployed model
pipeline, so a raw CRM record goes through exactly the same transformations
as the training data (no train/serve skew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_YES_NO = {"Yes": 1, "No": 0}


def _strip_text(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.astype(str).str.strip()
    return s


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Type-coerce and encode the raw feature columns.

    * TotalCharges -> float. The 11 blanks in the source are all tenure == 0
      customers (signed up, not yet billed), so they become 0.0. Any other
      gap is imputed as tenure * MonthlyCharges.
    * Two-level Yes/No fields -> 0/1; gender -> gender_male (0/1).
    * Multi-level categoricals stay as strings for the one-hot encoder.
    """
    out = df.copy()
    for col in out.columns:
        if col in config.RAW_FEATURE_COLS:
            out[col] = _strip_text(out[col])

    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    out["SeniorCitizen"] = pd.to_numeric(out["SeniorCitizen"], errors="coerce").astype("Int64")

    tc = pd.to_numeric(out["TotalCharges"].astype(str).str.strip(), errors="coerce")
    tc = tc.mask(tc.isna() & (out["tenure"] == 0), 0.0)
    tc = tc.fillna(out["tenure"] * out["MonthlyCharges"])
    out["TotalCharges"] = tc.astype(float)

    for col in config.BINARY_YES_NO_COLS:
        out[col] = out[col].map(_YES_NO).astype("Int64")
    out["gender_male"] = (out["gender"] == "Male").astype(int)
    out = out.drop(columns=["gender"])

    for col in config.CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].astype(object)

    int_cols = config.BINARY_YES_NO_COLS + ["SeniorCitizen"]
    out[int_cols] = out[int_cols].astype(float)  # nullable -> float for sklearn
    return out


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full training-time clean: features + target + integrity checks."""
    if df[config.ID_COL].duplicated().any():
        raise ValueError("Duplicate customerID values in source data")
    out = clean_features(df)
    out[config.TARGET_COL] = _strip_text(df[config.TARGET_COL]).map(_YES_NO)
    if out[config.TARGET_COL].isna().any():
        raise ValueError("Unexpected values in the Churn column")
    out[config.TARGET_COL] = out[config.TARGET_COL].astype(int)

    feature_cols = [c for c in out.columns if c not in (config.ID_COL, config.TARGET_COL)]
    n_null = int(out[feature_cols].isna().sum().sum())
    if n_null:
        bad = out[feature_cols].isna().sum()
        raise ValueError(f"Nulls remain after cleaning: {bad[bad > 0].to_dict()}")
    if np.isinf(out[feature_cols].select_dtypes("number")).any().any():
        raise ValueError("Infinite values after cleaning")
    return out
