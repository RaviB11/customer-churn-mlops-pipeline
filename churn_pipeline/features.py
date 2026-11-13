"""Feature engineering and the sklearn preprocessing step.

Engineered features (all computable from data known *before* the churn month):

* avg_monthly_charge -- TotalCharges / tenure (MonthlyCharges when tenure=0)
* charge_delta       -- MonthlyCharges - avg_monthly_charge. Positive means the
                        customer now pays more than their lifetime average
                        (price rise or upgrade); a plausible churn trigger.
* num_services       -- phone + internet + multiple lines + add-ons held
* auto_pay           -- 1 if the payment method is an automatic one
* tenure_bucket      -- 0-6m, 7-12m, 13-24m, 25-48m, 49-72m, 72m+
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config
from .cleaning import clean_features


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tenure = out["tenure"].astype(float)
    safe_tenure = tenure.where(tenure > 0)
    out["avg_monthly_charge"] = (out["TotalCharges"] / safe_tenure).fillna(out["MonthlyCharges"])
    out["charge_delta"] = out["MonthlyCharges"] - out["avg_monthly_charge"]

    addons = sum((out[c] == "Yes").astype(int) for c in config.ADDON_SERVICE_COLS)
    out["num_services"] = (
        out["PhoneService"].fillna(0).astype(int)
        + (out["InternetService"] != "No").astype(int)
        + (out["MultipleLines"] == "Yes").astype(int)
        + addons
    )
    out["auto_pay"] = out["PaymentMethod"].astype(str).str.contains("automatic").astype(int)
    out["tenure_bucket"] = pd.cut(tenure, bins=config.TENURE_BINS, labels=config.TENURE_LABELS).astype(object)
    return out


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw-format records -> model-ready columns. Used as the first pipeline
    step so the deployed model accepts records straight from the CRM."""
    out = engineer_features(clean_features(df))
    return out[config.NUMERIC_COLS + config.BINARY_MODEL_COLS + config.CATEGORICAL_COLS]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_COLS),
            ("bin", "passthrough", config.BINARY_MODEL_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_COLS,
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_model_pipeline(estimator) -> Pipeline:
    """Full serving pipeline: raw record -> cleaned -> engineered -> encoded -> model."""
    return Pipeline(
        steps=[
            ("clean_engineer", FunctionTransformer(clean_and_engineer, validate=False)),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def feature_names(pipeline: Pipeline) -> np.ndarray:
    return pipeline.named_steps["preprocess"].get_feature_names_out()
