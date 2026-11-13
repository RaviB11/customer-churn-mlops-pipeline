"""Serving layer: load the trained bundle and score raw customer records."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .monitoring import PredictionLogger, validate_records


class SchemaError(ValueError):
    """Input records fail hard validation (missing columns, bad values)."""


def risk_tier(prob: float, threshold: float) -> str:
    """Three tiers for the retention team's queue.

    high   -> above max(0.5, threshold): more likely to churn than not
    medium -> above the recall-driven decision threshold
    low    -> below the threshold
    """
    if prob >= max(0.5, threshold):
        return "high"
    if prob >= threshold:
        return "medium"
    return "low"


class ChurnPredictor:
    def __init__(
        self, model_path: Path | str = config.MODEL_PATH, log_path: Path | str | None = config.PREDICTION_LOG
    ):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.model_version = bundle["model_version"]
        self.model_name = bundle["model_name"]
        self.metadata = {k: v for k, v in bundle.items() if k not in ("pipeline",)}
        self.logger = PredictionLogger(log_path) if log_path else None

    def predict(self, records: pd.DataFrame | list[dict] | dict, log: bool = True) -> pd.DataFrame:
        df = pd.DataFrame([records]) if isinstance(records, dict) else pd.DataFrame(records)
        df = df.drop(columns=[config.TARGET_COL], errors="ignore")
        check = validate_records(df)
        if check["errors"]:
            raise SchemaError("; ".join(check["errors"]))

        probs = self.pipeline.predict_proba(df[config.RAW_FEATURE_COLS])[:, 1].astype(float)
        flags = probs >= self.threshold
        ids = (
            df[config.ID_COL] if config.ID_COL in df.columns else pd.Series([None] * len(df), index=df.index)
        )
        out = pd.DataFrame(
            {
                config.ID_COL: ids.to_numpy(),
                "churn_probability": np.round(probs, 4),
                "churn_flag": flags,
                "risk_tier": [risk_tier(p, self.threshold) for p in probs],
            }
        )
        out.attrs["warnings"] = check["warnings"]
        if log and self.logger:
            self.logger.log(df, probs, flags, self.model_version, check["warnings"])
        return out


@lru_cache(maxsize=1)
def get_predictor(model_path: str = str(config.MODEL_PATH)) -> ChurnPredictor:
    return ChurnPredictor(model_path)
