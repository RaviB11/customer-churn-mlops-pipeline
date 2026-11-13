"""REST API for CRM integration.

Run:  uvicorn churn_pipeline.api:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import config
from .predict import ChurnPredictor, SchemaError

app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")
_predictor: ChurnPredictor | None = None


def predictor() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor(
            os.getenv("CHURN_MODEL_PATH", str(config.MODEL_PATH)),
            os.getenv("CHURN_PREDICTION_LOG", str(config.PREDICTION_LOG)),
        )
    return _predictor


class CustomerRecord(BaseModel):
    """One customer in the CRM's raw format. Category fields are free strings
    on purpose: an unseen value (e.g. a new payment method) is scored and
    flagged as a warning rather than rejected."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85",
            }
        }
    )

    customerID: str | None = None
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: str | float | None = None


class Prediction(BaseModel):
    customerID: str | None
    churn_probability: float
    churn_flag: bool
    risk_tier: str
    threshold: float
    model_version: str
    warnings: list[str] = []


def _score(records: list[CustomerRecord]) -> list[Prediction]:
    p = predictor()
    try:
        out = p.predict([r.model_dump() for r in records])
    except SchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [
        Prediction(
            **row,
            threshold=p.threshold,
            model_version=p.model_version,
            warnings=out.attrs.get("warnings", []),
        )
        for row in out.to_dict(orient="records")
    ]


@app.get("/health")
def health() -> dict:
    p = predictor()
    return {"status": "ok", "model_version": p.model_version, "threshold": p.threshold}


@app.post("/predict", response_model=Prediction)
def predict_one(record: CustomerRecord) -> Prediction:
    return _score([record])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(records: list[CustomerRecord]) -> list[Prediction]:
    if not records:
        raise HTTPException(status_code=422, detail="Empty batch")
    if len(records) > 5000:
        raise HTTPException(status_code=413, detail="Batch limit is 5000 records")
    return _score(records)
