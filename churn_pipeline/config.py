"""Central configuration: paths, column groups and the expected input schema.

Everything that another module would otherwise hard-code lives here, so a
change to the source schema only has to be made in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (override the project root with CHURN_PROJECT_ROOT, e.g. in Docker)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(os.getenv("CHURN_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
WAREHOUSE_DB = DATA_DIR / "warehouse" / "telco.db"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", MODELS_DIR / "churn_model.joblib"))
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MONITORING_DIR = PROJECT_ROOT / "monitoring"
PREDICTION_LOG = Path(os.getenv("CHURN_PREDICTION_LOG", MONITORING_DIR / "prediction_log.jsonl"))
# Prediction-log privacy (audit P1-E). Customer IDs are stored only as a salted
# SHA-256 hash (still joinable to later churn outcomes by whoever holds the
# salt), and demographic fields are left out unless explicitly enabled.
LOG_DEMOGRAPHICS = os.getenv("CHURN_LOG_DEMOGRAPHICS", "0") == "1"
LOG_ID_SALT = os.getenv("CHURN_LOG_SALT", "demo-only-salt-set-CHURN_LOG_SALT-in-production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "telco-churn"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Decision threshold policy: pick the highest threshold whose cross-validated
# recall on the TRAINING set (out-of-fold) is at least this value. The
# retention team said missing a churner costs more than a wasted offer, so we
# prioritise recall. Change here, retrain, and the threshold moves with it.
TARGET_RECALL = 0.80

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
ID_COL = "customerID"
TARGET_COL = "Churn"

DEMOGRAPHIC_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents"]
ACCOUNT_COLS = [
    "tenure",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
RAW_FEATURE_COLS = DEMOGRAPHIC_COLS + ACCOUNT_COLS + SERVICE_COLS

# Two-level fields converted to 0/1 during cleaning
BINARY_YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# Multi-level fields that are one-hot encoded inside the model pipeline
CATEGORICAL_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "tenure_bucket",
]

# Add-on services counted in the engineered `num_services` feature
ADDON_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_charge",
    "charge_delta",
    "num_services",
]
BINARY_MODEL_COLS = [
    "gender_male",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "auto_pay",
]

TENURE_BINS = [-1, 6, 12, 24, 48, 72, float("inf")]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49-72m", "72m+"]

# Categories seen in the IBM source data. New values are still scored (the
# one-hot encoder ignores unknown levels) but are flagged by monitoring.
EXPECTED_CATEGORIES: dict[str, list] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}
NUMERIC_RANGES = {"tenure": (0, 120), "MonthlyCharges": (0, 1000), "TotalCharges": (0, 100000)}
