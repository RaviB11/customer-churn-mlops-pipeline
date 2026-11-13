import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CHURN_DISABLE_MLFLOW", "1")

from churn_pipeline import config, etl  # noqa: E402


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return etl.load_raw_csv()


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """A fast-mode model trained once per test session (no MLflow, temp path)."""
    from churn_pipeline.modeling import train_all

    path = tmp_path_factory.mktemp("model") / "model.joblib"
    run = train_all(source="csv", fast=True, save_path=path, track=False)
    return run, path


@pytest.fixture()
def predictor(trained, tmp_path):
    from churn_pipeline.predict import ChurnPredictor

    _, path = trained
    return ChurnPredictor(path, log_path=tmp_path / "log.jsonl")


@pytest.fixture()
def sample_records(raw_df) -> pd.DataFrame:
    return raw_df.drop(columns=[config.TARGET_COL]).head(25).copy()
