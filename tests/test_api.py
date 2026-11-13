import pytest
from fastapi.testclient import TestClient

from churn_pipeline import api, config


@pytest.fixture()
def client(trained, tmp_path, monkeypatch):
    _, path = trained
    monkeypatch.setenv("CHURN_MODEL_PATH", str(path))
    monkeypatch.setenv("CHURN_PREDICTION_LOG", str(tmp_path / "api_log.jsonl"))
    api._predictor = None
    yield TestClient(api.app)
    api._predictor = None


def _record(raw_df, i=0):
    rec = raw_df.drop(columns=[config.TARGET_COL]).iloc[i].to_dict()
    rec["SeniorCitizen"], rec["tenure"] = int(rec["SeniorCitizen"]), int(rec["tenure"])
    return rec


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_one(client, raw_df):
    r = client.post("/predict", json=_record(raw_df))
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_tier"] in {"high", "medium", "low"}


def test_predict_batch(client, raw_df):
    r = client.post("/predict/batch", json=[_record(raw_df, i) for i in range(3)])
    assert r.status_code == 200 and len(r.json()) == 3


def test_bad_input_returns_422(client, raw_df):
    rec = _record(raw_df)
    rec["Dependents"] = "Sometimes"
    assert client.post("/predict", json=rec).status_code == 422
    rec = _record(raw_df)
    del rec["Contract"]
    assert client.post("/predict", json=rec).status_code == 422
