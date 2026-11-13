import json

import pytest

from churn_pipeline import config
from churn_pipeline.predict import SchemaError, risk_tier


def test_probabilities_valid(predictor, sample_records):
    out = predictor.predict(sample_records)
    assert len(out) == len(sample_records)
    assert out["churn_probability"].between(0, 1).all()
    assert (out["churn_flag"] == (out["churn_probability"] >= predictor.threshold)).all()


def test_single_dict_record(predictor, sample_records):
    rec = sample_records.iloc[0].to_dict()
    out = predictor.predict(rec)
    assert out[config.ID_COL].iloc[0] == rec[config.ID_COL]


def test_blank_total_charges_new_customer(predictor, sample_records):
    rec = sample_records.iloc[[0]].copy()
    rec["tenure"], rec["TotalCharges"] = 0, " "
    assert predictor.predict(rec)["churn_probability"].notna().all()


def test_unseen_category_scored_with_warning(predictor, sample_records):
    rec = sample_records.iloc[[0]].copy()
    rec["PaymentMethod"] = "Digital wallet"
    out = predictor.predict(rec)
    assert out["churn_probability"].between(0, 1).all()
    assert any("Digital wallet" in w for w in out.attrs["warnings"])


def test_missing_column_rejected(predictor, sample_records):
    with pytest.raises(SchemaError, match="Missing required columns"):
        predictor.predict(sample_records.drop(columns=["Contract"]))


def test_invalid_binary_value_rejected(predictor, sample_records):
    bad = sample_records.iloc[[0]].copy()
    bad["Partner"] = "Maybe"
    with pytest.raises(SchemaError):
        predictor.predict(bad)


def test_negative_tenure_rejected(predictor, sample_records):
    bad = sample_records.iloc[[0]].copy()
    bad["tenure"] = -3
    with pytest.raises(SchemaError):
        predictor.predict(bad)


def test_predictions_are_logged(predictor, sample_records):
    predictor.predict(sample_records.head(5))
    lines = predictor.logger.path.read_text().splitlines()
    assert len(lines) == 5
    entry = json.loads(lines[0])
    assert {"timestamp_utc", "model_version", "features", "churn_probability", "churn_flag"} <= entry.keys()
    assert config.TARGET_COL not in entry["features"]


@pytest.mark.parametrize("p,t,expected", [(0.9, 0.27, "high"), (0.3, 0.27, "medium"), (0.1, 0.27, "low")])
def test_risk_tiers(p, t, expected):
    assert risk_tier(p, t) == expected


@pytest.mark.parametrize("value", [0.9, 2, -1, "yes"])
def test_invalid_senior_citizen_rejected_cleanly(predictor, sample_records, value):
    """Audit P1-B: 0.9 used to pass validation as 0 and then crash with TypeError."""
    bad = sample_records.iloc[[0]].copy()
    bad["SeniorCitizen"] = value
    with pytest.raises(SchemaError, match="SeniorCitizen"):
        predictor.predict(bad)


def test_cli_rejects_fractional_senior_citizen(trained, sample_records, tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    _, model_path = trained
    bad = sample_records.head(2).copy()
    bad["SeniorCitizen"] = 0.9
    csv = tmp_path / "bad.csv"
    bad.to_csv(csv, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "predict_churn.py"
    res = subprocess.run(
        [sys.executable, str(script), "--input", str(csv), "--model", str(model_path), "--no-log"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "SeniorCitizen" in res.stderr and "Traceback" not in res.stderr
