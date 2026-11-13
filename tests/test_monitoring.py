import joblib
import numpy as np

from churn_pipeline.monitoring import drift_report, psi, read_prediction_log


def test_psi_zero_for_identical():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0


def test_psi_large_for_shift():
    assert psi([0.5, 0.5], [0.05, 0.95]) > 0.25


def test_drift_report_flags_price_rise_and_new_payment_method(predictor, trained, raw_df):
    _, path = trained
    ref = joblib.load(path)["reference_profile"]
    batch = raw_df.drop(columns=["Churn"]).sample(400, random_state=0)
    predictor.predict(batch)
    table, summary = drift_report(read_prediction_log(predictor.logger.path), ref)
    assert "MonthlyCharges" not in summary["features_significant"]  # unshifted traffic

    shifted = batch.copy()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 1.5
    shifted.loc[shifted.index[:40], "PaymentMethod"] = "Digital wallet"
    predictor.logger.path.unlink()
    predictor.predict(shifted)
    table, summary = drift_report(read_prediction_log(predictor.logger.path), ref)
    assert "MonthlyCharges" in summary["features_significant"]
    pm = table.set_index("feature").loc["PaymentMethod"]
    assert "Digital wallet" in pm["unseen_categories"]
    assert np.isfinite(table["psi"]).all()


def test_log_is_privacy_minimised_by_default(predictor, sample_records):
    """Audit P1-E: no raw customer ID and no demographic fields in the log."""
    import json

    from churn_pipeline import config
    from churn_pipeline.monitoring import hash_customer_id

    predictor.predict(sample_records.head(3))
    entries = [json.loads(x) for x in predictor.logger.path.read_text().splitlines()]
    raw_ids = set(sample_records[config.ID_COL].head(3))
    for e in entries:
        assert "customer_id" not in e
        assert e["customer_id_hash"] not in raw_ids and len(e["customer_id_hash"]) == 20
        assert not set(config.DEMOGRAPHIC_COLS) & set(e["features"])
        assert raw_ids.isdisjoint(json.dumps(e).split('"'))
    assert entries[0]["customer_id_hash"] == hash_customer_id(sample_records[config.ID_COL].iloc[0])


def test_demographic_logging_is_opt_in(tmp_path, sample_records):
    import json

    from churn_pipeline import config
    from churn_pipeline.monitoring import PredictionLogger

    log = PredictionLogger(tmp_path / "full.jsonl", include_demographics=True)
    log.log(sample_records.head(1), [0.3], [True], "v-test")
    entry = json.loads(log.path.read_text())
    assert set(config.DEMOGRAPHIC_COLS) <= set(entry["features"])
