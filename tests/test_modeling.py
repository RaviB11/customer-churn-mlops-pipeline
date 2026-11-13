from churn_pipeline import config
from churn_pipeline.evaluation import compute_metrics


def test_all_models_trained(trained):
    run, path = trained
    assert set(run.results) == {"decision_tree", "logistic_regression", "xgboost"}
    assert path.exists() and path.with_suffix(".json").exists()


def test_models_beat_chance_clearly(trained):
    run, _ = trained
    for r in run.results.values():
        assert r.test_metrics["roc_auc"] > 0.75, r.name


def test_threshold_meets_recall_target_out_of_fold(trained):
    run, _ = trained
    for r in run.results.values():
        oof = compute_metrics(run.y_train, r.oof_prob, r.threshold)
        assert oof["recall"] >= config.TARGET_RECALL - 1e-9, r.name


def test_selected_model_has_best_cv_auc(trained):
    run, _ = trained
    best = max(r.cv_auc_mean for r in run.results.values())
    assert run.results[run.best_name].cv_auc_mean == best


def test_test_set_disjoint_from_train(trained):
    run, _ = trained
    assert not set(run.X_train.index) & set(run.X_test.index)


def test_csv_and_warehouse_give_identical_modelling_data(tmp_path, monkeypatch):
    """Audit P1-A: both sources must reach the positional split in the same row
    order, otherwise they produce different holdout sets."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from churn_pipeline import etl
    from churn_pipeline.modeling import load_modelling_data

    monkeypatch.setattr(config, "WAREHOUSE_DB", tmp_path / "wh.db")
    etl.build_warehouse(db_path=tmp_path / "wh.db")
    splits = {}
    for source in ("csv", "warehouse"):
        X, y, ids = load_modelling_data(source)
        *_, ids_te = train_test_split(
            X, y, ids, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
        )
        splits[source] = (X, y, ids_te)
    assert splits["csv"][2].tolist() == splits["warehouse"][2].tolist()
    pd.testing.assert_series_equal(splits["csv"][1], splits["warehouse"][1])


def test_bundle_records_the_holdout_ids(trained):
    import joblib

    run, path = trained
    bundle = joblib.load(path)
    assert bundle["holdout_ids"] == list(run.ids_test)
    assert len(bundle["holdout_ids"]) == len(run.X_test)


def test_sample_customers_are_holdout_customers(raw_df):
    """data/sample must contain only customers the model never trained on."""
    import pandas as pd

    from churn_pipeline.modeling import load_modelling_data, split_holdout

    X, y, ids = load_modelling_data("csv")
    *_, ids_tr, ids_te = split_holdout(X, y, ids)
    sample = pd.read_csv(config.DATA_DIR / "sample" / "new_customers.csv")
    assert sample[config.ID_COL].isin(set(ids_te)).all()
    assert not sample[config.ID_COL].isin(set(ids_tr)).any()
    assert config.TARGET_COL not in sample.columns


def test_traffic_pool_is_the_model_holdout(trained):
    from churn_pipeline.modeling import load_holdout_records, load_modelling_data, split_holdout

    run, path = trained
    pool = load_holdout_records(path)
    assert pool[config.ID_COL].tolist() == list(run.ids_test)
    X, y, ids = load_modelling_data("csv")
    *_, ids_tr, _ = split_holdout(X, y, ids)
    assert not set(pool[config.ID_COL]) & set(ids_tr)
