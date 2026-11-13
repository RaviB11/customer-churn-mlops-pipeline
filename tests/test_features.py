import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from churn_pipeline import config
from churn_pipeline.cleaning import clean_dataset, clean_features
from churn_pipeline.features import build_model_pipeline, clean_and_engineer, engineer_features


def test_avg_monthly_charge_for_new_customers(raw_df):
    feats = engineer_features(clean_features(raw_df))
    new = feats["tenure"] == 0
    assert np.allclose(feats.loc[new, "avg_monthly_charge"], feats.loc[new, "MonthlyCharges"])
    assert feats["avg_monthly_charge"].notna().all()


def test_tenure_bucket_boundaries():
    df = pd.DataFrame({"tenure": [0, 6, 7, 12, 13, 24, 25, 48, 49, 72]})
    for c in config.RAW_FEATURE_COLS:
        if c != "tenure":
            df[c] = "No"
    df["MonthlyCharges"], df["TotalCharges"], df["SeniorCitizen"] = 50.0, "100", 0
    df["gender"], df["InternetService"], df["PaymentMethod"] = "Male", "DSL", "Mailed check"
    out = engineer_features(clean_features(df))
    assert out["tenure_bucket"].tolist() == [
        "0-6m",
        "0-6m",
        "7-12m",
        "7-12m",
        "13-24m",
        "13-24m",
        "25-48m",
        "25-48m",
        "49-72m",
        "49-72m",
    ]


def test_num_services_range(raw_df):
    feats = engineer_features(clean_features(raw_df))
    assert feats["num_services"].between(0, 9).all()


def test_model_inputs_exclude_target_and_id(raw_df):
    cols = clean_and_engineer(raw_df).columns
    assert config.TARGET_COL not in cols and config.ID_COL not in cols


def test_target_cannot_influence_predictions(trained, raw_df):
    """Leakage guard: flipping the Churn column must not change any score."""
    run, _ = trained
    pipe = run.results[run.best_name].pipeline
    X = raw_df.head(200).copy()
    flipped = X.copy()
    flipped[config.TARGET_COL] = np.where(X[config.TARGET_COL] == "Yes", "No", "Yes")
    assert np.allclose(pipe.predict_proba(X)[:, 1], pipe.predict_proba(flipped)[:, 1])


def test_no_single_feature_is_a_near_perfect_predictor(raw_df):
    """A feature with AUC ~1 on its own is almost always a leak."""
    y = clean_dataset(raw_df)[config.TARGET_COL]
    feats = clean_and_engineer(raw_df)
    for col in config.NUMERIC_COLS + config.BINARY_MODEL_COLS:
        auc = roc_auc_score(y, feats[col].astype(float))
        assert max(auc, 1 - auc) < 0.95, col
    for col in config.CATEGORICAL_COLS:  # each one-hot level on its own
        for level in feats[col].unique():
            auc = roc_auc_score(y, (feats[col] == level).astype(float))
            assert max(auc, 1 - auc) < 0.95, f"{col}={level}"


def test_shuffled_labels_give_chance_performance(raw_df):
    """If the pipeline scores well on shuffled labels, information is leaking."""
    y = clean_dataset(raw_df)[config.TARGET_COL].sample(frac=1, random_state=0).to_numpy()
    X = raw_df[config.RAW_FEATURE_COLS]
    pipe = build_model_pipeline(DecisionTreeClassifier(max_depth=4, random_state=0))
    auc = cross_val_score(pipe, X, y, cv=3, scoring="roc_auc").mean()
    assert 0.42 < auc < 0.58


def test_tree_leaf_rules_cover_all_training_rows(trained):
    from churn_pipeline.explain import leaf_rules

    run, _ = trained
    rules = leaf_rules(run.results["decision_tree"].pipeline, run.X_train, run.y_train)
    assert rules["customers"].sum() == len(run.X_train)
    assert rules["churn_rate"].between(0, 1).all()
    assert rules["rule"].str.contains("Contract").any()
