import numpy as np
import pandas as pd

from churn_pipeline.evaluation import (
    calibration_table,
    choose_threshold,
    compute_metrics,
    expected_calibration_error,
    paired_bootstrap_auc_diff,
    subgroup_metrics,
)

rng = np.random.default_rng(0)
P = rng.random(3000)
Y = (rng.random(3000) < P).astype(int)  # perfectly calibrated by construction


def test_calibration_of_calibrated_scores_is_small():
    t = calibration_table(Y, P)
    assert t["customers"].sum() == len(Y)
    assert expected_calibration_error(Y, P) < 0.05


def test_calibration_detects_miscalibration():
    assert expected_calibration_error(Y, np.clip(P * 0.4, 0, 1)) > 0.2


def test_brier_reported():
    assert 0 < compute_metrics(Y, P)["brier"] < 0.25


def test_bootstrap_identical_models_is_zero():
    out = paired_bootstrap_auc_diff(Y, P, P, n_boot=200)
    assert out["observed_diff"] == 0 and out["ci95_low"] == 0 and out["ci95_high"] == 0


def test_bootstrap_detects_clearly_better_model():
    noisy = np.clip(P + rng.normal(0, 0.5, len(P)), 0, 1)
    out = paired_bootstrap_auc_diff(Y, P, noisy, n_boot=300)
    assert out["ci95_low"] > 0 and out["ci95_low"] <= out["observed_diff"] <= out["ci95_high"]


def test_threshold_meets_recall_target():
    t = choose_threshold(Y, P, 0.8)
    assert compute_metrics(Y, P, t)["recall"] >= 0.8


def test_subgroup_metrics_cover_all_rows():
    groups = pd.Series(np.where(rng.random(len(Y)) < 0.5, "A", "B"))
    out = subgroup_metrics(Y, P, groups, 0.5)
    assert out["customers"].sum() == len(Y) and set(out["group"]) == {"A", "B"}
