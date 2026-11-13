"""Train all models, pick the best by cross-validated ROC-AUC, save artifacts.

python scripts/train.py            # full run (grid search, 5-fold CV)
python scripts/train.py --fast     # quick smoke run (small grid, 3-fold)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from churn_pipeline import config, etl  # noqa: E402
from churn_pipeline.evaluation import (  # noqa: E402
    calibration_table,
    decile_table,
    expected_calibration_error,
    paired_bootstrap_auc_diff,
    subgroup_metrics,
)
from churn_pipeline.modeling import comparison_table, train_all  # noqa: E402
from churn_pipeline.predict import risk_tier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["warehouse", "csv"], default="warehouse")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--target-recall", type=float, default=config.TARGET_RECALL)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    etl.build_warehouse()
    etl.write_data_manifest()
    run = train_all(source=args.source, fast=args.fast, target_recall=args.target_recall)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    table = comparison_table(run.results)
    table.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)

    best = run.results[run.best_name]
    decile_table(run.y_test, best.test_prob).to_csv(config.REPORTS_DIR / "test_decile_lift.csv", index=False)

    # Sample report: every held-out customer, ranked by churn probability
    sample = run.X_test.copy()
    sample.insert(0, config.ID_COL, run.ids_test.to_numpy())
    sample["churn_probability"] = best.test_prob.round(4)
    sample["churn_flag"] = best.test_prob >= best.threshold
    sample["risk_tier"] = [risk_tier(p, best.threshold) for p in best.test_prob]
    sample["actual_churn"] = run.y_test.to_numpy()
    sample.sort_values("churn_probability", ascending=False).to_csv(
        config.REPORTS_DIR / "sample_predictions_test_set.csv", index=False
    )

    # Calibration of the shipped model (audit P1-C)
    calibration_table(run.y_test, best.test_prob).to_csv(
        config.REPORTS_DIR / "calibration_test_set.csv", index=False
    )
    calibration = {
        "brier_test": best.test_metrics["brier"],
        "brier_constant_rate_baseline": float(((run.y_test - run.y_train.mean()) ** 2).mean()),
        "ece_test_10_bins": expected_calibration_error(run.y_test, best.test_prob),
    }

    # Paired bootstrap of test AUC differences (audit P2-A)
    bootstrap = {
        f"{a}_minus_{b}": paired_bootstrap_auc_diff(
            run.y_test, run.results[a].test_prob, run.results[b].test_prob
        )
        for a, b in [("xgboost", "logistic_regression"), ("xgboost", "decision_tree")]
    }

    # Subgroup performance for the sensitive attributes among the inputs (audit P2-F)
    groups = {
        "gender": run.X_test["gender"],
        "senior_citizen": run.X_test["SeniorCitizen"].map({0: "No", 1: "Yes"}),
    }
    sub = pd.concat(
        subgroup_metrics(run.y_test, best.test_prob, g, best.threshold).assign(attribute=attr)
        for attr, g in groups.items()
    )
    sub[["attribute"] + [c for c in sub.columns if c != "attribute"]].to_csv(
        config.REPORTS_DIR / "subgroup_metrics_test_set.csv", index=False
    )

    metrics = {
        name: {
            "cv_roc_auc_mean": r.cv_auc_mean,
            "cv_roc_auc_std": r.cv_auc_std,
            "best_params": r.best_params,
            "test_at_threshold": r.test_metrics,
            "test_at_0.5": r.test_metrics_at_05,
        }
        for name, r in run.results.items()
    }
    metrics["selected_model"] = run.best_name
    metrics["calibration"] = calibration
    metrics["paired_bootstrap_test_auc"] = bootstrap
    metrics["data_sha256"] = run.data_sha256
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n" + table.to_string(index=False))
    print(f"\nSelected model: {run.best_name} (threshold {best.threshold:.4f})")
    print(f"Saved bundle:   {run.bundle_path}")
    print("\nCalibration:", {k: round(v, 4) for k, v in calibration.items()})
    for k, v in bootstrap.items():
        lo, hi = v["ci95_low"], v["ci95_high"]
        print(f"Test AUC {k}: {v['observed_diff']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(sub.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
