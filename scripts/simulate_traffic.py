"""Simulate CRM traffic through the predictor so the monitoring has data.

Two batches are replayed from the model's own holdout customers (IDs stored in
the model bundle at training time, so they are never training customers):
  1. "normal"  -- records as-is
  2. "drifted" -- a hypothetical future month: monthly charges +15%, more
                  month-to-month contracts, and a payment method the model
                  has never seen ("Digital wallet"). These changes are
                  synthetic, chosen only to show what the drift report catches.

    python scripts/simulate_traffic.py --n 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from churn_pipeline import config  # noqa: E402
from churn_pipeline.modeling import load_holdout_records  # noqa: E402
from churn_pipeline.predict import ChurnPredictor  # noqa: E402


def drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    out["MonthlyCharges"] = (out["MonthlyCharges"] * 1.15).round(2)
    flip = rng.random(len(out)) < 0.30
    out.loc[flip, "Contract"] = "Month-to-month"
    new_pm = rng.random(len(out)) < 0.10
    out.loc[new_pm, "PaymentMethod"] = "Digital wallet"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="records per batch")
    ap.add_argument("--log", type=Path, default=config.PREDICTION_LOG)
    ap.add_argument("--model", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--reset", action="store_true", help="delete the existing log first")
    args = ap.parse_args()
    if args.reset and args.log.exists():
        args.log.unlink()

    test = load_holdout_records(args.model).drop(columns=[config.TARGET_COL])
    rng = np.random.default_rng(7)
    predictor = ChurnPredictor(args.model, log_path=args.log)
    normal = test.sample(min(args.n, len(test)), random_state=1)
    drifted = drift(test.sample(min(args.n, len(test)), random_state=2), rng)

    for name, batch in [("normal", normal), ("drifted", drifted)]:
        preds = predictor.predict(batch)
        print(
            f"{name:8s}: {len(preds)} scored, mean p={preds['churn_probability'].mean():.3f}, "
            f"flagged={preds['churn_flag'].mean():.1%}, warnings={preds.attrs['warnings']}"
        )
    print(f"Log: {args.log}")


if __name__ == "__main__":
    main()
