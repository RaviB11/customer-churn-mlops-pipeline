"""Regenerate data/sample/ from the trained model's holdout customers.

    python scripts/make_sample_customers.py

Writes new_customers.csv (8 holdout customers, Churn column removed) and
customer.json (the first of them) so the CLI and API demos never score a
customer the model was trained on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from churn_pipeline import config  # noqa: E402
from churn_pipeline.modeling import load_holdout_records  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=config.DATA_DIR / "sample")
    args = ap.parse_args()

    holdout = load_holdout_records(args.model)
    sample = holdout.sample(args.n, random_state=3).drop(columns=[config.TARGET_COL])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out_dir / "new_customers.csv", index=False)

    rec = sample.iloc[0].to_dict()
    rec["SeniorCitizen"], rec["tenure"] = int(rec["SeniorCitizen"]), int(rec["tenure"])
    rec["MonthlyCharges"] = float(rec["MonthlyCharges"])
    (args.out_dir / "customer.json").write_text(json.dumps(rec, indent=2))
    print(f"Wrote {len(sample)} holdout customers to {args.out_dir}")


if __name__ == "__main__":
    main()
