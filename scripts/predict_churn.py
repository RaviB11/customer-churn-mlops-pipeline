"""Score new customers from a CSV (CRM export format) or a single JSON record.

python scripts/predict_churn.py --input data/sample/new_customers.csv
python scripts/predict_churn.py --input data/sample/new_customers.csv --output preds.csv
python scripts/predict_churn.py --json '{"gender": "Female", ...}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from churn_pipeline import config  # noqa: E402
from churn_pipeline.predict import ChurnPredictor, SchemaError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="CSV of customer records")
    src.add_argument("--json", help="one customer record as a JSON string")
    ap.add_argument("--output", type=Path, help="write predictions CSV here (default: print)")
    ap.add_argument("--model", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--no-log", action="store_true", help="do not append to the prediction log")
    args = ap.parse_args()

    records = (
        pd.read_csv(args.input, dtype={"TotalCharges": str})
        if args.input
        else pd.DataFrame([json.loads(args.json)])
    )
    predictor = ChurnPredictor(args.model)
    try:
        preds = predictor.predict(records, log=not args.no_log)
    except SchemaError as exc:
        print(f"Input rejected: {exc}", file=sys.stderr)
        return 2

    for w in preds.attrs.get("warnings", []):
        print(f"WARNING: {w}", file=sys.stderr)
    preds = preds.sort_values("churn_probability", ascending=False)
    if args.output:
        preds.to_csv(args.output, index=False)
        print(f"Wrote {len(preds)} predictions to {args.output}")
    else:
        print(preds.to_string(index=False))
    print(
        f"\nModel {predictor.model_version} | flag threshold {predictor.threshold:.4f} | "
        f"flagged {int(preds['churn_flag'].sum())} of {len(preds)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
