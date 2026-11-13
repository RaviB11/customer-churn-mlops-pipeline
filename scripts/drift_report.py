"""Compare logged prediction traffic with the training profile.

python scripts/drift_report.py                       # whole log
python scripts/drift_report.py --last 500            # most recent 500 requests
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402

from churn_pipeline import config  # noqa: E402
from churn_pipeline.monitoring import drift_report, read_prediction_log  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=config.PREDICTION_LOG)
    ap.add_argument("--model", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--last", type=int, help="only the most recent N requests")
    ap.add_argument("--out", type=Path, default=config.REPORTS_DIR / "drift_report.csv")
    args = ap.parse_args()

    log_df = read_prediction_log(args.log)
    if log_df.empty:
        print("No predictions logged yet.")
        return 1
    if args.last:
        log_df = log_df.tail(args.last)
    reference = joblib.load(args.model)["reference_profile"]
    table, summary = drift_report(log_df, reference)
    table.to_csv(args.out, index=False)
    print(json.dumps(summary, indent=2))
    print("\n" + table.head(12).to_string(index=False))
    print(f"\nFull table: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
