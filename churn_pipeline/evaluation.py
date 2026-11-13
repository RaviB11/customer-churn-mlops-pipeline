"""Model evaluation: statistical metrics plus retention-team business metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y_true, y_prob, target_recall: float) -> float:
    """Highest threshold whose recall is still >= target_recall.

    Highest = fewest offers sent while still catching the target share of
    churners. Must be called on out-of-fold TRAINING predictions, never on the
    test set, or the reported test metrics are optimistic.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # recall[i] corresponds to thresholds[i] for i < len(thresholds)
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(thresholds.min())
    return float(thresholds[ok].max())


def top_k_lift(y_true, y_prob, frac: float = 0.10) -> dict:
    """Churn rate in the top `frac` of scores vs the overall churn rate."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_prob))
    k = max(1, int(round(frac * len(y_true))))
    top = y_true[order[:k]]
    base = y_true.mean()
    return {
        f"top{int(frac * 100)}_churn_rate": float(top.mean()),
        f"top{int(frac * 100)}_lift": float(top.mean() / base) if base else float("nan"),
        f"top{int(frac * 100)}_capture": float(top.sum() / y_true.sum()) if y_true.sum() else 0.0,
    }


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        # Business framing: customers who WOULD be flagged for outreach (no
        # offers were sent; this is offline scoring of historical data)
        "customers_flagged": int(tp + fp),
        "churners_missed": int(fn),
    }
    metrics.update(top_k_lift(y_true, y_prob, 0.10))
    return metrics


def decile_table(y_true, y_prob) -> pd.DataFrame:
    """Customers ranked by score and split into 10 equal groups (1 = riskiest)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, 10, labels=list(range(1, 11)))
    g = df.groupby("decile", observed=True).agg(
        customers=("y", "size"), churners=("y", "sum"), avg_score=("p", "mean")
    )
    g["churn_rate"] = g["churners"] / g["customers"]
    g["lift"] = g["churn_rate"] / df["y"].mean()
    g["cum_capture"] = g["churners"].cumsum() / df["y"].sum()
    return g.reset_index()


def calibration_table(y_true, y_prob, edges=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)) -> pd.DataFrame:
    """Predicted vs observed churn rate by score band (reliability check).

    Probabilities drive the high/medium/low tiers, so they should mean what
    they say: customers scored ~0.6 should churn ~60% of the time.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["band"] = pd.cut(df["p"], bins=list(edges), include_lowest=True)
    g = df.groupby("band", observed=True).agg(
        customers=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")
    )
    g["gap"] = g["observed_rate"] - g["mean_predicted"]
    return g.reset_index().assign(band=lambda d: d["band"].astype(str))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Customer-weighted mean |observed - predicted| over equal-width bins."""
    t = calibration_table(y_true, y_prob, edges=np.linspace(0, 1, n_bins + 1))
    return float((t["gap"].abs() * t["customers"]).sum() / t["customers"].sum())


def paired_bootstrap_auc_diff(y_true, prob_a, prob_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUC(a) - AUC(b) on the same customers, with a 95% percentile interval.

    Resamples customers with replacement and scores both models on each
    resample, so the interval reflects the paired comparison. An interval that
    includes 0 means the test set cannot distinguish the two models; it is not
    proof that they are equivalent.
    """
    y, a, b = np.asarray(y_true), np.asarray(prob_a), np.asarray(prob_b)
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    while len(diffs) < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    diffs = np.array(diffs)
    return {
        "observed_diff": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "share_of_resamples_a_better": float((diffs > 0).mean()),
        "n_boot": n_boot,
    }


def subgroup_metrics(y_true, y_prob, groups: pd.Series, threshold: float) -> pd.DataFrame:
    """Recall, precision and flag rate per subgroup at the deployed threshold."""
    rows = []
    y, p, g = np.asarray(y_true), np.asarray(y_prob), np.asarray(groups)
    for level in pd.unique(g):
        m = g == level
        pred = p[m] >= threshold
        tp = int((pred & (y[m] == 1)).sum())
        rows.append(
            {
                "group": level,
                "customers": int(m.sum()),
                "actual_churn_rate": float(y[m].mean()),
                "flag_rate": float(pred.mean()),
                "recall": tp / max(int((y[m] == 1).sum()), 1),
                "precision": tp / max(int(pred.sum()), 1),
                "roc_auc": float(roc_auc_score(y[m], p[m])) if 0 < y[m].mean() < 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)
