"""Explainability helpers: tree rules, logistic coefficients, SHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import export_text

from .features import feature_names


def tree_rules(pipeline, max_depth: int = 4) -> str:
    """Raw sklearn text dump of the tree (numeric splits on standardised
    values). For business-readable rules in original units use `leaf_rules`."""
    return export_text(
        pipeline.named_steps["model"],
        feature_names=list(feature_names(pipeline)),
        max_depth=max_depth,
        show_weights=False,
    )


def logistic_coefficients(pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    coefs = model.coef_.ravel()
    df = pd.DataFrame({"feature": feature_names(pipeline), "coefficient": coefs, "odds_ratio": np.exp(coefs)})
    return df.reindex(df["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def tree_importances(pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    df = pd.DataFrame({"feature": feature_names(pipeline), "importance": model.feature_importances_})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def shap_values(pipeline, X_raw: pd.DataFrame, max_rows: int = 1000):
    """SHAP values for a tree model on (a sample of) raw-format records."""
    import shap

    X = X_raw.sample(min(max_rows, len(X_raw)), random_state=0)
    transformed = pipeline[:-1].transform(X)
    names = list(feature_names(pipeline))
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    values = explainer(pd.DataFrame(transformed, columns=names))
    return values


def shap_importance(values) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": values.feature_names, "mean_abs_shap": np.abs(values.values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def leaf_rules(pipeline, X_raw: pd.DataFrame, y) -> pd.DataFrame:
    """One row per decision-tree leaf: the if/then path, size and churn rate.

    Thresholds are shown in original units; one-hot splits read as
    "Contract is Month-to-month" / "Contract is not Month-to-month".
    """
    prep = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = list(feature_names(pipeline))
    scaler = prep.named_transformers_["num"]
    num_cols = list(prep.transformers_[0][2])
    unscale = {c: (m, s) for c, m, s in zip(num_cols, scaler.mean_, scaler.scale_, strict=True)}
    cat_cols = list(prep.transformers_[2][2])
    tree = model.tree_

    def describe(feat: str, go_right: bool, thr: float) -> str:
        if feat in unscale:
            m, s = unscale[feat]
            return f"{feat} {'>' if go_right else '<='} {thr * s + m:.1f}"
        for c in cat_cols:
            if feat.startswith(c + "_"):
                return f"{c} {'is' if go_right else 'is not'} {feat[len(c) + 1 :]}"
        return f"{feat} = {'Yes' if go_right else 'No'}"

    paths: dict[int, list[str]] = {}

    def walk(node: int, conds: list[str]) -> None:
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:  # leaf
            paths[node] = conds
            return
        feat, thr = names[tree.feature[node]], tree.threshold[node]
        walk(left, conds + [describe(feat, False, thr)])
        walk(right, conds + [describe(feat, True, thr)])

    walk(0, [])
    leaves = model.apply(pipeline[:-1].transform(X_raw))
    stats = pd.DataFrame({"leaf": leaves, "y": np.asarray(y)}).groupby("leaf")["y"].agg(["size", "mean"])
    rows = [
        {"rule": " AND ".join(paths[leaf]), "customers": int(r["size"]), "churn_rate": float(r["mean"])}
        for leaf, r in stats.iterrows()
    ]
    return pd.DataFrame(rows).sort_values("churn_rate", ascending=False).reset_index(drop=True)
