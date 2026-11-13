"""Training: baseline decision tree, logistic regression and tuned XGBoost.

Discipline used here, so the numbers in the report are honest:
* 80/20 stratified split; the test set is touched once, at the very end.
* All model choices (hyper-parameters, decision threshold, which model ships)
  are made on 5-fold cross-validation inside the training set.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config, etl
from .cleaning import clean_dataset
from .evaluation import choose_threshold, compute_metrics
from .features import build_model_pipeline, clean_and_engineer
from .monitoring import build_reference_profile

log = logging.getLogger(__name__)

XGB_GRID = {
    # First pass (depth 2-4, lr 0.03-0.1, 200-400 trees) chose the shallowest,
    # slowest corner, so the grid was widened in that direction.
    "model__max_depth": [1, 2, 3],
    "model__learning_rate": [0.01, 0.03, 0.1],
    "model__n_estimators": [200, 400],
    "model__min_child_weight": [5, 10],
    "model__colsample_bytree": [0.8, 1.0],
    "model__subsample": [0.8],
}
XGB_GRID_FAST = {"model__max_depth": [2, 3], "model__n_estimators": [100]}


@dataclass
class ModelResult:
    name: str
    pipeline: object
    threshold: float
    cv_auc_mean: float
    cv_auc_std: float
    oof_prob: np.ndarray
    test_prob: np.ndarray
    test_metrics: dict
    test_metrics_at_05: dict
    best_params: dict = field(default_factory=dict)


@dataclass
class TrainingRun:
    results: dict
    best_name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ids_test: pd.Series
    bundle_path: Path | None
    data_sha256: str


def load_modelling_data(source: str = "warehouse") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return raw-format features X, target y and customer IDs.

    X stays in raw CRM format because the serving pipeline cleans it itself;
    clean_dataset is still run here to enforce integrity checks and build y.
    """
    if source == "warehouse":
        db = config.WAREHOUSE_DB
        if not db.exists():
            etl.build_warehouse(db_path=db)
        raw = etl.load_from_warehouse(db)
    else:
        raw = etl.load_raw_csv()
    # train_test_split is positional, so both sources must share one row order
    # or they produce different holdout sets (audit P1-A).
    raw = raw.sort_values(config.ID_COL, kind="stable").reset_index(drop=True)
    clean = clean_dataset(raw)
    return raw[config.RAW_FEATURE_COLS], clean[config.TARGET_COL], raw[config.ID_COL]


def split_holdout(X, y, ids):
    """The one train/holdout split used everywhere (training, sample customers,
    traffic simulation). Returns X_tr, X_te, y_tr, y_te, ids_tr, ids_te."""
    return train_test_split(
        X, y, ids, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )


def load_holdout_records(model_path: Path | str | None = None) -> pd.DataFrame:
    """Raw-format records (with Churn) for the holdout customers recorded in a
    model bundle, so demos always use customers that model never trained on."""
    bundle = joblib.load(model_path or config.MODEL_PATH)
    if "holdout_ids" not in bundle:
        raise KeyError("Model bundle has no holdout_ids; retrain with scripts/train.py")
    raw = etl.load_raw_csv().set_index(config.ID_COL)
    return raw.loc[bundle["holdout_ids"]].reset_index()


def model_specs(fast: bool = False) -> dict:
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=config.RANDOM_STATE,
    )
    return {
        "decision_tree": (
            DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=config.RANDOM_STATE),
            None,
        ),
        "logistic_regression": (LogisticRegression(max_iter=5000, C=1.0), None),
        "xgboost": (xgb, XGB_GRID_FAST if fast else XGB_GRID),
    }


def _fit_one(name, estimator, grid, X_tr, y_tr, X_te, y_te, cv, target_recall) -> ModelResult:
    pipe = build_model_pipeline(estimator)
    best_params: dict = {}
    if grid:
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_tr, y_tr)
        pipe = search.best_estimator_
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        i = search.best_index_
        cv_mean = float(search.cv_results_["mean_test_score"][i])
        cv_std = float(search.cv_results_["std_test_score"][i])
        log.info("%s best params %s (CV AUC %.4f)", name, best_params, cv_mean)

    # Out-of-fold probabilities on the training set -> threshold choice
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    if not grid:
        from sklearn.metrics import roc_auc_score

        fold_aucs = [roc_auc_score(y_tr.iloc[va], oof[va]) for _, va in cv.split(X_tr, y_tr)]
        cv_mean, cv_std = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
        pipe.fit(X_tr, y_tr)

    threshold = choose_threshold(y_tr, oof, target_recall)
    test_prob = pipe.predict_proba(X_te)[:, 1]
    return ModelResult(
        name=name,
        pipeline=pipe,
        threshold=threshold,
        cv_auc_mean=cv_mean,
        cv_auc_std=cv_std,
        oof_prob=oof,
        test_prob=test_prob,
        test_metrics=compute_metrics(y_te, test_prob, threshold),
        test_metrics_at_05=compute_metrics(y_te, test_prob, 0.5),
        best_params=best_params,
    )


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for r in results.values():
        m = r.test_metrics
        rows.append(
            {
                "model": r.name,
                "cv_roc_auc": round(r.cv_auc_mean, 4),
                "cv_roc_auc_std": round(r.cv_auc_std, 4),
                "test_roc_auc": round(m["roc_auc"], 4),
                "test_pr_auc": round(m["pr_auc"], 4),
                "threshold": round(r.threshold, 4),
                "test_recall": round(m["recall"], 4),
                "test_precision": round(m["precision"], 4),
                "test_f1": round(m["f1"], 4),
                "test_accuracy": round(m["accuracy"], 4),
                "accuracy_at_0.5": round(r.test_metrics_at_05["accuracy"], 4),
                "recall_at_0.5": round(r.test_metrics_at_05["recall"], 4),
                "top10_lift": round(m["top10_lift"], 3),
                "customers_flagged": m["customers_flagged"],
                "churners_missed": m["churners_missed"],
            }
        )
    return pd.DataFrame(rows)


def _log_mlflow(run: TrainingRun, fast: bool) -> None:
    if os.getenv("CHURN_DISABLE_MLFLOW") == "1":
        return
    try:
        import mlflow
    except ImportError:  # tracking is optional
        log.warning("mlflow not installed; skipping experiment tracking")
        return
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"train-{datetime.now():%Y%m%d-%H%M%S}"):
        mlflow.set_tags({"data_sha256": run.data_sha256, "fast_mode": str(fast)})
        mlflow.log_params(
            {
                "test_size": config.TEST_SIZE,
                "cv_folds": config.CV_FOLDS,
                "target_recall": config.TARGET_RECALL,
                "selected_model": run.best_name,
            }
        )
        for r in run.results.values():
            with mlflow.start_run(run_name=r.name, nested=True):
                mlflow.set_tag("data_sha256", run.data_sha256)
                mlflow.log_params(r.best_params or {"default_params": True})
                mlflow.log_metric("cv_roc_auc", r.cv_auc_mean)
                mlflow.log_metric("cv_roc_auc_std", r.cv_auc_std)
                for k, v in r.test_metrics.items():
                    mlflow.log_metric(f"test_{k}", float(v))
        if run.bundle_path:
            mlflow.log_artifact(str(run.bundle_path))


def train_all(
    source: str = "warehouse",
    fast: bool = False,
    target_recall: float = config.TARGET_RECALL,
    save_path: Path | str | None = config.MODEL_PATH,
    track: bool = True,
) -> TrainingRun:
    X, y, ids = load_modelling_data(source)
    X_tr, X_te, y_tr, y_te, _, ids_te = split_holdout(X, y, ids)
    cv = StratifiedKFold(
        n_splits=3 if fast else config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    results = {}
    for name, (est, grid) in model_specs(fast).items():
        log.info("Training %s", name)
        results[name] = _fit_one(name, est, grid, X_tr, y_tr, X_te, y_te, cv, target_recall)

    # Ship the model with the best *cross-validated* AUC (not test AUC).
    best_name = max(results, key=lambda k: results[k].cv_auc_mean)
    data_sha = etl.file_sha256(config.RAW_CSV)

    bundle_path = None
    if save_path is not None:
        bundle_path = save_bundle(
            results[best_name], X_tr, data_sha, target_recall, save_path, holdout_ids=list(ids_te)
        )

    run = TrainingRun(results, best_name, X_tr, X_te, y_tr, y_te, ids_te, bundle_path, data_sha)
    if track:
        _log_mlflow(run, fast)
    return run


def save_bundle(result: ModelResult, X_train, data_sha, target_recall, path, holdout_ids=None) -> Path:
    """Persist everything the serving layer needs in one file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(timezone.utc)
    version = f"{result.name}-{trained_at:%Y%m%d%H%M}-{data_sha[:8]}"
    reference = build_reference_profile(clean_and_engineer(X_train), result.oof_prob)
    bundle = {
        "pipeline": result.pipeline,
        "model_name": result.name,
        "model_version": version,
        "threshold": result.threshold,
        "target_recall": target_recall,
        "trained_at_utc": trained_at.isoformat(timespec="seconds"),
        "data_sha256": data_sha,
        "raw_feature_cols": config.RAW_FEATURE_COLS,
        "cv_roc_auc": result.cv_auc_mean,
        "test_metrics": result.test_metrics,
        "best_params": result.best_params,
        "reference_profile": reference,
        "holdout_ids": holdout_ids or [],
        "env": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    joblib.dump(bundle, path)
    meta = {k: v for k, v in bundle.items() if k not in ("pipeline", "reference_profile", "holdout_ids")}
    meta["n_holdout_ids"] = len(bundle["holdout_ids"])
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    return path
