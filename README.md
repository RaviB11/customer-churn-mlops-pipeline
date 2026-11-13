# Telco Customer Churn Prediction

[![CI](https://github.com/RaviB11/customer-churn-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/RaviB11/customer-churn-mlops-pipeline/actions/workflows/ci.yml)

End-to-end churn-risk scoring prototype for a hypothetical telecom retention team: simulated SQLite warehouse → cleaning → feature engineering → three models (decision tree, logistic regression, tuned XGBoost) → packaged predictor, CLI, REST API and Dockerfile → privacy-minimised prediction logging and drift monitoring.

**Scope, plainly:** a locally served, container-packaged prototype trained and evaluated on a random holdout of one public historical snapshot. Drift monitoring is demonstrated on synthetic traffic. "Flagged customers" are offline predictions — no retention offers were sent, so nothing here measures churn prevented or ROI. See [`LIMITATIONS_AND_PRODUCTION_PLAN.md`](LIMITATIONS_AND_PRODUCTION_PLAN.md).

**Data:** [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) — 7,043 customers, 21 columns, CC BY 4.0.

## Results (held-out test set, 1,409 customers)

| Model | CV ROC-AUC (5-fold) | Test ROC-AUC | Test PR-AUC | Recall @ threshold | Precision @ threshold | Flagged (churners caught) |
|---|---|---|---|---|---|---|
| Decision tree (baseline, depth 4) | 0.829 ± 0.016 | 0.838 | 0.613 | 84.8% | 48.5% | 654 (317) |
| Logistic regression | 0.847 ± 0.016 | 0.846 | 0.664 | 80.7% | 52.0% | 581 (302) |
| **XGBoost, tuned (selected)** | **0.850 ± 0.016** | **0.850** | **0.673** | **80.5%** | **51.9%** | **580 (301)** |

Thresholds are chosen on out-of-fold *training* predictions as the highest cut-off that still catches ≥ 80% of churners. Full table: `reports/model_comparison.csv`.

Takeaways:
* XGBoost beats the tree baseline (paired-bootstrap 95% CI for the test-AUC difference: +0.003 to +0.022).
* Its edge over logistic regression is small and **not clearly separable on this test set** (+0.004; 95% CI −0.0002 to +0.008). That is not proof of equivalence; logistic regression is a defensible choice if readable coefficients matter more.
* Probabilities are well calibrated (Brier 0.134 vs 0.195 for a constant-rate guess; expected calibration error 0.021), so the risk tiers can be read at face value.
* The top 10% of customers by score churned at 2.9× the average rate.
* Subgroup recall and precision by gender and senior-citizen status are in `reports/subgroup_metrics_test_set.csv`. See [`REPORT.md`](REPORT.md) for the business write-up and `notebooks/churn_analysis.ipynb` for the full analysis with charts.

## Quick start

Requires Python 3.12.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python scripts\train.py                                     # ~1-2 min: builds warehouse, trains, saves model
python scripts\predict_churn.py --input data\sample\new_customers.csv
pytest -q
```

**macOS / Linux**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/train.py
python scripts/predict_churn.py --input data/sample/new_customers.csv
pytest -q
```

### Scoring new customers (CLI)

```powershell
python scripts\predict_churn.py --input customers.csv --output predictions.csv
python scripts\predict_churn.py --json '{"gender":"Female","SeniorCitizen":0, ...}'
```
Input is the CRM's raw format (same columns as the source CSV, `Churn` not needed; blank `TotalCharges` is fine for new customers). Output per customer: `churn_probability`, `churn_flag` (at or above the model's threshold), and `risk_tier` (high / medium / low).

### REST API

```powershell
uvicorn churn_pipeline.api:app --port 8000
# Interactive docs: http://localhost:8000/docs
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -InFile data\sample\customer.json
```
Endpoints: `GET /health`, `POST /predict` (one record), `POST /predict/batch` (up to 5,000). Malformed records return HTTP 422 with the reason; an unseen category (e.g. a new payment method) is scored and returned with a warning.

### Docker

```powershell
docker build -t telco-churn .
docker run -p 8000:8000 -v ${PWD}/monitoring:/app/monitoring telco-churn
```
The image trains the model during the build so the pickled pipeline always matches the image's library versions. The Dockerfile has not yet been built and run outside CI; a production design would use a separate inference-only image that pulls a registered model.

Only load model bundles you trained yourself: joblib/pickle files can execute code when loaded.

### Monitoring

```powershell
python scripts\simulate_traffic.py --n 500 --reset   # replays the model's holdout customers, plus a synthetic drifted batch
python scripts\drift_report.py --last 500            # PSI per feature vs training profile
```
Every prediction (CLI or API) is appended to `monitoring/prediction_log.jsonl` with timestamp, request ID, model version, non-demographic inputs, score and flag. Customer IDs are stored only as a salted SHA-256 hash (set `CHURN_LOG_SALT` outside the demo); demographic fields are logged only if `CHURN_LOG_DEMOGRAPHICS=1`. The drift report compares recent traffic with the training distribution stored in the model bundle (PSI; rule-of-thumb bands < 0.10 stable, 0.10–0.25 moderate, > 0.25 significant) and lists unseen categories by name.

### Experiment tracking

Each `train.py` run is logged to MLflow (SQLite backend, `mlflow.db`) with parameters, CV and test metrics for every model, the model bundle, and the SHA-256 of the source CSV as the data version (`data_sha256` tag; also in `data/manifest.json`).
```powershell
pip install mlflow   # full package, for the UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Set `CHURN_DISABLE_MLFLOW=1` to skip tracking (CI does this).

## Project layout

```
churn_pipeline/
  config.py       paths, column groups, expected schema, threshold policy
  etl.py          CSV load, SQLite warehouse (4 normalised tables), SQL join, data hashing
  cleaning.py     TotalCharges fix, Yes/No encoding, integrity checks
  features.py     engineered features + sklearn preprocessing; full serving pipeline
  modeling.py     split, CV, grid search, threshold selection, model selection, MLflow, bundle
  evaluation.py   ROC/PR-AUC, precision/recall, lift, decile table
  explain.py      decision-tree rules, logistic coefficients, SHAP
  predict.py      ChurnPredictor (validate -> score -> log)
  monitoring.py   schema validation, privacy-minimised prediction log, PSI drift report
  api.py          FastAPI service
scripts/          train.py, predict_churn.py, simulate_traffic.py, drift_report.py,
                  make_sample_customers.py (rebuilds data/sample from the model's holdout)
tests/            pytest suite (data quality, warehouse integrity, cleaning, features, leakage guards, models,
                  evaluation, predictor, CLI, API, monitoring/privacy)
notebooks/        churn_analysis.ipynb (executed, with outputs)
reports/          metrics, comparison table, decile lift, calibration, subgroup metrics, sample predictions,
                  drift report, figures/
LIMITATIONS_AND_PRODUCTION_PLAN.md   what this prototype does not show, and what production would need
data/             raw CSV, sample inputs, manifest (warehouse DB is generated)
```

## Method notes

* **No test-set tuning.** 80/20 stratified split; hyper-parameters, threshold and the shipped model are all chosen by 5-fold CV on the training set. The test set is scored once.
* **One pipeline for training and serving.** The saved sklearn pipeline starts from raw CRM records (cleaning → features → encoding → model), so there is no train/serve skew.
* **Leakage checks** (in `tests/test_features.py`): target and ID are never inputs; flipping the `Churn` column leaves every score unchanged; no single numeric feature or category level has AUC ≥ 0.95; a model on shuffled labels scores at chance. These guard against obvious target contamination; without event timestamps they cannot prove every feature was available at prediction time. `TotalCharges` is not derived from the outcome but is ~99.96% correlated with tenure × monthly charge, so it adds little on its own.
* **One split everywhere.** Rows are sorted by `customerID` before the split, so the CSV and warehouse paths produce identical holdout sets. The holdout IDs are saved in the model bundle, and the sample customers and simulated traffic are drawn only from them (all regression-tested).
* **Warehouse integrity.** Loading checks key uniqueness, missing and orphan rows across the four tables and that the join keeps every customer.
* **Known limitations:** random holdout of a single snapshot (for churned customers `tenure` is measured at departure); XGBoost hyper-parameters were tuned on the same folds later used for the out-of-fold threshold, which may bias the threshold slightly (the held-out test result is unaffected). Both are covered in the limitations document.

## CI

`.github/workflows/ci.yml` is defined but **has not yet run on GitHub**. On every push and pull request it runs: ruff lint and format check, the pytest suite on Python 3.11 and 3.12, a fast training smoke run, then builds the Docker image and calls the API inside the container.

## License and attribution

Dataset: IBM, CC BY 4.0. Code: add the license of your choice.
