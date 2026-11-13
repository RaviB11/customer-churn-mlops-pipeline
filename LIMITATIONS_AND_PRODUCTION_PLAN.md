# Limitations and Production Plan

This repository is a **locally served, container-packaged prototype** built on the public IBM Telco sample. This page lists what it does not demonstrate and what a production version for a real telecom client would need. It is a plan, not a list of implemented features.

## What the results do and do not show

| The project shows | It does not show |
|---|---|
| Offline ranking and classification quality on a random 20% holdout of one historical snapshot | Performance on future months |
| Which customers the model *would* flag for outreach | That outreach prevents churn, or any revenue saved |
| Well-calibrated probabilities on this sample | That calibration holds as the customer base changes |
| Drift detection on synthetic, deliberately shifted traffic | Live monitoring of real customers or of model accuracy over time |
| Successful GitHub Actions CI checks, Docker image build, and containerized API smoke tests | A publicly hosted production service or real-customer traffic |
| A simulated four-table SQLite warehouse with integrity checks | Integration with a real data warehouse |

## 1. Validation design (temporal)

**Now:** stratified random holdout of a single snapshot. For churned customers, `tenure` is measured at departure.

**Production:** build monthly point-in-time snapshots — features as of the end of month *t*, label = churned during month *t+1* — then train on earlier months and evaluate on later ones (rolling-origin backtest). This tests whether the model generalises forward in time and confirms that every feature is available at scoring time, which the current leakage tests cannot prove without timestamps.

## 2. Threshold selection rigour

**Now:** XGBoost hyper-parameters are chosen by grid search on the five training folds, and the same folds then produce the out-of-fold scores used to set the threshold. Each out-of-fold score still comes from a model that never saw that customer's label, but the hyper-parameter choice did see those folds' results, so the threshold may be slightly optimistic. The held-out test result is unaffected.

**Production:** use nested cross-validation, or a three-way split (train / threshold-tuning / test), and re-check the threshold on each new backtest month.

## 3. Calibration over time

**Now:** expected calibration error ≈ 0.02 on the test set; no recalibration applied.

**Production:** recompute the calibration table each month once outcomes arrive. If the gap widens, refit a calibrator (isotonic or Platt) on recent labelled data. Risk tiers depend on calibrated probabilities.

## 4. Monitoring with real outcomes

**Now:** PSI on logged inputs and scores, plus unseen-category detection.

**Production:**
- Join each month's churn outcomes to past predictions (via the hashed customer ID) and track recall, precision and calibration by month.
- Alert when input drift, score drift or realised recall crosses an agreed limit.
- Retrain only after validating on new labelled data; do not retrain automatically on drift alone.

## 5. Privacy and data handling

**Now (implemented):** the prediction log stores a salted SHA-256 hash of the customer ID, excludes demographic fields by default, and keeps only the inputs needed for drift checks.

**Production:**
- Keep the salt in a secrets manager, never in code.
- Restrict log access to named roles.
- Set retention and rotation periods agreed with the client's privacy team.
- Encrypt logs at rest.
- Never publish logs containing real customers.
- Confirm the legal basis for using each input field.

## 6. Fairness and responsible use

**Now:** `gender` and `SeniorCitizen` are model inputs. Test-set subgroup results (`reports/subgroup_metrics_test_set.csv`):
- Recall and precision are close for men and women.
- Senior citizens are flagged more often, in line with their higher churn rate in the sample.

**Production:**
- Review with the client whether these attributes may be used, and compare a model without them.
- Agree acceptable subgroup gaps.
- Monitor subgroup recall and precision monthly.
- Check that retention offers themselves do not treat groups unfairly.

None of this constitutes a completed responsible-AI review.

## 7. Measuring business impact

**Now:** none. Flag counts are offline predictions; no offers were sent.

**Production:**
- Randomly hold back a control group among flagged customers.
- Compare churn between treated and control groups to estimate saves per offer.
- Combine with offer cost and customer value to set the threshold by expected net value instead of a fixed recall target.

## 8. Deployment architecture

**Now:** FastAPI serves predictions locally, and the Docker image has been successfully built and smoke-tested through GitHub Actions. The CI workflow validates dependencies, linting, automated tests, model training, predictions, Docker packaging, and API availability.

The Dockerfile retrains the model at image-build time to maintain dependency compatibility. However, this couples training with serving and packages training data into the image.

The project has not been deployed as a publicly hosted production service. It does not demonstrate production authentication, service availability, autoscaling, or live customer monitoring.

**Production:**
- Separate the training pipeline from a small inference-only image that pulls a versioned model from a model registry.
- Add authentication, TLS and rate limiting to the API.
- Add centralised logging and health monitoring.
- Only ever load model artifacts produced by the team's own pipeline, because pickle files can execute code.

## 9. Data platform

**Now:** a simulated warehouse — one CSV split into four SQLite tables, with key-uniqueness, orphan-row and join-cardinality checks on load.

**Production:**
- Read from the client's warehouse through governed views.
- Enforce primary and foreign keys at the source.
- Add data-contract tests on schema and category values before training or scoring.
