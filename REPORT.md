# Customer Churn Model — Findings for the Retention Team

All figures come from `reports/metrics.json`, `reports/model_comparison.csv`, `reports/test_decile_lift.csv`, `reports/calibration_test_set.csv` and `reports/subgroup_metrics_test_set.csv`, produced by `scripts/train.py` on the IBM Telco sample (7,043 customers; 26.5% churned in the last month). Model results are on 1,409 held-out customers the models never saw during training or tuning — a random holdout from one historical snapshot, scored offline. No retention offers were sent; "flagged" means the model would have selected the customer for outreach.

## 1. Who is leaving

| Segment | Churn rate | Customers |
|---|---|---|
| Month-to-month contract | 42.7% | 3,875 |
| One-year contract | 11.3% | 1,473 |
| Two-year contract | 2.8% | 1,695 |
| Tenure 0–6 months | 52.9% | 1,481 |
| Tenure 49–72 months | 9.5% | 2,239 |
| Fiber-optic internet | 41.9% | 3,096 |
| Pays by electronic check | 45.3% | 2,365 |
| **All customers** | **26.5%** | **7,043** |

The highest-risk profile is a **month-to-month fiber-optic customer in their first ~15 months, paying by electronic check, without online security or tech support**. Among internet customers, churn falls steadily with each add-on service held (from about half of those with none to about 5% of those with all six).

These are associations in historical data, not proven causes; they indicate where to test interventions.

## 2. How well we can predict it

Three models were compared, from most to least interpretable:

| | Decision tree (baseline) | Logistic regression | XGBoost (selected) |
|---|---|---|---|
| Ranking quality, test ROC-AUC | 0.838 | 0.846 | 0.850 |
| Churners caught (recall) | 84.8% | 80.7% | 80.5% |
| Offers that reach a churner (precision) | 48.5% | 52.0% | 51.9% |
| Customers flagged (churners among them) | 654 (317) | 581 (302) | 580 (301) |

* **XGBoost ranks customers better than the baseline tree.** On the same 1,409 customers, the 95% bootstrap interval for the AUC difference is +0.003 to +0.022. At their own thresholds the tree flags 74 more customers and catches 16 more churners, so per flag XGBoost is more precise (51.9% vs 48.5%); flag counts alone are not a like-for-like comparison.
* **XGBoost's edge over logistic regression is small and not clearly separable on this test set** (+0.004 AUC; 95% interval −0.0002 to +0.008). This does not show the two are equivalent — only that this data cannot tell them apart with confidence. XGBoost was selected by the rule fixed in advance (best cross-validated AUC); logistic regression is a sound alternative if directly readable coefficients matter more.
* **Accuracy is not the right headline.** At the default 50% cut-off all three models are about 80% accurate but miss roughly half of churners. The operating cut-off (27.4% probability) trades some accuracy (75%) for catching four in five churners.

## 3. How to use the scores

**If the team has a fixed outreach budget**, rank customers by probability and contact from the top:

| Contact the riskiest… | Churn rate in that group | Share of all churners reached |
|---|---|---|
| 10% | 75.9% (2.9× average) | 28.6% |
| 20% | — | 49.7% |
| 30% | — | 67.4% |
| 40% | — | 79.1% |

**If the goal is coverage**, use the `churn_flag`. (The ranking table above and the flag figures below describe different targeting policies, so they are not directly comparable.) On the test set it flagged 580 of 1,409 customers; 301 of them churned (51.9%), and 73 churners were not flagged.

Risk tiers returned by the API, with actual test-set churn rates:

| Tier | Rule | Customers | Actual churn rate |
|---|---|---|---|
| High | probability ≥ 50% | 301 | 65.4% |
| Medium | 27.4% ≤ probability < 50% | 279 | 37.3% |
| Low | below 27.4% | 829 | 8.8% |

**Sample report — highest-risk held-out customers** (full list: `reports/sample_predictions_test_set.csv`)

| customerID | Contract | Tenure | Internet | Payment | Monthly $ | Churn probability | Tier | Actually churned |
|---|---|---|---|---|---|---|---|---|
| 5419-JPRRN | Month-to-month | 1 | Fiber optic | Electronic check | 101.45 | 0.925 | high | yes |
| 4910-GMJOT | Month-to-month | 1 | Fiber optic | Electronic check | 94.60 | 0.910 | high | yes |
| 0107-YHINA | Month-to-month | 1 | Fiber optic | Electronic check | 99.75 | 0.906 | high | yes |
| 8149-RSOUN | Month-to-month | 1 | Fiber optic | Electronic check | 93.85 | 0.900 | high | yes |
| 6521-YYTYI | Month-to-month | 1 | Fiber optic | Electronic check | 93.30 | 0.886 | high | yes |

### Can the probabilities be taken at face value?

Yes, on this data. Predicted and observed churn rates agree closely across score bands: the expected calibration error is about 2 percentage points, and the Brier score is 0.134 against 0.195 for a constant-rate guess (`reports/calibration_test_set.csv`). Recalibration is not needed.

### Does it work equally across customer groups?

| Group | Test customers | Actual churn rate | Flagged | Recall | Precision |
|---|---|---|---|---|---|
| Female | 716 | 25.7% | 40.9% | 81.0% | 50.9% |
| Male | 693 | 27.4% | 41.4% | 80.0% | 53.0% |
| Not senior citizen | 1,175 | 23.7% | 36.9% | 77.1% | 49.5% |
| Senior citizen | 234 | 40.6% | 62.4% | 90.5% | 58.9% |

Results by gender are close. Senior citizens are flagged much more often, in line with their higher churn rate in this sample; the group is small. Gender and senior-citizen status are model inputs, so whether they may be used — and whether outreach rules should differ by group — needs a fairness and legal review with the client before any real use.

## 4. Realistic expectations

* About **half of flagged customers would not have left anyway**, and about **one in five churners will not be flagged** at the current setting.
* The threshold is a business choice. It is set to catch ~80% of churners because missing a churner was described as the costlier error. **No offer cost or customer-value figures were available, so none were assumed.** With those two numbers the threshold can be set to maximise expected net savings instead.
* These results measure *prediction*, not *retention*. Whether an offer changes the outcome must be measured by holding back a random control group when offers go out.

## 5. Suggested interventions to test

1. Contract-upgrade incentives for month-to-month customers in their first year.
2. Bundled online security / tech support for fiber-optic customers without them.
3. Auto-pay enrolment incentive for electronic-check payers.

## 6. Operating the model

* Scores are available locally via CLI (`scripts/predict_churn.py`) or REST API (`/predict`, `/predict/batch`), with a Dockerfile for container packaging. This is a prototype, not a hosted production service.
* Every prediction is logged with a hashed customer ID and no demographic fields. `scripts/drift_report.py` flags shifts in the incoming customer mix and unseen values. It was demonstrated on a synthetic batch (15% higher monthly charges with cumulative charges left unchanged, plus a new "Digital wallet" payment method); both were flagged. This tests the monitoring, not real-world model accuracy, which requires joining later churn outcomes to past predictions.
* Recommended cadence: run the drift report monthly; retrain when key features show significant shift or when new labelled churn data is available.

## 7. Limitations and next steps

* Random holdout of a single historical snapshot; production training and validation should use monthly point-in-time snapshots, evaluated on later months.
* No cost data; add offer cost and customer lifetime value to set the threshold economically.
* Measure true intervention impact with a control group before scaling offers.
* Full list, including privacy and fairness controls: `LIMITATIONS_AND_PRODUCTION_PLAN.md`.
