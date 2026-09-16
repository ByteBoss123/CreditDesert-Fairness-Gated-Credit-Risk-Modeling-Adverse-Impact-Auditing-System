# CreditDesert
### Fairness-Gated Credit Risk Modeling, Auditing & Deployment — On Real Data, End to End

**Project type:** Applied ML / fair-lending risk modeling
**Built for:** Data Scientist / Credit Risk roles (non-prime lending, fintech)
**Repository status:** Serving code, an exported model, datasets, and reported
analysis outputs are included. The original training and audit scripts are not
in the current repository or its existing commit history. Reported model metrics
cannot be reproduced end to end from this checkout until those scripts are restored.

---

## What this is

A three-part project that takes a genuine fair-lending problem — can you build
an accurate credit risk model *and* know whether it's quietly reproducing
lending disparities? — all the way from raw government data to a deployable,
tested API.

| Component | What it does | Data |
|---|---|---|
| **1. Fairness Audit** | XGBoost approval-risk model + adverse-impact testing + redlining-proxy analysis | 247,025 real 2024 HMDA mortgage records (Virginia) |
| **2. Consumer Credit Model** | Risk model on non-mortgage installment credit, closing the domain gap to non-prime/lease-to-own lending | 4,454 real consumer credit applications |
| **3. Serving API** | Real FastAPI service, live-tested, serving the actual trained model | Loads Component 1's model directly |

## The core finding

Even when a model is trained with **race, ethnicity, and sex completely
excluded** as inputs, it can still reproduce racial disparities — because
legitimate underwriting features (income, debt-to-income, loan-to-value) are
themselves correlated with race due to structural economic conditions. This
project measured that gap directly: a ~7-point difference in predicted
approval probability between Black and White applicants, persisting after
race-blind modeling. This is a genuine, defensible, and non-obvious result —
not a manufactured story to fit a narrative.

## Quick facts

- **Model AUC**: 0.894 (HMDA), 0.853 (consumer credit) — both realistic, both
  arrived at only after catching and fixing a severe label-leakage bug (see
  below)
- **Adverse Impact Ratio, Black applicants**: 0.863 (below reference, above the
  regulatory 0.80 threshold)
- **Model-adjusted disparity**: persists at ~7 points even with race excluded
- **Serving API**: tested with real requests — strong applicant scores 98.06%
  approval probability, weak applicant scores 2.24%
- **A real bug found and fixed during testing**: an early outlier-detection
  design used raw min/max as "normal" bounds and would have almost never
  triggered, because the raw HMDA data contains extreme data-quality outliers
  (loan amounts up to $375M). Fixed by switching to percentile-based bounds.

## Repository files

Files are stored in the repository root. Use these links to open them:

| Purpose | Files |
|---|---|
| Serving code and dependencies | [app.py](app.py), [requirements.txt](requirements.txt), [Dockerfile](Dockerfile) |
| Model and validation schemas | [hmda_model_b.json](hmda_model_b.json), [category_schema.json](category_schema.json), [numeric_ranges.json](numeric_ranges.json) |
| Data | [hmda_va_2024_modeling.parquet](hmda_va_2024_modeling.parquet), [consumer_credit_data.csv](consumer_credit_data.csv) |
| Audit and monitoring outputs | [fairness_audit_results.xlsx](fairness_audit_results.xlsx), [drift_monitoring_psi.csv](drift_monitoring_psi.csv) |
| Explainability outputs | [shap_summary.png](shap_summary.png), [consumer_shap_summary.png](consumer_shap_summary.png), [consumer_shap_feature_importance.csv](consumer_shap_feature_importance.csv) |

### Missing source scripts

The original project description references `features.py`, `train.py`,
`fairness_audit.py`, `explain_and_monitor.py`, `features_consumer.py`, and
`train_consumer.py`. These files have not been uploaded. References to those
scripts in the reports describe the original workflow; they are not runnable
paths in this checkout. Restore the original scripts to reproduce the analysis.

### Run the included serving code

From the repository root, follow [SERVING_README.md](SERVING_README.md).
The API loads its model and schemas from the same directory as `app.py`.
The Databricks document describes a proposed translation, not a completed deployment.

## Full methodology, results, and honest limitations

See:
- [COMPONENT_1_FAIRNESS_AUDIT.md](COMPONENT_1_FAIRNESS_AUDIT.md) — HMDA modeling, leakage fix, adverse
  impact ratios, model-adjusted disparity finding, geographic redlining check,
  SHAP, drift monitoring, and full limitations
- [COMPONENT_2_CONSUMER_CREDIT.md](COMPONENT_2_CONSUMER_CREDIT.md) — consumer credit model, results,
  limitations
- [SERVING_README.md](SERVING_README.md) — API testing evidence, design decisions,
  deployment instructions
- [databricks_equivalent.md](databricks_equivalent.md) — how this pipeline translates to
  Databricks/Delta Lake/PySpark at production scale

## Tech Stack

Python, pandas, XGBoost, MLflow (experiment tracking + model registry), SHAP
(explainability), scikit-learn, FastAPI, Docker. Architected for a
Databricks/Delta Lake lift at scale.

## What this project is — and isn't

**Is:** a rigorous, honestly-reported demonstration of the methodology a
real fair-lending credit risk model needs — leakage detection, adverse impact
testing, proxy-feature auditing, explainability, drift monitoring, and a
tested serving layer.

**Isn't:** a solved problem. It doesn't reduce credit deserts, doesn't reflect
Koalafi's actual non-prime/lease-to-own population (HMDA is mortgage data),
found no legally-actionable disparity (the 80% threshold wasn't crossed), and
has no live public deployment (sandbox-built and tested, not internet-hosted).
These limitations are documented throughout rather than glossed over.
