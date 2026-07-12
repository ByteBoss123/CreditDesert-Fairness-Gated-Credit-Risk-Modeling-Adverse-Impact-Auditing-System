# CreditDesert
### Fairness-Gated Credit Risk Modeling, Auditing & Deployment — On Real Data, End to End

**Project type:** Applied ML / fair-lending risk modeling
**Built for:** Data Scientist / Credit Risk roles (non-prime lending, fintech)
**Every number in this repo came from a script in this repo, run against real data.**
Nothing here is simulated, placeholder, or fabricated — including the mistakes,
which are documented rather than hidden.

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

## Repo Structure

```
CreditDesert/
├── README.md                          # This file
├── databricks_equivalent.md           # Honest translation to Databricks/PySpark at scale
├── data/
│   ├── hmda_va_2024_modeling.parquet  # Component 1: real HMDA data, cleaned
│   └── consumer_credit_data.csv       # Component 2: real consumer credit data
├── src/
│   ├── features.py                    # Component 1: feature engineering
│   ├── train.py                       # Component 1: XGBoost + MLflow, two-variant comparison
│   ├── fairness_audit.py              # Component 1: adverse impact ratio, model-adjusted disparity
│   ├── explain_and_monitor.py         # Component 1: SHAP + PSI drift monitoring
│   ├── features_consumer.py           # Component 2: feature engineering
│   └── train_consumer.py              # Component 2: XGBoost + MLflow
├── serving/
│   ├── SERVING_README.md              # Deployment details and honest scope limits
│   ├── app.py                         # FastAPI serving application
│   ├── hmda_model_b.json              # Real trained model (exported)
│   ├── category_schema.json           # Valid input codes, extracted from real data
│   ├── numeric_ranges.json            # Percentile-based outlier bounds
│   ├── Dockerfile
│   └── requirements.txt
└── outputs/
    ├── fairness_audit_results.xlsx    # Component 1: AIR tables, model-adjusted disparity, geographic check
    ├── shap_summary.png               # Component 1: SHAP explainability plot
    ├── drift_monitoring_psi.csv       # Component 1: PSI drift check
    ├── consumer_shap_summary.png      # Component 2: SHAP explainability plot
    └── consumer_shap_feature_importance.csv
```

## Full methodology, results, and honest limitations

See:
- **`docs/COMPONENT_1_FAIRNESS_AUDIT.md`** — HMDA modeling, leakage fix, adverse
  impact ratios, model-adjusted disparity finding, geographic redlining check,
  SHAP, drift monitoring, and full limitations
- **`docs/COMPONENT_2_CONSUMER_CREDIT.md`** — consumer credit model, results,
  limitations
- **`serving/SERVING_README.md`** — API testing evidence, design decisions,
  deployment instructions
- **`databricks_equivalent.md`** — how this pipeline translates to
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
