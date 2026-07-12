# Component 1: Fairness Audit & Risk Model (HMDA)

A credit approval risk model and fair-lending audit built on **247,025 real 2024
mortgage loan applications from Virginia**, sourced directly from the CFPB's public
HMDA (Home Mortgage Disclosure Act) database. Extends the fairness-gating
architecture used in TenantShield to a regulated credit-decisioning context.

**Every number in this README came from a script in this repo, run against real
government data — nothing here is simulated, estimated, or placeholder.**

---

## The Problem

Non-prime and near-prime lenders operate in a segment where credit decisions carry
real regulatory scrutiny: the Equal Credit Opportunity Act and Fair Housing Act
prohibit lending discrimination, and disparate-impact liability applies even when
no discriminatory intent exists. A risk model that is accurate but silently
reproduces historical lending disparities is a genuine legal and business risk,
not just an ethics footnote — this is exactly the tension a production credit
model at a company like Koalafi has to manage every day.

This project asks two questions on real data:
1. Can we build an accurate approval/denial risk model using only legitimate
   underwriting signals?
2. Does that model — even with race, ethnicity, and sex fully excluded as
   inputs — still reproduce lending disparities through proxy correlation?

## Data

- **Source**: [CFPB HMDA Snapshot National Loan-Level Dataset, 2024](https://ffiec.cfpb.gov/data-publication/2024), filtered to Virginia
- **Raw size**: 351,177 records, 99 fields
- **Modeling set**: 247,025 records after excluding withdrawn (4), incomplete (5),
  and purchased-loan (6) actions, which aren't underwriting decisions
- **Target**: `approved` = 1 for originated/approved-not-accepted loans (action codes
  1, 2, 8), 0 for denials (codes 3, 7) — the standard HMDA fair-lending convention
- **Baseline approval rate**: 77.89%

## Methodology

### 1. Leakage check (before trusting any metric)
An initial model using all available loan fields scored AUC = 0.9998 — an
unrealistic result that triggered a leakage investigation rather than a
celebration. Checking field missingness by outcome confirmed the cause:
`interest_rate`, `total_loan_costs`, and `rate_spread` are **100% missing for
denied applications** — they're only populated once a loan actually closes, so
their presence/absence directly encodes the label. All three were removed. The
reported model below uses only fields genuinely available at underwriting time
(loan amount, income, DTI, requested LTV, loan structure, tract economics).

### 2. Risk model
XGBoost binary classifier, trained on an 80/20 stratified split (197,620 train /
49,405 test), 300 rounds with early stopping, tracked in MLflow.

Two variants were trained to test a specific redlining-proxy hypothesis:

| Variant | Features | Test AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Model A** | includes `tract_minority_population_percent` | 0.8939 | 0.8934 | 0.9771 | 0.9334 |
| **Model B** | excludes it | 0.8944 | 0.8931 | 0.9772 | 0.9333 |

**Finding**: removing tract-level minority composition cost essentially zero
predictive performance (AUC actually ticked up slightly). This field carries
almost no legitimate underwriting signal — its main function is as a
correlate of race — so there's no accuracy trade-off in excluding it. All
fairness results below use **Model B**.

### 3. Fairness audit
Standard fair-lending metrics: raw approval-rate disparity and the **Adverse
Impact Ratio (AIR)** — the CFPB/EEOC "80% rule," where a group's approval rate
below 80% of the reference group's rate is a regulatory red flag.

| Group | Approval rate | AIR vs. reference | n |
|---|---|---|---|
| White (reference) | 80.59% | 1.000 | 139,304 |
| Asian | 78.31% | 0.972 | 18,986 |
| Black or African American | 69.55% | 0.863 | 34,408 |
| American Indian/Alaska Native | 68.16% | 0.846 | 1,228 |
| Not Hispanic or Latino (reference) | 78.80% | 1.000 | 180,729 |
| Hispanic or Latino | 73.17% | 0.929 | 17,253 |
| Male (reference) | 75.75% | 1.000 | 85,807 |
| Female | 74.15% | 0.979 | 56,483 |

None of these cross the 0.80 regulatory threshold in this dataset, but the
Black/White gap (13 points raw, AIR 0.863) is the largest and closest to the
line, and is worth flagging as a monitoring priority.

### 4. Does removing race actually remove the disparity?
This is the core question, and the honest answer is **no, not fully**. Model B
never sees race, ethnicity, or sex as inputs. Yet on the held-out test set:

| Race | Mean model-predicted approval probability | Actual approval rate | n |
|---|---|---|---|
| White | 79.08% | 80.48% | 27,931 |
| Asian | 79.15% | 78.21% | 3,842 |
| Black or African American | 72.21% | 69.42% | 6,852 |
| American Indian/Alaska Native | 73.54% | 72.61% | 241 |

A ~7-point gap in predicted approval probability persists between Black and
White applicants **even with race fully excluded from the model**. This isn't
a bug — it's a well-documented phenomenon in fair-lending literature: features
like income, DTI, and requested LTV are themselves correlated with race due to
structural economic disparities, so "race-blind" modeling does not guarantee
disparate-impact-free outcomes. A production deployment of this model would
need an explicit fairness gate (disparity monitoring + a defined action
threshold) rather than treating race exclusion as sufficient compliance.

### 5. Geographic check
Approval rate by neighborhood minority-population quartile is weaker and
non-monotonic (Q1: 77.9%, Q2: 79.5%, Q3: 78.5%, Q4 highest-minority: 75.7%) —
real but inconclusive as a standalone redlining signal in this state-level cut.
Reported honestly rather than forced into a cleaner story than the data supports.

### 6. Explainability (SHAP)
Top drivers of model predictions, by mean absolute SHAP value: debt-to-income
ratio (0.78), loan purpose (0.71), loan-to-value ratio (0.46), applicant credit
score type (0.22), loan amount (0.21). All legitimate underwriting factors — no
demographic field appears in the top drivers, consistent with Model B's design.
See `outputs/shap_summary.png`.

### 7. Drift monitoring
Population Stability Index (PSI) computed by splitting the test set into two
halves as a stand-in for sequential monitoring periods. All features and the
model's output score came back stable (PSI < 0.001 across the board) — **expected
and not a meaningful production result**, since this is a random split of one
year of data, not genuinely sequential periods. This demonstrates the monitoring
methodology (industry-standard PSI thresholds: <0.10 stable, 0.10-0.25 watch,
>0.25 retrain flag) rather than a real drift finding. In production this would
run on true month-over-month or quarter-over-quarter data.

## Honest Limitations

- **Single state, single year.** Virginia 2024 only. Findings may not generalize
  to other states or economic conditions.
- **HMDA covers mortgages, not the lease-to-own/installment credit that
  companies like Koalafi underwrite.** The methodology (fairness-gated risk
  modeling, adverse impact monitoring, proxy-feature auditing) transfers
  directly; the specific dataset and model do not.
- **No alternative/thin-file credit data.** This dataset can't speak to
  underwriting for applicants with limited traditional credit history —
  that would require a different data source entirely.
- **Drift monitoring is a methodology demo, not a real finding** (see above).
- **AIR below 0.80 was not observed for any group in this cut**, so this
  project cannot claim to have found a legally actionable disparity — only a
  measurable gap worth monitoring.

