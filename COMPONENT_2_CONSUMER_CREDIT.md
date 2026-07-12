# Component 2: Consumer Installment Credit Risk Model

HMDA is mortgage data. Koalafi underwrites lease-to-own/installment credit for
non-prime consumers — a genuinely different population. This component closes
that domain gap using a second real dataset: consumer credit-scoring data
originally compiled by Dr. Lluís Belanche Muñoz (source:
[github.com/gastonstat/CreditScoring](https://github.com/gastonstat/CreditScoring)),
4,454 real credit applications with income, assets, debt, employment tenure,
prior-default records, and loan amount/price.

**This dataset has no race/ethnicity/sex fields — it cannot support a fairness
audit.** It exists solely to demonstrate the risk-modeling half of the story on
a non-mortgage population. Component 1 (HMDA) remains the fairness-audit
centerpiece of this project.

### Results
- **Bad rate**: 28.15% (1,254 bad / 4,454 total) — a realistic base rate for
  non-prime consumer credit
- **XGBoost model**: Test AUC = 0.8529, Precision = 0.6907, Recall = 0.5339, F1 = 0.6022
- **Leakage check performed**: verified no field is near-100%-missing
  conditional on outcome (the pattern that caused the HMDA leakage bug).
  Income has a higher missing rate for bad applicants (17.3% vs 5.1% for
  good) — plausible legitimate missingness (harder to verify income for
  higher-risk applicants), not leakage.

### Top model drivers (SHAP)
`amount_to_income` ratio, `records` (prior default flag), `seniority`
(employment tenure), `amount_to_price` ratio, `assets`, `job` type — all
legitimate underwriting signals. See `outputs/consumer_shap_summary.png`.

### Honest limitations of this component
- Small dataset (4,454 rows) — less statistically robust than the 247K-row
  HMDA model
- Population/country of origin not stated in the source documentation —
  flagged as unspecified rather than assumed
- Lower recall (0.53) than the HMDA model reflects genuine difficulty:
  smaller sample size and a harder classification task (predicting future
  default vs. predicting an already-made approval decision)
- No demographic fields, so this component cannot be fairness-audited on its
  own — it must be read alongside Component 1, not as a replacement for it
