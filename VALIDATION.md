# Repair validation — 2026-10-19

Executed locally against the included real datasets:

- Five regression tests passed: split completeness/disjointness, PSI overflow,
  constant-reference shift, and invalid finite-sample handling.
- Consumer model trained on 2,672 rows, validated on 891, tested on 891:
  ROC-AUC 0.864399, PR-AUC 0.715700, Brier 0.131054 (constant baseline 0.202348).
- HMDA A: 148,215 / 49,405 / 49,405 rows; test ROC-AUC 0.882746.
- HMDA B: same partitions; test ROC-AUC 0.882346, PR-AUC 0.951995.
  The positive label is lender approval, not repayment/default.
- All three models registered as version 1 in a local SQLite MLflow registry,
  tagged `candidate_not_approved`; no production promotion.
- Fairness report, SHAP computation and simulated-period PSI script executed.
- Existing API returned HTTP 200 for health and a synthetic valid prediction.
  This verifies local serving only, not an internet deployment or customer use.

Fresh JSON reports, environment versions and registry metadata are in
`evidence/repair_2026-10-19/`. Historical root-level reports remain unchanged.
The new model files and local MLflow database are generated under `outputs/`
and are not committed; rerun the documented commands to recreate them.

The test partition is excluded from early stopping. It is still from the same
historical dataset and does not replace external or temporal validation.
