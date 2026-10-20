# Model card and review checklist

**Owner:** repository maintainer. **Status:** research candidate, not approved for lending.

**Targets:** HMDA historical lender approval/denial; consumer dataset good/bad
outcomes. These targets represent different tasks and must be reported separately.
**Data:** bundled HMDA Virginia 2024 parquet and consumer credit CSV. Training
records SHA-256 source fingerprints. Verify upstream provenance and permitted
use before any deployment beyond the portfolio setting.

**Validation:** stratified 60/20/20 split, training-only categorical vocabularies,
validation early stopping, untouched final partition during fitting, and saved
split indices. This does not establish time stability, causality or generalization
to another lender/population. Check repeated applicants and group leakage before
making stronger claims. Label semantics and data availability at prediction time
require review.

**Governance artifacts:** MLflow run/model IDs, source hash, evaluation report,
candidate-only registry version, and descriptive group/SHAP/PSI analysis.
These artifacts are not regulatory certification or an organizational approval.

**Promotion:** no automated production promotion. A reviewer must verify source
rights, feature timing, external validation, subgroup uncertainty, calibration,
business thresholds, security, and serving/training schema parity. Record reviewer,
date, model version and decision before promotion in a real system.

**Rollback:** the demo API keeps the original checked-in artifact; training never
replaces it. Any deployment must package model and schemas together and retain
the prior approved bundle. No rollback has been exercised in a production system.

**Monitoring:** local logs and simulated PSI demonstrate mechanics. Actual
performance monitoring requires delayed labels and an agreed evaluation window.
No customer usage, default reduction or financial savings are asserted.
