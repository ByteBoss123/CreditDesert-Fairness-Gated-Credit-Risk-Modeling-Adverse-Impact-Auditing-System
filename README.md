# CreditDesert: credit modeling and monitoring demonstration

Two research models are included: HMDA mortgage **approval/denial prediction**
and consumer credit good/bad outcome prediction. HMDA approval probability is
not probability of repayment, default or creditworthiness.

## Reproduce training and candidate registration

Use Python 3.11 or 3.12 in a virtual environment:

```bash
python -m pip install -r requirements-training.txt
python src/train_consumer.py --register
python src/train.py --register
python src/fairness_audit.py
python src/explain_and_monitor.py
python -m pytest tests -q
```

Data defaults to the repository root. Override `CREDITDESERT_DATA_DIR` if needed.
Generated files go to `outputs/`, configurable via `CREDITDESERT_OUTPUT_DIR`.
`MLFLOW_TRACKING_URI` can point to an authorized tracking server; by default a
local SQLite store is used. The `--register` flag registers candidate versions
and tags them `candidate_not_approved`. No production alias is assigned.

Each model uses disjoint 60/20/20 stratified train/validation/test partitions.
Category dictionaries are learned from training rows only. Early stopping uses
validation data; the saved model is sliced to the selected iteration before
final test evaluation. MLflow records parameters, metrics, source checksum,
split indices, model artifact and the evaluation report. PR-AUC, ROC-AUC, Brier
score and fixed-threshold precision/recall/F1 are reported.

These are random holdouts, not temporal or lender-held-out validation. The fixed
0.5 threshold is illustrative and has not been optimized for business costs.
Repeated development against this dataset limits claims of external validity.

## Serving and monitoring

`app.py` and the checked-in HMDA model support a local FastAPI demonstration.
See `SERVING_README.md`. New training writes new models to `outputs/`; it does
not silently replace the historical served model or its category schemas.
The API records prediction counts, score distributions and range warnings.
The batch monitor computes SHAP and PSI on two simulated periods from a holdout;
it is not live production monitoring. PSI now covers values outside reference
bounds and explicitly rejects samples with no finite observations.

## Governance scope

See `MODEL_CARD.md` for intended use, review requirements and limitations.
Group disparity reports are exploratory. Excluding protected attributes or
crossing a numerical threshold does not establish fairness or compliance.
Mean predicted probabilities by group are descriptive comparisons, not causal
or fully adjusted estimates. No lending decisions should use this demonstration.

## Evidence and historical outputs

Root-level model files, charts and reports are historical artifacts. The older
training used its test partition for early stopping, so historical AUC values
are not untouched-test estimates. New results are separately recorded under
`outputs/` and summarized in `VALIDATION.md` when executed. Original methodology
files are retained as historical context and may describe the earlier protocol.
Databricks translation notes are a proposal, not deployment evidence.
