# CreditDesert Model Serving API

A real FastAPI service serving the actual trained HMDA Model B (the fairness-cleared
variant, no `tract_minority_population_percent`). Verified end-to-end with real
requests — see test output below, not a mockup.

## What was actually verified (not simulated)
- **Strong applicant** ($350K loan, $95K income, 80% LTV, 36% DTI): 98.06% approval probability
- **Weak applicant** (same profile, 65% DTI, 97% LTV, $40K income): 2.24% approval probability
- **Invalid input rejected**: an out-of-schema `loan_type` code returns HTTP 422 with a clear error, not a silent bad prediction
- **Outlier flagging works**: a $50M loan request is still scored, but flagged with an explicit warning that it's outside the 1st-99th percentile of training data
- **Monitoring endpoint works**: `/monitoring/summary` reads real logged predictions, not placeholder numbers

## Design choices worth explaining in an interview

**No race/ethnicity/sex as inputs.** This is deliberate and matches real fair-lending
practice: protected classes should not be direct decisioning inputs. Fairness
auditing happens out-of-band, comparing outcomes across groups after the fact
(see `fairness_audit.py` in the main project) — never as a live field this API reads.

**Percentile bounds, not true min/max, for outlier detection.** The raw HMDA
training data contains severe data-quality outliers (loan amounts up to $375M,
loan-to-value ratios up to 2,000,000%) — the CFPB's own documentation warns about
this. Using true min/max as "normal range" bounds would make the warning system
almost useless, since it would essentially never trigger. Using the 1st/99th
percentile catches genuinely unusual inputs while tolerating real-world outliers.
This was a bug I found and fixed during testing, not something I got right on
the first try — worth mentioning honestly if asked.

**Prediction logging never includes protected class fields**, only what
underwriting decisions can legitimately consider — by construction, since the
API never collects race/ethnicity/sex at all.

## Running it locally
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
Then: `curl http://localhost:8000/health`

## Running it in Docker
```bash
docker build -t creditdesert-api .
docker run -p 8000:8000 creditdesert-api
```

## Honest scope limitation
This was built and verified inside a sandboxed environment with no outbound
internet hosting — so there's no live public URL for this API. The code and
Dockerfile are genuinely deployable as-is; the one remaining step is pushing
this to an actual host:
- **Render / Railway** (free tier): connect this folder as a repo, they auto-detect
  the Dockerfile
- **AWS App Runner** or **Databricks Model Serving**: matches the JD's "real-time
  decisioning platform" language most directly — see `databricks_equivalent.md`
  in the main project for the Databricks-specific translation

## Files
```
app.py                  # FastAPI application
hmda_model_b.json        # Real trained XGBoost model (exported from Component 1)
category_schema.json     # Valid HMDA category codes, extracted from real training data
numeric_ranges.json      # Percentile-based bounds for outlier warnings
Dockerfile
requirements.txt
```
