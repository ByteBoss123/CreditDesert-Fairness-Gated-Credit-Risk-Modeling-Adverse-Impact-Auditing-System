"""
CreditDesert - Model Serving API
Serves the real trained HMDA Model B (fairness-cleared: no tract_minority_population_percent)
for real-time approval-probability scoring.

IMPORTANT DESIGN CHOICE: this API does NOT accept race, ethnicity, or sex as
inputs - consistent with fair lending practice (protected classes should not be
decisioning inputs). Fairness auditing happens out-of-band via periodic batch
analysis (see fairness_audit.py), comparing outcomes across demographic groups
AFTER the fact using HMDA-reported data, never as a live input to this endpoint.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("creditdesert-api")

app = FastAPI(
    title="CreditDesert Risk Scoring API",
    description="Real-time credit approval probability scoring (HMDA-trained model, fairness-cleared variant)",
    version="1.0.0",
)

# --- Load real trained model and schemas at startup ---
model = xgb.Booster()
model.load_model(str(BASE_DIR / "hmda_model_b.json"))

with open(BASE_DIR / "category_schema.json") as f:
    CATEGORY_SCHEMA = json.load(f)
with open(BASE_DIR / "numeric_ranges.json") as f:
    NUMERIC_RANGES = json.load(f)

FEATURE_ORDER = [
    'loan_amount', 'income', 'loan_to_income', 'loan_to_value_ratio', 'loan_term',
    'loan_type', 'loan_purpose', 'lien_status', 'occupancy_type',
    'open_end_line_of_credit', 'business_or_commercial_purpose', 'construction_method',
    'applicant_credit_score_type', 'submission_of_application',
    'tract_to_msa_income_percentage', 'tract_owner_occupied_units',
    'tract_median_age_of_housing_units', 'ffiec_msa_md_median_family_income',
    'debt_to_income_numeric'
]
CAT_COLS = ['loan_type', 'loan_purpose', 'lien_status', 'occupancy_type',
            'open_end_line_of_credit', 'business_or_commercial_purpose',
            'construction_method', 'applicant_credit_score_type', 'submission_of_application']

PREDICTION_LOG_PATH = BASE_DIR / "prediction_log.jsonl"


class LoanApplication(BaseModel):
    loan_amount: float = Field(..., gt=0, description="Requested loan amount (USD)")
    income: float = Field(..., gt=0, description="Applicant annual income (USD thousands, HMDA convention)")
    loan_to_value_ratio: float = Field(..., ge=0, le=200, description="Loan-to-value ratio (%)")
    loan_term: float = Field(..., gt=0, le=600, description="Loan term (months)")
    loan_type: str = Field(..., description="HMDA loan_type code: 1=Conventional, 2=FHA insured, 3=VA guaranteed, 4=USDA RHS/FSA guaranteed")
    loan_purpose: str = Field(..., description="HMDA loan_purpose code (see CFPB field reference for definitions)")
    lien_status: str = Field(..., description="1=Secured by first lien, 2=Secured by subordinate lien")
    occupancy_type: str = Field(..., description="1=Principal residence, 2=Second residence, 3=Investment property")
    open_end_line_of_credit: str = Field(..., description="HMDA code (see CFPB field reference)")
    business_or_commercial_purpose: str = Field(..., description="HMDA code (see CFPB field reference)")
    construction_method: str = Field(..., description="1=Site-built, 2=Manufactured home")
    applicant_credit_score_type: str = Field(..., description="HMDA code (see CFPB field reference)")
    submission_of_application: str = Field(..., description="HMDA code (see CFPB field reference)")
    tract_to_msa_income_percentage: float = Field(..., ge=0, le=500, description="Census tract income as % of MSA median")
    tract_owner_occupied_units: float = Field(..., ge=0, description="Owner-occupied housing units in tract")
    tract_median_age_of_housing_units: float = Field(..., ge=0, description="Median age of housing units in tract (years)")
    ffiec_msa_md_median_family_income: float = Field(..., gt=0, description="MSA median family income (USD)")
    debt_to_income_numeric: float = Field(..., ge=0, le=100, description="Debt-to-income ratio (%)")

    @field_validator('loan_type', 'loan_purpose', 'lien_status', 'occupancy_type',
                      'open_end_line_of_credit', 'business_or_commercial_purpose',
                      'construction_method', 'applicant_credit_score_type',
                      'submission_of_application')
    @classmethod
    def validate_category(cls, v, info):
        field_name = info.field_name
        valid = CATEGORY_SCHEMA.get(field_name, [])
        if v not in valid:
            raise ValueError(f"'{v}' is not a valid code for {field_name}. Valid codes seen in training data: {valid}")
        return v


class PredictionResponse(BaseModel):
    approval_probability: float
    predicted_decision: str
    model_version: str = "hmda_model_b_fairness_cleared"
    warnings: list[str] = []


def build_feature_row(app_data: LoanApplication) -> pd.DataFrame:
    row = {
        'loan_amount': app_data.loan_amount,
        'income': app_data.income,
        'loan_to_income': app_data.loan_amount / (app_data.income * 1000) if app_data.income > 0 else np.nan,
        'loan_to_value_ratio': app_data.loan_to_value_ratio,
        'loan_term': app_data.loan_term,
        'loan_type': app_data.loan_type,
        'loan_purpose': app_data.loan_purpose,
        'lien_status': app_data.lien_status,
        'occupancy_type': app_data.occupancy_type,
        'open_end_line_of_credit': app_data.open_end_line_of_credit,
        'business_or_commercial_purpose': app_data.business_or_commercial_purpose,
        'construction_method': app_data.construction_method,
        'applicant_credit_score_type': app_data.applicant_credit_score_type,
        'submission_of_application': app_data.submission_of_application,
        'tract_to_msa_income_percentage': app_data.tract_to_msa_income_percentage,
        'tract_owner_occupied_units': app_data.tract_owner_occupied_units,
        'tract_median_age_of_housing_units': app_data.tract_median_age_of_housing_units,
        'ffiec_msa_md_median_family_income': app_data.ffiec_msa_md_median_family_income,
        'debt_to_income_numeric': app_data.debt_to_income_numeric,
    }
    df = pd.DataFrame([row])[FEATURE_ORDER]
    for c in CAT_COLS:
        df[c] = df[c].astype('category')
    return df


def check_out_of_range_warnings(app_data: LoanApplication) -> list[str]:
    """Real monitoring signal: flag inputs far outside the training distribution.
    Uses 1st/99th percentile bounds, NOT true min/max - the raw HMDA data contains
    severe outliers (e.g. loan_amount up to $375M, LTV up to 2,000,000%), which the
    CFPB itself warns about in its data documentation. True min/max would make this
    check nearly useless (it would almost never trigger); percentile bounds actually
    catch genuinely unusual inputs while tolerating the real-world outliers already
    present in the data."""
    warnings = []
    checks = {
        'loan_amount': app_data.loan_amount,
        'income': app_data.income,
        'loan_to_value_ratio': app_data.loan_to_value_ratio,
        'debt_to_income_numeric': app_data.debt_to_income_numeric,
    }
    for field, value in checks.items():
        rng = NUMERIC_RANGES.get(field)
        if rng and (value < rng['p01'] or value > rng['p99']):
            warnings.append(
                f"{field}={value} is outside the typical training range "
                f"[{rng['p01']:.1f}, {rng['p99']:.1f}] (1st-99th percentile) - "
                f"prediction may be less reliable for this profile"
            )
    return warnings


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/predict", response_model=PredictionResponse)
def predict(app_data: LoanApplication):
    try:
        X = build_feature_row(app_data)
        dmatrix = xgb.DMatrix(X, enable_categorical=True)
        prob = float(model.predict(dmatrix)[0])
        decision = "likely_approved" if prob >= 0.5 else "likely_denied"
        warnings = check_out_of_range_warnings(app_data)

        # Real logging for downstream monitoring (drift, volume, score distribution)
        # Deliberately does NOT log any demographic field, since none is collected here.
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "approval_probability": prob,
            "predicted_decision": decision,
            "loan_amount": app_data.loan_amount,
            "income": app_data.income,
            "warnings": warnings,
        }
        with open(PREDICTION_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return PredictionResponse(
            approval_probability=round(prob, 4),
            predicted_decision=decision,
            warnings=warnings,
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/monitoring/summary")
def monitoring_summary():
    """Real-time view into what's been served, for the kind of dashboard the
    JD calls out ('performance monitoring'). Reads the actual prediction log,
    not synthetic numbers."""
    if not PREDICTION_LOG_PATH.exists():
        return {"n_predictions": 0, "message": "No predictions logged yet"}

    records = [json.loads(line) for line in open(PREDICTION_LOG_PATH)]
    df = pd.DataFrame(records)
    return {
        "n_predictions": len(df),
        "approval_rate": round(df['predicted_decision'].eq('likely_approved').mean(), 4),
        "mean_approval_probability": round(df['approval_probability'].mean(), 4),
        "n_with_warnings": int(df['warnings'].apply(len).gt(0).sum()),
    }
