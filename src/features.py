"""
CreditDesert - Feature Engineering
Builds legitimate underwriting features (no demographic fields) for the risk model,
and keeps demographic fields separate for the fairness audit.
"""
import pandas as pd
import numpy as np

def to_numeric_safe(series):
    return pd.to_numeric(series, errors='coerce')

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # --- Legitimate underwriting signals (what a real credit model would use) ---
    out['loan_amount'] = to_numeric_safe(df['loan_amount'])
    out['income'] = to_numeric_safe(df['income'])
    out['loan_to_income'] = out['loan_amount'] / (out['income'] * 1000).replace(0, np.nan)

    # NOTE: property_value, interest_rate, total_loan_costs, and rate_spread are
    # deliberately excluded. EDA confirmed interest_rate is 100% missing for denied
    # applications (only assigned once a loan actually closes), so including it
    # (or other post-closing fields) causes severe target leakage rather than signal.
    out['loan_to_value_ratio'] = to_numeric_safe(df['loan_to_value_ratio'])
    out['loan_term'] = to_numeric_safe(df['loan_term'])

    # Loan structure flags
    out['loan_type'] = df['loan_type'].astype('category')
    out['loan_purpose'] = df['loan_purpose'].astype('category')
    out['lien_status'] = df['lien_status'].astype('category')
    out['occupancy_type'] = df['occupancy_type'].astype('category')
    out['open_end_line_of_credit'] = df['open-end_line_of_credit'].astype('category')
    out['business_or_commercial_purpose'] = df['business_or_commercial_purpose'].astype('category')
    out['construction_method'] = df['construction_method'].astype('category')
    out['applicant_credit_score_type'] = df['applicant_credit_score_type'].astype('category')
    out['submission_of_application'] = df['submission_of_application'].astype('category')

    # Neighborhood / tract-level economic context (legitimate, not individual demographic)
    out['tract_minority_population_percent'] = to_numeric_safe(df['tract_minority_population_percent'])
    out['tract_to_msa_income_percentage'] = to_numeric_safe(df['tract_to_msa_income_percentage'])
    out['tract_owner_occupied_units'] = to_numeric_safe(df['tract_owner_occupied_units'])
    out['tract_median_age_of_housing_units'] = to_numeric_safe(df['tract_median_age_of_housing_units'])
    out['ffiec_msa_md_median_family_income'] = to_numeric_safe(df['ffiec_msa_md_median_family_income'])

    # Convert DTI bucket strings to an ordered numeric proxy where possible
    dti_map = {
        '<20%': 10, '20%-<30%': 25, '30%-<36%': 33, '36': 36, '37': 37, '38': 38,
        '39': 39, '40': 40, '41': 41, '42': 42, '43': 43, '44': 44, '45': 45,
        '46': 46, '47': 47, '48': 48, '49': 49, '50%-60%': 55, '>60%': 65,
    }
    out['debt_to_income_numeric'] = df['debt_to_income_ratio'].astype(str).map(dti_map)

    return out

def build_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Kept separate from model features - used ONLY for the fairness audit, never as model inputs."""
    return df[['derived_race', 'derived_ethnicity', 'derived_sex',
               'tract_minority_population_percent']].copy()
