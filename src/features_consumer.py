"""
CreditDesert - Component 2: Consumer Installment Credit Risk Model
Domain: non-mortgage consumer credit (income/assets/debt/employment based
underwriting) - a much closer proxy for Koalafi's lease-to-own, non-prime
population than the HMDA mortgage data in Component 1.

NOTE: this dataset has no race/ethnicity/sex fields, so it CANNOT support a
fairness audit. It exists purely to close the "mortgage vs. installment credit"
domain gap for the risk-modeling half of the story. The fairness audit
(Component 1, HMDA) remains the fairness centerpiece of this project.
"""
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['seniority'] = df['Seniority']            # months at current job
    out['time'] = df['Time']                       # requested loan duration (months)
    out['age'] = df['Age']
    out['expenses'] = df['Expenses']
    out['income'] = df['Income']
    out['assets'] = df['Assets']
    out['debt'] = df['Debt']
    out['amount'] = df['Amount']                    # requested loan amount
    out['price'] = df['Price']                      # price of good being financed
    out['amount_to_income'] = out['amount'] / out['income'].replace(0, np.nan)
    out['debt_to_income'] = out['debt'] / out['income'].replace(0, np.nan)
    out['amount_to_price'] = out['amount'] / out['price'].replace(0, np.nan)

    out['home'] = df['Home'].astype('category')
    out['marital'] = df['Marital'].astype('category')
    out['records'] = df['Records'].astype('category')  # prior default on record - legitimate bureau-style signal
    out['job'] = df['Job'].astype('category')

    return out

def build_target(df: pd.DataFrame) -> pd.Series:
    return (df['Status'] == 'bad').astype(int)  # 1 = bad/default risk, matches "risk model" framing
