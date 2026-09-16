"""
CreditDesert - Fairness Audit
Standard fair-lending metrics: raw denial-rate disparity by group, and the
Adverse Impact Ratio (AIR) / "80% rule" used by CFPB and EEOC in practice.
Also checks whether MODEL predictions (not just raw outcomes) show disparity
after controlling for legitimate underwriting factors.
"""
import pickle
import pandas as pd
import numpy as np

with open('/home/claude/creditdesert/outputs/model_results.pkl', 'rb') as f:
    results = pickle.load(f)

df = pd.read_parquet('/home/claude/creditdesert/data/hmda_va_2024_modeling.parquet')

def air_table(group_col, label, actual_col='approved'):
    sub = df[df[group_col].isin(['White', 'Black or African American', 'Asian',
                                   'Hispanic or Latino', 'Not Hispanic or Latino',
                                   'Male', 'Female',
                                   'American Indian or Alaska Native'])]
    rates = sub.groupby(group_col)[actual_col].agg(['mean', 'count']).rename(
        columns={'mean': 'approval_rate', 'count': 'n'})
    rates = rates.sort_values('approval_rate', ascending=False)
    reference_rate = rates['approval_rate'].max()
    rates['adverse_impact_ratio'] = (rates['approval_rate'] / reference_rate).round(4)
    rates['approval_rate'] = rates['approval_rate'].round(4)
    rates['flag_below_80pct_rule'] = rates['adverse_impact_ratio'] < 0.80
    print(f"\n=== {label}: raw approval rate & Adverse Impact Ratio (80% rule) ===")
    print(rates)
    return rates

race_table = air_table('derived_race', 'Race')
eth_table = air_table('derived_ethnicity', 'Ethnicity')
sex_table = air_table('derived_sex', 'Sex')

# --- Model-adjusted disparity: does the trained model's predicted approval
# probability still differ by race AFTER controlling for legitimate financial factors? ---
print("\n=== Model-predicted approval rate by race (test set, Model B - no tract minority %) ===")
res_b = results['B']
demo_test = res_b['demo_test'].copy()
demo_test['pred_prob'] = res_b['preds']
demo_test['actual'] = res_b['y_test'].values

model_race = demo_test[demo_test['derived_race'].isin(
    ['White', 'Black or African American', 'Asian', 'American Indian or Alaska Native'])
].groupby('derived_race').agg(
    mean_predicted_approval_prob=('pred_prob', 'mean'),
    actual_approval_rate=('actual', 'mean'),
    n=('actual', 'count')
).round(4)
print(model_race)

# --- Geographic "credit desert" check: denial rate by tract minority composition quartile ---
print("\n=== Geographic disparity: approval rate by tract minority-population quartile ===")
df['tract_minority_quartile'] = pd.qcut(
    pd.to_numeric(df['tract_minority_population_percent'], errors='coerce'),
    4, labels=['Q1 (least minority)', 'Q2', 'Q3', 'Q4 (most minority)']
)
geo_table = df.groupby('tract_minority_quartile', observed=True)['approved'].agg(['mean', 'count']).rename(
    columns={'mean': 'approval_rate', 'count': 'n'})
geo_table['approval_rate'] = geo_table['approval_rate'].round(4)
print(geo_table)

# Save all outputs
with pd.ExcelWriter('/home/claude/creditdesert/outputs/fairness_audit_results.xlsx') as writer:
    race_table.to_excel(writer, sheet_name='Race_AIR')
    eth_table.to_excel(writer, sheet_name='Ethnicity_AIR')
    sex_table.to_excel(writer, sheet_name='Sex_AIR')
    model_race.to_excel(writer, sheet_name='Model_Adjusted_Race')
    geo_table.to_excel(writer, sheet_name='Geographic_Quartile')

print("\nSaved fairness_audit_results.xlsx")
