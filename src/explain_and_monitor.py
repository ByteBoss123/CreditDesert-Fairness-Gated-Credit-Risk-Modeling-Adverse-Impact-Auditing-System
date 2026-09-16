"""
CreditDesert - Explainability & Drift Monitoring
SHAP-based feature attribution (matches JD: "explainability, root-cause analysis")
and a Population Stability Index (PSI) drift check simulating quarterly monitoring
(matches JD: "performance monitoring, drift detection").
"""
import pickle
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

with open('/home/claude/creditdesert/outputs/model_results.pkl', 'rb') as f:
    results = pickle.load(f)

res = results['B']  # Model B: no tract_minority_population_percent, the fairer variant
model = res['model']
X_test = res['X_test']

# --- SHAP explainability ---
explainer = shap.TreeExplainer(model)
X_sample = X_test.sample(min(5000, len(X_test)), random_state=42)
shap_values = explainer.shap_values(X_sample)

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X_sample.columns)
mean_abs_shap = mean_abs_shap.sort_values(ascending=False)
print("=== Top 10 features by mean |SHAP value| (global importance) ===")
print(mean_abs_shap.head(10).round(4))

mean_abs_shap.to_csv('/home/claude/creditdesert/outputs/shap_feature_importance.csv')

# Save a SHAP summary plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
shap.summary_plot(shap_values, X_sample, show=False, max_display=12)
plt.tight_layout()
plt.savefig('/home/claude/creditdesert/outputs/shap_summary.png', dpi=150)
plt.close()
print("Saved shap_summary.png")

# --- Drift monitoring: Population Stability Index (PSI) ---
# Simulates the "quarterly monitoring" scenario: split test set in half to
# stand in for two time periods, and check whether feature distributions
# and score distributions have drifted using the industry-standard PSI metric.
def psi(expected, actual, bins=10):
    expected = pd.to_numeric(expected, errors='coerce').dropna()
    actual = pd.to_numeric(actual, errors='coerce').dropna()
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints = np.unique(breakpoints)
    e_counts = np.histogram(expected, breakpoints)[0] / len(expected)
    a_counts = np.histogram(actual, breakpoints)[0] / len(actual)
    e_counts = np.where(e_counts == 0, 1e-6, e_counts)
    a_counts = np.where(a_counts == 0, 1e-6, a_counts)
    return np.sum((a_counts - e_counts) * np.log(a_counts / e_counts))

n = len(X_test)
period_a = X_test.iloc[:n // 2]
period_b = X_test.iloc[n // 2:]

print("\n=== Drift check (PSI) between two simulated monitoring periods ===")
print("PSI interpretation: <0.10 stable | 0.10-0.25 moderate shift | >0.25 significant drift, retrain flag")
drift_results = {}
for col in ['loan_amount', 'income', 'loan_to_income', 'loan_to_value_ratio', 'debt_to_income_numeric']:
    score = psi(period_a[col], period_b[col])
    drift_results[col] = round(score, 4)
    flag = 'RETRAIN FLAG' if score > 0.25 else ('watch' if score > 0.10 else 'stable')
    print(f"{col}: PSI={score:.4f} -> {flag}")

# Score-level drift (model output distribution)
preds_a = model.predict(xgb.DMatrix(period_a, enable_categorical=True))
preds_b = model.predict(xgb.DMatrix(period_b, enable_categorical=True))
score_psi = psi(pd.Series(preds_a), pd.Series(preds_b))
print(f"model_predicted_score: PSI={score_psi:.4f}")
drift_results['model_predicted_score'] = round(score_psi, 4)

pd.Series(drift_results, name='PSI').to_csv('/home/claude/creditdesert/outputs/drift_monitoring_psi.csv')
print("\nSaved drift_monitoring_psi.csv")
