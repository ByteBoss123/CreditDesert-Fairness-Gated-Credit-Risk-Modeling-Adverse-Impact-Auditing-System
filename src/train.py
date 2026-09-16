"""
CreditDesert - Risk Model Training
Trains an XGBoost approval/denial model with MLflow experiment tracking.
Trains TWO variants to test whether tract-level minority composition acts as a redlining proxy:
  - Model A: includes tract_minority_population_percent
  - Model B: excludes it (economic tract features only)
"""
import sys
sys.path.insert(0, '/home/claude/creditdesert/src')
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, brier_score_loss
from features import build_features, build_demographics

mlflow.set_tracking_uri('sqlite:////home/claude/creditdesert/mlruns/mlflow.db')
mlflow.set_experiment('CreditDesert_HMDA_VA_2024')

df = pd.read_parquet('/home/claude/creditdesert/data/hmda_va_2024_modeling.parquet')
y = df['approved']
X_full = build_features(df)
demo = build_demographics(df)

cat_cols = X_full.select_dtypes(include='category').columns.tolist()
for c in cat_cols:
    X_full[c] = X_full[c].astype(str).astype('category')

def train_variant(feature_df, variant_name, drop_cols=None):
    X = feature_df.copy()
    if drop_cols:
        X = X.drop(columns=drop_cols)

    X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
    }

    with mlflow.start_run(run_name=variant_name):
        mlflow.log_params(params)
        mlflow.log_param('n_features', X.shape[1])
        mlflow.log_param('n_train', len(X_train))
        mlflow.log_param('n_test', len(X_test))
        mlflow.log_param('variant', variant_name)

        evals_result = {}
        model = xgb.train(
            params, dtrain, num_boost_round=300,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=20, evals_result=evals_result, verbose_eval=False
        )

        preds = model.predict(dtest)
        auc = roc_auc_score(y_test, preds)
        brier = brier_score_loss(y_test, preds)
        pred_labels = (preds >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred_labels, average='binary')

        mlflow.log_metric('test_auc', auc)
        mlflow.log_metric('test_brier', brier)
        mlflow.log_metric('test_precision', precision)
        mlflow.log_metric('test_recall', recall)
        mlflow.log_metric('test_f1', f1)
        mlflow.log_metric('best_iteration', model.best_iteration)
        mlflow.xgboost.log_model(model, artifact_path='model')

        run_id = mlflow.active_run().info.run_id

    print(f"[{variant_name}] AUC={auc:.4f} Brier={brier:.4f} P={precision:.4f} R={recall:.4f} F1={f1:.4f} n_feat={X.shape[1]} run_id={run_id}")

    return {
        'model': model, 'X_test': X_test, 'y_test': y_test, 'preds': preds,
        'demo_test': demo_test, 'auc': auc, 'brier': brier,
        'precision': precision, 'recall': recall, 'f1': f1, 'run_id': run_id,
        'variant': variant_name
    }

if __name__ == '__main__':
    results_a = train_variant(X_full, 'ModelA_with_tract_minority_pct')
    results_b = train_variant(X_full, 'ModelB_without_tract_minority_pct',
                               drop_cols=['tract_minority_population_percent'])

    import pickle
    with open('/home/claude/creditdesert/outputs/model_results.pkl', 'wb') as f:
        pickle.dump({'A': results_a, 'B': results_b}, f)
    print("\nSaved model_results.pkl")
