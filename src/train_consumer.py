"""
CreditDesert - Component 2: Consumer Installment Credit Risk Model Training
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
from features_consumer import build_features, build_target

mlflow.set_tracking_uri('sqlite:////home/claude/creditdesert/mlruns/mlflow.db')
mlflow.set_experiment('CreditDesert_ConsumerCredit_Component2')

df = pd.read_csv('/home/claude/creditdesert/data/consumer_credit_data.csv')
X = build_features(df)
y = build_target(df)

cat_cols = X.select_dtypes(include='category').columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype(str).astype('category')

print(f"Rows: {len(X)}, Bad rate: {y.mean():.4f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
}

with mlflow.start_run(run_name='ConsumerCredit_XGB'):
    mlflow.log_params(params)
    mlflow.log_param('n_features', X.shape[1])
    mlflow.log_param('n_train', len(X_train))
    mlflow.log_param('n_test', len(X_test))

    model = xgb.train(
        params, dtrain, num_boost_round=300,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=20, verbose_eval=False
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

print(f"AUC={auc:.4f} Brier={brier:.4f} Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} run_id={run_id}")

import pickle
with open('/home/claude/creditdesert/outputs/consumer_model_results.pkl', 'wb') as f:
    pickle.dump({'model': model, 'X_test': X_test, 'y_test': y_test, 'preds': preds,
                 'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1}, f)
print("Saved consumer_model_results.pkl")
