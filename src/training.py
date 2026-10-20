"""Three-way evaluation and opt-in MLflow candidate registration."""
import hashlib
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_fscore_support
from paths import OUTPUTS, TRACKING_URI


def split_indices(y):
    train, remainder = train_test_split(np.arange(len(y)), test_size=.4, stratify=y, random_state=42)
    valid, test = train_test_split(remainder, test_size=.5, stratify=y.iloc[remainder], random_state=43)
    return train, valid, test


def train_model(X, y, name, source, register=False):
    train, valid, test = split_indices(y)
    # Fit category dictionaries on training data only; unseen values become missing.
    X = X.copy()
    for col in X.select_dtypes(include='category').columns:
        categories = X.iloc[train][col].astype(str).unique().tolist()
        X[col] = pd.Categorical(X[col].astype(str), categories=categories)
    mats = [xgb.DMatrix(X.iloc[idx], label=y.iloc[idx], enable_categorical=True) for idx in (train, valid, test)]
    params = dict(objective='binary:logistic', eval_metric='auc', max_depth=4,
                  eta=.05, subsample=.8, colsample_bytree=.8, seed=42, nthread=2)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment('CreditDesert_reproducible')
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        mlflow.log_params(dict(n_train=len(train), n_validation=len(valid), n_test=len(test), n_features=X.shape[1]))
        mlflow.set_tags(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             intended_use='portfolio_research_only', evaluation='60_20_20_stratified_holdout'))
        model = xgb.train(params, mats[0], num_boost_round=300,
                          evals=[(mats[1], 'validation')], early_stopping_rounds=20, verbose_eval=False)
        # Slice to the validation-selected iteration before testing and serialization.
        model = model[:model.best_iteration + 1]
        preds = model.predict(mats[2])
        precision, recall, f1, _ = precision_recall_fscore_support(y.iloc[test], preds >= .5, average='binary', zero_division=0)
        metrics = dict(test_auc=float(roc_auc_score(y.iloc[test], preds)),
                       test_pr_auc=float(average_precision_score(y.iloc[test], preds)),
                       test_brier=float(brier_score_loss(y.iloc[test], preds)),
                       test_precision=float(precision), test_recall=float(recall), test_f1=float(f1),
                       baseline_brier=float(brier_score_loss(y.iloc[test], np.full(len(test), y.iloc[train].mean()))))
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path='model')
        split_manifest = {label: idx.tolist() for label, idx in zip(('train','validation','test'), (train,valid,test))}
        mlflow.log_dict(split_manifest, 'split_indices.json')
        registration = None
        if register:
            version = mlflow.register_model(f'runs:/{run.info.run_id}/model', name)
            client = mlflow.MlflowClient()
            client.set_model_version_tag(name, version.version, 'review_status', 'candidate_not_approved')
            registration = dict(name=name, version=version.version)
        report = dict(model=name, metrics=metrics, run_id=run.info.run_id, registration=registration,
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      split_counts={k:len(v) for k,v in split_manifest.items()},
                      limitations=['Random holdout, not temporal validation', 'Research demonstration; no lending approval or production promotion'])
        (OUTPUTS / f'{name}_evaluation.json').write_text(json.dumps(report, indent=2))
        mlflow.log_dict(report, 'evaluation.json')
        model.save_model(OUTPUTS / f'{name}.json')
    print(json.dumps(report, indent=2))
    return dict(model=model, X_test=X.iloc[test], y_test=y.iloc[test], preds=preds,
                auc=metrics['test_auc'], brier=metrics['test_brier'], precision=precision,
                recall=recall, f1=f1, run_id=run.info.run_id, test_indices=test, variant=name)
