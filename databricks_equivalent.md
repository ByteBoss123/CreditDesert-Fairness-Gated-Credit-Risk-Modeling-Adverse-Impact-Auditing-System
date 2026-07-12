# Databricks / Production Pipeline Equivalent

This project was built locally with pandas + XGBoost for fast iteration and easy
verification of every metric. The JD calls out Databricks and Snowflake exposure,
so this note is an honest translation layer: what changes, and what doesn't, if this
pipeline were deployed on Databricks.

## What stays identical
- Feature engineering logic (`src/features.py`) — pure pandas transformations
  translate to PySpark DataFrame operations 1:1 (`.withColumn`, `.groupBy`, etc.)
- Model choice (XGBoost) — `xgboost.spark.SparkXGBClassifier` is a drop-in
  distributed equivalent of the `xgb.train()` calls used here
- MLflow tracking — already used in this project locally; on Databricks this is
  the same API, just pointed at the Databricks-hosted MLflow tracking server
  instead of a local SQLite backend, with zero code changes to logging calls
- SHAP explainability — works identically inside a Databricks notebook

## What would change in a real deployment
```python
# Local (this project):
df = pd.read_parquet('data/hmda_va_2024_modeling.parquet')

# Databricks:
df = spark.read.format("delta").load("/mnt/lakehouse/hmda/va_2024")
df = df.withColumn("loan_to_income",
        F.col("loan_amount") / (F.col("income") * 1000))
```

- **Ingestion**: raw HMDA CSVs would land in a Bronze Delta table via Auto Loader
  (`cloudFiles` format), with schema inference explicitly enabled
  (`cloudFiles.inferColumnTypes = true`) to avoid the all-StringType trap.
- **Feature store**: Silver/Gold Delta tables replace the local parquet file,
  with Delta's ACID guarantees and time travel replacing manual versioning.
- **Training at scale**: `SparkXGBClassifier` for distributed training across
  multiple states/years instead of single-machine XGBoost.
- **Drift monitoring in production**: scheduled as a Databricks Job (weekly/monthly),
  writing PSI scores to a monitoring Delta table with alerting instead of the
  one-off script used here.
- **Model registry & serving**: `mlflow.register_model()` + Databricks Model
  Serving endpoint for real-time decisioning, matching the JD's
  "real-time decisioning platform" language.

## Why it wasn't built on Databricks directly for this project
Honest reason: this was built to validate the modeling and fairness methodology
end-to-end, quickly, on one state's data (247K rows) — a scale where a local
pandas/XGBoost pipeline is faster to iterate on and easier to audit line-by-line
than a distributed cluster. The architecture above is what scaling this to
national HMDA data (~15M+ records/year) on Databricks would actually look like.
