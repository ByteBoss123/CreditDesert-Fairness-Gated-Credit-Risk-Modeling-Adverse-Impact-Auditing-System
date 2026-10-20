"""Portable paths; generated artifacts never overwrite historical evidence."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('CREDITDESERT_DATA_DIR', ROOT))
OUTPUTS = Path(os.environ.get('CREDITDESERT_OUTPUT_DIR', ROOT / 'outputs'))
OUTPUTS.mkdir(parents=True, exist_ok=True)
TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///' + str(OUTPUTS / 'mlflow.db'))
