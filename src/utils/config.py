"""Project-wide config loaded from .env."""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parents[2]
RAW_DIR         = ROOT_DIR / os.getenv("RAW_DATA_DIR",       "data/raw")
PROCESSED_DIR   = ROOT_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
FEATURES_DIR    = ROOT_DIR / os.getenv("FEATURES_DATA_DIR",  "data/features")
MODELS_DIR      = ROOT_DIR / os.getenv("MODELS_DIR",         "models")

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_URI        = os.getenv("MLFLOW_TRACKING_URI",   "mlruns")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "RetailPulse")

# ── Model config ───────────────────────────────────────────────────────────
CHURN_THRESHOLD_DAYS  = int(os.getenv("CHURN_THRESHOLD_DAYS",  90))
FORECAST_HORIZON_DAYS = int(os.getenv("FORECAST_HORIZON_DAYS", 90))
LEAD_TIME_DAYS        = int(os.getenv("LEAD_TIME_DAYS",         7))
SERVICE_LEVEL         = float(os.getenv("SERVICE_LEVEL",       0.95))

# Ensure output dirs exist
for _d in (PROCESSED_DIR, FEATURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
