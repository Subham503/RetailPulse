"""
Model Performance Tracker.

Logs predictions vs actuals over time.
Computes rolling metrics: AUC, MAPE, silhouette.

Usage:
    python -m monitoring.performance_tracker
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score, silhouette_score
from src.utils.config import PROCESSED_DIR, MODELS_DIR
from src.utils.logging_config import setup_logger

METRICS_PATH = Path(__file__).resolve().parent / "metrics_log.json"


def load_metrics_log() -> list:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return []


def save_metrics_log(log: list) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(log, indent=2))


def compute_churn_metrics() -> dict | None:
    """Recompute churn AUC from saved predictions."""
    p = PROCESSED_DIR / "rfm_with_churn.csv"
    if not p.exists():
        return None

    rfm = pd.read_csv(p)
    if "Churned" not in rfm.columns or "ChurnProb" not in rfm.columns:
        return None

    auc = roc_auc_score(rfm["Churned"], rfm["ChurnProb"])
    return {"model": "churn_xgboost", "metric": "auc_roc", "value": round(auc, 4)}


def compute_forecast_metrics() -> dict | None:
    """Compute MAPE from forecast vs actuals overlap."""
    p = PROCESSED_DIR / "forecast.csv"
    if not p.exists():
        return None

    fc = pd.read_csv(p, parse_dates=["ds"])
    actual   = fc[fc["type"] == "actual"].copy()
    actual["y_actual"] = pd.to_numeric(actual["y_actual"], errors="coerce")
    actual["yhat"]     = pd.to_numeric(actual.get("yhat", np.nan), errors="coerce")

    mask = actual["y_actual"].notna() & actual["yhat"].notna() & (actual["y_actual"] != 0)
    if mask.sum() == 0:
        return None

    mape = float(np.mean(np.abs(
        (actual.loc[mask, "y_actual"] - actual.loc[mask, "yhat"]) / actual.loc[mask, "y_actual"]
    )) * 100)

    return {"model": "sarima_forecast", "metric": "mape", "value": round(mape, 2)}


def compute_segmentation_metrics() -> dict | None:
    """Compute silhouette score for current clustering."""
    p = PROCESSED_DIR / "rfm_clustered.csv"
    if not p.exists():
        return None

    rfm = pd.read_csv(p)
    if "Cluster" not in rfm.columns:
        return None

    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"]  = np.log1p(X["Monetary"])

    from sklearn.preprocessing import StandardScaler
    X_sc = StandardScaler().fit_transform(X)

    sil = silhouette_score(X_sc, rfm["Cluster"], sample_size=min(2000, len(rfm)), random_state=42)
    return {"model": "kmeans_segmentation", "metric": "silhouette", "value": round(sil, 4)}


def run() -> None:
    setup_logger()
    logger.info("=" * 50)
    logger.info("MONITORING — PERFORMANCE TRACKER")
    logger.info("=" * 50)

    log = load_metrics_log()
    timestamp = datetime.now().isoformat()
    entry = {"timestamp": timestamp, "metrics": []}

    for fn in [compute_churn_metrics, compute_forecast_metrics, compute_segmentation_metrics]:
        result = fn()
        if result:
            entry["metrics"].append(result)
            status = "✅" if _meets_target(result) else "⚠️"
            logger.info(f"  {status} {result['model']:<30} {result['metric']}: {result['value']}")

    log.append(entry)
    save_metrics_log(log)
    logger.success(f"Metrics logged → {METRICS_PATH}")
    logger.info(f"Total log entries: {len(log)}")


def _meets_target(m: dict) -> bool:
    targets = {"auc_roc": (0.80, ">="), "mape": (15.0, "<="), "silhouette": (0.35, ">=")}
    if m["metric"] not in targets:
        return True
    threshold, op = targets[m["metric"]]
    return m["value"] >= threshold if op == ">=" else m["value"] <= threshold


if __name__ == "__main__":
    run()