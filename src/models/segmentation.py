"""
Customer Segmentation — KMeans on RFM features.

Steps:
  1. Load rfm.csv
  2. Log-transform + StandardScale R, F, M
  3. Elbow method (k=2–10) + silhouette scoring
  4. Train final KMeans at best k
  5. Save model + scaler + labelled CSV

Usage:
    python -m src.models.segmentation
"""

import sys
import joblib
import numpy as np
import pandas as pd
from loguru import logger

import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.utils.config import PROCESSED_DIR, MODELS_DIR
from src.utils.logging_config import setup_logger
from src.utils.mlflow_utils import setup_mlflow

INPUT_PATH  = PROCESSED_DIR / "rfm.csv"
OUTPUT_PATH = PROCESSED_DIR / "rfm_clustered.csv"
K_RANGE     = range(2, 11)


def prepare_features(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Log-transform skewed cols, then StandardScale."""
    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"]  = np.log1p(X["Monetary"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def find_best_k(X: np.ndarray) -> tuple[int, list, list]:
    """Elbow + silhouette to pick optimal k."""
    inertias, silhouettes = [], []

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=42)
        silhouettes.append(sil)
        logger.info(f"  k={k:>2} | inertia={km.inertia_:>12.1f} | silhouette={sil:.4f}")

    best_k = list(K_RANGE)[int(np.argmax(silhouettes))]
    logger.info(f"Best k = {best_k}  (silhouette = {max(silhouettes):.4f})")
    return best_k, inertias, silhouettes


def train_final_model(X: np.ndarray, k: int) -> KMeans:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    return km


def label_clusters(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Attach cluster id and human-readable cluster label."""
    rfm = rfm.copy()
    rfm["Cluster"] = labels

    # Rank clusters by Monetary mean → assign labels
    cluster_monetary = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False)
    rank_map = {
        0: "High Value",
        1: "Mid Value",
        2: "Low Value",
        3: "Dormant",
        4: "Occasional",
    }
    cluster_rank = {cid: rank_map.get(i, f"Cluster {i}") for i, cid in enumerate(cluster_monetary.index)}
    rfm["ClusterLabel"] = rfm["Cluster"].map(cluster_rank)
    return rfm


def run() -> dict:
    setup_logger()
    setup_mlflow()

    logger.info("=" * 50)
    logger.info("MODEL 1/4 — CUSTOMER SEGMENTATION (KMeans)")
    logger.info("=" * 50)

    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}. Run `make features` first.")
        sys.exit(1)

    rfm = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded RFM: {len(rfm):,} customers")

    X_scaled, scaler = prepare_features(rfm)

    logger.info("Searching for best k...")
    best_k, inertias, silhouettes = find_best_k(X_scaled)

    with mlflow.start_run(run_name="segmentation_kmeans"):
        mlflow.log_param("best_k",          best_k)
        mlflow.log_param("k_range",         f"{min(K_RANGE)}-{max(K_RANGE)}")
        mlflow.log_param("n_customers",     len(rfm))
        mlflow.log_metric("silhouette_score", round(max(silhouettes), 4))
        mlflow.log_metric("inertia",          round(inertias[best_k - min(K_RANGE)], 2))

        km = train_final_model(X_scaled, best_k)
        mlflow.sklearn.log_model(km, "kmeans_model")

        rfm_clustered = label_clusters(rfm, km.labels_)

        # Cluster summary
        summary = rfm_clustered.groupby("ClusterLabel").agg(
            count=("CustomerID", "count"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
        ).round(1)

        logger.info("\nCluster Summary:\n" + summary.to_string())

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(km,     MODELS_DIR / "kmeans.joblib")
    joblib.dump(scaler, MODELS_DIR / "rfm_scaler.joblib")
    rfm_clustered.to_csv(OUTPUT_PATH, index=False)

    logger.success(f"Model  → {MODELS_DIR}/kmeans.joblib")
    logger.success(f"Scaler → {MODELS_DIR}/rfm_scaler.joblib")
    logger.success(f"Data   → {OUTPUT_PATH}")

    return {"model": km, "scaler": scaler, "rfm_clustered": rfm_clustered, "best_k": best_k}


if __name__ == "__main__":
    run()