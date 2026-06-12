"""
Customer Segmentation — KMeans on RFM features.

Forces k=6 segments as per business requirement (6-8 meaningful segments).
Also runs elbow/silhouette analysis for documentation.

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

# Business requirement: 6-8 meaningful segments
FORCED_K  = 6
K_RANGE   = range(2, 11)

# Segment labels mapped by cluster rank (Monetary mean descending)
SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal Customers",
    2: "Potential Loyalists",
    3: "At-Risk Customers",
    4: "Hibernating",
    5: "Lost Customers",
}


def prepare_features(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Log-transform skewed cols, then StandardScale."""
    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"]  = np.log1p(X["Monetary"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def elbow_analysis(X: np.ndarray) -> tuple[list, list]:
    """Run elbow + silhouette for documentation purposes."""
    inertias, silhouettes = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=42)
        silhouettes.append(sil)
        logger.info(f"  k={k:>2} | inertia={km.inertia_:>12.1f} | silhouette={sil:.4f}")
    return inertias, silhouettes


def label_clusters(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Assign business segment names ranked by Monetary value."""
    rfm = rfm.copy()
    rfm["Cluster"] = labels

    # Rank clusters by avg Monetary (descending) → assign labels
    cluster_monetary = (
        rfm.groupby("Cluster")["Monetary"]
        .mean()
        .sort_values(ascending=False)
    )

    rank_map = {
        cid: SEGMENT_LABELS.get(i, f"Segment {i+1}")
        for i, cid in enumerate(cluster_monetary.index)
    }
    rfm["ClusterLabel"] = rfm["Cluster"].map(rank_map)
    return rfm


def run() -> dict:
    setup_logger()
    setup_mlflow()

    logger.info("=" * 50)
    logger.info("MODEL 1/4 — CUSTOMER SEGMENTATION (KMeans k=6)")
    logger.info("=" * 50)

    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}. Run `make features` first.")
        sys.exit(1)

    rfm = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded RFM: {len(rfm):,} customers")

    X_scaled, scaler = prepare_features(rfm)

    # Elbow analysis (for documentation)
    logger.info("Elbow analysis across k=2-10:")
    inertias, silhouettes = elbow_analysis(X_scaled)

    best_sil_k = list(K_RANGE)[int(np.argmax(silhouettes))]
    logger.info(f"Best silhouette k={best_sil_k} — using k={FORCED_K} per business requirement")

    # Train final model at k=6
    km = KMeans(n_clusters=FORCED_K, random_state=42, n_init=10)
    km.fit(X_scaled)
    sil_score = silhouette_score(X_scaled, km.labels_,
                                  sample_size=min(3000, len(X_scaled)), random_state=42)

    with mlflow.start_run(run_name="segmentation_kmeans_k6"):
        mlflow.log_param("k",              FORCED_K)
        mlflow.log_param("k_range",        f"{min(K_RANGE)}-{max(K_RANGE)}")
        mlflow.log_param("best_elbow_k",   best_sil_k)
        mlflow.log_param("forced_k",       FORCED_K)
        mlflow.log_param("n_customers",    len(rfm))
        mlflow.log_metric("silhouette_k6", round(sil_score, 4))
        mlflow.sklearn.log_model(km, "kmeans_k6_model")

        rfm_clustered = label_clusters(rfm, km.labels_)

        summary = (
            rfm_clustered.groupby("ClusterLabel")
            .agg(
                Count        = ("CustomerID", "count"),
                Avg_Recency  = ("Recency",    "mean"),
                Avg_Frequency= ("Frequency",  "mean"),
                Avg_Monetary = ("Monetary",   "mean"),
            )
            .round(1)
            .sort_values("Avg_Monetary", ascending=False)
        )
        logger.info(f"\nCluster Summary (k=6):\n{summary.to_string()}")
        logger.info(f"Silhouette @ k=6: {sil_score:.4f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(km,     MODELS_DIR / "kmeans.joblib")
    joblib.dump(scaler, MODELS_DIR / "rfm_scaler.joblib")
    rfm_clustered.to_csv(OUTPUT_PATH, index=False)

    logger.success(f"Model  → {MODELS_DIR}/kmeans.joblib")
    logger.success(f"Scaler → {MODELS_DIR}/rfm_scaler.joblib")
    logger.success(f"Data   → {OUTPUT_PATH}")

    return {
        "model": km, "scaler": scaler,
        "rfm_clustered": rfm_clustered,
        "silhouette": sil_score,
        "k": FORCED_K,
    }


if __name__ == "__main__":
    run()