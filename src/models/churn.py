"""
Churn Prediction — XGBoost classifier on RFM features.
Includes SHAP explainability.

Usage:
    python -m src.models.churn
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from src.utils.config import PROCESSED_DIR, MODELS_DIR, CHURN_THRESHOLD_DAYS
from src.utils.logging_config import setup_logger
from src.utils.mlflow_utils import setup_mlflow

INPUT_PATH   = PROCESSED_DIR / "rfm.csv"
OUTPUT_PATH  = PROCESSED_DIR / "rfm_with_churn.csv"
SHAP_DIR     = MODELS_DIR.parent / "reports" / "shap"
FEATURE_COLS = ["Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "RFM_Total"]


def label_churn(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm["Churned"] = (rfm["Recency"] >= CHURN_THRESHOLD_DAYS).astype(int)
    rate = rfm["Churned"].mean()
    logger.info(f"Churn threshold : {CHURN_THRESHOLD_DAYS} days")
    logger.info(f"Churn rate      : {rate:.2%}  ({rfm['Churned'].sum()} / {len(rfm)})")
    return rfm


def build_features(rfm: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available = [c for c in FEATURE_COLS if c in rfm.columns]
    X = rfm[available].copy()
    X["Monetary"]  = np.log1p(X["Monetary"])
    X["Frequency"] = np.log1p(X["Frequency"])
    return X, rfm["Churned"]


def train_baseline(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_tr, y_train)
    auc = roc_auc_score(y_test, lr.predict_proba(X_te)[:, 1])
    logger.info(f"Baseline (LogReg) AUC : {auc:.4f}")
    return lr, scaler, auc


def train_xgboost(X_train, X_test, y_train, y_test):
    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos, "random_state": 42,
        "eval_metric": "auc", "verbosity": 0,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"XGBoost AUC : {auc:.4f}  (target ≥ 0.88)")
    logger.info("\n" + classification_report(y_test, (y_proba >= 0.5).astype(int)))
    return model, auc, params


def generate_shap_plots(model, X_train: pd.DataFrame) -> None:
    """Generate and save SHAP summary + bar plots."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating SHAP explanations...")

    explainer   = shap.TreeExplainer(model)
    sample      = X_train.sample(min(500, len(X_train)), random_state=42)
    shap_values = explainer.shap_values(sample)

    # Summary dot plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot (mean absolute SHAP)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.success(f"SHAP plots → {SHAP_DIR}")

    # Log feature importance from SHAP
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_shap, index=sample.columns).sort_values(ascending=False)
    logger.info("SHAP Feature Importance:")
    for feat, val in importance.items():
        logger.info(f"  {feat:<20}: {val:.4f}")


def run() -> dict:
    setup_logger()
    setup_mlflow()

    logger.info("=" * 50)
    logger.info("MODEL 3/4 — CHURN PREDICTION (XGBoost + SHAP)")
    logger.info("=" * 50)

    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}. Run `make features` first.")
        sys.exit(1)

    rfm = pd.read_csv(INPUT_PATH)
    rfm = label_churn(rfm)
    X, y = build_features(rfm)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}  Test: {len(X_test)}")

    lr, scaler, lr_auc = train_baseline(X_train, X_test, y_train, y_test)

    with mlflow.start_run(run_name="churn_xgboost_shap"):
        xgb_model, xgb_auc, params = train_xgboost(X_train, X_test, y_train, y_test)
        mlflow.log_params(params)
        mlflow.log_metric("lr_baseline_auc", round(lr_auc, 4))
        mlflow.log_metric("xgb_auc",         round(xgb_auc, 4))
        mlflow.xgboost.log_model(xgb_model,  "xgb_churn_model")

    # SHAP
    generate_shap_plots(xgb_model, X_train)

    # Attach predictions to full RFM
    rfm["ChurnProb"]  = xgb_model.predict_proba(X)[:, 1].round(4)
    rfm["ChurnLabel"] = (rfm["ChurnProb"] >= 0.5).astype(int)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, MODELS_DIR / "xgb_churn.joblib")
    joblib.dump(scaler,    MODELS_DIR / "churn_scaler.joblib")
    rfm.to_csv(OUTPUT_PATH, index=False)

    logger.success(f"Model  → {MODELS_DIR}/xgb_churn.joblib")
    logger.success(f"Data   → {OUTPUT_PATH}")
    logger.info(f"AUC    : {xgb_auc:.4f}")

    return {"model": xgb_model, "auc": xgb_auc, "rfm_with_churn": rfm}


if __name__ == "__main__":
    run()