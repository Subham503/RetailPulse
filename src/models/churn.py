"""
Churn Prediction — XGBoost classifier on RFM features.

Churn definition: customer inactive for >= 90 days.

Steps:
  1. Load rfm.csv
  2. Label churned customers
  3. Train LogisticRegression (baseline) + XGBoost (primary)
  4. Evaluate AUC-ROC, classification report
  5. Attach ChurnProb to rfm → save rfm_with_churn.csv

Usage:
    python -m src.models.churn
"""

import sys
import joblib
import numpy as np
import pandas as pd
from loguru import logger

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.utils.config import PROCESSED_DIR, MODELS_DIR, CHURN_THRESHOLD_DAYS
from src.utils.logging_config import setup_logger
from src.utils.mlflow_utils import setup_mlflow

INPUT_PATH  = PROCESSED_DIR / "rfm.csv"
OUTPUT_PATH = PROCESSED_DIR / "rfm_with_churn.csv"

FEATURE_COLS = ["Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "RFM_Total"]


def label_churn(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add Churned label: 1 if Recency >= CHURN_THRESHOLD_DAYS."""
    rfm = rfm.copy()
    rfm["Churned"] = (rfm["Recency"] >= CHURN_THRESHOLD_DAYS).astype(int)
    rate = rfm["Churned"].mean()
    logger.info(f"Churn threshold: {CHURN_THRESHOLD_DAYS} days")
    logger.info(f"Churn rate: {rate:.2%}  ({rfm['Churned'].sum()} / {len(rfm)} customers)")
    return rfm


def build_features(rfm: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build X, y for churn model."""
    available = [c for c in FEATURE_COLS if c in rfm.columns]
    X = rfm[available].copy()
    X["Monetary"]  = np.log1p(X["Monetary"])
    X["Frequency"] = np.log1p(X["Frequency"])
    y = rfm["Churned"]
    return X, y


def train_baseline(X_train, X_test, y_train, y_test) -> tuple[LogisticRegression, StandardScaler, float]:
    """Logistic Regression baseline."""
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_tr_sc, y_train)
    auc = roc_auc_score(y_test, lr.predict_proba(X_te_sc)[:, 1])
    logger.info(f"Baseline (LogReg) AUC : {auc:.4f}")
    return lr, scaler, auc


def train_xgboost(X_train, X_test, y_train, y_test) -> tuple[xgb.XGBClassifier, float]:
    """XGBoost primary model."""
    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    params = {
        "n_estimators":    300,
        "max_depth":       4,
        "learning_rate":   0.05,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos,
        "random_state":    42,
        "eval_metric":     "auc",
        "verbosity":       0,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"XGBoost AUC          : {auc:.4f}  (target ≥ 0.80)")

    report = classification_report(y_test, (y_proba >= 0.5).astype(int))
    logger.info(f"\n{report}")

    cm = confusion_matrix(y_test, (y_proba >= 0.5).astype(int))
    logger.info(f"Confusion matrix:\n{cm}")

    return model, auc, params


def run() -> dict:
    setup_logger()
    setup_mlflow()

    logger.info("=" * 50)
    logger.info("MODEL 3/4 — CHURN PREDICTION (XGBoost)")
    logger.info("=" * 50)

    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}. Run `make features` first.")
        sys.exit(1)

    rfm = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded: {len(rfm):,} customers")

    rfm = label_churn(rfm)
    X, y = build_features(rfm)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # Baseline
    lr, scaler, lr_auc = train_baseline(X_train, X_test, y_train, y_test)

    # Primary
    with mlflow.start_run(run_name="churn_xgboost"):
        xgb_model, xgb_auc, params = train_xgboost(X_train, X_test, y_train, y_test)

        mlflow.log_params(params)
        mlflow.log_metric("lr_baseline_auc",  round(lr_auc, 4))
        mlflow.log_metric("xgb_auc",          round(xgb_auc, 4))
        mlflow.log_metric("churn_threshold",   CHURN_THRESHOLD_DAYS)
        mlflow.xgboost.log_model(xgb_model,   "xgb_churn_model")

    # Attach churn probability to full RFM
    rfm["ChurnProb"] = xgb_model.predict_proba(X)[:, 1].round(4)
    rfm["ChurnLabel"] = (rfm["ChurnProb"] >= 0.5).astype(int)

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, MODELS_DIR / "xgb_churn.joblib")
    joblib.dump(scaler,    MODELS_DIR / "churn_scaler.joblib")
    rfm.to_csv(OUTPUT_PATH, index=False)

    logger.success(f"Model  → {MODELS_DIR}/xgb_churn.joblib")
    logger.success(f"Data   → {OUTPUT_PATH}")
    logger.info(f"AUC    : {xgb_auc:.4f}  {'✅ target met' if xgb_auc >= 0.80 else '⚠ below target'}")

    return {"model": xgb_model, "auc": xgb_auc, "rfm_with_churn": rfm}


if __name__ == "__main__":
    run()